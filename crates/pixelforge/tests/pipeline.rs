//! The library cases of tests/test_pipeline.py (the CLI cases are in tests/cli.rs) and
//! `test_mcda_palette_source_with_box_and_kopf` of tests/test_branches.py: `pipeline::run` /
//! `run_loaded` on paths and arrays, the stats, the saved files and the log lines.

mod common;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Mutex, Once};
use std::thread::ThreadId;

use common::*;
use pixelforge::config::Config;
use pixelforge::io;
use pixelforge::pipeline::{self, Input, Output};
use pixelforge::pyjson::{py_round, Json};
use pixelforge::{color, palette, parallel, preprocess, quantize};

// ---------------------------------------------------------------------------------------
// Log capture (pytest's caplog): one global logger for this binary; tests run on parallel
// threads, so each record carries its thread and a test only sees its own.

struct Record {
    thread: ThreadId,
    target: String,
    level: log::Level,
    message: String,
}

struct Collector;

static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

impl log::Log for Collector {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push(Record {
            thread: std::thread::current().id(),
            target: record.target().to_string(),
            level: record.level(),
            message: record.args().to_string(),
        });
    }

    fn flush(&self) {}
}

static COLLECTOR: Collector = Collector;

/// Install the collector (once) at INFO, the level every test but one captures at.
fn capture() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        log::set_logger(&COLLECTOR).unwrap();
        log::set_max_level(log::LevelFilter::Info);
    });
}

/// The records this thread logged since the last call (`caplog.records` + `caplog.clear()`).
fn take_records() -> Vec<Record> {
    let me = std::thread::current().id();
    let mut all = RECORDS.lock().unwrap();
    let (mine, others): (Vec<Record>, Vec<Record>) = all.drain(..).partition(|r| r.thread == me);
    *all = others;
    mine
}

const PIPELINE: &str = "pixelforge::pipeline";

// ---------------------------------------------------------------------------------------
// Helpers

fn stat<'a>(out: &'a Output, key: &str) -> &'a Json {
    match &out.stats {
        Json::Object(items) => {
            &items
                .iter()
                .find(|(k, _)| k == key)
                .unwrap_or_else(|| panic!("no stats[{key:?}]"))
                .1
        }
        other => panic!("stats is not an object: {other:?}"),
    }
}

fn stat_keys(out: &Output) -> BTreeSet<String> {
    match &out.stats {
        Json::Object(items) => items.iter().map(|(k, _)| k.clone()).collect(),
        _ => panic!("stats is not an object"),
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn run_loaded(loaded: &io::Loaded, config: &Config) -> Output {
    pipeline::run_loaded(loaded, config, Vec::new(), false).unwrap()
}

fn assert_same_output(a: &Output, b: &Output) {
    assert_eq!(a.indices, b.indices);
    assert_eq!(a.palette, b.palette);
    assert_eq!(a.image, b.image);
}

/// `np.unique(indices)` as a set.
fn unique(indices: &[i64]) -> BTreeSet<i64> {
    indices.iter().copied().collect()
}

/// `_opaque(rgb)`: an opaque `io.Loaded` from float RGB (`np.clip(np.round(rgb), 0, 255)`).
fn opaque(rgb: &[f64], h: usize, w: usize) -> io::Loaded {
    let rgba: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|p| {
            let c = |v: f64| v.round_ties_even().clamp(0.0, 255.0) as u8;
            [c(p[0]), c(p[1]), c(p[2]), 255]
        })
        .collect();
    io::from_rgba(&rgba, h, w, "test".into())
}

fn gerstner_dither(dither: &str, strength: Option<f64>) -> Config {
    let mut pairs = vec![
        ("method", s("gerstner")),
        ("out_width", int(24)),
        ("out_height", int(24)),
        ("palette_size", int(8)),
        ("denoise", s("none")),
        ("key_bg", b(false)),
        ("outline", s("none")),
        ("remove_orphans", b(false)),
        ("fix_jaggies", b(false)),
        ("dither", s(dither)),
    ];
    if let Some(x) = strength {
        pairs.push(("dither_strength", float(x)));
    }
    config(&pairs)
}

/// A decoded PNG as Pillow sees the raw image: colour type, PLTE, tRNS and one sample per
/// pixel (indexed images unpacked from their bit depth).
struct RawPng {
    color_type: png::ColorType,
    trns: Option<Vec<u8>>,
    samples: Vec<u8>,
}

fn decode_raw(path: &Path) -> RawPng {
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut buf).unwrap();
    let info = reader.info();
    let (w, h) = (info.width as usize, info.height as usize);
    let depth = info.bit_depth as usize;
    let channels = info.color_type.samples();
    let mut samples = Vec::new();
    for y in 0..h {
        let row = &buf[y * frame.line_size..(y + 1) * frame.line_size];
        if depth == 8 {
            samples.extend_from_slice(&row[..w * channels]);
        } else {
            assert_eq!(channels, 1);
            let per_byte = 8 / depth;
            for x in 0..w {
                let byte = row[x / per_byte];
                let shift = 8 - depth * (x % per_byte + 1);
                samples.push((byte >> shift) & ((1u8 << depth) - 1));
            }
        }
    }
    RawPng {
        color_type: info.color_type,
        trns: info.trns.as_ref().map(|t| t.to_vec()),
        samples,
    }
}

/// `np.asarray(Image.open(path).convert("RGBA"))`.
fn decode_rgba(path: &Path) -> Vec<u8> {
    io::decode_bytes("x.png", &std::fs::read(path).unwrap(), io::MAX_INPUT_PIXELS)
        .unwrap()
        .rgba
}

// ---------------------------------------------------------------------------------------
// tests/test_pipeline.py

#[test]
fn library_api() {
    let tmp = TempDir::new("api");
    let cfg = config(&[
        ("preset", s("sprite")),
        ("method", s("gerstner")),
        ("palette_size", int(12)),
        ("out_height", int(48)),
    ]);
    let result = run_path(&fixture_path("circle_alpha"), &cfg).unwrap();
    let (h, w) = (result.indices.h, result.indices.w);
    // uint8 (h, w, 4) image, uint8 (k, 3) palette, (h, w) indices with transparency.
    assert_eq!(result.image.len(), h * w * 4);
    assert!(!result.palette.is_empty());
    assert_eq!(result.indices.indices.len(), h * w);
    assert_eq!(result.indices.indices.iter().min(), Some(&-1));
    let keys = stat_keys(&result);
    for key in ["iterations", "timings", "final_palette_size", "config_hash"] {
        assert!(keys.contains(key), "{key}");
    }
    let paths = result.save(&tmp.join("out/hero")).unwrap();
    assert!(tmp.join("out/hero.png").is_file() && paths.len() == 5);
}

#[test]
fn named_palette_and_dither() {
    let base = [
        ("preset", s("background")),
        ("method", s("box")),
        ("palette_name", s("pico8")),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("tileset", b(false)),
        ("remove_orphans", b(false)),
    ];
    let path = fixture_path("gradient6");
    let plain = run_path(&path, &config(&merge(&[&base, &[("dither", s("none"))]]))).unwrap();
    let dithered = run_path(
        &path,
        &config(&merge(&[
            &base,
            &[("dither", s("bayer4")), ("dither_strength", float(1.0))],
        ])),
    )
    .unwrap();
    let pico8 = palette::parse_hex_lines("pico8").unwrap();
    for result in [&plain, &dithered] {
        // A named hardware palette is used as-is: no ramp shift, no saturation.
        let hex: Vec<String> = result
            .palette
            .iter()
            .map(|&c| color::rgb8_to_hex(c)[1..].to_string())
            .collect();
        assert_eq!(hex, pico8);
        assert!(unique(&result.indices.indices).len() >= 6);
    }
    assert_ne!(plain.indices.indices, dithered.indices.indices);
    assert_eq!(stat(&plain, "final_palette_size"), &Json::Int(16));
}

#[test]
fn outline_index_in_stats() {
    let base = [
        ("preset", s("sprite")),
        ("method", s("box")),
        ("out_height", int(16)),
    ];
    let path = fixture_path("circle_alpha");
    let result = run_path(&path, &config(&base)).unwrap();
    assert_eq!(
        stat(&result, "outline_index"),
        &Json::Int(result.palette.len() as i64 - 1)
    );
    // json.dumps(result.stats) succeeds.
    pixelforge::pyjson::dumps(&result.stats, pixelforge::pyjson::Layout::INDENT2);
    let plain = run_path(&path, &config(&merge(&[&base, &[("outline", s("none"))]]))).unwrap();
    assert_eq!(stat(&plain, "outline_index"), &Json::Null);
}

#[test]
fn gerstner_dither_changes_smooth_gradient() {
    let n = 64;
    let mut rgb = Vec::with_capacity(n * n * 3);
    for i in 0..n {
        for j in 0..n {
            let (y, x) = (i as f64 / 63.0, j as f64 / 63.0);
            rgb.extend([40.0 + 200.0 * x, 30.0 + 75.0 * (x + y), 20.0 + 100.0 * y]);
        }
    }
    // Palette entries are far enough apart that dithering the palette colors themselves (the
    // old behavior) changed no pixel at all here.
    let loaded = opaque(&rgb, n, n);
    let plain = run_loaded(&loaded, &gerstner_dither("none", None));
    let dithered = run_loaded(&loaded, &gerstner_dither("bayer4", Some(1.0)));
    let (p, d) = (&plain.indices.indices, &dithered.indices.indices);
    let opaque_mask: Vec<bool> = p.iter().map(|&i| i >= 0).collect();
    assert_eq!(opaque_mask, d.iter().map(|&i| i >= 0).collect::<Vec<_>>());
    let n_opaque = opaque_mask.iter().filter(|&&m| m).count();
    let changed = (0..p.len())
        .filter(|&k| opaque_mask[k] && p[k] != d[k])
        .count();
    assert!(changed as f64 / n_opaque as f64 > 0.05);
    assert!((0..d.len()).filter(|&k| opaque_mask[k]).all(|k| d[k] >= 0));
    assert!(*d.iter().max().unwrap() < dithered.palette.len() as i64);
    // Same palette (β applied exactly once); only the index image changes.
    assert_eq!(plain.palette, dithered.palette);
}

#[test]
fn gerstner_auto_dither_leaves_hard_edge_alone() {
    let n = 64;
    let mut rgb = Vec::with_capacity(n * n * 3);
    for _ in 0..n {
        for j in 0..n {
            rgb.extend(if j < 32 {
                [200.0, 40.0, 40.0]
            } else {
                [40.0, 40.0, 200.0]
            });
        }
    }
    let loaded = opaque(&rgb, n, n);
    let plain = run_loaded(&loaded, &gerstner_dither("none", None));
    let auto = run_loaded(&loaded, &gerstner_dither("auto", Some(1.0)));
    // Output columns on both sides of the red/blue boundary at x = 12.
    let edge = |out: &Output| -> Vec<i64> {
        let w = out.indices.w;
        (0..out.indices.h)
            .flat_map(|y| (10..14).map(move |x| (y, x)))
            .map(|(y, x)| out.indices.indices[y * w + x])
            .collect()
    };
    assert_eq!(unique(&edge(&plain)).len(), 2);
    assert_eq!(edge(&plain), edge(&auto));
}

#[test]
fn local_std_ignores_transparent_cells() {
    let (h, w) = (5, 5);
    let mut lab = vec![0.0; h * w * 3];
    let mut mask = vec![false; h * w];
    for y in 1..4 {
        for x in 1..4 {
            mask[y * w + x] = true;
            lab[3 * (y * w + x)] = 50.0; // flat opaque 3×3 block; transparent cells hold L = 0
        }
    }
    // Unmasked: the silhouette looks like detail.
    assert!(quantize::local_std(&lab, None, h, w)[w + 1] > 20.0);
    let std = quantize::local_std(&lab, Some(&mask), h, w);
    let max = |inside: bool| {
        (0..h * w)
            .filter(|&k| mask[k] == inside)
            .map(|k| std[k])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    assert!(max(true) < 1e-3 && max(false) == 0.0);
    lab[3 * (2 * w + 2)] = 80.0; // real detail inside the opaque block is still seen
    assert!(quantize::local_std(&lab, Some(&mask), h, w)[2 * w + 2] > 5.0);
}

/// `_opaque_square()`: 64 × 64, red-brown top quarter, blue below.
fn opaque_square() -> io::Loaded {
    let size = 64;
    let mut rgba = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for _ in 0..size {
            rgba.extend(if y >= size / 4 {
                [40, 120, 200, 255]
            } else {
                [200, 60, 30, 255]
            });
        }
    }
    io::from_rgba(&rgba, size, size, "square".into())
}

fn has_outline_ring(result: &Output) -> bool {
    let Json::Int(ring) = *stat(result, "outline_index") else {
        return false;
    };
    let (h, w, idx) = (result.indices.h, result.indices.w, &result.indices.indices);
    let mut border = Vec::new();
    for x in 1..w - 1 {
        border.push(idx[x]);
        border.push(idx[(h - 1) * w + x]);
    }
    for y in 1..h - 1 {
        border.push(idx[y * w]);
        border.push(idx[y * w + w - 1]);
    }
    border.iter().all(|&i| i == ring)
}

fn shape(out: &Output) -> (usize, usize) {
    (out.indices.h, out.indices.w)
}

#[test]
fn canvas_is_exactly_the_requested_size_with_outline() {
    let base = [
        ("preset", s("sprite")),
        ("method", s("box")),
        ("key_bg", b(false)),
    ];
    let result = run_loaded(
        &opaque_square(),
        &config(&merge(&[
            &base,
            &[("out_width", int(32)), ("out_height", int(32))],
        ])),
    );
    assert_eq!(shape(&result), (32, 32));
    assert!(has_outline_ring(&result));
    let w = result.indices.w;
    for y in 1..31 {
        for x in 1..31 {
            assert!(result.indices.indices[y * w + x] >= 0);
        }
    }
    assert_eq!(stat(&result, "outline_margin"), &Json::Int(1));
    assert_eq!(stat(&result, "outline_clipped"), &Json::Bool(false));
    let only_height = run_loaded(
        &opaque_square(),
        &config(&merge(&[&base, &[("out_height", int(24))]])),
    );
    assert_eq!(shape(&only_height), (24, 24));
    assert!(has_outline_ring(&only_height));
}

#[test]
fn canvas_is_exactly_the_requested_size_without_outline() {
    let result = run_loaded(
        &opaque_square(),
        &config(&[
            ("preset", s("sprite")),
            ("method", s("box")),
            ("key_bg", b(false)),
            ("outline", s("none")),
            ("out_width", int(32)),
            ("out_height", int(32)),
        ]),
    );
    assert_eq!(shape(&result), (32, 32));
    assert!(result.indices.indices.iter().all(|&i| i >= 0));
    assert_eq!(stat(&result, "outline_margin"), &Json::Int(0));
    assert_eq!(stat(&result, "outline_index"), &Json::Null);
    assert_eq!(stat(&result, "outline_clipped"), &Json::Bool(false));
}

#[test]
fn default_sprite_longest_edge_is_exactly_64() {
    let result = run_path(
        &fixture_path("circle_alpha"),
        &config(&[("preset", s("sprite")), ("method", s("box"))]),
    )
    .unwrap();
    assert_eq!(result.indices.h.max(result.indices.w), 64);
    assert_eq!(stat(&result, "outline_margin"), &Json::Int(1));
}

#[test]
fn tileset_outline_is_clipped_not_padded() {
    let cfg = config(&[
        ("preset", s("sprite")),
        ("method", s("box")),
        ("key_bg", b(false)),
        ("tileset", b(true)),
        ("tile_size", int(8)),
        ("out_width", int(32)),
        ("out_height", int(32)),
    ]);
    let result = run_loaded(&opaque_square(), &cfg);
    assert_eq!(shape(&result), (32, 32));
    assert_eq!(stat(&result, "outline_margin"), &Json::Int(0));
    assert_eq!(stat(&result, "outline_clipped"), &Json::Bool(true));
}

/// `_sprite_rgba()`: 16 × 16 RGBA with three opaque blocks.
fn sprite_rgba() -> Vec<u8> {
    let mut img = Image::filled(16, 16, &[0, 0, 0, 0]);
    let mut fill = |ys: std::ops::Range<usize>, xs: std::ops::Range<usize>, px: [u8; 4]| {
        for y in ys {
            for x in xs.clone() {
                img.px_mut(y, x).copy_from_slice(&px);
            }
        }
    };
    fill(3..13, 3..13, [200, 60, 40, 255]);
    fill(5..9, 5..11, [40, 90, 210, 255]);
    fill(10..12, 4..8, [250, 240, 200, 255]);
    img.data
}

#[test]
fn transparent_index_first() {
    let tmp = TempDir::new("first");
    let rgba = sprite_rgba();
    let loaded = io::from_rgba(&rgba, 16, 16, io::rgba8_digest(16, 16, &rgba));
    let cfg = config(&[
        ("method", s("box")),
        ("out_width", int(16)),
        ("out_height", int(16)),
        ("crop_to_alpha", b(false)),
        ("key_bg", b(false)),
        ("remove_bg", b(false)),
        ("denoise", s("none")),
        ("dither", s("none")),
        ("outline", s("none")),
        ("remove_orphans", b(false)),
        ("fix_jaggies", b(false)),
        ("transparent_index", s("first")),
    ]);
    let result = run_loaded(&loaded, &cfg);
    let idx = &result.indices.indices;
    assert!(idx.contains(&-1) && idx.iter().min() == Some(&-1));
    result.save(&tmp.join("s")).unwrap();
    let png = decode_raw(&tmp.join("s.png"));
    // mode "P" with info["transparency"] == 0: tRNS matches b"^\xff*\x00\xff*$" at index 0.
    assert_eq!(png.color_type, png::ColorType::Indexed);
    let trns = png.trns.expect("a tRNS chunk");
    assert_eq!(trns[0], 0);
    assert!(trns[1..].iter().all(|&a| a == 255));
    let shifted: Vec<u8> = idx.iter().map(|&i| (i + 1) as u8).collect();
    assert_eq!(png.samples, shifted);
    assert_eq!(decode_rgba(&tmp.join("s.png")), result.image);
    let entries = read_json(&tmp.join("s_palette.json"));
    let entries = entries.as_array().unwrap();
    assert_eq!(
        entries[0],
        serde_json::json!({"hex": null, "transparent": true, "used": true})
    );
    assert_eq!(entries.len(), result.palette.len() + 1);
    let hex_path = tmp.join("s_palette.hex");
    let text = std::fs::read_to_string(&hex_path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], palette::TRANSPARENT_HEX_LINE);
    assert_eq!(lines.len(), result.palette.len() + 1);
    assert_eq!(
        palette::load_palette(hex_path.to_str().unwrap()).unwrap(),
        result.palette
    );

    assert!(config_err(&[("transparent_index", s("middle"))]).contains("transparent_index"));
}

#[test]
fn unused_named_palette_entries_are_kept_and_flagged() {
    let tmp = TempDir::new("unused");
    let cfg = config(&[
        ("method", s("box")),
        ("out_height", int(16)),
        ("palette_name", s("pico8")),
        ("dither", s("none")),
    ]);
    let result = run_path(&fixture_path("two_color"), &cfg).unwrap();
    result.save(&tmp.join("t")).unwrap();
    let entries = read_json(&tmp.join("t_palette.json"));
    let entries = entries.as_array().unwrap();
    assert_eq!(entries.len(), 16);
    let used: Vec<i64> = unique(&result.indices.indices)
        .into_iter()
        .filter(|&i| i >= 0)
        .collect();
    let flagged: Vec<i64> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e["used"] == serde_json::Value::Bool(true))
        .map(|(i, _)| i as i64)
        .collect();
    assert_eq!(flagged, used);
    let meta = read_json(&tmp.join("t_meta.json"));
    let unused = meta["stats"]["colors_unused"].as_i64().unwrap();
    assert_eq!(unused, 16 - used.len() as i64);
    assert!(unused > 0);
}

/// `_three_bands()`: opaque 160 × 96, red left 16 columns, blue right 16, green between.
fn three_bands() -> io::Loaded {
    let (width, height) = (160, 96);
    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..height {
        for x in 0..width {
            rgba.extend(if x < 16 {
                [220, 30, 30, 255]
            } else if x >= width - 16 {
                [30, 30, 220, 255]
            } else {
                [40, 180, 60, 255]
            });
        }
    }
    io::from_rgba(&rgba, height, width, "bands".into())
}

fn tile_fit(fit: &str) -> Output {
    let cfg = config(&[
        ("preset", s("background")),
        ("method", s("box")),
        ("denoise", s("none")),
        ("dither", s("none")),
        ("out_width", int(64)),
        ("fit", s(fit)),
        ("palette_size", int(4)),
    ]);
    run_loaded(&three_bands(), &cfg)
}

#[test]
fn tileset_fit_pad_centers_aspect_preserving_content() {
    let result = tile_fit("pad");
    // warnings.simplefilter("error"): no warning at all.
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(shape(&result), (48, 64));
    let (h, w, idx) = (48, 64, &result.indices.indices);
    let opaque_rows: Vec<usize> = (0..h)
        .filter(|&y| (0..w).any(|x| idx[y * w + x] >= 0))
        .collect();
    let (first, last) = (opaque_rows[0], *opaque_rows.last().unwrap());
    let (top, bottom) = (first as i64, (h - 1 - last) as i64);
    assert!(opaque_rows.len() == 38 && (top - bottom).abs() <= 1);
    assert!((first..=last).all(|y| (0..w).all(|x| idx[y * w + x] >= 0)));
    assert_eq!(stat(&result, "fit"), &Json::str("pad"));
    assert_eq!(
        stat(&result, "content_size"),
        &Json::List(vec![Json::Int(64), Json::Int(38)])
    );
}

#[test]
fn tileset_fit_stretch_and_crop_fill_the_canvas() {
    let stretch = tile_fit("stretch");
    // pytest.warns(UserWarning, match="aspect").
    assert!(
        stretch.warnings.iter().any(|w| w.contains("aspect")),
        "{:?}",
        stretch.warnings
    );
    let crop = tile_fit("crop");
    for (result, fit) in [(&stretch, "stretch"), (&crop, "crop")] {
        assert_eq!(shape(result), (48, 64));
        assert!(result.indices.indices.iter().all(|&i| i >= 0));
        assert_eq!(stat(result, "fit"), &Json::str(fit));
        assert_eq!(
            stat(result, "content_size"),
            &Json::List(vec![Json::Int(64), Json::Int(48)])
        );
    }
    let has = |result: &Output, rgb: [i32; 3]| {
        result.image.chunks_exact(4).any(|p| {
            (0..3)
                .map(|c| (i32::from(p[c]) - rgb[c]).abs())
                .max()
                .unwrap()
                < 60
        })
    };
    // Crop takes the middle 128 of the 160 columns, so neither edge band survives.
    assert!(has(&stretch, [220, 30, 30]) && has(&stretch, [30, 30, 220]));
    assert!(!has(&crop, [220, 30, 30]) && !has(&crop, [30, 30, 220]));
    assert!(has(&crop, [40, 180, 60]));
}

#[test]
fn run_accepts_array_and_pil_image() {
    let path = fixture_path("circle_alpha");
    let cfg = config(&[
        ("preset", s("sprite")),
        ("method", s("box")),
        ("out_width", int(16)),
        ("out_height", int(16)),
    ]);
    let from_path = run_path(&path, &cfg).unwrap();
    // np.asarray(im.convert("RGBA")); a PIL image goes through the same conversion as the
    // array (Python only), so `from_pil` has no Rust counterpart.
    let rgba = fixture("circle_alpha");
    assert_eq!(rgba.c, 4);
    let from_array = run_image(&rgba, &cfg).unwrap();
    let floats: Vec<f64> = rgba.data.iter().map(|&v| f64::from(v) / 255.0).collect();
    let from_float = pipeline::run(
        Input::Array {
            shape: &rgba.shape(),
            data: io::ArrayData::F64(&floats),
        },
        &cfg,
        None,
    )
    .unwrap();
    for other in [&from_array, &from_float] {
        assert_same_output(&from_path, other);
    }
    // In-memory images hash shape + RGBA bytes, not the file bytes.
    assert_eq!(from_path.input_sha256, io::sha256_file(&path).unwrap());
    assert_ne!(from_array.input_sha256, from_path.input_sha256);
    assert_eq!(from_array.input_sha256, from_float.input_sha256);

    let mut opaque_rgb = Image::filled(12, 12, &[0, 0, 0]);
    for y in 0..12 {
        for x in 6..12 {
            opaque_rgb.px_mut(y, x).copy_from_slice(&[255, 255, 255]);
        }
    }
    let rgb_result = run_image(
        &opaque_rgb,
        &config(&[
            ("preset", s("sprite")),
            ("method", s("box")),
            ("out_width", int(8)),
            ("out_height", int(8)),
        ]),
    )
    .unwrap();
    assert_eq!(shape(&rgb_result), (8, 8));

    let err = |shape: &[usize], data: io::ArrayData<'_>, max: Option<u64>| -> String {
        match pipeline::run(Input::Array { shape, data }, &cfg, max) {
            Err(pipeline::Error::Pixelforge(e)) => e.to_string(),
            Err(e) => panic!("not a PixelforgeError: {e:?}"),
            Ok(_) => panic!("accepted"),
        }
    };
    let budget = (rgba.h * rgba.w - 1) as u64;
    assert!(
        err(&rgba.shape(), io::ArrayData::U8(&rgba.data), Some(budget)).contains("input budget")
    );
    assert!(err(&[8, 8], io::ArrayData::U8(&[0; 64]), None).contains("shape"));
    assert!(err(&[8, 8, 3], io::ArrayData::F64(&[2.0; 192]), None).contains("[0, 1]"));
    assert!(err(&[8, 8, 3], io::ArrayData::Other("int32"), None).contains("uint8 or float"));
}

#[test]
fn unknown_palette_fails_fast() {
    capture();
    let tmp = TempDir::new("palette");
    assert!(config_err(&[("palette_name", s("nope"))]).contains("unknown palette 'nope'"));

    // Python monkeypatched preprocess.run / downscale.get to fail if a stage ran before the
    // palette was validated. Rust-observable equivalent: no stage logged its INFO line.
    take_records();
    let missing = tmp.join("missing.hex");
    let cfg = config(&[("palette_name", s(missing.to_str().unwrap()))]);
    match run_path(&fixture_path("two_color"), &cfg) {
        Err(pipeline::Error::Config(e)) => assert!(e.0.contains("unknown palette"), "{e}"),
        other => panic!("expected a ConfigError, got {:?}", other.err()),
    }
    let stages: Vec<String> = take_records()
        .into_iter()
        .filter(|r| r.target == PIPELINE)
        .map(|r| r.message)
        .collect();
    assert!(stages.is_empty(), "a stage ran: {stages:?}");

    let bad = tmp.join("bad.hex");
    std::fs::write(&bad, "not a color\n").unwrap();
    let cfg = config(&[("palette_name", s(bad.to_str().unwrap()))]);
    match run_path(&fixture_path("two_color"), &cfg) {
        Err(pipeline::Error::Config(e)) => assert!(e.0.contains("not a hex color"), "{e}"),
        other => panic!("expected a ConfigError, got {:?}", other.err()),
    }
}

#[test]
fn meta_records_matting_versions_only_with_remove_bg() {
    let tmp = TempDir::new("matting");
    let cfg = config(&[
        ("method", s("box")),
        ("out_height", int(16)),
        ("palette_size", int(4)),
    ]);
    let mut result = run_path(&fixture_path("circle_alpha"), &cfg).unwrap();
    result.save(&tmp.join("plain")).unwrap();
    let meta = read_json(&tmp.join("plain_meta.json"));
    assert!(meta.get("remove_bg_versions").is_none());
    // Saving does not run rembg, so a result relabelled remove_bg=True exercises the meta.
    result.config = replace(&cfg, &[("remove_bg", b(true))]);
    result.save(&tmp.join("matted")).unwrap();
    let meta = read_json(&tmp.join("matted_meta.json"));
    let versions = meta["remove_bg_versions"].as_object().unwrap();
    let keys: BTreeSet<&str> = versions.keys().map(String::as_str).collect();
    assert_eq!(keys, BTreeSet::from(["rembg", "onnxruntime"]));
    assert!(versions.values().all(|v| v.is_null() || v.is_string()));
}

#[test]
fn meta_environment_and_rounded_palette_lab() {
    // Python sets PIXELFORGE_THREADS=3 first; env vars are not set in-process here (tests run
    // on parallel threads): tests/cli.rs `meta_environment_threads_from_env` checks that part
    // on the binary. The native environment is {platform, threads, native}, not Python's
    // interpreter and library versions.
    let tmp = TempDir::new("environment");
    let cfg = config(&[
        ("method", s("box")),
        ("out_height", int(16)),
        ("palette_size", int(4)),
    ]);
    let result = run_path(&fixture_path("circle_alpha"), &cfg).unwrap();
    result.save(&tmp.join("a")).unwrap();
    let meta = read_json(&tmp.join("a_meta.json"));
    let env = meta["environment"].as_object().unwrap();
    let keys: BTreeSet<&str> = env.keys().map(String::as_str).collect();
    assert_eq!(keys, BTreeSet::from(["platform", "threads", "native"]));
    assert_eq!(env["native"], pipeline::VERSION);
    assert_eq!(env["platform"], pipeline::platform().as_str());
    assert_eq!(env["threads"], parallel::thread_count() as u64);
    // The environment is not hashed.
    assert_eq!(meta["config_hash"], cfg.hash().unwrap().as_str());
    for entry in read_json(&tmp.join("a_palette.json")).as_array().unwrap() {
        for v in entry["lab"].as_array().unwrap() {
            let v = v.as_f64().unwrap();
            assert_eq!(v, py_round(v, 6));
        }
    }
}

#[test]
fn downscaler_receives_alpha_weight() {
    // Red square with a 4-px white fringe at alpha 128 (opaque) and one at 127 (transparent):
    // the downscaler sees the fringe at weight 128/255, not as a full-weight opaque pixel.
    // Python recorded the downscaler's arguments through a monkeypatched downscale.get in the
    // reference orchestration; here the weight the pipeline hands every downscaler
    // (`Preprocessed.weight`, see pipeline::downscale) is checked directly.
    let mut img = Image::filled(64, 64, &[0, 0, 0, 0]);
    for y in 8..56 {
        for x in 8..56 {
            img.px_mut(y, x).copy_from_slice(&[220, 0, 20, 255]);
        }
        for x in 56..60 {
            img.px_mut(y, x).copy_from_slice(&[255, 255, 255, 128]);
        }
        for x in 4..8 {
            img.px_mut(y, x).copy_from_slice(&[255, 255, 255, 127]);
        }
    }
    let cfg = config(&[
        ("method", s("box")),
        ("out_width", int(16)),
        ("out_height", int(16)),
        ("palette_size", int(4)),
        ("alpha_threshold", int(128)),
    ]);
    let loaded = img.loaded();
    let pre = preprocess::run(&loaded.rgb, &loaded.alpha, loaded.h, loaded.w, &cfg).unwrap();
    assert_eq!(pre.weight.len(), pre.mask.len());
    for (w, m) in pre.weight.iter().zip(&pre.mask) {
        assert_eq!(*w > 0.0, *m);
    }
    let mut levels: Vec<f64> = pre.weight.clone();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let expected = [0.0, 128.0 / 255.0, 1.0];
    assert_eq!(levels.len(), 3, "{levels:?}");
    for (a, e) in levels.iter().zip(expected) {
        assert!((a - e).abs() <= 1e-8 + 1e-5 * e.abs(), "{levels:?}");
    }
    // The pipeline runs on that input (Python also compared it with the reference
    // orchestration, which has no Rust counterpart).
    let native = run_image(&img, &cfg).unwrap();
    assert_eq!(shape(&native), (16, 16));
}

#[test]
fn run_loaded_logs_one_info_record_per_stage() {
    capture();
    let loaded = io::load(&fixture_path("checker_tiles"), io::MAX_INPUT_PIXELS).unwrap();
    let cfg = config(&[
        ("preset", s("background")),
        ("method", s("kopf")),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("tile_size", int(8)),
        ("kopf_max_iters", int(5)),
    ]);
    take_records();
    // caplog.at_level("INFO"): the logger's max level is INFO (the other tests capture at
    // INFO too, so switching the global level does not affect them).
    log::set_max_level(log::LevelFilter::Info);
    let logged = run_loaded(&loaded, &cfg);
    let stages: Vec<String> = take_records()
        .into_iter()
        .filter(|r| r.target == PIPELINE && r.level == log::Level::Info)
        .map(|r| r.message.split_whitespace().next().unwrap().to_string())
        .collect();
    assert_eq!(
        stages,
        [
            "preprocess",
            "downscale",
            "palette/quantize",
            "postprocess",
            "tiles"
        ]
    );
    log::set_max_level(log::LevelFilter::Trace);
    let verbose = run_loaded(&loaded, &cfg);
    log::set_max_level(log::LevelFilter::Info);
    assert!(take_records()
        .iter()
        .any(|r| r.message.starts_with("kopf iter 1/5")));
    // Logging never changes the result.
    assert_eq!(logged.indices, verbose.indices);
    assert_eq!(logged.palette, verbose.palette);
}

// ---------------------------------------------------------------------------------------
// tests/test_branches.py

fn mcda_palette_source_with(method: &str) {
    capture();
    // The pipeline's "palette/quantize" INFO line names the palette source that was used.
    let sources = || -> Vec<String> {
        take_records()
            .into_iter()
            .filter(|r| r.target == PIPELINE && r.message.starts_with("palette/quantize"))
            .map(|r| {
                let tail = r.message.rsplit_once('(').unwrap().1;
                tail.trim_end_matches(')').to_string()
            })
            .collect()
    };
    let image = small("gradient6", 4);
    let cfg = config(&merge(&[
        &[
            ("method", s(method)),
            ("palette_source", s("mcda")),
            ("palette_size", int(6)),
            ("kopf_max_iters", int(10)),
        ],
        &fast(),
        &raw(),
    ]));
    take_records();
    let result = run_image(&image, &cfg).unwrap();
    assert_eq!(sources(), ["mcda"]);
    assert!((2..=6).contains(&result.palette.len()));
    assert_eq!(shape(&result), (16, 16));
    assert!(result.indices.indices.iter().all(|&i| i >= 0));
    let median = run_image(
        &image,
        &replace(&cfg, &[("palette_source", s("median_cut"))]),
    )
    .unwrap();
    assert_eq!(sources(), ["median_cut"]); // median_cut does not go through mcda
    assert_ne!(median.palette, result.palette);
}

#[test]
fn mcda_palette_source_with_box_and_kopf() {
    for method in ["box", "kopf"] {
        mcda_palette_source_with(method);
    }
}
