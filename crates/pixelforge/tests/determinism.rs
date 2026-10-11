//! tests/test_determinism.py in process: two runs of every method, preset and fixture write
//! byte-identical files, the config hash is stable, `seed` only reaches gerstner, and the
//! parallel kernels give the same bits on 1, 2, 3 and 8 threads.
//!
//! The fresh-process case (`test_outputs_identical_across_processes_and_environments`,
//! different `PIXELFORGE_THREADS` per process) is tests/cli.rs
//! `outputs_identical_across_processes_and_thread_counts` (1, 2, 3 and 8 threads, every
//! method and both presets, on the binary).

mod common;

use std::collections::BTreeMap;
use std::path::Path;

use common::*;
use pixelforge::config::{Config, Value};
use pixelforge::{filters, io, kopf_run, parallel, preprocess};

const METHODS: [&str; 3] = ["box", "kopf", "gerstner"];
/// `sorted(PRESETS)`.
const PRESETS: [&str; 2] = ["background", "sprite"];
/// Fixtures 1, 4, 6.
const FIXTURE_NAMES: [&str; 3] = ["line_diag", "circle_alpha", "noisy_gradient"];

/// `PRESETS[preset]`: the background preset is run at 64×64 (4 × 4 tiles).
fn preset_args(preset: &str) -> Vec<(&'static str, Value)> {
    if preset == "background" {
        vec![("out_width", int(64)), ("out_height", int(64))]
    } else {
        Vec::new()
    }
}

/// `convert` of one fixture into `outdir` (as the CLI does: run, then save to
/// `outdir/<stem>`), returning every written file. `_meta.json` without
/// `stats.timings` and `environment` (wall clock and host only).
fn convert(name: &str, outdir: &Path, cfg: &Config) -> BTreeMap<String, Vec<u8>> {
    let result = run_path(&fixture_path(name), cfg).unwrap();
    result.save(&outdir.join(name)).unwrap();
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(outdir).unwrap() {
        let p = entry.unwrap().path();
        let file = p.file_name().unwrap().to_string_lossy().into_owned();
        let mut data = std::fs::read(&p).unwrap();
        if file.ends_with("_meta.json") {
            let mut meta: serde_json::Value = serde_json::from_slice(&data).unwrap();
            meta["stats"].as_object_mut().unwrap().remove("timings");
            meta.as_object_mut().unwrap().remove("environment");
            data = serde_json::to_vec(&meta).unwrap();
        }
        files.insert(file, data);
    }
    files
}

fn pngs(files: &BTreeMap<String, Vec<u8>>) -> BTreeMap<&String, &Vec<u8>> {
    files.iter().filter(|(k, _)| k.ends_with(".png")).collect()
}

/// `_assert_byte_identical`: the PNGs (the files Python compares) are the expected set and
/// byte-identical between two runs; so is every other output file (`_meta.json` without
/// timings and environment).
fn assert_byte_identical(name: &str, preset: &str, method: &str, extra: &[(&'static str, Value)]) {
    let tmp = TempDir::new("determinism");
    let cfg = config(&merge(&[
        &[("preset", s(preset)), ("method", s(method))],
        &preset_args(preset),
        extra,
    ]));
    let first = convert(name, &tmp.join("a"), &cfg);
    let second = convert(name, &tmp.join("b"), &cfg);
    let mut expected = vec![format!("{name}.png"), format!("{name}_preview.png")];
    if preset == "background" {
        expected.push(format!("{name}_tileset.png"));
    }
    expected.sort();
    let names: Vec<&String> = pngs(&first).into_keys().collect();
    assert_eq!(
        names,
        expected.iter().collect::<Vec<_>>(),
        "{name} {preset} {method}"
    );
    assert!(
        pngs(&first) == pngs(&second),
        "{name} {preset} {method}: PNGs differ"
    );
    assert_eq!(
        first.keys().collect::<Vec<_>>(),
        second.keys().collect::<Vec<_>>()
    );
    for (file, data) in &first {
        assert!(
            data == &second[file],
            "{name} {preset} {method}: {file} differs"
        );
    }
}

fn cli_output_is_byte_identical(method: &str) {
    for name in FIXTURE_NAMES {
        for preset in PRESETS {
            assert_byte_identical(name, preset, method, &[]);
        }
    }
}

#[test]
fn cli_output_is_byte_identical_box() {
    cli_output_is_byte_identical("box");
}

#[test]
fn cli_output_is_byte_identical_gerstner() {
    cli_output_is_byte_identical("gerstner");
}

#[test]
#[ignore] // slow (pytest -m slow)
fn cli_output_is_byte_identical_kopf() {
    cli_output_is_byte_identical("kopf");
}

#[test]
fn kopf_output_is_byte_identical_small() {
    // No keying: at 32 px the keyed 2-px line of line_diag covers no cell by half.
    let small = [
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("kopf_max_iters", int(10)),
        ("key_bg", b(false)),
    ];
    for name in FIXTURE_NAMES {
        for preset in PRESETS {
            assert_byte_identical(name, preset, "kopf", &small);
        }
    }
}

#[test]
fn config_hash_is_stable_across_runs() {
    for preset in PRESETS {
        let mut hashes = std::collections::BTreeSet::new();
        for method in METHODS {
            for _ in 0..3 {
                let cfg = config(&[("preset", s(preset)), ("method", s(method))]);
                hashes.insert(cfg.hash().unwrap());
            }
        }
        assert_eq!(hashes.len(), METHODS.len());
    }
    let default = config(&[]);
    let dict: Vec<(String, Value)> = default
        .to_dict()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let again = Config::new(&dict).unwrap();
    assert_eq!(default.hash().unwrap(), again.hash().unwrap());
}

/// `_decoded`: (colour type, palette, transparency, pixels) of a PNG; the text chunks (which
/// embed the config) excluded.
type Decoded = (png::ColorType, Option<Vec<u8>>, Option<Vec<u8>>, Vec<u8>);

fn decoded(path: &Path) -> Decoded {
    let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut buf).unwrap();
    buf.truncate(frame.buffer_size());
    let info = reader.info();
    (
        info.color_type,
        info.palette.as_ref().map(|p| p.to_vec()),
        info.trns.as_ref().map(|t| t.to_vec()),
        buf,
    )
}

#[derive(PartialEq, Debug)]
struct SeedRun {
    png: Decoded,
    preview: Decoded,
    palette: Vec<u8>,
    palette_hex: Vec<u8>,
}

fn convert_with_seed(outdir: &Path, method: &str, seed: Option<i64>) -> SeedRun {
    let mut pairs = vec![
        ("method", s(method)),
        ("out_height", int(16)),
        ("palette_size", int(4)),
        ("denoise", s("none")),
        ("kopf_max_iters", int(10)),
    ];
    if let Some(seed) = seed {
        pairs.push(("seed", int(seed)));
    }
    let result = run_path(&fixture_path("gradient6"), &config(&pairs)).unwrap();
    result.save(&outdir.join("gradient6")).unwrap();
    SeedRun {
        png: decoded(&outdir.join("gradient6.png")),
        preview: decoded(&outdir.join("gradient6_preview.png")),
        palette: std::fs::read(outdir.join("gradient6_palette.json")).unwrap(),
        palette_hex: std::fs::read(outdir.join("gradient6_palette.hex")).unwrap(),
    }
}

/// seed feeds only gerstner's optional MCDA jitter: box and kopf ignore it entirely. The
/// PNG bytes differ by the embedded config (seed is a field), so pixels, palette and
/// transparency are compared after decoding, plus the palette files byte for byte.
#[test]
fn seed_only_affects_gerstner() {
    for method in METHODS {
        let tmp = TempDir::new("seed");
        let seeded = convert_with_seed(&tmp.join("seeded"), method, Some(7));
        let unseeded = convert_with_seed(&tmp.join("unseeded"), method, None);
        if method == "gerstner" {
            // The jitter may or may not change the result; only record which.
            eprintln!("gerstner_seed_changes_output: {}", seeded != unseeded);
        } else {
            assert_eq!(seeded, unseeded, "{method}");
            let again = convert_with_seed(&tmp.join("again"), method, Some(12345));
            assert_eq!(again, unseeded, "{method}");
        }
    }
}

// --- thread counts ----------------------------------------------------------------------
// In process the kernels that take a thread count are called with 1, 2, 3 and 8 threads
// (PIXELFORGE_THREADS is never set in process: tests run on parallel threads). gerstner
// and the pipeline read the thread count from the environment; they are covered on the
// binary by tests/cli.rs `outputs_identical_across_processes_and_thread_counts`.

const THREADS: [usize; 4] = [1, 2, 3, 8];

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn kopf_identical_across_thread_counts() {
    for name in FIXTURE_NAMES {
        let cfg = config(&[
            ("preset", s("sprite")),
            ("method", s("kopf")),
            ("out_width", int(32)),
            ("out_height", int(32)),
            ("kopf_max_iters", int(10)),
            ("key_bg", b(false)),
        ]);
        let loaded = io::load(&fixture_path(name), io::MAX_INPUT_PIXELS).unwrap();
        let pre = preprocess::run(&loaded.rgb, &loaded.alpha, loaded.h, loaded.w, &cfg).unwrap();
        // More than one band (32 rows each), so the threads really split the work.
        assert!(parallel::n_bands(pre.h) > 1, "{name}: {} rows", pre.h);
        let (tw, th) = (pre.target_width() as usize, pre.target_height() as usize);
        let runs: Vec<_> = THREADS
            .iter()
            .map(|&t| {
                kopf_run::run(
                    &pre.lab,
                    &pre.mask,
                    pre.h,
                    pre.w,
                    tw,
                    th,
                    &cfg,
                    Some(&pre.weight),
                    t,
                )
            })
            .collect();
        for (run, t) in runs.iter().zip(THREADS).skip(1) {
            assert_eq!(bits(&run.small_lab), bits(&runs[0].small_lab), "{name} {t}");
            assert_eq!(run.small_mask, runs[0].small_mask, "{name} {t}");
            assert_eq!(
                format!("{:?}", run.stats),
                format!("{:?}", runs[0].stats),
                "{name} {t}"
            );
        }
    }
}

#[test]
fn bilateral_identical_across_thread_counts() {
    let image = fixture("noisy_gradient");
    let rgb: Vec<f64> = image.data.iter().map(|&v| f64::from(v) / 255.0).collect();
    let runs: Vec<Vec<u64>> = THREADS
        .iter()
        .map(|&t| {
            bits(&filters::denoise_bilateral(
                &rgb, image.h, image.w, 0.08, 1.5, t,
            ))
        })
        .collect();
    for (run, t) in runs.iter().zip(THREADS).skip(1) {
        assert!(run == &runs[0], "{t} threads");
    }
}

#[test]
fn run_bands_identical_across_thread_counts() {
    // Per-band float sums merged in band order: the same bits for every thread count.
    let rows = 200;
    let values: Vec<f64> = (0..rows * 7)
        .map(|i| ((i * 7919) % 1000) as f64 * 1.000_000_1e-3 - 0.37)
        .collect();
    let total = |threads: usize| -> (Vec<u64>, u64) {
        let bands: Vec<(usize, usize)> = (0..parallel::n_bands(rows))
            .map(|b| parallel::band_rows(b, rows))
            .collect();
        let sums = parallel::run_bands(bands, threads, |_, (y0, y1)| {
            values[y0 * 7..y1 * 7].iter().sum::<f64>()
        });
        (bits(&sums), sums.iter().sum::<f64>().to_bits())
    };
    let first = total(1);
    for t in &THREADS[1..] {
        assert_eq!(total(*t), first, "{t} threads");
    }
}
