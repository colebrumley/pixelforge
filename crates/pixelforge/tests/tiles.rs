//! Port of tests/test_tiles.py (and test_branches.py's tile-size check): tile extraction,
//! dedupe with flips and tolerance, reconstruction, seamless edges, Tiled map and CSV
//! export, tileset column counts.

mod common;
use common::*;

use std::path::Path;

use pixelforge::color;
use pixelforge::config::Config;
use pixelforge::io;
use pixelforge::pipeline::{self, Output};
use pixelforge::pyjson::Json;
use pixelforge::tiles::{self, TileConfig, Tilemap, Tiles};

// ---------------------------------------------------------------------------------------
// Index images

/// A row-major `i64` index image (`np.ndarray` of int64, `< 0` transparent).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Idx {
    h: usize,
    w: usize,
    v: Vec<i64>,
}

impl Idx {
    fn full(h: usize, w: usize, value: i64) -> Idx {
        Idx {
            h,
            w,
            v: vec![value; h * w],
        }
    }

    fn zeros(h: usize, w: usize) -> Idx {
        Idx::full(h, w, 0)
    }

    fn set(&mut self, y: usize, x: usize, value: i64) {
        self.v[y * self.w + x] = value;
    }

    fn get(&self, y: usize, x: usize) -> i64 {
        self.v[y * self.w + x]
    }

    /// `a[:, ::-1]`.
    fn fliplr(&self) -> Idx {
        let mut out = self.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                out.set(y, x, self.get(y, self.w - 1 - x));
            }
        }
        out
    }

    /// `a[::-1, :]`.
    fn flipud(&self) -> Idx {
        let mut out = self.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                out.set(y, x, self.get(self.h - 1 - y, x));
            }
        }
        out
    }
}

/// `np.hstack(parts)`.
fn hstack(parts: &[&Idx]) -> Idx {
    let h = parts[0].h;
    let w = parts.iter().map(|p| p.w).sum();
    let mut v = Vec::with_capacity(h * w);
    for y in 0..h {
        for p in parts {
            assert_eq!(p.h, h);
            v.extend_from_slice(&p.v[y * p.w..(y + 1) * p.w]);
        }
    }
    Idx { h, w, v }
}

/// `np.vstack(parts)`.
fn vstack(parts: &[Idx]) -> Idx {
    let w = parts[0].w;
    let mut v = Vec::new();
    for p in parts {
        assert_eq!(p.w, w);
        v.extend_from_slice(&p.v);
    }
    Idx {
        h: v.len() / w,
        w,
        v,
    }
}

/// `np.block(rows)`.
fn block(rows: &[&[&Idx]]) -> Idx {
    vstack(&rows.iter().map(|r| hstack(r)).collect::<Vec<_>>())
}

fn tile_config(pairs: &[(&str, pixelforge::config::Value)]) -> TileConfig {
    TileConfig::from(&config(pairs))
}

fn extract(idx: &Idx, palette_lab: &[f64], config: &TileConfig) -> Tiles {
    tiles::extract(&idx.v, idx.h, idx.w, palette_lab, config).unwrap()
}

/// `tiles.reconstruct(out.tiles, out.tilemap)` as an index image.
fn reconstruct(stack: &[i64], tilemap: &Tilemap) -> Idx {
    let (v, h, w) = tiles::reconstruct(stack, tilemap).unwrap();
    Idx { h, w, v }
}

fn lab_gray(lightness: &[f64]) -> Vec<f64> {
    lightness.iter().flat_map(|&l| [l, 0.0, 0.0]).collect()
}

/// `palette_lab[i]` with numpy's negative indexing (-1 = the last entry).
fn lab_of(palette_lab: &[f64], i: i64) -> &[f64] {
    let k = palette_lab.len() / 3;
    let i = if i < 0 {
        (k as i64 + i) as usize
    } else {
        i as usize
    };
    &palette_lab[i * 3..i * 3 + 3]
}

fn stat(out: &Output, key: &str) -> i64 {
    match &out.stats {
        Json::Object(items) => match items.iter().find(|(k, _)| k == key) {
            Some((_, Json::Int(v))) => *v,
            other => panic!("stats[{key:?}] = {other:?}"),
        },
        other => panic!("stats is not an object: {other:?}"),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

// ---------------------------------------------------------------------------------------

#[test]
fn test_checker_tiles_dedupe_with_flips() {
    let config = config(&[
        ("preset", s("background")),
        ("out_width", int(256)),
        ("out_height", int(256)),
    ]);
    assert!(config.tileset && config.tile_size == 16);
    let result = run_path(&fixture_path("checker_tiles"), &config).unwrap();
    let n_tiles = stat(&result, "tiles") as usize;
    let tilemap = result.tilemap.as_ref().unwrap();
    let entries = &tilemap.tiles;
    let flips_used = entries.iter().any(|&[_, fh, fv]| fh != 0 || fv != 0);
    assert!(n_tiles == 2 || (n_tiles == 1 && flips_used));
    let changed = stat(&result, "tiles_rerender_px_changed");
    assert!((0..=256 * 256).contains(&changed));
    assert!(tilemap.width_tiles == 16 && tilemap.height_tiles == 16);
    let cols = tilemap.tileset_columns as usize;
    assert!(entries.len() == 256 && cols == (n_tiles as f64).sqrt().ceil() as usize);

    // The tilemap reconstructs the index image exactly.
    let t = tilemap.tile_size as usize;
    let sheet = result.tileset_indices.as_ref().unwrap();
    let mut stack = Vec::with_capacity(n_tiles * t * t);
    for i in 0..n_tiles {
        let (y0, x0) = ((i / cols) * t, (i % cols) * t);
        for y in y0..y0 + t {
            stack.extend_from_slice(&sheet.indices[y * sheet.w + x0..y * sheet.w + x0 + t]);
        }
    }
    let back = reconstruct(&stack, tilemap);
    assert_eq!((back.h, back.w), (result.indices.h, result.indices.w));
    assert_eq!(back.v, result.indices.indices);
    let tileset = result.tileset.as_ref().unwrap();
    assert_eq!((sheet.h, sheet.w), (16, cols * 16));
    assert_eq!(tileset.len(), 16 * cols * 16 * 4);
    // Every output color is one of the two checker colors (the denoiser may shift them by a
    // level or two, which median cut keeps as separate palette entries).
    let source = color::rgb8_to_lab(&[60, 90, 160, 230, 200, 90]);
    let mut used: Vec<i64> = result.indices.indices.clone();
    used.sort_unstable();
    used.dedup();
    let worst = used
        .iter()
        .map(|&i| {
            let lab = lab_of(&result.palette_lab, i);
            source
                .chunks_exact(3)
                .map(|s| color::delta_e(lab, s)[0])
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(worst < 12.0, "{worst}");
}

#[test]
fn test_extract_exact_near_and_flipped() {
    let palette_lab = color::rgb8_to_lab(&[0, 0, 0, 255, 255, 255, 250, 250, 250]);
    let mut a = Idx::zeros(4, 4);
    for y in 0..4 {
        a.set(y, 0, 1); // asymmetric tile
    }
    let mut near = a.clone();
    near.set(3, 3, 0);
    for y in 0..4 {
        near.set(y, 0, 2); // almost the same white
    }
    let other = Idx::full(4, 4, 1);
    let idx = block(&[&[&a, &a.fliplr()], &[&near, &other]]);
    let out = extract(&idx, &palette_lab, &tile_config(&[("tile_size", int(4))]));
    assert_eq!(out.n_tiles, 2);
    assert_eq!(
        out.tilemap.tiles,
        vec![[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]]
    );
    let strict = extract(
        &idx,
        &palette_lab,
        &tile_config(&[("tile_size", int(4)), ("tile_dedupe_tolerance", float(0.0))]),
    );
    assert_eq!(strict.n_tiles, 3);
    assert_eq!(reconstruct(&strict.tiles, &strict.tilemap), idx);
}

#[test]
fn test_seamless_edges_match() {
    let config = config(&[
        ("preset", s("background")),
        ("method", s("kopf")),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("tileset", b(false)),
        ("seamless", b(true)),
    ]);
    let result = run_path(&fixture_path("seamless_src"), &config).unwrap();
    let (h, w) = (result.indices.h, result.indices.w);
    assert_eq!((h, w), (32, 32));
    let lab = |y: usize, x: usize| lab_of(&result.palette_lab, result.indices.indices[y * w + x]);
    let de = |p: &[f64], q: &[f64]| color::delta_e(p, q)[0];
    let left_right = mean(
        &(0..h)
            .map(|y| de(lab(y, 0), lab(y, w - 1)))
            .collect::<Vec<_>>(),
    );
    let top_bottom = mean(
        &(0..w)
            .map(|x| de(lab(0, x), lab(h - 1, x)))
            .collect::<Vec<_>>(),
    );
    // lab[:, 1:-2] against lab[:, 2:-1].
    let mut inner = Vec::new();
    for y in 0..h {
        for x in 1..w - 2 {
            inner.push(de(lab(y, x), lab(y, x + 1)));
        }
    }
    let interior = mean(&inner);
    assert!(interior > 0.0);
    assert!(left_right < 3.0 * interior, "{left_right} vs {interior}");
    assert!(top_bottom < 3.0 * interior, "{top_bottom} vs {interior}");
}

#[test]
fn test_sparse_stars_stay_distinct() {
    // The review's "6 stars": a near-black sky with 6 single white pixels, each in its own
    // tile at a different spot. Mean ΔE per tile is ~0.4, far under the default tolerance,
    // but the star pixel's ΔE (~95) exceeds the per-pixel cap, so nothing is lost.
    let palette_lab = color::rgb8_to_lab(&[8, 8, 12, 255, 255, 255]);
    let mut idx = Idx::zeros(64, 64);
    for (y, x) in [(1, 2), (5, 22), (20, 7), (27, 41), (40, 50), (54, 13)] {
        idx.set(y, x, 1);
    }
    let config = tile_config(&[("tile_size", int(16))]);
    assert!(config.tile_dedupe_tolerance > 0.0);
    let out = extract(&idx, &palette_lab, &config);
    assert_eq!(out.n_tiles, 7); // 6 star tiles + 1 plain sky tile
    assert_eq!(reconstruct(&out.tiles, &out.tilemap), idx);
}

#[test]
fn test_single_pixel_small_delta_still_merges() {
    let palette_lab = lab_gray(&[20.0, 24.0]); // ΔE 4, under the per-pixel cap
    let a = Idx::zeros(16, 16);
    let mut b = a.clone();
    b.set(7, 9, 1);
    let out = extract(
        &hstack(&[&a, &b]),
        &palette_lab,
        &tile_config(&[("tile_size", int(16))]),
    );
    assert_eq!(out.n_tiles, 1);
    assert_eq!(out.tilemap.tiles, vec![[0, 0, 0], [0, 0, 0]]);
}

#[test]
fn test_near_duplicate_picks_nearest_not_earliest() {
    let palette_lab = lab_gray(&[50.0, 51.5, 52.5]);
    let a = Idx::zeros(4, 4); // L 50
    let b = Idx::full(4, 4, 2); // L 52.5: ΔE 2.5 from A, stays distinct
    let c = Idx::full(4, 4, 1); // L 51.5: ΔE 1.5 from A, 1.0 from B
    let out = extract(
        &hstack(&[&a, &b, &c]),
        &palette_lab,
        &tile_config(&[("tile_size", int(4))]),
    );
    assert_eq!(out.n_tiles, 2);
    assert_eq!(out.tilemap.tiles, vec![[0, 0, 0], [1, 0, 0], [1, 0, 0]]);
}

#[test]
fn test_near_duplicate_tie_prefers_lowest_id() {
    let palette_lab = lab_gray(&[50.0, 51.0, 52.5]);
    let a = Idx::zeros(4, 4);
    let b = Idx::full(4, 4, 2);
    let c = Idx::full(4, 4, 1); // ΔE 1.0 from A, 1.5 from B
    let cfg = tile_config(&[("tile_size", int(4))]);
    let out = extract(&hstack(&[&a, &b, &c]), &palette_lab, &cfg);
    assert_eq!(out.tilemap.tiles[2], [0, 0, 0]);
    let tie = lab_gray(&[50.0, 51.25, 52.5]); // equidistant from A and B
    let out = extract(&hstack(&[&a, &b, &c]), &tie, &cfg);
    assert_eq!(out.tilemap.tiles[2], [0, 0, 0]);
}

#[test]
fn test_flipped_copies_map_to_one_tile() {
    let palette_lab = color::rgb8_to_lab(&[0, 0, 0, 255, 0, 0, 0, 0, 255]);
    let mut a = Idx::zeros(4, 4);
    a.set(0, 0, 1);
    a.set(0, 1, 1);
    a.set(1, 0, 2);
    a.set(3, 3, -1); // transparency survives flips too
    let idx = hstack(&[&a, &a.fliplr(), &a.flipud(), &a.flipud().fliplr(), &a]);
    for tol in [0.0, 2.0] {
        let out = extract(
            &idx,
            &palette_lab,
            &tile_config(&[("tile_size", int(4)), ("tile_dedupe_tolerance", float(tol))]),
        );
        assert_eq!(out.n_tiles, 1);
        assert_eq!(
            out.tilemap.tiles,
            vec![[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0]]
        );
        assert_eq!(reconstruct(&out.tiles, &out.tilemap), idx);
    }
}

#[test]
fn test_transparency_mismatch_never_merges() {
    let palette_lab = color::rgb8_to_lab(&[0, 0, 0]);
    let a = Idx::zeros(16, 16);
    let mut b = a.clone();
    b.set(0, 0, -1); // mean ΔE 0.39, but one clear/opaque pixel
    let out = extract(
        &hstack(&[&a, &b]),
        &palette_lab,
        &tile_config(&[("tile_size", int(16))]),
    );
    assert_eq!(out.n_tiles, 2);
}

/// A small deterministic generator (splitmix64) in place of `np.random.default_rng(7)`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `rng.integers(lo, hi)` (hi exclusive).
    fn integer(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo) as u64) as i64
    }

    fn idx(&mut self, h: usize, w: usize, lo: i64, hi: i64) -> Idx {
        Idx {
            h,
            w,
            v: (0..h * w).map(|_| self.integer(lo, hi)).collect(),
        }
    }
}

#[test]
fn test_zero_tolerance_round_trips() {
    // Not numpy's PCG64 stream: a splitmix64 stream draws the same kinds of values
    // (palette 0..256, indices -1..6) in the same layout.
    let mut rng = Rng(7);
    let rgb: Vec<u8> = (0..18).map(|_| rng.integer(0, 256) as u8).collect();
    let palette_lab = color::rgb8_to_lab(&rgb);
    for _ in 0..5 {
        let base = rng.idx(8, 8, -1, 6);
        // A few repeated / flipped / one-pixel-off tiles so that exact merges do happen.
        let mut near = base.clone();
        near.set(2, 5, (near.get(2, 5) + 1).rem_euclid(6));
        let random = rng.idx(8, 8, -1, 6);
        let idx = block(&[
            &[&base, &base.fliplr(), &near],
            &[&random, &base.flipud(), &base],
        ]);
        let out = extract(
            &idx,
            &palette_lab,
            &tile_config(&[("tile_size", int(8)), ("tile_dedupe_tolerance", float(0.0))]),
        );
        assert_eq!(out.n_tiles, 3);
        assert_eq!(reconstruct(&out.tiles, &out.tilemap), idx);
    }
}

#[test]
fn test_tileset_columns() {
    let cases: [(usize, i64, i64); 9] = [
        (1, 0, 1),
        (2, 0, 2),
        (4, 0, 2),
        (5, 0, 3),
        (15, 0, 4),
        (16, 0, 4),
        (17, 0, 5),
        (15, 16, 16),
        (15, 3, 3),
    ];
    for (n, requested, expected) in cases {
        assert_eq!(tiles::tileset_columns(n, requested), expected, "n={n}");
        let lightness: Vec<f64> = (0..n).map(|i| (3 * i) as f64).collect();
        let palette_lab = lab_gray(&lightness);
        let parts: Vec<Idx> = (0..n).map(|i| Idx::full(2, 2, i as i64)).collect();
        let idx = hstack(&parts.iter().collect::<Vec<_>>());
        let out = extract(
            &idx,
            &palette_lab,
            &tile_config(&[("tile_size", int(2)), ("tileset_columns", int(requested))]),
        );
        assert!(out.n_tiles == n && out.tilemap.tileset_columns == expected);
        let expected = expected as usize;
        let (sheet, sh, sw) = tiles::tileset_indices(&out.tiles, 2, expected);
        let rows = n.div_ceil(expected);
        assert_eq!((sh, sw), (rows * 2, expected * 2));
        // Only the last row has padding.
        assert!(sheet[..(rows - 1) * 2 * sw].iter().all(|&v| v >= 0));
        assert_eq!(
            sheet.iter().filter(|&&v| v < 0).count(),
            (rows * expected - n) * 4
        );
    }
}

#[test]
fn test_tileset_columns_validated() {
    for bad in [-1, 257] {
        let err = config_err(&[("tileset_columns", int(bad))]);
        assert!(err.contains("tileset_columns"), "{err}");
    }
}

#[test]
fn test_gids_and_csv() {
    let tilemap = Tilemap {
        tile_size: 4,
        width_tiles: 3,
        height_tiles: 2,
        tileset_columns: 2,
        tiles: vec![
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [2, 1, 1],
            [1, 0, 0],
            [0, 0, 0],
        ],
    };
    assert_eq!(
        tiles::gids(&tilemap, true),
        vec![1, 0x8000_0002, 0x4000_0001, 0xC000_0003, 2, 1]
    );
    assert_eq!(tiles::tilemap_csv(&tilemap), "1,2,1\n3,2,1\n");
}

/// 32×32 RGBA: an asymmetric 8×8 motif in all four flips plus two plain tiles.
fn flip_scene() -> Image {
    let mut motif = Image::filled(8, 8, &[20, 30, 120, 255]);
    for y in 0..8 {
        for x in 0..8 {
            if y < 3 && x < 5 {
                motif.px_mut(y, x).copy_from_slice(&[230, 200, 40, 255]);
            }
            if y >= 5 && x >= 6 {
                motif.px_mut(y, x).copy_from_slice(&[200, 40, 40, 255]);
            }
        }
    }
    let flipped = |fh: bool, fv: bool| {
        let mut out = motif.clone();
        for y in 0..8 {
            for x in 0..8 {
                let (sy, sx) = (if fv { 7 - y } else { y }, if fh { 7 - x } else { x });
                out.px_mut(y, x).copy_from_slice(motif.px(sy, sx));
            }
        }
        out
    };
    let plain = Image::filled(8, 8, &[40, 160, 60, 255]);
    let clear = Image::filled(8, 8, &[0, 0, 0, 0]);
    let row = [
        motif.clone(),
        flipped(true, false),
        flipped(false, true),
        flipped(true, true),
    ];
    let rev: Vec<Image> = row.iter().rev().cloned().collect();
    let rows: [Vec<Image>; 4] = [
        row.to_vec(),
        vec![plain.clone(), clear.clone(), plain.clone(), motif.clone()],
        rev,
        vec![clear.clone(), clear, motif, plain],
    ];
    let mut scene = Image::filled(32, 32, &[0, 0, 0, 0]);
    for (ty, r) in rows.iter().enumerate() {
        for (tx, tile) in r.iter().enumerate() {
            for y in 0..8 {
                for x in 0..8 {
                    scene
                        .px_mut(ty * 8 + y, tx * 8 + x)
                        .copy_from_slice(tile.px(y, x));
                }
            }
        }
    }
    scene
}

fn scene_config(overrides: &[(&'static str, pixelforge::config::Value)]) -> Config {
    let base = vec![
        ("preset", s("background")),
        ("method", s("box")),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("tile_size", int(8)),
        ("palette_size", int(4)),
        ("denoise", s("none")),
        ("dither", s("none")),
        ("remove_orphans", b(false)),
        ("fix_jaggies", b(false)),
        ("palette_ramps", b(false)),
    ];
    config(&merge(&[&base, overrides]))
}

fn run_scene(config: &Config) -> Output {
    let scene = flip_scene();
    let loaded = io::from_rgba(&scene.data, scene.h, scene.w, String::new());
    pipeline::run_loaded(&loaded, config, Vec::new(), false).unwrap()
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The PNG chunks `(type, data)` of a file, in order.
fn png_chunks(path: &Path) -> Vec<([u8; 4], Vec<u8>)> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let mut pos = 8;
    let mut out = Vec::new();
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = bytes[pos + 4..pos + 8].try_into().unwrap();
        out.push((kind, bytes[pos + 8..pos + 8 + len].to_vec()));
        pos += 12 + len;
    }
    out
}

fn chunk(chunks: &[([u8; 4], Vec<u8>)], kind: &[u8; 4]) -> Option<Vec<u8>> {
    chunks
        .iter()
        .find(|(k, _)| k == kind)
        .map(|(_, d)| d.clone())
}

/// `Image.open(path).size`: (width, height) from IHDR.
fn png_size(path: &Path) -> (i64, i64) {
    let ihdr = chunk(&png_chunks(path), b"IHDR").unwrap();
    let be = |i: usize| i64::from(u32::from_be_bytes(ihdr[i..i + 4].try_into().unwrap()));
    (be(0), be(4))
}

/// `np.asarray(Image.open(path).convert("RGBA"))`, flattened.
fn decode_rgba(path: &Path) -> Vec<u8> {
    let loaded = io::load(path, io::MAX_INPUT_PIXELS).unwrap();
    let mut out = Vec::with_capacity(loaded.h * loaded.w * 4);
    for (p, &a) in loaded.rgb.chunks_exact(3).zip(&loaded.alpha) {
        out.extend(p.iter().map(|v| (v * 255.0).round() as u8));
        out.push(a);
    }
    out
}

#[test]
fn test_tiled_export_matches_tilemap() {
    let tmp = TempDir::new("tiled_export");
    let result = run_scene(&scene_config(&[]));
    let paths = result.save(&tmp.join("scene")).unwrap();
    let path_of = |name: &str| {
        paths
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, p)| p.clone())
            .unwrap()
    };
    assert_eq!(path_of("tilemap_tmj"), tmp.join("scene.tmj"));
    let tilemap = read_json(&tmp.join("scene_tilemap.json"));
    let mut keys: Vec<&str> = tilemap
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "height_tiles",
            "tile_size",
            "tiles",
            "tileset_columns",
            "width_tiles"
        ]
    );
    let entries: Vec<[i64; 3]> = tilemap["tiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let e = e.as_array().unwrap();
            [0, 1, 2].map(|i| e[i].as_i64().unwrap())
        })
        .collect();
    assert!(entries.iter().any(|e| e[1] != 0) && entries.iter().any(|e| e[2] != 0));
    let n = stat(&result, "tiles");

    let tmj = read_json(&tmp.join("scene.tmj"));
    assert!(tmj["orientation"] == "orthogonal" && tmj["renderorder"] == "right-down");
    let geom = ["tilewidth", "tileheight", "width", "height"].map(|k| tmj[k].as_i64().unwrap());
    assert_eq!(geom, [8, 8, 4, 4]);
    let layers = tmj["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 1);
    let layer = &layers[0];
    assert!(layer["name"] == "background" && layer["type"] == "tilelayer");
    assert_eq!(
        (layer["width"].as_i64(), layer["height"].as_i64()),
        (Some(4), Some(4))
    );
    let data: Vec<i64> = layer["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(data.len(), 4 * 4);
    for (gid, &[tile_id, fh, fv]) in data.iter().zip(&entries) {
        assert_eq!(gid & tiles::GID_FLIP_H != 0, fh != 0);
        assert_eq!(gid & tiles::GID_FLIP_V != 0, fv != 0);
        assert_eq!(gid & 0x1FFF_FFFF, tile_id + 1);
        assert!((1..=n).contains(&(gid & 0x1FFF_FFFF)));
    }

    let tilesets = tmj["tilesets"].as_array().unwrap();
    assert_eq!(tilesets.len(), 1);
    let tileset = &tilesets[0];
    assert!(tileset["firstgid"] == 1 && tileset["tilecount"] == n);
    assert!(tileset["margin"] == 0 && tileset["spacing"] == 0);
    assert!(tileset["tilewidth"] == 8 && tileset["tileheight"] == 8);
    let columns = tileset["columns"].as_i64().unwrap();
    assert_eq!(columns, tilemap["tileset_columns"].as_i64().unwrap());
    assert_eq!(columns, (n as f64).sqrt().ceil() as i64);
    assert_eq!(tileset["name"], "scene");
    let image = tmp.join(tileset["image"].as_str().unwrap());
    assert_eq!(image.file_name().unwrap(), "scene_tileset.png");
    assert_eq!(image, path_of("tileset"));
    let size = png_size(&image);
    assert_eq!(
        size,
        (
            tileset["imagewidth"].as_i64().unwrap(),
            tileset["imageheight"].as_i64().unwrap()
        )
    );
    assert_eq!(size.0, columns * 8);

    let csv = std::fs::read_to_string(tmp.join("scene_tilemap.csv")).unwrap();
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 4);
    let values: Vec<i64> = rows
        .iter()
        .flat_map(|r| r.split(',').map(|v| v.parse::<i64>().unwrap()))
        .collect();
    assert_eq!(values, entries.iter().map(|e| e[0] + 1).collect::<Vec<_>>());
}

/// Pillow's `info["transparency"]` for a palette PNG whose tRNS holds only 0 and 255: the
/// index of the first 0.
fn transparency(chunks: &[([u8; 4], Vec<u8>)]) -> Option<usize> {
    chunk(chunks, b"tRNS").and_then(|t| t.iter().position(|&a| a == 0))
}

#[test]
fn test_image_and_tileset_share_palette_layout() {
    for layout in ["first", "last"] {
        let tmp = TempDir::new("palette_layout");
        let result = run_scene(&scene_config(&[("transparent_index", s(layout))]));
        assert!(result.indices.indices.iter().any(|&i| i < 0));
        result.save(&tmp.join("scene")).unwrap();
        let im = png_chunks(&tmp.join("scene.png"));
        let sheet = png_chunks(&tmp.join("scene_tileset.png"));
        assert_eq!(im[0].1[9], 3, "{layout}: scene.png is not a palette PNG");
        assert_eq!(
            sheet[0].1[9], 3,
            "{layout}: the tileset is not a palette PNG"
        );
        // getpalette() and info["transparency"].
        assert_eq!(chunk(&im, b"PLTE"), chunk(&sheet, b"PLTE"), "{layout}");
        assert_eq!(chunk(&im, b"tRNS"), chunk(&sheet, b"tRNS"), "{layout}");
        assert!(transparency(&im).is_some(), "{layout}");
        assert_eq!(transparency(&im), transparency(&sheet), "{layout}");
        if layout == "first" {
            assert_eq!(transparency(&im), Some(0));
        }
        assert_eq!(
            decode_rgba(&tmp.join("scene.png")),
            result.image,
            "{layout}"
        );
        assert_eq!(
            &decode_rgba(&tmp.join("scene_tileset.png")),
            result.tileset.as_ref().unwrap(),
            "{layout}"
        );
    }
}

// From tests/test_branches.py.
#[test]
fn test_tiles_extract_rejects_sizes_not_divisible_by_the_tile() {
    for (h, w) in [(16, 24), (24, 16), (20, 20)] {
        let idx = vec![0i64; h * w];
        let err = tiles::extract(
            &idx,
            h,
            w,
            &[0.0, 0.0, 0.0],
            &tile_config(&[("tile_size", int(16))]),
        )
        .unwrap_err();
        let expected = format!("tile_size (16) must divide the output size ({w}x{h})");
        assert!(err.contains(&expected), "{err}");
    }
}
