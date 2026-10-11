//! The downscalers' behavioural tests: ports of tests/test_box.py, tests/test_kopf.py,
//! tests/test_gerstner.py and test_branches.py::test_seamless_with_gerstner.
//!
//! The kernels take LAB arrays and masks, built with `pixelforge::color` (or from a
//! preprocessed fixture, the `preprocessed` fixture of tests/conftest.py).

mod common;
use common::*;

use pixelforge::annealer::{SplitMix64, Tunables};
use pixelforge::config::{Config, Value};
use pixelforge::downscale::{neighbor_mean4, Downscaled, StatValue};
use pixelforge::preprocess::{self, Preprocessed};
use pixelforge::{boxfilter, color, gerstner, gerstner_run, io, kopf_run};

/// Threads for kopf's per-pixel loops (its results do not depend on the count).
const THREADS: usize = 2;
/// "dark" = closer to black than to white (test_kopf.py's DARK_L).
const DARK_L: f64 = 50.0;

// ---------------------------------------------------------------------------------------
// Helpers

/// The `preprocessed` fixture: load + preprocess a fixture, returning (Preprocessed, Config).
fn preprocessed(name: &str, pairs: &[(&str, Value)]) -> (Preprocessed, Config) {
    let config = config(pairs);
    let loaded = io::load(&fixture_path(name), io::MAX_INPUT_PIXELS).unwrap();
    let pre = preprocess::run(&loaded.rgb, &loaded.alpha, loaded.h, loaded.w, &config)
        .unwrap_or_else(|e| panic!("preprocess {name}: {e}"));
    (pre, config)
}

fn box_run(lab: &[f64], mask: &[bool], h: usize, w: usize, wo: usize, ho: usize) -> Downscaled {
    boxfilter::run(lab, mask, h, w, wo, ho, None).unwrap()
}

fn box_pre(pre: &Preprocessed, wo: usize, ho: usize, weight: Option<&[f64]>) -> Downscaled {
    boxfilter::run(&pre.lab, &pre.mask, pre.h, pre.w, wo, ho, weight).unwrap()
}

fn kopf_pre(pre: &Preprocessed, wo: usize, ho: usize, config: &Config) -> Downscaled {
    kopf_run::run(
        &pre.lab, &pre.mask, pre.h, pre.w, wo, ho, config, None, THREADS,
    )
}

fn gerstner_pre(
    pre: &Preprocessed,
    wo: usize,
    ho: usize,
    config: &Config,
    weight: Option<&[f64]>,
) -> Downscaled {
    gerstner_run::run(&pre.lab, &pre.mask, pre.h, pre.w, wo, ho, config, weight).unwrap()
}

fn stat(out: &Downscaled, name: &str) -> StatValue {
    out.stats
        .iter()
        .find(|(k, _)| *k == name)
        .unwrap_or_else(|| panic!("no stat {name}: {:?}", out.stats))
        .1
}

fn stat_int(out: &Downscaled, name: &str) -> i64 {
    match stat(out, name) {
        StatValue::Int(i) => i,
        v => panic!("{name} is not an int: {v:?}"),
    }
}

fn stat_bool(out: &Downscaled, name: &str) -> bool {
    match stat(out, name) {
        StatValue::Bool(v) => v,
        v => panic!("{name} is not a bool: {v:?}"),
    }
}

fn stat_float(out: &Downscaled, name: &str) -> f64 {
    match stat(out, name) {
        StatValue::Float(v) => v,
        v => panic!("{name} is not a float: {v:?}"),
    }
}

/// CIE76 ΔE between two LAB triples.
fn de(a: &[f64], b: &[f64]) -> f64 {
    color::delta_e(&a[..3], &b[..3])[0]
}

/// `np.isclose(a, b)` (rtol 1e-5, atol 1e-8).
fn isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// LAB triple of output pixel (y, x).
fn px(out: &Downscaled, y: usize, x: usize) -> &[f64] {
    let i = 3 * (y * out.w + x);
    &out.small_lab[i..i + 3]
}

fn l_at(out: &Downscaled, y: usize, x: usize) -> f64 {
    out.small_lab[3 * (y * out.w + x)]
}

fn indices(out: &Downscaled) -> &[i64] {
    out.indices.as_deref().expect("gerstner indices")
}

fn palette(out: &Downscaled) -> &[f64] {
    out.palette_lab.as_deref().expect("gerstner palette")
}

fn palette_len(out: &Downscaled) -> usize {
    palette(out).len() / 3
}

/// Minimum L per row over the opaque cells (`np.where(mask, L, inf).min(axis=1)`).
fn row_min_l(out: &Downscaled) -> Vec<f64> {
    (0..out.h)
        .map(|y| {
            (0..out.w)
                .filter(|&x| out.small_mask[y * out.w + x])
                .map(|x| l_at(out, y, x))
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// `mask.reshape(ho, f, wo, f).any(axis=(1, 3))`: the output cells holding an opaque pixel.
fn support(mask: &[bool], h: usize, w: usize, ho: usize, wo: usize) -> Vec<bool> {
    assert!(h.is_multiple_of(ho) && w.is_multiple_of(wo) && h / ho == w / wo);
    let f = h / ho;
    let mut out = vec![false; ho * wo];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                out[(y / f) * wo + x / f] = true;
            }
        }
    }
    out
}

/// Number of 8-connected components of the true cells (`ndimage.label`, 3×3 structure).
fn components8(grid: &[bool], h: usize, w: usize) -> usize {
    let mut seen = vec![false; h * w];
    let mut n = 0;
    for start in 0..h * w {
        if !grid[start] || seen[start] {
            continue;
        }
        n += 1;
        seen[start] = true;
        let mut stack = vec![start];
        while let Some(i) = stack.pop() {
            let (y, x) = ((i / w) as i64, (i % w) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if grid[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
    }
    n
}

/// A solid LAB image.
fn solid_lab(h: usize, w: usize, lab: &[f64]) -> Vec<f64> {
    lab[..3].repeat(h * w)
}

// ---------------------------------------------------------------------------------------
// tests/test_box.py

#[test]
fn test_solid_color_is_returned_exactly() {
    let rgb8 = [200u8, 120, 40];
    let lab = solid_lab(96, 128, &color::rgb8_to_lab(&rgb8));
    let mask = vec![true; 96 * 128];
    let out = box_run(&lab, &mask, 96, 128, 16, 12);
    assert!((out.h, out.w) == (12, 16) && out.small_mask.iter().all(|&m| m));
    assert_eq!(out.small_lab, lab[..3].repeat(12 * 16));
    assert!(color::lab_to_rgb8(&out.small_lab)
        .chunks_exact(3)
        .all(|p| p == rgb8));
}

#[test]
fn test_non_integer_ratio_and_mask() {
    let (h, w) = (50, 70);
    let mut lab = vec![0.0; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            lab[3 * (y * w + x)] = x as f64;
        }
    }
    let mask: Vec<bool> = (0..h * w).map(|i| i % w >= 35).collect();
    let out = box_run(&lab, &mask, h, w, 10, 8);
    for y in 0..8 {
        for x in 0..10 {
            assert_eq!(out.small_mask[y * 10 + x], x >= 5, "small_mask[{y}, {x}]");
        }
    }
    // np.arange(35, 70).reshape(5, 7).mean(axis=1)
    for (j, x) in (5..10).enumerate() {
        let expected = (0..7).map(|k| (35 + 7 * j + k) as f64).sum::<f64>() / 7.0;
        assert!(isclose(l_at(&out, 0, x), expected), "column {x}");
    }
}

#[test]
fn box_test_empty_mask_raises_pixelforge_error() {
    let lab = vec![0.0; 16 * 16 * 3];
    let err = boxfilter::run(&lab, &[false; 256], 16, 16, 8, 8, None).unwrap_err();
    assert!(err.0.contains("no opaque pixels"), "{}", err.0);
}

/// Red square with a 4-px white fringe at alpha 128 (right) and 127 (left), RGBA8.
fn fringed_square() -> Vec<u8> {
    let mut rgba = vec![0u8; 64 * 64 * 4];
    let mut fill = |y0: usize, y1: usize, x0: usize, x1: usize, px: [u8; 4]| {
        for y in y0..y1 {
            for x in x0..x1 {
                rgba[4 * (y * 64 + x)..4 * (y * 64 + x) + 4].copy_from_slice(&px);
            }
        }
    };
    fill(8, 56, 8, 56, [220, 0, 20, 255]);
    fill(8, 56, 56, 60, [255, 255, 255, 128]);
    fill(8, 56, 4, 8, [255, 255, 255, 127]);
    rgba
}

#[test]
fn test_semi_transparent_fringe_pulls_in_proportion_to_alpha() {
    let rgba = fringed_square();
    let rgb: Vec<f64> = rgba
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]].map(|v| f64::from(v) / 255.0))
        .collect();
    let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
    let config = config(&[
        ("method", s("box")),
        ("out_width", int(16)),
        ("out_height", int(16)),
        ("palette_size", int(4)),
        ("alpha_threshold", int(128)),
        ("outline", s("none")),
        ("denoise", s("none")),
    ]);
    let pre = preprocess::run(&rgb, &alpha, 64, 64, &config).unwrap();
    let mut unique = pre.weight.clone();
    unique.sort_by(f64::total_cmp);
    unique.dedup();
    assert_eq!(unique.len(), 3, "{unique:?}");
    for (u, e) in unique.iter().zip([0.0, 128.0 / 255.0, 1.0]) {
        assert!(isclose(*u, e), "{unique:?}");
    }
    assert!(pre
        .weight
        .iter()
        .zip(&pre.mask)
        .all(|(&w, &m)| (w > 0.0) == m));
    let plain = box_pre(&pre, 16, 16, None);
    let out = box_pre(&pre, 16, 16, Some(&pre.weight));
    assert_eq!(out.small_mask, plain.small_mask); // membership stays binary
                                                  // The output column that mixes red with the alpha-128 fringe is the alpha-weighted mean.
    let ref_lab = color::rgb8_to_lab(&[220, 0, 20, 255, 255, 255]);
    let (red_l, white_l) = (ref_lab[0], ref_lab[3]);
    let mixed: Vec<usize> = (0..16)
        .filter(|&x| (l_at(&plain, 8, x) - (red_l + white_l) / 2.0).abs() < 25.0)
        .collect();
    assert_eq!(mixed.len(), 1, "{mixed:?}");
    let col = mixed[0];
    let (hi, wi) = (pre.h, pre.w);
    let xs: Vec<usize> = (0..wi)
        .filter(|&x| boxfilter::cell_index(wi, 16)[x] == col)
        .collect();
    let ys: Vec<usize> = (0..hi)
        .filter(|&y| boxfilter::cell_index(hi, 16)[y] == 8)
        .collect();
    let (mut num, mut den) = (0.0, 0.0);
    for &y in &ys {
        for &x in &xs {
            let w = pre.weight[y * wi + x];
            num += w * pre.lab[3 * (y * wi + x)];
            den += w;
        }
    }
    let expected = num / den;
    assert!(
        isclose(l_at(&out, 8, col), expected),
        "{} vs {expected}",
        l_at(&out, 8, col)
    );
    assert!(l_at(&out, 8, col) < l_at(&plain, 8, col) - 5.0);
    // The alpha-127 fringe is transparent (and cropped).
    assert!(!(0..hi).any(|y| pre.mask[y * wi]));
}

#[test]
fn box_test_all_ones_weight_matches_unweighted() {
    for name in ["line_diag", "circle_alpha", "noisy_gradient"] {
        let (pre, _config) = preprocessed(
            name,
            &[
                ("method", s("box")),
                ("out_width", int(16)),
                ("out_height", int(16)),
            ],
        );
        let plain = box_pre(&pre, 16, 16, None);
        let ones = box_pre(&pre, 16, 16, Some(&vec![1.0; pre.h * pre.w]));
        assert_eq!(ones.small_lab, plain.small_lab, "{name}");
        assert_eq!(ones.small_mask, plain.small_mask, "{name}");
    }
}

// ---------------------------------------------------------------------------------------
// tests/test_kopf.py

#[test]
fn test_thin_line_stays_connected() {
    let (pre, config) = preprocessed(
        "line_diag",
        &[
            ("method", s("kopf")),
            ("out_width", int(32)),
            ("out_height", int(32)),
            ("key_bg", b(false)),
        ],
    );
    let out = kopf_pre(&pre, 32, 32, &config);
    let dark: Vec<bool> = out
        .small_lab
        .chunks_exact(3)
        .map(|p| p[0] < DARK_L)
        .collect();
    // Every step of the diagonal is dark, and the dark pixels form one 8-connected line.
    assert!((0..32).all(|i| dark[i * 32 + i]));
    assert_eq!(components8(&dark, 32, 32), 1);
    assert!(dark.iter().filter(|&&d| d).count() <= 3 * 32); // a line, not a smear

    // The box baseline averages the line into the white background: nothing dark survives.
    let baseline = box_pre(&pre, 32, 32, None);
    let min_l = |o: &Downscaled| {
        o.small_lab
            .chunks_exact(3)
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min)
    };
    assert!(!baseline.small_lab.chunks_exact(3).any(|p| p[0] < DARK_L));
    assert!(min_l(&out) < min_l(&baseline) - 30.0);
}

#[test]
fn test_converges() {
    let (pre, config) = preprocessed(
        "two_color",
        &[
            ("method", s("kopf")),
            ("out_width", int(8)),
            ("out_height", int(8)),
        ],
    );
    let out = kopf_pre(&pre, 8, 8, &config);
    assert!(stat_bool(&out, "converged"));
    assert!(stat_int(&out, "iterations") < config.kopf_max_iters);
    // The vertical edge stays sharp: 4 red columns, 4 blue columns, no blended column.
    let first = &pre.lab[..3];
    let last = &pre.lab[3 * (pre.w - 1)..3 * pre.w];
    for y in 0..8 {
        for x in 0..8 {
            let target = if x < 4 { first } else { last };
            assert!(de(px(&out, y, x), target) < 3.0, "({y}, {x})");
        }
    }
}

#[test]
fn test_mask_respected() {
    let (pre, config) = preprocessed(
        "circle_alpha",
        &[
            ("method", s("kopf")),
            ("out_width", int(32)),
            ("out_height", int(32)),
            ("crop_to_alpha", b(false)),
        ],
    );
    let out = kopf_pre(&pre, 32, 32, &config);
    // Support of the disc at 32×32: output cells that contain at least one opaque input pixel.
    let sup = support(&pre.mask, pre.h, pre.w, 32, 32);
    assert!(out.small_mask.iter().any(|&m| m));
    assert!(!out.small_mask.iter().zip(&sup).any(|(&m, &s)| m && !s));
    assert!(out.small_lab.iter().all(|v| v.is_finite()));
}

#[test]
fn test_max_iters_is_respected() {
    let (pre, config) = preprocessed(
        "noisy_gradient",
        &[
            ("method", s("kopf")),
            ("out_width", int(16)),
            ("out_height", int(16)),
            ("kopf_max_iters", int(3)),
        ],
    );
    let out = kopf_pre(&pre, 16, 16, &config);
    assert!(stat_int(&out, "iterations") == 3 && !stat_bool(&out, "converged"));
    assert!((out.h, out.w) == (16, 16) && out.small_mask.iter().all(|&m| m));
}

#[test]
fn test_solid_color_converges_quickly() {
    // A non-integer ratio, where border truncation used to keep max |Δμ| above kopf_tol.
    let lab = solid_lab(50, 50, &color::rgb_to_lab(&[0.3, 0.6, 0.2]));
    let config = config(&[
        ("method", s("kopf")),
        ("out_width", int(32)),
        ("out_height", int(32)),
    ]);
    let out = kopf_run::run(&lab, &[true; 2500], 50, 50, 32, 32, &config, None, THREADS);
    assert!(stat_bool(&out, "converged") && stat_int(&out, "iterations") < 10);
    let max_de = out
        .small_lab
        .chunks_exact(3)
        .map(|p| de(p, &lab))
        .fold(0.0, f64::max);
    assert!(max_de < 1e-6, "{max_de}");
}

#[test]
fn test_thin_line_converges() {
    let (pre, config) = preprocessed(
        "line_diag",
        &[
            ("method", s("kopf")),
            ("out_width", int(32)),
            ("out_height", int(32)),
            ("key_bg", b(false)),
        ],
    );
    let out = kopf_pre(&pre, 32, 32, &config);
    assert!(stat_bool(&out, "converged") && stat_int(&out, "iterations") < config.kopf_max_iters);
}

#[test]
fn kopf_test_all_ones_weight_matches_unweighted() {
    for name in ["line_diag", "circle_alpha"] {
        let (pre, config) = preprocessed(
            name,
            &[
                ("method", s("kopf")),
                ("out_width", int(16)),
                ("out_height", int(16)),
            ],
        );
        let plain = kopf_pre(&pre, 16, 16, &config);
        let ones_w = vec![1.0; pre.h * pre.w];
        let ones = kopf_run::run(
            &pre.lab,
            &pre.mask,
            pre.h,
            pre.w,
            16,
            16,
            &config,
            Some(&ones_w),
            THREADS,
        );
        assert_eq!(ones.small_lab, plain.small_lab, "{name}");
        assert_eq!(ones.small_mask, plain.small_mask, "{name}");
    }
}

#[test]
fn test_low_weight_pixels_pull_kernels_less() {
    // Columns alternate L 40 and L 60, both equally far from the initial kernel color (L 50),
    // so after one iteration every kernel's color is the spatial mean of its window: 50
    // unweighted, and closer to 40 when the light columns carry a quarter of the weight.
    let n = 32;
    let mut lab = vec![0.0; n * n * 3];
    let mut weight = vec![1.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let odd = x % 2 == 1;
            lab[3 * (y * n + x)] = if odd { 60.0 } else { 40.0 };
            if odd {
                weight[y * n + x] = 0.25;
            }
        }
    }
    let mask = vec![true; n * n];
    let config = config(&[
        ("method", s("kopf")),
        ("denoise", s("none")),
        ("kopf_max_iters", int(1)),
    ]);
    let plain = kopf_run::run(&lab, &mask, n, n, 8, 8, &config, None, THREADS);
    let out = kopf_run::run(&lab, &mask, n, n, 8, 8, &config, Some(&weight), THREADS);
    assert_eq!(out.small_mask, plain.small_mask);
    let mean_l = |o: &Downscaled| o.small_lab.chunks_exact(3).map(|p| p[0]).sum::<f64>() / 64.0;
    assert!((mean_l(&plain) - 50.0).abs() < 0.5, "{}", mean_l(&plain));
    assert!((mean_l(&out) - 44.0).abs() < 1.0, "{}", mean_l(&out));
}

/// Orange (L≈68) right of a transparent margin of `edge` px, with a 2-px dark (L≈6)
/// vertical line `offset` px inside the silhouette: (lab, mask), 64 × 64.
fn line_near_silhouette(edge: usize, offset: usize) -> (Vec<f64>, Vec<bool>) {
    let n = 64;
    let mut rgb = Vec::with_capacity(n * n * 3);
    let mut mask = Vec::with_capacity(n * n);
    for _y in 0..n {
        for x in 0..n {
            if x >= edge + offset && x < edge + offset + 2 {
                rgb.extend([20.0 / 255.0; 3]);
            } else {
                rgb.extend([240.0 / 255.0, 140.0 / 255.0, 30.0 / 255.0]);
            }
            mask.push(x >= edge);
        }
    }
    (color::rgb_to_lab(&rgb), mask)
}

#[test]
fn test_line_near_silhouette_not_worse_than_box() {
    // Kernels whose window the silhouette cuts used to fire the shape constraint on every
    // iteration and saturate σ, so a line 4 px inside the edge came out fainter than with box.
    let (lab, mask) = line_near_silhouette(4, 4);
    let config = config(&[
        ("method", s("kopf")),
        ("out_width", int(16)),
        ("out_height", int(16)),
    ]);
    let out = kopf_run::run(&lab, &mask, 64, 64, 16, 16, &config, None, THREADS);
    let baseline = box_run(&lab, &mask, 64, 64, 16, 16);
    let kopf_rows = row_min_l(&out);
    let kopf_min = kopf_rows.iter().copied().fold(f64::INFINITY, f64::min);
    let box_min = row_min_l(&baseline)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    assert!(kopf_min <= box_min + 2.0, "{kopf_min} vs {box_min}");
    assert!(kopf_rows.iter().all(|&l| l < DARK_L), "{kopf_rows:?}"); // visible on every row
}

/// test_line_two_px_inside_silhouette_shows_on_every_row[out_size, edge].
fn line_two_px_inside_silhouette(out_size: usize, edge: usize) {
    // A kernel whose cell is mostly transparent used to claim the line's pixels and then be
    // dropped from the mask, so at 64→8 the line vanished (pure fill color on every row).
    let (lab, mask) = line_near_silhouette(edge, 2);
    let config = config(&[
        ("method", s("kopf")),
        ("out_width", int(out_size as i64)),
        ("out_height", int(out_size as i64)),
    ]);
    let out = kopf_run::run(
        &lab, &mask, 64, 64, out_size, out_size, &config, None, THREADS,
    );
    let baseline = box_run(&lab, &mask, 64, 64, out_size, out_size);
    let box_min = row_min_l(&baseline)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let rows = row_min_l(&out);
    assert!(
        rows.iter().all(|&l| l < box_min + 2.0),
        "out_size {out_size}, edge {edge}: {rows:?} vs box {box_min}"
    );
}

#[test]
fn test_line_two_px_inside_silhouette_shows_on_every_row_16_4() {
    line_two_px_inside_silhouette(16, 4);
}

#[test]
fn test_line_two_px_inside_silhouette_shows_on_every_row_8_4() {
    line_two_px_inside_silhouette(8, 4);
}

#[test]
fn test_line_two_px_inside_silhouette_shows_on_every_row_8_6() {
    line_two_px_inside_silhouette(8, 6);
}

#[test]
fn test_line_two_px_inside_silhouette_shows_on_every_row_8_9() {
    line_two_px_inside_silhouette(8, 9);
}

#[test]
fn test_silhouette_mask_matches_box() {
    let (pre, config) = preprocessed(
        "circle_alpha",
        &[
            ("method", s("kopf")),
            ("out_width", int(32)),
            ("out_height", int(32)),
            ("crop_to_alpha", b(false)),
        ],
    );
    let out = kopf_pre(&pre, 32, 32, &config);
    assert_eq!(out.small_mask, box_pre(&pre, 32, 32, None).small_mask);
}

#[test]
fn test_silhouette_does_not_saturate_sigma() {
    let (pre, config) = preprocessed(
        "circle_alpha",
        &[
            ("method", s("kopf")),
            ("out_width", int(16)),
            ("out_height", int(16)),
            ("crop_to_alpha", b(false)),
            ("prereduce_max_ratio", int(4)),
        ],
    );
    assert_eq!((pre.h, pre.w), (64, 64));
    let out = kopf_pre(&pre, 16, 16, &config);
    assert!((stat_int(&out, "sigma_capped") as f64) < 0.1 * 16.0 * 16.0);
}

// ---------------------------------------------------------------------------------------
// tests/test_gerstner.py

#[test]
fn test_two_color_image_gives_two_colors() {
    let (pre, config) = preprocessed(
        "two_color",
        &[
            ("method", s("gerstner")),
            ("palette_size", int(2)),
            ("out_width", int(8)),
            ("out_height", int(8)),
            ("saturation_beta", float(1.0)),
        ],
    );
    let out = gerstner_pre(&pre, 8, 8, &config, None);
    assert_eq!(palette_len(&out), 2);
    let rb = color::rgb8_to_lab(&[255, 0, 0, 0, 0, 255]);
    let (red, blue) = (&rb[..3], &rb[3..]);
    let idx = indices(&out);
    let (left, right) = (idx[0], idx[7]);
    assert_ne!(left, right);
    let pal = palette(&out);
    assert!(de(&pal[3 * left as usize..], red) < 3.0);
    assert!(de(&pal[3 * right as usize..], blue) < 3.0);
    for y in 0..8 {
        for x in 0..8 {
            let want = if x < 4 { left } else { right };
            assert_eq!(idx[y * 8 + x], want, "({y}, {x})");
        }
    }
    assert!(out.small_mask.iter().all(|&m| m));
}

#[test]
#[allow(non_snake_case)]
fn test_palette_grows_to_K() {
    let (pre, config) = preprocessed(
        "gradient6",
        &[
            ("method", s("gerstner")),
            ("palette_size", int(6)),
            ("out_width", int(16)),
            ("out_height", int(16)),
        ],
    );
    let out = gerstner_pre(&pre, 16, 16, &config, None);
    assert!(stat_int(&out, "palette_size") == 6 && palette_len(&out) == 6);
    assert!(!stat_bool(&out, "hit_iteration_cap"));
    let mut unique = indices(&out).to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), 6);
}

#[test]
fn test_iteration_cap() {
    let (pre, config) = preprocessed(
        "gradient6",
        &[
            ("method", s("gerstner")),
            ("palette_size", int(6)),
            ("out_width", int(16)),
            ("out_height", int(16)),
            ("g_max_iters", int(3)),
        ],
    );
    let out = gerstner_pre(&pre, 16, 16, &config, None);
    assert!(stat_int(&out, "iterations") == 3 && stat_bool(&out, "hit_iteration_cap"));
    let k = palette_len(&out);
    assert!((1..=6).contains(&k));
    let idx = indices(&out);
    assert!((out.h, out.w) == (16, 16) && idx.len() == 256);
    assert!(idx.iter().all(|&i| i >= 0 && (i as usize) < k));
    assert!(out.small_lab.iter().all(|v| v.is_finite()) && out.small_mask.iter().all(|&m| m));
}

#[test]
fn test_transparent_superpixels_are_masked_out() {
    let (pre, config) = preprocessed(
        "circle_alpha",
        &[
            ("method", s("gerstner")),
            ("palette_size", int(4)),
            ("out_width", int(32)),
            ("out_height", int(32)),
            ("crop_to_alpha", b(false)),
        ],
    );
    let out = gerstner_pre(&pre, 32, 32, &config, None);
    let sup = support(&pre.mask, pre.h, pre.w, 32, 32);
    assert!(!out.small_mask.iter().zip(&sup).any(|(&m, &s)| m && !s));
    assert!((12..20).all(|y| (12..20).all(|x| out.small_mask[y * 32 + x])));
}

/// `_run_lab`: gerstner on an opaque LAB image at a quarter of its size.
fn run_lab(
    lab: &[f64],
    h: usize,
    w: usize,
    palette_size: i64,
    overrides: &[(&str, Value)],
    tunables: Tunables,
) -> (Downscaled, Config) {
    let mut pairs = vec![
        ("method", s("gerstner")),
        ("palette_size", int(palette_size)),
        ("out_width", int((w / 4) as i64)),
        ("out_height", int((h / 4) as i64)),
    ];
    pairs.extend(overrides.iter().cloned());
    let config = config(&pairs);
    let mask = vec![true; h * w];
    let out =
        gerstner_run::run_with(lab, &mask, h, w, w / 4, h / 4, &config, None, tunables).unwrap();
    (out, config)
}

/// 32 × 32, L 50 on the left half and L 54 on the right.
fn close_halves() -> Vec<f64> {
    let mut lab = vec![0.0; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            lab[3 * (y * 32 + x)] = if x >= 16 { 54.0 } else { 50.0 };
        }
    }
    lab
}

#[test]
fn test_close_colors_split() {
    // ΔE 4 is clearly visible; after the bilateral filter the halves sit ≈ 2.8 apart, which
    // the 3·2δ growth rule alone never reached (1 color, annealing ran down to T ≈ 1e-3).
    let (out, config) = run_lab(&close_halves(), 32, 32, 2, &[], Tunables::default());
    assert_eq!(palette_len(&out), 2);
    let idx = indices(&out);
    for y in 0..8 {
        for x in 0..8 {
            let want = if x < 4 { idx[0] } else { idx[7] };
            assert_eq!(idx[y * 8 + x], want, "({y}, {x})");
        }
    }
    assert_ne!(idx[0], idx[7]);
    assert!(stat_float(&out, "final_temperature") >= 1e-2 * config.g_T_final);
}

#[test]
fn test_single_color_stops_early() {
    let lab = solid_lab(32, 32, &[60.0, 10.0, -20.0]);
    let (out, config) = run_lab(&lab, 32, 32, 8, &[], Tunables::default());
    assert_eq!(palette_len(&out), 1);
    assert!(!stat_bool(&out, "hit_iteration_cap"));
    assert!(stat_int(&out, "iterations") <= 5);
    // Not the backstop.
    assert!(stat_float(&out, "final_temperature") >= 1e-2 * config.g_T_final);
}

#[test]
#[allow(non_snake_case)]
fn test_three_colors_with_large_K() {
    let colors = [[30.0, 20.0, 10.0], [60.0, -30.0, 40.0], [85.0, 5.0, -40.0]];
    let mut lab = vec![0.0; 32 * 32 * 3];
    for y in 0..32 {
        for x in 0..32 {
            let c = if x < 11 {
                0
            } else if x < 22 {
                1
            } else {
                2
            };
            lab[3 * (y * 32 + x)..3 * (y * 32 + x) + 3].copy_from_slice(&colors[c]);
        }
    }
    let (out, _) = run_lab(
        &lab,
        32,
        32,
        16,
        &[("saturation_beta", float(1.0))],
        Tunables::default(),
    );
    assert_eq!(palette_len(&out), 3);
    for x in [0, 16, 31] {
        let c = &lab[3 * x..3 * x + 3];
        let best = palette(&out)
            .chunks_exact(3)
            .map(|p| de(p, c))
            .fold(f64::INFINITY, f64::min);
        assert!(best < 3.0, "column {x}: {best}");
    }
}

#[test]
fn test_checker_tiles_iteration_count() {
    // T0 ∝ spread (not variance): 14 iterations here, down from 25. Bound = 14 + 20%.
    let (pre, config) = preprocessed(
        "checker_tiles",
        &[
            ("method", s("gerstner")),
            ("palette_size", int(8)),
            ("out_width", int(16)),
            ("out_height", int(16)),
        ],
    );
    let out = gerstner_pre(&pre, 16, 16, &config, None);
    assert!(!stat_bool(&out, "hit_iteration_cap"));
    assert!(stat_int(&out, "iterations") <= 16);
}

#[test]
fn test_stalled_pair_ends_annealing() {
    // With the spread cap disabled the L 50 / L 54 pair keeps creeping apart without ever
    // becoming ready; annealing must stop once it stalls, not at the 1e-3·T_final backstop.
    // (Python monkeypatches gerstner.SPLIT_SPREAD; here it is the Tunables field.)
    let tunables = Tunables {
        split_spread: 1e9,
        ..Tunables::default()
    };
    let (out, config) = run_lab(&close_halves(), 32, 32, 2, &[], tunables);
    assert!(!stat_bool(&out, "hit_iteration_cap"));
    assert!(stat_float(&out, "final_temperature") >= 1e-2 * config.g_T_final);
}

#[test]
fn gerstner_test_empty_mask_raises_pixelforge_error() {
    let lab = vec![0.0; 16 * 16 * 3];
    let config = config(&[("method", s("gerstner"))]);
    let err = gerstner_run::run(&lab, &[false; 256], 16, 16, 8, 8, &config, None).unwrap_err();
    assert!(err.0.contains("no opaque pixels"), "{}", err.0);
}

#[test]
fn gerstner_test_all_ones_weight_matches_unweighted() {
    for name in ["two_color", "circle_alpha"] {
        let (pre, config) = preprocessed(
            name,
            &[
                ("method", s("gerstner")),
                ("out_width", int(16)),
                ("out_height", int(16)),
            ],
        );
        let plain = gerstner_pre(&pre, 16, 16, &config, None);
        let ones = gerstner_pre(&pre, 16, 16, &config, Some(&vec![1.0; pre.h * pre.w]));
        assert_eq!(ones.small_lab, plain.small_lab, "{name}");
        assert_eq!(ones.small_mask, plain.small_mask, "{name}");
        assert_eq!(ones.indices, plain.indices, "{name}");
    }
}

#[test]
fn test_palette_prior_follows_coverage_and_sums_to_one() {
    let uniform = gerstner_run::palette_prior(&[1.0; 7]);
    assert_eq!(uniform, vec![1.0 / 7.0; 7]);
    let prior = gerstner_run::palette_prior(&[1.0, 1.0, 0.5, 128.0 / 255.0]);
    assert!(isclose(prior.iter().sum(), 1.0));
    assert!(prior[0] == prior[1] && isclose(prior[2] / prior[0], 0.5));
}

#[test]
fn test_semi_transparent_superpixels_get_less_palette_mass() {
    // Opaque block with a band at weight 0.2 on the right. The Python test records the
    // annealer's prior on the Python reference (tests/gerstner_reference.py) and checks it
    // sums to one with the band's superpixels holding a fifth of the mass; that prior is
    // internal to the Rust stage, so only the run itself and palette_prior are checked here.
    let mut lab = vec![0.0; 32 * 32 * 3];
    let mut weight = vec![1.0; 32 * 32];
    for y in 0..32 {
        for x in 0..32 {
            lab[3 * (y * 32 + x)] = if x >= 24 { 90.0 } else { 50.0 };
            if x >= 24 {
                weight[y * 32 + x] = 0.2;
            }
        }
    }
    let mask = vec![true; 32 * 32];
    let config = config(&[
        ("method", s("gerstner")),
        ("palette_size", int(2)),
        ("denoise", s("none")),
        ("g_max_iters", int(3)),
    ]);
    let out = gerstner_run::run(&lab, &mask, 32, 32, 8, 8, &config, Some(&weight)).unwrap();
    assert!(out.small_mask.iter().all(|&m| m));
    // The superpixel prior as the run computes it from the coverage (mean weight): the
    // band's superpixels (columns 6 and 7 of the 8 × 8 grid) get a fifth of the mass.
    let coverage: Vec<f64> = (0..64)
        .map(|k| if k % 8 >= 6 { 0.2 } else { 1.0 })
        .collect();
    let first = gerstner_run::palette_prior(&coverage);
    assert!(isclose(first.iter().sum(), 1.0));
    for k in 0..64 {
        if k % 8 >= 6 {
            assert!(isclose(first[k] / first[0], 0.2));
        } else {
            assert_eq!(first[k], first[0]);
        }
    }
    assert_eq!(
        gerstner_run::palette_prior(&[1.0; 64]),
        vec![1.0 / 64.0; 64]
    );
}

#[test]
fn test_shared_grid_helpers() {
    // neighbor_mean4 against the explicit loop. The Python test's _laplacian_smooth and
    // _Grid checks are on private helpers of the Rust stage (gerstner_run.rs) and not
    // reachable from an integration test.
    let (h, w) = (3, 4);
    let grid: Vec<f64> = (0..12).map(|i| (i as f64).powf(1.5)).collect();
    let mut expected = vec![0.0; 12];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let near: Vec<f64> = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                .iter()
                .map(|(dy, dx)| (y + dy, x + dx))
                .filter(|&(ny, nx)| ny >= 0 && ny < h as i64 && nx >= 0 && nx < w as i64)
                .map(|(ny, nx)| grid[ny as usize * w + nx as usize])
                .collect();
            expected[y as usize * w + x as usize] = near.iter().sum::<f64>() / near.len() as f64;
        }
    }
    let got = neighbor_mean4(&grid, h, w);
    for (g, e) in got.iter().zip(&expected) {
        assert!(isclose(*g, *e), "{got:?} vs {expected:?}");
    }
    // Box's cell index is what the superpixel grid's home cells use.
    assert_eq!(
        boxfilter::cell_index(20, 5),
        vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    );
}

/// The bilateral filter before the symmetric-weight sharing: one exp() per offset
/// (`_bilateral_reference`, with the stage's `libm::exp`).
fn bilateral_reference(
    image: &[f64],
    active: &[f64],
    h: usize,
    w: usize,
    sigma_color: f64,
    sigma_spatial: f64,
) -> Vec<f64> {
    let radius = (2.0 * sigma_spatial).ceil() as i64;
    let mut num = vec![0.0; h * w * 3];
    let mut den = vec![0.0; h * w];
    let inv_color = 1.0 / (2.0 * sigma_color * sigma_color);
    let inv_spatial = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    let p = (y * w as i64 + x) as usize;
                    let (ny, nx) = (y + dy, x + dx);
                    let inside = ny >= 0 && ny < h as i64 && nx >= 0 && nx < w as i64;
                    let q = if inside {
                        (ny * w as i64 + nx) as usize
                    } else {
                        0
                    };
                    let neighbor = if inside {
                        [image[3 * q], image[3 * q + 1], image[3 * q + 2]]
                    } else {
                        [0.0; 3]
                    };
                    let a = if inside { active[q] } else { 0.0 };
                    let d0 = neighbor[0] - image[3 * p];
                    let d1 = neighbor[1] - image[3 * p + 1];
                    let d2 = neighbor[2] - image[3 * p + 2];
                    let dist2 = d0 * d0 + d1 * d1 + d2 * d2;
                    let mut weight = pixelforge::annealer::exp(
                        -dist2 * inv_color - ((dx * dx + dy * dy) as f64) * inv_spatial,
                    );
                    weight *= a;
                    for c in 0..3 {
                        num[3 * p + c] += weight * neighbor[c];
                    }
                    den[p] += weight;
                }
            }
        }
    }
    let mut out = image.to_vec();
    for p in 0..h * w {
        if den[p] > 0.0 {
            for c in 0..3 {
                out[3 * p + c] = num[3 * p + c] / den[p];
            }
        }
    }
    out
}

#[test]
fn test_reference_bilateral_filter_weight_sharing_is_bitwise() {
    // Python checks its numpy reference's weight sharing; here the same check runs on the
    // Rust stage (gerstner::bilateral_filter), against the one-exp()-per-offset reference.
    // The random image comes from SplitMix64, not numpy's default_rng(7).
    for sigma_spatial in [0.4, 1.0, 1.5, 2.3] {
        let mut rng = SplitMix64::new(7);
        let (h, w) = (41, 37);
        let scale = [100.0, 160.0, 160.0];
        let shift = [0.0, 80.0, 80.0];
        let image: Vec<f64> = (0..h * w * 3)
            .map(|i| rng.uniform(0.0, 1.0) * scale[i % 3] - shift[i % 3])
            .collect();
        let active: Vec<f64> = (0..h * w)
            .map(|_| f64::from(u8::from(rng.uniform(0.0, 1.0) > 0.2)))
            .collect();
        let expected = bilateral_reference(&image, &active, h, w, 12.0, sigma_spatial);
        let got = gerstner::bilateral_filter(&image, &active, h, w, 12.0, sigma_spatial);
        let same = got
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "sigma_spatial {sigma_spatial}");
    }
}

// ---------------------------------------------------------------------------------------
// tests/test_branches.py

#[test]
fn test_seamless_with_gerstner() {
    let image = small("seamless_src", 2); // 64 × 64, period 64
    let cfg = config(&[
        ("preset", s("background")),
        ("method", s("gerstner")),
        ("tileset", b(false)),
        ("seamless", b(true)),
        ("denoise", s("none")),
        ("out_width", int(16)),
        ("out_height", int(16)),
    ]);
    let result = run_image(&image, &cfg).unwrap();
    let idx = &result.indices;
    assert!((idx.h, idx.w) == (16, 16) && idx.indices.iter().all(|&i| i >= 0));
    let lab_at = |y: usize, x: usize| {
        let k = idx.indices[y * 16 + x] as usize;
        &result.palette_lab[3 * k..3 * k + 3]
    };
    let mean = |pairs: Vec<(&[f64], &[f64])>| {
        let n = pairs.len() as f64;
        pairs.into_iter().map(|(a, b)| de(a, b)).sum::<f64>() / n
    };
    // lab[:, 1:-2] vs lab[:, 2:-1]
    let interior = mean(
        (0..16)
            .flat_map(|y| (1..14).map(move |x| (y, x)))
            .map(|(y, x)| (lab_at(y, x), lab_at(y, x + 1)))
            .collect(),
    );
    assert!(interior > 0.0);
    let columns = mean((0..16).map(|y| (lab_at(y, 0), lab_at(y, 15))).collect());
    let rows = mean((0..16).map(|x| (lab_at(0, x), lab_at(15, x))).collect());
    assert!(columns < 3.0 * interior, "{columns} vs {interior}");
    assert!(rows < 3.0 * interior, "{rows} vs {interior}");
    let plain = run_image(&image, &replace(&cfg, &[("seamless", b(false))])).unwrap();
    assert_eq!((plain.indices.h, plain.indices.w), (16, 16));
}
