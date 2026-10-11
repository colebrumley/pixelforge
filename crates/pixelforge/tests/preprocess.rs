//! Port of tests/test_preprocess.py (and test_branches.py's `test_median_denoise` and
//! `test_remove_bg_missing_dependency_message`): keying with fringe, layouts and `fit`,
//! pre-reduction, weights, enhance and seamless padding.

mod common;
use common::*;

use pixelforge::color;
use pixelforge::config::{Config, Value};
use pixelforge::enhance;
use pixelforge::filters::{self, Mode};
use pixelforge::io;
use pixelforge::layout::{self, Layout};
use pixelforge::mask;
use pixelforge::pipeline::{self, Output};
use pixelforge::preprocess::{self, Preprocessed};
use pixelforge::pyjson::Json;

// ---------------------------------------------------------------------------------------
// Helpers

fn pre(rgb: &[f64], alpha: &[u8], h: usize, w: usize, config: &Config) -> Preprocessed {
    preprocess::run(rgb, alpha, h, w, config).unwrap()
}

fn set(rgb: &mut [f64], w: usize, y: usize, x: usize, c: [f64; 3]) {
    rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
}

fn fill_rect(
    rgb: &mut [f64],
    w: usize,
    ys: std::ops::Range<usize>,
    xs: std::ops::Range<usize>,
    c: [f64; 3],
) {
    for y in ys {
        for x in xs.clone() {
            set(rgb, w, y, x, c);
        }
    }
}

/// `_sprite_on_white(size)`.
fn sprite_on_white(size: usize) -> (Vec<f64>, Vec<u8>) {
    let mut rgb = vec![1.0; size * size * 3];
    fill_rect(&mut rgb, size, 20..76, 30..66, [0.8, 0.2, 0.1]);
    fill_rect(&mut rgb, size, 40..50, 40..56, [1.0, 1.0, 1.0]);
    (rgb, vec![255; size * size])
}

const RED: [f64; 3] = [0.85, 0.0, 0.07];

/// `_disc_on_white(ramp, size, radius)`: the RGB image and the subject coverage.
fn disc_on_white(ramp: f64, size: usize, radius: f64) -> (Vec<f64>, Vec<f64>) {
    let c = (size - 1) as f64 / 2.0;
    let mut rgb = Vec::with_capacity(size * size * 3);
    let mut coverage = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let d = (y as f64 - c).hypot(x as f64 - c);
            let cov = if ramp > 0.0 {
                ((radius + ramp / 2.0 - d) / ramp).clamp(0.0, 1.0)
            } else if d <= radius {
                1.0
            } else {
                0.0
            };
            for r in RED {
                rgb.push(cov * r + (1.0 - cov));
            }
            coverage.push(cov);
        }
    }
    (rgb, coverage)
}

fn disc() -> (Vec<f64>, Vec<f64>) {
    disc_on_white(3.0, 128, 40.0)
}

fn rgba_of(rgb: &[f64], alpha: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .zip(alpha)
        .flat_map(|(p, &a)| {
            [
                (p[0] * 255.0).round_ties_even() as u8,
                (p[1] * 255.0).round_ties_even() as u8,
                (p[2] * 255.0).round_ties_even() as u8,
                a,
            ]
        })
        .collect()
}

fn run_loaded(loaded: &io::Loaded, config: &Config) -> Output {
    pipeline::run_loaded(loaded, config, Vec::new(), false).unwrap()
}

fn stat<'a>(out: &'a Output, key: &str) -> &'a Json {
    match &out.stats {
        Json::Object(items) => &items.iter().find(|(k, _)| k == key).unwrap().1,
        other => panic!("stats is not an object: {other:?}"),
    }
}

fn layout_of(width: i64, height: i64, config: &Config) -> (i64, i64, Option<Layout>) {
    let r = layout::resolve_layout(width, height, config).unwrap();
    (r.out_width, r.out_height, r.layout)
}

fn dims(width: i64, height: i64, config: &Config) -> (i64, i64) {
    layout::resolve_dims(width, height, config).unwrap()
}

/// `np.allclose(a, b, rtol, atol)`.
fn allclose_tol(a: &[f64], b: &[f64], rtol: f64, atol: f64) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .all(|(&x, &y)| (x - y).abs() <= atol + rtol * y.abs())
}

fn allclose(a: &[f64], b: &[f64]) -> bool {
    allclose_tol(a, b, 1e-5, 1e-8)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for v in values {
        sum += v;
        n += 1;
    }
    sum / n as f64
}

/// A deterministic stand-in for `np.random.default_rng(seed).random(n)` (the tests only
/// need a random field, not numpy's exact values).
fn random(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

/// `ndimage.gaussian_filter(rgb, (sigma, sigma, 0), mode=mode)` on an `(h, w, 3)` image.
fn gaussian_rgb(rgb: &[f64], h: usize, w: usize, sigma: f64, mode: Mode) -> Vec<f64> {
    let mut out = vec![0.0; rgb.len()];
    for c in 0..3 {
        let mut plane: Vec<f64> = rgb.iter().skip(c).step_by(3).copied().collect();
        plane = filters::gaussian_filter1d(&plane, h, w, sigma, 0, mode);
        plane = filters::gaussian_filter1d(&plane, h, w, sigma, 1, mode);
        for (i, v) in plane.into_iter().enumerate() {
            out[i * 3 + c] = v;
        }
    }
    out
}

/// `np.roll(a, (dy, dx), axis=(0, 1))` on an `(h, w, c)` array.
fn roll(a: &[f64], h: usize, w: usize, c: usize, dy: usize, dx: usize) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for y in 0..h {
        for x in 0..w {
            let (ty, tx) = ((y + dy) % h, (x + dx) % w);
            out[(ty * w + tx) * c..(ty * w + tx) * c + c]
                .copy_from_slice(&a[(y * w + x) * c..(y * w + x) * c + c]);
        }
    }
    out
}

/// `a[y0:y1, x0:x1]` of an `(h, w, c)` array, flattened.
fn crop(
    a: &[f64],
    w: usize,
    c: usize,
    ys: std::ops::Range<usize>,
    xs: std::ops::Range<usize>,
) -> Vec<f64> {
    let mut out = Vec::new();
    for y in ys {
        out.extend_from_slice(&a[(y * w + xs.start) * c..(y * w + xs.end) * c]);
    }
    out
}

// ---------------------------------------------------------------------------------------
// Keying

#[test]
fn test_flat_background_is_keyed_but_enclosed_regions_are_kept() {
    let (rgb, alpha) = sprite_on_white(96);
    let p = pre(
        &rgb,
        &alpha,
        96,
        96,
        &config(&[
            ("preset", s("sprite")),
            ("denoise", s("none")),
            ("out_height", int(16)),
        ]),
    );
    assert!(p.background_keyed);
    // Cropped to the sprite's bounding box padded by 1 px; only that ring is transparent.
    assert_eq!((p.h, p.w), (58, 38));
    for y in 1..p.h - 1 {
        for x in 1..p.w - 1 {
            assert!(p.mask[y * p.w + x]);
        }
    }
    assert!(!p.mask[..p.w].iter().any(|&m| m));
    assert!(!(0..p.h).any(|y| p.mask[y * p.w]));
}

#[test]
fn test_keying_is_off_for_backgrounds_and_when_disabled() {
    let (rgb, alpha) = sprite_on_white(96);
    for c in [
        config(&[("preset", s("background"))]),
        config(&[("preset", s("sprite")), ("key_bg", b(false))]),
    ] {
        let p = pre(&rgb, &alpha, 96, 96, &c);
        assert!(!p.background_keyed && p.mask.iter().all(|&m| m));
    }
}

#[test]
fn test_images_without_a_flat_border_are_left_alone() {
    for name in ["two_color", "gradient6", "noisy_gradient", "checker_tiles"] {
        let loaded = io::load(&fixture_path(name), io::MAX_INPUT_PIXELS).unwrap();
        let p = pre(
            &loaded.rgb,
            &loaded.alpha,
            loaded.h,
            loaded.w,
            &config(&[("preset", s("sprite"))]),
        );
        assert!(!p.background_keyed && p.mask.iter().all(|&m| m), "{name}");
    }
}

#[test]
fn test_existing_alpha_takes_precedence() {
    let loaded = io::load(&fixture_path("circle_alpha"), io::MAX_INPUT_PIXELS).unwrap();
    let p = pre(
        &loaded.rgb,
        &loaded.alpha,
        loaded.h,
        loaded.w,
        &config(&[("preset", s("sprite"))]),
    );
    assert!(!p.background_keyed);
}

#[test]
fn test_solid_image_is_not_keyed_away() {
    let p = pre(
        &[0.5; 32 * 32 * 3],
        &[255; 32 * 32],
        32,
        32,
        &config(&[("preset", s("sprite"))]),
    );
    assert!(!p.background_keyed && p.mask.iter().all(|&m| m));
}

#[test]
fn test_tolerance_and_validation() {
    let (mut rgb, alpha) = sprite_on_white(96);
    fill_rect(&mut rgb, 96, 0..96, 0..48, [0.9; 3]); // two-tone background, 0.1 apart
    fill_rect(&mut rgb, 96, 20..76, 30..66, [0.8, 0.2, 0.1]);
    let strict = pre(
        &rgb,
        &alpha,
        96,
        96,
        &config(&[("preset", s("sprite")), ("key_bg_tolerance", float(0.02))]),
    );
    assert!(!strict.background_keyed);
    let loose = pre(
        &rgb,
        &alpha,
        96,
        96,
        &config(&[("preset", s("sprite")), ("key_bg_tolerance", float(0.15))]),
    );
    assert!(loose.background_keyed);
    config_err(&[("key_bg_tolerance", float(1.5))]);
}

#[test]
fn test_pipeline_reports_keying_and_palette_ignores_background() {
    let (rgb, alpha) = sprite_on_white(96);
    let loaded = io::from_rgba(&rgba_of(&rgb, &alpha), 96, 96, "test".into());
    let result = run_loaded(
        &loaded,
        &config(&[("preset", s("sprite")), ("method", s("box"))]),
    );
    assert_eq!(stat(&result, "background_keyed"), &Json::Bool(true));
    // The background is gone: the canvas is the sprite plus its outline ring, whose corners
    // stay transparent.
    let ix = &result.indices;
    assert_eq!(ix.indices[0], -1);
    assert_eq!(ix.indices[ix.h * ix.w - 1], -1);
    for y in 2..ix.h - 2 {
        for x in 2..ix.w - 2 {
            assert!(ix.indices[y * ix.w + x] >= 0);
        }
    }
    assert!(ix.h == 64 && ix.w < 50);
}

// ---------------------------------------------------------------------------------------
// Layout

#[test]
fn test_resolve_dims_reserves_the_outline_margin_inside_the_canvas() {
    assert_eq!(
        layout::outline_margin(&config(&[("preset", s("sprite"))])),
        1
    );
    assert_eq!(
        layout::outline_margin(&config(&[("preset", s("sprite")), ("outline", s("none"))])),
        0
    );
    assert_eq!(
        layout::outline_margin(&config(&[("preset", s("sprite")), ("tileset", b(true))])),
        0
    );
    let sprite = config(&[("preset", s("sprite"))]);
    // Longest edge is exactly 64; the other side keeps the aspect of the 62-px inner area.
    assert_eq!(dims(100, 50, &sprite), (64, 33)); // round(62 / 2) + 2
    assert_eq!(dims(50, 100, &sprite), (33, 64));
    assert_eq!(
        dims(
            64,
            64,
            &replace(&sprite, &[("out_width", int(32)), ("out_height", int(32))])
        ),
        (32, 32)
    );
    assert_eq!(
        dims(64, 48, &replace(&sprite, &[("out_height", int(24))])),
        (31, 24)
    ); // 22·4/3
    assert_eq!(
        dims(100, 50, &replace(&sprite, &[("outline", s("none"))])),
        (64, 32)
    );
    assert_eq!(dims(1000, 10, &sprite), (64, 8)); // never below 8
    let p = pre(
        &[0.5; 64 * 64 * 3],
        &[255; 64 * 64],
        64,
        64,
        &replace(
            &sprite,
            &[
                ("out_width", int(32)),
                ("out_height", int(32)),
                ("key_bg", b(false)),
            ],
        ),
    );
    assert!(p.outline_margin() == 1 && (p.out_width(), p.out_height()) == (32, 32));
    assert_eq!((p.target_width(), p.target_height()), (30, 30));
}

// ---------------------------------------------------------------------------------------
// Fringe

#[test]
fn test_anti_aliased_fringe_is_keyed_out() {
    let (rgb, coverage) = disc();
    let halo: Vec<bool> = coverage.iter().map(|&c| c > 0.0 && c < 0.5).collect();
    let before = mask::flat_background(&rgb, 128, 128, 0.08, 0).unwrap();
    let after = mask::flat_background(&rgb, 128, 128, 0.08, 3).unwrap();
    let halo_n = halo.iter().filter(|&&h| h).count() as f64;
    let kept = |bg: &[bool]| halo.iter().zip(bg).filter(|(&h, &k)| h && !k).count() as f64;
    assert!(kept(&before) > 0.5 * halo_n); // the bug: most of it stays
    assert!(kept(&after) <= 0.1 * halo_n);
    // No subject pixel is lost.
    assert!(!after.iter().zip(&coverage).any(|(&k, &c)| k && c >= 0.5));
}

#[test]
fn test_hard_edges_and_enclosed_regions_are_unchanged_by_unmixing() {
    let (rgb, coverage) = disc_on_white(0.0, 128, 40.0);
    let keyed = mask::flat_background(&rgb, 128, 128, 0.08, 3).unwrap();
    let expected: Vec<bool> = coverage.iter().map(|&c| c == 0.0).collect();
    assert_eq!(keyed, expected);
    let (rgb, _alpha) = sprite_on_white(96);
    let keyed = mask::flat_background(&rgb, 96, 96, 0.08, 3).unwrap();
    for y in 20..76 {
        for x in 30..66 {
            // 40:50 × 40:56 lies inside 20:76 × 30:66.
            assert!(!keyed[y * 96 + x]);
        }
    }
}

#[test]
fn test_fringe_zero_reproduces_plain_keying() {
    let (rgb, _) = disc();
    let alpha = vec![255u8; 128 * 128];
    let plain = mask::flat_background(&rgb, 128, 128, 0.08, 0).unwrap();
    let p = pre(
        &rgb,
        &alpha,
        128,
        128,
        &config(&[
            ("preset", s("sprite")),
            ("denoise", s("none")),
            ("out_height", int(128)),
            ("crop_to_alpha", b(false)),
            ("key_bg_fringe", int(0)),
        ]),
    );
    let inverted: Vec<bool> = plain.iter().map(|&k| !k).collect();
    assert_eq!(p.mask, inverted);
    for bad in [-1, 9] {
        config_err(&[("key_bg_fringe", int(bad))]);
    }
}

#[test]
fn test_palette_is_not_spent_on_the_halo() {
    let (rgb, _) = disc();
    let rgba = rgba_of(&rgb, &vec![255u8; 128 * 128]);
    let red = color::rgb_to_lab(&RED);
    let (red_l, red_c) = (red[0], red[1].hypot(red[2]));
    let pinkish = |fringe: i64| {
        let result = run_loaded(
            &io::from_rgba(&rgba, 128, 128, "disc".into()),
            &config(&[
                ("preset", s("sprite")),
                ("method", s("box")),
                ("palette_size", int(4)),
                ("key_bg_fringe", int(fringe)),
            ]),
        );
        let rgb: Vec<f64> = result
            .palette
            .iter()
            .flat_map(|p| p.iter().map(|&v| f64::from(v) / 255.0))
            .collect();
        let lab = color::rgb_to_lab(&rgb);
        // Red mixed with white: lighter than the red by > 5 L and below 85 % of its chroma
        // (the dark auto-outline entry is darker, so it does not count).
        lab.chunks_exact(3)
            .filter(|l| l[0] > red_l + 5.0 && l[1].hypot(l[2]) < 0.85 * red_c)
            .count()
    };
    assert!(pinkish(0) >= 3); // without unmixing three of four entries are pinks
    assert!(pinkish(3) <= 1);
}

// ---------------------------------------------------------------------------------------
// Scale, canvas and fit

#[test]
fn test_resolve_dims_with_scale() {
    let sprite = config(&[("preset", s("sprite"))]);
    assert_eq!(
        dims(64, 64, &replace(&sprite, &[("scale", int(2))])),
        (32, 32)
    ); // margin inside
    assert_eq!(
        dims(100, 50, &replace(&sprite, &[("scale", float(0.5))])),
        (200, 100)
    );
    assert_eq!(
        dims(30, 10, &replace(&sprite, &[("scale", int(4))])),
        (8, 8)
    ); // never below 8
    let tiles = config(&[("preset", s("background")), ("scale", int(3))]);
    assert_eq!(dims(100, 50, &tiles), (48, 32)); // ceil(33, 17) to 16
}

#[test]
fn test_resolve_dims_with_canvas() {
    let sprite = config(&[("preset", s("sprite")), ("canvas", s("48x32"))]);
    // The subject fits inside the 46x30 area left by the outline margin, centered.
    assert_eq!(layout_of(100, 50, &sprite), (48, 32, Some((46, 23, 0, 3))));
    assert_eq!(layout_of(50, 100, &sprite), (48, 32, Some((15, 30, 15, 0))));
    assert_eq!(dims(100, 50, &sprite), (48, 32));
    let plain = replace(&sprite, &[("outline", s("none"))]);
    assert_eq!(layout_of(100, 50, &plain), (48, 32, Some((48, 24, 0, 4))));
    // With a scale the subject keeps it, and is only shrunk when it would not fit.
    let scaled = replace(&sprite, &[("scale", int(1))]);
    assert_eq!(layout_of(20, 10, &scaled), (48, 32, Some((20, 10, 13, 10))));
    assert_eq!(layout_of(200, 10, &scaled), (48, 32, Some((46, 2, 0, 14))));
    let p = pre(
        &[0.5; 64 * 64 * 3],
        &[255; 64 * 64],
        64,
        64,
        &replace(&sprite, &[("key_bg", b(false))]),
    );
    assert_eq!(
        (p.inner_width(), p.inner_height(), p.target_width()),
        (30, 30, 30)
    );
    assert_eq!(p.canvas_pad(), ((1, 1), (9, 9)));
}

#[test]
fn test_tileset_fit_pad_keeps_aspect_and_pads_centered() {
    let pad = config(&[("preset", s("background")), ("out_width", int(64))]);
    assert_eq!(pad.fit, "pad");
    assert_eq!(layout_of(160, 96, &pad), (64, 48, Some((64, 38, 0, 5))));
    // Default 256 longest edge: content 256x171 (rounded), canvas padded up to 176.
    let bg = config(&[("preset", s("background"))]);
    assert_eq!(layout_of(300, 200, &bg), (256, 176, Some((256, 171, 0, 2))));
    assert_eq!(layout_of(200, 300, &bg), (176, 256, Some((171, 256, 2, 0))));
    assert_eq!(
        layout::fit_mode(&config(&[
            ("preset", s("background")),
            ("out_width", int(64)),
            ("out_height", int(48)),
        ])),
        None
    );
    assert_eq!(layout::fit_mode(&config(&[("preset", s("sprite"))])), None);
}

#[test]
fn test_tileset_fit_stretch_and_crop_fill_the_canvas() {
    for fit in ["stretch", "crop"] {
        let c = config(&[
            ("preset", s("background")),
            ("out_width", int(64)),
            ("fit", s(fit)),
        ]);
        assert_eq!(layout_of(160, 96, &c), (64, 48, None));
    }
    assert_eq!(layout::fit_crop(160, 96, 64, 48), (0, 96, 16, 144));
    assert_eq!(layout::fit_crop(96, 160, 64, 48), (44, 116, 0, 96));
    assert!(config_err(&[("fit", s("zoom"))]).contains("fit"));
}

#[test]
fn test_tileset_longest_edge_never_exceeds_the_target() {
    for fit in ["pad", "stretch", "crop"] {
        let c = config(&[
            ("preset", s("background")),
            ("tile_size", int(24)),
            ("fit", s(fit)),
        ]);
        let (out_w, out_h, _) = layout_of(300, 200, &c);
        assert!((out_w, out_h) == (240, 168) && out_w.max(out_h) <= 256);
    }
    // The Python test's `raises` block wrapped `Config(...)` too, which is what rejects it
    // (tile_size must be <= 512), before `resolve_layout` runs.
    let err = config_err(&[("preset", s("background")), ("tile_size", int(1000))]);
    assert!(err.contains("tile_size"), "{err}");
}

#[test]
fn test_stretch_warns_about_aspect_change_and_pad_does_not() {
    let warned = layout::resolve_layout(
        160,
        96,
        &config(&[
            ("preset", s("background")),
            ("out_width", int(64)),
            ("fit", s("stretch")),
        ]),
    )
    .unwrap();
    assert_eq!(warned.warnings.len(), 1);
    assert!(
        warned.warnings[0].contains("aspect ratio from 1.667"),
        "{}",
        warned.warnings[0]
    );
    for fit in ["pad", "crop"] {
        let c = config(&[
            ("preset", s("background")),
            ("out_width", int(64)),
            ("fit", s(fit)),
        ]);
        assert!(layout::resolve_layout(160, 96, &c)
            .unwrap()
            .warnings
            .is_empty());
    }
    // Within 2 %: no warning.
    let c = config(&[
        ("preset", s("background")),
        ("out_width", int(64)),
        ("fit", s("stretch")),
    ]);
    assert!(layout::resolve_layout(128, 95, &c)
        .unwrap()
        .warnings
        .is_empty());
}

// ---------------------------------------------------------------------------------------
// Pre-reduction and weights

/// `_reduce_config(**overrides)`.
fn reduce_config(overrides: &[(&'static str, Value)]) -> Config {
    let base: Vec<(&'static str, Value)> = vec![
        ("preset", s("sprite")),
        ("key_bg", b(false)),
        ("crop_to_alpha", b(false)),
        ("denoise", s("none")),
        ("out_width", int(16)),
        ("out_height", int(16)),
        ("prereduce_max_ratio", int(8)),
    ];
    config(&merge(&[&base, overrides]))
}

#[test]
fn test_prereduce_factor_and_disable() {
    let rgb = vec![0.25; 256 * 256 * 3];
    let alpha = vec![255u8; 256 * 256];
    let p = pre(&rgb, &alpha, 256, 256, &reduce_config(&[]));
    assert!(p.prereduce_factor == 2 && (p.h, p.w) == (128, 128));
    let off = pre(
        &rgb,
        &alpha,
        256,
        256,
        &reduce_config(&[("prereduce_max_ratio", int(0))]),
    );
    assert!(off.prereduce_factor == 1 && (off.h, off.w) == (256, 256));
    // Below 2 × ratio × output on either axis there is nothing to gain.
    let narrow = pre(
        &vec![0.25; 256 * 250 * 3],
        &vec![255u8; 256 * 250],
        256,
        250,
        &reduce_config(&[]),
    );
    assert_eq!(narrow.prereduce_factor, 1);
    config_err(&[("prereduce_max_ratio", int(-1))]);
}

#[test]
fn test_prereduce_block_mean_is_exact_and_remainder_is_trimmed() {
    let (h, w) = (5, 7);
    // Dyadic: exact in float64.
    let rgb: Vec<f64> = (0..h * w * 3).map(|i| (i % 16) as f64 / 16.0).collect();
    let small = layout::prereduce(&rgb, &vec![true; h * w], None, h, w, 2);
    assert!((small.h, small.w) == (2, 3) && small.mask.iter().all(|&m| m));
    let at = |y: usize, x: usize, c: usize| rgb[(y * w + x) * 3 + c];
    let mut expected = Vec::new();
    for by in 0..2 {
        for bx in 0..3 {
            for c in 0..3 {
                let (y, x) = (2 * by, 2 * bx);
                expected.push(
                    (at(y, x, c) + at(y + 1, x, c) + at(y, x + 1, c) + at(y + 1, x + 1, c)) / 4.0,
                );
            }
        }
    }
    assert_eq!(small.rgb, expected);
}

#[test]
fn test_prereduce_mask_threshold_and_transparent_fill_does_not_bleed() {
    let (h, w) = (4, 8);
    let red = [0.8, 0.2, 0.1];
    let mut rgb = vec![0.0; h * w * 3]; // transparent pixels are black ...
    fill_rect(&mut rgb, w, 0..h, 0..3, red); // ... the opaque part is red
    let mut m = vec![false; h * w];
    for y in 0..h {
        for x in 0..3 {
            m[y * w + x] = true; // columns 0-2 opaque
        }
    }
    m[4] = true; // one opaque pixel in block (0, 2)
    set(&mut rgb, w, 0, 4, red);
    let small = layout::prereduce(&rgb, &m, None, h, w, 2);
    // Block columns: 0 fully opaque, 1 half opaque (kept), 2 a quarter (dropped), 3 empty.
    assert_eq!(
        small.mask,
        vec![true, true, false, false, true, true, false, false]
    );
    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(&small.rgb[(y * 4 + x) * 3..(y * 4 + x) * 3 + 3], &red);
        }
    }
}

#[test]
fn test_pipeline_reports_prereduce_factor() {
    let mut rgba = vec![255u8; 256 * 256 * 4];
    for y in 0..256 {
        for x in 0..128 {
            rgba[(y * 256 + x) * 4..(y * 256 + x) * 4 + 3].copy_from_slice(&[40, 40, 40]);
        }
    }
    let result = run_loaded(
        &io::from_rgba(&rgba, 256, 256, "test".into()),
        &reduce_config(&[("method", s("box"))]),
    );
    assert_eq!(stat(&result, "prereduce_factor"), &Json::Int(2));
}

#[test]
fn test_opaque_input_has_unit_weight() {
    let rgb = random(0, 40 * 24 * 3);
    let p = pre(
        &rgb,
        &vec![255u8; 40 * 24],
        40,
        24,
        &config(&[
            ("out_width", int(16)),
            ("out_height", int(16)),
            ("key_bg", b(false)),
            ("seamless", b(true)),
        ]),
    );
    assert_eq!(p.weight.len(), p.mask.len());
    assert!(p.weight.iter().all(|&v| v == 1.0));
}

#[test]
fn test_weight_is_alpha_on_the_mask_through_repeat_and_prereduce() {
    let mut alpha = vec![255u8; 64];
    for y in 0..8 {
        alpha[y * 8 + 6] = 200;
        alpha[y * 8 + 7] = 100; // below alpha_threshold: transparent
    }
    let p = pre(
        &[0.5; 8 * 8 * 3],
        &alpha,
        8,
        8,
        &config(&[
            ("out_width", int(16)),
            ("out_height", int(16)),
            ("denoise", s("none")),
            ("crop_to_alpha", b(false)),
            ("outline", s("none")),
        ]),
    );
    assert_eq!((p.h, p.w), (16, 16)); // repeated 2x
    let a = 200.0 / 255.0;
    for y in 0..16 {
        let row = &p.weight[y * 16..y * 16 + 16];
        assert!(row[..12].iter().all(|&v| v == 1.0));
        assert!(allclose(&row[12..14], &[a, a]));
        assert!(row[14..].iter().all(|&v| v == 0.0));
    }
    assert!(p.weight.iter().zip(&p.mask).all(|(&v, &m)| m || v == 0.0));

    let mut big = vec![0u8; 256 * 256];
    for y in 0..256 {
        for x in 0..256 {
            big[y * 256 + x] = alpha[(y / 32) * 8 + x / 32];
        }
    }
    let reduced = pre(
        &vec![0.5; 256 * 256 * 3],
        &big,
        256,
        256,
        &reduce_config(&[]),
    );
    assert_eq!(reduced.prereduce_factor, 2);
    assert_eq!((reduced.h, reduced.w), (128, 128));
    for y in 0..128 {
        let row = &reduced.weight[y * 128..y * 128 + 128];
        assert!(row[..96].iter().all(|&v| v == 1.0));
        assert!(allclose(&row[96..112], &[a; 16]));
        assert!(row[112..].iter().all(|&v| v == 0.0));
    }
}

#[test]
fn test_prereduce_weight_is_the_block_mean() {
    let mut m = vec![true; 8];
    m[3] = false;
    m[7] = false;
    let mut weight: Vec<f64> = m.iter().map(|&k| if k { 0.5 } else { 0.0 }).collect();
    weight[0] = 1.0;
    weight[4] = 1.0;
    let small = layout::prereduce(&[0.0; 2 * 4 * 3], &m, Some(&weight), 2, 4, 2);
    assert!(small.mask.iter().all(|&k| k));
    assert!(allclose(&small.weight.unwrap(), &[0.75, 0.25]));
}

// ---------------------------------------------------------------------------------------
// Enhance

/// `_soft_feature()`: a mid-gray field with a faint dark dot the size of one output pixel.
fn soft_feature() -> (Vec<f64>, Vec<bool>) {
    let mut rgb = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        for x in 0..64 {
            let (dy, dx) = (y as f64 - 32.0, x as f64 - 32.0);
            let mut luma = 0.6 - 0.15 * (-(dy * dy + dx * dx) / (2.0 * 3.0f64.powi(2))).exp();
            luma += 0.1 * (x as f64 / 63.0); // a gradient much larger than the feature
            rgb.extend_from_slice(&[luma; 3]);
        }
    }
    (rgb, vec![true; 64 * 64])
}

fn enh(rgb: &[f64], m: &[bool], h: usize, w: usize, cell: f64, c: &Config) -> Vec<f64> {
    enhance::enhance(rgb, m, h, w, (cell, cell), c, None, false)
}

#[test]
fn test_enhance_off_is_identity() {
    let (rgb, m) = soft_feature();
    assert_eq!(enh(&rgb, &m, 64, 64, 8.0, &config(&[])), rgb);
}

#[test]
fn test_enhance_deepens_a_small_dark_feature() {
    let cases: [&[(&str, Value)]; 3] = [
        &[("enhance", float(1.0))],
        &[("ink", float(1.0))],
        &[("enhance", float(1.0)), ("ink", float(0.6))],
    ];
    let (rgb, m) = soft_feature();
    // Same column, off the dot.
    let contrast = |image: &[f64]| {
        mean(crop(image, 64, 3, 4..12, 28..36)) - mean(crop(image, 64, 3, 32..33, 32..33))
    };
    for overrides in cases {
        let out = enh(&rgb, &m, 64, 64, 8.0, &config(overrides));
        assert_eq!(out.len(), rgb.len());
        assert!(
            out.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "{overrides:?}"
        );
        assert!(contrast(&out) > 1.5 * contrast(&rgb), "{overrides:?}");
    }
}

#[test]
fn test_enhance_keeps_hue_and_ignores_transparent_pixels() {
    let (rgb, mut m) = soft_feature();
    let mut rgb: Vec<f64> = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[0] * 1.0, p[1] * 0.6, p[2] * 0.4])
        .collect();
    for y in 0..64 {
        for x in 0..16 {
            m[y * 64 + x] = false;
        }
    }
    fill_rect(&mut rgb, 64, 0..64, 0..16, [0.0, 1.0, 0.0]); // arbitrary color under alpha 0
    let c = config(&[("enhance", float(1.0)), ("ink", float(0.5))]);
    let out = enh(&rgb, &m, 64, 64, 8.0, &c);
    assert_eq!(
        crop(&out, 64, 3, 0..64, 0..16),
        crop(&rgb, 64, 3, 0..64, 0..16)
    );
    let inside: Vec<&[f64]> = out
        .chunks_exact(3)
        .zip(&m)
        .filter(|(_, &k)| k)
        .map(|(p, _)| p)
        .collect();
    let lit: Vec<&&[f64]> = inside.iter().filter(|p| p[0] > 0.0).collect();
    assert!(!lit.is_empty());
    let g: Vec<f64> = lit.iter().map(|p| p[1] / p[0]).collect();
    let bl: Vec<f64> = lit.iter().map(|p| p[2] / p[0]).collect();
    assert!(allclose(&g, &vec![0.6; g.len()]));
    assert!(allclose(&bl, &vec![0.4; bl.len()]));
    // The green under the mask did not leak into the opaque edge.
    let mut other = rgb.clone();
    fill_rect(&mut other, 64, 0..64, 0..16, [1.0, 0.0, 1.0]);
    let out2 = enh(&other, &m, 64, 64, 8.0, &c);
    for (i, &k) in m.iter().enumerate() {
        if k {
            assert_eq!(&out[i * 3..i * 3 + 3], &out2[i * 3..i * 3 + 3]);
        }
    }
}

#[test]
fn test_enhance_does_not_clip_saturated_colors() {
    // Brightening stops at the pixel's brightest channel: shading and hue both survive.
    let shade: Vec<f64> = (0..64)
        .map(|i| 0.2 + (0.95 - 0.2) / 63.0 * i as f64)
        .collect();
    for color_ in [[0.05, 0.1, 1.0], [1.0, 0.08, 0.05], [1.0, 0.75, 0.6]] {
        for strength in [0.5, 1.0, 4.0] {
            let mut rgb = Vec::with_capacity(64 * 64 * 3);
            for _y in 0..64 {
                for &sh in &shade {
                    rgb.extend(color_.iter().map(|&c| sh * c));
                }
            }
            let out = enh(
                &rgb,
                &[true; 64 * 64],
                64,
                64,
                8.0,
                &config(&[("enhance", float(strength))]),
            );
            let top = (0..3).fold(0, |best, i| if color_[i] > color_[best] { i } else { best });
            let row: Vec<f64> = (0..64).map(|x| out[(32 * 64 + x) * 3 + top]).collect();
            assert!(
                row.windows(2).all(|p| p[1] - p[0] >= -1e-9),
                "{color_:?} {strength}"
            );
            let mut rounded: Vec<i64> = row
                .iter()
                .map(|v| (v * 1000.0).round_ties_even() as i64)
                .collect();
            rounded.sort();
            rounded.dedup();
            assert!(rounded.len() > 20, "{color_:?} {strength}");
            let maxc = color_.iter().cloned().fold(f64::MIN, f64::max);
            let (mut got, mut want) = (Vec::new(), Vec::new());
            for p in out.chunks_exact(3).filter(|p| p[top] > 0.0) {
                for c in 0..3 {
                    got.push(p[c] / p[top]);
                    want.push(color_[c] / maxc);
                }
            }
            assert!(allclose(&got, &want), "{color_:?} {strength}");
        }
    }
}

#[test]
fn test_enhance_leaves_a_nearly_flat_subject_nearly_flat() {
    // One 8-bit step of difference is not stretched to black and white.
    let mut rgb = vec![128.0 / 255.0; 32 * 32 * 3];
    fill_rect(&mut rgb, 32, 0..32, 16..32, [129.0 / 255.0; 3]);
    let m = vec![true; 32 * 32];
    let c = config(&[("enhance", float(1.0))]);
    let out = enh(&rgb, &m, 32, 32, 2.0, &c);
    let max = out.iter().cloned().fold(f64::MIN, f64::max);
    let min = out.iter().cloned().fold(f64::MAX, f64::min);
    assert!(max - min < 0.05);
    assert!((mean(out.iter().copied()) - mean(rgb.iter().copied())).abs() < 0.02);
    let flat = vec![0.3; 32 * 32 * 3];
    assert!(allclose(&enh(&flat, &m, 32, 32, 2.0, &c), &flat));
}

#[test]
fn test_ink_does_not_depend_on_enhance() {
    // A faint dark dot is inked the same with or without the levels stretch.
    let mut rgb = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        for x in 0..64 {
            let (dy, dx) = (y as f64 - 32.0, x as f64 - 32.0);
            let luma = 0.5 + 0.05 * (x as f64 / 63.0) - 0.05 * (-(dy * dy + dx * dx) / 18.0).exp();
            rgb.extend_from_slice(&[luma; 3]);
        }
    }
    let m = vec![true; 64 * 64];
    let mut touched = Vec::new();
    for strength in [0.0, 0.25, 1.0] {
        let plain = enh(
            &rgb,
            &m,
            64,
            64,
            8.0,
            &config(&[("enhance", float(strength))]),
        );
        let inked = enh(
            &rgb,
            &m,
            64,
            64,
            8.0,
            &config(&[("enhance", float(strength)), ("ink", float(1.0))]),
        );
        let at = (32 * 64 + 32) * 3;
        assert!(inked[at] < 0.5 * plain[at], "{strength}");
        let t: Vec<bool> = inked
            .chunks_exact(3)
            .zip(plain.chunks_exact(3))
            .map(|(a, b)| max_abs_diff(a, b) > 1e-6)
            .collect();
        touched.push(t);
    }
    assert!(touched[0] == touched[1] && touched[0] == touched[2]);
}

#[test]
fn test_enhance_weighs_pixels_by_alpha() {
    // alpha_threshold=0 admits alpha-0 pixels; their RGB must barely count.
    let (h, w) = (40, 40);
    let rgb = gaussian_rgb(&random(0, h * w * 3), h, w, 2.0, Mode::Nearest);
    let m = vec![true; h * w];
    let mut weight = vec![1.0; h * w];
    for y in 0..h {
        for x in 0..10 {
            weight[y * w + x] = 1.0 / 255.0;
        }
    }
    let c = config(&[("enhance", float(1.0)), ("ink", float(0.5))]);
    let mut outs = Vec::new();
    let mut unweighted = Vec::new();
    for hidden in [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]] {
        let mut image = rgb.clone();
        fill_rect(&mut image, w, 0..h, 0..10, hidden);
        let out = enhance::enhance(&image, &m, h, w, (4.0, 4.0), &c, Some(&weight), false);
        outs.push(crop(&out, w, 3, 0..h, 10..w));
        let out = enhance::enhance(&image, &m, h, w, (4.0, 4.0), &c, None, false);
        unweighted.push(crop(&out, w, 3, 0..h, 10..w));
    }
    assert!(max_abs_diff(&outs[0], &outs[1]) < 0.03);
    assert!(max_abs_diff(&unweighted[0], &unweighted[1]) > 0.1);
}

#[test]
fn test_enhance_keeps_a_seamless_tile_periodic() {
    let cases: [&[(&str, Value)]; 3] = [
        &[("enhance", float(1.0))],
        &[("ink", float(1.0))],
        &[
            ("enhance", float(1.0)),
            ("enhance_radius", float(16.0)),
            ("ink", float(1.0)),
        ],
    ];
    let (h, w) = (48, 60);
    let rgb = gaussian_rgb(&random(1, h * w * 3), h, w, 3.0, Mode::Wrap);
    let m = vec![true; h * w];
    for overrides in cases {
        let c = config(overrides);
        let out = enhance::enhance(&rgb, &m, h, w, (6.0, 6.0), &c, None, true);
        assert!(max_abs_diff(&out, &rgb) > 0.1, "{overrides:?}");
        let rolled = enhance::enhance(
            &roll(&rgb, h, w, 3, 7, 31),
            &m,
            h,
            w,
            (6.0, 6.0),
            &c,
            None,
            true,
        );
        assert!(
            allclose_tol(&roll(&out, h, w, 3, 7, 31), &rolled, 1e-5, 1e-9),
            "{overrides:?}"
        );
        // Without wrap the same tile is filtered as if it ended at its border.
        let unwrapped = enhance::enhance(&rgb, &m, h, w, (6.0, 6.0), &c, None, false);
        assert!(!allclose_tol(&out, &unwrapped, 1e-5, 1e-3), "{overrides:?}");
    }
}

#[test]
fn test_seamless_padding_is_the_wrapped_enhanced_image() {
    let cases: [&[(&'static str, Value)]; 2] = [&[("enhance", float(1.0))], &[("ink", float(1.0))]];
    let loaded = io::load(&fixture_path("seamless_src"), io::MAX_INPUT_PIXELS).unwrap();
    for overrides in cases {
        let base: Vec<(&'static str, Value)> = vec![
            ("preset", s("background")),
            ("tileset", b(false)),
            ("seamless", b(true)),
            ("out_width", int(16)),
            ("out_height", int(16)),
            ("method", s("box")),
            ("denoise", s("none")),
            ("dither", s("none")),
        ];
        let c = config(&merge(&[&base, overrides]));
        let p = pre(&loaded.rgb, &loaded.alpha, loaded.h, loaded.w, &c);
        assert_eq!(p.prereduce_factor, 1);
        let py = (p.h - loaded.h) / 2;
        let px = (p.w - loaded.w) / 2;
        let (ch, cw) = (p.h - 2 * py, p.w - 2 * px);
        for y in 0..p.h {
            for x in 0..p.w {
                let sy = py + (y + ch - py % ch) % ch;
                let sx = px + (x + cw - px % cw) % cw;
                assert_eq!(
                    &p.lab[(y * p.w + x) * 3..(y * p.w + x) * 3 + 3],
                    &p.lab[(sy * p.w + sx) * 3..(sy * p.w + sx) * 3 + 3],
                    "{overrides:?} ({y}, {x})"
                );
            }
        }
        let plain = pre(
            &loaded.rgb,
            &loaded.alpha,
            loaded.h,
            loaded.w,
            &replace(&c, &[("enhance", float(0.0)), ("ink", float(0.0))]),
        );
        assert!(!allclose(&plain.lab, &p.lab), "{overrides:?}");
    }
}

#[test]
fn test_wide_blur_has_the_requested_width_at_constant_cost() {
    for sigma in [4.0, 7.3, 25.0] {
        let mut impulse = vec![0.0; 2001];
        impulse[1000] = 1.0;
        let kernel = enhance::gaussian(&impulse, 1, 2001, (0.1, sigma), Mode::Nearest);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-6, "{sigma}: {sum}");
        let var: f64 = kernel
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let x = i as f64 - 1000.0;
                k * x * x
            })
            .sum();
        assert!(
            (var.sqrt() - sigma).abs() <= 0.05 * sigma,
            "{sigma}: {}",
            var.sqrt()
        );
    }
}

#[test]
fn test_explicit_levels_replace_the_percentiles() {
    let (rgb, m) = soft_feature();
    let auto = enh(&rgb, &m, 64, 64, 8.0, &config(&[("enhance", float(1.0))]));
    let levels = config(&[
        ("enhance", float(1.0)),
        ("enhance_black", float(0.0)),
        ("enhance_white", float(1.0)),
    ]);
    let fixed = enh(&rgb, &m, 64, 64, 8.0, &levels);
    assert!(!allclose(&auto, &fixed));
    // With black 0 and white 1 the stretch is the identity: only the local contrast remains.
    let lifted: Vec<f64> = rgb.iter().map(|v| v + 0.05).collect();
    let brighter = enh(&lifted, &m, 64, 64, 8.0, &levels);
    let diff: Vec<f64> = crop(&brighter, 64, 3, 8..56, 8..56)
        .iter()
        .zip(crop(&fixed, 64, 3, 8..56, 8..56))
        .map(|(a, b)| a - b)
        .collect();
    assert!(allclose_tol(&diff, &vec![0.05; diff.len()], 1e-5, 5e-3));
}

#[test]
fn test_enhance_radius_is_in_output_pixels() {
    // The same picture at twice the resolution gives (nearly) the same output.
    let small = fixture("noisy_gradient");
    let mut big = Image::filled(small.h * 2, small.w * 2, &[0, 0, 0]);
    for y in 0..big.h {
        for x in 0..big.w {
            big.px_mut(y, x).copy_from_slice(small.px(y / 2, x / 2));
        }
    }
    let c = config(&[
        ("preset", s("background")),
        ("method", s("box")),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("denoise", s("none")),
        ("prereduce_max_ratio", int(0)),
        ("enhance", float(1.5)),
        ("ink", float(0.5)),
        ("dither", s("none")),
    ]);
    let a = run_image(&small, &c).unwrap().image;
    let bb = run_image(&big, &c).unwrap().image;
    assert_eq!(a.len(), bb.len());
    let diff = mean(
        a.iter()
            .zip(&bb)
            .map(|(&x, &y)| (i32::from(x) - i32::from(y)).abs() as f64),
    );
    assert!(diff < 2.0, "{diff}");
    let plain = run_image(
        &small,
        &replace(&c, &[("enhance", float(0.0)), ("ink", float(0.0))]),
    )
    .unwrap()
    .image;
    assert_ne!(a, plain);
}

// ---------------------------------------------------------------------------------------
// From tests/test_branches.py

#[test]
fn test_median_denoise() {
    let image = small("noisy_gradient", 4);
    let loaded = image.loaded();
    let fast_kw = fast();
    let median_kw = merge(&[&fast_kw, &[("denoise", s("median"))]]);
    let none = pre(
        &loaded.rgb,
        &loaded.alpha,
        loaded.h,
        loaded.w,
        &config(&fast_kw),
    );
    let median = pre(
        &loaded.rgb,
        &loaded.alpha,
        loaded.h,
        loaded.w,
        &config(&median_kw),
    );
    assert_eq!(median.lab.len(), none.lab.len());
    assert_eq!((median.h, median.w), (none.h, none.w));
    assert!(!allclose(&median.lab, &none.lab));
    // A 3×3 median removes a single outlier pixel.
    let mut flat = Image::filled(32, 32, &[128, 128, 128]);
    flat.px_mut(10, 10).copy_from_slice(&[255, 255, 255]);
    let spot = flat.loaded();
    let out = pre(&spot.rgb, &spot.alpha, spot.h, spot.w, &config(&median_kw));
    let l: Vec<f64> = out.lab.iter().step_by(3).copied().collect();
    let ptp =
        l.iter().cloned().fold(f64::MIN, f64::max) - l.iter().cloned().fold(f64::MAX, f64::min);
    assert!(ptp < 1e-9);
    let result = run_image(
        &image,
        &config(&merge(&[&[("method", s("box"))], &median_kw])),
    )
    .unwrap();
    assert_eq!((result.indices.h, result.indices.w), (16, 16));
}

/// The Rust library runs the external `rembg` command instead of importing the package, so
/// "missing dependency" means no `rembg` on PATH. The PATH is not changed here (tests run on
/// parallel threads): the test only runs where no `rembg` is installed. The `run` part
/// (`remove_bg=True` fails with this ConfigError) is
/// tests/pipeline_oracle.rs::remove_bg_without_the_command_is_a_config_error.
#[test]
fn test_remove_bg_missing_dependency_message() {
    if std::process::Command::new(preprocess::REMBG_COMMAND)
        .arg("--version")
        .output()
        .is_ok()
    {
        return;
    }
    let err =
        preprocess::remove_background_external(&[0.5; 8 * 8 * 3], &[255; 64], 8, 8).unwrap_err();
    assert_eq!(
        err.to_string(),
        "--remove-bg requires the optional 'rembg' dependency; install it with: pip install \
         'pixelforge[bg]'"
    );
    assert!(matches!(err, preprocess::Error::Config(_)));
}
