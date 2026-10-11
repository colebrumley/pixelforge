//! Port of the dithering tests of tests/test_branches.py (`test_bayer8_dither`,
//! `test_zero_dither_strength_is_no_dither`).

mod common;

use common::*;
use pixelforge::config::{Config, Value};
use pixelforge::quantize;

/// `_gray_ramp()`: 64 × 64 RGB, every row `np.round(np.linspace(30, 220, 64))`.
fn gray_ramp() -> Image {
    let step = (220.0 - 30.0) / 63.0;
    let ramp: Vec<u8> = (0..64)
        .map(|i| {
            let v = if i == 63 {
                220.0
            } else {
                30.0 + i as f64 * step
            };
            v.round_ties_even() as u8
        })
        .collect();
    let mut data = Vec::with_capacity(64 * 64 * 3);
    for _ in 0..64 {
        for &v in &ramp {
            data.extend_from_slice(&[v, v, v]);
        }
    }
    Image {
        h: 64,
        w: 64,
        c: 3,
        data,
    }
}

/// `_dither_config(**overrides)`.
fn dither_config(overrides: &[(&'static str, Value)]) -> Config {
    let base = [
        ("method", s("box")),
        ("palette_size", int(4)),
        ("palette_source", s("median_cut")),
    ];
    config(&merge(&[&base, &fast(), &raw(), overrides]))
}

/// `offsets[y0:y0+n, x0:x0+n]` of a `(16, 16, 3)` offset image, flattened.
fn block(offsets: &[f64], y0: usize, x0: usize, n: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for y in y0..y0 + n {
        let row = (y * 16 + x0) * 3;
        out.extend_from_slice(&offsets[row..row + n * 3]);
    }
    out
}

#[test]
fn test_bayer8_dither() {
    let image = gray_ramp();
    let plain = run_image(&image, &dither_config(&[("dither", s("none"))])).unwrap();
    let bayer8 = run_image(
        &image,
        &dither_config(&[("dither", s("bayer8")), ("dither_strength", float(1.0))]),
    )
    .unwrap();
    assert_eq!(plain.palette, bayer8.palette); // dithering only re-indexes
    assert_ne!(plain.indices.indices, bayer8.indices.indices);

    let offsets = quantize::dither_offsets(
        &vec![0.0; 16 * 16 * 3],
        None,
        16,
        16,
        &dither_config(&[("dither", s("bayer8"))]),
    )
    .unwrap();
    assert_eq!(block(&offsets, 0, 0, 8), block(&offsets, 8, 8, 8)); // period 8
    assert_ne!(block(&offsets, 0, 0, 4), block(&offsets, 4, 4, 4));
    let mut channel0: Vec<f64> = block(&offsets, 0, 0, 8).into_iter().step_by(3).collect();
    channel0.sort_by(f64::total_cmp);
    channel0.dedup();
    assert_eq!(channel0.len(), 64);
}

#[test]
fn test_zero_dither_strength_is_no_dither() {
    let image = gray_ramp();
    let plain = run_image(&image, &dither_config(&[("dither", s("none"))])).unwrap();
    for dither in ["bayer4", "bayer8", "auto"] {
        let zero = run_image(
            &image,
            &dither_config(&[("dither", s(dither)), ("dither_strength", float(0.0))]),
        )
        .unwrap();
        assert_eq!(plain.indices.indices, zero.indices.indices, "{dither}");
        assert_eq!(plain.palette, zero.palette, "{dither}");
    }
}
