//! `pixelforge::color` against the scikit-image outputs recorded in tests/oracle
//! (docs/rust-migration-plan.md, Phase 1 oracles, Phase 2 colour).
//!
//! The conversions use `libm`'s `pow`/`cbrt` where numpy uses the platform's, and a plain
//! left-to-right dot product where skimage uses a BLAS matrix product, so floats may differ in
//! the last bits. Measured on macOS arm64 (2026-10-10): rgb_to_lab 5.7e-14, rgb8_to_lab
//! 5.7e-14, lab_to_rgb 6.7e-16, lab_to_rgb8 exact (0 of 600 differ). The bound is set well above that and
//! far below anything visible.

use std::path::PathBuf;

use pixelforge::color;
use serde_json::Value;

const BOUND: f64 = 1e-9;

fn oracle(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/oracle")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn flat_f64(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
        .collect()
}

fn flat_u8(v: &Value) -> Vec<u8> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| u8::try_from(x.as_u64().unwrap()).unwrap())
        })
        .collect()
}

fn max_abs_diff(name: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max = 0.0_f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(
            d <= BOUND,
            "{name}[{i}]: got {g:?}, want {w:?} (|d| = {d:e})"
        );
        max = max.max(d);
    }
    eprintln!(
        "{name}: max |deviation| = {max:e} over {} values",
        got.len()
    );
    max
}

#[test]
fn rgb_to_lab_matches_lab_lattice() {
    let o = oracle("lab_lattice.json");
    let rgb = flat_f64(&o["rgb"]);
    assert_eq!(rgb.len(), 3000);
    max_abs_diff(
        "lab_lattice rgb_to_lab",
        &color::rgb_to_lab(&rgb),
        &flat_f64(&o["lab"]),
    );
}

#[test]
fn rgb8_to_lab_matches_oracle() {
    let o = oracle("rgb8_lab.json");
    let rgb8 = flat_u8(&o["rgb8"]);
    assert_eq!(rgb8.len(), 264 * 3);
    max_abs_diff(
        "rgb8_lab rgb8_to_lab",
        &color::rgb8_to_lab(&rgb8),
        &flat_f64(&o["lab"]),
    );
}

#[test]
fn lab_to_rgb_matches_roundtrip_oracle() {
    let o = oracle("lab_roundtrip.json");
    let lab = flat_f64(&o["lab"]);
    assert_eq!(lab.len(), 600);
    max_abs_diff(
        "lab_roundtrip lab_to_rgb",
        &color::lab_to_rgb(&lab),
        &flat_f64(&o["lab_to_rgb"]),
    );
}

#[test]
fn lab_to_rgb8_matches_roundtrip_oracle_exactly() {
    let o = oracle("lab_roundtrip.json");
    let lab = flat_f64(&o["lab"]);
    let want = flat_u8(&o["lab_to_rgb8"]);
    let python_float = flat_f64(&o["lab_to_rgb"]);
    let got = color::lab_to_rgb8(&lab);
    assert_eq!(got.len(), want.len());
    let mut mismatches = 0;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if g == w {
            continue;
        }
        // Allowed only where the value sits within 1e-9 of a .5 rounding boundary.
        let scaled = python_float[i] * 255.0;
        let frac = scaled - scaled.floor();
        assert!(
            (frac - 0.5).abs() < 1e-9 && g.abs_diff(*w) == 1,
            "lab_to_rgb8[{i}] (lab {:?}): got {g}, want {w}, python value*255 = {scaled:?}",
            &lab[i / 3 * 3..i / 3 * 3 + 3]
        );
        eprintln!("lab_to_rgb8[{i}]: {g} vs {w}, on a .5 boundary ({scaled:?})");
        mismatches += 1;
    }
    eprintln!(
        "lab_roundtrip lab_to_rgb8: {mismatches} of {} differ",
        got.len()
    );
}

#[test]
fn lattice_round_trip_within_one_level() {
    let n = 18;
    let mut rgb = Vec::new();
    for r in 0..n {
        for g in 0..n {
            for b in 0..n {
                for c in [r, g, b] {
                    rgb.push(f64::from(c) / f64::from(n - 1));
                }
            }
        }
    }
    let back = color::lab_to_rgb(&color::rgb_to_lab(&rgb));
    let max = rgb
        .iter()
        .zip(&back)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("rgb -> lab -> rgb on a {n}^3 lattice: max error {max:e}");
    assert!(max <= 1.0 / 255.0, "round trip error {max}");

    // Every 8-bit grey and a coarse 8-bit lattice survive rgb8 -> lab -> rgb8 exactly.
    let mut rgb8 = Vec::new();
    for v in 0..=255u8 {
        rgb8.extend_from_slice(&[v, v, v]);
    }
    for r in (0..=255u8).step_by(17) {
        for g in (0..=255u8).step_by(17) {
            for b in (0..=255u8).step_by(17) {
                rgb8.extend_from_slice(&[r, g, b]);
            }
        }
    }
    assert_eq!(color::lab_to_rgb8(&color::rgb8_to_lab(&rgb8)), rgb8);
}

#[test]
fn nearest_index_matches_squared_distance_argmin() {
    // Lattice colours against the first 16 of them shifted: brute force equals a plain
    // first-minimum scan of delta_e.
    let o = oracle("lab_lattice.json");
    let lab = flat_f64(&o["lab"]);
    let palette: Vec<f64> = lab[..48].iter().map(|v| v + 0.25).collect();
    let idx = color::nearest_index(&lab, &palette);
    for (p, &k) in lab.chunks_exact(3).zip(&idx) {
        let reps: Vec<f64> = std::iter::repeat_n(p, 16).flatten().copied().collect();
        let d = color::delta_e(&reps, &palette);
        let first_min = (0..16).fold(0, |b, j| if d[j] < d[b] { j } else { b });
        assert_eq!(k as usize, first_min);
    }
}
