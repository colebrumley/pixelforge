//! Port of tests/test_color.py: LAB round trip, white point, 8-bit rounding and hex helpers,
//! nearest-colour search. The numpy-wrapper checks (dtypes, leading shapes, contiguity and
//! the binding's ValueError texts) have no Rust counterpart: the Rust API takes flat slices
//! of interleaved triples. Where a part of such a test is about values, that part is ported.

use std::path::PathBuf;

use pixelforge::color;
use serde_json::Value;

/// A small deterministic generator (splitmix64) standing in for `np.random.default_rng`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else {
        err.downcast_ref::<String>().cloned().unwrap_or_default()
    }
}

#[test]
fn test_round_trip_lattice() {
    let axis: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
    let mut rgb = Vec::new();
    for &r in &axis {
        for &g in &axis {
            for &b in &axis {
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
    }
    assert_eq!(rgb.len() / 3, 1000);
    let back = color::lab_to_rgb(&color::rgb_to_lab(&rgb));
    assert!(max_abs_diff(&back, &rgb) <= 1.0 / 255.0);
}

#[test]
fn test_white_and_black() {
    let lab = color::rgb_to_lab(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert!((lab[0] - 100.0).abs() <= 0.01);
    assert!(lab[3].abs() <= 0.01);
}

#[test]
fn test_rgb8_rounding_and_hex() {
    let rgb8: [u8; 9] = [255, 0, 0, 12, 34, 56, 0, 0, 0];
    assert_eq!(color::lab_to_rgb8(&color::rgb8_to_lab(&rgb8)), rgb8);
    assert_eq!(color::rgb8_to_hex([12, 34, 56]), "#0c2238");
    assert_eq!(color::hex_to_rgb8("#0c2238").unwrap(), [12, 34, 56]);
}

#[test]
fn test_nearest_index() {
    let palette = color::rgb8_to_lab(&[0, 0, 0, 255, 255, 255, 255, 0, 0]);
    let pixels = color::rgb8_to_lab(&[250, 10, 10, 20, 20, 20, 240, 240, 240]);
    assert_eq!(color::nearest_index(&pixels, &palette), vec![2, 0, 1]);
}

// ------------------------------------------------------------------ native conversions

fn oracle(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/oracle")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
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

#[test]
fn test_native_matches_skimage_oracle() {
    let lattice = oracle("lab_lattice.json");
    let lab = color::rgb_to_lab(&flat_f64(&lattice["rgb"]));
    assert!(max_abs_diff(&lab, &flat_f64(&lattice["lab"])) <= 1e-9);
    let rt = oracle("lab_roundtrip.json");
    let lab_in = flat_f64(&rt["lab"]);
    assert!(max_abs_diff(&color::lab_to_rgb(&lab_in), &flat_f64(&rt["lab_to_rgb"])) <= 1e-9);
    assert_eq!(color::lab_to_rgb8(&lab_in), flat_u8(&rt["lab_to_rgb8"]));
    let r8 = oracle("rgb8_lab.json");
    let got = color::rgb8_to_lab(&flat_u8(&r8["rgb8"]));
    assert!(max_abs_diff(&got, &flat_f64(&r8["lab"])) <= 1e-9);
}

/// The value half of the Python test: shapes (3,), (0, 3), (5, 3), (4, 6, 3), (2, 1, 2, 3)
/// are 1, 0, 5, 24 and 4 triples of a flat slice here (leading shapes and dtypes are numpy's).
#[test]
fn test_native_keeps_leading_shape() {
    let mut rng = Rng(7);
    for n in [1usize, 0, 5, 24, 4] {
        let rgb8: Vec<u8> = (0..n * 3).map(|_| rng.byte()).collect();
        let rgb: Vec<f64> = rgb8.iter().map(|&v| f64::from(v) / 255.0).collect();
        let lab = color::rgb_to_lab(&rgb);
        assert_eq!(lab.len(), n * 3);
        assert_eq!(color::rgb8_to_lab(&rgb8), lab, "{n} triples");
        let back = color::lab_to_rgb(&lab);
        assert_eq!(back.len(), n * 3);
        assert_eq!(color::lab_to_rgb8(&lab), rgb8, "{n} triples");
        // The conversion is per pixel: converting one triple at a time gives the same values.
        let one_by_one: Vec<f64> = rgb.chunks(3).flat_map(color::rgb_to_lab).collect();
        assert_eq!(one_by_one, lab);
    }
}

/// Only the "not triples" error has a Rust counterpart (a panic on a flat length that is not
/// a multiple of 3); the other cases test the binding's dtype and contiguity checks.
#[test]
fn test_python_wrappers_accept_non_contiguous_and_lists() {
    let err = std::panic::catch_unwind(|| color::rgb_to_lab(&[0.0; 8])).unwrap_err();
    let msg = panic_message(err);
    assert!(msg.contains("not a multiple of 3"), "{msg}");
}

#[test]
fn test_native_nearest_index_errors() {
    // "nearest_index: palette must not be empty" (a panic in Rust).
    let err = std::panic::catch_unwind(|| color::nearest_index(&[0.0; 6], &[])).unwrap_err();
    let msg = panic_message(err);
    assert!(msg.contains("nearest_index: empty palette"), "{msg}");
    // No pixels and no palette: an empty result.
    assert!(color::nearest_index(&[], &[]).is_empty());
}

#[test]
fn test_nearest_index_ties_go_to_lowest_index() {
    let palette = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0];
    let pixels = [
        5.0, 0.0, 0.0, // exact duplicate entries 2 and 3
        2.5, 0.0, 0.0, // halfway between entries 0 and 2
        7.5, 0.0, 0.0, // halfway between entries 2 and 1
        -1.0, 0.0, 0.0,
    ];
    assert_eq!(color::nearest_index(&pixels, &palette), vec![2, 0, 1, 0]);
}

#[test]
fn test_nearest_index_matches_brute_force_argmin() {
    let mut rng = Rng(11);
    let palette: Vec<f64> = (0..37 * 3).map(|_| rng.uniform(-50.0, 50.0)).collect();
    let pixels: Vec<f64> = (0..500 * 3).map(|_| rng.uniform(-60.0, 60.0)).collect();
    let expected: Vec<i64> = pixels
        .chunks(3)
        .map(|p| {
            let d: Vec<f64> = palette
                .chunks(3)
                .map(|q| (0..3).map(|i| (p[i] - q[i]).powi(2)).sum())
                .collect();
            let mut best = 0;
            for (k, &v) in d.iter().enumerate() {
                if v < d[best] {
                    best = k;
                }
            }
            best as i64
        })
        .collect();
    assert_eq!(color::nearest_index(&pixels, &palette), expected);
}
