//! `pixelforge::filters` and `pixelforge::enhance::percentile` against the scipy, numpy and
//! scikit-image outputs recorded in tests/oracle (ndimage_filters.json,
//! skimage_bilateral.json). Uniform and median filters and percentiles are bitwise; the
//! Gaussian filter and the bilateral differ from the recordings in the last bits only: the
//! recording machine's scipy and scikit-image builds contract part of their `acc += a * b`
//! loops into fused multiply-adds (the port does not, Decision 6), and numpy's `exp` and
//! `libm::exp` differ in a few kernel values' last bit.

use std::path::PathBuf;

use pixelforge::enhance::percentile;
use pixelforge::filters::{
    denoise_bilateral, gaussian_filter1d, median_filter3_nearest, uniform_filter1d, Mode,
};
use serde_json::Value;

fn oracle(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/oracle")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// A 1-D or 2-D array as (h, w, values, one_d); a 1-D array of length n is (n, 1).
fn grid(v: &Value) -> (usize, usize, Vec<f64>, bool) {
    let items = v.as_array().unwrap();
    if items[0].is_array() {
        let w = items[0].as_array().unwrap().len();
        let data: Vec<f64> = items.iter().flat_map(floats).collect();
        (items.len(), w, data, false)
    } else {
        (items.len(), 1, floats(v), true)
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn ndimage_filters_match_scipy() {
    let o = oracle("ndimage_filters.json");
    let arrays = o["arrays"].as_object().unwrap();
    let (mut uniform, mut gauss, mut median) = (0, 0, 0);
    let mut gauss_max = 0.0f64;
    for f in o["filters"].as_array().unwrap() {
        let p = &f["params"];
        let func = f["func"].as_str().unwrap();
        let input = f["input"].as_str().unwrap();
        let (h, w, x, one_d) = grid(&arrays[input]);
        let (_, _, want, _) = grid(&f["output"]);
        let mode = p["mode"].as_str().map(|m| Mode::parse(m).unwrap());
        let axis = p["axis"].as_u64().unwrap_or(0) as usize;
        match func {
            "uniform_filter1d" => {
                let size = p["size"].as_u64().unwrap() as usize;
                let got = uniform_filter1d(&x, h, w, size, axis, mode.unwrap());
                assert_eq!(bits(&got), bits(&want), "{func} {input} {p}");
                uniform += 1;
            }
            "uniform_filter" => {
                let m = mode.unwrap();
                let mut got = uniform_filter1d(&x, h, w, 3, 0, m);
                if !one_d {
                    got = uniform_filter1d(&got, h, w, 3, 1, m);
                }
                assert_eq!(bits(&got), bits(&want), "{func} {input} {p}");
                uniform += 1;
            }
            "gaussian_filter1d" => {
                let sigma = p["sigma"].as_f64().unwrap();
                let got = gaussian_filter1d(&x, h, w, sigma, axis, mode.unwrap());
                let d = max_abs_diff(&got, &want);
                // Measured: 2.2e-16 (macOS arm64 scipy's FMA contraction; sigma 3.9 also exp).
                assert!(d <= 1e-12, "{func} {input} {p}: {d:e}");
                gauss_max = gauss_max.max(d);
                gauss += 1;
            }
            "median_filter" => {
                // A 1-D array is the (1, n) case.
                let (hh, ww) = if one_d { (1, h) } else { (h, w) };
                let got = median_filter3_nearest(&x, hh, ww);
                assert_eq!(bits(&got), bits(&want), "{func} {input} {p}");
                median += 1;
            }
            other => panic!("unexpected filter {other}"),
        }
    }
    assert_eq!((uniform, gauss, median), (43, 20, 3));
    eprintln!("gaussian_filter1d max deviation {gauss_max:e}");
}

#[test]
fn percentiles_match_numpy_bitwise() {
    let o = oracle("ndimage_filters.json");
    let samples = o["samples"].as_object().unwrap();
    let mut n = 0;
    for case in o["percentile"].as_array().unwrap() {
        let input = case["input"].as_str().unwrap();
        let values: Vec<f64> = match &samples[input] {
            v if v.as_array().unwrap()[0].is_array() => grid(v).2,
            v => floats(v),
        };
        let got = percentile(&values, &floats(&case["q"]));
        assert_eq!(bits(&got), bits(&floats(&case["output"])), "{input}");
        n += 1;
    }
    assert_eq!(n, 6);
}

#[test]
fn bilateral_matches_skimage() {
    let o = oracle("skimage_bilateral.json");
    let mut worst = 0.0f64;
    let mut flips = 0;
    for case in o["cases"].as_array().unwrap() {
        let size = case["size"].as_u64().unwrap() as usize;
        let rgb: Vec<f64> = grid(&case["filled_rgb8"])
            .2
            .iter()
            .map(|v| v / 255.0)
            .collect();
        let want = grid(&case["output"]).2;
        let sc = case["sigma_color"].as_f64().unwrap();
        let ss = case["sigma_spatial"].as_f64().unwrap();
        let got = denoise_bilateral(&rgb, size, size, sc, ss, 2);
        let d = max_abs_diff(&got, &want);
        flips += got
            .iter()
            .zip(&want)
            .filter(|(a, b)| (*a * 255.0).round_ties_even() != (*b * 255.0).round_ties_even())
            .count();
        // Measured: 3.3e-16 over the 5 cases; 1e-9 is the bound the Python tests use too.
        assert!(d <= 1e-9, "{} {sc} {ss}: {d:e}", case["fixture"]);
        worst = worst.max(d);
    }
    assert_eq!(flips, 0, "8-bit values that would change");
    eprintln!("bilateral max deviation {worst:e}");
}
