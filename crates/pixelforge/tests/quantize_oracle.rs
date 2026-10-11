//! `pixelforge::quantize`'s 3-wide uniform filter against scipy's outputs recorded in
//! tests/oracle/ndimage_filters.json: `uniform_filter1d` (size 3, mode "nearest", both
//! axes) and `uniform_filter` (size 3, mode "nearest"), bit for bit.

use std::path::PathBuf;

use pixelforge::quantize::{uniform_filter1d_3_nearest, uniform_filter_3_nearest};
use serde_json::Value;

fn oracle() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/oracle/ndimage_filters.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// A 1-D or 2-D array as (h, w, values); a 1-D array of length n is (n, 1).
fn grid(v: &Value) -> (usize, usize, Vec<f64>) {
    let items = v.as_array().unwrap();
    if items[0].is_array() {
        let w = items[0].as_array().unwrap().len();
        let data: Vec<f64> = items
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect();
        (items.len(), w, data)
    } else {
        (
            items.len(),
            1,
            items.iter().map(|x| x.as_f64().unwrap()).collect(),
        )
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn uniform_filters_match_scipy_bitwise() {
    let o = oracle();
    let arrays = o["arrays"].as_object().unwrap();
    let mut checked = 0;
    for f in o["filters"].as_array().unwrap() {
        let params = &f["params"];
        if params["size"].as_i64() != Some(3) || params["mode"].as_str() != Some("nearest") {
            continue;
        }
        let func = f["func"].as_str().unwrap();
        let input = f["input"].as_str().unwrap();
        let (h, w, x) = grid(&arrays[input]);
        let one_d = !arrays[input].as_array().unwrap()[0].is_array();
        let (_, _, want) = grid(&f["output"]);
        let got = match func {
            "uniform_filter1d" => {
                uniform_filter1d_3_nearest(&x, h, w, params["axis"].as_u64().unwrap() as usize)
            }
            // A 1-D array is filtered along its only axis.
            "uniform_filter" if one_d => uniform_filter1d_3_nearest(&x, h, w, 0),
            "uniform_filter" => uniform_filter_3_nearest(&x, h, w),
            _ => continue,
        };
        assert_eq!(bits(&got), bits(&want), "{func} {input} {params}");
        checked += 1;
    }
    assert_eq!(
        checked, 8,
        "expected 5 uniform_filter1d and 3 uniform_filter cases"
    );
}

#[test]
fn a_length_one_axis_is_still_filtered() {
    // (h, 1): the axis-1 pass computes (v + v + v) / 3, which need not give v back.
    let x = [0.1, 0.7, 0.30000000000000004];
    let along0 = uniform_filter1d_3_nearest(&x, 3, 1, 0);
    let expected: Vec<f64> = along0.iter().map(|&v| (0.0 + v + v + v) / 3.0).collect();
    assert_eq!(bits(&uniform_filter_3_nearest(&x, 3, 1)), bits(&expected));
}
