//! `pixelforge::mask::edt` against `scipy.ndimage.distance_transform_edt(~mask,
//! return_distances=True, return_indices=True)` recorded in tests/oracle/edt_indices.json:
//! distances, row and column indices bitwise identical on all eight masks (which pins
//! scipy's choice among equidistant opaque pixels).

use std::path::PathBuf;

use pixelforge::mask;
use serde_json::Value;

fn oracle(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/oracle")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn rows(v: &Value) -> Vec<&Vec<Value>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap())
        .collect()
}

#[test]
fn edt_matches_scipy_bitwise() {
    let data = oracle("edt_indices.json");
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mask_rows: Vec<&str> = case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        let (h, w) = (mask_rows.len(), mask_rows[0].len());
        let m: Vec<bool> = mask_rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| b == b'1'))
            .collect();
        let (dist, r, c) = mask::edt(&m, h, w);
        let want_d: Vec<f64> = rows(&case["distances"])
            .into_iter()
            .flat_map(|row| row.iter().map(|x| x.as_f64().unwrap()))
            .collect();
        let want_r: Vec<i64> = rows(&case["indices_row"])
            .into_iter()
            .flat_map(|row| row.iter().map(|x| x.as_i64().unwrap()))
            .collect();
        let want_c: Vec<i64> = rows(&case["indices_col"])
            .into_iter()
            .flat_map(|row| row.iter().map(|x| x.as_i64().unwrap()))
            .collect();
        assert_eq!(want_d.len(), h * w, "{name}: oracle size");
        for i in 0..h * w {
            assert_eq!(
                (r[i] as i64, c[i] as i64),
                (want_r[i], want_c[i]),
                "{name}: indices at ({}, {})",
                i / w,
                i % w
            );
            assert_eq!(
                dist[i].to_bits(),
                want_d[i].to_bits(),
                "{name}: distance at ({}, {})",
                i / w,
                i % w
            );
        }
    }
}
