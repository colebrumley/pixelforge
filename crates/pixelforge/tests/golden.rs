//! tests/test_golden.py: the output regression guard. Every fixture/method/preset is run
//! with `cargo xtask golden`'s config and its pixel digest and config hash compared with
//! tests/golden.json.

mod common;

use std::collections::BTreeMap;

use common::*;
use pixelforge::config::Config;
use sha2::{Digest, Sha256};

const GOLDEN_FIXTURES: [&str; 3] = ["two_color", "circle_alpha", "gradient6"];
const GOLDEN_METHODS: [&str; 3] = ["box", "kopf", "gerstner"];
const GOLDEN_PRESETS: [&str; 2] = ["sprite", "background"];

struct Entry {
    config_hash: String,
    pixels_sha256: String,
}

fn golden_text() -> String {
    std::fs::read_to_string(repo_root().join("tests/golden.json")).unwrap()
}

fn golden() -> BTreeMap<String, serde_json::Value> {
    serde_json::from_str(&golden_text()).unwrap()
}

/// `golden_keys()`: fixture/method/preset in the script's order.
fn golden_keys() -> Vec<String> {
    let mut keys = Vec::new();
    for fixture in GOLDEN_FIXTURES {
        for method in GOLDEN_METHODS {
            for preset in GOLDEN_PRESETS {
                keys.push(format!("{fixture}/{method}/{preset}"));
            }
        }
    }
    keys
}

/// `golden_config`: 32 × 32 output without denoising or background keying.
fn golden_config(method: &str, preset: &str) -> Config {
    config(&[
        ("preset", s(preset)),
        ("method", s(method)),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("denoise", s("none")),
        ("key_bg", b(false)),
    ])
}

/// `pixel_digest`: SHA-256 of `"rgba HxWx4\n"` + the RGBA bytes, then `"indices HxW\n"` +
/// the indices as little-endian int32.
fn pixel_digest(rgba: &[u8], indices: &[i64], h: usize, w: usize) -> String {
    assert_eq!(rgba.len(), h * w * 4);
    assert_eq!(indices.len(), h * w);
    let mut digest = Sha256::new();
    digest.update(format!("rgba {h}x{w}x4\n").as_bytes());
    digest.update(rgba);
    digest.update(format!("indices {h}x{w}\n").as_bytes());
    for &i in indices {
        digest.update(i32::try_from(i).unwrap().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// `compute_entry(fixture, method, preset)`.
fn compute_entry(key: &str) -> Entry {
    let parts: Vec<&str> = key.split('/').collect();
    let [fixture, method, preset] = parts[..] else {
        panic!("bad key {key}")
    };
    let cfg = golden_config(method, preset);
    let result = run_path(&fixture_path(fixture), &cfg).unwrap();
    Entry {
        config_hash: cfg.hash().unwrap(),
        pixels_sha256: pixel_digest(
            &result.image,
            &result.indices.indices,
            result.indices.h,
            result.indices.w,
        ),
    }
}

/// kopf on the 256-px fixtures takes 2-5 s per case (Python; `-m slow`).
fn is_slow(key: &str) -> bool {
    ["circle_alpha", "gradient6"].iter().any(|f| {
        GOLDEN_PRESETS
            .iter()
            .any(|p| key == format!("{f}/kopf/{p}"))
    })
}

#[test]
fn golden_file_lists_every_case() {
    let golden = golden();
    let mut expected = golden_keys();
    expected.sort();
    assert_eq!(golden.keys().cloned().collect::<Vec<_>>(), expected);
    assert_eq!(expected.len(), 18);
    for entry in golden.values() {
        let keys: Vec<&String> = entry.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["config_hash", "pixels_sha256"]);
    }
    // The file is `json.dumps(golden, indent=2, sort_keys=True) + "\n"` (serde_json's maps
    // are sorted, and its pretty printer indents by 2 with Python's separators).
    assert_eq!(
        golden_text(),
        serde_json::to_string_pretty(&golden).unwrap() + "\n"
    );
}

fn output_matches_golden(key: &str) {
    let golden = golden();
    let expected = Entry {
        config_hash: golden[key]["config_hash"].as_str().unwrap().to_string(),
        pixels_sha256: golden[key]["pixels_sha256"].as_str().unwrap().to_string(),
    };
    let actual = compute_entry(key);
    let hint = "If this output change is intentional, run `cargo xtask golden` and commit \
                tests/golden.json with the change.";
    assert_eq!(
        actual.config_hash, expected.config_hash,
        "{key}: config hash changed (a default, a preset or the version changed). {hint}"
    );
    assert_eq!(
        actual.pixels_sha256, expected.pixels_sha256,
        "{key}: output pixels changed. {hint}"
    );
}

fn output_matches_golden_for(fixture: &str) {
    for key in golden_keys() {
        if key.starts_with(&format!("{fixture}/")) && !is_slow(&key) {
            output_matches_golden(&key);
        }
    }
}

#[test]
fn output_matches_golden_two_color() {
    output_matches_golden_for("two_color");
}

#[test]
fn output_matches_golden_circle_alpha() {
    output_matches_golden_for("circle_alpha");
}

#[test]
fn output_matches_golden_gradient6() {
    output_matches_golden_for("gradient6");
}

#[test]
#[ignore] // slow
fn output_matches_golden_kopf_256px() {
    let slow: Vec<String> = golden_keys().into_iter().filter(|k| is_slow(k)).collect();
    assert_eq!(slow.len(), 4);
    for key in slow {
        output_matches_golden(&key);
    }
}
