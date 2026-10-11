//! `xtask golden [--check]`: regenerate `tests/golden.json` (`scripts/update_golden.py`),
//! the SHA-256 of the pixels each method/preset produces on three fixtures.
//!
//! Run it only when an output change is intentional (an algorithm fix, a new default), then
//! review and commit the diff together with the change.

use std::path::{Path, PathBuf};

use pixelforge::config::{Config, Value};
use pixelforge::pipeline::{self, Input, Output};
use pixelforge::pyjson::{dumps, Json, Layout};
use sha2::{Digest, Sha256};

use crate::{fail, Fail};

pub const GOLDEN_FIXTURES: [&str; 3] = ["two_color", "circle_alpha", "gradient6"];
pub const GOLDEN_METHODS: [&str; 3] = ["box", "kopf", "gerstner"];
pub const GOLDEN_PRESETS: [&str; 2] = ["sprite", "background"];

pub fn golden_path() -> PathBuf {
    crate::root().join("tests").join("golden.json")
}

/// 32 × 32 output without denoising (fast) or background keying (fixtures are synthetic).
pub fn golden_config(method: &str, preset: &str) -> Result<Config, Fail> {
    Config::new(&[
        ("preset".into(), Value::str(preset)),
        ("method".into(), Value::str(method)),
        ("out_width".into(), Value::Int(32)),
        ("out_height".into(), Value::Int(32)),
        ("denoise".into(), Value::str("none")),
        ("key_bg".into(), Value::Bool(false)),
    ])
    .map_err(|e| fail(1, e.0))
}

/// SHA-256 of the decoded RGBA pixels and the palette indices: `"rgba HxWx4\n"` + the RGBA
/// bytes, then `"indices HxW\n"` + the indices as int32 little-endian. Hashes arrays, not
/// PNG bytes (the encoded bytes vary between encoders).
pub fn pixel_digest(out: &Output) -> String {
    let (h, w) = (out.indices.h, out.indices.w);
    let mut digest = Sha256::new();
    digest.update(format!("rgba {h}x{w}x4\n").as_bytes());
    digest.update(&out.image);
    digest.update(format!("indices {h}x{w}\n").as_bytes());
    for &i in &out.indices.indices {
        digest.update((i as i32).to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `{"config_hash": …, "pixels_sha256": …}` for one key.
pub fn compute_entry(fixture: &str, method: &str, preset: &str) -> Result<Json, Fail> {
    let config = golden_config(method, preset)?;
    let path = crate::fixtures::default_dir().join(format!("{fixture}.png"));
    let out = pipeline::run(Input::Path(&path), &config, None)
        .map_err(|e| fail(1, format!("{fixture}/{method}/{preset}: {e}")))?;
    let hash = config.hash().map_err(|e| fail(1, e.to_string()))?;
    Ok(Json::Object(vec![
        ("config_hash".into(), Json::Str(hash)),
        ("pixels_sha256".into(), Json::Str(pixel_digest(&out))),
    ]))
}

/// `fixture/method/preset` for every fixture × method × preset, in that nesting order.
pub fn golden_keys() -> Vec<String> {
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

pub fn compute_all() -> Result<Json, Fail> {
    let mut entries = Vec::new();
    for key in golden_keys() {
        let parts: Vec<&str> = key.split('/').collect();
        entries.push((key.clone(), compute_entry(parts[0], parts[1], parts[2])?));
    }
    Ok(Json::Object(entries))
}

/// `json.dumps(golden, indent=2, sort_keys=True) + "\n"`.
pub fn render(golden: &Json) -> String {
    dumps(golden, Layout::INDENT2_SORTED) + "\n"
}

/// `xtask golden [--check]` against `path`.
pub fn main(path: &Path, check: bool) -> Result<(), Fail> {
    let text = render(&compute_all()?);
    if check {
        let current = std::fs::read_to_string(path).unwrap_or_default();
        if current != text {
            return Err(fail(1, format!("{} is out of date", path.display())));
        }
        return Ok(());
    }
    std::fs::write(path, text).map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
    println!("{}", path.display());
    Ok(())
}
