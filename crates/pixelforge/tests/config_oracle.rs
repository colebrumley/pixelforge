//! The Rust Config against the Python oracle (tests/oracle/config_corpus.json and
//! config_errors.json, recorded by tests/oracle/record_oracles.py): canonical JSON, hash,
//! to_dict, palette and every ConfigError message, byte for byte.
//!
//! Palette paths in the corpus are relative to the repository root, so the tests run with
//! the working directory set there (the same directory for every test in this binary).

use pixelforge::config::{Config, Value};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(&root).unwrap();
    root
}

fn load(name: &str) -> serde_json::Value {
    let path = repo_root().join("tests/oracle").join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A JSON value as the Python object it was recorded from (tagged non-JSON values
/// included: `{"$float": "nan"}`, `{"$int": "<decimal>"}`).
fn value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if n.is_u64() {
                Value::int_from_decimal(&n.to_string())
            } else {
                Value::Float(n.as_f64().unwrap())
            }
        }
        serde_json::Value::String(s) => Value::Str(s.clone()),
        serde_json::Value::Array(items) => Value::List(items.iter().map(value).collect()),
        serde_json::Value::Object(map) => {
            if let Some(f) = map.get("$float") {
                Value::Float(match f.as_str().unwrap() {
                    "nan" => f64::NAN,
                    "inf" => f64::INFINITY,
                    "-inf" => f64::NEG_INFINITY,
                    other => panic!("unknown $float {other}"),
                })
            } else if let Some(i) = map.get("$int") {
                Value::int_from_decimal(i.as_str().unwrap())
            } else {
                panic!("untagged object {map:?}")
            }
        }
    }
}

fn kwargs(v: &serde_json::Value) -> Vec<(String, Value)> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), value(v)))
        .collect()
}

#[test]
fn corpus() {
    let doc = load("config_corpus.json");
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 100, "{} cases", cases.len());
    for case in cases {
        let label = case["kwargs"].to_string();
        let mut config =
            Config::new(&kwargs(&case["kwargs"])).unwrap_or_else(|e| panic!("{label}: {e}"));
        if let Some(chain) = case.get("replace") {
            for changes in chain.as_array().unwrap() {
                config = config
                    .replace(&kwargs(changes))
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
            }
        }
        assert_eq!(
            config.canonical_json(),
            case["canonical_json"].as_str().unwrap(),
            "{label}"
        );
        assert_eq!(
            config.hash().unwrap(),
            case["hash"].as_str().unwrap(),
            "{label}"
        );
        let dict = config.to_dict();
        let want = case["to_dict"].as_object().unwrap();
        assert_eq!(dict.len(), want.len(), "{label}");
        for (name, v) in &dict {
            assert_eq!(*v, value(&want[*name]), "{label}: {name}");
        }
        match case.get("palette_sha256") {
            Some(sha) => {
                config
                    .validate_palette()
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(
                    config.palette_sha256().unwrap().as_deref(),
                    sha.as_str(),
                    "{label}"
                );
                let lines: Vec<String> = case["palette"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|l| l.as_str().unwrap().to_string())
                    .collect();
                assert_eq!(
                    config.palette_hex_lines().unwrap(),
                    Some(lines.clone()),
                    "{label}"
                );
                let provenance = config.provenance().unwrap();
                assert_eq!(provenance.palette, Some(lines), "{label}");
            }
            None => {
                assert_eq!(config.palette_sha256().unwrap(), None, "{label}");
                assert_eq!(config.palette_name, None, "{label}");
            }
        }
    }
}

#[test]
fn errors() {
    let doc = load("config_errors.json");
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 104);
    let mut bad = Vec::new();
    for case in cases {
        let want = case["message"].as_str().unwrap();
        match Config::new(&kwargs(&case["kwargs"])) {
            Ok(_) => bad.push(format!("{}: no error, want {want:?}", case["python"])),
            Err(e) if e.0 != want => bad.push(format!(
                "{}:\n  got  {:?}\n  want {want:?}",
                case["python"], e.0
            )),
            Err(_) => {}
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} differ:\n{}",
        bad.len(),
        cases.len(),
        bad.join("\n")
    );
}

#[test]
fn versions_match_the_oracle() {
    let doc = load("versions.json");
    assert_eq!(doc["pixelforge"].as_str(), Some(env!("CARGO_PKG_VERSION")));
}
