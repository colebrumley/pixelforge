//! The whole Rust pipeline against the Python oracle (tests/oracle/outputs_small, recorded by
//! tests/oracle/record_oracles.py with `Result.save`): for each `<name>_config.json`, run
//! `pipeline::run` on the fixture, save, and compare every output. Text files byte for byte;
//! `_meta.json` without `environment` and `stats.timings`, as a structure and re-serialised
//! (so the layout is checked too); PNGs decoded (mode, size, palette, transparency, text,
//! indices and their SHA-256). `_meta.json`'s `config_hash` and `version` are the recording
//! version's (`versions.json`): the hash is checked at that version, then both are replaced
//! by the crate's before comparing.
//!
//! The first end-to-end test with no Python involved.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pixelforge::config::{Config, Value};
use pixelforge::io;
use pixelforge::pipeline::{self, Input};
use pixelforge::pyjson::{dumps, Json, Layout};
use sha2::{Digest, Sha256};

fn repo_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(&root).unwrap();
    root
}

fn oracle_dir() -> PathBuf {
    repo_root().join("tests/oracle/outputs_small")
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn kwargs(v: &serde_json::Value) -> Vec<(String, Value)> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            let value = match v {
                serde_json::Value::Null => Value::None,
                serde_json::Value::Bool(b) => Value::Bool(*b),
                serde_json::Value::Number(n) => match n.as_i64() {
                    Some(i) => Value::Int(i),
                    None => Value::Float(n.as_f64().unwrap()),
                },
                serde_json::Value::String(s) => Value::Str(s.clone()),
                other => panic!("unexpected kwarg {other}"),
            };
            (k.clone(), value)
        })
        .collect()
}

/// A parsed JSON value as `pyjson::Json` (keys in serde_json's sorted order).
fn to_json(v: &serde_json::Value) -> Json {
    match v {
        serde_json::Value::Null => Json::Null,
        serde_json::Value::Bool(b) => Json::Bool(*b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) if !n.is_f64() => Json::Int(i),
            _ => Json::Float(n.as_f64().unwrap()),
        },
        serde_json::Value::String(s) => Json::Str(s.clone()),
        serde_json::Value::Array(items) => Json::List(items.iter().map(to_json).collect()),
        serde_json::Value::Object(map) => {
            Json::Object(map.iter().map(|(k, v)| (k.clone(), to_json(v))).collect())
        }
    }
}

/// The `.png.json` form of a written PNG (see tests/oracle/README.md).
fn decode_png(path: &Path, full: bool) -> serde_json::Value {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut buf).unwrap();
    let info = reader.info();
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "{}", path.display());
    let (w, h) = (info.width as usize, info.height as usize);
    let mut out = serde_json::Map::new();
    let mode = match info.color_type {
        png::ColorType::Indexed => "P",
        png::ColorType::Rgba => "RGBA",
        other => panic!("unexpected colour type {other:?}"),
    };
    out.insert("mode".into(), mode.into());
    out.insert("size".into(), serde_json::json!([w, h]));
    let text: BTreeMap<String, String> = io::read_png_text(path).unwrap().into_iter().collect();
    out.insert("text".into(), serde_json::to_value(text).unwrap());
    // Pillow's "transparency" for a P image whose tRNS is 255... then one 0: that index.
    let transparency = match info.trns.as_deref() {
        None => serde_json::Value::Null,
        Some(trns) => {
            let zeros: Vec<usize> = (0..trns.len()).filter(|&i| trns[i] == 0).collect();
            assert_eq!(zeros.len(), 1, "tRNS has one transparent entry");
            assert!(trns.iter().all(|&a| a == 0 || a == 255));
            assert_eq!(zeros[0], trns.len() - 1);
            zeros[0].into()
        }
    };
    out.insert("transparency".into(), transparency);
    if let Some(plte) = info.palette.as_deref() {
        out.insert("palette".into(), serde_json::to_value(plte).unwrap());
    }
    let data = &buf[..frame.buffer_size()];
    out.insert(
        "pixels_sha256".into(),
        format!("{:x}", Sha256::digest(data)).into(),
    );
    if full {
        let channels = if mode == "P" { 1 } else { 4 };
        let rows: Vec<serde_json::Value> = data
            .chunks_exact(w * channels)
            .map(|row| {
                if channels == 1 {
                    serde_json::to_value(row).unwrap()
                } else {
                    serde_json::to_value(row.chunks_exact(4).collect::<Vec<_>>()).unwrap()
                }
            })
            .collect();
        assert_eq!(rows.len(), h);
        out.insert("pixels".into(), rows.into());
    }
    serde_json::Value::Object(out)
}

/// The pixelforge version the oracle was recorded with (`tests/oracle/versions.json`).
fn recorded_version() -> String {
    let path = repo_root().join("tests/oracle/versions.json");
    read_json(&path)["pixelforge"].as_str().unwrap().to_string()
}

/// `Config.hash()` as pixelforge `version` computes it (see config_oracle.rs).
fn hash_at(config: &Config, version: &str) -> String {
    let mut payload = format!("{}\n{version}", config.canonical_json());
    if let Some(sha) = config.palette_sha256().unwrap() {
        payload.push_str("\npalette:");
        payload.push_str(&sha);
    }
    format!("{:x}", Sha256::digest(payload.as_bytes()))
}

fn strip_meta(mut meta: serde_json::Value) -> serde_json::Value {
    let obj = meta.as_object_mut().unwrap();
    obj.remove("environment").expect("meta has environment");
    obj.get_mut("stats")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("timings")
        .expect("stats has timings");
    meta
}

#[test]
fn outputs_small_match_the_python_pipeline() {
    let dir = oracle_dir();
    let mut configs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with("_config.json"))
        .collect();
    configs.sort();
    assert_eq!(configs.len(), 2, "two recorded runs");
    let tmp =
        std::env::temp_dir().join(format!("pixelforge-pipeline-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    for config_path in configs {
        let file = config_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let prefix = file.strip_suffix("_config.json").unwrap().to_string();
        let spec = read_json(&config_path);
        let config = Config::new(&kwargs(&spec["kwargs"])).unwrap();
        let fixture = repo_root().join(spec["fixture"].as_str().unwrap());
        let out = pipeline::run(Input::Path(&fixture), &config, None).unwrap();
        let paths = out.save(&tmp.join(&prefix)).unwrap();
        let mut compared = 0;
        for (name, path) in &paths {
            let base = path.file_name().unwrap().to_string_lossy().into_owned();
            if base.ends_with(".png") {
                let want = read_json(&dir.join(format!("{base}.json")));
                let got = decode_png(path, name != "preview");
                assert_eq!(got, want, "{base}");
            } else if name == "meta" {
                // The recorded `config_hash` and `version` are the recording version's.
                let recorded = std::fs::read_to_string(dir.join(&base)).unwrap();
                let recorded_hash = read_json(&dir.join(&base))["config_hash"]
                    .as_str()
                    .unwrap()
                    .to_string();
                let version = recorded_version();
                assert_eq!(recorded_hash, hash_at(&config, &version), "{base}");
                let want_text = recorded
                    .replace(&recorded_hash, &config.hash().unwrap())
                    .replace(
                        &format!("\"version\": \"{version}\""),
                        &format!("\"version\": \"{}\"", pipeline::VERSION),
                    );
                let got = strip_meta(read_json(path));
                let want: serde_json::Value = serde_json::from_str(&want_text).unwrap();
                assert_eq!(got, want, "{base}");
                let text = dumps(&to_json(&got), Layout::INDENT2_SORTED) + "\n";
                assert_eq!(text, want_text, "{base} re-serialised");
                // And the file itself: the same text with the two host-dependent keys.
                let written = std::fs::read_to_string(path).unwrap();
                assert!(
                    written.contains("\n  \"environment\": {\n"),
                    "{base} layout"
                );
            } else {
                let want = std::fs::read(dir.join(&base)).unwrap();
                let got = std::fs::read(path).unwrap();
                assert_eq!(
                    String::from_utf8_lossy(&got),
                    String::from_utf8_lossy(&want),
                    "{base}"
                );
                assert_eq!(got, want, "{base}");
            }
            compared += 1;
        }
        let expected = if config.tileset { 9 } else { 5 };
        assert_eq!(compared, expected, "{prefix}: every output compared");
        // Every recorded file of this run was produced.
        for entry in std::fs::read_dir(&dir).unwrap() {
            let base = entry.unwrap().file_name().to_string_lossy().into_owned();
            if base.starts_with(&format!("{prefix}_")) || base.starts_with(&format!("{prefix}.")) {
                if base == file {
                    continue;
                }
                let produced = base.strip_suffix(".json").filter(|b| b.ends_with(".png"));
                let produced = produced.unwrap_or(&base);
                assert!(tmp.join(produced).is_file(), "{produced} not written");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn array_input_hash_and_stats_order() {
    // A 2-colour 8×8 RGB array: hashed as `rgba8 HxW\n` + bytes, stats in Python's order.
    let mut data = vec![0u8; 8 * 8 * 3];
    for (i, px) in data.chunks_exact_mut(3).enumerate() {
        px.copy_from_slice(if i % 8 < 4 {
            &[200, 30, 30]
        } else {
            &[30, 30, 200]
        });
    }
    let config = Config::new(&[
        ("method".into(), Value::str("box")),
        ("denoise".into(), Value::str("none")),
        ("key_bg".into(), Value::Bool(false)),
        ("out_width".into(), Value::Int(8)),
        ("out_height".into(), Value::Int(8)),
        ("outline".into(), Value::str("none")),
    ])
    .unwrap();
    let out = pipeline::run(
        Input::Array {
            shape: &[8, 8, 3],
            data: io::ArrayData::U8(&data),
        },
        &config,
        None,
    )
    .unwrap();
    let mut rgba = Vec::new();
    for px in data.chunks_exact(3) {
        rgba.extend_from_slice(px);
        rgba.push(255);
    }
    assert_eq!(out.input_sha256, io::rgba8_digest(8, 8, &rgba));
    let Json::Object(stats) = &out.stats else {
        panic!("stats is an object")
    };
    let keys: Vec<&str> = stats.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        [
            "iterations",
            "final_palette_size",
            "colors_used",
            "colors_unused",
            "config_hash",
            "background_keyed",
            "fit",
            "content_size",
            "outline_index",
            "outline_margin",
            "outline_clipped",
            "prereduce_factor",
            "method",
            "timings"
        ]
    );
    let Some((_, Json::Object(timings))) = stats.iter().find(|(k, _)| k == "timings") else {
        panic!("timings")
    };
    let names: Vec<&str> = timings.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        names,
        [
            "load",
            "preprocess",
            "downscale",
            "palette_quantize",
            "postprocess"
        ]
    );
    assert_eq!(out.image.len(), 8 * 8 * 4);
    assert_eq!(out.palette.len(), 2);
}

#[test]
fn remove_bg_without_the_command_is_a_config_error() {
    // Only meaningful where no `rembg` executable is installed (the CI and dev machines).
    if std::process::Command::new(pixelforge::preprocess::REMBG_COMMAND)
        .arg("--version")
        .output()
        .is_ok()
    {
        return;
    }
    let config = Config::new(&[
        ("remove_bg".into(), Value::Bool(true)),
        ("denoise".into(), Value::str("none")),
    ])
    .unwrap();
    let data = vec![0u8; 16 * 16 * 3];
    let err = pipeline::run(
        Input::Array {
            shape: &[16, 16, 3],
            data: io::ArrayData::U8(&data),
        },
        &config,
        None,
    )
    .unwrap_err();
    match err {
        pipeline::Error::Config(e) => assert_eq!(e.0, pixelforge::preprocess::REMBG_MISSING),
        other => panic!("expected a ConfigError, got {other:?}"),
    }
}
