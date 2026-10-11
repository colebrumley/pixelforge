//! Port of tests/test_config.py: Config defaults, presets, validation messages, coercion,
//! `replace`, hashing, canonical JSON, provenance and the CLI's preset-shadowing warning.
//!
//! Python-only checks with no Rust counterpart are noted where they occur: dataclass
//! frozenness and read-only `PRESETS` (Rust enforces both at compile time: the fields need
//! a `mut` binding, `PRESETS` is a `const` slice), Python's `hash()` (the Rust `Config` has
//! no `std::hash::Hash`; the content hash `Config::hash` is compared instead) and the
//! exception class hierarchy / `__all__`.

mod common;

use std::cell::RefCell;

use common::*;
use pixelforge::config::{Config, FieldValue, Value, PRESETS};

/// `PRESETS[name]`.
fn preset(name: &str) -> &'static [(&'static str, FieldValue)] {
    PRESETS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, values)| *values)
        .unwrap()
}

/// `PRESETS[preset][key]` as a runtime value.
fn preset_value(name: &str, key: &str) -> Value {
    let (_, v) = preset(name).iter().find(|(k, _)| *k == key).unwrap();
    Value::from(*v)
}

fn get(config: &Config, name: &str) -> Value {
    config.get(name).unwrap()
}

fn hash(config: &Config) -> String {
    config.hash().unwrap()
}

/// `Config(**d)` for a dict of owned names.
fn from_pairs(pairs: Vec<(String, Value)>) -> Config {
    Config::new(&pairs).unwrap()
}

/// `Config(**cfg.to_dict())`.
fn from_dict(config: &Config) -> Config {
    from_pairs(
        config
            .to_dict()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// `json.loads(text)` of a flat object, as Config keyword arguments.
fn loads(text: &str) -> Vec<(String, Value)> {
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
    let serde_json::Value::Object(map) = parsed else {
        panic!("not a JSON object: {text}");
    };
    map.into_iter()
        .map(|(k, v)| {
            let value = match v {
                serde_json::Value::Null => Value::None,
                serde_json::Value::Bool(b) => Value::Bool(b),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Value::Int(i)
                    } else if n.is_u64() {
                        Value::BigInt(n.to_string())
                    } else {
                        Value::Float(n.as_f64().unwrap())
                    }
                }
                serde_json::Value::String(s) => Value::Str(s),
                other => panic!("unexpected JSON value {other}"),
            };
            (k, value)
        })
        .collect()
}

/// The keys of a flat JSON object in the order they appear in the text.
fn json_keys(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut keys = Vec::new();
    let mut depth = 0;
    let mut expect_key = false;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '{' | '[' => {
                depth += 1;
                expect_key = chars[i] == '{' && depth == 1;
            }
            '}' | ']' => depth -= 1,
            ',' if depth == 1 => expect_key = true,
            '"' => {
                let mut s = String::new();
                i += 1;
                while chars[i] != '"' {
                    if chars[i] == '\\' {
                        s.push(chars[i]);
                        i += 1;
                    }
                    s.push(chars[i]);
                    i += 1;
                }
                if expect_key {
                    keys.push(s);
                    expect_key = false;
                }
            }
            _ => {}
        }
        i += 1;
    }
    keys
}

/// Building the config fails (a `ValueError` in Python) and the message
/// contains `name`.
fn assert_err_mentions(pairs: &[(&str, Value)], name: &str) {
    let message = config_err(pairs);
    assert!(
        message.contains(name),
        "Config({pairs:?}): {message:?} does not mention {name}"
    );
}

#[test]
fn test_defaults_then_preset_then_explicit() {
    let sprite = config(&[]);
    assert_eq!(
        (
            sprite.preset.as_str(),
            sprite.method.as_str(),
            sprite.dither.as_str(),
            sprite.outline.as_str()
        ),
        ("sprite", "gerstner", "none", "auto")
    );
    let background = config(&[("preset", s("background"))]);
    for (key, value) in preset("background") {
        assert_eq!(get(&background, key), Value::from(*value), "{key}");
    }
    assert_eq!(
        config(&[
            ("preset", s("background")),
            ("method", s("box")),
            ("dither", s("bayer8"))
        ])
        .method,
        "box"
    );
    assert!(!sprite.remove_bg); // opt-in only, whatever is installed
}

#[test]
fn test_hash_is_stable_and_sensitive() {
    let a = config(&[("palette_size", int(12))]);
    let b_ = config(&[("palette_size", int(12))]);
    assert!(hash(&a) == hash(&b_) && hash(&a).len() == 64);
    assert_ne!(hash(&a), hash(&config(&[("palette_size", int(13))])));
    assert_eq!(
        hash(&config(&[("saturation_beta", int(1))])),
        hash(&config(&[("saturation_beta", float(1.0))]))
    );
    let canonical = a.canonical_json();
    assert!(!canonical.contains(' '));
    let keys = json_keys(&canonical);
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
    assert_eq!(keys.len(), Config::field_names().len());
    assert_eq!(from_pairs(loads(&canonical)), a);
}

#[test]
fn test_validation_errors() {
    let cases: Vec<Vec<(&str, Value)>> = vec![
        vec![("out_width", int(4))],
        vec![("out_height", int(7))],
        vec![("palette_size", int(1))],
        vec![("palette_size", int(257))],
        vec![("method", s("nearest"))],
        vec![("g_alpha", float(1.0))],
        vec![("g_alpha", float(0.0))],
        vec![("preset", s("tiles"))],
        vec![
            ("tileset", b(true)),
            ("out_width", int(40)),
            ("out_height", int(32)),
        ],
        vec![("outline", s("red"))],
        vec![("nonsense", int(1))],
        vec![("palette_size", s("16"))],
        vec![("dither", s("floyd"))],
    ];
    for kwargs in &cases {
        config_err(kwargs);
    }
}

#[test]
fn test_replace_reapplies_changed_preset() {
    let background = replace(
        &config(&[("preset", s("sprite"))]),
        &[("preset", s("background"))],
    );
    assert_eq!(background, config(&[("preset", s("background"))]));
    for key in ["method", "tileset", "outline", "palette_size"] {
        assert_eq!(
            get(&background, key),
            preset_value("background", key),
            "{key}"
        );
    }
    let kept = replace(
        &config(&[("preset", s("sprite")), ("palette_size", int(12))]),
        &[("preset", s("background"))],
    );
    assert!(kept.palette_size == 12 && kept.method == "kopf" && kept.tileset);
    // Fields set through replace() are explicit in the result and survive a later preset change.
    let chained = replace(
        &replace(&config(&[]), &[("method", s("box"))]),
        &[("preset", s("background"))],
    );
    assert!(chained.method == "box" && chained.tileset);
}

#[test]
fn test_replace_keeps_everything_else() {
    let cfg = config(&[
        ("preset", s("background")),
        ("palette_size", int(12)),
        ("dither", s("bayer8")),
        ("seed", int(3)),
    ]);
    let boxed = replace(&cfg, &[("method", s("box"))]);
    assert_eq!(boxed.method, "box");
    let without_method = |c: &Config| -> Vec<(&'static str, Value)> {
        c.to_dict()
            .into_iter()
            .filter(|(k, _)| *k != "method")
            .collect()
    };
    assert_eq!(without_method(&boxed), without_method(&cfg));
}

#[test]
fn test_round_trip_hashable_and_frozen() {
    let cfg = config(&[("preset", s("background")), ("palette_size", int(12))]);
    let again = from_dict(&cfg);
    // Python's hash(again) == hash(cfg): the Rust Config has no std Hash; equality and the
    // content hash stand in for it.
    assert!(again == cfg && hash(&again) == hash(&cfg));
    assert!(cfg.to_dict().iter().all(|(k, _)| !k.contains("explicit")));
    assert!(!cfg.canonical_json().contains("_explicit"));
    assert!(!Config::field_names().contains(&"_explicit"));
    // len({cfg, again, Config()}) == 2
    let mut distinct: Vec<Config> = Vec::new();
    for c in [cfg.clone(), again, config(&[])] {
        if !distinct.contains(&c) {
            distinct.push(c);
        }
    }
    assert_eq!(distinct.len(), 2);
    // `cfg.method = "box"` raising FrozenInstanceError has no runtime counterpart: mutating a
    // field needs a `mut` binding, checked at compile time.
}

thread_local! {
    /// Warnings logged on the current test thread (the CLI's stderr).
    static LOGGED: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct CaptureLogger;

impl log::Log for CaptureLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        LOGGED.with(|l| l.borrow_mut().push(record.args().to_string()));
    }

    fn flush(&self) {}
}

static CAPTURE: CaptureLogger = CaptureLogger;

/// What was logged on this thread since the last call (`capsys.readouterr().err`).
fn read_err() -> String {
    LOGGED.with(|l| std::mem::take(&mut *l.borrow_mut()).join("\n"))
}

#[test]
fn test_cli_warns_when_json_shadows_preset() {
    use pixelforge::cli::build_config;

    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Warn);
    let tmp = TempDir::new("cli-warns");
    let path = tmp.join("cfg.json");
    std::fs::write(
        &path,
        r#"{"preset": "sprite", "method": "gerstner", "palette_size": 16}"#,
    )
    .unwrap();
    read_err();
    let flags = vec![
        ("preset".to_string(), s("background")),
        ("palette_size".to_string(), int(8)),
    ];
    let config = build_config(Some(&path), &flags).unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(
        (
            config.preset.as_str(),
            config.method.as_str(),
            config.palette_size
        ),
        ("background", "gerstner", 8)
    );
    let err = read_err();
    assert!(
        err.contains("warning") && err.contains("method") && !err.contains("palette_size"),
        "{err:?}"
    );
    build_config(Some(&path), &[("palette_size".to_string(), int(8))])
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(read_err(), "");
}

#[test]
fn test_upper_bounds() {
    let cases: [(&str, i64); 9] = [
        ("out_width", 4096),
        ("out_height", 4096),
        ("scale_preview", 64),
        ("denoise_sigma_spatial", 16),
        ("kopf_max_iters", 1000),
        ("g_max_iters", 10000),
        ("saturation_beta", 5),
        ("tile_size", 512),
        ("seed", i64::MAX),
    ];
    for (name, limit) in cases {
        config(&[(name, int(limit))]);
        let over = match limit.checked_add(1) {
            Some(v) => int(v),
            None => Value::int_from_decimal("9223372036854775808"),
        };
        assert_err_mentions(&[(name, over)], name);
    }
}

#[test]
fn test_lower_and_range_bounds() {
    let cases: Vec<(&str, Value)> = vec![
        ("seed", int(-1)),
        ("dither_variance_threshold", float(-0.1)),
        ("tile_dedupe_tolerance", float(-1.0)),
        ("key_bg_tolerance", float(-0.01)),
        ("g_m", float(0.0)),
        ("kopf_tol", float(0.0)),
        ("seed", Value::int_from_decimal("18446744073709551616")), // 2**64
        (
            "orphan_min_region",
            Value::int_from_decimal("-9223372036854775809"), // -2**63 - 1
        ),
    ];
    for (name, value) in cases {
        config_err(&[(name, value)]);
    }
}

#[test]
fn test_non_finite_floats_are_rejected() {
    let names = [
        "saturation_beta",
        "g_m",
        "kopf_tol",
        "dither_strength",
        "dither_variance_threshold",
        "tile_dedupe_tolerance",
        "key_bg_tolerance",
        "g_T_final",
    ];
    for name in names {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_err_mentions(&[(name, float(value))], name);
        }
    }
}

#[test]
fn test_non_finite_values_for_int_fields_are_rejected() {
    for value in [f64::INFINITY, f64::NAN] {
        assert_err_mentions(&[("kopf_max_iters", float(value))], "kopf_max_iters");
    }
}

#[test]
fn test_outline_is_normalised_to_lowercase() {
    let upper = config(&[("outline", s("#AABBCC"))]);
    let lower = config(&[("outline", s("#aabbcc"))]);
    assert!(upper.outline == "#aabbcc" && hash(&upper) == hash(&lower));
    for bad in ["#aabbcc\n", "#aabbccdd", "aabbcc"] {
        assert_err_mentions(&[("outline", s(bad))], "outline");
    }
}

#[test]
fn test_scale_and_canvas_validation() {
    let cases: Vec<Vec<(&str, Value)>> = vec![
        vec![("scale", int(0))],
        vec![("scale", float(-1.5))],
        vec![("scale", int(4097))],
        vec![("scale", float(f64::NAN))],
        vec![("canvas", s("64"))],
        vec![("canvas", s("64x"))],
        vec![("canvas", s("64X64"))],
        vec![("canvas", s(" 64x64"))],
        vec![("canvas", s("7x64"))],
        vec![("canvas", s("64x4097"))],
        vec![("canvas", s("\u{ff16}\u{ff14}x64"))], // full-width "６４"
        vec![("canvas", int(64))],
        vec![("scale", int(2)), ("out_width", int(32))],
        vec![("canvas", s("64x64")), ("out_height", int(64))],
        // tileset: tile_size must divide it
        vec![("preset", s("background")), ("canvas", s("40x32"))],
    ];
    for kwargs in &cases {
        config_err(kwargs);
    }
}

#[test]
fn test_scale_and_canvas_accepted() {
    let cfg = config(&[("scale", int(2)), ("canvas", s("48x32"))]);
    assert!(cfg.scale == Some(2.0) && cfg.canvas_size() == Some((48, 32)));
    assert_eq!(config(&[("scale", int(4096))]).scale, Some(4096.0));
    assert_eq!(
        config(&[("canvas", s("4096x8"))]).canvas_size(),
        Some((4096, 8))
    );
    assert_eq!(
        config(&[("preset", s("background")), ("canvas", s("64x32"))]).canvas_size(),
        Some((64, 32))
    );
    // Python's hash(cfg) == hash(Config(scale=2.0, ...)): equality and the content hash.
    let same = config(&[("scale", float(2.0)), ("canvas", s("48x32"))]);
    assert!(same == cfg && hash(&same) == hash(&cfg));
    assert_eq!(from_pairs(loads(&cfg.canonical_json())), cfg);
    assert_eq!(config(&[]).canvas_size(), None);
}

#[test]
fn test_config_error_is_value_error_and_pixelforge_error() {
    // The class hierarchy (ConfigError subclasses ValueError and PixelforgeError) and
    // `pixelforge.__all__` are Python-only; in Rust a ConfigError is an std Error that the
    // pipeline's error converts from.
    let e = Config::new(&kw(&[("palette_size", int(1))])).unwrap_err();
    assert!(e.0.contains("palette_size"), "{e}");
    let as_std: &dyn std::error::Error = &e;
    assert!(as_std.to_string().contains("palette_size"));
    let pipeline_error: pixelforge::pipeline::Error = e.into();
    assert!(matches!(
        pipeline_error,
        pixelforge::pipeline::Error::Config(_)
    ));
    assert_err_mentions(&[("nope", int(1))], "unknown config field");
}

#[test]
fn test_negative_seed_rejected() {
    assert_err_mentions(&[("seed", int(-1))], "seed must be >= 0");
}

#[test]
fn test_presets_are_read_only() {
    // Assigning into PRESETS / PRESET_LONGEST_EDGE raising TypeError has no runtime
    // counterpart: both are `const` slices, immutable at compile time.
    let before = config(&[("preset", s("sprite"))]).to_dict();
    assert_eq!(config(&[("preset", s("sprite"))]).to_dict(), before);
    assert_eq!(config(&[("preset", s("sprite"))]).palette_size, 16);
    assert_eq!(
        pixelforge::config::PRESET_LONGEST_EDGE,
        &[("sprite", 64), ("background", 256)]
    );
}

#[test]
fn test_hash_covers_palette_file_contents() {
    let tmp = TempDir::new("palette-hash");
    let path = tmp.join("p.hex");
    let name = path.to_str().unwrap();
    std::fs::write(&path, "ff0000\n00ff00\n").unwrap();
    let before = config(&[("palette_name", s(name))]);
    let before_hash = hash(&before); // reads and caches the palette lines
    std::fs::write(&path, "ff0000\n0000ff\n").unwrap();
    let after = config(&[("palette_name", s(name))]);
    assert_eq!(before.canonical_json(), after.canonical_json());
    assert_eq!(before_hash, hash(&before));
    assert_ne!(hash(&before), hash(&after));
    // bundled: same way
    let pico8 = config(&[("palette_name", s("pico8"))]);
    assert_eq!(pico8.palette_hex_lines().unwrap().unwrap().len(), 16);
    let provenance = after.provenance().unwrap();
    assert_eq!(
        provenance.palette,
        Some(vec!["ff0000".to_string(), "0000ff".to_string()])
    );
    assert_eq!(provenance.config, after.to_dict());
    assert_eq!(provenance.palette_sha256.unwrap().len(), 64);
    assert_eq!(config(&[]).provenance().unwrap().palette, None);
    assert_eq!(config(&[]).provenance().unwrap().palette_sha256, None);
}

/// Every raise in Config._validate, with the start of the message it must produce.
fn validate_messages() -> Vec<(Vec<(&'static str, Value)>, String)> {
    let mut cases: Vec<(Vec<(&'static str, Value)>, String)> = vec![
        (
            vec![("palette_name", s("no_such_palette"))],
            "unknown palette 'no_such_palette'".into(),
        ),
        (
            vec![("out_width", int(7))],
            "out_width must be >= 8, got 7".into(),
        ),
        (
            vec![("out_height", int(7))],
            "out_height must be >= 8, got 7".into(),
        ),
        (vec![("scale", int(0))], "scale must be > 0, got 0.0".into()),
        (
            vec![("canvas", s("7x64"))],
            "canvas must be \"WxH\" with W, H in [8, 4096]".into(),
        ),
        (
            vec![("scale", int(2)), ("out_width", int(32))],
            "scale and canvas cannot be combined with out_width".into(),
        ),
        (
            vec![("canvas", s("32x32")), ("out_height", int(32))],
            "scale and canvas cannot be combined with out_width".into(),
        ),
        (
            vec![("palette_size", int(257))],
            "palette_size must be in [2, 256], got 257".into(),
        ),
        (
            vec![("method", s("nearest"))],
            "method must be one of ['box', 'kopf', 'gerstner']".into(),
        ),
        (
            vec![("denoise", s("gauss"))],
            "denoise must be one of ['none', 'bilateral', 'median']".into(),
        ),
        (
            vec![("dither", s("floyd"))],
            "dither must be one of ['none', 'bayer4', 'bayer8', 'auto']".into(),
        ),
        (
            vec![("palette_source", s("kmeans"))],
            "palette_source must be one of".into(),
        ),
        (
            vec![("transparent_index", s("middle"))],
            "transparent_index must be one of ['last', 'first']".into(),
        ),
        (
            vec![("fit", s("zoom"))],
            "fit must be one of ['pad', 'stretch', 'crop']".into(),
        ),
        (
            vec![("outline", s("red"))],
            "outline must be \"none\", \"auto\" or \"#rrggbb\", got 'red'".into(),
        ),
        (
            vec![("g_alpha", float(1.0))],
            "g_alpha must be in (0, 1), got 1.0".into(),
        ),
        (
            vec![("alpha_threshold", int(256))],
            "alpha_threshold must be in [0, 255], got 256".into(),
        ),
        (
            vec![("dither_strength", float(1.5))],
            "dither_strength must be in [0, 1], got 1.5".into(),
        ),
        (
            vec![("key_bg_tolerance", float(1.5))],
            "key_bg_tolerance must be in [0, 1], got 1.5".into(),
        ),
        (
            vec![("orphan_max_delta", float(-1.0))],
            "orphan_max_delta must be finite and >= 0, got -1.0".into(),
        ),
        (
            vec![("key_bg_fringe", int(9))],
            "key_bg_fringe must be in [0, 8], got 9".into(),
        ),
        (
            vec![("prereduce_max_ratio", int(-1))],
            "prereduce_max_ratio must be >= 0, got -1".into(),
        ),
        (
            vec![("outline_darken", float(1.5))],
            "outline_darken must be in [0, 1], got 1.5".into(),
        ),
    ];
    for name in [
        "kopf_max_iters",
        "g_max_iters",
        "tile_size",
        "scale_preview",
        "orphan_min_region",
    ] {
        cases.push((vec![(name, int(0))], format!("{name} must be >= 1, got 0")));
    }
    for name in [
        "g_T_final",
        "g_m",
        "kopf_tol",
        "saturation_beta",
        "denoise_sigma_color",
        "denoise_sigma_spatial",
        "g_bilateral_sigma_color",
        "g_bilateral_sigma_spatial",
    ] {
        cases.push((
            vec![(name, float(0.0))],
            format!("{name} must be > 0, got 0.0"),
        ));
    }
    cases.extend([
        (
            vec![("dither_variance_threshold", float(-0.5))],
            "dither_variance_threshold must be >= 0, got -0.5".into(),
        ),
        (
            vec![("tile_dedupe_tolerance", float(-0.5))],
            "tile_dedupe_tolerance must be >= 0, got -0.5".into(),
        ),
        (
            vec![("tileset_columns", int(257))],
            "tileset_columns must be 0 (automatic) or in [1, 256], got 257".into(),
        ),
        (vec![("seed", int(-1))], "seed must be >= 0, got -1".into()),
        (
            vec![("scale_preview", int(65))],
            "scale_preview must be <= 64, got 65".into(),
        ),
        (
            vec![
                ("preset", s("background")),
                ("out_width", int(40)),
                ("out_height", int(32)),
            ],
            "tile_size (16) must divide out_width (40) when tileset=True".into(),
        ),
        (
            vec![
                ("preset", s("background")),
                ("out_width", int(32)),
                ("out_height", int(40)),
            ],
            "tile_size (16) must divide out_height (40) when tileset=True".into(),
        ),
        (
            vec![("preset", s("background")), ("canvas", s("40x32"))],
            "tile_size (16) must divide canvas (40x32) when tileset=True".into(),
        ),
    ]);
    cases
}

#[test]
fn test_every_validation_message() {
    let cases = validate_messages();
    assert_eq!(cases.len(), 44);
    for (kwargs, message) in &cases {
        let got = config_err(kwargs);
        assert!(got.starts_with(message.as_str()), "{kwargs:?}: {got:?}");
    }
}

#[test]
fn test_enhance_fields_are_bounded() {
    let cases: Vec<Vec<(&str, Value)>> = vec![
        vec![("enhance", float(-0.1))],
        vec![("enhance", float(4.5))],
        vec![("ink", float(-0.1))],
        vec![("ink", float(1.1))],
        vec![("enhance_radius", int(0))],
        vec![("enhance_radius", int(17))],
        vec![("enhance_black", float(-0.1))],
        vec![("enhance_white", float(1.1))],
        vec![("enhance_black", float(0.5)), ("enhance_white", float(0.5))],
    ];
    for kwargs in &cases {
        config_err(kwargs);
    }
}

#[test]
fn test_enhance_is_off_in_every_preset() {
    for name in ["sprite", "background"] {
        let config = config(&[("preset", s(name))]);
        assert!(config.enhance == 0.0 && config.ink == 0.0, "{name}");
    }
}
