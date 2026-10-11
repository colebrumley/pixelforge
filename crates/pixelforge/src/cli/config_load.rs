//! `build_config`: field defaults ← preset ← `--config` JSON ← explicit flags, with the
//! Python CLI's checks and messages. Every failure is a config error (exit 2), except JSON
//! nested beyond the recursion limit, which Python reported as a plain error.

use std::io::Read;
use std::path::Path;

use super::json::{self, Json, JsonError};
use super::{warn, CliError};
use crate::config::{Config, Value, PRESETS};
use crate::io::{open_regular, IoError};

/// `MAX_CONFIG_FILE_BYTES`: the largest `--config` file read.
pub const MAX_CONFIG_FILE_BYTES: u64 = 1024 * 1024;

/// `_read_config_text`: a regular file of at most [`MAX_CONFIG_FILE_BYTES`], decoded as
/// UTF-8.
pub fn read_config_text(path: &Path) -> Result<String, CliError> {
    let shown = path.display();
    let mut file = open_regular(path).map_err(|e| match e {
        IoError::Pixelforge(e) => CliError::config(e.0),
        e => CliError::config(e.to_string()),
    })?;
    let mut data = Vec::new();
    file.by_ref()
        .take(MAX_CONFIG_FILE_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|e| CliError::config(crate::io::os_error_message(&e, &shown.to_string())))?;
    if data.len() as u64 > MAX_CONFIG_FILE_BYTES {
        return Err(CliError::config(format!(
            "{shown} is larger than {MAX_CONFIG_FILE_BYTES} bytes"
        )));
    }
    String::from_utf8(data)
        .map_err(|_| CliError::config(format!("{shown} is not a UTF-8 text file")))
}

/// `build_config(config_path, flags)`: `flags` are the fields given on the command line
/// (in field order); the JSON object in `config_path` supplies the others it names.
pub fn build_config(
    config_path: Option<&Path>,
    flags: &[(String, Value)],
) -> Result<Config, CliError> {
    let mut values: Vec<(String, Value)> = Vec::new();
    if let Some(path) = config_path {
        let text = read_config_text(path)?;
        let loaded = json::loads(&text).map_err(|e| match e {
            JsonError::Value(m) => CliError::config(m),
            JsonError::Recursion(m) => CliError::runtime(m),
        })?;
        let Json::Object(pairs) = loaded else {
            return Err(CliError::config(format!(
                "{} must contain a JSON object of Config fields",
                path.display()
            )));
        };
        values.extend(pairs.iter().map(|(k, v)| (k.clone(), v.to_value())));
    }
    let preset = flags.iter().find(|(name, _)| name == "preset");
    if let (Some((_, Value::Str(preset))), Some(path)) = (preset, config_path) {
        let preset_fields = PRESETS
            .iter()
            .find(|(name, _)| name == preset)
            .map(|(_, values)| *values)
            .unwrap_or_default();
        let mut shadowed: Vec<&str> = values
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| preset_fields.iter().any(|(f, _)| f == k))
            .filter(|k| !flags.iter().any(|(f, _)| f == k))
            .collect();
        shadowed.sort();
        shadowed.dedup();
        if !shadowed.is_empty() {
            warn(&format!(
                "warning: --preset {preset} does not override {} from {}; the JSON values win",
                shadowed.join(", "),
                path.display()
            ));
        }
    }
    values.extend(flags.iter().cloned());
    let config = Config::new(&values)?;
    config.validate_palette()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("pf-config-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn config_error(path: &Path) -> String {
        match build_config(Some(path), &[]) {
            Err(e) if e.config => e.message,
            Err(e) => panic!("not a config error: {}", e.message),
            Ok(_) => panic!("accepted {}", path.display()),
        }
    }

    #[test]
    fn json_values_reach_the_config() {
        let path = write(
            "ok.json",
            br#"{"method": "box", "palette_size": 4, "scale": 2}"#,
        );
        let flags = vec![("palette_size".to_string(), Value::Int(6))];
        let config = build_config(Some(&path), &flags).unwrap();
        assert_eq!(config.method, "box");
        assert_eq!(config.palette_size, 6);
        assert_eq!(config.scale, Some(2.0));
    }

    #[test]
    fn json_numbers_are_checked_like_python() {
        let big = write("big.json", br#"{"seed": 123456789012345678901234567890}"#);
        assert_eq!(
            config_error(&big),
            "seed must fit in a signed 64-bit integer, got 123456789012345678901234567890"
        );
        let inf = write("inf.json", br#"{"saturation_beta": 1e400}"#);
        assert_eq!(
            config_error(&inf),
            "saturation_beta must be a finite number, got inf"
        );
        let nan = write("nan.json", br#"{"saturation_beta": NaN}"#);
        assert_eq!(config_error(&nan), "config JSON must not contain NaN");
        let list = write("list.json", br#"{"palette_size": [1, [2.0, "x"]]}"#);
        assert_eq!(
            config_error(&list),
            "palette_size must be of type int, got [1, [2.0, 'x']]"
        );
        let object = write("object.json", br#"{"canvas": {"w": 8}}"#);
        assert_eq!(
            config_error(&object),
            "canvas must be of type str | None, got {'w': 8}"
        );
        let float_int = write("float_int.json", br#"{"palette_size": 8.0, "seed": null}"#);
        let config = build_config(Some(&float_int), &[]).unwrap();
        assert_eq!((config.palette_size, config.seed), (8, None));
    }

    #[test]
    fn file_checks() {
        let array = write("array.json", b"[1]");
        assert!(config_error(&array).ends_with("must contain a JSON object of Config fields"));
        let latin1 = write("latin1.json", b"{\"canvas\": \"\xe9\"}");
        assert!(config_error(&latin1).ends_with("is not a UTF-8 text file"));
        let mut huge = b"{\"palette_size\": 4".to_vec();
        huge.resize(huge.len() + MAX_CONFIG_FILE_BYTES as usize, b' ');
        huge.push(b'}');
        let huge = write("huge.json", &huge);
        assert!(config_error(&huge).ends_with("is larger than 1048576 bytes"));
        let missing = huge.with_file_name("missing.json");
        assert!(config_error(&missing).starts_with("[Errno 2] No such file or directory: '"));
        let dir = huge.parent().unwrap();
        assert!(config_error(dir).ends_with("is not a regular file"));
        let bad = write("bad.json", b"{\"a\" 1}");
        assert_eq!(
            config_error(&bad),
            "Expecting ':' delimiter: line 1 column 6 (char 5)"
        );
    }
}
