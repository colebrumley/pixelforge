//! The command-line options, generated from the Config field table, and the
//! value parsers that reproduce click's conversions: `int()` / `float()` on the text,
//! `click.Path` checks, `IntRange(min=1)`.
//!
//! Every field is a `--kebab-case` option that defaults to "not given" (the preset or the
//! `--config` JSON applies). Booleans are `--flag` / `--no-flag` pairs where the last one
//! given wins; choice fields accept only their choices; everything else is converted as
//! click's `INT`, `FLOAT` and `STRING` types do. A repeated option keeps its last value.

use std::ffi::OsStr;
use std::path::PathBuf;

use clap::builder::{PossibleValuesParser, TypedValueParser, ValueParser};
use clap::error::ErrorKind;
use clap::parser::ValueSource;
use clap::{Arg, ArgAction, ArgMatches, Command};

use crate::config::{FieldKind, FieldSpec, FieldValue, Value, FIELDS, PRESETS};
use crate::io::MAX_INPUT_PIXELS;
use crate::pyrepr::{float_repr, str_repr};
use crate::pyunicode;

/// Old spellings kept working as hidden options: `(flag, field)`.
pub const FLAG_ALIASES: &[(&str, &str)] = &[("g-T-final", "g_T_final")];

/// The clap id of the negative half of a boolean pair.
fn negative_id(name: &str) -> String {
    format!("no_{name}")
}

fn alias_id(name: &str) -> String {
    format!("_alias_{name}")
}

/// `--kebab-case` spelling of a field name (without the dashes).
pub fn flag_name(name: &str) -> String {
    name.replace('_', "-").to_lowercase()
}

/// `_fmt`: how `--help` shows a default.
fn fmt_default(value: FieldValue) -> String {
    match value {
        FieldValue::Bool(true) => "on".into(),
        FieldValue::Bool(false) => "off".into(),
        FieldValue::None => "none".into(),
        FieldValue::Int(i) => i.to_string(),
        FieldValue::Float(x) => float_repr(x),
        FieldValue::Str(s) => s.into(),
    }
}

/// `_help`: the field's help text plus its default, per preset where the presets differ.
pub fn field_help(spec: &FieldSpec) -> String {
    let per_preset: Vec<(&str, FieldValue)> = PRESETS
        .iter()
        .map(|(preset, values)| {
            let value = values
                .iter()
                .find(|(k, _)| *k == spec.name)
                .map_or(spec.default, |(_, v)| *v);
            (*preset, value)
        })
        .collect();
    let reprs: Vec<String> = per_preset
        .iter()
        .map(|(_, v)| Value::from(*v).repr())
        .collect();
    let shown = if reprs.iter().any(|r| *r != reprs[0]) {
        per_preset
            .iter()
            .map(|(preset, v)| format!("{} ({preset})", fmt_default(*v)))
            .collect::<Vec<_>>()
            .join(", ")
    } else if spec.default == FieldValue::None {
        return spec.help.to_string();
    } else {
        fmt_default(spec.default)
    };
    format!("{}  [default: {shown}]", spec.help)
}

// ---------------------------------------------------------------------------------------
// Value parsers

/// `int(text)` for a `str`: surrounding whitespace, a sign, decimal digits of any script and
/// single underscores between digits; the value as `Int`, or `BigInt` beyond i64.
pub fn py_int(text: &str) -> Option<Value> {
    let mut ascii = String::with_capacity(text.len());
    for c in text.chars() {
        if (c as u32) < 128 {
            ascii.push(c);
        } else if pyunicode::is_space(c) {
            ascii.push(' ');
        } else {
            ascii.push(char::from_digit(pyunicode::decimal_value(c)?, 10)?);
        }
    }
    let t = ascii.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'));
    let (negative, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return None;
    }
    if !body.bytes().all(|b| b.is_ascii_digit() || b == b'_') {
        return None;
    }
    let digits = body.replace('_', "");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Some(Value::Int(0));
    }
    let decimal = if negative {
        format!("-{digits}")
    } else {
        digits.to_string()
    };
    Some(Value::int_from_decimal(&decimal))
}

/// click's text for a parameter in "Invalid value for ...".
fn param_name(arg: Option<&Arg>) -> String {
    let Some(arg) = arg else {
        return "value".into();
    };
    match (arg.get_short(), arg.get_long()) {
        (Some(s), Some(l)) => format!("'-{s}' / '--{l}'"),
        (None, Some(l)) => format!("'--{l}'"),
        _ => format!(
            "'{}'",
            arg.get_value_names()
                .and_then(|n| n.first())
                .map_or_else(|| arg.get_id().to_string(), |n| n.to_string())
        ),
    }
}

fn invalid(cmd: &Command, arg: Option<&Arg>, message: String) -> clap::Error {
    let mut err = clap::Error::raw(
        ErrorKind::ValueValidation,
        format!("Invalid value for {}: {message}", param_name(arg)),
    );
    err = err.format(&mut cmd.clone());
    err
}

fn text(value: &OsStr) -> String {
    value.to_string_lossy().into_owned()
}

/// The kinds of conversion click applied to option values.
#[derive(Clone, Copy, Debug)]
enum Conv {
    /// `click.INT`: `int(text)`.
    Int,
    /// `click.FLOAT`: `float(text)`.
    Float,
    /// `click.STRING`.
    Str,
}

#[derive(Clone, Debug)]
struct ClickValue(Conv);

impl TypedValueParser for ClickValue {
    type Value = Value;

    fn parse_ref(
        &self,
        cmd: &Command,
        arg: Option<&Arg>,
        value: &OsStr,
    ) -> Result<Value, clap::Error> {
        let s = value.to_str();
        let bad = |kind: &str| {
            invalid(
                cmd,
                arg,
                format!("{} is not a valid {kind}.", str_repr(&text(value))),
            )
        };
        match self.0 {
            Conv::Int => s.and_then(py_int).ok_or_else(|| bad("integer")),
            Conv::Float => s
                .and_then(pyunicode::parse_float)
                .map(Value::Float)
                .ok_or_else(|| bad("float")),
            Conv::Str => Ok(Value::Str(text(value))),
        }
    }
}

/// `click.IntRange(min=1)` for `--max-input-pixels` (values beyond u64 saturate).
#[derive(Clone, Debug)]
struct PositiveInt;

impl TypedValueParser for PositiveInt {
    type Value = u64;

    fn parse_ref(
        &self,
        cmd: &Command,
        arg: Option<&Arg>,
        value: &OsStr,
    ) -> Result<u64, clap::Error> {
        let parsed = value.to_str().and_then(py_int).ok_or_else(|| {
            invalid(
                cmd,
                arg,
                format!("{} is not a valid integer.", str_repr(&text(value))),
            )
        })?;
        let out_of_range =
            |shown: String| invalid(cmd, arg, format!("{shown} is not in the range x>=1."));
        match parsed {
            Value::Int(i) if i >= 1 => Ok(i as u64),
            Value::Int(i) => Err(out_of_range(i.to_string())),
            Value::BigInt(s) if s.starts_with('-') => Err(out_of_range(s)),
            Value::BigInt(s) => Ok(s.parse::<u64>().unwrap_or(u64::MAX)),
            _ => unreachable!("py_int returns integers"),
        }
    }
}

/// `click.Path(file_okay=..., dir_okay=...)` without `exists`: a path that exists must be
/// of an accepted kind ("File 'x' is a directory." / "Directory 'x' is a file.").
#[derive(Clone, Debug)]
struct ClickPath {
    /// The path may name a directory (click: "Directory" when files are not okay).
    dir_okay: bool,
}

impl TypedValueParser for ClickPath {
    type Value = PathBuf;

    fn parse_ref(
        &self,
        cmd: &Command,
        arg: Option<&Arg>,
        value: &OsStr,
    ) -> Result<PathBuf, clap::Error> {
        let path = PathBuf::from(value);
        if let Ok(meta) = std::fs::metadata(&path) {
            let shown = str_repr(&text(value));
            if self.dir_okay && meta.is_file() {
                return Err(invalid(cmd, arg, format!("Directory {shown} is a file.")));
            }
            if !self.dir_okay && meta.is_dir() {
                return Err(invalid(cmd, arg, format!("File {shown} is a directory.")));
            }
        }
        Ok(path)
    }
}

/// A file argument or option (`click.Path(dir_okay=False)`).
pub fn file_path() -> ValueParser {
    ValueParser::new(ClickPath { dir_okay: false })
}

/// A directory argument or option (`click.Path(file_okay=False)`).
pub fn dir_path() -> ValueParser {
    ValueParser::new(ClickPath { dir_okay: true })
}

// ---------------------------------------------------------------------------------------
// The options

/// A boolean `--flag` (on/off, help hidden for the negative half).
pub fn flag(id: &'static str, long: &'static str, help: &'static str) -> Arg {
    Arg::new(id)
        .long(long)
        .action(ArgAction::SetTrue)
        .help(help)
}

fn field_arg(spec: &FieldSpec) -> Vec<Arg> {
    let long = flag_name(spec.name);
    let help = field_help(spec);
    if spec.kind == FieldKind::Bool {
        let negative = negative_id(spec.name);
        return vec![
            Arg::new(spec.name)
                .long(long.clone())
                .action(ArgAction::SetTrue)
                .overrides_with(negative.clone())
                .help(help),
            Arg::new(negative)
                .long(format!("no-{long}"))
                .action(ArgAction::SetTrue)
                .overrides_with(spec.name)
                .hide(true),
        ];
    }
    let arg = Arg::new(spec.name)
        .long(long)
        .action(ArgAction::Set)
        .num_args(1)
        .allow_hyphen_values(true)
        .help(help);
    let arg = match (spec.choices, spec.kind) {
        (Some(choices), _) => arg
            .value_name(choices.join("|"))
            .hide_possible_values(true)
            .value_parser(PossibleValuesParser::new(choices.iter().copied()).map(Value::Str)),
        (None, FieldKind::Int) => arg
            .value_name("INTEGER")
            .value_parser(ClickValue(Conv::Int)),
        (None, FieldKind::Float) => arg
            .value_name("FLOAT")
            .value_parser(ClickValue(Conv::Float)),
        (None, _) => arg.value_name("TEXT").value_parser(ClickValue(Conv::Str)),
    };
    vec![arg]
}

/// `config_options`: `--config`, `--max-input-pixels`, one option per Config field (in field
/// order) and the hidden aliases.
pub fn config_args(mut cmd: Command) -> Command {
    cmd = cmd
        .arg(
            Arg::new("config")
                .long("config")
                .value_name("FILE")
                .allow_hyphen_values(true)
                .value_parser(file_path())
                .help("JSON file of Config fields; flags override it."),
        )
        .arg(
            Arg::new("max_input_pixels")
                .long("max-input-pixels")
                .value_name("INTEGER")
                .allow_hyphen_values(true)
                .value_parser(ValueParser::new(PositiveInt))
                .help(format!(
                    "Refuse inputs with more pixels than this [default: {MAX_INPUT_PIXELS}]."
                )),
        );
    for spec in FIELDS {
        for arg in field_arg(spec) {
            cmd = cmd.arg(arg);
        }
    }
    for (old, name) in FLAG_ALIASES {
        cmd = cmd.arg(
            Arg::new(alias_id(name))
                .long(*old)
                .num_args(1)
                .allow_hyphen_values(true)
                .value_parser(ClickValue(Conv::Float))
                .hide(true),
        );
    }
    cmd.after_help(
        "Every on/off option also has a --no- form that turns it off (e.g. --no-key-bg); \
         the last one given wins.",
    )
}

/// The options `config_args` added, as given on the command line.
#[derive(Clone, Debug, Default)]
pub struct ConfigArgs {
    pub config_path: Option<PathBuf>,
    pub max_input_pixels: Option<u64>,
    /// The fields given explicitly, in field order (the preset and JSON apply to the rest).
    pub flags: Vec<(String, Value)>,
}

fn given(m: &ArgMatches, id: &str) -> bool {
    m.value_source(id) == Some(ValueSource::CommandLine)
}

/// Read the options `config_args` added.
pub fn config_args_from(m: &ArgMatches) -> ConfigArgs {
    let mut flags = Vec::new();
    for spec in FIELDS {
        let value = if spec.kind == FieldKind::Bool {
            if given(m, spec.name) {
                Some(Value::Bool(true))
            } else if given(m, &negative_id(spec.name)) {
                Some(Value::Bool(false))
            } else {
                None
            }
        } else {
            m.get_one::<Value>(spec.name).cloned()
        };
        // An old spelling applies only when the field's own flag is not given.
        let value = value.or_else(|| {
            FLAG_ALIASES
                .iter()
                .find(|(_, name)| *name == spec.name)
                .and_then(|(_, name)| m.get_one::<Value>(&alias_id(name)).cloned())
        });
        if let Some(v) = value {
            flags.push((spec.name.to_string(), v));
        }
    }
    ConfigArgs {
        config_path: m.get_one::<PathBuf>("config").cloned(),
        max_input_pixels: m.get_one::<u64>("max_input_pixels").copied(),
        flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::field;

    #[test]
    fn help_shows_per_preset_defaults_like_click() {
        let help = |name: &str| field_help(field(name).unwrap());
        assert_eq!(
            help("palette_size"),
            "Number of palette colors K; ignored with --palette-name.  \
             [default: 16 (sprite), 32 (background)]"
        );
        assert_eq!(
            help("key_bg"),
            "Key out a flat opaque background color.  [default: on (sprite), off (background)]"
        );
        assert_eq!(
            help("remove_bg"),
            "Remove the background with the external rembg command (opt-in, needs rembg on PATH).  \
             [default: off]"
        );
        assert_eq!(
            help("kopf_tol"),
            "kopf convergence tolerance (RMS color change in the unit cube).  [default: 0.001]"
        );
        assert_eq!(
            help("denoise_sigma_spatial"),
            "Denoise spatial sigma in input pixels (after pre-reduction).  [default: 2.0]"
        );
        assert_eq!(
            help("seed"),
            "Seed for the optional MCDA jitter (unset = fully analytic)."
        );
        assert_eq!(
            help("preset"),
            "Starting values for the other fields: sprite or background.  [default: sprite]"
        );
    }

    #[test]
    fn ints_parse_like_python() {
        assert_eq!(py_int("16"), Some(Value::Int(16)));
        assert_eq!(py_int(" +1_000 "), Some(Value::Int(1000)));
        assert_eq!(py_int("-007"), Some(Value::Int(-7)));
        assert_eq!(py_int("-0"), Some(Value::Int(0)));
        assert_eq!(py_int("\u{663}\u{664}"), Some(Value::Int(34)));
        assert_eq!(
            py_int("99999999999999999999"),
            Some(Value::BigInt("99999999999999999999".into()))
        );
        for bad in [
            "", "-", "1.0", "1e3", "_1", "1_", "1__0", "0x10", "1 2", "abc", "+-1",
        ] {
            assert_eq!(py_int(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn every_field_has_a_flag() {
        let cmd = config_args(Command::new("t"));
        for spec in FIELDS {
            let arg = cmd
                .get_arguments()
                .find(|a| a.get_id() == spec.name)
                .unwrap_or_else(|| panic!("no flag for {}", spec.name));
            assert_eq!(arg.get_long(), Some(flag_name(spec.name).as_str()));
            assert!(!arg.is_hide_set(), "{}", spec.name);
        }
    }
}
