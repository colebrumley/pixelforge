//! `Config`: construction (defaults ← preset ← explicit values), `_coerce`, `_validate`
//! with Python's messages, `replace`, the palette, canonical JSON, `hash` and `provenance`.
//! Ported from `pixelforge/config.py`.

use super::json::dumps_sorted;
use super::{
    field, FieldKind, FieldSpec, Value, DENOISERS, DITHERS, FIELDS, FITS, METHODS, PALETTE_SOURCES,
    PRESETS, PRESET_NAMES, TRANSPARENT_INDICES, UPPER_BOUND_ORDER,
};
use crate::palette::{self, PaletteError, INLINE_PREFIX, PALETTE_SUFFIXES};
use crate::pyrepr::{float_repr, str_list_repr, str_repr};
use crate::pyunicode;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::OnceLock;

/// The package version, part of every `hash()`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// An invalid configuration (Python's `ConfigError`); the message is Python's, byte for byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

fn err<T>(message: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(message.into()))
}

/// Conversion between a typed Config field and a coerced [`Value`].
trait FieldType: Sized {
    fn from_value(v: Value) -> Self;
    fn to_value(&self) -> Value;
}

impl FieldType for bool {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Bool(b) => b,
            other => unreachable!("coerced bool, got {other:?}"),
        }
    }
    fn to_value(&self) -> Value {
        Value::Bool(*self)
    }
}

impl FieldType for i64 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Int(i) => i,
            other => unreachable!("coerced int, got {other:?}"),
        }
    }
    fn to_value(&self) -> Value {
        Value::Int(*self)
    }
}

impl FieldType for f64 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Float(x) => x,
            other => unreachable!("coerced float, got {other:?}"),
        }
    }
    fn to_value(&self) -> Value {
        Value::Float(*self)
    }
}

impl FieldType for String {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Str(s) => s,
            other => unreachable!("coerced str, got {other:?}"),
        }
    }
    fn to_value(&self) -> Value {
        Value::Str(self.clone())
    }
}

impl<T: FieldType> FieldType for Option<T> {
    fn from_value(v: Value) -> Self {
        match v {
            Value::None => None,
            v => Some(T::from_value(v)),
        }
    }
    fn to_value(&self) -> Value {
        match self {
            None => Value::None,
            Some(t) => t.to_value(),
        }
    }
}

macro_rules! config_struct {
    ($($name:ident: $ty:ty,)*) => {
        /// A validated pixelforge configuration: one typed field per entry of
        /// [`FIELDS`](super::FIELDS), in the same order.
        #[allow(non_snake_case)]
        #[derive(Clone, Debug)]
        pub struct Config {
            $(pub $name: $ty,)*
            /// The field names the caller gave explicitly (`_explicit`); `replace` re-applies
            /// a changed preset to every other field. Not part of equality or hashing.
            explicit: BTreeSet<String>,
            /// `palette_hex_lines()`, once resolved.
            palette_lines: OnceLock<Vec<String>>,
        }

        /// The struct's field names in declaration order (tested equal to `FIELDS`).
        #[cfg(test)]
        const STRUCT_FIELDS: &[&str] = &[$(stringify!($name)),*];

        impl Config {
            /// Build from coerced values, one per field.
            fn from_coerced(mut values: BTreeMap<&'static str, Value>, explicit: BTreeSet<String>) -> Config {
                Config {
                    $($name: FieldType::from_value(
                        values.remove(stringify!($name)).expect("every field has a value"),
                    ),)*
                    explicit,
                    palette_lines: OnceLock::new(),
                }
            }

            /// The value of the field `name`, or `None` for an unknown name.
            pub fn get(&self, name: &str) -> Option<Value> {
                match name {
                    $(stringify!($name) => Some(self.$name.to_value()),)*
                    _ => None,
                }
            }
        }

        /// Field values only, as Python's dataclass equality (`-0.0 == 0.0`).
        impl PartialEq for Config {
            fn eq(&self, other: &Config) -> bool {
                true $(&& self.$name == other.$name)*
            }
        }
    };
}

config_struct! {
    preset: String,
    out_width: Option<i64>,
    out_height: Option<i64>,
    scale: Option<f64>,
    canvas: Option<String>,
    remove_bg: bool,
    key_bg: bool,
    key_bg_tolerance: f64,
    key_bg_fringe: i64,
    crop_to_alpha: bool,
    alpha_threshold: i64,
    denoise: String,
    denoise_sigma_color: f64,
    denoise_sigma_spatial: f64,
    prereduce_max_ratio: i64,
    enhance: f64,
    enhance_black: Option<f64>,
    enhance_white: Option<f64>,
    enhance_radius: f64,
    ink: f64,
    method: String,
    kopf_max_iters: i64,
    kopf_tol: f64,
    palette_size: i64,
    palette_name: Option<String>,
    palette_source: String,
    palette_ramps: bool,
    saturation_beta: f64,
    g_m: f64,
    g_alpha: f64,
    g_T_final: f64,
    g_eps_palette: f64,
    g_eps_cluster: f64,
    g_laplacian: f64,
    g_bilateral_sigma_color: f64,
    g_bilateral_sigma_spatial: f64,
    g_max_iters: i64,
    dither: String,
    dither_strength: f64,
    dither_variance_threshold: f64,
    remove_orphans: bool,
    orphan_min_region: i64,
    orphan_max_delta: f64,
    fix_jaggies: bool,
    outline: String,
    outline_darken: f64,
    tile_size: i64,
    tileset: bool,
    fit: String,
    tile_dedupe_tolerance: f64,
    seamless: bool,
    tileset_columns: i64,
    transparent_index: String,
    seed: Option<i64>,
    scale_preview: i64,
}

/// `Config.provenance()`: everything needed to reproduce a run besides the input image.
#[derive(Clone, Debug, PartialEq)]
pub struct Provenance {
    pub config: Vec<(&'static str, Value)>,
    pub palette_sha256: Option<String>,
    pub palette: Option<Vec<String>>,
    pub version: &'static str,
}

/// `-2**63 <= x <= 2**63 - 1` for an integral float.
fn float_fits_i64(x: f64) -> bool {
    (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&x)
}

/// `_coerce`: check `value` against the field's type and convert it, so equal configs hash
/// equally. Bools are not ints; an int must be exact (an integral float is accepted) and fit
/// in 64 bits; floats must be finite (an int or a numeric string is converted, as
/// `float(value)` does); `outline` is lowercased; `None` only for optional fields.
fn coerce(spec: &FieldSpec, value: Value) -> Result<Value, ConfigError> {
    let name = spec.name;
    if value == Value::None {
        if spec.optional {
            return Ok(Value::None);
        }
        return err(format!("{name} must not be None"));
    }
    let type_error = |v: &Value| {
        ConfigError(format!(
            "{name} must be of type {}, got {}",
            spec.python_type(),
            v.repr()
        ))
    };
    let range_error = |v: &Value| {
        ConfigError(format!(
            "{name} must fit in a signed 64-bit integer, got {}",
            v.repr()
        ))
    };
    let finite_error =
        |v: &Value| ConfigError(format!("{name} must be a finite number, got {}", v.repr()));
    match spec.kind {
        FieldKind::Bool => match value {
            Value::Bool(_) => Ok(value),
            v => Err(type_error(&v)),
        },
        FieldKind::Int => match value {
            Value::Int(_) => Ok(value),
            Value::BigInt(ref s) => match s.parse::<i64>() {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => Err(range_error(&value)),
            },
            // int(nan) is a ValueError and int(inf) an OverflowError: both type errors.
            Value::Float(x) if !x.is_finite() || x.fract() != 0.0 => Err(type_error(&value)),
            Value::Float(x) if float_fits_i64(x) => Ok(Value::Int(x as i64)),
            Value::Float(_) => Err(range_error(&value)),
            v => Err(type_error(&v)),
        },
        FieldKind::Float => {
            let x = match value {
                Value::Int(i) => i as f64,
                // float(huge int) raises OverflowError: a type error.
                Value::BigInt(ref s) => match s.parse::<f64>() {
                    Ok(x) if x.is_finite() => x,
                    _ => return Err(type_error(&value)),
                },
                Value::Float(x) => x,
                Value::Str(ref s) => match pyunicode::parse_float(s) {
                    Some(x) => x,
                    None => return Err(type_error(&value)),
                },
                v => return Err(type_error(&v)),
            };
            if x.is_finite() {
                Ok(Value::Float(x))
            } else {
                Err(finite_error(&value))
            }
        }
        FieldKind::Str => match value {
            // "#AABBCC" and "#aabbcc" are the same outline and must hash equally.
            Value::Str(s) if name == "outline" => Ok(Value::Str(s.to_lowercase())),
            Value::Str(_) => Ok(value),
            v => Err(type_error(&v)),
        },
    }
}

/// `WxH` with ASCII digits (`([0-9]+)x([0-9]+)`, full match), or `None`.
fn parse_canvas(text: &str) -> Option<(u64, u64)> {
    let (w, h) = text.split_once('x')?;
    let num = |s: &str| -> Option<u64> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        // Only the comparison with [8, 4096] matters; saturate longer numbers.
        Some(s.parse::<u64>().unwrap_or(u64::MAX))
    };
    Some((num(w)?, num(h)?))
}

fn choice(name: &str, value: &str, allowed: &[&str]) -> Result<(), ConfigError> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        err(format!(
            "{name} must be one of {}, got {}",
            str_list_repr(allowed),
            str_repr(value)
        ))
    }
}

/// `str()` of a number field value as Python prints it in a message.
fn num(v: &Value) -> String {
    v.repr()
}

impl Config {
    /// The field names in declaration order.
    pub fn field_names() -> Vec<&'static str> {
        FIELDS.iter().map(|f| f.name).collect()
    }

    /// `Config(**kwargs)`: field defaults ← preset ← `kwargs`, coerced and validated. Unknown
    /// names are rejected; a repeated name takes its last value.
    pub fn new(kwargs: &[(String, Value)]) -> Result<Config, ConfigError> {
        let mut given: BTreeMap<&str, &Value> = BTreeMap::new();
        for (name, value) in kwargs {
            given.insert(name.as_str(), value);
        }
        let unknown: Vec<&str> = given
            .keys()
            .copied()
            .filter(|n| field(n).is_none())
            .collect();
        if !unknown.is_empty() {
            return err(format!("unknown config field(s): {}", unknown.join(", ")));
        }
        let default_preset = Value::from(field("preset").expect("preset field").default);
        let preset_value = given.get("preset").copied().unwrap_or(&default_preset);
        let preset = match preset_value {
            Value::Str(s) if PRESET_NAMES.contains(&s.as_str()) => s.as_str(),
            other => {
                let mut names = PRESET_NAMES.to_vec();
                names.sort();
                return err(format!(
                    "preset must be one of {}, got {}",
                    str_list_repr(&names),
                    other.repr()
                ));
            }
        };
        let overrides = PRESETS
            .iter()
            .find(|(n, _)| *n == preset)
            .map(|(_, values)| *values)
            .expect("preset name checked");
        let mut values = BTreeMap::new();
        for spec in FIELDS {
            let value = match given.get(spec.name) {
                Some(v) => (*v).clone(),
                None => match overrides.iter().find(|(k, _)| *k == spec.name) {
                    Some((_, v)) => Value::from(*v),
                    None => Value::from(spec.default),
                },
            };
            values.insert(spec.name, coerce(spec, value)?);
        }
        let explicit = given.keys().map(|k| k.to_string()).collect();
        let config = Config::from_coerced(values, explicit);
        config.validate()?;
        Ok(config)
    }

    /// The field names given explicitly when this config was built.
    pub fn explicit(&self) -> &BTreeSet<String> {
        &self.explicit
    }

    /// A copy with `changes` applied: `Config(**{explicit fields}, **changes)`, so a changed
    /// preset is re-applied to every field the caller never set.
    pub fn replace(&self, changes: &[(String, Value)]) -> Result<Config, ConfigError> {
        let mut values: Vec<(String, Value)> = self
            .explicit
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    self.get(name).expect("explicit names are fields"),
                )
            })
            .collect();
        values.extend(changes.iter().cloned());
        Config::new(&values)
    }

    /// Every field and its value, in field order (`asdict`).
    pub fn to_dict(&self) -> Vec<(&'static str, Value)> {
        FIELDS
            .iter()
            .map(|f| (f.name, self.get(f.name).expect("struct has every field")))
            .collect()
    }

    /// `json.dumps(to_dict(), sort_keys=True, separators=(",", ":"))`, byte for byte.
    pub fn canonical_json(&self) -> String {
        dumps_sorted(&self.to_dict())
    }

    /// `(width, height)` of `canvas`, or `None`.
    pub fn canvas_size(&self) -> Option<(i64, i64)> {
        let (w, h) = parse_canvas(self.canvas.as_deref()?)?;
        Some((w as i64, h as i64))
    }

    /// The resolved `palette_name` colours as `rrggbb` lines (read once, then cached), or
    /// `None` without a palette.
    pub fn palette_hex_lines(&self) -> Result<Option<Vec<String>>, PaletteError> {
        let Some(name) = &self.palette_name else {
            return Ok(None);
        };
        if let Some(lines) = self.palette_lines.get() {
            return Ok(Some(lines.clone()));
        }
        let lines = palette::parse_hex_lines(name)?;
        Ok(Some(self.palette_lines.get_or_init(|| lines).clone()))
    }

    /// Fill the palette cache with lines the caller already resolved (the Python Config's
    /// cache), so a run uses the same reading of the palette file as the caller's
    /// `hash()`. No effect without `palette_name` or once the cache is filled.
    pub fn seed_palette_hex_lines(&self, lines: Vec<String>) {
        if self.palette_name.is_some() {
            let _ = self.palette_lines.set(lines);
        }
    }

    /// SHA-256 of the resolved palette's newline-joined hex lines, or `None`.
    pub fn palette_sha256(&self) -> Result<Option<String>, PaletteError> {
        Ok(self
            .palette_hex_lines()?
            .map(|lines| format!("{:x}", Sha256::digest(lines.join("\n").as_bytes()))))
    }

    /// SHA-256 (lowercase hex) of the canonical JSON, the version and, with `palette_name`,
    /// the palette's contents, so rewriting a palette file changes the hash.
    pub fn hash(&self) -> Result<String, PaletteError> {
        let mut payload = format!("{}\n{VERSION}", self.canonical_json());
        if let Some(sha) = self.palette_sha256()? {
            payload.push_str("\npalette:");
            payload.push_str(&sha);
        }
        Ok(format!("{:x}", Sha256::digest(payload.as_bytes())))
    }

    /// Everything needed to reproduce a run besides the input image.
    pub fn provenance(&self) -> Result<Provenance, PaletteError> {
        Ok(Provenance {
            config: self.to_dict(),
            palette_sha256: self.palette_sha256()?,
            palette: self.palette_hex_lines()?,
            version: VERSION,
        })
    }

    /// Load `palette_name` (file read and parse); a `ConfigError` with the loader's message
    /// if it is unusable. `new` only checks the name's form, so building a Config never
    /// reads a file.
    pub fn validate_palette(&self) -> Result<(), ConfigError> {
        match &self.palette_name {
            None => Ok(()),
            Some(name) => palette::load_palette(name)
                .map(|_| ())
                .map_err(|e| ConfigError(e.0)),
        }
    }

    /// `_validate`: the checks and messages of `Config._validate`, in its order.
    fn validate(&self) -> Result<(), ConfigError> {
        if let Some(name) = &self.palette_name {
            let bundled = palette::bundled_palettes();
            let lower = name.to_lowercase();
            if !bundled.contains(&name.as_str())
                && !PALETTE_SUFFIXES.iter().any(|s| lower.ends_with(s))
                && !name.starts_with(INLINE_PREFIX)
            {
                return err(format!(
                    "unknown palette {}: not one of {}, not a path to a .hex or .gpl file and \
                     not 'hex:rrggbb,...'",
                    str_repr(name),
                    str_list_repr(&bundled)
                ));
            }
        }
        for (name, v) in [
            ("out_width", self.out_width),
            ("out_height", self.out_height),
        ] {
            if let Some(v) = v {
                if v < 8 {
                    return err(format!("{name} must be >= 8, got {v}"));
                }
            }
        }
        if let Some(scale) = self.scale {
            if scale <= 0.0 {
                return err(format!("scale must be > 0, got {}", float_repr(scale)));
            }
        }
        if let Some(canvas) = &self.canvas {
            let ok = matches!(parse_canvas(canvas),
                Some((w, h)) if (8..=4096).contains(&w) && (8..=4096).contains(&h));
            if !ok {
                return err(format!(
                    "canvas must be \"WxH\" with W, H in [8, 4096], got {}",
                    str_repr(canvas)
                ));
            }
        }
        if (self.scale.is_some() || self.canvas.is_some())
            && (self.out_width.is_some() || self.out_height.is_some())
        {
            return err("scale and canvas cannot be combined with out_width/out_height");
        }
        if !(2..=256).contains(&self.palette_size) {
            return err(format!(
                "palette_size must be in [2, 256], got {}",
                self.palette_size
            ));
        }
        choice("method", &self.method, METHODS)?;
        choice("denoise", &self.denoise, DENOISERS)?;
        choice("dither", &self.dither, DITHERS)?;
        choice("palette_source", &self.palette_source, PALETTE_SOURCES)?;
        choice(
            "transparent_index",
            &self.transparent_index,
            TRANSPARENT_INDICES,
        )?;
        choice("fit", &self.fit, FITS)?;
        if self.outline != "none" && self.outline != "auto" && !is_hex_color(&self.outline) {
            return err(format!(
                "outline must be \"none\", \"auto\" or \"#rrggbb\", got {}",
                str_repr(&self.outline)
            ));
        }
        if !(self.g_alpha > 0.0 && self.g_alpha < 1.0) {
            return err(format!(
                "g_alpha must be in (0, 1), got {}",
                float_repr(self.g_alpha)
            ));
        }
        if !(0..=255).contains(&self.alpha_threshold) {
            return err(format!(
                "alpha_threshold must be in [0, 255], got {}",
                self.alpha_threshold
            ));
        }
        if !(0.0..=1.0).contains(&self.dither_strength) {
            return err(format!(
                "dither_strength must be in [0, 1], got {}",
                float_repr(self.dither_strength)
            ));
        }
        if !(0.0..=1.0).contains(&self.key_bg_tolerance) {
            return err(format!(
                "key_bg_tolerance must be in [0, 1], got {}",
                float_repr(self.key_bg_tolerance)
            ));
        }
        if !(self.orphan_max_delta.is_finite() && self.orphan_max_delta >= 0.0) {
            return err(format!(
                "orphan_max_delta must be finite and >= 0, got {}",
                float_repr(self.orphan_max_delta)
            ));
        }
        if !(0..=8).contains(&self.key_bg_fringe) {
            return err(format!(
                "key_bg_fringe must be in [0, 8], got {}",
                self.key_bg_fringe
            ));
        }
        for (name, v) in [
            ("enhance_black", self.enhance_black),
            ("enhance_white", self.enhance_white),
        ] {
            if let Some(v) = v {
                if !(0.0..=1.0).contains(&v) {
                    return err(format!("{name} must be in [0, 1], got {}", float_repr(v)));
                }
            }
        }
        if let (Some(black), Some(white)) = (self.enhance_black, self.enhance_white) {
            if black >= white {
                return err(format!(
                    "enhance_black ({}) must be below enhance_white ({})",
                    float_repr(black),
                    float_repr(white)
                ));
            }
        }
        if self.prereduce_max_ratio < 0 {
            return err(format!(
                "prereduce_max_ratio must be >= 0, got {}",
                self.prereduce_max_ratio
            ));
        }
        if !(0.0..=1.0).contains(&self.outline_darken) {
            return err(format!(
                "outline_darken must be in [0, 1], got {}",
                float_repr(self.outline_darken)
            ));
        }
        for (name, v) in [
            ("kopf_max_iters", self.kopf_max_iters),
            ("g_max_iters", self.g_max_iters),
            ("tile_size", self.tile_size),
            ("scale_preview", self.scale_preview),
            ("orphan_min_region", self.orphan_min_region),
        ] {
            if v < 1 {
                return err(format!("{name} must be >= 1, got {v}"));
            }
        }
        for (name, v) in [
            ("g_T_final", self.g_T_final),
            ("g_m", self.g_m),
            ("kopf_tol", self.kopf_tol),
            ("saturation_beta", self.saturation_beta),
            ("denoise_sigma_color", self.denoise_sigma_color),
            ("denoise_sigma_spatial", self.denoise_sigma_spatial),
            ("g_bilateral_sigma_color", self.g_bilateral_sigma_color),
            ("g_bilateral_sigma_spatial", self.g_bilateral_sigma_spatial),
            ("enhance_radius", self.enhance_radius),
        ] {
            if v <= 0.0 {
                return err(format!("{name} must be > 0, got {}", float_repr(v)));
            }
        }
        for (name, v) in [
            ("dither_variance_threshold", self.dither_variance_threshold),
            ("tile_dedupe_tolerance", self.tile_dedupe_tolerance),
            ("enhance", self.enhance),
            ("ink", self.ink),
        ] {
            if v < 0.0 {
                return err(format!("{name} must be >= 0, got {}", float_repr(v)));
            }
        }
        if !(0..=256).contains(&self.tileset_columns) {
            return err(format!(
                "tileset_columns must be 0 (automatic) or in [1, 256], got {}",
                self.tileset_columns
            ));
        }
        if let Some(seed) = self.seed {
            if seed < 0 {
                return err(format!("seed must be >= 0, got {seed}"));
            }
        }
        for name in UPPER_BOUND_ORDER {
            let bound = field(name)
                .and_then(|f| f.upper_bound)
                .expect("UPPER_BOUND_ORDER names bounded fields");
            let v = self.get(name).expect("bounded fields exist");
            let above = match v {
                Value::Int(i) => i > bound,
                // Every float field's bound is small enough to be exact as an f64.
                Value::Float(x) => x > bound as f64,
                _ => false,
            };
            if above {
                return err(format!("{name} must be <= {bound}, got {}", num(&v)));
            }
        }
        if self.tileset {
            for (name, v) in [
                ("out_width", self.out_width),
                ("out_height", self.out_height),
            ] {
                if let Some(v) = v {
                    if v % self.tile_size != 0 {
                        return err(format!(
                            "tile_size ({}) must divide {name} ({v}) when tileset=True",
                            self.tile_size
                        ));
                    }
                }
            }
            if let (Some(canvas), Some((w, h))) = (&self.canvas, self.canvas_size()) {
                if w % self.tile_size != 0 || h % self.tile_size != 0 {
                    return err(format!(
                        "tile_size ({}) must divide canvas ({canvas}) when tileset=True",
                        self.tile_size
                    ));
                }
            }
        }
        Ok(())
    }
}

/// `#[0-9a-f]{6}`, the whole string (lowercase only: `outline` is lowercased first).
fn is_hex_color(s: &str) -> bool {
    s.len() == 7
        && s.starts_with('#')
        && s.bytes()
            .skip(1)
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kw(items: &[(&str, Value)]) -> Vec<(String, Value)> {
        items
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn error(items: &[(&str, Value)]) -> String {
        Config::new(&kw(items)).unwrap_err().0
    }

    #[test]
    fn struct_fields_follow_the_table() {
        assert_eq!(STRUCT_FIELDS, Config::field_names().as_slice());
        let c = Config::new(&[]).unwrap();
        for f in FIELDS {
            let v = c.get(f.name).unwrap();
            let kind_ok = matches!(
                (&v, f.kind),
                (Value::Bool(_), FieldKind::Bool)
                    | (Value::Int(_), FieldKind::Int)
                    | (Value::Float(_), FieldKind::Float)
                    | (Value::Str(_), FieldKind::Str)
            ) || (f.optional && v == Value::None);
            assert!(kind_ok, "{}: {v:?}", f.name);
        }
    }

    #[test]
    fn presets_and_explicit_values() {
        let sprite = Config::new(&[]).unwrap();
        assert_eq!(sprite.preset, "sprite");
        assert!(sprite.key_bg);
        assert_eq!(sprite.dither, "none");
        let bg = Config::new(&kw(&[("preset", Value::str("background"))])).unwrap();
        assert_eq!(
            (bg.method.as_str(), bg.palette_size, bg.tileset),
            ("kopf", 32, true)
        );
        let c = Config::new(&kw(&[
            ("preset", Value::str("background")),
            ("palette_size", Value::Int(8)),
        ]))
        .unwrap();
        assert_eq!(c.palette_size, 8);
        let r = c.replace(&kw(&[("preset", Value::str("sprite"))])).unwrap();
        assert_eq!((r.method.as_str(), r.palette_size), ("gerstner", 8));
    }

    #[test]
    fn coercion() {
        let c = Config::new(&kw(&[
            ("palette_size", Value::Float(16.0)),
            ("g_m", Value::Int(9_007_199_254_740_993)),
            ("saturation_beta", Value::str(" 1_2.5e-1 ")),
            ("outline", Value::str("#AbCdEf")),
        ]))
        .unwrap();
        assert_eq!(c.palette_size, 16);
        assert_eq!(c.g_m, 9_007_199_254_740_992.0);
        assert_eq!(c.saturation_beta, 1.25);
        assert_eq!(c.outline, "#abcdef");
        assert_eq!(
            error(&[("palette_size", Value::Float(1e30))]),
            "palette_size must fit in a signed 64-bit integer, got 1e+30"
        );
        assert_eq!(
            error(&[("palette_size", Value::Float(f64::INFINITY))]),
            "palette_size must be of type int, got inf"
        );
        assert_eq!(
            error(&[("g_m", Value::str("nan"))]),
            "g_m must be a finite number, got 'nan'"
        );
        assert_eq!(
            error(&[("g_m", Value::BigInt(format!("1{}", "0".repeat(400))))]),
            format!("g_m must be of type float, got 1{}", "0".repeat(400))
        );
        assert_eq!(
            error(&[("seed", Value::Other("(1,)".to_string()))]),
            "seed must be of type int | None, got (1,)"
        );
        assert_eq!(
            error(&[("preset", Value::List(vec![]))]),
            "preset must be one of ['background', 'sprite'], got []"
        );
    }

    #[test]
    fn equality_ignores_explicit() {
        let a = Config::new(&kw(&[("palette_size", Value::Int(16))])).unwrap();
        let b = Config::new(&[]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.canonical_json(), b.canonical_json());
        assert_ne!(a.explicit(), b.explicit());
    }
}
