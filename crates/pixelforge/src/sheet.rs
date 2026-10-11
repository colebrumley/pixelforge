//! The `compare` command's library pieces (docs/rust-migration-plan.md, Section 4.1
//! "compare" and Phase 7), ported from `pixelforge/cli.py`: the side-by-side sheet, the
//! `--prepass` columns, `--sweep` parsing, the per-cell `flags` text, the preview-scale cap
//! and the labelled proof sheet.
//!
//! Everything here reproduces the Python functions exactly except the proof sheet's label
//! text. Python draws it with Pillow's default font; this module draws it with an embedded
//! 6×10 bitmap font (below), so the label sizes, and therefore the sheet's pixels and size,
//! differ from the Python version's. The layout rules (padding, label bars, column widths,
//! row heights, gaps, cell positions) are the same, given the label sizes.
//!
//! RGBA images are `h * w * 4` bytes in C order (row-major, RGBA per pixel), as everywhere
//! in the crate (`io`, `pipeline::Output::image`).

use crate::config::{ConfigError, FieldKind, Value, FIELDS};
use crate::pyrepr::{float_repr, str_repr};
use crate::pyunicode;

/// An RGBA8 image: `h * w * 4` bytes, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    pub h: usize,
    pub w: usize,
    pub data: Vec<u8>,
}

impl RgbaImage {
    /// A fully transparent image.
    pub fn transparent(h: usize, w: usize) -> RgbaImage {
        RgbaImage {
            h,
            w,
            data: vec![0; h * w * 4],
        }
    }

    /// An image from its bytes; panics if `data` is not `h * w * 4` long.
    pub fn new(h: usize, w: usize, data: Vec<u8>) -> RgbaImage {
        assert_eq!(
            data.len(),
            h * w * 4,
            "RgbaImage: data does not match the shape"
        );
        RgbaImage { h, w, data }
    }

    /// The pixel at row `y`, column `x`.
    pub fn pixel(&self, y: usize, x: usize) -> [u8; 4] {
        let i = (y * self.w + x) * 4;
        [
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ]
    }

    fn set(&mut self, y: usize, x: usize, colour: [u8; 4]) {
        let i = (y * self.w + x) * 4;
        self.data[i..i + 4].copy_from_slice(&colour);
    }

    /// `im` copied over this image with its top-left corner at (`x`, `y`), alpha included
    /// (Pillow's `paste` without a mask); the part outside this image is dropped.
    pub fn paste(&mut self, im: &RgbaImage, x: usize, y: usize) {
        if x >= self.w || y >= self.h {
            return;
        }
        let cols = im.w.min(self.w - x);
        for r in 0..im.h.min(self.h - y) {
            let src = r * im.w * 4;
            let dst = ((y + r) * self.w + x) * 4;
            self.data[dst..dst + cols * 4].copy_from_slice(&im.data[src..src + cols * 4]);
        }
    }

    /// Fill the rectangle of rows `y0..y1` and columns `x0..x1` (clipped to the image).
    pub fn fill(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, colour: [u8; 4]) {
        for y in y0..y1.min(self.h) {
            for x in x0..x1.min(self.w) {
                self.set(y, x, colour);
            }
        }
    }
}

/// `cli.side_by_side`: the images in a row on a transparent canvas, top-aligned, `gap`
/// pixels apart. No images give a 0×0 image (Python raises on an empty list).
pub fn side_by_side(images: &[RgbaImage], gap: usize) -> RgbaImage {
    if images.is_empty() {
        return RgbaImage::transparent(0, 0);
    }
    let height = images.iter().map(|im| im.h).max().unwrap_or(0);
    let width = images.iter().map(|im| im.w).sum::<usize>() + gap * (images.len() - 1);
    let mut sheet = RgbaImage::transparent(height, width);
    let mut x = 0;
    for im in images {
        sheet.paste(im, x, 0);
        x += im.w + gap;
    }
    sheet
}

/// `compare --prepass`: the columns of the proof sheet, from no pre-pass to the strongest,
/// as `(enhance, ink)` overrides.
pub const PREPASS_COLUMNS: [[(&str, f64); 2]; 4] = [
    [("enhance", 0.0), ("ink", 0.0)],
    [("enhance", 0.5), ("ink", 0.0)],
    [("enhance", 1.0), ("ink", 0.0)],
    [("enhance", 1.0), ("ink", 0.6)],
];

/// [`PREPASS_COLUMNS`] as Config overrides, in the form [`parse_sweep`] returns.
pub fn prepass_columns() -> Vec<Vec<(String, Value)>> {
    PREPASS_COLUMNS
        .iter()
        .map(|col| {
            col.iter()
                .map(|&(name, v)| (name.to_string(), Value::Float(v)))
                .collect()
        })
        .collect()
}

/// The most values one `--sweep` may give.
pub const MAX_SWEEP_VALUES: usize = 12;
/// The words `--sweep` accepts for a boolean field (compared lower-cased).
pub const TRUE_WORDS: [&str; 4] = ["on", "true", "yes", "1"];
pub const FALSE_WORDS: [&str; 4] = ["off", "false", "no", "0"];
/// The proof sheet's label bars, label text and the padding around a label (also the gap
/// between cells).
pub const LABEL_BG: [u8; 4] = [40, 40, 40, 255];
pub const LABEL_FG: [u8; 4] = [255, 255, 255, 255];
pub const LABEL_PAD: usize = 6;
/// Pixels of cell previews on one proof sheet (RGBA, so 4 bytes each, held twice while the
/// sheet is assembled); the preview scale is lowered to stay within it.
pub const PROOF_SHEET_MAX_PIXELS: u64 = 32_000_000;

/// CPython's default limit on the digits `int(str)` converts (`sys.get_int_max_str_digits`).
const INT_MAX_STR_DIGITS: usize = 4300;

/// `int(s)` for a `str` in base 10: `None` where Python raises ValueError.
///
/// Like CPython: Unicode whitespace stripped, an optional sign, decimal digits of any script,
/// single underscores only between two digits, at most 4300 digits (leading zeros count).
fn parse_int(s: &str) -> Option<Value> {
    let t = pyunicode::strip(s);
    let (negative, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let mut digits = String::with_capacity(body.len());
    let mut prev_digit = false;
    let mut prev_underscore = false;
    for c in body.chars() {
        if c == '_' {
            if !prev_digit {
                return None;
            }
            prev_digit = false;
            prev_underscore = true;
        } else {
            let d = pyunicode::decimal_value(c)?;
            digits.push(char::from_digit(d, 10)?);
            prev_digit = true;
            prev_underscore = false;
        }
    }
    if digits.is_empty() || prev_underscore || digits.len() > INT_MAX_STR_DIGITS {
        return None;
    }
    let significant = digits.trim_start_matches('0');
    let text = match (significant.is_empty(), negative) {
        (true, _) => "0".to_string(),
        (false, true) => format!("-{significant}"),
        (false, false) => significant.to_string(),
    };
    Some(Value::int_from_decimal(&text))
}

/// `cli.parse_sweep`: `"palette-size=8,12,16"` → one Config override per value
/// (`[[("palette_size", Int(8))], ...]`), with Python's messages.
///
/// The field is named as its flag or as the Config field (case-insensitive); `none` unsets an
/// optional field and booleans take on/off (also true/false, yes/no, 1/0). Values are split
/// on commas, so fields whose values contain one (inline palettes) cannot be swept.
pub fn parse_sweep(text: &str) -> Result<Vec<Vec<(String, Value)>>, ConfigError> {
    let (name, values) = match text.split_once('=') {
        Some((n, v)) => (n, Some(v)),
        None => (text, None),
    };
    let key = pyunicode::strip(name)
        .trim_start_matches('-')
        .replace('-', "_")
        .to_lowercase();
    let spec = FIELDS.iter().find(|f| f.name.to_lowercase() == key);
    let (spec, values) = match (spec, values) {
        (Some(spec), Some(values)) => (spec, values),
        _ => {
            return Err(ConfigError(format!(
                "--sweep takes FIELD=VALUE[,VALUE...] with a Config field, got {}",
                str_repr(text)
            )))
        }
    };
    if spec.name == "method" {
        return Err(ConfigError(
            "--sweep cannot vary method: the methods are the rows of the sheet".to_string(),
        ));
    }
    let raws: Vec<&str> = values.split(',').map(pyunicode::strip).collect();
    if raws.len() > MAX_SWEEP_VALUES || raws.contains(&"") {
        return Err(ConfigError(format!(
            "--sweep takes 1 to {MAX_SWEEP_VALUES} non-empty values, got {}",
            str_repr(text)
        )));
    }
    let base = spec.kind.python_name();
    let mut columns = Vec::with_capacity(raws.len());
    for raw in raws {
        let lower = raw.to_lowercase();
        let value = if lower == "none" && spec.optional {
            Value::None
        } else {
            match spec.kind {
                FieldKind::Bool => {
                    if TRUE_WORDS.contains(&lower.as_str()) {
                        Value::Bool(true)
                    } else if FALSE_WORDS.contains(&lower.as_str()) {
                        Value::Bool(false)
                    } else {
                        return Err(ConfigError(format!(
                            "--sweep {} takes on/off values, got {}",
                            spec.name,
                            str_repr(raw)
                        )));
                    }
                }
                FieldKind::Int | FieldKind::Float => {
                    let parsed = if spec.kind == FieldKind::Int {
                        parse_int(raw)
                    } else {
                        pyunicode::parse_float(raw).map(Value::Float)
                    };
                    parsed.ok_or_else(|| {
                        ConfigError(format!(
                            "--sweep {}: {} is not {base}",
                            spec.name,
                            str_repr(raw)
                        ))
                    })?
                }
                FieldKind::Str => Value::Str(raw.to_string()),
            }
        };
        columns.push(vec![(spec.name.to_string(), value)]);
    }
    Ok(columns)
}

/// Python's `f"{x:g}"`: 6 significant digits (rounded half to even on the exact binary
/// value), trailing zeros and a bare point removed, exponent form (`1e-05`, `1.5e+16`) when
/// the rounded decimal exponent is below -4 or at least 6; `inf`, `-inf`, `nan`; `-0`.
pub fn format_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    const PRECISION: i32 = 6;
    let sci = format!("{:.*e}", (PRECISION - 1) as usize, x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let strip = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    let mut out = String::new();
    if x.is_sign_negative() {
        out.push('-');
    }
    if (-4..PRECISION).contains(&exp) {
        let body = if exp >= 0 {
            let int_len = exp as usize + 1;
            format!("{}.{}", &digits[..int_len], &digits[int_len..])
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        out.push_str(&strip(body));
    } else {
        out.push_str(&strip(format!("{}.{}", &digits[..1], &digits[1..])));
        let sign = if exp < 0 { '-' } else { '+' };
        out.push_str(&format!("e{sign}{:02}", exp.abs()));
    }
    out
}

/// `cli._show`: a value as the sheet labels and the `flags` text spell it. Booleans are
/// `on`/`off`, None is `none`; a float is its `%g` form when that reads back as the same
/// float (`1`, `0.6`), else its `repr` (every digit), so a cell is reproducible from its
/// text; anything else is `str(value)`.
pub fn show(value: &Value) -> String {
    match value {
        Value::Bool(b) => if *b { "on" } else { "off" }.to_string(),
        Value::None => "none".to_string(),
        Value::Float(x) => {
            let short = format_g(*x);
            if short.parse::<f64>() == Ok(*x) {
                short
            } else {
                float_repr(*x)
            }
        }
        Value::Int(i) => i.to_string(),
        Value::BigInt(s) => s.clone(),
        Value::Str(s) => s.clone(),
        Value::List(_) | Value::Other(_) => value.repr(),
    }
}

/// `shlex.quote(s)`: `s` unchanged if it is non-empty and only has the characters
/// `[A-Za-z0-9@%+=:,./_-]`, else in single quotes with each `'` written as `'"'"'`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = |c: char| c.is_ascii_alphanumeric() || "@%+=:,./_-".contains(c);
    if s.chars().all(safe) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// `cli.override_flags`: the `convert` flags that apply `overrides`, shell-quoted and
/// space-separated. A boolean is `--flag` or `--no-flag`; None ("not set") has no flag, so a
/// value it unsets in the base config stays set (the cell's full config reproduces it).
pub fn override_flags(overrides: &[(String, Value)]) -> String {
    let mut flags = Vec::with_capacity(overrides.len());
    for (name, value) in overrides {
        let flag = format!("--{}", name.replace('_', "-").to_lowercase());
        match value {
            Value::Bool(true) => flags.push(flag),
            Value::Bool(false) => flags.push(format!("--no-{}", &flag[2..])),
            Value::None => {}
            other => flags.push(format!("{flag} {}", shlex_quote(&show(other)))),
        }
    }
    flags.join(" ")
}

/// `math.isqrt(n)`: the largest integer whose square is at most `n`.
fn isqrt(n: u64) -> u64 {
    // f64::sqrt is IEEE-exact; the correction loops fix the rounding of `n` to f64.
    let mut r = (n as f64).sqrt() as u64;
    while r.checked_mul(r).is_none_or(|sq| sq > n) {
        r -= 1;
    }
    while (r + 1).checked_mul(r + 1).is_some_and(|sq| sq <= n) {
        r += 1;
    }
    r
}

/// `cli.preview_scale`: `scales` lowered by one common cap so the previews of images of
/// `sizes` (`(h, w)`, native resolution) fit [`PROOF_SHEET_MAX_PIXELS`]; with the warning
/// Python prints (without the `pixelforge: ` prefix) when the cap lowers the largest scale.
pub fn preview_scale(sizes: &[(usize, usize)], scales: &[i64]) -> (Vec<i64>, Option<String>) {
    let native: u128 = sizes.iter().map(|&(h, w)| h as u128 * w as u128).sum();
    let quotient = u128::from(PROOF_SHEET_MAX_PIXELS) / native.max(1);
    let cap = isqrt(quotient as u64).max(1) as i64;
    let warning = scales.iter().max().filter(|&&m| cap < m).map(|m| {
        format!(
            "proof sheet previews scaled {cap}x instead of {m}x to stay within \
             {PROOF_SHEET_MAX_PIXELS} pixels"
        )
    });
    (scales.iter().map(|&s| s.min(cap)).collect(), warning)
}

/// The width of a glyph cell of the label font and its height. Row 0 of a cell is blank,
/// rows 1–7 hold capitals and digits (baseline on row 7), rows 8–9 descenders; column 5 is
/// blank (the space between letters).
pub const GLYPH_ADVANCE: usize = 6;
pub const GLYPH_HEIGHT: usize = 10;
/// The cell row of a glyph's first bitmap row.
const GLYPH_TOP: usize = 1;

/// The label font: printable ASCII (32–126), 5 columns × 9 rows per glyph (cell rows 1–9),
/// top row first, bit 4 the leftmost column. An original design for pixelforge in the
/// manner of the common 5×7 character-LCD fonts, dedicated to the public domain (CC0).
/// Anything outside printable ASCII is drawn as `?`.
#[rustfmt::skip]
const GLYPHS: [[u8; 9]; 95] = [
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // ' '
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100, 0b00000, 0b00000], // '!'
    [0b01010, 0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '"'
    [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010, 0b00000, 0b00000], // '#'
    [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100, 0b00000, 0b00000], // '$'
    [0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011, 0b00000, 0b00000], // '%'
    [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101, 0b00000, 0b00000], // '&'
    [0b00100, 0b00100, 0b01000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '\''
    [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00000, 0b00000], // '('
    [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000, 0b00000, 0b00000], // ')'
    [0b00000, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0b00000, 0b00000, 0b00000], // '*'
    [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000, 0b00000, 0b00000], // '+'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100, 0b00100, 0b01000], // ','
    [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '-'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100, 0b00000, 0b00000], // '.'
    [0b00000, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000, 0b00000], // '/'
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110, 0b00000, 0b00000], // '0'
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110, 0b00000, 0b00000], // '1'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111, 0b00000, 0b00000], // '2'
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110, 0b00000, 0b00000], // '3'
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010, 0b00000, 0b00000], // '4'
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110, 0b00000, 0b00000], // '5'
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000], // '6'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00000, 0b00000], // '7'
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000], // '8'
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100, 0b00000, 0b00000], // '9'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000, 0b00000, 0b00000], // ':'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00100, 0b01000, 0b00000], // ';'
    [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010, 0b00000, 0b00000], // '<'
    [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000, 0b00000], // '='
    [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000, 0b00000, 0b00000], // '>'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100, 0b00000, 0b00000], // '?'
    [0b01110, 0b10001, 0b00001, 0b01101, 0b10101, 0b10101, 0b01110, 0b00000, 0b00000], // '@'
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0b00000, 0b00000], // 'A'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110, 0b00000, 0b00000], // 'B'
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110, 0b00000, 0b00000], // 'C'
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100, 0b00000, 0b00000], // 'D'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111, 0b00000, 0b00000], // 'E'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b00000, 0b00000], // 'F'
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111, 0b00000, 0b00000], // 'G'
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0b00000, 0b00000], // 'H'
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110, 0b00000, 0b00000], // 'I'
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100, 0b00000, 0b00000], // 'J'
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001, 0b00000, 0b00000], // 'K'
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111, 0b00000, 0b00000], // 'L'
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001, 0b00000, 0b00000], // 'M'
    [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b00000, 0b00000], // 'N'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000], // 'O'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000, 0b00000, 0b00000], // 'P'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101, 0b00000, 0b00000], // 'Q'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001, 0b00000, 0b00000], // 'R'
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110, 0b00000, 0b00000], // 'S'
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00000], // 'T'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000], // 'U'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00000, 0b00000], // 'V'
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010, 0b00000, 0b00000], // 'W'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001, 0b00000, 0b00000], // 'X'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00000], // 'Y'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111, 0b00000, 0b00000], // 'Z'
    [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110, 0b00000, 0b00000], // '['
    [0b00000, 0b10000, 0b01000, 0b00100, 0b00010, 0b00001, 0b00000, 0b00000, 0b00000], // '\\'
    [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110, 0b00000, 0b00000], // ']'
    [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '^'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111, 0b00000], // '_'
    [0b01000, 0b00100, 0b00010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '`'
    [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111, 0b00000, 0b00000], // 'a'
    [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b11110, 0b00000, 0b00000], // 'b'
    [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110, 0b00000, 0b00000], // 'c'
    [0b00001, 0b00001, 0b01101, 0b10011, 0b10001, 0b10001, 0b01111, 0b00000, 0b00000], // 'd'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110, 0b00000, 0b00000], // 'e'
    [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000, 0b00000, 0b00000], // 'f'
    [0b00000, 0b00000, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110], // 'g'
    [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001, 0b00000, 0b00000], // 'h'
    [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110, 0b00000, 0b00000], // 'i'
    [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100], // 'j'
    [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b00000, 0b00000], // 'k'
    [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110, 0b00000, 0b00000], // 'l'
    [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101, 0b00000, 0b00000], // 'm'
    [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001, 0b00000, 0b00000], // 'n'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000], // 'o'
    [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000], // 'p'
    [0b00000, 0b00000, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001], // 'q'
    [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000, 0b00000, 0b00000], // 'r'
    [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110, 0b00000, 0b00000], // 's'
    [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110, 0b00000, 0b00000], // 't'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101, 0b00000, 0b00000], // 'u'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00000, 0b00000], // 'v'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010, 0b00000, 0b00000], // 'w'
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b00000, 0b00000], // 'x'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110], // 'y'
    [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111, 0b00000, 0b00000], // 'z'
    [0b00011, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00011, 0b00000, 0b00000], // '{'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00000], // '|'
    [0b11000, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b11000, 0b00000, 0b00000], // '}'
    [0b00000, 0b00000, 0b01000, 0b10101, 0b00010, 0b00000, 0b00000, 0b00000, 0b00000], // '~'
];

fn glyph(c: char) -> &'static [u8; 9] {
    let code = c as u32;
    let index = if (32..127).contains(&code) {
        code
    } else {
        u32::from(b'?')
    };
    &GLYPHS[(index - 32) as usize]
}

/// The size `(width, height)` of a label box: the text in the label font, one glyph cell
/// per character, plus [`LABEL_PAD`] on every side.
pub fn label_size(text: &str) -> (usize, usize) {
    (
        GLYPH_ADVANCE * text.chars().count() + 2 * LABEL_PAD,
        GLYPH_HEIGHT + 2 * LABEL_PAD,
    )
}

/// Draw `text` in the label font with its first cell's top-left corner at (`x`, `y`);
/// pixels outside the image are dropped.
pub fn draw_text(image: &mut RgbaImage, x: usize, y: usize, text: &str, colour: [u8; 4]) {
    for (i, c) in text.chars().enumerate() {
        let left = x + i * GLYPH_ADVANCE;
        for (r, bits) in glyph(c).iter().enumerate() {
            let py = y + GLYPH_TOP + r;
            for col in 0..5 {
                let px = left + col;
                if bits & (0b10000 >> col) != 0 && py < image.h && px < image.w {
                    image.set(py, px, colour);
                }
            }
        }
    }
}

/// `cli.proof_sheet`: a labelled grid of RGBA images, `cells[row][column]`, transparent
/// between the cells. Column labels sit in a bar along the top, row labels in a bar down the
/// left; each column is as wide as its widest cell or label, each row as tall as its tallest
/// cell or label, with [`LABEL_PAD`] pixels before every column and row.
///
/// The layout is Python's; the label sizes are [`label_size`]'s (Python measures Pillow's
/// default font), so the sheets differ in size and pixels from the Python version's.
///
/// Panics unless there is at least one row and one column, one row label per row and one
/// cell per column label in every row.
pub fn proof_sheet<R: AsRef<str>, C: AsRef<str>>(
    cells: &[Vec<RgbaImage>],
    row_labels: &[R],
    column_labels: &[C],
) -> RgbaImage {
    assert!(
        !row_labels.is_empty() && !column_labels.is_empty(),
        "proof_sheet: no rows or no columns"
    );
    assert_eq!(
        cells.len(),
        row_labels.len(),
        "proof_sheet: one label per row"
    );
    assert!(
        cells.iter().all(|row| row.len() == column_labels.len()),
        "proof_sheet: one cell per column label in every row"
    );
    let gap = LABEL_PAD;
    let left = row_labels
        .iter()
        .map(|l| label_size(l.as_ref()).0)
        .max()
        .unwrap_or(0);
    let top = column_labels
        .iter()
        .map(|l| label_size(l.as_ref()).1)
        .max()
        .unwrap_or(0);
    let widths: Vec<usize> = column_labels
        .iter()
        .enumerate()
        .map(|(c, label)| {
            cells
                .iter()
                .map(|row| row[c].w)
                .fold(label_size(label.as_ref()).0, usize::max)
        })
        .collect();
    let heights: Vec<usize> = cells
        .iter()
        .zip(row_labels)
        .map(|(row, label)| {
            row.iter()
                .map(|im| im.h)
                .fold(label_size(label.as_ref()).1, usize::max)
        })
        .collect();
    let width = left + widths.iter().sum::<usize>() + gap * widths.len();
    let height = top + heights.iter().sum::<usize>() + gap * heights.len();
    let mut sheet = RgbaImage::transparent(height, width);
    sheet.fill(0, 0, width, top, LABEL_BG);
    sheet.fill(0, 0, left, height, LABEL_BG);
    // The first column (row) starts one gap after the label bar, the next one gap after it.
    let offsets = |sizes: &[usize], start: usize| -> Vec<usize> {
        let mut pos = start + gap;
        sizes
            .iter()
            .map(|&s| {
                let p = pos;
                pos += s + gap;
                p
            })
            .collect()
    };
    let xs = offsets(&widths, left);
    let ys = offsets(&heights, top);
    for (&x, label) in xs.iter().zip(column_labels) {
        draw_text(
            &mut sheet,
            x + LABEL_PAD,
            LABEL_PAD,
            label.as_ref(),
            LABEL_FG,
        );
    }
    for ((&y, label), row) in ys.iter().zip(row_labels).zip(cells) {
        draw_text(
            &mut sheet,
            LABEL_PAD,
            y + LABEL_PAD,
            label.as_ref(),
            LABEL_FG,
        );
        for (&x, im) in xs.iter().zip(row) {
            sheet.paste(im, x, y);
        }
    }
    sheet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_format_like_python() {
        let cases: &[(f64, &str)] = &[
            (1.0, "1"),
            (0.6, "0.6"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (1e16, "1e+16"),
            (123456.7, "123457"),
            (1234567.0, "1.23457e+06"),
            (999999.5, "1e+06"),
            (0.30000000000000004, "0.3"),
            (-0.0, "-0"),
            (0.0, "0"),
            (5e-324, "4.94066e-324"),
            // Exact ties at the sixth digit round half to even, as Python does.
            (1234565.0, "1.23456e+06"),
            (1234575.0, "1.23458e+06"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
            (-2.5, "-2.5"),
            (1.5e300, "1.5e+300"),
        ];
        for &(x, want) in cases {
            assert_eq!(format_g(x), want, "{x:?}");
        }
    }

    #[test]
    fn show_like_python() {
        let f = |x: f64| show(&Value::Float(x));
        assert_eq!(f(1.0), "1");
        assert_eq!(f(0.6), "0.6");
        assert_eq!(f(1e-5), "1e-05");
        assert_eq!(f(1e16), "1e+16");
        assert_eq!(f(123456.7), "123456.7");
        assert_eq!(f(1234567.0), "1234567.0");
        assert_eq!(f(0.30000000000000004), "0.30000000000000004");
        assert_eq!(f(-0.0), "-0");
        assert_eq!(f(5e-324), "4.94066e-324"); // reads back as 5e-324
        assert_eq!(f(f64::NAN), "nan");
        assert_eq!(f(f64::INFINITY), "inf");
        assert_eq!(show(&Value::Bool(true)), "on");
        assert_eq!(show(&Value::Bool(false)), "off");
        assert_eq!(show(&Value::None), "none");
        assert_eq!(show(&Value::Int(-3)), "-3");
        assert_eq!(show(&Value::str("a b")), "a b");
    }

    #[test]
    fn shlex_quote_like_python() {
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(
            shlex_quote("abc-1.5_x/y:z,w=v+u@%"),
            "abc-1.5_x/y:z,w=v+u@%"
        );
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
        assert_eq!(shlex_quote("$HOME"), "'$HOME'");
        assert_eq!(shlex_quote("é"), "'é'");
        assert_eq!(shlex_quote("'"), "''\"'\"''");
        assert_eq!(shlex_quote("#ff0000"), "'#ff0000'");
    }

    #[test]
    fn override_flags_text() {
        let overrides = vec![
            ("method".to_string(), Value::str("kopf")),
            ("tileset".to_string(), Value::Bool(false)),
            ("dither_auto".to_string(), Value::Bool(true)),
            ("palette".to_string(), Value::None),
            ("g_T_final".to_string(), Value::Float(0.6)),
            ("outline".to_string(), Value::str("#000000")),
        ];
        assert_eq!(
            override_flags(&overrides),
            "--method kopf --no-tileset --dither-auto --g-t-final 0.6 --outline '#000000'"
        );
    }

    #[test]
    fn sweep_parsing() {
        let cols = parse_sweep("--palette-size=8, 12 ,16").unwrap();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[1], vec![("palette_size".to_string(), Value::Int(12))]);
        assert_eq!(
            parse_sweep("method=box").unwrap_err().0,
            "--sweep cannot vary method: the methods are the rows of the sheet"
        );
        assert_eq!(
            parse_sweep("nope=1").unwrap_err().0,
            "--sweep takes FIELD=VALUE[,VALUE...] with a Config field, got 'nope=1'"
        );
        assert_eq!(
            parse_sweep("palette_size=8,,16").unwrap_err().0,
            "--sweep takes 1 to 12 non-empty values, got 'palette_size=8,,16'"
        );
        assert_eq!(
            parse_sweep("palette_size=8.5").unwrap_err().0,
            "--sweep palette_size: '8.5' is not int"
        );
        assert_eq!(parse_int("1_000"), Some(Value::Int(1000)));
        assert_eq!(parse_int("-007"), Some(Value::Int(-7)));
        assert_eq!(parse_int("1__0"), None);
        assert_eq!(parse_int(&"1".repeat(4301)), None);
    }

    #[test]
    fn isqrt_exact() {
        for n in [0u64, 1, 2, 3, 4, 15, 16, 17, 99, 100, 32_000_000, u64::MAX] {
            let r = isqrt(n);
            assert!(
                r * r <= n && (r + 1).checked_mul(r + 1).is_none_or(|s| s > n),
                "{n}"
            );
        }
    }

    #[test]
    fn preview_scale_cap() {
        assert_eq!(preview_scale(&[(64, 64)], &[8]), (vec![8], None));
        let (scales, warning) = preview_scale(&[(1000, 1000), (1000, 1000)], &[8, 2]);
        assert_eq!(scales, vec![4, 2]);
        assert_eq!(
            warning.unwrap(),
            "proof sheet previews scaled 4x instead of 8x to stay within 32000000 pixels"
        );
    }

    #[test]
    fn side_by_side_layout() {
        let a = RgbaImage::new(2, 1, vec![1; 8]);
        let b = RgbaImage::new(1, 2, vec![2; 8]);
        let s = side_by_side(&[a, b], 3);
        assert_eq!((s.h, s.w), (2, 6));
        assert_eq!(s.pixel(1, 0), [1; 4]);
        assert_eq!(s.pixel(0, 1), [0; 4]);
        assert_eq!(s.pixel(0, 4), [2; 4]);
        assert_eq!(s.pixel(1, 4), [0; 4]);
    }

    #[test]
    fn label_text_pixels() {
        let text = "pixelforge 0.1.0";
        let (w, h) = label_size(text);
        assert_eq!((w, h), (6 * 16 + 12, 22));
        let mut im = RgbaImage::transparent(h, w);
        im.fill(0, 0, w, h, LABEL_BG);
        draw_text(&mut im, LABEL_PAD, LABEL_PAD, text, LABEL_FG);
        // (x, y) within the text area.
        let on = |x: usize, y: usize| im.pixel(LABEL_PAD + y, LABEL_PAD + x) == LABEL_FG;
        // 'p': the top of the bowl on row 3 (x 0..=3), the stem down to the descender rows.
        assert!(on(0, 3) && on(3, 3) && !on(4, 3));
        assert!(on(0, 8) && on(0, 9) && !on(1, 9));
        // Row 0 of every cell and column 5 of the first cell are blank.
        assert!((0..w - 2 * LABEL_PAD).all(|x| !on(x, 0)));
        assert!((0..GLYPH_HEIGHT).all(|y| !on(5, y)));
        // 'i' (cell 1): the dot on row 1, a blank row 2, the stem from row 3.
        assert!(on(6 + 2, 1) && !on(6 + 2, 2) && on(6 + 2, 3));
        // ' ' (cell 10) is empty; '0' (cell 11) has its slash through the centre (row 4).
        assert!((0..GLYPH_HEIGHT).all(|y| (0..6).all(|x| !on(60 + x, y))));
        assert!(on(66 + 2, 4) && on(66, 2) && on(66 + 4, 2) && !on(66 + 2, 3));
        // '.' (cell 12): two by two on rows 6-7.
        assert!(on(72 + 1, 6) && on(72 + 2, 7) && !on(72 + 1, 5));
        // The padding stays background.
        assert_eq!(im.pixel(0, 0), LABEL_BG);
        assert_eq!(im.pixel(h - 1, w - 1), LABEL_BG);
        // Non-ASCII text draws '?'.
        let mut q = RgbaImage::transparent(10, 6);
        let mut e = RgbaImage::transparent(10, 6);
        draw_text(&mut q, 0, 0, "?", LABEL_FG);
        draw_text(&mut e, 0, 0, "é", LABEL_FG);
        assert_eq!(q, e);
    }

    #[test]
    fn proof_sheet_layout() {
        let cell = |h, w, v| RgbaImage::new(h, w, vec![v; h * w * 4]);
        let cells = vec![
            vec![cell(30, 40, 1), cell(5, 5, 2)],
            vec![cell(10, 10, 3), cell(50, 20, 4)],
        ];
        let sheet = proof_sheet(&cells, &["box", "gerstner"], &["a", "bb"]);
        let left = label_size("gerstner").0;
        let top = 22;
        let widths = [40, label_size("bb").0.max(20)];
        let heights = [30, 50];
        assert_eq!(sheet.w, left + widths[0] + widths[1] + 2 * LABEL_PAD);
        assert_eq!(sheet.h, top + heights[0] + heights[1] + 2 * LABEL_PAD);
        let x1 = left + LABEL_PAD + widths[0] + LABEL_PAD;
        let y1 = top + LABEL_PAD + heights[0] + LABEL_PAD;
        assert_eq!(sheet.pixel(top + LABEL_PAD, left + LABEL_PAD), [1; 4]);
        assert_eq!(sheet.pixel(y1 + 49, x1 + 19), [4; 4]);
        assert_eq!(sheet.pixel(y1, x1 - 1), [0; 4]);
        assert_eq!(sheet.pixel(0, sheet.w - 1), LABEL_BG);
        assert_eq!(sheet.pixel(sheet.h - 1, 0), LABEL_BG);
        assert_eq!(sheet.pixel(sheet.h - 1, sheet.w - 1), [0; 4]);
    }
}
