//! `json.loads(text, parse_constant=_reject_json_constant)` as the Python CLI reads `--config`.
//!
//! A port of CPython 3.11's C scanner (`Modules/_json.c`) and `json.decoder`, so that the
//! values and every error message are Python's: positions count code points, messages are
//! `"{msg}: line L column C (char P)"`, `NaN` / `Infinity` / `-Infinity` reach the constant
//! hook (which rejects them), integers keep their exact value (`Int`, or `BigInt` outside
//! i64), anything with a fraction or exponent is a correctly rounded float (overflowing to
//! infinity as `float()` does), duplicate keys keep the last value, and nesting deeper than
//! Python's recursion limit allows is a `RecursionError` (exit 1, not a config error).
//!
//! The one difference: a `\uXXXX` escape of a lone surrogate (which a Python `str` can hold
//! and a Rust `String` cannot) becomes U+FFFD.

use crate::config::Value;
use crate::pyrepr::str_repr;

/// Python's default `sys.get_int_max_str_digits()`.
const INT_MAX_STR_DIGITS: usize = 4300;

/// The deepest nesting Python's scanner accepted from the CLI (`python -m pixelforge`: the
/// limit of 1000 frames minus the ones already on the stack); one more is a RecursionError.
pub const MAX_DEPTH: usize = 981;

/// A failure of `json.loads`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JsonError {
    /// A `ValueError` (`JSONDecodeError`, the constant hook, the integer digit limit): a
    /// config error in the CLI.
    Value(String),
    /// `RecursionError`: reported as a plain error (exit 1), as Python's CLI did.
    Recursion(String),
}

/// A parsed JSON value. Objects keep their pairs in first-insertion order with the last
/// value of a repeated key, as a Python dict built from them.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(Value),
    Float(f64),
    Str(String),
    List(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The value as a Config keyword argument: lists stay lists, objects become their
    /// Python `repr` (never accepted by a field, but named in its error).
    pub fn to_value(&self) -> Value {
        match self {
            Json::Null => Value::None,
            Json::Bool(b) => Value::Bool(*b),
            Json::Int(v) => v.clone(),
            Json::Float(x) => Value::Float(*x),
            Json::Str(s) => Value::Str(s.clone()),
            Json::List(items) => Value::List(items.iter().map(Json::to_value).collect()),
            Json::Object(_) => Value::Other(self.repr()),
        }
    }

    /// Python's `repr` of the decoded value.
    pub fn repr(&self) -> String {
        match self {
            Json::Object(pairs) => {
                let parts: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| format!("{}: {}", str_repr(k), v.repr()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            Json::List(items) => {
                let parts: Vec<String> = items.iter().map(Json::repr).collect();
                format!("[{}]", parts.join(", "))
            }
            other => other.to_value().repr(),
        }
    }
}

/// `json.loads(text, parse_constant=reject)` where `reject(name)` raises
/// `ValueError(f"config JSON must not contain {name}")`.
pub fn loads(text: &str) -> Result<Json, JsonError> {
    let s: Vec<char> = text.chars().collect();
    let p = Parser { s: &s, depth: 0 };
    if s.first() == Some(&'\u{feff}') {
        return Err(p.error("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0));
    }
    let start = p.skip_ws(0);
    let mut p = p;
    let (value, end) = match p.scan_once(start) {
        Ok(v) => v,
        Err(Fail::Stop(idx)) => return Err(p.error("Expecting value", idx)),
        Err(Fail::Error(e)) => return Err(e),
    };
    let end = p.skip_ws(end);
    if end != s.len() {
        return Err(p.error("Extra data", end));
    }
    Ok(value)
}

enum Fail {
    /// `StopIteration(idx)`: no value starts at `idx` ("Expecting value" at the top).
    Stop(usize),
    Error(JsonError),
}

impl From<JsonError> for Fail {
    fn from(e: JsonError) -> Fail {
        Fail::Error(e)
    }
}

struct Parser<'a> {
    s: &'a [char],
    depth: usize,
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

impl Parser<'_> {
    /// `JSONDecodeError(msg, doc, pos)` as `str()` shows it.
    fn error(&self, msg: &str, pos: usize) -> JsonError {
        let before = &self.s[..pos.min(self.s.len())];
        let lineno = before.iter().filter(|&&c| c == '\n').count() + 1;
        let colno = match before.iter().rposition(|&c| c == '\n') {
            Some(nl) => pos - nl,
            None => pos + 1,
        };
        JsonError::Value(format!("{msg}: line {lineno} column {colno} (char {pos})"))
    }

    fn skip_ws(&self, mut idx: usize) -> usize {
        while idx < self.s.len() && is_ws(self.s[idx]) {
            idx += 1;
        }
        idx
    }

    fn matches_at(&self, idx: usize, word: &str) -> bool {
        let n = word.chars().count();
        idx + n <= self.s.len() && self.s[idx..idx + n].iter().copied().eq(word.chars())
    }

    fn scan_once(&mut self, idx: usize) -> Result<(Json, usize), Fail> {
        let s = self.s;
        if idx >= s.len() {
            return Err(Fail::Stop(idx));
        }
        match s[idx] {
            '"' => {
                let (text, end) = self.scanstring(idx + 1)?;
                return Ok((Json::Str(text), end));
            }
            '{' => return self.nested(idx, "object"),
            '[' => return self.nested(idx, "array"),
            'n' if self.matches_at(idx, "null") => return Ok((Json::Null, idx + 4)),
            't' if self.matches_at(idx, "true") => return Ok((Json::Bool(true), idx + 4)),
            'f' if self.matches_at(idx, "false") => return Ok((Json::Bool(false), idx + 5)),
            'N' if self.matches_at(idx, "NaN") => return Err(constant("NaN")),
            'I' if self.matches_at(idx, "Infinity") => return Err(constant("Infinity")),
            '-' if self.matches_at(idx, "-Infinity") => return Err(constant("-Infinity")),
            _ => {}
        }
        self.match_number(idx)
    }

    fn nested(&mut self, idx: usize, kind: &str) -> Result<(Json, usize), Fail> {
        if self.depth >= MAX_DEPTH {
            return Err(Fail::Error(JsonError::Recursion(format!(
                "maximum recursion depth exceeded while decoding a JSON {kind} from a unicode \
                 string"
            ))));
        }
        self.depth += 1;
        let result = if kind == "object" {
            self.parse_object(idx + 1)
        } else {
            self.parse_array(idx + 1)
        };
        self.depth -= 1;
        result
    }

    fn match_number(&self, start: usize) -> Result<(Json, usize), Fail> {
        let s = self.s;
        let end_idx = s.len() - 1;
        let digit = |i: usize| s[i].is_ascii_digit();
        let mut idx = start;
        if s[idx] == '-' {
            idx += 1;
            if idx > end_idx {
                return Err(Fail::Stop(start));
            }
        }
        if ('1'..='9').contains(&s[idx]) {
            idx += 1;
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
        } else if s[idx] == '0' {
            idx += 1;
        } else {
            return Err(Fail::Stop(start));
        }
        let mut is_float = false;
        if idx < end_idx && s[idx] == '.' && digit(idx + 1) {
            is_float = true;
            idx += 2;
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
        }
        if idx < end_idx && (s[idx] == 'e' || s[idx] == 'E') {
            let e_start = idx;
            idx += 1;
            if idx < end_idx && (s[idx] == '-' || s[idx] == '+') {
                idx += 1;
            }
            while idx <= end_idx && digit(idx) {
                idx += 1;
            }
            if digit(idx - 1) {
                is_float = true;
            } else {
                idx = e_start;
            }
        }
        let text: String = s[start..idx].iter().collect();
        if is_float {
            // Correctly rounded; out-of-range magnitudes give ±inf / ±0.0, as float() does.
            let x: f64 = text.parse().expect("matched a JSON number");
            return Ok((Json::Float(x), idx));
        }
        let digits = text.trim_start_matches('-');
        if digits.len() > INT_MAX_STR_DIGITS {
            return Err(Fail::Error(JsonError::Value(format!(
                "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string \
                 conversion: value has {} digits; use sys.set_int_max_str_digits() to \
                 increase the limit",
                digits.len()
            ))));
        }
        let value = if text == "-0" {
            Value::Int(0)
        } else {
            Value::int_from_decimal(&text)
        };
        Ok((Json::Int(value), idx))
    }

    /// `scanstring_unicode`: the string whose opening quote is at `end - 1`.
    fn scanstring(&self, end: usize) -> Result<(String, usize), Fail> {
        let s = self.s;
        let len = s.len();
        let begin = end - 1;
        let mut out = String::new();
        let mut end = end;
        // A high surrogate waiting for its low half (an unpaired one becomes U+FFFD).
        loop {
            let mut next = end;
            let mut c = '\0';
            while next < len {
                c = s[next];
                if c == '"' || c == '\\' {
                    break;
                }
                if (c as u32) <= 0x1f {
                    return Err(self.error("Invalid control character at", next).into());
                }
                next += 1;
            }
            if next == len || !(c == '"' || c == '\\') {
                return Err(self.error("Unterminated string starting at", begin).into());
            }
            out.extend(&s[end..next]);
            next += 1;
            if c == '"' {
                end = next;
                break;
            }
            if next == len {
                return Err(self.error("Unterminated string starting at", begin).into());
            }
            let esc = s[next];
            if esc != 'u' {
                end = next + 1;
                let ch = match esc {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return Err(self.error("Invalid \\escape", end - 2).into()),
                };
                out.push(ch);
                continue;
            }
            next += 1;
            end = next + 4;
            if end >= len {
                return Err(self.error("Invalid \\uXXXX escape", next - 1).into());
            }
            let mut u = 0u32;
            while next < end {
                let d = s[next]
                    .to_digit(16)
                    .filter(|_| s[next].is_ascii_hexdigit())
                    .ok_or_else(|| self.error("Invalid \\uXXXX escape", end - 5))?;
                u = (u << 4) | d;
                next += 1;
            }
            if (0xd800..=0xdbff).contains(&u)
                && end + 6 < len
                && s[next] == '\\'
                && s[next + 1] == 'u'
            {
                next += 2;
                end += 6;
                let mut u2 = 0u32;
                while next < end {
                    let d = s[next]
                        .to_digit(16)
                        .filter(|_| s[next].is_ascii_hexdigit())
                        .ok_or_else(|| self.error("Invalid \\uXXXX escape", end - 5))?;
                    u2 = (u2 << 4) | d;
                    next += 1;
                }
                if (0xdc00..=0xdfff).contains(&u2) {
                    u = 0x10000 + (((u - 0xd800) << 10) | (u2 - 0xdc00));
                } else {
                    end -= 6;
                }
            }
            out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
        }
        Ok((out, end))
    }

    fn parse_object(&mut self, start: usize) -> Result<(Json, usize), Fail> {
        let s = self.s;
        let len = s.len();
        let mut pairs: Vec<(String, Json)> = Vec::new();
        let mut idx = self.skip_ws(start);
        if idx >= len || s[idx] != '}' {
            loop {
                if idx >= len || s[idx] != '"' {
                    return Err(self
                        .error("Expecting property name enclosed in double quotes", idx)
                        .into());
                }
                let (key, next) = self.scanstring(idx + 1)?;
                idx = self.skip_ws(next);
                if idx >= len || s[idx] != ':' {
                    return Err(self.error("Expecting ':' delimiter", idx).into());
                }
                idx = self.skip_ws(idx + 1);
                let (value, next) = self.scan_once(idx)?;
                match pairs.iter_mut().find(|(k, _)| *k == key) {
                    Some(pair) => pair.1 = value,
                    None => pairs.push((key, value)),
                }
                idx = self.skip_ws(next);
                if idx < len && s[idx] == '}' {
                    break;
                }
                if idx >= len || s[idx] != ',' {
                    return Err(self.error("Expecting ',' delimiter", idx).into());
                }
                idx = self.skip_ws(idx + 1);
            }
        }
        Ok((Json::Object(pairs), idx + 1))
    }

    fn parse_array(&mut self, start: usize) -> Result<(Json, usize), Fail> {
        let s = self.s;
        let len = s.len();
        let mut items = Vec::new();
        let mut idx = self.skip_ws(start);
        if idx >= len || s[idx] != ']' {
            loop {
                let (value, next) = self.scan_once(idx)?;
                items.push(value);
                idx = self.skip_ws(next);
                if idx < len && s[idx] == ']' {
                    break;
                }
                if idx >= len || s[idx] != ',' {
                    return Err(self.error("Expecting ',' delimiter", idx).into());
                }
                idx = self.skip_ws(idx + 1);
            }
        }
        Ok((Json::List(items), idx + 1))
    }
}

fn constant(name: &str) -> Fail {
    Fail::Error(JsonError::Value(format!(
        "config JSON must not contain {name}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(text: &str) -> Value {
        loads(text).unwrap().to_value()
    }

    fn err(text: &str) -> String {
        match loads(text) {
            Err(JsonError::Value(m)) | Err(JsonError::Recursion(m)) => m,
            Ok(v) => panic!("{text:?} parsed as {v:?}"),
        }
    }

    #[test]
    fn numbers_map_like_python() {
        assert_eq!(value("16"), Value::Int(16));
        assert_eq!(value("-0"), Value::Int(0));
        assert_eq!(value("-9223372036854775808"), Value::Int(i64::MIN));
        assert_eq!(
            value("9223372036854775808"),
            Value::BigInt("9223372036854775808".into())
        );
        assert_eq!(
            value("-123456789012345678901234567890"),
            Value::BigInt("-123456789012345678901234567890".into())
        );
        assert_eq!(value("1.0"), Value::Float(1.0));
        assert_eq!(value("1e2"), Value::Float(100.0));
        assert_eq!(value("1E-2"), Value::Float(0.01));
        assert_eq!(value("0.1"), Value::Float(0.1));
        assert_eq!(value("1e400"), Value::Float(f64::INFINITY));
        assert_eq!(value("-1e400"), Value::Float(f64::NEG_INFINITY));
        assert_eq!(value("1e-400"), Value::Float(0.0));
        // Correctly rounded (halfway case of 2^53 + 1).
        assert_eq!(
            value("9007199254740993.0"),
            Value::Float(9007199254740992.0)
        );
        let digits = "1".repeat(4301);
        assert!(err(&digits).starts_with("Exceeds the limit (4300 digits)"));
        assert_eq!(value(&"9".repeat(4300)), Value::BigInt("9".repeat(4300)));
    }

    #[test]
    fn constants_are_rejected() {
        assert_eq!(err("NaN"), "config JSON must not contain NaN");
        assert_eq!(
            err(r#"{"a": [1, Infinity]}"#),
            "config JSON must not contain Infinity"
        );
        assert_eq!(err("-Infinity"), "config JSON must not contain -Infinity");
        // The constant is met before the later syntax error.
        assert_eq!(err("[NaN, }"), "config JSON must not contain NaN");
    }

    #[test]
    fn nested_values() {
        assert_eq!(
            value(r#"[1, 2.5, "a", null, true, [false]]"#),
            Value::List(vec![
                Value::Int(1),
                Value::Float(2.5),
                Value::str("a"),
                Value::None,
                Value::Bool(true),
                Value::List(vec![Value::Bool(false)]),
            ])
        );
        assert_eq!(
            value(r#"{"b": {"x": 1}, "a": [1.0], "b": {"y": "z"}}"#),
            Value::Other("{'b': {'y': 'z'}, 'a': [1.0]}".into())
        );
        assert_eq!(value("{}"), Value::Other("{}".into()));
        assert_eq!(value(r#""é😀\n""#), Value::str("é😀\n"));
        assert_eq!(value(r#""\ud800x""#), Value::str("\u{fffd}x"));
    }

    #[test]
    fn error_messages_are_pythons() {
        assert_eq!(err(""), "Expecting value: line 1 column 1 (char 0)");
        assert_eq!(
            err("{"),
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        );
        assert_eq!(
            err("{\"a\" 1}"),
            "Expecting ':' delimiter: line 1 column 6 (char 5)"
        );
        assert_eq!(
            err("{\"a\": }"),
            "Expecting value: line 1 column 7 (char 6)"
        );
        assert_eq!(
            err("{\"a\": 1,}"),
            "Expecting property name enclosed in double quotes: line 1 column 9 (char 8)"
        );
        assert_eq!(
            err("[1 2]"),
            "Expecting ',' delimiter: line 1 column 4 (char 3)"
        );
        assert_eq!(err("[1,]"), "Expecting value: line 1 column 4 (char 3)");
        assert_eq!(err("{}\n x"), "Extra data: line 2 column 2 (char 4)");
        assert_eq!(
            err("\"abc"),
            "Unterminated string starting at: line 1 column 1 (char 0)"
        );
        assert_eq!(
            err("\"a\nb\""),
            "Invalid control character at: line 1 column 3 (char 2)"
        );
        assert_eq!(err(r#""\x""#), "Invalid \\escape: line 1 column 2 (char 1)");
        assert_eq!(
            err(r#""\u12g4""#),
            "Invalid \\uXXXX escape: line 1 column 3 (char 2)"
        );
        assert_eq!(
            err("\u{feff}{}"),
            "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)"
        );
        let deep = "[".repeat(MAX_DEPTH + 1);
        assert!(matches!(loads(&deep), Err(JsonError::Recursion(_))));
    }
}
