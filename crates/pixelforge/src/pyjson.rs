//! `json.dumps` for arbitrary values with Python's layouts, and Python's `round(x, n)`.
//!
//! The pipeline writes `_palette.json` (`indent=2`), `_meta.json` (`indent=2, sort_keys`),
//! `_tilemap.json` (`separators=(",", ":")`) and `.tmj` (`indent=1`); these files must be
//! byte-identical to the Python version's. Python's rules: without `indent` the item
//! separator is `", "` and the key separator `": "`; with `indent` the item separator is
//! `","` followed by a newline and `indent` spaces per level, the key separator stays
//! `": "`, and empty containers are written as `[]` / `{}`. Compact mode is
//! `separators=(",", ":")`. Strings use `ensure_ascii` escaping, floats Python's `repr`.

use crate::pyrepr::{float_repr, json_string};

/// A JSON value with ordered object keys (insertion order, as a Python dict).
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn str(s: &str) -> Json {
        Json::Str(s.to_string())
    }
}

/// How `json.dumps` was called.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// `indent=N`; `None` is the default single-line layout.
    pub indent: Option<usize>,
    /// `sort_keys=True` (keys sorted by code point).
    pub sort_keys: bool,
    /// `separators=(",", ":")`; only meaningful without `indent`.
    pub compact: bool,
}

impl Layout {
    pub const COMPACT: Layout = Layout {
        indent: None,
        sort_keys: false,
        compact: true,
    };
    pub const INDENT1: Layout = Layout {
        indent: Some(1),
        sort_keys: false,
        compact: false,
    };
    pub const INDENT2: Layout = Layout {
        indent: Some(2),
        sort_keys: false,
        compact: false,
    };
    pub const INDENT2_SORTED: Layout = Layout {
        indent: Some(2),
        sort_keys: true,
        compact: false,
    };
}

/// `json.dumps(value, **layout)`.
pub fn dumps(value: &Json, layout: Layout) -> String {
    let mut out = String::new();
    write(&mut out, value, layout, 0);
    out
}

fn newline(out: &mut String, indent: usize, level: usize) {
    out.push('\n');
    for _ in 0..indent * level {
        out.push(' ');
    }
}

fn write(out: &mut String, value: &Json, layout: Layout, level: usize) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Float(x) if x.is_nan() => out.push_str("NaN"),
        Json::Float(x) if x.is_infinite() => {
            out.push_str(if *x > 0.0 { "Infinity" } else { "-Infinity" })
        }
        Json::Float(x) => out.push_str(&float_repr(*x)),
        Json::Str(s) => out.push_str(&json_string(s)),
        Json::List(items) => {
            out.push('[');
            if !items.is_empty() {
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    match layout.indent {
                        Some(n) => newline(out, n, level + 1),
                        None if i > 0 && !layout.compact => out.push(' '),
                        None => {}
                    }
                    write(out, item, layout, level + 1);
                }
                if let Some(n) = layout.indent {
                    newline(out, n, level);
                }
            }
            out.push(']');
        }
        Json::Object(items) => {
            let mut entries: Vec<&(String, Json)> = items.iter().collect();
            if layout.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            if !entries.is_empty() {
                for (i, (key, item)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    match layout.indent {
                        Some(n) => newline(out, n, level + 1),
                        None if i > 0 && !layout.compact => out.push(' '),
                        None => {}
                    }
                    out.push_str(&json_string(key));
                    out.push_str(if layout.compact && layout.indent.is_none() {
                        ":"
                    } else {
                        ": "
                    });
                    write(out, item, layout, level + 1);
                }
                if let Some(n) = layout.indent {
                    newline(out, n, level);
                }
            }
            out.push('}');
        }
    }
}

/// Python's `round(x, ndigits)` for a finite `x` and `ndigits >= 0`: the decimal string of
/// `x` correctly rounded to `ndigits` places (half to even on an exact tie), read back as a
/// float. Rust's fixed-precision formatting is exact and rounds ties to even, like
/// CPython's `_Py_dg_dtoa` mode 3; `tests/test_pyjson_native.py` checks this against
/// Python on random values and on exact ties.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Json {
        Json::Object(vec![
            (
                "b".into(),
                Json::List(vec![Json::Int(1), Json::Float(2.5), Json::Null]),
            ),
            ("a".into(), Json::Object(vec![])),
            ("c".into(), Json::List(vec![])),
            ("d".into(), Json::str("x\"y")),
        ])
    }

    #[test]
    fn default_layout() {
        assert_eq!(
            dumps(
                &sample(),
                Layout {
                    indent: None,
                    sort_keys: false,
                    compact: false
                }
            ),
            r#"{"b": [1, 2.5, null], "a": {}, "c": [], "d": "x\"y"}"#
        );
    }

    #[test]
    fn compact_and_sorted() {
        assert_eq!(
            dumps(
                &sample(),
                Layout {
                    indent: None,
                    sort_keys: true,
                    compact: true
                }
            ),
            r#"{"a":{},"b":[1,2.5,null],"c":[],"d":"x\"y"}"#
        );
    }

    #[test]
    fn indent_two() {
        assert_eq!(
            dumps(&sample(), Layout::INDENT2),
            "{\n  \"b\": [\n    1,\n    2.5,\n    null\n  ],\n  \"a\": {},\n  \"c\": [],\n  \"d\": \"x\\\"y\"\n}"
        );
    }

    #[test]
    fn round_half_even_on_exact_ties() {
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(0.375, 2), 0.38);
        assert_eq!(py_round(2.5, 0), 2.0);
        assert_eq!(py_round(-0.0, 6), -0.0);
        assert_eq!(py_round(53.123456789, 6), 53.123457);
    }
}
