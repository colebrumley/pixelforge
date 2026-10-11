//! Python's text forms of values, byte for byte: `repr(float)` (also `str(float)` and the
//! float form `json.dumps` writes), `repr(str)`, and `json.dumps(str)` with
//! `ensure_ascii=True`. The canonical Config JSON and the Config error messages are built
//! from these.

use crate::pyunicode;
use std::fmt::Write;

/// `repr(x)` for a float.
///
/// The digits are the shortest that round-trip (Rust's `{:e}` and Python's `repr` both
/// produce them). Python's layout: with `d1.d2…dn × 10^e`, exponent form `d1.d2…dne±XX`
/// (at least two exponent digits, no point if there is one digit) when `e < -4` or
/// `e >= 16`, else positional with at least one digit after the point (`1.0`, `0.0001`,
/// `1000000000000000.0`). `-0.0` keeps its sign; non-finite values are `inf`, `-inf`, `nan`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let (digits, exp) = shortest_digits(x.abs());
    let mut out = String::with_capacity(digits.len() + 8);
    if x.is_sign_negative() {
        out.push('-');
    }
    let n = digits.len() as i32;
    if !(-4..16).contains(&exp) {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exp < 0 { '-' } else { '+' };
        write!(out, "e{sign}{:02}", exp.abs()).unwrap();
    } else if exp < 0 {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let int_len = exp + 1;
        if n <= int_len {
            out.push_str(&digits);
            for _ in 0..(int_len - n) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..int_len as usize]);
            out.push('.');
            out.push_str(&digits[int_len as usize..]);
        }
    }
    out
}

/// Split `{:e}` / `{:.Ne}` output into its digits and decimal exponent.
fn split_sci(sci: &str) -> (String, i32) {
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    (mantissa.chars().filter(|&c| c != '.').collect(), exp)
}

/// The digits and decimal exponent of Python's `repr` for a finite `x >= 0`: the fewest
/// digits that read back as `x` and, among those, the ones closest to `x`.
///
/// Rust's `{:e}` gives the fewest digits, but when two strings of that length round-trip and
/// `x` lies exactly halfway between them it does not pick the one Python (David Gay's dtoa,
/// mode 0) picks, e.g. 1605046810404663.25 → Rust `…663.3`, Python `…663.2`. So the length is
/// taken from `{:e}` and the digits from `{:.Ne}`, which rounds the exact binary value
/// correctly (ties to even), whenever that string round-trips.
fn shortest_digits(x: f64) -> (String, i32) {
    let shortest = split_sci(&format!("{x:e}"));
    let precision = shortest.0.len() - 1;
    let rounded = format!("{x:.precision$e}");
    if rounded.parse::<f64>() == Ok(x) {
        split_sci(&rounded)
    } else {
        shortest
    }
}

/// `repr(s)` for a `str`: single quotes unless the text has a `'` and no `"`; `\\`, the
/// quote, `\t`, `\n`, `\r` escaped; other ASCII controls and DEL as `\xhh`; non-ASCII kept
/// when printable, else `\xhh`, `\uhhhh` or `\Uhhhhhhhh`.
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        let u = c as u32;
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if u < 0x20 || u == 0x7f => write!(out, "\\x{u:02x}").unwrap(),
            _ if u < 0x7f || pyunicode::is_printable(c) => out.push(c),
            _ if u <= 0xff => write!(out, "\\x{u:02x}").unwrap(),
            _ if u <= 0xffff => write!(out, "\\u{u:04x}").unwrap(),
            _ => write!(out, "\\U{u:08x}").unwrap(),
        }
    }
    out.push(quote);
    out
}

/// `repr(list_of_str)`, e.g. `['box', 'kopf']`.
pub fn str_list_repr<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| str_repr(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}

/// `json.dumps(s)` (`ensure_ascii=True`): `\"`, `\\`, `\n`, `\r`, `\t`, `\b`, `\f`; every
/// other character outside U+0020..U+007E as `\uxxxx` (lowercase hex), characters beyond
/// the BMP as a surrogate pair.
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(out, "\\u{unit:04x}").unwrap();
                }
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // the exact binary values are the point
    fn float_layout() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-1.5, "-1.5"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.00011, "0.00011"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (9999999999999998.0, "9999999999999998.0"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.1 + 0.2, "0.30000000000000004"),
            (123456789.12345679, "123456789.12345679"),
            (1e22, "1e+22"),
            // Exactly halfway between two shortest candidates: ties to even, as dtoa.
            (-1605046810404663.25, "-1605046810404663.2"),
            (249455451002943.125, "249455451002943.12"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ];
        for &(x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:?}");
        }
    }

    /// Every one of the 10,000 recorded Python reprs: parse, format, compare.
    #[test]
    fn float_repr_oracle() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/oracle/float_repr.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        let reprs = doc["repr"].as_array().unwrap();
        assert_eq!(reprs.len(), 10_000);
        let mut bad = Vec::new();
        for r in reprs {
            let r = r.as_str().unwrap();
            let x: f64 = r.parse().unwrap();
            let got = float_repr(x);
            if got != r {
                bad.push(format!("{r} -> {got}"));
            }
        }
        assert!(
            bad.is_empty(),
            "{} mismatches: {:?}",
            bad.len(),
            &bad[..bad.len().min(20)]
        );
    }

    #[test]
    fn strings() {
        assert_eq!(str_repr("tiles"), "'tiles'");
        assert_eq!(str_repr("a'b"), "\"a'b\"");
        assert_eq!(str_repr("a'b\""), "'a\\'b\"'");
        assert_eq!(
            str_repr("p\u{e4}lette \u{2713}.hex"),
            "'p\u{e4}lette \u{2713}.hex'"
        );
        assert_eq!(str_repr("a\0\\\t"), "'a\\x00\\\\\\t'");
        assert_eq!(
            str_repr("\x7f\u{85}\u{2028}\u{e0001}\u{ad}\u{1f600}"),
            "'\\x7f\\x85\\u2028\\U000e0001\\xad\u{1f600}'"
        );
        assert_eq!(
            str_list_repr(&["background", "sprite"]),
            "['background', 'sprite']"
        );
    }

    #[test]
    fn json_strings() {
        assert_eq!(
            json_string("a\"b\\c\n\r\t\u{8}\u{c}\u{1}\u{7f}"),
            "\"a\\\"b\\\\c\\n\\r\\t\\b\\f\\u0001\\u007f\""
        );
        assert_eq!(
            json_string("\u{e4}\u{2713}\u{1f600}"),
            "\"\\u00e4\\u2713\\ud83d\\ude00\""
        );
    }
}
