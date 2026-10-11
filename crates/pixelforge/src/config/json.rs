//! The canonical JSON writer (docs/rust-migration-plan.md, Decision 7): exactly
//! `json.dumps(obj, sort_keys=True, separators=(",", ":"))` for a flat object of Config
//! values. Keys sorted by code point (byte order of UTF-8 is the same order), strings with
//! `ensure_ascii` escaping, `true`/`false`/`null`, integers in decimal, floats as Python's
//! `repr`.

use super::Value;
use crate::pyrepr::{float_repr, json_string};

/// One value as `json.dumps` writes it.
pub(super) fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::None => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::BigInt(s) => out.push_str(s),
        // json.dumps writes NaN and Infinity for non-finite floats; a Config holds none.
        Value::Float(x) if x.is_nan() => out.push_str("NaN"),
        Value::Float(x) if x.is_infinite() => {
            out.push_str(if *x > 0.0 { "Infinity" } else { "-Infinity" })
        }
        Value::Float(x) => out.push_str(&float_repr(*x)),
        Value::Str(s) => out.push_str(&json_string(s)),
        Value::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Other(r) => out.push_str(&json_string(r)),
    }
}

/// `json.dumps(dict(items), sort_keys=True, separators=(",", ":"))`.
pub(super) fn dumps_sorted(items: &[(&str, Value)]) -> String {
    let mut sorted: Vec<&(&str, Value)> = items.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::from("{");
    for (i, (key, value)) in sorted.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&json_string(key));
        out.push(':');
        write_value(&mut out, value);
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_compact() {
        let items = [
            ("b", Value::Float(1e-5)),
            ("a", Value::str("p\u{e4}\"")),
            ("C", Value::None),
            ("g_T", Value::Bool(true)),
            ("g_a", Value::Int(-3)),
        ];
        assert_eq!(
            dumps_sorted(&items),
            r#"{"C":null,"a":"p\u00e4\"","b":1e-05,"g_T":true,"g_a":-3}"#
        );
    }

    /// `canonical_json_of_list` in float_repr.json: SHA-256 of
    /// `json.dumps(floats, separators=(",", ":"))` over the 10,000 floats.
    #[test]
    fn float_list_hash() {
        use sha2::{Digest, Sha256};
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/oracle/float_repr.json"
        );
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let floats: Vec<Value> = doc["repr"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| Value::Float(r.as_str().unwrap().parse().unwrap()))
            .collect();
        let mut out = String::new();
        write_value(&mut out, &Value::List(floats));
        let digest = format!("{:x}", Sha256::digest(out.as_bytes()));
        assert_eq!(digest, doc["canonical_json_of_list"].as_str().unwrap());
    }
}
