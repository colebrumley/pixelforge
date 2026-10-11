//! `json.loads` into [`pixelforge::pyjson::Json`]: object keys in file order (as a Python
//! dict), integers as `Int`, numbers with a fraction or exponent as `Float` (read exactly),
//! so `dumps(loads(text))` reproduces a file `json.dumps` wrote. Plus small accessors.

use std::fmt;

use pixelforge::pyjson::Json;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};

struct Parsed(Json);

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Parsed;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_unit<E: de::Error>(self) -> Result<Parsed, E> {
        Ok(Parsed(Json::Null))
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Parsed, E> {
        Ok(Parsed(Json::Bool(v)))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Parsed, E> {
        Ok(Parsed(Json::Int(v)))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Parsed, E> {
        i64::try_from(v)
            .map(|i| Parsed(Json::Int(i)))
            .map_err(|_| E::custom(format!("integer {v} out of range")))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Parsed, E> {
        Ok(Parsed(Json::Float(v)))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Parsed, E> {
        Ok(Parsed(Json::Str(v.to_string())))
    }
    fn visit_string<E: de::Error>(self, v: String) -> Result<Parsed, E> {
        Ok(Parsed(Json::Str(v)))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Parsed, A::Error> {
        let mut items = Vec::new();
        while let Some(Parsed(item)) = seq.next_element()? {
            items.push(item);
        }
        Ok(Parsed(Json::List(items)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Parsed, A::Error> {
        let mut items: Vec<(String, Json)> = Vec::new();
        while let Some((key, Parsed(value))) = map.next_entry::<String, Parsed>()? {
            // A repeated key keeps its first position and its last value (a Python dict).
            match items.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = value,
                None => items.push((key, value)),
            }
        }
        Ok(Parsed(Json::Object(items)))
    }
}

impl<'de> de::Deserialize<'de> for Parsed {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Parsed, D::Error> {
        d.deserialize_any(JsonVisitor)
    }
}

/// Parse JSON text.
pub fn loads(text: &str) -> Result<Json, String> {
    serde_json::from_str::<Parsed>(text)
        .map(|p| p.0)
        .map_err(|e| e.to_string())
}

/// `obj.get(key)` for an object (`None` for anything else).
pub fn get<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
    match value {
        Json::Object(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

pub fn as_f64(value: &Json) -> Option<f64> {
    match value {
        Json::Int(i) => Some(*i as f64),
        Json::Float(x) => Some(*x),
        _ => None,
    }
}

pub fn as_i64(value: &Json) -> Option<i64> {
    match value {
        Json::Int(i) => Some(*i),
        _ => None,
    }
}

pub fn as_str(value: &Json) -> Option<&str> {
    match value {
        Json::Str(s) => Some(s),
        _ => None,
    }
}

pub fn as_bool(value: &Json) -> Option<bool> {
    match value {
        Json::Bool(b) => Some(*b),
        _ => None,
    }
}

pub fn as_list(value: &Json) -> Option<&[Json]> {
    match value {
        Json::List(items) => Some(items),
        _ => None,
    }
}

pub fn entries(value: &Json) -> &[(String, Json)] {
    match value {
        Json::Object(items) => items,
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelforge::pyjson::{dumps, Layout};

    #[test]
    fn keeps_order_and_number_kinds() {
        let text = "{\n  \"b\": 1,\n  \"a\": [\n    2.5,\n    1e-05,\n    null,\n    true\n  ]\n}";
        let value = loads(text).unwrap();
        assert_eq!(dumps(&value, Layout::INDENT2), text);
        assert_eq!(get(&value, "b"), Some(&Json::Int(1)));
    }
}
