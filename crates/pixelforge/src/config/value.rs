//! Runtime values for Config fields: what a caller passes (Python keyword arguments through
//! the binding, later JSON and CLI flags), before `_coerce` checks and converts them.

use super::FieldValue;
use crate::pyrepr::{float_repr, str_repr};

/// A value given for a Config field, typed as the source had it.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    /// An integer outside the i64 range, as decimal text (optional leading `-`). Kept so
    /// the out-of-range messages can name it.
    BigInt(String),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    /// Any other value, as its Python `repr`. Never accepted by a field.
    Other(String),
}

impl Value {
    /// An integer from decimal text: `Int` if it fits in an i64, else `BigInt`.
    pub fn int_from_decimal(text: &str) -> Value {
        match text.parse::<i64>() {
            Ok(i) => Value::Int(i),
            Err(_) => Value::BigInt(text.to_string()),
        }
    }

    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    /// Python's `repr` of the value.
    pub fn repr(&self) -> String {
        match self {
            Value::None => "None".to_string(),
            Value::Bool(true) => "True".to_string(),
            Value::Bool(false) => "False".to_string(),
            Value::Int(i) => i.to_string(),
            Value::BigInt(s) => s.clone(),
            Value::Float(x) => float_repr(*x),
            Value::Str(s) => str_repr(s),
            Value::List(items) => {
                let parts: Vec<String> = items.iter().map(Value::repr).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Other(r) => r.clone(),
        }
    }
}

impl From<FieldValue> for Value {
    fn from(v: FieldValue) -> Value {
        match v {
            FieldValue::None => Value::None,
            FieldValue::Bool(b) => Value::Bool(b),
            FieldValue::Int(i) => Value::Int(i),
            FieldValue::Float(x) => Value::Float(x),
            FieldValue::Str(s) => Value::Str(s.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprs() {
        assert_eq!(Value::None.repr(), "None");
        assert_eq!(Value::Bool(true).repr(), "True");
        assert_eq!(Value::Float(1.0).repr(), "1.0");
        assert_eq!(Value::Float(f64::NAN).repr(), "nan");
        assert_eq!(Value::str("16").repr(), "'16'");
        assert_eq!(
            Value::List(vec![Value::Int(1), Value::Float(2.0), Value::str("a")]).repr(),
            "[1, 2.0, 'a']"
        );
        assert_eq!(
            Value::int_from_decimal("-9223372036854775808"),
            Value::Int(i64::MIN)
        );
        assert_eq!(
            Value::int_from_decimal("9223372036854775808"),
            Value::BigInt("9223372036854775808".to_string())
        );
    }
}
