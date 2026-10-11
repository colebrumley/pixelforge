//! `json_dumps` and `py_round` for `tests/test_pyjson_native.py`.

use pixelforge::pyjson::{dumps, py_round, Json, Layout};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

pub(crate) fn to_json(obj: &Bound<'_, PyAny>) -> PyResult<Json> {
    if obj.is_none() {
        Ok(Json::Null)
    } else if let Ok(b) = obj.cast::<PyBool>() {
        Ok(Json::Bool(b.is_true()))
    } else if let Ok(i) = obj.cast::<PyInt>() {
        Ok(Json::Int(i.extract()?))
    } else if let Ok(f) = obj.cast::<PyFloat>() {
        Ok(Json::Float(f.value()))
    } else if let Ok(s) = obj.cast::<PyString>() {
        Ok(Json::Str(s.to_str()?.to_string()))
    } else if let Ok(list) = obj.cast::<PyList>() {
        list.iter()
            .map(|v| to_json(&v))
            .collect::<PyResult<Vec<_>>>()
            .map(Json::List)
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        tuple
            .iter()
            .map(|v| to_json(&v))
            .collect::<PyResult<Vec<_>>>()
            .map(Json::List)
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        let mut items = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err("json_dumps: keys must be str"))?
                .to_str()?
                .to_string();
            items.push((key, to_json(&v)?));
        }
        Ok(Json::Object(items))
    } else {
        Err(PyTypeError::new_err(format!(
            "json_dumps: unsupported value {}",
            obj.get_type().name()?
        )))
    }
}

/// `json.dumps(obj, indent=indent, sort_keys=sort_keys, separators=(",", ":") if compact)`.
#[pyfunction]
#[pyo3(signature = (obj, indent=None, sort_keys=false, compact=false))]
fn json_dumps(
    obj: &Bound<'_, PyAny>,
    indent: Option<usize>,
    sort_keys: bool,
    compact: bool,
) -> PyResult<String> {
    Ok(dumps(
        &to_json(obj)?,
        Layout {
            indent,
            sort_keys,
            compact,
        },
    ))
}

/// Python's `round(x, ndigits)`.
#[pyfunction]
fn round_float(x: f64, ndigits: usize) -> f64 {
    py_round(x, ndigits)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(json_dumps, m)?)?;
    m.add_function(wrap_pyfunction!(round_float, m)?)?;
    Ok(())
}
