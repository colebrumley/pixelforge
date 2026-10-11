//! The Config field table (`pixelforge::config`) as Python objects, for
//! tests/test_field_table.py to compare with pixelforge/config.py, and the Rust `Config`
//! itself (canonical JSON, hash, errors, `replace`, palettes) for tests/test_config_native.py.
//!
//! Keyword arguments are converted to `pixelforge::config::Value` as they are, without any
//! coercion, so the Rust `_coerce` sees what Python's would: ints beyond i64 as `BigInt`,
//! NaN and infinities as floats, lists as lists, anything else as its `repr`.

use pixelforge::config::{Config, FieldValue, Value, FIELDS, PRESETS, PRESET_LONGEST_EDGE};
use pixelforge::palette;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

fn field_value<'py>(py: Python<'py>, v: &FieldValue) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &Value::from(*v))
}

fn to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::None => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::BigInt(s) => py.get_type::<PyInt>().call1((s.as_str(),))?,
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Other(r) => r.into_pyobject(py)?.into_any(),
    })
}

/// A Python object as a Config `Value`, unconverted.
fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::None);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract::<bool>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(match obj.extract::<i64>() {
            Ok(i) => Value::Int(i),
            Err(_) => {
                let text = obj
                    .py()
                    .get_type::<PyInt>()
                    .call_method1("__repr__", (obj,))?;
                Value::BigInt(text.extract::<String>()?)
            }
        });
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Value::Float(obj.extract::<f64>()?));
    }
    if obj.is_instance_of::<PyString>() {
        // A str with lone surrogates has no Rust form; it can only be rejected.
        return Ok(match obj.extract::<String>() {
            Ok(s) => Value::Str(s),
            Err(_) => Value::Other(obj.repr()?.extract::<String>()?),
        });
    }
    if obj.is_exact_instance_of::<PyList>() {
        let items = obj
            .try_iter()?
            .map(|item| from_py(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::List(items));
    }
    Ok(Value::Other(obj.repr()?.extract::<String>()?))
}

fn kwargs(dict: &Bound<'_, PyDict>) -> PyResult<Vec<(String, Value)>> {
    dict.iter()
        .map(|(k, v)| {
            let key = k
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("keywords must be strings"))?;
            Ok((key, from_py(&v)?))
        })
        .collect()
}

/// `pixelforge.errors.ConfigError(message)`.
fn config_error_py(py: Python<'_>, message: &str) -> PyErr {
    match py
        .import("pixelforge.errors")
        .and_then(|m| m.getattr("ConfigError"))
        .and_then(|cls| cls.call1((message,)))
    {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

pub(crate) fn build(py: Python<'_>, kw: &Bound<'_, PyDict>) -> PyResult<Config> {
    Config::new(&kwargs(kw)?).map_err(|e| config_error_py(py, &e.0))
}

/// Every Config field in declaration order, as dicts with keys name, type (the Python
/// annotation, e.g. "int | None"), default, help, choices (list or None) and upper_bound
/// (int or None).
#[pyfunction]
fn config_fields(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let out = PyList::empty(py);
    for f in FIELDS {
        let d = PyDict::new(py);
        d.set_item("name", f.name)?;
        d.set_item("type", f.python_type())?;
        d.set_item("default", field_value(py, &f.default)?)?;
        d.set_item("help", f.help)?;
        d.set_item("choices", f.choices.map(|c| c.to_vec()))?;
        d.set_item("upper_bound", f.upper_bound)?;
        out.append(d)?;
    }
    Ok(out)
}

/// {"presets": {name: {field: value}}, "longest_edge": {name: int}}, in table order.
#[pyfunction]
fn config_presets(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let presets = PyDict::new(py);
    for (name, values) in PRESETS {
        let d = PyDict::new(py);
        for (key, v) in *values {
            d.set_item(*key, field_value(py, v)?)?;
        }
        presets.set_item(*name, d)?;
    }
    let edges = PyDict::new(py);
    for (name, edge) in PRESET_LONGEST_EDGE {
        edges.set_item(*name, *edge)?;
    }
    let out = PyDict::new(py);
    out.set_item("presets", presets)?;
    out.set_item("longest_edge", edges)?;
    Ok(out)
}

/// `Config(**kwargs).canonical_json()` computed in Rust; ConfigError if invalid.
#[pyfunction]
fn config_canonical_json(py: Python<'_>, kwargs: &Bound<'_, PyDict>) -> PyResult<String> {
    Ok(build(py, kwargs)?.canonical_json())
}

/// `Config(**kwargs).hash()` computed in Rust; ConfigError if invalid, ValueError if the
/// palette cannot be loaded.
#[pyfunction]
fn config_hash(py: Python<'_>, kwargs: &Bound<'_, PyDict>) -> PyResult<String> {
    build(py, kwargs)?
        .hash()
        .map_err(|e| PyValueError::new_err(e.0))
}

/// `Config(**kwargs).to_dict()` computed in Rust (field order).
#[pyfunction]
fn config_to_dict<'py>(
    py: Python<'py>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (name, v) in build(py, kwargs)?.to_dict() {
        out.set_item(name, to_py(py, &v)?)?;
    }
    Ok(out)
}

/// `Config(**kwargs).replace(**c1).replace(**c2)...canonical_json()` computed in Rust;
/// `changes` is one dict or a list of dicts applied in order.
#[pyfunction]
fn config_replace_canonical_json(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    changes: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let chain: Vec<Bound<'_, PyDict>> = if let Ok(d) = changes.cast::<PyDict>() {
        vec![d.clone()]
    } else {
        changes
            .try_iter()?
            .map(|item| {
                let item = item?;
                let d = item.cast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err("changes must be a dict or a list of dicts")
                })?;
                Ok(d.clone())
            })
            .collect::<PyResult<_>>()?
    };
    let mut config = build(py, kwargs)?;
    for d in &chain {
        config = config
            .replace(&self::kwargs(d)?)
            .map_err(|e| config_error_py(py, &e.0))?;
    }
    Ok(config.canonical_json())
}

/// The ConfigError message `Config(**kwargs)` raises in Rust, or None when it is valid.
#[pyfunction]
fn config_error(kwargs: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
    Ok(Config::new(&self::kwargs(kwargs)?).err().map(|e| e.0))
}

/// The ConfigError message of `Config(**kwargs).validate_palette()` in Rust, or None when
/// the palette loads (or there is none). ConfigError if the kwargs themselves are invalid.
#[pyfunction]
fn config_validate_palette(py: Python<'_>, kwargs: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
    Ok(build(py, kwargs)?.validate_palette().err().map(|e| e.0))
}

/// `palette.parse_hex_lines(name)` in Rust: lowercase 'rrggbb' lines; ValueError with
/// Python's message if the palette cannot be loaded.
#[pyfunction]
fn palette_hex_lines(name: &str) -> PyResult<Vec<String>> {
    palette::parse_hex_lines(name).map_err(|e| PyValueError::new_err(e.0))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(config_fields, m)?)?;
    m.add_function(wrap_pyfunction!(config_presets, m)?)?;
    m.add_function(wrap_pyfunction!(config_canonical_json, m)?)?;
    m.add_function(wrap_pyfunction!(config_hash, m)?)?;
    m.add_function(wrap_pyfunction!(config_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(config_replace_canonical_json, m)?)?;
    m.add_function(wrap_pyfunction!(config_error, m)?)?;
    m.add_function(wrap_pyfunction!(config_validate_palette, m)?)?;
    m.add_function(wrap_pyfunction!(palette_hex_lines, m)?)?;
    Ok(())
}
