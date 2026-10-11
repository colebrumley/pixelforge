//! The Rust `Config` as the Python class `NativeConfig` (wrapped by `pixelforge.Config`),
//! and the field table and presets as Python values.
//!
//! Keyword arguments are converted to `pixelforge::config::Value` as they are, without any
//! coercion, so the library's coercion and validation see what Python passed: ints beyond
//! i64 as `BigInt`, NaN and infinities as floats, lists as lists, anything else as its
//! `repr` (rejected with the library's message).

use pixelforge::config::{Config, FieldValue, Value, FIELDS, PRESETS};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

use crate::package_error;

pub(crate) fn to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
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

fn field_value<'py>(py: Python<'py>, v: &FieldValue) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &Value::from(*v))
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

fn config_error(py: Python<'_>, message: &str) -> PyErr {
    package_error(py, "ConfigError", message)
}

/// A validated configuration (`Config::new`): field defaults ← preset ← the keyword
/// arguments. Immutable; `replace` returns a new one.
#[pyclass(name = "NativeConfig", module = "pixelforge._native", frozen)]
pub struct NativeConfig {
    pub(crate) config: Config,
}

#[pymethods]
impl NativeConfig {
    /// `NativeConfig(kwargs)`; `ConfigError` with the library's message if invalid.
    #[new]
    fn new(py: Python<'_>, kwargs: &Bound<'_, PyDict>) -> PyResult<Self> {
        let config = Config::new(&self::kwargs(kwargs)?).map_err(|e| config_error(py, &e.0))?;
        Ok(NativeConfig { config })
    }

    /// Every field and its value, in field order.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, v) in self.config.to_dict() {
            out.set_item(name, to_py(py, &v)?)?;
        }
        Ok(out)
    }

    /// The canonical JSON embedded in every PNG (sorted keys, no whitespace).
    fn canonical_json(&self) -> String {
        self.config.canonical_json()
    }

    /// SHA-256 of the canonical JSON, the version and the palette's colours (the
    /// `config_hash` of `_meta.json`); `ConfigError` if `palette_name` cannot be read.
    fn hash(&self, py: Python<'_>) -> PyResult<String> {
        self.config.hash().map_err(|e| config_error(py, &e.0))
    }

    /// A copy with `changes` applied; a changed preset is re-applied to every field that
    /// was not given explicitly.
    fn replace(&self, py: Python<'_>, changes: &Bound<'_, PyDict>) -> PyResult<NativeConfig> {
        let config = self
            .config
            .replace(&kwargs(changes)?)
            .map_err(|e| config_error(py, &e.0))?;
        Ok(NativeConfig { config })
    }

    /// Load `palette_name` (file read and parse); `ConfigError` if it is unusable.
    fn validate_palette(&self, py: Python<'_>) -> PyResult<()> {
        self.config
            .validate_palette()
            .map_err(|e| config_error(py, &e.0))
    }

    /// `{"config": dict, "palette_sha256": str | None, "palette": [str] | None,
    /// "version": str}`: everything needed to reproduce a run besides the input image.
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = self
            .config
            .provenance()
            .map_err(|e| config_error(py, &e.0))?;
        let config = PyDict::new(py);
        for (name, v) in &p.config {
            config.set_item(*name, to_py(py, v)?)?;
        }
        let out = PyDict::new(py);
        out.set_item("config", config)?;
        out.set_item("palette_sha256", p.palette_sha256)?;
        out.set_item("palette", p.palette)?;
        out.set_item("version", p.version)?;
        Ok(out)
    }

    /// `(width, height)` of `canvas`, or None.
    fn canvas_size(&self) -> Option<(i64, i64)> {
        self.config.canvas_size()
    }
}

/// The field names in declaration order.
#[pyfunction]
fn config_fields() -> Vec<&'static str> {
    FIELDS.iter().map(|f| f.name).collect()
}

/// `{preset: {field: value}}`: each preset's overrides of the field defaults.
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
    Ok(presets)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeConfig>()?;
    m.add_function(wrap_pyfunction!(config_fields, m)?)?;
    m.add_function(wrap_pyfunction!(config_presets, m)?)?;
    Ok(())
}
