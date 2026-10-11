//! The Config field table (`pixelforge::config`) as Python objects, for
//! tests/test_field_table.py to compare with pixelforge/config.py.

use pixelforge::config::{Value, FIELDS, PRESETS, PRESET_LONGEST_EDGE};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn value<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match *v {
        Value::None => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
    })
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
        d.set_item("default", value(py, &f.default)?)?;
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
            d.set_item(*key, value(py, v)?)?;
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(config_fields, m)?)?;
    m.add_function(wrap_pyfunction!(config_presets, m)?)?;
    Ok(())
}
