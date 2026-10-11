//! The `Downscaled` result as a Python dict, and `neighbor_mean4` for its parity test.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pixelforge::downscale::{Downscaled, StatValue};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[allow(dead_code)]
fn array2<'py, T: numpy::Element>(
    py: Python<'py>,
    v: Vec<T>,
    h: usize,
    w: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    PyArray1::from_vec(py, v)
        .reshape([h, w])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[allow(dead_code)]
fn array3<'py>(
    py: Python<'py>,
    v: Vec<f64>,
    h: usize,
    w: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    PyArray1::from_vec(py, v)
        .reshape([h, w, 3])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// `Downscaled` → dict with the keys of `pixelforge.downscale.Downscaled` (None for the
/// optional arrays) and `stats` as a dict in insertion order.
// Used by the kopf and gerstner bindings (Phase 5); remove the allow when they land.
#[allow(dead_code)]
pub(crate) fn downscaled_to_py<'py>(
    py: Python<'py>,
    d: Downscaled,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let (h, w) = (d.h, d.w);
    out.set_item("small_lab", array3(py, d.small_lab, h, w)?)?;
    out.set_item("small_mask", array2(py, d.small_mask, h, w)?)?;
    out.set_item(
        "palette_lab",
        d.palette_lab
            .map(|p| {
                let k = p.len() / 3;
                array2(py, p, k, 3)
            })
            .transpose()?,
    )?;
    out.set_item(
        "palette_lab_unsaturated",
        d.palette_lab_unsaturated
            .map(|p| {
                let k = p.len() / 3;
                array2(py, p, k, 3)
            })
            .transpose()?,
    )?;
    out.set_item(
        "indices",
        d.indices.map(|v| array2(py, v, h, w)).transpose()?,
    )?;
    out.set_item(
        "mean_lab",
        d.mean_lab.map(|v| array3(py, v, h, w)).transpose()?,
    )?;
    let stats = PyDict::new(py);
    for (name, value) in d.stats {
        match value {
            StatValue::Int(i) => stats.set_item(name, i)?,
            StatValue::Bool(b) => stats.set_item(name, b)?,
            StatValue::Float(f) => stats.set_item(name, f)?,
        }
    }
    out.set_item("stats", stats)?;
    Ok(out)
}

/// `pixelforge.downscale.neighbor_mean4` on an (H, W) float64 array.
#[pyfunction]
fn neighbor_mean4<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = grid.shape();
    let (h, w) = (shape[0], shape[1]);
    let values: Vec<f64> = grid.as_array().iter().copied().collect();
    let out = pixelforge::downscale::neighbor_mean4(&values, h, w);
    out.into_pyarray(py)
        .reshape([h, w])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neighbor_mean4, m)?)?;
    Ok(())
}
