//! Python wrappers of `pixelforge::sheet` (the `compare` command's sheets, `--sweep` parsing
//! and flag text), for tests/test_sheet_native.py to compare with `pixelforge/cli.py`.
//!
//! Images are C-contiguous `(H, W, 4)` uint8 arrays. `ConfigError`s are raised as
//! `pixelforge.errors.ConfigError` with the library's message.

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pixelforge::config::Value;
use pixelforge::sheet::{self, RgbaImage};

use crate::color::py_shape;
use crate::config::{config_error_py, from_py, to_py};

fn image(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<RgbaImage> {
    let arr = obj.cast::<PyArray3<u8>>().map_err(|_| {
        PyValueError::new_err(format!("{what} must be a (H, W, 4) uint8 numpy.ndarray"))
    })?;
    let shape = arr.shape().to_vec();
    if shape[2] != 4 {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape (H, W, 4), got {}",
            py_shape(&shape)
        )));
    }
    let ro = arr.try_readonly()?;
    let data = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))?;
    Ok(RgbaImage::new(shape[0], shape[1], data.to_vec()))
}

fn images(list: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<RgbaImage>> {
    list.try_iter()?.map(|item| image(&item?, what)).collect()
}

fn to_array<'py>(py: Python<'py>, im: RgbaImage) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = Array3::from_shape_vec((im.h, im.w, 4), im.data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// `cli.side_by_side(images, gap)`.
#[pyfunction]
fn sheet_side_by_side<'py>(
    py: Python<'py>,
    images_list: &Bound<'py, PyAny>,
    gap: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let ims = images(images_list, "images")?;
    to_array(py, sheet::side_by_side(&ims, gap))
}

/// `cli.parse_sweep(text)`: a list of one-entry dicts; `ConfigError` as Python's.
#[pyfunction]
fn sheet_parse_sweep<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
    let columns = sheet::parse_sweep(text).map_err(|e| config_error_py(py, &e.0))?;
    let out = PyList::empty(py);
    for column in columns {
        let dict = PyDict::new(py);
        for (name, value) in &column {
            dict.set_item(name, to_py(py, value)?)?;
        }
        out.append(dict)?;
    }
    Ok(out)
}

/// `cli.PREPASS_COLUMNS` as a list of dicts.
#[pyfunction]
fn sheet_prepass_columns(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let out = PyList::empty(py);
    for column in sheet::prepass_columns() {
        let dict = PyDict::new(py);
        for (name, value) in &column {
            dict.set_item(name, to_py(py, value)?)?;
        }
        out.append(dict)?;
    }
    Ok(out)
}

/// `cli._show(value)`.
#[pyfunction]
fn sheet_show(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(sheet::show(&from_py(value)?))
}

/// `cli.override_flags(overrides)`.
#[pyfunction]
fn sheet_override_flags(overrides: &Bound<'_, PyDict>) -> PyResult<String> {
    let pairs = overrides
        .iter()
        .map(|(k, v)| {
            let key = k
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("override names must be strings"))?;
            Ok((key, from_py(&v)?))
        })
        .collect::<PyResult<Vec<(String, Value)>>>()?;
    Ok(sheet::override_flags(&pairs))
}

/// `cli.preview_scale` for images of `sizes` (`(h, w)` pairs): `(scales, warning)`, the
/// warning text without the `pixelforge: ` prefix, or None.
#[pyfunction]
fn sheet_preview_scale(sizes: Vec<(usize, usize)>, scales: Vec<i64>) -> (Vec<i64>, Option<String>) {
    sheet::preview_scale(&sizes, &scales)
}

/// `cli.proof_sheet(cells, row_labels, column_labels)` with the embedded label font.
#[pyfunction]
fn sheet_proof_sheet<'py>(
    py: Python<'py>,
    cells: &Bound<'py, PyAny>,
    row_labels: Vec<String>,
    column_labels: Vec<String>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let rows = cells
        .try_iter()?
        .map(|row| images(&row?, "cells"))
        .collect::<PyResult<Vec<_>>>()?;
    if row_labels.is_empty() || column_labels.is_empty() {
        return Err(PyValueError::new_err("proof_sheet: no rows or no columns"));
    }
    if rows.len() != row_labels.len() || rows.iter().any(|r| r.len() != column_labels.len()) {
        return Err(PyValueError::new_err(
            "proof_sheet: cells must have one row per row label and one cell per column label",
        ));
    }
    let sheet = py.detach(|| sheet::proof_sheet(&rows, &row_labels, &column_labels));
    to_array(py, sheet)
}

/// The `(width, height)` of a proof-sheet label box for `text`.
#[pyfunction]
fn sheet_label_size(text: &str) -> (usize, usize) {
    sheet::label_size(text)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sheet_side_by_side, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_parse_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_prepass_columns, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_show, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_override_flags, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_preview_scale, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_proof_sheet, m)?)?;
    m.add_function(wrap_pyfunction!(sheet_label_size, m)?)?;
    Ok(())
}
