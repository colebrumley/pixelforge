//! Python wrappers of `pixelforge::preprocess`: the whole preprocessing stage behind
//! `pixelforge.preprocess.run`, and the two subject helpers `batch` uses
//! (`subject_bbox`, `subject_brightness`). `remove_bg` is applied by the Python caller, which
//! passes the replaced alpha.
//!
//! Arguments are checked as in `mask.rs` (rgb float64 `(H, W, 3)`, alpha uint8 `(H, W)`,
//! C-contiguous, non-empty); the Config arrives as `config.to_dict()`. Errors keep Python's
//! class (`ConfigError`, `PixelforgeError`) and message; the layout's aspect warning is a
//! `UserWarning`, as `resolve_layout` emits it.

use std::ffi::CString;

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pixelforge::config::Config;
use pixelforge::preprocess::{self, Error};

use crate::color::{shaped, slice, value_error};
use crate::config::build;
use crate::mask::{array, nonempty, pixelforge_error, same_size};

fn error_py(py: Python<'_>, e: Error) -> PyErr {
    match e {
        Error::Config(e) => pixelforge_error(py, "ConfigError", &e.0),
        Error::Pixelforge(e) => pixelforge_error(py, "PixelforgeError", &e.0),
    }
}

/// The checked image and alpha with their size, and the rebuilt Config.
#[allow(clippy::type_complexity)]
fn inputs<'py>(
    py: Python<'py>,
    func: &str,
    rgb: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<(
    PyReadonlyArrayDyn<'py, f64>,
    PyReadonlyArrayDyn<'py, u8>,
    usize,
    usize,
    Config,
)> {
    let config = build(py, config)?;
    let (rshape, rarr) = array::<f64>(func, "rgb", "float64", rgb, true)?;
    let (ashape, aarr) = array::<u8>(func, "alpha", "uint8", alpha, false)?;
    same_size(func, &rshape, &ashape, ("rgb", "alpha"))?;
    nonempty(func, &ashape)?;
    Ok((rarr, aarr, ashape[0], ashape[1], config))
}

/// `preprocess.run(rgb, alpha, Config(**config))` with `alpha` already through rembg when
/// `remove_bg` is set. Returns a dict with every `Preprocessed` field: lab float64
/// (H, W, 3), mask bool (H, W), weight float64 (H, W), out_width, out_height, pad_out,
/// outline_margin, prereduce_factor (ints), background_keyed (bool), layout (tuple of four
/// ints or None), fit (str or None).
#[pyfunction]
fn preprocess_run<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    const F: &str = "preprocess_run";
    let (rarr, aarr, h, w, config) = inputs(py, F, rgb, alpha, config)?;
    let (r, a) = (slice(F, "rgb", &rarr)?, slice(F, "alpha", &aarr)?);
    let pre = py
        .detach(|| preprocess::run(r, a, h, w, &config))
        .map_err(|e| error_py(py, e))?;
    for text in &pre.warnings {
        let message = CString::new(text.as_str()).map_err(|e| value_error(F, e.to_string()))?;
        PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 1)?;
    }
    let out = PyDict::new(py);
    let (ph, pw) = (pre.h, pre.w);
    out.set_item("lab", shaped(py, &[ph, pw, 3], pre.lab))?;
    out.set_item("mask", shaped(py, &[ph, pw], pre.mask))?;
    out.set_item("weight", shaped(py, &[ph, pw], pre.weight))?;
    out.set_item("out_width", pre.geometry.out_width)?;
    out.set_item("out_height", pre.geometry.out_height)?;
    out.set_item("pad_out", pre.geometry.pad_out)?;
    out.set_item("outline_margin", pre.geometry.outline_margin)?;
    out.set_item("layout", pre.geometry.layout)?;
    out.set_item("background_keyed", pre.background_keyed)?;
    out.set_item("fit", pre.fit)?;
    out.set_item("prereduce_factor", pre.prereduce_factor)?;
    Ok(out)
}

/// `preprocess.subject_bbox(rgb, alpha, Config(**config))` with `alpha` already through
/// rembg: the crop box (y0, y1, x0, x1).
#[pyfunction]
fn subject_bbox<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<(usize, usize, usize, usize)> {
    const F: &str = "subject_bbox";
    let (rarr, aarr, h, w, config) = inputs(py, F, rgb, alpha, config)?;
    let (r, a) = (slice(F, "rgb", &rarr)?, slice(F, "alpha", &aarr)?);
    py.detach(|| preprocess::subject_bbox(r, a, h, w, &config))
        .map_err(|e| pixelforge_error(py, "PixelforgeError", &e.0))
}

/// `preprocess.subject_brightness(rgb, alpha, Config(**config))` with `alpha` already
/// through rembg: float64 (N,), the largest channel of the subject's mostly opaque pixels.
#[pyfunction]
fn subject_brightness<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    const F: &str = "subject_brightness";
    let (rarr, aarr, h, w, config) = inputs(py, F, rgb, alpha, config)?;
    let (r, a) = (slice(F, "rgb", &rarr)?, slice(F, "alpha", &aarr)?);
    let value = py
        .detach(|| preprocess::subject_brightness(r, a, h, w, &config))
        .map_err(|e| pixelforge_error(py, "PixelforgeError", &e.0))?;
    let n = value.len();
    Ok(shaped(py, &[n], value))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_run, m)?)?;
    m.add_function(wrap_pyfunction!(subject_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(subject_brightness, m)?)?;
    Ok(())
}
