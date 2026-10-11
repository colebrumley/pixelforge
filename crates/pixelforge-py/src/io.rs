//! Image input: `load(path, max_pixels)` and `from_array(array, max_pixels)`, both
//! returning `(rgb (H, W, 3) float64, alpha (H, W) uint8, sha256)`.
//!
//! Errors: `PixelforgeError` with the library's message, `OSError` (its errno subclass,
//! e.g. `FileNotFoundError`) built as `OSError(errno, strerror, path)`. The multi-frame
//! notice is a `UserWarning`.

use std::ffi::CString;
use std::path::PathBuf;

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pixelforge::io::{self, ArrayData, IoError, Loaded};

use crate::package_error;

pub(crate) fn io_error(py: Python<'_>, err: IoError) -> PyErr {
    match err {
        IoError::Pixelforge(e) => package_error(py, "PixelforgeError", &e.0),
        IoError::Os { error, path } => match error.raw_os_error() {
            Some(code) => {
                let built = py.import("os").and_then(|os| {
                    let strerror = os.getattr("strerror")?.call1((code,))?;
                    py.import("builtins")?
                        .getattr("OSError")?
                        .call1((code, strerror, path))
                });
                match built {
                    Ok(exc) => PyErr::from_value(exc),
                    Err(e) => e,
                }
            }
            None => PyErr::from(error),
        },
    }
}

/// Emit each text as a `UserWarning` attributed `stacklevel` frames up.
pub(crate) fn warn(py: Python<'_>, texts: &[String], stacklevel: i32) -> PyResult<()> {
    for text in texts {
        let message =
            CString::new(text.as_str()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, stacklevel)?;
    }
    Ok(())
}

type LoadedTuple<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<u8>>, String);

fn loaded_arrays<'py>(py: Python<'py>, loaded: Loaded) -> PyResult<LoadedTuple<'py>> {
    let (h, w) = (loaded.h, loaded.w);
    let rgb = Array3::from_shape_vec((h, w, 3), loaded.rgb)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let alpha = Array2::from_shape_vec((h, w), loaded.alpha)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((rgb.into_pyarray(py), alpha.into_pyarray(py), loaded.sha256))
}

/// The budget as the library takes it: the default when None, a negative budget refuses
/// everything.
pub(crate) fn budget(max_pixels: Option<i64>) -> u64 {
    match max_pixels {
        None => io::MAX_INPUT_PIXELS,
        Some(n) => u64::try_from(n).unwrap_or(0),
    }
}

/// Decode an image file (PNG, JPEG, GIF, WEBP, BMP, TIFF; regular files only; EXIF
/// orientation applied; the first frame of an animation, with a `UserWarning`).
#[pyfunction]
#[pyo3(signature = (path, max_pixels=None))]
fn load<'py>(
    py: Python<'py>,
    path: PathBuf,
    max_pixels: Option<i64>,
) -> PyResult<LoadedTuple<'py>> {
    let budget = budget(max_pixels);
    let loaded = py
        .detach(|| io::load(&path, budget))
        .map_err(|e| io_error(py, e))?;
    warn(py, &loaded.warnings, 2)?;
    loaded_arrays(py, loaded)
}

/// An `(H, W, 3)` or `(H, W, 4)` array of uint8, or of float in [0, 1]; hashed as its shape
/// plus RGBA bytes. `PixelforgeError` for other shapes or dtypes and above the budget.
#[pyfunction]
#[pyo3(signature = (array, max_pixels=None))]
fn from_array<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    max_pixels: Option<i64>,
) -> PyResult<LoadedTuple<'py>> {
    let np = py.import("numpy")?;
    let arr = np.getattr("asarray")?.call1((array,))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let dtype = arr.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let name = dtype.str()?.to_string();
    let budget = budget(max_pixels);
    let shape_ok = shape.len() == 3 && (shape[2] == 3 || shape[2] == 4);
    let result = if !shape_ok {
        io::from_image_array(&shape, ArrayData::Other(""), budget)
    } else if kind == "f" {
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", np.getattr("float64")?)?;
        let contiguous = np
            .getattr("ascontiguousarray")?
            .call((arr,), Some(&kwargs))?;
        let typed = contiguous.cast::<PyArrayDyn<f64>>()?.try_readonly()?;
        let data = typed
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        io::from_image_array(&shape, ArrayData::F64(data), budget)
    } else if name == "uint8" {
        let contiguous = np.getattr("ascontiguousarray")?.call1((arr,))?;
        let typed = contiguous.cast::<PyArrayDyn<u8>>()?.try_readonly()?;
        let data = typed
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        io::from_image_array(&shape, ArrayData::U8(data), budget)
    } else {
        io::from_image_array(&shape, ArrayData::Other(&name), budget)
    };
    let loaded = result.map_err(|e| package_error(py, "PixelforgeError", &e.0))?;
    loaded_arrays(py, loaded)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(from_array, m)?)?;
    m.add("MAX_INPUT_PIXELS", io::MAX_INPUT_PIXELS)?;
    Ok(())
}
