//! Python wrappers of `pixelforge::io` (`pixelforge.io` delegates to these).
//!
//! Errors keep Python's classes: `PixelforgeError` with the library's message, `OSError`
//! (its errno subclass, e.g. `FileNotFoundError`) built as `OSError(errno, strerror, path)`
//! so `str(exc)` matches `os.open`'s, `IndexError` for an index beyond the palette. The
//! multi-frame notice is a `UserWarning` attributed to the caller of `pixelforge.io.load`.

use std::ffi::CString;
use std::path::PathBuf;

use numpy::ndarray::{Array2, Array3, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pixelforge::io::{self, ArrayData, IoError, Loaded, TransparentIndex};

use crate::color::py_shape;
use crate::mask::pixelforge_error;

fn io_error(py: Python<'_>, err: IoError) -> PyErr {
    match err {
        IoError::Pixelforge(e) => pixelforge_error(py, "PixelforgeError", &e.0),
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

type LoadedTuple<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<u8>>, String);

fn loaded_arrays<'py>(py: Python<'py>, loaded: Loaded) -> PyResult<LoadedTuple<'py>> {
    let (h, w) = (loaded.h, loaded.w);
    let rgb = Array3::from_shape_vec((h, w, 3), loaded.rgb)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let alpha = Array2::from_shape_vec((h, w), loaded.alpha)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((rgb.into_pyarray(py), alpha.into_pyarray(py), loaded.sha256))
}

/// The budget as the library takes it: a negative budget refuses everything.
fn budget(max_pixels: i64) -> u64 {
    u64::try_from(max_pixels).unwrap_or(0)
}

/// `io.load(path, max_pixels)`: `(rgb (H, W, 3) float64, alpha (H, W) uint8, sha256)`.
/// Emits `"{path} has {n} frames; using the first"` as a `UserWarning` (stacklevel 2).
#[pyfunction]
fn io_load<'py>(py: Python<'py>, path: PathBuf, max_pixels: i64) -> PyResult<LoadedTuple<'py>> {
    let result = py.detach(|| io::load(&path, budget(max_pixels)));
    let loaded = result.map_err(|e| io_error(py, e))?;
    for text in &loaded.warnings {
        let message =
            CString::new(text.as_str()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 2)?;
    }
    loaded_arrays(py, loaded)
}

/// `io.from_image(array, max_pixels)` for array inputs: `(rgb, alpha, sha256)`.
#[pyfunction]
fn io_from_image_array<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    max_pixels: i64,
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
    let loaded = result.map_err(|e| pixelforge_error(py, "PixelforgeError", &e.0))?;
    loaded_arrays(py, loaded)
}

/// `io.from_rgba(rgba, sha256)` for a C-contiguous (H, W, 4) uint8 array: `(rgb, alpha,
/// sha256)`.
#[pyfunction]
fn io_from_rgba<'py>(
    py: Python<'py>,
    rgba: &Bound<'py, PyAny>,
    sha256: String,
) -> PyResult<LoadedTuple<'py>> {
    let arr = rgba.cast::<PyArray3<u8>>().map_err(|_| {
        PyValueError::new_err("io_from_rgba: rgba must be a (H, W, 4) uint8 numpy.ndarray")
    })?;
    let shape = arr.shape().to_vec();
    if shape[2] != 4 {
        return Err(PyValueError::new_err(format!(
            "io_from_rgba: rgba must have shape (H, W, 4), got {}",
            py_shape(&shape)
        )));
    }
    let ro = arr.try_readonly()?;
    let data = ro
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let loaded = io::from_rgba(data, shape[0], shape[1], sha256);
    loaded_arrays(py, loaded)
}

/// `io.sha256_file(path)`.
#[pyfunction]
fn io_sha256_file(py: Python<'_>, path: PathBuf) -> PyResult<String> {
    py.detach(|| io::sha256_file(&path))
        .map_err(|e| io_error(py, e))
}

/// `io.open_regular(path)`: a file descriptor opened for reading (the caller wraps it with
/// `os.fdopen`); `PixelforgeError` for FIFOs, devices and directories.
#[pyfunction]
fn io_open_regular(py: Python<'_>, path: PathBuf) -> PyResult<i64> {
    let file = io::open_regular(&path).map_err(|e| io_error(py, e))?;
    #[cfg(unix)]
    {
        use std::os::fd::IntoRawFd;
        Ok(i64::from(file.into_raw_fd()))
    }
    #[cfg(not(unix))]
    {
        drop(file);
        Err(PyValueError::new_err(
            "io_open_regular: only supported on Unix",
        ))
    }
}

fn palette_triples(palette: &Bound<'_, PyAny>) -> PyResult<Vec<[u8; 3]>> {
    let arr = palette
        .cast::<PyArray2<u8>>()
        .map_err(|_| PyValueError::new_err("palette_rgb8 must be a (K, 3) uint8 numpy.ndarray"))?;
    let shape = arr.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "palette_rgb8 must have shape (K, 3), got {}",
            py_shape(shape)
        )));
    }
    let ro = arr.try_readonly()?;
    let flat = ro
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

/// `io.indices_to_rgba(indices, palette_rgb8)`: int64 indices of any shape (< 0
/// transparent) to RGBA8 of shape `indices.shape + (4,)`.
#[pyfunction]
fn io_indices_to_rgba<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    palette_rgb8: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let idx = indices
        .cast::<PyArrayDyn<i64>>()
        .map_err(|_| PyValueError::new_err("indices must be an int64 numpy.ndarray"))?;
    let mut shape = idx.shape().to_vec();
    let ro = idx.try_readonly()?;
    let data = ro
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let palette = palette_triples(palette_rgb8)?;
    let out = io::indices_to_rgba(data, &palette).map_err(PyIndexError::new_err)?;
    shape.push(4);
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

fn upscale_typed<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    arr: &Bound<'py, PyArrayDyn<T>>,
    factor: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = arr.shape().to_vec();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "upscale_nearest: the array must have at least 2 dimensions, got {}",
            py_shape(&shape)
        )));
    }
    let channels: usize = shape[2..].iter().product();
    let ro = arr.try_readonly()?;
    let data = ro
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = io::upscale_nearest(data, shape[0], shape[1], channels, factor);
    let mut out_shape = shape.clone();
    out_shape[0] *= factor;
    out_shape[1] *= factor;
    let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into_any())
}

/// `io.upscale_nearest(arr, factor)` for C-contiguous uint8, int64 or float64 arrays of at
/// least two dimensions: every pixel repeated `factor` times along the first two axes.
#[pyfunction]
fn io_upscale_nearest<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    factor: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let factor = usize::try_from(factor)
        .map_err(|_| PyValueError::new_err("negative dimensions are not allowed"))?;
    if let Ok(a) = arr.cast::<PyArrayDyn<u8>>() {
        upscale_typed(py, a, factor)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<i64>>() {
        upscale_typed(py, a, factor)
    } else if let Ok(a) = arr.cast::<PyArrayDyn<f64>>() {
        upscale_typed(py, a, factor)
    } else {
        Err(PyValueError::new_err(
            "upscale_nearest: the array must be uint8, int64 or float64",
        ))
    }
}

/// `io.save_png(path, indices, palette_rgb8, text, scale, transparent_index)`: `indices` a
/// C-contiguous (H, W) int64 array, `palette_rgb8` (K, 3) uint8, `text` a dict of str or
/// None, `transparent_index` "first" (anything else is "last", as in Python).
#[pyfunction]
#[pyo3(signature = (path, indices, palette_rgb8, text, scale, transparent_index))]
fn io_save_png(
    py: Python<'_>,
    path: PathBuf,
    indices: &Bound<'_, PyAny>,
    palette_rgb8: &Bound<'_, PyAny>,
    text: Option<&Bound<'_, PyDict>>,
    scale: i64,
    transparent_index: &str,
) -> PyResult<()> {
    let idx = indices
        .cast::<PyArray2<i64>>()
        .map_err(|_| PyValueError::new_err("indices must be a (H, W) int64 numpy.ndarray"))?;
    let shape = idx.shape().to_vec();
    let ro = idx.try_readonly()?;
    let data = ro
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let palette = palette_triples(palette_rgb8)?;
    let mut pairs = Vec::new();
    if let Some(text) = text {
        for (k, v) in text.iter() {
            pairs.push((k.extract::<String>()?, v.extract::<String>()?));
        }
    }
    let ti = if transparent_index == "first" {
        TransparentIndex::First
    } else {
        TransparentIndex::Last
    };
    let scale = usize::try_from(scale.max(1)).unwrap_or(1);
    py.detach(|| io::save_png(&path, data, shape[0], shape[1], &palette, &pairs, scale, ti))
        .map_err(|e| io_error(py, e))
}

/// `io.read_png_text(path)`: the text chunks as a dict (last value of a repeated key).
#[pyfunction]
fn io_read_png_text<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let pairs = py
        .detach(|| io::read_png_text(&path))
        .map_err(|e| io_error(py, e))?;
    let out = PyDict::new(py);
    for (k, v) in pairs {
        out.set_item(k, v)?;
    }
    Ok(out)
}

/// The constants `pixelforge.io` exports.
#[pyfunction]
fn io_constants(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("MAX_INPUT_PIXELS", io::MAX_INPUT_PIXELS)?;
    out.set_item(
        "INPUT_FORMATS",
        PyList::new(py, io::INPUT_FORMATS)?.to_tuple(),
    )?;
    out.set_item("META_CONFIG_KEY", io::META_CONFIG_KEY)?;
    out.set_item("META_INPUT_KEY", io::META_INPUT_KEY)?;
    out.set_item("META_PALETTE_SHA_KEY", io::META_PALETTE_SHA_KEY)?;
    out.set_item("META_PALETTE_KEY", io::META_PALETTE_KEY)?;
    Ok(out)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(io_load, m)?)?;
    m.add_function(wrap_pyfunction!(io_from_image_array, m)?)?;
    m.add_function(wrap_pyfunction!(io_from_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(io_sha256_file, m)?)?;
    m.add_function(wrap_pyfunction!(io_open_regular, m)?)?;
    m.add_function(wrap_pyfunction!(io_indices_to_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(io_upscale_nearest, m)?)?;
    m.add_function(wrap_pyfunction!(io_save_png, m)?)?;
    m.add_function(wrap_pyfunction!(io_read_png_text, m)?)?;
    m.add_function(wrap_pyfunction!(io_constants, m)?)?;
    Ok(())
}
