//! Python wrappers of `pixelforge::filters` (scipy.ndimage's 1-D filters, the 3×3 median,
//! skimage's bilateral) and `pixelforge::enhance` (percentile, levels, the enhance pass), for
//! the parity tests tests/test_filters_native.py and tests/test_enhance_native.py. Argument
//! checks in the colour binding's style (each failure a `ValueError` naming the function and
//! the argument), then the library call with the GIL released. A Config arrives as
//! `config.to_dict()` and is rebuilt (and validated) in Rust.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods};
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pixelforge::filters::{self, Mode};
use pixelforge::{enhance as en, parallel};

use crate::color::{py_shape, slice, triples, value_error};

/// The argument as a C-contiguous `T` array of exactly `ndim` dimensions.
fn array<'py, T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    want: &str,
    obj: &Bound<'py, PyAny>,
    ndim: usize,
) -> PyResult<(Vec<usize>, PyReadonlyArrayDyn<'py, T>)> {
    let arr = obj.cast::<PyArrayDyn<T>>().map_err(|_| {
        let got = obj
            .getattr("dtype")
            .map(|d| d.to_string())
            .unwrap_or_else(|_| obj.get_type().to_string());
        value_error(
            func,
            format!("{name} must be a {dtype} numpy.ndarray, got {got}"),
        )
    })?;
    let shape = arr.shape().to_vec();
    if shape.len() != ndim {
        return Err(value_error(
            func,
            format!("{name} must have shape {want}, got {}", py_shape(&shape)),
        ));
    }
    if !arr.is_c_contiguous() {
        return Err(value_error(
            func,
            format!("{name} must be C-contiguous (use numpy.ascontiguousarray)"),
        ));
    }
    let ro = arr
        .try_readonly()
        .map_err(|e| value_error(func, format!("cannot read {name}: {e}")))?;
    Ok((shape, ro))
}

/// An RGB image: float64 `(H, W, 3)`.
fn rgb_image<'py>(
    func: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(usize, usize, PyReadonlyArrayDyn<'py, f64>)> {
    let (shape, arr) = triples::<f64>(func, "rgb", "float64", obj, None)?;
    if shape.len() != 3 {
        return Err(value_error(
            func,
            format!("rgb must have shape (H, W, 3), got {}", py_shape(&shape)),
        ));
    }
    Ok((shape[0], shape[1], arr))
}

/// An `(H, W)` array of the image's shape.
fn plane<'py, T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    obj: &Bound<'py, PyAny>,
    h: usize,
    w: usize,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    let (shape, arr) = array::<T>(func, name, dtype, "(H, W)", obj, 2)?;
    if shape != [h, w] {
        return Err(value_error(
            func,
            format!(
                "{name} must have the image's shape ({h}, {w}), got {}",
                py_shape(&shape)
            ),
        ));
    }
    Ok(arr)
}

fn mode_arg(func: &str, mode: &str) -> PyResult<Mode> {
    Mode::parse(mode).ok_or_else(|| {
        value_error(
            func,
            format!("mode must be 'nearest' or 'wrap', got {mode:?}"),
        )
    })
}

fn axis_arg(func: &str, axis: usize) -> PyResult<usize> {
    if axis > 1 {
        return Err(value_error(
            func,
            format!("axis must be 0 or 1, got {axis}"),
        ));
    }
    Ok(axis)
}

fn grid<'py>(py: Python<'py>, h: usize, w: usize, out: Vec<f64>) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((h, w), out)
        .expect("h × w values")
        .into_pyarray(py)
}

fn image<'py>(py: Python<'py>, h: usize, w: usize, out: Vec<f64>) -> Bound<'py, PyArray3<f64>> {
    Array3::from_shape_vec((h, w, 3), out)
        .expect("h × w × 3 values")
        .into_pyarray(py)
}

/// `scipy.ndimage.uniform_filter1d(arr, size, axis=axis, mode=mode)` for a float64 `(H, W)`
/// array, mode "nearest" or "wrap", bit for bit.
#[pyfunction]
fn uniform_filter1d<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    size: usize,
    axis: usize,
    mode: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "uniform_filter1d";
    let (shape, a) = array::<f64>(F, "arr", "float64", "(H, W)", arr, 2)?;
    let (mode, axis) = (mode_arg(F, mode)?, axis_arg(F, axis)?);
    if size == 0 {
        return Err(value_error(F, "size must be at least 1".into()));
    }
    let data = slice(F, "arr", &a)?;
    let (h, w) = (shape[0], shape[1]);
    let out = py.detach(|| filters::uniform_filter1d(data, h, w, size, axis, mode));
    Ok(grid(py, h, w, out))
}

/// `scipy.ndimage.gaussian_filter1d(arr, sigma, axis=axis, mode=mode)` (truncate 4) for a
/// float64 `(H, W)` array, mode "nearest" or "wrap".
#[pyfunction]
fn gaussian_filter1d<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    sigma: f64,
    axis: usize,
    mode: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "gaussian_filter1d";
    let (shape, a) = array::<f64>(F, "arr", "float64", "(H, W)", arr, 2)?;
    let (mode, axis) = (mode_arg(F, mode)?, axis_arg(F, axis)?);
    if sigma.is_nan() || sigma < 0.0 {
        return Err(value_error(F, format!("sigma must be >= 0, got {sigma}")));
    }
    let data = slice(F, "arr", &a)?;
    let (h, w) = (shape[0], shape[1]);
    let out = py.detach(|| filters::gaussian_filter1d(data, h, w, sigma, axis, mode));
    Ok(grid(py, h, w, out))
}

/// scipy's `_gaussian_kernel1d(sigma, 0, int(4 * sigma + 0.5))`, with `libm::exp`.
#[pyfunction]
fn gaussian_kernel1d(py: Python<'_>, sigma: f64) -> Bound<'_, PyArray1<f64>> {
    Array1::from_vec(filters::gaussian_kernel1d(sigma)).into_pyarray(py)
}

/// `scipy.ndimage.median_filter(arr, size=3, mode="nearest")` for a float64 `(H, W)` array,
/// bit for bit.
#[pyfunction]
fn median_filter3<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "median_filter3";
    let (shape, a) = array::<f64>(F, "arr", "float64", "(H, W)", arr, 2)?;
    let data = slice(F, "arr", &a)?;
    let (h, w) = (shape[0], shape[1]);
    let out = py.detach(|| filters::median_filter3_nearest(data, h, w));
    Ok(grid(py, h, w, out))
}

/// `preprocess.denoise(rgb, config)`: rgb float64 `(H, W, 3)`, the Config as `to_dict()` ->
/// float64 `(H, W, 3)` (a copy for `denoise="none"`). The bilateral runs on
/// `thread_count()` threads; the result does not depend on it.
#[pyfunction]
fn denoise<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    const F: &str = "denoise";
    let (h, w, a) = rgb_image(F, rgb)?;
    let config = crate::config::build(py, config)?;
    let data = slice(F, "rgb", &a)?;
    let out = py.detach(|| match config.denoise.as_str() {
        "median" => filters::denoise_median(data, h, w),
        "bilateral" => filters::denoise_bilateral(
            data,
            h,
            w,
            config.denoise_sigma_color,
            config.denoise_sigma_spatial,
            parallel::thread_count(),
        ),
        _ => data.to_vec(),
    });
    Ok(image(py, h, w, out))
}

/// `np.percentile(values, qs)` (linear) for a float64 1-D array, bit for bit.
#[pyfunction]
fn percentile(values: &Bound<'_, PyAny>, qs: Vec<f64>) -> PyResult<Vec<f64>> {
    const F: &str = "percentile";
    let (_, a) = array::<f64>(F, "values", "float64", "(N,)", values, 1)?;
    let data = slice(F, "values", &a)?;
    Ok(en::percentile(data, &qs))
}

/// `preprocess.enhance_levels(values, config)` -> (black, white).
#[pyfunction]
fn enhance_levels(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    config: &Bound<'_, PyDict>,
) -> PyResult<(f64, f64)> {
    const F: &str = "enhance_levels";
    let (_, a) = array::<f64>(F, "values", "float64", "(N,)", values, 1)?;
    let config = crate::config::build(py, config)?;
    Ok(en::enhance_levels(slice(F, "values", &a)?, &config))
}

/// `preprocess.subject_brightness` after `_subject`: the largest channel of the subject's
/// level pixels. rgb float64 `(H, W, 3)`, mask bool `(H, W)`, weight float64 `(H, W)` ->
/// float64 `(N,)`.
#[pyfunction]
fn subject_brightness_from<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    weight: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    const F: &str = "subject_brightness_from";
    let (h, w, a) = rgb_image(F, rgb)?;
    let m = plane::<bool>(F, "mask", "bool", mask, h, w)?;
    let wt = plane::<f64>(F, "weight", "float64", weight, h, w)?;
    let (a, m, wt) = (
        slice(F, "rgb", &a)?,
        slice(F, "mask", &m)?,
        slice(F, "weight", &wt)?,
    );
    let out = py.detach(|| en::subject_brightness(a, m, wt));
    Ok(Array1::from_vec(out).into_pyarray(py))
}

/// `preprocess._stretch(values, lo, hi)` for a float64 1-D array, bit for bit.
#[pyfunction]
fn enhance_stretch<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    lo: f64,
    hi: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    const F: &str = "enhance_stretch";
    let (_, a) = array::<f64>(F, "values", "float64", "(N,)", values, 1)?;
    let out = en::stretch(slice(F, "values", &a)?, lo, hi);
    Ok(Array1::from_vec(out).into_pyarray(py))
}

/// `preprocess._gaussian(arr, (sigma_y, sigma_x), mode)` for a float64 `(H, W)` array: the
/// Gaussian kernel below 4 pixels, three running means from there (bitwise on that path).
#[pyfunction]
fn enhance_gaussian<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    sigma: (f64, f64),
    mode: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "enhance_gaussian";
    let (shape, a) = array::<f64>(F, "arr", "float64", "(H, W)", arr, 2)?;
    let mode = mode_arg(F, mode)?;
    if sigma.0.is_nan() || sigma.0 < 0.0 || sigma.1.is_nan() || sigma.1 < 0.0 {
        return Err(value_error(F, format!("sigma must be >= 0, got {sigma:?}")));
    }
    let data = slice(F, "arr", &a)?;
    let (h, w) = (shape[0], shape[1]);
    let out = py.detach(|| en::gaussian(data, h, w, sigma, mode));
    Ok(grid(py, h, w, out))
}

/// `preprocess.enhance(rgb, mask, cell, config, weight, wrap)`: rgb float64 `(H, W, 3)`,
/// mask bool `(H, W)`, cell (y, x) input pixels per output pixel, the Config as `to_dict()`,
/// weight float64 `(H, W)` or None -> float64 `(H, W, 3)`.
#[pyfunction]
#[pyo3(signature = (rgb, mask, cell, config, weight=None, wrap=false))]
fn enhance<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    cell: (f64, f64),
    config: &Bound<'py, PyDict>,
    weight: Option<&Bound<'py, PyAny>>,
    wrap: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    const F: &str = "enhance";
    let (h, w, a) = rgb_image(F, rgb)?;
    let m = plane::<bool>(F, "mask", "bool", mask, h, w)?;
    let wt = weight
        .filter(|o| !o.is_none())
        .map(|o| plane::<f64>(F, "weight", "float64", o, h, w))
        .transpose()?;
    let config = crate::config::build(py, config)?;
    let a = slice(F, "rgb", &a)?;
    let m = slice(F, "mask", &m)?;
    let wt = wt.as_ref().map(|x| slice(F, "weight", x)).transpose()?;
    let out = py.detach(|| en::enhance(a, m, h, w, cell, &config, wt, wrap));
    Ok(image(py, h, w, out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uniform_filter1d, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter1d, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_kernel1d, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter3, m)?)?;
    m.add_function(wrap_pyfunction!(denoise, m)?)?;
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(enhance_levels, m)?)?;
    m.add_function(wrap_pyfunction!(subject_brightness_from, m)?)?;
    m.add_function(wrap_pyfunction!(enhance_stretch, m)?)?;
    m.add_function(wrap_pyfunction!(enhance_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(enhance, m)?)?;
    Ok(())
}
