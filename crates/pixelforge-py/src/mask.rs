//! Python wrappers of `pixelforge::mask` and `pixelforge::layout` (the exact parts of
//! preprocessing), for tests/test_mask_native.py and tests/test_layout_native.py to compare
//! with `pixelforge/preprocess.py` until the integration step switches it over.
//!
//! Arguments are checked like `color.rs` does (dtype, shape, C-contiguity; a `ValueError`
//! naming the function and the argument). Images are float64 `(H, W, 3)`, masks bool
//! `(H, W)`, weights float64 `(H, W)`, alpha uint8 `(H, W)`. Errors the Python code raises
//! keep their class (`ConfigError`, `PixelforgeError`) and message.

use std::ffi::CString;

use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use pixelforge::{layout, mask};

use crate::color::{py_shape, shaped, slice, value_error};
use crate::config::build;

/// `pixelforge.errors.<class>(message)` as a Python exception.
pub(crate) fn pixelforge_error(py: Python<'_>, class: &str, message: &str) -> PyErr {
    match py
        .import("pixelforge.errors")
        .and_then(|m| m.getattr(class))
        .and_then(|cls| cls.call1((message,)))
    {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

/// The argument as a C-contiguous `T` array of `ndim` dimensions (the last of length 3 when
/// `rgb`): its shape and a readonly view.
pub(crate) fn array<'py, T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    obj: &Bound<'py, PyAny>,
    rgb: bool,
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
    let ok = if rgb {
        shape.len() == 3 && shape[2] == 3
    } else {
        shape.len() == 2
    };
    if !ok {
        let want = if rgb { "(H, W, 3)" } else { "(H, W)" };
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

/// `(h, w)` of an image and its mask, which must agree (and be non-empty when `nonempty`).
pub(crate) fn same_size(func: &str, a: &[usize], b: &[usize], names: (&str, &str)) -> PyResult<()> {
    if a[..2] != b[..2] {
        return Err(value_error(
            func,
            format!(
                "{} and {} must have the same height and width, got {} and {}",
                names.0,
                names.1,
                py_shape(a),
                py_shape(b)
            ),
        ));
    }
    Ok(())
}

pub(crate) fn nonempty(func: &str, shape: &[usize]) -> PyResult<()> {
    if shape[0] == 0 || shape[1] == 0 {
        return Err(value_error(
            func,
            format!("the image must not be empty, got {}", py_shape(shape)),
        ));
    }
    Ok(())
}

fn as_i64(v: Vec<i32>) -> Vec<i64> {
    v.into_iter().map(i64::from).collect()
}

/// `scipy.ndimage.distance_transform_edt(~mask, return_distances=True,
/// return_indices=True)` as `_fill_transparent` and `_unmix_fringe` call it: the transform of
/// the *background*, so the features are the `True` pixels of `mask`. Returns (distances
/// float64 (H, W), rows int64 (H, W), cols int64 (H, W)): for every pixel the distance to, and
/// the coordinates of, the `True` pixel scipy picks (bitwise scipy's, ties included). Without
/// any `True` pixel: rows -1, cols 0, as scipy.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn edt<'py>(
    py: Python<'py>,
    mask: &Bound<'py, PyAny>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i64>>,
    Bound<'py, PyArrayDyn<i64>>,
)> {
    const F: &str = "edt";
    let (shape, arr) = array::<bool>(F, "mask", "bool", mask, false)?;
    let m = slice(F, "mask", &arr)?;
    let (h, w) = (shape[0], shape[1]);
    let (d, r, c) = py.detach(|| mask::edt(m, h, w));
    Ok((
        shaped(py, &shape, d),
        shaped(py, &shape, as_i64(r)),
        shaped(py, &shape, as_i64(c)),
    ))
}

/// `preprocess._fill_transparent(rgb, mask)`.
#[pyfunction]
fn fill_transparent<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    const F: &str = "fill_transparent";
    let (rshape, rarr) = array::<f64>(F, "rgb", "float64", rgb, true)?;
    let (mshape, marr) = array::<bool>(F, "mask", "bool", mask, false)?;
    same_size(F, &rshape, &mshape, ("rgb", "mask"))?;
    let (r, m) = (slice(F, "rgb", &rarr)?, slice(F, "mask", &marr)?);
    let (h, w) = (mshape[0], mshape[1]);
    let out = py.detach(|| mask::fill_transparent(r, m, h, w));
    Ok(shaped(py, &rshape, out))
}

/// `ndimage.label(mask)` (4-connected): (labels int64 (H, W), number of components). Labels
/// are numbered in raster order of each component's first pixel.
#[pyfunction]
fn label4<'py>(
    py: Python<'py>,
    mask: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArrayDyn<i64>>, usize)> {
    const F: &str = "label4";
    let (shape, arr) = array::<bool>(F, "mask", "bool", mask, false)?;
    let m = slice(F, "mask", &arr)?;
    let (labels, n) = py.detach(|| mask::label4(m, shape[0], shape[1]));
    Ok((shaped(py, &shape, labels), n))
}

/// `ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=iterations)`.
#[pyfunction]
fn binary_dilation8<'py>(
    py: Python<'py>,
    mask: &Bound<'py, PyAny>,
    iterations: usize,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    const F: &str = "binary_dilation8";
    let (shape, arr) = array::<bool>(F, "mask", "bool", mask, false)?;
    let m = slice(F, "mask", &arr)?;
    let out = py.detach(|| mask::binary_dilation8(m, shape[0], shape[1], iterations));
    Ok(shaped(py, &shape, out))
}

/// `preprocess.flat_background(rgb, tolerance, fringe)`: the background mask or None.
#[pyfunction]
fn flat_background<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    tolerance: f64,
    fringe: usize,
) -> PyResult<Option<Bound<'py, PyArrayDyn<bool>>>> {
    const F: &str = "flat_background";
    let (shape, arr) = array::<f64>(F, "rgb", "float64", rgb, true)?;
    nonempty(F, &shape)?;
    let r = slice(F, "rgb", &arr)?;
    let (h, w) = (shape[0], shape[1]);
    let out = py.detach(|| mask::flat_background(r, h, w, tolerance, fringe));
    Ok(out.map(|m| shaped(py, &shape[..2], m)))
}

/// `preprocess._unmix_fringe(rgb, background, reference, passes)`; `reference` is the three
/// backdrop channels.
#[pyfunction]
fn unmix_fringe<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    background: &Bound<'py, PyAny>,
    reference: [f64; 3],
    passes: usize,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    const F: &str = "unmix_fringe";
    let (rshape, rarr) = array::<f64>(F, "rgb", "float64", rgb, true)?;
    let (bshape, barr) = array::<bool>(F, "background", "bool", background, false)?;
    same_size(F, &rshape, &bshape, ("rgb", "background"))?;
    let (r, b) = (slice(F, "rgb", &rarr)?, slice(F, "background", &barr)?);
    let (h, w) = (bshape[0], bshape[1]);
    let out = py.detach(|| mask::unmix_fringe(r, b, h, w, &reference, passes));
    Ok(shaped(py, &bshape, out))
}

/// `preprocess._subject(rgb, alpha, Config(**config))` without `remove_bg` (the caller
/// replaces alpha first): (mask bool, weight float64, background_keyed, (y0, y1, x0, x1)).
/// `PixelforgeError` for a fully transparent image, `ConfigError` for an invalid config.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn subject<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<bool>>,
    Bound<'py, PyArrayDyn<f64>>,
    bool,
    (usize, usize, usize, usize),
)> {
    const F: &str = "subject";
    let config = build(py, config)?;
    let (rshape, rarr) = array::<f64>(F, "rgb", "float64", rgb, true)?;
    let (ashape, aarr) = array::<u8>(F, "alpha", "uint8", alpha, false)?;
    same_size(F, &rshape, &ashape, ("rgb", "alpha"))?;
    nonempty(F, &ashape)?;
    let (r, a) = (slice(F, "rgb", &rarr)?, slice(F, "alpha", &aarr)?);
    let (h, w) = (ashape[0], ashape[1]);
    let s = py
        .detach(|| mask::subject(r, a, h, w, &config))
        .map_err(|e| pixelforge_error(py, "PixelforgeError", &e.0))?;
    Ok((
        shaped(py, &ashape, s.mask),
        shaped(py, &ashape, s.weight),
        s.background_keyed,
        s.crop,
    ))
}

/// `preprocess.resolve_layout(width, height, Config(**config))`: (out_width, out_height,
/// layout or None). Emits Python's `UserWarning` for a stretched aspect; `ConfigError` with
/// Python's message when tile_size exceeds the derived size.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn resolve_layout(
    py: Python<'_>,
    width: i64,
    height: i64,
    config: &Bound<'_, PyDict>,
) -> PyResult<(i64, i64, Option<(i64, i64, i64, i64)>)> {
    let config = build(py, config)?;
    let r = layout::resolve_layout(width, height, &config)
        .map_err(|e| pixelforge_error(py, "ConfigError", &e.0))?;
    for text in &r.warnings {
        let message = CString::new(text.as_str())
            .map_err(|e| value_error("resolve_layout", e.to_string()))?;
        PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 1)?;
    }
    Ok((r.out_width, r.out_height, r.layout))
}

/// `preprocess.outline_margin(Config(**config))`.
#[pyfunction]
fn outline_margin(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<i64> {
    Ok(layout::outline_margin(&build(py, config)?))
}

/// `preprocess.fit_mode(Config(**config))`.
#[pyfunction]
fn fit_mode(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
    Ok(layout::fit_mode(&build(py, config)?).map(str::to_string))
}

/// `preprocess.fit_crop(width, height, out_width, out_height)`: (y0, y1, x0, x1).
#[pyfunction]
fn fit_crop(
    width: i64,
    height: i64,
    out_width: i64,
    out_height: i64,
) -> PyResult<(i64, i64, i64, i64)> {
    if width <= 0 || height <= 0 || out_width <= 0 || out_height <= 0 {
        return Err(value_error("fit_crop", "sizes must be positive".into()));
    }
    Ok(layout::fit_crop(width, height, out_width, out_height))
}

/// `preprocess.prereduce_factor(h, w, out_h, out_w, max_ratio)`.
#[pyfunction]
fn prereduce_factor(h: i64, w: i64, out_h: i64, out_w: i64, max_ratio: i64) -> PyResult<i64> {
    if out_h <= 0 || out_w <= 0 {
        return Err(value_error(
            "prereduce_factor",
            "output sizes must be positive".into(),
        ));
    }
    Ok(layout::prereduce_factor(h, w, out_h, out_w, max_ratio))
}

/// `preprocess.prereduce(rgb, mask, factor, weight)`: (rgb, mask) without a weight, (rgb,
/// mask, weight) with one.
#[pyfunction]
#[pyo3(signature = (rgb, mask, factor, weight=None))]
fn prereduce<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    factor: usize,
    weight: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    const F: &str = "prereduce";
    if factor == 0 {
        return Err(value_error(F, "factor must be >= 1".into()));
    }
    let (rshape, rarr) = array::<f64>(F, "rgb", "float64", rgb, true)?;
    let (mshape, marr) = array::<bool>(F, "mask", "bool", mask, false)?;
    same_size(F, &rshape, &mshape, ("rgb", "mask"))?;
    let (r, m) = (slice(F, "rgb", &rarr)?, slice(F, "mask", &marr)?);
    let warr = match weight {
        Some(obj) => {
            let (wshape, warr) = array::<f64>(F, "weight", "float64", obj, false)?;
            same_size(F, &wshape, &mshape, ("weight", "mask"))?;
            Some(warr)
        }
        None => None,
    };
    let wt = match &warr {
        Some(a) => Some(slice(F, "weight", a)?),
        None => None,
    };
    let (h, w) = (mshape[0], mshape[1]);
    let out = py.detach(|| layout::prereduce(r, m, wt, h, w, factor));
    let rgb = shaped(py, &[out.h, out.w, 3], out.rgb);
    let mask = shaped(py, &[out.h, out.w], out.mask);
    match out.weight {
        Some(wv) => {
            let weight = shaped(py, &[out.h, out.w], wv);
            PyTuple::new(py, [rgb.into_any(), mask.into_any(), weight.into_any()])
        }
        None => PyTuple::new(py, [rgb.into_any(), mask.into_any()]),
    }
}

type Planes<'py> = (
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<bool>>,
    Bound<'py, PyArrayDyn<f64>>,
);

/// The three planes of `repeat` / `wrap_pad`, checked.
#[allow(clippy::type_complexity)]
fn planes<'py>(
    func: &str,
    rgb: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    weight: &Bound<'py, PyAny>,
) -> PyResult<(
    Vec<usize>,
    PyReadonlyArrayDyn<'py, f64>,
    PyReadonlyArrayDyn<'py, bool>,
    PyReadonlyArrayDyn<'py, f64>,
)> {
    let (rshape, rarr) = array::<f64>(func, "rgb", "float64", rgb, true)?;
    let (mshape, marr) = array::<bool>(func, "mask", "bool", mask, false)?;
    let (wshape, warr) = array::<f64>(func, "weight", "float64", weight, false)?;
    same_size(func, &rshape, &mshape, ("rgb", "mask"))?;
    same_size(func, &wshape, &mshape, ("weight", "mask"))?;
    Ok((mshape, rarr, marr, warr))
}

/// The integer repeat factor `run` uses for a `w`×`h` input and an `out_w`×`out_h` output;
/// `PixelforgeError` with Python's message when the repeated input would be too large.
#[pyfunction]
fn repeat_factor(py: Python<'_>, h: usize, w: usize, out_w: i64, out_h: i64) -> PyResult<usize> {
    if h == 0 || w == 0 {
        return Err(value_error(
            "repeat_factor",
            "sizes must be positive".into(),
        ));
    }
    layout::repeat_factor(h, w, out_w, out_h)
        .map_err(|e| pixelforge_error(py, "PixelforgeError", &e.0))
}

/// `np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)` of rgb, mask and weight.
#[pyfunction]
fn repeat<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    weight: &Bound<'py, PyAny>,
    factor: usize,
) -> PyResult<Planes<'py>> {
    const F: &str = "repeat";
    let (shape, rarr, marr, warr) = planes(F, rgb, mask, weight)?;
    let (r, m, wt) = (
        slice(F, "rgb", &rarr)?,
        slice(F, "mask", &marr)?,
        slice(F, "weight", &warr)?,
    );
    let (h, w) = (shape[0], shape[1]);
    let (ro, mo, wo) = py.detach(|| layout::repeat(r, m, wt, h, w, factor));
    let (oh, ow) = (h * factor, w * factor);
    Ok((
        shaped(py, &[oh, ow, 3], ro),
        shaped(py, &[oh, ow], mo),
        shaped(py, &[oh, ow], wo),
    ))
}

/// `np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode="wrap")` of rgb, mask and weight.
#[pyfunction]
fn wrap_pad<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    weight: &Bound<'py, PyAny>,
    pad_y: usize,
    pad_x: usize,
) -> PyResult<Planes<'py>> {
    const F: &str = "wrap_pad";
    let (shape, rarr, marr, warr) = planes(F, rgb, mask, weight)?;
    nonempty(F, &shape)?;
    let (r, m, wt) = (
        slice(F, "rgb", &rarr)?,
        slice(F, "mask", &marr)?,
        slice(F, "weight", &warr)?,
    );
    let (h, w) = (shape[0], shape[1]);
    let (ro, mo, wo) = py.detach(|| layout::wrap_pad(r, m, wt, h, w, pad_y, pad_x));
    let (oh, ow) = (h + 2 * pad_y, w + 2 * pad_x);
    Ok((
        shaped(py, &[oh, ow, 3], ro),
        shaped(py, &[oh, ow], mo),
        shaped(py, &[oh, ow], wo),
    ))
}

/// The seamless wrap padding (pad_y, pad_x) in input pixels for an h×w input downscaled to
/// inner_h×inner_w.
#[pyfunction]
fn seamless_pads(h: usize, w: usize, inner_h: i64, inner_w: i64) -> PyResult<(usize, usize)> {
    if inner_h <= 0 || inner_w <= 0 {
        return Err(value_error(
            "seamless_pads",
            "inner sizes must be positive".into(),
        ));
    }
    Ok(layout::seamless_pads(h, w, inner_h, inner_w))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edt, m)?)?;
    m.add_function(wrap_pyfunction!(fill_transparent, m)?)?;
    m.add_function(wrap_pyfunction!(label4, m)?)?;
    m.add_function(wrap_pyfunction!(binary_dilation8, m)?)?;
    m.add_function(wrap_pyfunction!(flat_background, m)?)?;
    m.add_function(wrap_pyfunction!(unmix_fringe, m)?)?;
    m.add_function(wrap_pyfunction!(subject, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_layout, m)?)?;
    m.add_function(wrap_pyfunction!(outline_margin, m)?)?;
    m.add_function(wrap_pyfunction!(fit_mode, m)?)?;
    m.add_function(wrap_pyfunction!(fit_crop, m)?)?;
    m.add_function(wrap_pyfunction!(prereduce_factor, m)?)?;
    m.add_function(wrap_pyfunction!(prereduce, m)?)?;
    m.add_function(wrap_pyfunction!(repeat_factor, m)?)?;
    m.add_function(wrap_pyfunction!(repeat, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_pad, m)?)?;
    m.add_function(wrap_pyfunction!(seamless_pads, m)?)?;
    Ok(())
}
