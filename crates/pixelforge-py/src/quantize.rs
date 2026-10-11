//! Python wrappers of `pixelforge::quantize` and the palette builders of
//! `pixelforge::palette` (median cut, hue ramps, `build`, the `.hex` text): argument checks
//! in the colour binding's style (each failure a `ValueError` naming the function and the
//! argument), then the library call with the GIL released. A Config arrives as
//! `config.to_dict()` and is rebuilt (and validated) in Rust.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pixelforge::{palette, quantize};

use crate::color::{py_shape, slice, triples, value_error};

/// The argument as a C-contiguous `T` array of exactly `ndim` dimensions.
fn array<'py, T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    want: &str,
    obj: &Bound<'py, PyAny>,
    ndim: usize,
) -> PyResult<(Vec<usize>, numpy::PyReadonlyArrayDyn<'py, T>)> {
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

/// A LAB image: float64 `(H, W, 3)`.
fn lab_image<'py>(
    func: &str,
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(usize, usize, numpy::PyReadonlyArrayDyn<'py, f64>)> {
    let (shape, arr) = triples::<f64>(func, name, "float64", obj, None)?;
    if shape.len() != 3 {
        return Err(value_error(
            func,
            format!("{name} must have shape (H, W, 3), got {}", py_shape(&shape)),
        ));
    }
    Ok((shape[0], shape[1], arr))
}

/// An optional bool mask whose shape is `(h, w)`.
fn mask_arg<'py>(
    func: &str,
    obj: Option<&Bound<'py, PyAny>>,
    h: usize,
    w: usize,
) -> PyResult<Option<numpy::PyReadonlyArrayDyn<'py, bool>>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    let (shape, arr) = array::<bool>(func, "mask", "bool", "(H, W)", obj, 2)?;
    if shape != [h, w] {
        return Err(value_error(
            func,
            format!(
                "mask must have the image's shape ({h}, {w}), got {}",
                py_shape(&shape)
            ),
        ));
    }
    Ok(Some(arr))
}

fn triples_of(data: &[f64]) -> Vec<[f64; 3]> {
    data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn palette_array<'py>(py: Python<'py>, colors: &[[f64; 3]]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = colors.iter().flatten().copied().collect();
    Array2::from_shape_vec((colors.len(), 3), flat)
        .expect("three values per colour")
        .into_pyarray(py)
}

/// The `n`×`n` Bayer threshold matrix (n a power of two), normalised to B/n² − 0.5.
#[pyfunction]
fn bayer(py: Python<'_>, n: usize) -> Bound<'_, PyArray2<f64>> {
    let (side, m) = quantize::bayer(n);
    Array2::from_shape_vec((side, side), m)
        .expect("side × side values")
        .into_pyarray(py)
}

/// `scipy.ndimage.uniform_filter(arr, size=3, mode="nearest")` for a float64 `(H, W)`
/// array, bit for bit.
#[pyfunction]
fn uniform_filter3<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "uniform_filter3";
    let (shape, a) = array::<f64>(F, "arr", "float64", "(H, W)", arr, 2)?;
    let data = slice(F, "arr", &a)?;
    let (h, w) = (shape[0], shape[1]);
    let out = py.detach(|| quantize::uniform_filter_3_nearest(data, h, w));
    Ok(Array2::from_shape_vec((h, w), out)
        .expect("h × w values")
        .into_pyarray(py))
}

/// Local LAB standard deviation over a 3×3 window: lab float64 `(H, W, 3)`, optional mask
/// bool `(H, W)` (True = opaque) -> float64 `(H, W)`.
#[pyfunction]
#[pyo3(signature = (lab, mask=None))]
fn local_std<'py>(
    py: Python<'py>,
    lab: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "local_std";
    let (h, w, larr) = lab_image(F, "lab", lab)?;
    let marr = mask_arg(F, mask, h, w)?;
    let l = slice(F, "lab", &larr)?;
    let m = marr.as_ref().map(|a| slice(F, "mask", a)).transpose()?;
    let out = py.detach(|| quantize::local_std(l, m, h, w));
    Ok(Array2::from_shape_vec((h, w), out)
        .expect("h × w values")
        .into_pyarray(py))
}

/// Per-pixel LAB dither offsets float64 `(H, W, 3)`, or None when dithering is off:
/// small_lab float64 `(H, W, 3)`, optional mask bool `(H, W)`, the Config as `to_dict()`.
#[pyfunction]
#[pyo3(signature = (small_lab, mask, config))]
fn dither_offsets<'py>(
    py: Python<'py>,
    small_lab: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Option<Bound<'py, PyArray3<f64>>>> {
    const F: &str = "dither_offsets";
    let (h, w, larr) = lab_image(F, "small_lab", small_lab)?;
    let marr = mask_arg(F, mask, h, w)?;
    let config = crate::config::build(py, config)?;
    let l = slice(F, "small_lab", &larr)?;
    let m = marr.as_ref().map(|a| slice(F, "mask", a)).transpose()?;
    let out = py.detach(|| quantize::dither_offsets(l, m, h, w, &config));
    Ok(out.map(|o| {
        Array3::from_shape_vec((h, w, 3), o)
            .expect("h × w × 3 values")
            .into_pyarray(py)
    }))
}

/// Index image int64 `(H, W)`: the nearest palette colour of every opaque pixel after the
/// dither offsets, -1 where transparent. small_lab float64 `(H, W, 3)`, mask bool `(H, W)`,
/// palette_lab float64 `(K, 3)`, the Config as `to_dict()`.
#[pyfunction]
fn quantize_run<'py>(
    py: Python<'py>,
    small_lab: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    palette_lab: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    const F: &str = "quantize_run";
    let (h, w, larr) = lab_image(F, "small_lab", small_lab)?;
    let marr = mask_arg(F, Some(mask), h, w)?
        .ok_or_else(|| value_error(F, "mask must not be None".into()))?;
    let (kshape, karr) = triples::<f64>(F, "palette_lab", "float64", palette_lab, Some(2))?;
    let config = crate::config::build(py, config)?;
    let l = slice(F, "small_lab", &larr)?;
    let m = slice(F, "mask", &marr)?;
    let k = slice(F, "palette_lab", &karr)?;
    if kshape[0] == 0 && m.iter().any(|&b| b) {
        return Err(value_error(F, "palette_lab must not be empty".into()));
    }
    let out = py.detach(|| quantize::run(l, m, k, h, w, &config));
    Ok(Array2::from_shape_vec((h, w), out)
        .expect("h × w values")
        .into_pyarray(py))
}

/// Median-cut palette (`palette.median_cut`): pixels float64 `(N, 3)` -> float64 `(K', 3)`,
/// K' <= k. ValueError for no pixels.
#[pyfunction]
fn median_cut<'py>(
    py: Python<'py>,
    pixels: &Bound<'py, PyAny>,
    k: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "median_cut";
    let (_, parr) = triples::<f64>(F, "pixels", "float64", pixels, Some(2))?;
    let p = triples_of(slice(F, "pixels", &parr)?);
    let out = py
        .detach(|| palette::median_cut(&p, k))
        .map_err(|e| PyValueError::new_err(e.0))?;
    Ok(palette_array(py, &out))
}

/// Hue-shifted ramps (`palette.regularize_ramps`): float64 `(K, 3)` -> float64 `(K, 3)`.
#[pyfunction]
fn regularize_ramps<'py>(
    py: Python<'py>,
    palette: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "regularize_ramps";
    let (_, parr) = triples::<f64>(F, "palette", "float64", palette, Some(2))?;
    let p = triples_of(slice(F, "palette", &parr)?);
    let out = py.detach(|| palette::regularize_ramps(&p));
    Ok(palette_array(py, &out))
}

/// `palette.build` without the mcda source: pixels float64 `(N, 3)`, the Config as
/// `to_dict()` -> palette float64 `(K, 3)` in LAB. ValueError for the mcda source, a palette
/// that cannot be loaded, or no pixels.
#[pyfunction]
fn palette_build<'py>(
    py: Python<'py>,
    pixels: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    const F: &str = "palette_build";
    let (_, parr) = triples::<f64>(F, "pixels", "float64", pixels, Some(2))?;
    let config = crate::config::build(py, config)?;
    let p = triples_of(slice(F, "pixels", &parr)?);
    let out = py
        .detach(|| palette::build(&p, &config))
        .map_err(|e| PyValueError::new_err(e.0))?;
    Ok(palette_array(py, &out))
}

/// The text of a `.hex` palette file (`palette.write_hex`): palette_rgb8 uint8 `(K, 3)`.
#[pyfunction]
fn palette_hex_text(palette_rgb8: &Bound<'_, PyAny>, transparent_first: bool) -> PyResult<String> {
    const F: &str = "palette_hex_text";
    let (_, arr) = triples::<u8>(F, "palette_rgb8", "uint8", palette_rgb8, Some(2))?;
    let colors: Vec<[u8; 3]> = slice(F, "palette_rgb8", &arr)?
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    Ok(palette::hex_text(&colors, transparent_first))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bayer, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_filter3, m)?)?;
    m.add_function(wrap_pyfunction!(local_std, m)?)?;
    m.add_function(wrap_pyfunction!(dither_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_run, m)?)?;
    m.add_function(wrap_pyfunction!(median_cut, m)?)?;
    m.add_function(wrap_pyfunction!(regularize_ramps, m)?)?;
    m.add_function(wrap_pyfunction!(palette_build, m)?)?;
    m.add_function(wrap_pyfunction!(palette_hex_text, m)?)?;
    Ok(())
}
