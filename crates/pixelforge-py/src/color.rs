//! Python wrappers of `pixelforge::color`: argument checking (dtype, shape, contiguity,
//! each failure a `ValueError` naming the function and the argument), then the library call
//! with the GIL released.
//!
//! The conversions accept any leading shape `(..., 3)` and return the same shape;
//! `nearest_index` takes `(N, 3)` pixels and a `(K, 3)` palette.

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pixelforge::color;

fn value_error(func: &str, msg: String) -> PyErr {
    PyValueError::new_err(format!("{func}: {msg}"))
}

/// A shape as numpy prints it: `(2, 4)`, `(5,)`, `()`.
fn py_shape(shape: &[usize]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => {
            let parts: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    }
}

/// The argument as a C-contiguous `T` array whose last axis has length 3 (and, with
/// `ndim`, exactly that many dimensions): its shape and a copy-free readonly view.
fn triples<'py, T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    obj: &Bound<'py, PyAny>,
    ndim: Option<usize>,
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
    let want = match ndim {
        Some(2) => "(N, 3)",
        _ => "(..., 3)",
    };
    if shape.last() != Some(&3) || ndim.is_some_and(|n| n != shape.len()) {
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

fn slice<'a, T: numpy::Element>(
    func: &str,
    name: &str,
    arr: &'a numpy::PyReadonlyArrayDyn<'_, T>,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|e| value_error(func, format!("{name}: {e}")))
}

fn shaped<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<T>,
) -> Bound<'py, PyArrayDyn<T>> {
    ArrayD::from_shape_vec(IxDyn(shape), data)
        .expect("output length matches the input shape")
        .into_pyarray(py)
}

/// sRGB in [0, 1], float64 `(..., 3)` -> CIELAB (D65), float64 of the same shape.
#[pyfunction]
pub fn rgb_to_lab<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let (shape, arr) = triples::<f64>("rgb_to_lab", "rgb", "float64", rgb, None)?;
    let data = slice("rgb_to_lab", "rgb", &arr)?;
    let out = py.detach(|| color::rgb_to_lab(data));
    Ok(shaped(py, &shape, out))
}

/// CIELAB float64 `(..., 3)` -> sRGB in [0, 1] (clipped), float64 of the same shape.
#[pyfunction]
pub fn lab_to_rgb<'py>(
    py: Python<'py>,
    lab: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let (shape, arr) = triples::<f64>("lab_to_rgb", "lab", "float64", lab, None)?;
    let data = slice("lab_to_rgb", "lab", &arr)?;
    let out = py.detach(|| color::lab_to_rgb(data));
    Ok(shaped(py, &shape, out))
}

/// 8-bit sRGB, uint8 `(..., 3)` -> CIELAB, float64 of the same shape.
#[pyfunction]
pub fn rgb8_to_lab<'py>(
    py: Python<'py>,
    rgb8: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let (shape, arr) = triples::<u8>("rgb8_to_lab", "rgb8", "uint8", rgb8, None)?;
    let data = slice("rgb8_to_lab", "rgb8", &arr)?;
    let out = py.detach(|| color::rgb8_to_lab(data));
    Ok(shaped(py, &shape, out))
}

/// CIELAB float64 `(..., 3)` -> 8-bit sRGB (rounded half to even), uint8 of the same shape.
#[pyfunction]
pub fn lab_to_rgb8<'py>(
    py: Python<'py>,
    lab: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let (shape, arr) = triples::<f64>("lab_to_rgb8", "lab", "float64", lab, None)?;
    let data = slice("lab_to_rgb8", "lab", &arr)?;
    let out = py.detach(|| color::lab_to_rgb8(data));
    Ok(shaped(py, &shape, out))
}

/// Index of the nearest palette colour (CIE76, ties to the lowest index) for every row:
/// pixels float64 `(N, 3)`, palette float64 `(K, 3)` -> int64 `(N,)`.
#[pyfunction]
pub fn nearest_index<'py>(
    py: Python<'py>,
    pixels: &Bound<'py, PyAny>,
    palette: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    const F: &str = "nearest_index";
    let (pshape, parr) = triples::<f64>(F, "pixels", "float64", pixels, Some(2))?;
    let (kshape, karr) = triples::<f64>(F, "palette", "float64", palette, Some(2))?;
    if kshape[0] == 0 && pshape[0] != 0 {
        return Err(value_error(F, "palette must not be empty".into()));
    }
    let p = slice(F, "pixels", &parr)?;
    let k = slice(F, "palette", &karr)?;
    let out = py.detach(|| color::nearest_index(p, k));
    Ok(out.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rgb_to_lab, m)?)?;
    m.add_function(wrap_pyfunction!(lab_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(rgb8_to_lab, m)?)?;
    m.add_function(wrap_pyfunction!(lab_to_rgb8, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_index, m)?)?;
    Ok(())
}
