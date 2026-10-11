//! Python wrapper of `pixelforge::boxfilter::run` (`pixelforge.downscale.box.run`), and the
//! input checks every downscaler binding shares: lab float64 `(H, W, 3)`, mask bool
//! `(H, W)`, weight float64 `(H, W)` or None, all C-contiguous, and an output size of at
//! least 1 × 1. Failures are `ValueError`s naming the function and the argument; the
//! all-transparent input is Python's `PixelforgeError`.

use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pixelforge::boxfilter;

use crate::color::{slice, value_error};
use crate::config::build;
use crate::downscale::downscaled_to_py;
use crate::mask::{array, pixelforge_error, same_size};

/// The checked arrays of a downscaler call.
pub(crate) struct Inputs<'py> {
    pub lab: PyReadonlyArrayDyn<'py, f64>,
    pub mask: PyReadonlyArrayDyn<'py, bool>,
    pub weight: Option<PyReadonlyArrayDyn<'py, f64>>,
    pub h: usize,
    pub w: usize,
}

impl Inputs<'_> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn slices(&self, func: &str) -> PyResult<(&[f64], &[bool], Option<&[f64]>)> {
        Ok((
            slice(func, "lab", &self.lab)?,
            slice(func, "mask", &self.mask)?,
            self.weight
                .as_ref()
                .map(|w| slice(func, "weight", w))
                .transpose()?,
        ))
    }
}

pub(crate) fn inputs<'py>(
    func: &str,
    lab: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    out_width: usize,
    out_height: usize,
    weight: Option<&Bound<'py, PyAny>>,
) -> PyResult<Inputs<'py>> {
    let (lshape, lab) = array::<f64>(func, "lab", "float64", lab, true)?;
    let (mshape, mask) = array::<bool>(func, "mask", "bool", mask, false)?;
    same_size(func, &lshape, &mshape, ("lab", "mask"))?;
    let weight = match weight {
        None => None,
        Some(w) => {
            let (wshape, warr) = array::<f64>(func, "weight", "float64", w, false)?;
            same_size(func, &mshape, &wshape, ("mask", "weight"))?;
            Some(warr)
        }
    };
    if out_width == 0 || out_height == 0 {
        return Err(value_error(
            func,
            format!("the output size must be at least 1x1, got {out_width}x{out_height}"),
        ));
    }
    Ok(Inputs {
        lab,
        mask,
        weight,
        h: mshape[0],
        w: mshape[1],
    })
}

/// `box.run(lab, mask, out_width, out_height, Config(**config), weight)` as a dict with the
/// fields of `Downscaled` (stats `{"iterations": 1}`).
#[pyfunction]
#[pyo3(signature = (lab, mask, out_width, out_height, config, weight=None))]
fn box_run<'py>(
    py: Python<'py>,
    lab: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    out_width: usize,
    out_height: usize,
    config: &Bound<'py, PyDict>,
    weight: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    const F: &str = "box_run";
    build(py, config)?; // the box filter has no settings; the Config is still validated
    let inp = inputs(F, lab, mask, out_width, out_height, weight)?;
    let (l, m, w) = inp.slices(F)?;
    let (h, wd) = (inp.h, inp.w);
    let out = py
        .detach(|| boxfilter::run(l, m, h, wd, out_width, out_height, w))
        .map_err(|e| pixelforge_error(py, "PixelforgeError", &e.0))?;
    downscaled_to_py(py, out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(box_run, m)?)?;
    Ok(())
}
