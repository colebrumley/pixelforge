//! Python wrappers of the gerstner kernels in `pixelforge::gerstner`: argument checking
//! (dtype, shape, contiguity, value ranges, each failure a `ValueError`), then the library
//! call with the GIL released.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pixelforge::annealer::{self, PaletteAnnealer, Tunables};
use pixelforge::gerstner;
use pixelforge::gerstner_run as control;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_error(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

fn contiguous<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    arr: &'a numpy::PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|_| value_error(format!("{name} must be a C-contiguous array")))
}

fn check_shape(name: &str, got: &[usize], expected: &[usize]) -> PyResult<()> {
    if got != expected {
        return Err(value_error(format!(
            "{name} has shape {got:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

/// One modified-SLIC assignment pass over the (hi, wi) pixel grid; the body of
/// `gerstner._assign` fused into one loop per pixel.
///
/// For each pixel the candidates are the superpixels of grid cells
/// (clip(home_y + dy), clip(home_x + dx)), dy in −r..=r outer, dx inner, compared with a
/// strict `<` so the first one visited wins ties. Centers that may not be chosen carry
/// center_x = +inf. With `windowed`, a candidate also needs |px − cx| ≤ rx and
/// |py − cy| ≤ ry. The distance is coef·‖Δxy‖ (+ ‖Δlab‖ when `channels`/`sp_color` are
/// given), evaluated in the reference's operation order. Returns −1 where nothing qualified.
#[pyfunction]
#[pyo3(signature = (home_y, home_x, px, py, center_x, center_y, radius, rx, ry, coef,
                    windowed, channels=None, sp_color=None))]
#[allow(clippy::too_many_arguments)]
pub fn slic_assign<'py>(
    gil: Python<'py>,
    home_y: PyReadonlyArray1<'py, i64>,
    home_x: PyReadonlyArray1<'py, i64>,
    px: PyReadonlyArray1<'py, f64>,
    py: PyReadonlyArray1<'py, f64>,
    center_x: PyReadonlyArray2<'py, f64>,
    center_y: PyReadonlyArray2<'py, f64>,
    radius: i64,
    rx: f64,
    ry: f64,
    coef: f64,
    windowed: bool,
    channels: Option<PyReadonlyArray3<'py, f64>>,
    sp_color: Option<PyReadonlyArray3<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let hi = home_y.len();
    let wi = home_x.len();
    let (ho, wo) = (center_x.shape()[0], center_x.shape()[1]);
    if ho == 0 || wo == 0 {
        return Err(value_error("center_x must not be empty".into()));
    }
    check_shape("py", py.shape(), &[hi])?;
    check_shape("px", px.shape(), &[wi])?;
    check_shape("center_y", center_y.shape(), &[ho, wo])?;
    if !(0..=1 << 20).contains(&radius) {
        return Err(value_error(format!(
            "radius must be in 0..=2^20, got {radius}"
        )));
    }
    if ho.checked_mul(wo).is_none_or(|n| n > u32::MAX as usize) {
        return Err(value_error(format!("{ho}x{wo} superpixels is too many")));
    }
    let home_y = contiguous(&home_y, "home_y")?;
    let home_x = contiguous(&home_x, "home_x")?;
    let px = contiguous(&px, "px")?;
    let py = contiguous(&py, "py")?;
    let cx = contiguous(&center_x, "center_x")?;
    let cy = contiguous(&center_y, "center_y")?;

    let color = match (&channels, &sp_color) {
        (Some(ch), Some(sp)) => {
            check_shape("channels", ch.shape(), &[3, hi, wi])?;
            check_shape("sp_color", sp.shape(), &[ho, wo, 3])?;
            Some((contiguous(ch, "channels")?, contiguous(sp, "sp_color")?))
        }
        (None, None) => None,
        _ => {
            return Err(value_error(
                "channels and sp_color must be given together".into(),
            ))
        }
    };

    let (ho_i, wo_i) = (ho as i64, wo as i64);
    if let Some(&v) = home_y.iter().find(|&&v| v < 0 || v >= ho_i) {
        return Err(value_error(format!("home_y holds {v}, outside 0..{ho}")));
    }
    if let Some(&v) = home_x.iter().find(|&&v| v < 0 || v >= wo_i) {
        return Err(value_error(format!("home_x holds {v}, outside 0..{wo}")));
    }
    let best = gil.detach(|| {
        gerstner::slic_assign(
            home_y, home_x, px, py, cx, cy, ho, wo, radius, rx, ry, coef, windowed, color,
        )
    });
    let out = Array2::from_shape_vec((hi, wi), best).expect("shape matches length");
    Ok(out.into_pyarray(gil))
}

/// The `active` mask of `bilateral_filter`: float weights, bool or uint8.
#[derive(FromPyObject)]
pub enum ActiveMask<'py> {
    F64(PyReadonlyArray2<'py, f64>),
    Bool(PyReadonlyArray2<'py, bool>),
    U8(PyReadonlyArray2<'py, u8>),
}

impl ActiveMask<'_> {
    fn to_weights(&self) -> PyResult<(Vec<usize>, Vec<f64>)> {
        Ok(match self {
            ActiveMask::F64(a) => (a.shape().to_vec(), contiguous(a, "active")?.to_vec()),
            ActiveMask::Bool(a) => (
                a.shape().to_vec(),
                contiguous(a, "active")?
                    .iter()
                    .map(|&v| if v { 1.0 } else { 0.0 })
                    .collect(),
            ),
            ActiveMask::U8(a) => (
                a.shape().to_vec(),
                contiguous(a, "active")?
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect(),
            ),
        })
    }
}

/// Bilateral filter of an (h, w, 3) LAB image over the pixels with nonzero `active`.
///
/// radius = ceil(2·σ_s); weight(p, p+d) = exp(−‖I(p+d) − I(p)‖²·inv_color −
/// (dx² + dy²)·inv_spatial) · active(p+d), with inv_x = 1 / (2σ_x²); out = Σ w·I / Σ w,
/// accumulated per pixel over offsets in (dy, dx) lexicographic order like the reference.
/// Pixels whose weights sum to 0 keep their input. `exp` is `libm::exp` so the bits do not
/// depend on the platform's libm (they differ from numpy's exp at rounding level).
#[pyfunction]
pub fn bilateral_filter<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, f64>,
    active: ActiveMask<'py>,
    sigma_color: f64,
    sigma_spatial: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    check_shape("image", shape, &[h, w, 3])?;
    let (active_shape, act) = active.to_weights()?;
    check_shape("active", &active_shape, &[h, w])?;
    let img = contiguous(&image, "image")?;
    if !(sigma_spatial.is_finite() && sigma_spatial > 0.0) {
        return Err(value_error(format!(
            "sigma_spatial must be finite and > 0, got {sigma_spatial}"
        )));
    }
    if !(sigma_color.is_finite() && sigma_color > 0.0) {
        return Err(value_error(format!(
            "sigma_color must be finite and > 0, got {sigma_color}"
        )));
    }
    let out = py.detach(|| gerstner::bilateral_filter(img, &act, h, w, sigma_color, sigma_spatial));
    let out = Array3::from_shape_vec((h, w, 3), out).expect("shape matches length");
    Ok(out.into_pyarray(py))
}

/// The pixel `mask` of `superpixel_stats`: bool or uint8 (nonzero = opaque).
#[derive(FromPyObject)]
pub enum PixelMask<'py> {
    Bool(PyReadonlyArray2<'py, bool>),
    U8(PyReadonlyArray2<'py, u8>),
}

impl PixelMask<'_> {
    fn to_flags(&self) -> PyResult<(Vec<usize>, Vec<bool>)> {
        Ok(match self {
            PixelMask::Bool(a) => (a.shape().to_vec(), contiguous(a, "mask")?.to_vec()),
            PixelMask::U8(a) => (
                a.shape().to_vec(),
                contiguous(a, "mask")?.iter().map(|&v| v != 0).collect(),
            ),
        })
    }
}

/// Count, mass, colour sums and centre sums per superpixel; gerstner's seven per-iteration
/// `np.bincount` calls fused into one pass.
///
/// Pixels are visited in row-major order and only those with a nonzero `mask` and an
/// assignment `s >= 0` contribute: `count[s] += 1`, `mass[s] += w`, `sum_c[s] += w · col_c`,
/// `sum_x[s] += w · (x + 0.5)`, `sum_y[s] += w · (y + 0.5)`, the product rounded before the
/// add (without `weight`, w is not applied at all and mass counts pixels). `np.bincount`
/// adds each bin's values sequentially in input order, and its input is the masked pixel
/// list in row-major order, so these sums are bitwise equal to the reference.
///
/// SINGLE-THREADED ON PURPOSE: a banded parallel version would sum each bin as per-band
/// partial sums merged afterwards, which changes the rounding of the sums and therefore the
/// superpixel means, centres and the output pixels. Keep the one sequential pass.
///
/// Returns (count int64 (n_sp,), mass (n_sp,), colour sums (n_sp, 3), sum_x, sum_y).
#[pyfunction]
#[pyo3(signature = (assigned_to, mask, weight, channels, n_sp))]
#[allow(clippy::type_complexity)]
pub fn superpixel_stats<'py>(
    py: Python<'py>,
    assigned_to: PyReadonlyArray2<'py, i64>,
    mask: PixelMask<'py>,
    weight: Option<PyReadonlyArray2<'py, f64>>,
    channels: PyReadonlyArray3<'py, f64>,
    n_sp: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (hi, wi) = (assigned_to.shape()[0], assigned_to.shape()[1]);
    let (mask_shape, mask) = mask.to_flags()?;
    check_shape("mask", &mask_shape, &[hi, wi])?;
    check_shape("channels", channels.shape(), &[3, hi, wi])?;
    let assigned = contiguous(&assigned_to, "assigned_to")?;
    let ch = contiguous(&channels, "channels")?;
    let weight = match &weight {
        Some(w) => {
            check_shape("weight", w.shape(), &[hi, wi])?;
            Some(contiguous(w, "weight")?)
        }
        None => None,
    };
    let n_sp_i =
        i64::try_from(n_sp).map_err(|_| value_error(format!("n_sp {n_sp} is too large")))?;
    if let Some(&v) = assigned.iter().find(|&&v| v >= n_sp_i) {
        return Err(value_error(format!(
            "assigned_to holds {v}, outside -1..{n_sp}"
        )));
    }

    let stats = py.detach(|| gerstner::superpixel_stats(assigned, &mask, weight, ch, hi, wi, n_sp));
    let sums = Array2::from_shape_vec((n_sp, 3), stats.sums).expect("shape matches length");
    Ok((
        Array1::from(stats.count).into_pyarray(py),
        Array1::from(stats.mass).into_pyarray(py),
        sums.into_pyarray(py),
        Array1::from(stats.sum_x).into_pyarray(py),
        Array1::from(stats.sum_y).into_pyarray(py),
    ))
}

// ------------------------------------------------------------------- control flow, annealer

/// The annealer's tunable constants from a dict of overrides keyed by the Python module
/// constants' names (`PERTURB_DELTA`, `SPLIT_GROWTH`, `SPLIT_SPREAD`,
/// `EXHAUSTED_T_FRACTION`); the others keep their defaults.
fn tunables(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<Tunables> {
    let mut t = Tunables::default();
    if let Some(d) = overrides {
        for (key, value) in d.iter() {
            let key: String = key.extract()?;
            let v: f64 = value.extract()?;
            match key.as_str() {
                "PERTURB_DELTA" => t.perturb_delta = v,
                "SPLIT_GROWTH" => t.split_growth = v,
                "SPLIT_SPREAD" => t.split_spread = v,
                "EXHAUSTED_T_FRACTION" => t.exhausted_t_fraction = v,
                other => {
                    return Err(value_error(format!(
                        "tunables: unknown or fixed constant {other:?}"
                    )))
                }
            }
        }
    }
    Ok(t)
}

/// gerstner's constants as the library defines them, keyed like the Python module's.
#[pyfunction]
fn gerstner_constants(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("PERTURB_DELTA", annealer::PERTURB_DELTA)?;
    d.set_item("SPLIT_GROWTH", annealer::SPLIT_GROWTH)?;
    d.set_item("SPLIT_SPREAD", annealer::SPLIT_SPREAD)?;
    d.set_item("STALL_EVENTS", annealer::STALL_EVENTS)?;
    d.set_item("EXHAUSTED_T_FRACTION", annealer::EXHAUSTED_T_FRACTION)?;
    d.set_item("_MAX_SEARCH_RADIUS", control::MAX_SEARCH_RADIUS)?;
    d.set_item("PROGRESS_EVERY", control::PROGRESS_EVERY)?;
    Ok(d)
}

/// `pixelforge.errors.PixelforgeError(message)`.
fn pixelforge_error(py: Python<'_>, message: &str) -> PyErr {
    match py
        .import("pixelforge.errors")
        .and_then(|m| m.getattr("PixelforgeError"))
        .and_then(|cls| cls.call1((message,)))
    {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

/// `gerstner.run`: lab float64 (H, W, 3), mask bool (H, W), the output size, the Config as
/// `to_dict()`, an optional weight float64 (H, W) -> dict with the fields of
/// `pixelforge.downscale.Downscaled`. PixelforgeError when no pixel is opaque. Logs its
/// progress to the `pixelforge.gerstner` logger.
#[pyfunction]
#[pyo3(signature = (lab, mask, out_width, out_height, config, weight=None, tunables=None))]
#[allow(clippy::too_many_arguments)]
fn gerstner_run<'py>(
    py: Python<'py>,
    lab: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    out_width: usize,
    out_height: usize,
    config: &Bound<'py, PyDict>,
    weight: Option<PyReadonlyArray2<'py, f64>>,
    tunables: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    crate::pylog::sync_level(py)?;
    let shape = lab.shape();
    let (h, w) = (shape[0], shape[1]);
    check_shape("lab", shape, &[h, w, 3])?;
    check_shape("mask", mask.shape(), &[h, w])?;
    if h == 0 || w == 0 || out_width == 0 || out_height == 0 {
        return Err(value_error(format!(
            "input {h}x{w} and output {out_width}x{out_height} must not be empty"
        )));
    }
    let weight = match &weight {
        Some(wt) => {
            check_shape("weight", wt.shape(), &[h, w])?;
            Some(contiguous(wt, "weight")?)
        }
        None => None,
    };
    let lab = contiguous(&lab, "lab")?;
    let mask = contiguous(&mask, "mask")?;
    let config = crate::config::build(py, config)?;
    let tun = self::tunables(tunables)?;
    let out = py
        .detach(|| control::run_with(lab, mask, h, w, out_width, out_height, &config, weight, tun))
        .map_err(|e| pixelforge_error(py, &e.0))?;
    crate::downscale::downscaled_to_py(py, out)
}

fn points_of(points: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<[f64; 3]>> {
    let shape = points.shape();
    check_shape(name, shape, &[shape[0], 3])?;
    Ok(contiguous(points, name)?
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect())
}

fn triples_array<'py>(py: Python<'py>, v: &[[f64; 3]]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = v.iter().flatten().copied().collect();
    Array2::from_shape_vec((v.len(), 3), flat)
        .expect("three values per row")
        .into_pyarray(py)
}

/// `gerstner.principal_axis`: points float64 (N, 3) -> (unit axis float64 (3,), variance).
#[pyfunction]
fn principal_axis<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let pts = points_of(&points, "points")?;
    let (axis, variance) = annealer::principal_axis(&pts);
    Ok((Array1::from(axis.to_vec()).into_pyarray(py), variance))
}

/// `libm::exp` (`log` = false) or `libm::log` elementwise on a float64 (N,) array: the
/// transcendental functions the library uses in place of numpy's, for parity tests.
#[pyfunction]
fn libm_exp_log<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    log: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let f = if log { annealer::log } else { annealer::exp };
    let out: Vec<f64> = contiguous(&x, "x")?.iter().map(|&v| f(v)).collect();
    Ok(Array1::from(out).into_pyarray(py))
}

/// `gerstner.palette_prior`: coverage float64 (N,) -> prior float64 (N,).
#[pyfunction]
fn palette_prior<'py>(
    py: Python<'py>,
    coverage: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = contiguous(&coverage, "coverage")?;
    Ok(Array1::from(control::palette_prior(c)).into_pyarray(py))
}

/// `gerstner.PaletteAnnealer`, the library's annealer. The state attributes are writable so
/// tests can put a Python reference and this one into the same state.
#[pyclass(name = "PaletteAnnealer", module = "pixelforge._native")]
pub struct PyPaletteAnnealer {
    inner: PaletteAnnealer,
}

#[pymethods]
impl PyPaletteAnnealer {
    #[new]
    #[pyo3(signature = (init_points, palette_size, config, tunables=None))]
    fn new(
        py: Python<'_>,
        init_points: PyReadonlyArray2<'_, f64>,
        palette_size: usize,
        config: &Bound<'_, PyDict>,
        tunables: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let pts = points_of(&init_points, "init_points")?;
        if pts.is_empty() || palette_size == 0 {
            return Err(value_error(
                "PaletteAnnealer needs at least one point and palette_size >= 1".into(),
            ));
        }
        let config = crate::config::build(py, config)?;
        let tun = self::tunables(tunables)?;
        Ok(PyPaletteAnnealer {
            inner: PaletteAnnealer::with_tunables(&pts, palette_size, &config, tun),
        })
    }

    /// One iteration on points float64 (n, 3) with prior P(p_s) = weights float64 (n,).
    fn step(
        &mut self,
        py: Python<'_>,
        points: PyReadonlyArray2<'_, f64>,
        weights: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let pts = points_of(&points, "points")?;
        let w = contiguous(&weights, "weights")?;
        check_shape("weights", weights.shape(), &[pts.len()])?;
        let a = &self.inner;
        let pairs = a.clusters.len() / 2;
        let consistent = !a.clusters.is_empty()
            && a.prob.len() == a.clusters.len()
            && (!a.has_sub
                || (a.clusters.len().is_multiple_of(2)
                    && a.placed.len() == pairs
                    && a.history.len() == pairs));
        if !consistent {
            return Err(value_error(
                "PaletteAnnealer state is inconsistent (clusters/prob/placed/history)".into(),
            ));
        }
        let inner = &mut self.inner;
        py.detach(|| inner.step(&pts, w));
        Ok(())
    }

    #[getter]
    fn colors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        triples_array(py, &self.inner.colors())
    }

    #[getter]
    fn clusters<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        triples_array(py, &self.inner.clusters)
    }

    #[setter]
    fn set_clusters(&mut self, value: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        self.inner.clusters = points_of(&value, "clusters")?;
        Ok(())
    }

    #[getter]
    fn prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.inner.prob.clone()).into_pyarray(py)
    }

    #[setter]
    fn set_prob(&mut self, value: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.inner.prob = contiguous(&value, "prob")?.to_vec();
        Ok(())
    }

    #[getter]
    fn placed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.inner.placed.clone()).into_pyarray(py)
    }

    #[setter]
    fn set_placed(&mut self, value: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.inner.placed = contiguous(&value, "placed")?.to_vec();
        Ok(())
    }

    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self.inner.history.iter().flatten().copied().collect();
        Array2::from_shape_vec((self.inner.history.len(), annealer::STALL_EVENTS), flat)
            .expect("STALL_EVENTS values per row")
            .into_pyarray(py)
    }

    #[setter]
    fn set_history(&mut self, value: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let n = annealer::STALL_EVENTS;
        check_shape("history", value.shape(), &[value.shape()[0], n])?;
        self.inner.history = contiguous(&value, "history")?
            .chunks_exact(n)
            .map(|c| {
                let mut row = [0.0; annealer::STALL_EVENTS];
                row.copy_from_slice(c);
                row
            })
            .collect();
        Ok(())
    }

    #[getter(T)]
    fn temperature(&self) -> f64 {
        self.inner.t
    }

    #[setter(T)]
    fn set_temperature(&mut self, value: f64) {
        self.inner.t = value;
    }

    #[getter]
    fn has_sub(&self) -> bool {
        self.inner.has_sub
    }

    #[setter]
    fn set_has_sub(&mut self, value: bool) {
        self.inner.has_sub = value;
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.done
    }

    /// Palette index per point from the latest step, None before the first.
    #[getter]
    fn assign<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i64>>> {
        if self.inner.assign.is_empty() {
            return None;
        }
        let v: Vec<i64> = self.inner.assign.iter().map(|&a| a as i64).collect();
        Some(Array1::from(v).into_pyarray(py))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(slic_assign, m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, m)?)?;
    m.add_function(wrap_pyfunction!(superpixel_stats, m)?)?;
    m.add_function(wrap_pyfunction!(gerstner_constants, m)?)?;
    m.add_function(wrap_pyfunction!(gerstner_run, m)?)?;
    m.add_function(wrap_pyfunction!(principal_axis, m)?)?;
    m.add_function(wrap_pyfunction!(palette_prior, m)?)?;
    m.add_function(wrap_pyfunction!(libm_exp_log, m)?)?;
    m.add_class::<PyPaletteAnnealer>()?;
    Ok(())
}
