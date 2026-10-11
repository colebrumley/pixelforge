//! Python wrappers of the kopf kernels in `pixelforge::kopf`: argument checking (dtype,
//! shape, contiguity, writeability, each failure a `ValueError` naming the argument), then
//! the library call with the GIL released.

use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pixelforge::kopf::{self, Kernels, EM_Q, NSLOT};
use pixelforge::parallel::thread_count;

// ------------------------------------------------------------------ argument checking

fn value_error(func: &str, msg: String) -> PyErr {
    PyValueError::new_err(format!("{func}: {msg}"))
}

fn check_layout(
    func: &str,
    name: &str,
    got: &[usize],
    contiguous: bool,
    want: &[usize],
) -> PyResult<()> {
    if got != want {
        return Err(value_error(
            func,
            format!("{name} must have shape {want:?}, got {got:?}"),
        ));
    }
    if !contiguous {
        return Err(value_error(
            func,
            format!("{name} must be C-contiguous (use numpy.ascontiguousarray)"),
        ));
    }
    Ok(())
}

fn cast_array<'a, 'py, T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    obj: &'a Bound<'py, PyAny>,
) -> PyResult<&'a Bound<'py, PyArrayDyn<T>>> {
    obj.cast::<PyArrayDyn<T>>().map_err(|_| {
        let got = obj
            .getattr("dtype")
            .map(|d| d.to_string())
            .unwrap_or_else(|_| obj.get_type().to_string());
        value_error(
            func,
            format!("{name} must be a {dtype} numpy.ndarray, got {got}"),
        )
    })
}

/// The shape of a `dtype` array argument, before its shape is known to the caller.
fn shape_of<T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    obj: &Bound<'_, PyAny>,
    ndim: usize,
) -> PyResult<Vec<usize>> {
    let shape = cast_array::<T>(func, name, dtype, obj)?.shape().to_vec();
    if shape.len() != ndim {
        return Err(value_error(
            func,
            format!("{name} must be {ndim}-dimensional, got shape {shape:?}"),
        ));
    }
    Ok(shape)
}

fn readonly<'py, T: numpy::Element>(
    func: &str,
    name: &str,
    dtype: &str,
    obj: &Bound<'py, PyAny>,
    shape: &[usize],
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    let arr = cast_array::<T>(func, name, dtype, obj)?;
    check_layout(func, name, arr.shape(), arr.is_c_contiguous(), shape)?;
    arr.try_readonly()
        .map_err(|e| value_error(func, format!("cannot read {name}: {e}")))
}

fn readwrite<'py>(
    func: &str,
    name: &str,
    obj: &Bound<'py, PyAny>,
    shape: &[usize],
) -> PyResult<PyReadwriteArrayDyn<'py, f64>> {
    let arr = cast_array::<f64>(func, name, "float64", obj)?;
    check_layout(func, name, arr.shape(), arr.is_c_contiguous(), shape)?;
    arr.try_readwrite().map_err(|e| {
        value_error(
            func,
            format!("cannot write {name} (it must be writeable and must not alias another argument): {e}"),
        )
    })
}

fn f64s<'a>(func: &str, name: &str, arr: &'a PyReadonlyArrayDyn<'_, f64>) -> PyResult<&'a [f64]> {
    arr.as_slice()
        .map_err(|e| value_error(func, format!("{name}: {e}")))
}

fn u8s<'a>(func: &str, name: &str, arr: &'a PyReadonlyArrayDyn<'_, u8>) -> PyResult<&'a [u8]> {
    arr.as_slice()
        .map_err(|e| value_error(func, format!("{name}: {e}")))
}

fn f64s_mut<'a>(
    func: &str,
    name: &str,
    arr: &'a mut PyReadwriteArrayDyn<'_, f64>,
) -> PyResult<&'a mut [f64]> {
    arr.as_slice_mut()
        .map_err(|e| value_error(func, format!("{name}: {e}")))
}

/// The five (ho, wo) kernel-geometry arrays every function takes.
struct Geometry<'py> {
    mu_x: PyReadonlyArrayDyn<'py, f64>,
    mu_y: PyReadonlyArrayDyn<'py, f64>,
    inv_a: PyReadonlyArrayDyn<'py, f64>,
    inv_b: PyReadonlyArrayDyn<'py, f64>,
    inv_c: PyReadonlyArrayDyn<'py, f64>,
}

impl<'py> Geometry<'py> {
    fn new(func: &str, args: [(&str, &Bound<'py, PyAny>); 5], k: &[usize]) -> PyResult<Self> {
        let [mx, my, a, b, c] = args;
        Ok(Geometry {
            mu_x: readonly(func, mx.0, "float64", mx.1, k)?,
            mu_y: readonly(func, my.0, "float64", my.1, k)?,
            inv_a: readonly(func, a.0, "float64", a.1, k)?,
            inv_b: readonly(func, b.0, "float64", b.1, k)?,
            inv_c: readonly(func, c.0, "float64", c.1, k)?,
        })
    }

    /// The library's view of the five arrays.
    fn kernels(&self, func: &str, ho: usize, wo: usize) -> PyResult<Kernels<'_>> {
        Ok(Kernels {
            ho,
            wo,
            mu_x: f64s(func, "mu_x", &self.mu_x)?,
            mu_y: f64s(func, "mu_y", &self.mu_y)?,
            inv_a: f64s(func, "inv_a", &self.inv_a)?,
            inv_b: f64s(func, "inv_b", &self.inv_b)?,
            inv_c: f64s(func, "inv_c", &self.inv_c)?,
        })
    }
}

// ------------------------------------------------------------------ E-step + M-step

/// E-step and M-step sums for one kopf iteration, fused per pixel.
///
/// Writes γ (after normalization and the pixel weight) into `gamma_out` (hi, wi, 16), slot
/// s = jy·4 + jx (pixel-major, so each pixel writes one contiguous run), and the per-kernel sums into `sums_out` (ho, wo, 9): w, x, y, xx, xy, yy,
/// c0, c1, c2 with x = Σγ·dx etc. and dx, dy offsets from the μ passed in.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn kopf_em_step<'py>(
    py: Python<'py>,
    channels: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    pixel_weight: Option<&Bound<'py, PyAny>>,
    mu_x: &Bound<'py, PyAny>,
    mu_y: &Bound<'py, PyAny>,
    inv_a: &Bound<'py, PyAny>,
    inv_b: &Bound<'py, PyAny>,
    inv_c: &Bound<'py, PyAny>,
    nu: &Bound<'py, PyAny>,
    inv_color: &Bound<'py, PyAny>,
    ghost: &Bound<'py, PyAny>,
    gamma_out: &Bound<'py, PyAny>,
    sums_out: &Bound<'py, PyAny>,
) -> PyResult<()> {
    const F: &str = "kopf_em_step";
    let cs = shape_of::<f64>(F, "channels", "float64", channels, 3)?;
    if cs[0] != 3 {
        return Err(value_error(
            F,
            format!("channels must have shape (3, hi, wi), got {cs:?}"),
        ));
    }
    let (hi, wi) = (cs[1], cs[2]);
    let ks = shape_of::<f64>(F, "mu_x", "float64", mu_x, 2)?;
    let (ho, wo) = (ks[0], ks[1]);
    let channels = readonly::<f64>(F, "channels", "float64", channels, &[3, hi, wi])?;
    let mask = readonly::<u8>(F, "mask", "uint8", mask, &[hi, wi])?;
    let pixel_weight = pixel_weight
        .map(|w| readonly::<f64>(F, "pixel_weight", "float64", w, &[hi, wi]))
        .transpose()?;
    let geo = Geometry::new(
        F,
        [
            ("mu_x", mu_x),
            ("mu_y", mu_y),
            ("inv_a", inv_a),
            ("inv_b", inv_b),
            ("inv_c", inv_c),
        ],
        &[ho, wo],
    )?;
    let nu = readonly::<f64>(F, "nu", "float64", nu, &[ho, wo, 3])?;
    let inv_color = readonly::<f64>(F, "inv_color", "float64", inv_color, &[ho, wo])?;
    let ghost = readonly::<u8>(F, "ghost", "uint8", ghost, &[ho, wo])?;
    let mut gamma_out = readwrite(F, "gamma_out", gamma_out, &[hi, wi, NSLOT])?;
    let mut sums_out = readwrite(F, "sums_out", sums_out, &[ho, wo, EM_Q])?;

    let kernels = geo.kernels(F, ho, wo)?;
    let channels = f64s(F, "channels", &channels)?;
    let nu = f64s(F, "nu", &nu)?;
    let inv_color = f64s(F, "inv_color", &inv_color)?;
    let mask = u8s(F, "mask", &mask)?;
    let ghost = u8s(F, "ghost", &ghost)?;
    let pixel_weight = pixel_weight
        .as_ref()
        .map(|w| f64s(F, "pixel_weight", w))
        .transpose()?;
    let gamma = f64s_mut(F, "gamma_out", &mut gamma_out)?;
    let sums = f64s_mut(F, "sums_out", &mut sums_out)?;
    let threads = thread_count();
    py.detach(|| {
        kopf::em_step(
            hi,
            wi,
            &kernels,
            channels,
            mask,
            pixel_weight,
            nu,
            inv_color,
            ghost,
            gamma,
            sums,
            threads,
        )
    });
    Ok(())
}

// ------------------------------------------------------------------ C-step

/// The C-step's per-kernel sums from the E-step's γ (hi, wi, 16).
///
/// With (dx, dy) = (p_i − old μ_k) − (μ_k − old μ_k) and u = dx·ex + dy·ey for the axes
/// (1, 0), (0, 1), (√½, √½), (√½, −√½) (kopf.py's `axes` order): `both_out[a]` = Σ γ·u²,
/// `plus_out[a]` = Σ γ·u² over u > 0. `overlap_out[d]` = Σ_i γ_k(i)·γ_n(i) for the neighbor
/// n = k + d, d in (0, 1), (1, 0), (1, 1), (1, −1), over pixels where both slots exist.
/// All outputs are (4, ho, wo).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn kopf_c_step<'py>(
    py: Python<'py>,
    gamma: &Bound<'py, PyAny>,
    old_mu_x: &Bound<'py, PyAny>,
    old_mu_y: &Bound<'py, PyAny>,
    move_x: &Bound<'py, PyAny>,
    move_y: &Bound<'py, PyAny>,
    plus_out: &Bound<'py, PyAny>,
    both_out: &Bound<'py, PyAny>,
    overlap_out: &Bound<'py, PyAny>,
) -> PyResult<()> {
    const F: &str = "kopf_c_step";
    let gs = shape_of::<f64>(F, "gamma", "float64", gamma, 3)?;
    if gs[2] != NSLOT {
        return Err(value_error(
            F,
            format!("gamma must have shape (hi, wi, 16), got {gs:?}"),
        ));
    }
    let (hi, wi) = (gs[0], gs[1]);
    let ks = shape_of::<f64>(F, "old_mu_x", "float64", old_mu_x, 2)?;
    let (ho, wo) = (ks[0], ks[1]);
    let k = [ho, wo];
    let gamma = readonly::<f64>(F, "gamma", "float64", gamma, &[hi, wi, NSLOT])?;
    let old_mu_x = readonly::<f64>(F, "old_mu_x", "float64", old_mu_x, &k)?;
    let old_mu_y = readonly::<f64>(F, "old_mu_y", "float64", old_mu_y, &k)?;
    let move_x = readonly::<f64>(F, "move_x", "float64", move_x, &k)?;
    let move_y = readonly::<f64>(F, "move_y", "float64", move_y, &k)?;
    let mut plus_out = readwrite(F, "plus_out", plus_out, &[4, ho, wo])?;
    let mut both_out = readwrite(F, "both_out", both_out, &[4, ho, wo])?;
    let mut overlap_out = readwrite(F, "overlap_out", overlap_out, &[4, ho, wo])?;
    let gamma = f64s(F, "gamma", &gamma)?;
    let old_mu_x = f64s(F, "old_mu_x", &old_mu_x)?;
    let old_mu_y = f64s(F, "old_mu_y", &old_mu_y)?;
    let move_x = f64s(F, "move_x", &move_x)?;
    let move_y = f64s(F, "move_y", &move_y)?;
    let plus = f64s_mut(F, "plus_out", &mut plus_out)?;
    let both = f64s_mut(F, "both_out", &mut both_out)?;
    let overlap = f64s_mut(F, "overlap_out", &mut overlap_out)?;
    let threads = thread_count();
    py.detach(|| {
        kopf::c_step(
            hi, wi, ho, wo, gamma, old_mu_x, old_mu_y, move_x, move_y, plus, both, overlap, threads,
        )
    });
    Ok(())
}

// ------------------------------------------------------------------ starved-kernel refill

/// Spatial-only responsibilities for the final recoloring of starved kernels.
///
/// γ from −0.5·(p − μ)ᵀΣ⁻¹(p − μ) alone, over every in-image slot (no pixel mask, no ghost
/// test, no color term, no pixel weight), normalized per pixel. Then with g = γ·pixel_mass:
/// `mass_out` (ho, wo) = Σ g and `color_out` (ho, wo, 3) = Σ g·channel.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn kopf_spatial_refill<'py>(
    py: Python<'py>,
    channels: &Bound<'py, PyAny>,
    pixel_mass: &Bound<'py, PyAny>,
    mu_x: &Bound<'py, PyAny>,
    mu_y: &Bound<'py, PyAny>,
    inv_a: &Bound<'py, PyAny>,
    inv_b: &Bound<'py, PyAny>,
    inv_c: &Bound<'py, PyAny>,
    mass_out: &Bound<'py, PyAny>,
    color_out: &Bound<'py, PyAny>,
) -> PyResult<()> {
    const F: &str = "kopf_spatial_refill";
    let cs = shape_of::<f64>(F, "channels", "float64", channels, 3)?;
    if cs[0] != 3 {
        return Err(value_error(
            F,
            format!("channels must have shape (3, hi, wi), got {cs:?}"),
        ));
    }
    let (hi, wi) = (cs[1], cs[2]);
    let ks = shape_of::<f64>(F, "mu_x", "float64", mu_x, 2)?;
    let (ho, wo) = (ks[0], ks[1]);
    let channels = readonly::<f64>(F, "channels", "float64", channels, &[3, hi, wi])?;
    let pixel_mass = readonly::<f64>(F, "pixel_mass", "float64", pixel_mass, &[hi, wi])?;
    let geo = Geometry::new(
        F,
        [
            ("mu_x", mu_x),
            ("mu_y", mu_y),
            ("inv_a", inv_a),
            ("inv_b", inv_b),
            ("inv_c", inv_c),
        ],
        &[ho, wo],
    )?;
    let mut mass_out = readwrite(F, "mass_out", mass_out, &[ho, wo])?;
    let mut color_out = readwrite(F, "color_out", color_out, &[ho, wo, 3])?;
    let kernels = geo.kernels(F, ho, wo)?;
    let channels = f64s(F, "channels", &channels)?;
    let pixel_mass = f64s(F, "pixel_mass", &pixel_mass)?;
    let mass = f64s_mut(F, "mass_out", &mut mass_out)?;
    let color = f64s_mut(F, "color_out", &mut color_out)?;
    let threads = thread_count();
    py.detach(|| {
        kopf::spatial_refill(hi, wi, &kernels, channels, pixel_mass, mass, color, threads)
    });
    Ok(())
}
