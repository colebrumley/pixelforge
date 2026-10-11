//! kopf's per-pixel loops (pixelforge/downscale/kopf.py): the fused E-step + M-step sums,
//! the C-step's directional and overlap sums, and the spatial-only refill of starved kernels.
//!
//! Every input pixel sees at most NS × NS kernels: slot (jy, jx) is kernel
//! (base_y + jy, base_x + jx), valid iff it lies in the output grid and the pixel lies inside
//! its R_k window (|pos − (k + 0.5)| < RK_HALF_WIDTH on both axes), exactly as `_Axis` in the
//! numpy reference (pixelforge/downscale/_reference/kopf.py). Per-pixel quantities are
//! computed with the same operations in the same order as the reference; only the order in
//! which per-kernel sums accumulate differs (pixel order within a band, bands merged in
//! band order; see parallel.rs), and `exp` is `libm::exp`.
//!
//! γ is exchanged with Python pixel-major, shape (hi, wi, 16), slot s = jy·4 + jx: each pixel
//! writes and reads one contiguous run instead of 16 strided ones.

use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::parallel::{band_rows, n_bands, run_bands, thread_count, BAND_ROWS};

/// R_k half-width in output units (kopf.RK_HALF_WIDTH).
const RK_HALF_WIDTH: f64 = 2.0;
/// Slots per axis: ceil(2 · RK_HALF_WIDTH).
const NS: usize = 4;
const NSLOT: usize = NS * NS;
/// Values accumulated per kernel by the E+M step: w, x, y, xx, xy, yy, c0, c1, c2.
const EM_Q: usize = 9;
/// Values accumulated per kernel by the C-step: plus[4], both[4], overlap[4].
const C_Q: usize = 12;
/// Values accumulated per kernel by the refill: mass, c0, c1, c2.
const REFILL_Q: usize = 4;
/// Neighbor directions of the overlap sums, in kopf.py's order.
const OVERLAP_DIRS: [(i64, i64); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// Marks an invalid slot.
const NONE: usize = usize::MAX;
/// libm::exp(x) is exactly 0.0 for every x below this (its own underflow test), so skipping
/// the call there changes no bits.
const EXP_UNDERFLOW: f64 = -745.1332191019411; // the same f64 as libm's -745.13321910194110842

/// `libm::exp`, skipping the call where it underflows to exactly 0.
#[inline]
fn exp(x: f64) -> f64 {
    if x < EXP_UNDERFLOW {
        0.0
    } else {
        libm::exp(x)
    }
}

/// Window bookkeeping along one image axis (`_Axis` in the reference).
struct Axis {
    pos: Vec<f64>,
    base: Vec<i64>,
    /// Per pixel, the kernel of each slot, or `NONE` where the slot is invalid.
    slots: Vec<[usize; NS]>,
    n_out: usize,
}

impl Axis {
    fn new(n_in: usize, n_out: usize) -> Axis {
        let scale = n_out as f64 / n_in as f64;
        let pos: Vec<f64> = (0..n_in).map(|i| (i as f64 + 0.5) * scale).collect();
        let base = pos
            .iter()
            .map(|p| (p - 0.5 - RK_HALF_WIDTH).floor() as i64 + 1)
            .collect();
        let mut axis = Axis {
            pos,
            base,
            slots: Vec::new(),
            n_out,
        };
        axis.slots = (0..n_in)
            .map(|i| std::array::from_fn(|j| axis.slot(i, j).unwrap_or(NONE)))
            .collect();
        axis
    }

    /// Kernel of slot `j` for pixel `i`, if that slot is valid.
    fn slot(&self, i: usize, j: usize) -> Option<usize> {
        let k = self.base[i] + j as i64;
        if k >= 0
            && (k as usize) < self.n_out
            && (self.pos[i] - (k as f64 + 0.5)).abs() < RK_HALF_WIDTH
        {
            Some(k as usize)
        } else {
            None
        }
    }

    /// Kernel rows/cols `[k0, k1)` touched by pixels `[i0, i1)` (any slot, valid or not,
    /// clipped to the grid).
    fn kernel_span(&self, i0: usize, i1: usize) -> (usize, usize) {
        let clip = |k: i64| k.clamp(0, self.n_out as i64) as usize;
        if i0 >= i1 {
            return (0, 0);
        }
        let k0 = clip(self.base[i0]);
        let k1 = clip(self.base[i1 - 1] + NS as i64);
        (k0, k1.max(k0))
    }
}

/// A band's private accumulator: `q` values for each kernel in rows `[ky0, ky1)`.
struct BandAcc {
    offset: usize,
    data: Vec<f64>,
}

impl BandAcc {
    fn new(ay: &Axis, y0: usize, y1: usize, wo: usize, q: usize) -> BandAcc {
        let (ky0, ky1) = ay.kernel_span(y0, y1);
        BandAcc {
            offset: ky0 * wo,
            data: vec![0.0; (ky1 - ky0) * wo * q],
        }
    }

    /// The `q` accumulators of global kernel `k`.
    #[inline]
    fn at(&mut self, k: usize, q: usize) -> &mut [f64] {
        let start = (k - self.offset) * q;
        &mut self.data[start..start + q]
    }
}

/// Zero `out` and add every band's accumulator into it, in band order.
fn merge(out: &mut [f64], bands: &[BandAcc], q: usize) {
    out.fill(0.0);
    for band in bands {
        let start = band.offset * q;
        for (o, v) in out[start..start + band.data.len()]
            .iter_mut()
            .zip(&band.data)
        {
            *o += v;
        }
    }
}

/// Kernel parameters and inputs shared by the γ computations.
struct Model<'a> {
    ax: Axis,
    ay: Axis,
    wo: usize,
    mu_x: &'a [f64],
    mu_y: &'a [f64],
    inv_a: &'a [f64],
    inv_b: &'a [f64],
    inv_c: &'a [f64],
    /// (channels, nu, inv_color): the color term; None = spatial only.
    color: Option<(&'a [f64], &'a [f64], &'a [f64])>,
    /// Transparent pixels get γ = 0 (slot_valid with the pixel mask).
    mask: Option<&'a [u8]>,
    /// Ghost kernels take no pixels.
    ghost: Option<&'a [u8]>,
    /// Multiplied into γ after normalization.
    pixel_weight: Option<&'a [f64]>,
}

/// One pixel's slot kernels, offsets from μ, and γ.
struct PixelGamma {
    kernel: [usize; NSLOT],
    dx: [f64; NSLOT],
    dy: [f64; NSLOT],
    gamma: [f64; NSLOT],
}

impl Model<'_> {
    /// γ_k(i) for the pixel at (y, x): log-weights −0.5·quad − dist·inv_color (−∞ for invalid
    /// slots, ghost kernels and transparent pixels), shifted by the per-pixel maximum,
    /// exponentiated, normalized over the slots (γ = 0 where every slot is −∞), times the
    /// pixel weight. Slots without a kernel have `kernel == NONE` and γ = 0.
    #[inline]
    fn pixel(&self, y: usize, x: usize, n: usize) -> PixelGamma {
        let wi = self.ax.pos.len();
        let i = y * wi + x;
        let mut p = PixelGamma {
            kernel: [NONE; NSLOT],
            dx: [0.0; NSLOT],
            dy: [0.0; NSLOT],
            gamma: [0.0; NSLOT],
        };
        if let Some(mask) = self.mask {
            if mask[i] == 0 {
                return p;
            }
        }
        let (px, py) = (self.ax.pos[x], self.ay.pos[y]);
        let mut peak = f64::NEG_INFINITY;
        let mut any = false;
        let (row, col) = (&self.ay.slots[y], &self.ax.slots[x]);
        for (jy, &ky) in row.iter().enumerate() {
            if ky == NONE {
                continue;
            }
            for (jx, &kx) in col.iter().enumerate() {
                if kx == NONE {
                    continue;
                }
                let k = ky * self.wo + kx;
                if let Some(ghost) = self.ghost {
                    if ghost[k] != 0 {
                        continue;
                    }
                }
                let s = jy * NS + jx;
                let dx = px - self.mu_x[k];
                let dy = py - self.mu_y[k];
                let quad = self.inv_a[k] * dx * dx
                    + self.inv_b[k] * 2.0 * dx * dy
                    + self.inv_c[k] * dy * dy;
                let mut logw = quad * -0.5;
                if let Some((ch, nu, inv_color)) = self.color {
                    let d0 = ch[i] - nu[3 * k];
                    let d1 = ch[n + i] - nu[3 * k + 1];
                    let d2 = ch[2 * n + i] - nu[3 * k + 2];
                    let dist = (d0 * d0 + d1 * d1 + d2 * d2) * inv_color[k];
                    logw -= dist;
                }
                p.kernel[s] = k;
                p.dx[s] = dx;
                p.dy[s] = dy;
                p.gamma[s] = logw;
                any = true;
                peak = peak.max(logw);
            }
        }
        if !any {
            return p;
        }
        // The reference replaces a non-finite peak by 0 before shifting.
        if !peak.is_finite() {
            peak = 0.0;
        }
        let mut sum = 0.0;
        for s in 0..NSLOT {
            if p.kernel[s] != NONE {
                let e = exp(p.gamma[s] - peak);
                p.gamma[s] = e;
                sum += e;
            }
        }
        if sum == 0.0 {
            sum = 1.0; // every weight underflowed: γ = 0, as in the reference
        }
        let weight = self.pixel_weight.map(|w| w[i]);
        for s in 0..NSLOT {
            if p.kernel[s] != NONE {
                let mut g = p.gamma[s] / sum;
                if let Some(w) = weight {
                    g *= w;
                }
                p.gamma[s] = g;
            }
        }
        p
    }
}

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
}

/// Per-band views of a (hi, wi, 16) array: entry `b` holds band `b`'s rows, so bands can
/// write their pixels without sharing a mutable borrow.
fn split_gamma(gamma: &mut [f64], hi: usize, wi: usize) -> Vec<&mut [f64]> {
    if wi == 0 {
        return (0..n_bands(hi)).map(|_| <&mut [f64]>::default()).collect();
    }
    gamma.chunks_mut(BAND_ROWS * wi * NSLOT).collect()
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

    let model = Model {
        ax: Axis::new(wi, wo),
        ay: Axis::new(hi, ho),
        wo,
        mu_x: f64s(F, "mu_x", &geo.mu_x)?,
        mu_y: f64s(F, "mu_y", &geo.mu_y)?,
        inv_a: f64s(F, "inv_a", &geo.inv_a)?,
        inv_b: f64s(F, "inv_b", &geo.inv_b)?,
        inv_c: f64s(F, "inv_c", &geo.inv_c)?,
        color: Some((
            f64s(F, "channels", &channels)?,
            f64s(F, "nu", &nu)?,
            f64s(F, "inv_color", &inv_color)?,
        )),
        mask: Some(u8s(F, "mask", &mask)?),
        ghost: Some(u8s(F, "ghost", &ghost)?),
        pixel_weight: pixel_weight
            .as_ref()
            .map(|w| f64s(F, "pixel_weight", w))
            .transpose()?,
    };
    let gamma = f64s_mut(F, "gamma_out", &mut gamma_out)?;
    let sums = f64s_mut(F, "sums_out", &mut sums_out)?;
    let threads = thread_count();
    py.detach(|| em_step(&model, hi, wi, gamma, sums, threads));
    Ok(())
}

fn em_step(
    model: &Model,
    hi: usize,
    wi: usize,
    gamma: &mut [f64],
    sums: &mut [f64],
    threads: usize,
) {
    let n = hi * wi;
    let channels = model.color.map(|c| c.0).unwrap_or(&[]);
    let bands = run_bands(split_gamma(gamma, hi, wi), threads, |band, gamma| {
        let (y0, y1) = band_rows(band, hi);
        let mut acc = BandAcc::new(&model.ay, y0, y1, model.wo, EM_Q);
        for y in y0..y1 {
            for x in 0..wi {
                let i = y * wi + x;
                let local = (y - y0) * wi + x;
                let p = model.pixel(y, x, n);
                gamma[local * NSLOT..(local + 1) * NSLOT].copy_from_slice(&p.gamma);
                let (c0, c1, c2) = (channels[i], channels[n + i], channels[2 * n + i]);
                // Adding the ±0 terms of slots with γ = 0 changes no bits (accumulators start
                // at +0), so only invalid slots are skipped: fewer unpredictable branches.
                for s in 0..NSLOT {
                    if p.kernel[s] == NONE {
                        continue;
                    }
                    let g = p.gamma[s];
                    let (dx, dy) = (p.dx[s], p.dy[s]);
                    let (gx, gy) = (g * dx, g * dy);
                    let a = acc.at(p.kernel[s], EM_Q);
                    a[0] += g;
                    a[1] += gx;
                    a[2] += gy;
                    a[3] += gx * dx;
                    a[4] += gx * dy;
                    a[5] += gy * dy;
                    a[6] += g * c0;
                    a[7] += g * c1;
                    a[8] += g * c2;
                }
            }
        }
        acc
    });
    merge(sums, &bands, EM_Q);
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
    let args = CStepArgs {
        ax: Axis::new(wi, wo),
        ay: Axis::new(hi, ho),
        wo,
        gamma: f64s(F, "gamma", &gamma)?,
        old_mu_x: f64s(F, "old_mu_x", &old_mu_x)?,
        old_mu_y: f64s(F, "old_mu_y", &old_mu_y)?,
        move_x: f64s(F, "move_x", &move_x)?,
        move_y: f64s(F, "move_y", &move_y)?,
    };
    let plus = f64s_mut(F, "plus_out", &mut plus_out)?;
    let both = f64s_mut(F, "both_out", &mut both_out)?;
    let overlap = f64s_mut(F, "overlap_out", &mut overlap_out)?;
    let threads = thread_count();
    py.detach(|| c_step(&args, plus, both, overlap, threads));
    Ok(())
}

struct CStepArgs<'a> {
    ax: Axis,
    ay: Axis,
    wo: usize,
    gamma: &'a [f64],
    old_mu_x: &'a [f64],
    old_mu_y: &'a [f64],
    move_x: &'a [f64],
    move_y: &'a [f64],
}

fn c_step(
    args: &CStepArgs,
    plus: &mut [f64],
    both: &mut [f64],
    overlap: &mut [f64],
    threads: usize,
) {
    let (hi, wi, wo) = (args.ay.pos.len(), args.ax.pos.len(), args.wo);
    let r = 0.5_f64.sqrt();
    let axes: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (r, r), (r, -r)];
    let bands = run_bands(vec![(); n_bands(hi)], threads, |band, ()| {
        let (y0, y1) = band_rows(band, hi);
        let mut acc = BandAcc::new(&args.ay, y0, y1, wo, C_Q);
        for y in y0..y1 {
            let py = args.ay.pos[y];
            for x in 0..wi {
                let i = y * wi + x;
                let px = args.ax.pos[x];
                let mut kernel = [NONE; NSLOT];
                let mut g = [0.0; NSLOT];
                for (jy, &ky) in args.ay.slots[y].iter().enumerate() {
                    if ky == NONE {
                        continue;
                    }
                    for (jx, &kx) in args.ax.slots[x].iter().enumerate() {
                        if kx == NONE {
                            continue;
                        }
                        let s = jy * NS + jx;
                        kernel[s] = ky * wo + kx;
                        g[s] = args.gamma[i * NSLOT + s];
                    }
                }
                for s in 0..NSLOT {
                    let gs = g[s];
                    if kernel[s] == NONE || gs == 0.0 {
                        continue;
                    }
                    let k = kernel[s];
                    let dx = (px - args.old_mu_x[k]) - args.move_x[k];
                    let dy = (py - args.old_mu_y[k]) - args.move_y[k];
                    let a = acc.at(k, C_Q);
                    for (ai, &(ex, ey)) in axes.iter().enumerate() {
                        let u = dx * ex + dy * ey;
                        let t = gs * u * u;
                        a[4 + ai] += t;
                        // Branch-free; adding +0 where u <= 0 changes no bits (t >= 0).
                        a[ai] += if u > 0.0 { t } else { 0.0 };
                    }
                    let (jy, jx) = ((s / NS) as i64, (s % NS) as i64);
                    for (d, &(ndy, ndx)) in OVERLAP_DIRS.iter().enumerate() {
                        let (ny, nx) = (jy + ndy, jx + ndx);
                        if !(0..NS as i64).contains(&ny) || !(0..NS as i64).contains(&nx) {
                            continue;
                        }
                        // γ_n = 0 (or no kernel there) adds +0: no bits change.
                        a[8 + d] += gs * g[ny as usize * NS + nx as usize];
                    }
                }
            }
        }
        acc
    });
    let m = args.ay.n_out * wo;
    let mut merged = vec![0.0; m * C_Q];
    merge(&mut merged, &bands, C_Q);
    for k in 0..m {
        for a in 0..4 {
            plus[a * m + k] = merged[k * C_Q + a];
            both[a * m + k] = merged[k * C_Q + 4 + a];
            overlap[a * m + k] = merged[k * C_Q + 8 + a];
        }
    }
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
    let model = Model {
        ax: Axis::new(wi, wo),
        ay: Axis::new(hi, ho),
        wo,
        mu_x: f64s(F, "mu_x", &geo.mu_x)?,
        mu_y: f64s(F, "mu_y", &geo.mu_y)?,
        inv_a: f64s(F, "inv_a", &geo.inv_a)?,
        inv_b: f64s(F, "inv_b", &geo.inv_b)?,
        inv_c: f64s(F, "inv_c", &geo.inv_c)?,
        color: None,
        mask: None,
        ghost: None,
        pixel_weight: None,
    };
    let channels = f64s(F, "channels", &channels)?;
    let pixel_mass = f64s(F, "pixel_mass", &pixel_mass)?;
    let mass = f64s_mut(F, "mass_out", &mut mass_out)?;
    let color = f64s_mut(F, "color_out", &mut color_out)?;
    let threads = thread_count();
    py.detach(|| spatial_refill(&model, channels, pixel_mass, mass, color, threads));
    Ok(())
}

fn spatial_refill(
    model: &Model,
    channels: &[f64],
    pixel_mass: &[f64],
    mass: &mut [f64],
    color: &mut [f64],
    threads: usize,
) {
    let (hi, wi, wo) = (model.ay.pos.len(), model.ax.pos.len(), model.wo);
    let n = hi * wi;
    let bands = run_bands(vec![(); n_bands(hi)], threads, |band, ()| {
        let (y0, y1) = band_rows(band, hi);
        let mut acc = BandAcc::new(&model.ay, y0, y1, wo, REFILL_Q);
        for y in y0..y1 {
            for x in 0..wi {
                let i = y * wi + x;
                let p = model.pixel(y, x, n);
                for s in 0..NSLOT {
                    let g = p.gamma[s] * pixel_mass[i];
                    if p.kernel[s] == NONE || g == 0.0 {
                        continue;
                    }
                    let a = acc.at(p.kernel[s], REFILL_Q);
                    a[0] += g;
                    a[1] += g * channels[i];
                    a[2] += g * channels[n + i];
                    a[3] += g * channels[2 * n + i];
                }
            }
        }
        acc
    });
    let m = model.ay.n_out * wo;
    let mut merged = vec![0.0; m * REFILL_Q];
    merge(&mut merged, &bands, REFILL_Q);
    for k in 0..m {
        mass[k] = merged[k * REFILL_Q];
        for c in 0..3 {
            color[k * 3 + c] = merged[k * REFILL_Q + 1 + c];
        }
    }
}
