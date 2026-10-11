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
//!
//! The entry points take row-major slices and plain dimensions; the Python binding
//! (crates/pixelforge-py) validates dtypes, shapes and contiguity before calling them.

use crate::parallel::{band_rows, n_bands, run_bands, BAND_ROWS};

/// R_k half-width in output units (kopf.RK_HALF_WIDTH).
const RK_HALF_WIDTH: f64 = 2.0;
/// Slots per axis: ceil(2 · RK_HALF_WIDTH).
const NS: usize = 4;
pub const NSLOT: usize = NS * NS;
/// Values accumulated per kernel by the E+M step: w, x, y, xx, xy, yy, c0, c1, c2.
pub const EM_Q: usize = 9;
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

/// The (ho, wo) kernel geometry every entry point takes, row-major: centers μ and the
/// entries of Σ⁻¹ = [[inv_a, inv_b], [inv_b, inv_c]].
pub struct Kernels<'a> {
    pub ho: usize,
    pub wo: usize,
    pub mu_x: &'a [f64],
    pub mu_y: &'a [f64],
    pub inv_a: &'a [f64],
    pub inv_b: &'a [f64],
    pub inv_c: &'a [f64],
}

impl Kernels<'_> {
    fn check(&self) {
        let m = self.ho * self.wo;
        check_len("mu_x", self.mu_x, m);
        check_len("mu_y", self.mu_y, m);
        check_len("inv_a", self.inv_a, m);
        check_len("inv_b", self.inv_b, m);
        check_len("inv_c", self.inv_c, m);
    }
}

/// Panics unless `slice` holds `want` elements (the binding checks shapes first, so this
/// only guards direct Rust callers).
fn check_len<T>(name: &str, slice: &[T], want: usize) {
    assert_eq!(slice.len(), want, "{name} must hold {want} elements");
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
/// `channels` is (3, hi, wi), `mask` (hi, wi) (0 = transparent), `pixel_weight` (hi, wi),
/// `nu` (ho, wo, 3), `inv_color` and `ghost` (ho, wo). Writes γ (after normalization and
/// the pixel weight) into `gamma_out` (hi, wi, 16), slot s = jy·4 + jx (pixel-major, so
/// each pixel writes one contiguous run), and the per-kernel sums into `sums_out`
/// (ho, wo, 9): w, x, y, xx, xy, yy, c0, c1, c2 with x = Σγ·dx etc. and dx, dy offsets
/// from the μ passed in.
#[allow(clippy::too_many_arguments)]
pub fn em_step(
    hi: usize,
    wi: usize,
    kernels: &Kernels,
    channels: &[f64],
    mask: &[u8],
    pixel_weight: Option<&[f64]>,
    nu: &[f64],
    inv_color: &[f64],
    ghost: &[u8],
    gamma_out: &mut [f64],
    sums_out: &mut [f64],
    threads: usize,
) {
    let (ho, wo) = (kernels.ho, kernels.wo);
    kernels.check();
    check_len("channels", channels, 3 * hi * wi);
    check_len("mask", mask, hi * wi);
    if let Some(w) = pixel_weight {
        check_len("pixel_weight", w, hi * wi);
    }
    check_len("nu", nu, ho * wo * 3);
    check_len("inv_color", inv_color, ho * wo);
    check_len("ghost", ghost, ho * wo);
    check_len("gamma_out", gamma_out, hi * wi * NSLOT);
    check_len("sums_out", sums_out, ho * wo * EM_Q);
    let model = Model {
        ax: Axis::new(wi, wo),
        ay: Axis::new(hi, ho),
        wo,
        mu_x: kernels.mu_x,
        mu_y: kernels.mu_y,
        inv_a: kernels.inv_a,
        inv_b: kernels.inv_b,
        inv_c: kernels.inv_c,
        color: Some((channels, nu, inv_color)),
        mask: Some(mask),
        ghost: Some(ghost),
        pixel_weight,
    };
    em_step_bands(&model, hi, wi, gamma_out, sums_out, threads);
}

fn em_step_bands(
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
/// `old_mu_*` and `move_*` are (ho, wo); all outputs are (4, ho, wo).
#[allow(clippy::too_many_arguments)]
pub fn c_step(
    hi: usize,
    wi: usize,
    ho: usize,
    wo: usize,
    gamma: &[f64],
    old_mu_x: &[f64],
    old_mu_y: &[f64],
    move_x: &[f64],
    move_y: &[f64],
    plus_out: &mut [f64],
    both_out: &mut [f64],
    overlap_out: &mut [f64],
    threads: usize,
) {
    let m = ho * wo;
    check_len("gamma", gamma, hi * wi * NSLOT);
    check_len("old_mu_x", old_mu_x, m);
    check_len("old_mu_y", old_mu_y, m);
    check_len("move_x", move_x, m);
    check_len("move_y", move_y, m);
    check_len("plus_out", plus_out, 4 * m);
    check_len("both_out", both_out, 4 * m);
    check_len("overlap_out", overlap_out, 4 * m);
    let args = CStepArgs {
        ax: Axis::new(wi, wo),
        ay: Axis::new(hi, ho),
        wo,
        gamma,
        old_mu_x,
        old_mu_y,
        move_x,
        move_y,
    };
    c_step_bands(&args, plus_out, both_out, overlap_out, threads);
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

fn c_step_bands(
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
/// `mass_out` (ho, wo) = Σ g and `color_out` (ho, wo, 3) = Σ g·channel. `channels` is
/// (3, hi, wi) and `pixel_mass` (hi, wi).
#[allow(clippy::too_many_arguments)]
pub fn spatial_refill(
    hi: usize,
    wi: usize,
    kernels: &Kernels,
    channels: &[f64],
    pixel_mass: &[f64],
    mass_out: &mut [f64],
    color_out: &mut [f64],
    threads: usize,
) {
    let (ho, wo) = (kernels.ho, kernels.wo);
    kernels.check();
    check_len("channels", channels, 3 * hi * wi);
    check_len("pixel_mass", pixel_mass, hi * wi);
    check_len("mass_out", mass_out, ho * wo);
    check_len("color_out", color_out, ho * wo * 3);
    let model = Model {
        ax: Axis::new(wi, wo),
        ay: Axis::new(hi, ho),
        wo,
        mu_x: kernels.mu_x,
        mu_y: kernels.mu_y,
        inv_a: kernels.inv_a,
        inv_b: kernels.inv_b,
        inv_c: kernels.inv_c,
        color: None,
        mask: None,
        ghost: None,
        pixel_weight: None,
    };
    spatial_refill_bands(&model, channels, pixel_mass, mass_out, color_out, threads);
}

fn spatial_refill_bands(
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
