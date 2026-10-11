//! Content-adaptive downscaling (Kopf, Shamir, Peers 2013), REQUIREMENTS.md Section 6: the
//! control flow of `pixelforge/downscale/kopf.py` around the per-pixel loops in `kopf.rs`.
//!
//! A constrained EM over one anisotropic Gaussian kernel per output pixel. All spatial
//! quantities are in output-pixel units. Every input pixel is covered by at most 4 × 4
//! kernels (those whose R_k window contains it); the per-pixel loops over those slots (E-step
//! responsibilities γ, M-step sums, the C-step's directional and overlap sums, and the final
//! spatial-only refill) are `kopf::em_step`, `kopf::c_step` and `kopf::spatial_refill`.
//! Everything per kernel, i.e. the M-step and C-step algebra on (ho, wo) arrays, the
//! constraints, convergence and logging, is here, ported operation for operation from the
//! numpy code (docs/rust-migration-plan.md, Decisions 3 and 6):
//!
//! - elementwise expressions are evaluated in numpy's order, one rounding per ufunc;
//! - `np.mean` over the fed kernels is numpy's pairwise sum (`npsum`);
//! - `np.linalg.eigh` on the 2×2 covariances is LAPACK's `dsyevd` path for n = 2 (the
//!   `dsteqr` split tests, `dlaev2`, the rotation and the ascending sort), with the two
//!   fused multiply-adds numpy's Accelerate build performs, and the reconstruction
//!   `(V · diag(w)) @ Vᵀ` with the fused multiply-add of its BLAS `dgemm`; see `eigh2`;
//! - `SIGMA_GROWTH ** fires` is a table of the correctly rounded powers (what numpy's
//!   `pow` returns); `np.hypot` is `sqrt(fma(small, small, big²))`, which is what numpy's
//!   `hypot` computes on the macOS build (see `hypot`);
//! - the window counts (`win_y @ mask @ win_x.T`, `np.outer`) are integer arithmetic.
//!
//! The fused multiply-adds are `libm::fma` (pure Rust, correctly rounded, so the results are
//! the same on every platform; `f64::mul_add` would fall back to the platform's C `fma`
//! where the CPU has none, which the determinism guard forbids).

use crate::boxfilter::pixel_cells;
use crate::config::Config;
use crate::downscale::{neighbor_mean4, Downscaled, StatValue};
use crate::kopf::{self, Kernels, EM_Q, NSLOT};
use crate::npsum::pairwise_mean;

/// R_k half-width in output units.
pub const RK_HALF_WIDTH: f64 = 2.0;
pub const SIGMA_INIT: f64 = 1e-4;
pub const SIGMA_CAP: f64 = 1.0;
pub const SIGMA_GROWTH: f64 = 1.1;
pub const COV_INIT: f64 = 1.0 / 3.0;
pub const SINGULAR_MIN: f64 = 0.05;
pub const SINGULAR_MAX: f64 = 0.1;
pub const CLAMP_BOX: f64 = 0.25;
pub const DIRECTIONAL_VARIANCE_MAX: f64 = 0.2;
pub const OVERLAP_MIN: f64 = 0.08;
pub const ORIENTATION_MAX_DEGREES: f64 = 25.0;
/// A kernel holding less than this fraction of a cell's worth of pixels counts as starved.
pub const STARVED_FRACTION: f64 = 0.02;
/// Convergence allows σ to still change on fewer than this fraction of the kernels. On noisy
/// content 0.5–1 % of the kernels keep growing σ by 1.1× per iteration long after μ and ν
/// have settled (σ starts at 1e-4, so reaching the cap takes ~100 steps), and at 0.1 % no
/// real image converged within 50 iterations. At 2 % the result differs from the
/// 50-iteration one by a mean ΔE below 0.1 on the noisy fixtures.
pub const CHANGED_SIGMA_FRACTION: f64 = 0.02;
/// Iterations between INFO progress lines (DEBUG logs every one).
pub const PROGRESS_EVERY: usize = 10;

const NEIGHBORS8: [(i64, i64); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

const LOG_TARGET: &str = "pixelforge::kopf";

// ------------------------------------------------------------------ numerics helpers

/// `SIGMA_GROWTH ** n` for n = 0..=16, correctly rounded (bits of numpy's `1.1 ** n`, which
/// are the correctly rounded values; `libm::pow` is 1 ulp off at n = 4). A kernel fires at
/// most twice per neighbor direction (as k and as the neighbor), so `fires` <= 16.
const GROWTH_POW: [u64; 17] = [
    0x3ff0000000000000,
    0x3ff199999999999a,
    0x3ff35c28f5c28f5d,
    0x3ff54bc6a7ef9db4,
    0x3ff76cf41f212d79,
    0x3ff9c4a6223e186c,
    0x3ffc585058dde7ab,
    0x3fff2df1fb5a7ed7,
    0x40012611e3d82c2a,
    0x4002dd13add43095,
    0x4004bffc0c03023e,
    0x4006d32eda034f44,
    0x40091b805636d732,
    0x400b9e405ed5ecb8,
    0x400e6146ceb81dfe,
    0x4010b5808b4baa19,
    0x401261409939a182,
];

/// `SIGMA_GROWTH ** n`, as numpy computes it for an integer exponent array.
pub fn growth_pow(n: u32) -> f64 {
    match GROWTH_POW.get(n as usize) {
        Some(&bits) => f64::from_bits(bits),
        None => libm::pow(SIGMA_GROWTH, n as f64),
    }
}

/// `np.hypot(x, y)` as numpy's macOS build computes it: `sqrt(fma(small, small, big²))`
/// with `big`/`small` the larger/smaller magnitude. Bitwise equal to numpy there on 200,000
/// samples spanning 1e-16..10 (`libm::hypot` differs on ~10 % of them, by 1 ulp). The
/// arguments here are centroid differences (0 or >= 1 ulp of 1) and ±1, so no scaling for
/// overflow or underflow is needed.
pub fn hypot(x: f64, y: f64) -> f64 {
    let (a, b) = (x.abs(), y.abs());
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    libm::fma(small, small, big * big).sqrt()
}

/// Python's `'%.2e' % x`: two decimals, the exponent signed with at least two digits.
pub fn format_e2(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let s = format!("{x:.2e}");
    let (mantissa, exp) = s.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exp.abs())
}

// ------------------------------------------------------------------ 2×2 eigenproblem

/// LAPACK's `dlamch('E')` (relative machine epsilon, rounding) and `dlamch('S')`.
const EPS: f64 = f64::EPSILON * 0.5;
const SAFMIN: f64 = f64::MIN_POSITIVE;

/// LAPACK `dlaev2`: eigen-decomposition of [[a, b], [b, c]]. Returns (rt1, rt2, cs1, sn1):
/// rt1 the eigenvalue of larger absolute value, rt2 the other, (cs1, sn1) the unit right
/// eigenvector for rt1. Reference-LAPACK's formulas, with the two contractions numpy's
/// LAPACK (Accelerate) performs: `rt2 = fma(acmx/rt1, acmn, -(b/rt1)·b)` and
/// `1 + ct²` = `fma(ct, ct, 1)` (likewise `tn`); `1 + (ab/adf)²` is not fused.
fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let q = ab / adf;
        adf * (1.0 + q * q).sqrt()
    } else if adf < ab {
        let q = adf / ab;
        ab * (1.0 + q * q).sqrt()
    } else {
        ab * 2.0_f64.sqrt()
    };
    let (rt1, rt2, sgn1) = if sm < 0.0 {
        let rt1 = 0.5 * (sm - rt);
        let rt2 = libm::fma(acmx / rt1, acmn, -((b / rt1) * b));
        (rt1, rt2, -1)
    } else if sm > 0.0 {
        let rt1 = 0.5 * (sm + rt);
        let rt2 = libm::fma(acmx / rt1, acmn, -((b / rt1) * b));
        (rt1, rt2, 1)
    } else {
        (0.5 * rt, -0.5 * rt, 1)
    };
    let (cs, sgn2) = if df >= 0.0 {
        (df + rt, 1)
    } else {
        (df - rt, -1)
    };
    let (mut cs1, mut sn1);
    if cs.abs() > ab {
        let ct = -tb / cs;
        sn1 = 1.0 / libm::fma(ct, ct, 1.0).sqrt();
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1.0 / libm::fma(tn, tn, 1.0).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

/// `np.linalg.eigh` of the symmetric matrix whose lower triangle is [[a, ·], [b, c]]: the
/// eigenvalues ascending and the eigenvectors as columns, `v[row][col]`.
///
/// numpy calls LAPACK `dsyevd` (jobz 'V', uplo 'L'). For n = 2 its Householder reduction is
/// the identity (the reflector of a length-1 vector has τ = 0), so the tridiagonal matrix is
/// d = (a, c), e = b, and `dstedc` hands it to `dsteqr`. That splits the matrix where the
/// off-diagonal is negligible (two tests, below), otherwise solves it with `dlaev2` and
/// rotates the identity by (c, s): columns (c, s) for rt1 and (−s, c) for rt2; finally it
/// sorts the eigenvalues ascending, swapping columns. LAPACK's scaling of matrices with a
/// norm outside [2^-405, 2^511 / 3] is not reproduced: kopf's covariances (output pixels²
/// within a window of half-width 2) are 0 or far inside that range.
///
/// Bitwise equal to numpy (Accelerate) on 10,000 random SPD matrices plus diagonal,
/// equal-diagonal and near-diagonal ones (tests/test_kopf_run_native.py).
pub fn eigh2(a: f64, b: f64, c: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    let mut d = [a, c];
    let mut z = [[1.0, 0.0], [0.0, 1.0]];
    let e = b;
    // dsteqr: the split test before the block is processed ...
    let mut split = e == 0.0 || e.abs() <= (d[0].abs().sqrt() * d[1].abs().sqrt()) * EPS;
    if !split {
        // ... and the "small subdiagonal" test of the QL (|d2| >= |d1|) or QR iteration.
        let tst = e.abs() * e.abs();
        let eps2 = EPS * EPS;
        split = if d[1].abs() < d[0].abs() {
            tst <= (eps2 * d[1].abs()) * d[0].abs() + SAFMIN
        } else {
            tst <= (eps2 * d[0].abs()) * d[1].abs() + SAFMIN
        };
    }
    if !split {
        let (rt1, rt2, cs, sn) = dlaev2(d[0], e, d[1]);
        d = [rt1, rt2];
        z = [[cs, -sn], [sn, cs]];
    }
    if d[1] < d[0] {
        d.swap(0, 1);
        for row in &mut z {
            row.swap(0, 1);
        }
    }
    (d, z)
}

/// `_clamp_singular_values` on one covariance stored as [c00, c01, c10, c11]: eigenvalues
/// clipped to [SINGULAR_MIN, SINGULAR_MAX], then `(evecs * evals) @ evecs.T`, each entry
/// `fma(v[i][1]·w1, v[j][1], v[i][0]·w0·v[j][0])` as numpy's BLAS computes the 2×2 product.
/// `eigh` reads the lower triangle (c10).
pub fn clamp_singular_values(cov: [f64; 4]) -> [f64; 4] {
    let (w, v) = eigh2(cov[0], cov[2], cov[3]);
    let w = w.map(|x| clip(x, SINGULAR_MIN, SINGULAR_MAX));
    let vw = [
        [v[0][0] * w[0], v[0][1] * w[1]],
        [v[1][0] * w[0], v[1][1] * w[1]],
    ];
    let entry = |i: usize, j: usize| libm::fma(vw[i][1], v[j][1], vw[i][0] * v[j][0]);
    [entry(0, 0), entry(0, 1), entry(1, 0), entry(1, 1)]
}

/// `np.clip(x, lo, hi)` (NaN stays NaN).
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

// ------------------------------------------------------------------ window bookkeeping

/// Per kernel along one axis: the range of pixels its R_k window contains (`_window`), and
/// the number of pixels it would span on an unbounded lattice (`_window_span`).
fn window_ranges(n_in: usize, n_out: usize) -> Vec<(usize, usize)> {
    let scale = n_out as f64 / n_in as f64;
    (0..n_out)
        .map(|k| {
            let center = k as f64 + 0.5;
            let inside = |i: usize| ((i as f64 + 0.5) * scale - center).abs() < RK_HALF_WIDTH;
            // Positions increase with i, so the window is one contiguous run.
            match (0..n_in).find(|&i| inside(i)) {
                None => (0, 0),
                Some(lo) => {
                    let hi = (lo..n_in).find(|&i| !inside(i)).unwrap_or(n_in);
                    (lo, hi)
                }
            }
        })
        .collect()
}

fn window_span(n_in: usize, n_out: usize) -> Vec<u64> {
    let scale = n_out as f64 / n_in as f64;
    let n = n_in as i64;
    (0..n_out)
        .map(|k| {
            let center = k as f64 + 0.5;
            (-n..2 * n)
                .filter(|&j| ((j as f64 + 0.5) * scale - center).abs() < RK_HALF_WIDTH)
                .count() as u64
        })
        .collect()
}

/// Inverse covariances (inv_a, inv_b, inv_c) of `_inverse_cov`, from c00, c01, c11.
fn inverse_cov(cov: &[[f64; 4]]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut inv_a = Vec::with_capacity(cov.len());
    let mut inv_b = Vec::with_capacity(cov.len());
    let mut inv_c = Vec::with_capacity(cov.len());
    for c in cov {
        let (a, b, d) = (c[0], c[1], c[3]);
        let det = a * d - b * b;
        inv_a.push(d / det);
        inv_b.push(-b / det);
        inv_c.push(a / det);
    }
    (inv_a, inv_b, inv_c)
}

/// `_log_iteration`: DEBUG every iteration, INFO every PROGRESS_EVERY; read-only.
fn log_iteration(it: usize, max_iters: usize, d_mu: f64, d_nu: f64, changed: f64) {
    let level = if it.is_multiple_of(PROGRESS_EVERY) {
        log::Level::Info
    } else {
        log::Level::Debug
    };
    if log::log_enabled!(target: LOG_TARGET, level) {
        log::log!(
            target: LOG_TARGET,
            level,
            "kopf iter {it}/{max_iters} rms dmu={} dnu={} sigma_changed={:.1}%",
            format_e2(d_mu),
            format_e2(d_nu),
            100.0 * changed
        );
    }
}

// ------------------------------------------------------------------ run

/// `kopf.run`: `lab` (hi, wi, 3) CIELAB, `mask` (hi, wi), `weight` (hi, wi) or None,
/// row-major. `threads` goes to the per-pixel loops (results do not depend on it).
#[allow(clippy::too_many_arguments)]
pub fn run(
    lab: &[f64],
    mask: &[bool],
    hi: usize,
    wi: usize,
    out_width: usize,
    out_height: usize,
    config: &Config,
    weight: Option<&[f64]>,
    threads: usize,
) -> Downscaled {
    assert_eq!(lab.len(), hi * wi * 3, "kopf::run: lab is not (hi, wi, 3)");
    assert_eq!(mask.len(), hi * wi, "kopf::run: mask is not (hi, wi)");
    if let Some(w) = weight {
        assert_eq!(w.len(), hi * wi, "kopf::run: weight is not (hi, wi)");
    }
    let (wo, ho) = (out_width, out_height);
    let m = ho * wo;
    let n = hi * wi;
    let max_iters = config.kopf_max_iters.max(0) as usize;
    let tol = config.kopf_tol;

    // to_unit_cube, as (3, hi, wi) planes.
    let mut channels = vec![0.0f64; 3 * n];
    for i in 0..n {
        channels[i] = lab[3 * i] / 100.0;
        channels[n + i] = (lab[3 * i + 1] + 128.0) / 255.0;
        channels[2 * n + i] = (lab[3 * i + 2] + 128.0) / 255.0;
    }
    let mask_u8: Vec<u8> = mask.iter().map(|&b| b as u8).collect();
    // Per-pixel weight (alpha) multiplied into γ after the E-step, so semi-transparent pixels
    // pull the kernels less. Skipped when it is 1 on every opaque pixel (γ is 0 elsewhere).
    let pixel_weight: Option<Vec<f64>> = weight.and_then(|w| {
        let all_one = (0..n).filter(|&i| mask[i]).all(|i| w[i] == 1.0);
        (!all_one).then(|| (0..n).map(|i| if mask[i] { w[i] } else { 0.0 }).collect())
    });

    // INITIALIZE
    let grid_x: Vec<f64> = (0..m).map(|k| (k % wo) as f64 + 0.5).collect();
    let grid_y: Vec<f64> = (0..m).map(|k| (k / wo) as f64 + 0.5).collect();
    let mut mu_x = grid_x.clone();
    let mut mu_y = grid_y.clone();
    // Σ per kernel as [c00, c01, c10, c11]: the clamp's reconstruction is not exactly
    // symmetric, and `_inverse_cov` reads c01 while `eigh` reads c10.
    let mut cov = vec![[COV_INIT, 0.0, 0.0, COV_INIT]; m];
    let mut nu = vec![0.5f64; m * 3];
    let mut sigma = vec![SIGMA_INIT; m];

    let mut gamma = vec![0.0f64; n * NSLOT]; // γ per slot, written by the E-step
    let mut sums = vec![0.0f64; m * EM_Q]; // M-step sums w, x, y, xx, xy, yy, c0, c1, c2
    let mut plus = vec![0.0f64; 4 * m];
    let mut both = vec![0.0f64; 4 * m];
    let mut overlap_sums = vec![0.0f64; 4 * m];
    let mut weight_sum = vec![0.0f64; m];
    let mut iterations = 0usize;
    let mut converged = false;
    let cos_limit = libm::cos(ORIENTATION_MAX_DEGREES * (std::f64::consts::PI / 180.0));
    let starve_below = STARVED_FRACTION * (hi as f64 / ho as f64) * (wi as f64 / wo as f64);
    // Pixels in each kernel's R_k window: opaque ones, and all in-image ones. The window is
    // separable, so the counts are products of 0/1 matrices (exact integers).
    let win_y = window_ranges(hi, ho);
    let win_x = window_ranges(wi, wo);
    let span_y = window_span(hi, ho);
    let span_x = window_span(wi, wo);
    // Opaque pixels of row y inside kernel column kx's window (mask @ win_x.T).
    let mut row_counts = vec![0u64; hi * wo];
    for y in 0..hi {
        let mut prefix = Vec::with_capacity(wi + 1);
        prefix.push(0u64);
        for x in 0..wi {
            prefix.push(prefix[x] + mask[y * wi + x] as u64);
        }
        for (kx, &(x0, x1)) in win_x.iter().enumerate() {
            row_counts[y * wo + kx] = prefix[x1] - prefix[x0];
        }
    }
    let mut has_pixels = vec![false; m];
    let mut truncated = vec![false; m];
    for (ky, &(y0, y1)) in win_y.iter().enumerate() {
        for (kx, &(x0, x1)) in win_x.iter().enumerate() {
            let k = ky * wo + kx;
            let opaque: u64 = (y0..y1).map(|y| row_counts[y * wo + kx]).sum();
            let in_image = ((y1 - y0) * (x1 - x0)) as u64;
            // A kernel is truncated when its R_k window holds transparent pixels or reaches
            // past the image border (fewer in-image pixels than the window spans); see the
            // shape constraints.
            truncated[k] = opaque < in_image || in_image < span_y[ky] * span_x[kx];
            has_pixels[k] = opaque > 0;
        }
    }
    // DEVIATION: Section 6 — a kernel whose own output cell is less than half opaque (the box
    // filter's rule) is a "ghost": it takes no part in the E-step, the shape constraints or
    // the starved re-seeding, and it is transparent in the output. Left in the EM, such a
    // kernel keeps its tiny initial σ, matches a dark line 2–3 px inside the silhouette
    // exactly, claims the line's pixels from its opaque neighbors, and is then dropped by
    // small_mask, so the line vanished from the output altogether (pure fill color on
    // 256→32 where box still showed it). Its neighbors now take those pixels.
    let cell = pixel_cells(hi, wi, ho, wo);
    let mut cell_total = vec![0u64; m];
    let mut cell_opaque = vec![0u64; m];
    for (i, &k) in cell.iter().enumerate() {
        cell_total[k] += 1;
        cell_opaque[k] += mask[i] as u64;
    }
    let ghost: Vec<bool> = (0..m).map(|k| cell_opaque[k] * 2 < cell_total[k]).collect();
    let ghost_u8: Vec<u8> = ghost.iter().map(|&g| g as u8).collect();
    let shaped: Vec<bool> = (0..m).map(|k| !truncated[k] && !ghost[k]).collect();

    for _ in 0..max_iters {
        iterations += 1;
        // ------------------------------------------------------- E-STEP + M-STEP SUMS
        // Native, per pixel: log-weights −½ (p_i − μ_k)ᵀ Σ_k⁻¹ (p_i − μ_k) − |c_i − ν_k|² / 2σ_k²
        // over the pixel's slot kernels (−∞ for slots outside R_k, ghost kernels and
        // transparent pixels), normalized per pixel into γ, times the pixel weight; then the
        // γ-weighted sums with offsets from the current μ_k.
        //
        // DEVIATION: Section 6 — the per-kernel normalization w_k(i) /= Σ_{i∈R_k} w_k(i) is
        // skipped. It rescales every kernel relative to its own best-fitting pixels, so a
        // kernel sitting on a thin feature (which covers < 50% of its cell) can never become
        // more selective for the feature than for the background, and the darkest kernel in
        // reach takes all feature pixels from its neighbors: the 2-px line of
        // test_thin_line_stays_connected comes out as isolated dark blobs with gaps. Without
        // it the kernels compete on absolute bilateral fit and latch onto the line. Weights
        // stay in log space so the tiny initial σ cannot underflow; the spec's "sum == 0 →
        // uniform over R_k" guard is replaced by the re-seeding of starved kernels below.
        // Normalize per pixel: γ_k(i) = w_k(i) / Σ_n w_n(i), from the log-weights.
        let (inv_a, inv_b, inv_c) = inverse_cov(&cov);
        let inv_color: Vec<f64> = sigma.iter().map(|&s| 1.0 / (2.0 * s * s)).collect();
        kopf::em_step(
            hi,
            wi,
            &Kernels {
                ho,
                wo,
                mu_x: &mu_x,
                mu_y: &mu_y,
                inv_a: &inv_a,
                inv_b: &inv_b,
                inv_c: &inv_c,
            },
            &channels,
            &mask_u8,
            pixel_weight.as_deref(),
            &nu,
            &inv_color,
            &ghost_u8,
            &mut gamma,
            &mut sums,
            threads,
        );

        // ------------------------------------------------------------------ M-STEP
        for k in 0..m {
            weight_sum[k] = sums[k * EM_Q];
        }
        let fed: Vec<bool> = weight_sum.iter().map(|&w| w > 1e-12).collect(); // starved kernels keep their previous parameters
        let wsafe: Vec<f64> = (0..m)
            .map(|k| if fed[k] { weight_sum[k] } else { 1.0 })
            .collect();
        let (old_mu_x, old_mu_y) = (mu_x.clone(), mu_y.clone());
        let old_nu = nu.clone();
        let old_sigma = sigma.clone();
        // Σ_k uses the OLD μ_k: the sums' offsets are from it.
        let mut m_mu_x = mu_x.clone();
        let mut m_mu_y = mu_y.clone();
        for k in 0..m {
            if !fed[k] {
                continue;
            }
            let s = &sums[k * EM_Q..(k + 1) * EM_Q];
            let w = wsafe[k];
            let xy = s[4] / w;
            cov[k] = [s[3] / w, xy, xy, s[5] / w];
            m_mu_x[k] = mu_x[k] + s[1] / w;
            m_mu_y[k] = mu_y[k] + s[2] / w;
            for c in 0..3 {
                nu[3 * k + c] = s[6 + c] / w;
            }
        }

        // ------------------------------------------------------------------ C-STEP
        // Spatial: Laplacian smoothing, then clamp to a box around the grid position.
        // DEVIATION: Section 6 — the neighbor mean is taken over displacements from the grid
        // position, not raw positions. The two are identical for interior kernels; for border
        // kernels (fewer than 4 neighbors) the raw mean lies inside the image and would drag
        // them inward every iteration, a disturbance that then diffuses across the whole grid
        // and keeps |Δμ| above kopf_tol.
        let off_x: Vec<f64> = (0..m).map(|k| m_mu_x[k] - grid_x[k]).collect();
        let off_y: Vec<f64> = (0..m).map(|k| m_mu_y[k] - grid_y[k]).collect();
        let mean_x = neighbor_mean4(&off_x, ho, wo);
        let mean_y = neighbor_mean4(&off_y, ho, wo);
        for k in 0..m {
            mu_x[k] = grid_x[k] + clip(0.5 * off_x[k] + 0.5 * mean_x[k], -CLAMP_BOX, CLAMP_BOX);
            mu_y[k] = grid_y[k] + clip(0.5 * off_y[k] + 0.5 * mean_y[k], -CLAMP_BOX, CLAMP_BOX);
        }
        for c in cov.iter_mut() {
            *c = clamp_singular_values(*c);
        }

        // Shape constraints, with (p_i − μ_k) relative to the constrained μ_k. Native, per
        // pixel and slot: (dx, dy) = (p_i − old μ_k) − (μ_k − old μ_k), in this order so that
        // the result does not depend on whether the offsets were stored or are recomputed;
        // u = (dx, dy)·e for each axis e below; both = Σ γ u², plus = Σ_{u > 0} γ u²; and for
        // each neighbor direction (0, 1), (1, 0), (1, 1), (1, −1), Σ_i γ_k(i) γ_n(i).
        let move_x: Vec<f64> = (0..m).map(|k| mu_x[k] - old_mu_x[k]).collect();
        let move_y: Vec<f64> = (0..m).map(|k| mu_y[k] - old_mu_y[k]).collect();
        kopf::c_step(
            hi,
            wi,
            ho,
            wo,
            &gamma,
            &old_mu_x,
            &old_mu_y,
            &move_x,
            &move_y,
            &mut plus,
            &mut both,
            &mut overlap_sums,
            threads,
        );
        // Axes (ndy, ndx) = (0, 1), (1, 0), (1, 1), (−1, 1), i.e. e = (1, 0), (0, 1),
        // (√½, √½), (√½, −√½): `plus[a]` is the one-sided variance toward (ndy, ndx),
        // `both[a] − plus[a]` toward (−ndy, −ndx).
        let toward = |ndy: i64, ndx: i64, k: usize| -> f64 {
            match (ndy, ndx) {
                (0, 1) => plus[k],
                (0, -1) => both[k] - plus[k],
                (1, 0) => plus[m + k],
                (-1, 0) => both[m + k] - plus[m + k],
                (1, 1) => plus[2 * m + k],
                (-1, -1) => both[2 * m + k] - plus[2 * m + k],
                (-1, 1) => plus[3 * m + k],
                (1, -1) => both[3 * m + k] - plus[3 * m + k],
                _ => unreachable!("not a neighbor direction"),
            }
        };
        // Overlap with the neighbor n = k + (ndy, ndx): stored for (0, 1), (1, 0), (1, 1),
        // (1, −1) at k; the opposite directions read it at n (f is symmetric in (k, n)).
        let overlap = |ndy: i64, ndx: i64, k: usize, n: usize| -> f64 {
            match (ndy, ndx) {
                (0, 1) => overlap_sums[k],
                (0, -1) => overlap_sums[n],
                (1, 0) => overlap_sums[m + k],
                (-1, 0) => overlap_sums[m + n],
                (1, 1) => overlap_sums[2 * m + k],
                (-1, -1) => overlap_sums[2 * m + n],
                (1, -1) => overlap_sums[3 * m + k],
                (-1, 1) => overlap_sums[3 * m + n],
                _ => unreachable!("not a neighbor direction"),
            }
        };

        // DEVIATION: Section 6 — s and f are divided by Σ_i γ_k(i). As literal sums they scale
        // with rx·ry (≈ the number of input pixels per kernel), so the thresholds 0.2 and
        // 0.08 would fire for every kernel on every iteration and σ would saturate at its
        // cap, turning the method into a box filter. Normalized, s is a directional variance
        // in output units² and f a mean overlap, which is what the thresholds are scaled for.
        //
        // DEVIATION: Section 6 — the edge orientation o = Σ_{i∈R_k} ∇(γ_k / (γ_k + γ_n)) is
        // replaced by the direction between the two kernels' M-step centroids. The summed
        // gradient telescopes to the ratio on the border of the R_k window, where both γ are
        // (numerically) zero, so the literal estimate is noise: it fired for thousands of
        // kernels per iteration on a plain checkerboard and steadily blurred a result that was
        // exact after five iterations. The integral it stands for is the net normal of the
        // boundary between the two kernels' pixel sets, which is what the centroid direction
        // measures robustly.
        //
        // DEVIATION: Section 6 — both tests are skipped for kernels whose R_k window is
        // truncated by transparent pixels or the image border, and the orientation test is
        // skipped for pairs that share no pixels (f == 0: there is no edge between them).
        // A truncated window gives the kernel a lopsided footprint, so the one-sided variance
        // toward the cut side and the centroid direction are set by the cut, not by the
        // content, and growing σ never changes that geometry: on circle_alpha 256→32 the same
        // ~340 kernels in a 2–4-kernel band along the silhouette fired on every iteration,
        // 248 ended at SIGMA_CAP, and those kernels lost their color selectivity (a dark line
        // 4 px inside the silhouette came out fainter than with the box filter).
        let mut fires = vec![0u32; m];
        let whole: Vec<bool> = (0..m).map(|k| fed[k] && shaped[k]).collect();
        for &(ndy, ndx) in &NEIGHBORS8 {
            let norm = hypot(ndx as f64, ndy as f64);
            let (d_x, d_y) = (ndx as f64 / norm, ndy as f64 / norm);
            for ky in 0..ho {
                let ny = ky as i64 + ndy;
                for kx in 0..wo {
                    let nx = kx as i64 + ndx;
                    // Both tests need the neighbor to exist.
                    if ny < 0 || ny >= ho as i64 || nx < 0 || nx >= wo as i64 {
                        continue;
                    }
                    let k = ky * wo + kx;
                    let nb = ny as usize * wo + nx as usize;
                    let pair = whole[k] && shaped[nb];
                    let s_dir = toward(ndy, ndx, k) / wsafe[k];
                    let f = overlap(ndy, ndx, k, nb) / wsafe[k];
                    let o_x = m_mu_x[nb] - m_mu_x[k];
                    let o_y = m_mu_y[nb] - m_mu_y[k];
                    let o_len = hypot(o_x, o_y);
                    let cos_angle = (d_x * o_x + d_y * o_y) / if o_len > 0.0 { o_len } else { 1.0 };
                    let misaligned = o_len > 0.0 && cos_angle < cos_limit;
                    let fire = (whole[k] && s_dir > DIRECTIONAL_VARIANCE_MAX)
                        || (pair && f > 0.0 && f < OVERLAP_MIN && misaligned);
                    if fire {
                        fires[k] += 1; // σ_k *= 1.1
                        fires[nb] += 1; // σ_n *= 1.1
                    }
                }
            }
        }
        for k in 0..m {
            let grown = sigma[k] * growth_pow(fires[k]);
            // np.minimum: NaN propagates.
            sigma[k] = if grown > SIGMA_CAP { SIGMA_CAP } else { grown };
        }

        // Starved kernels: no pixel prefers the kernel's color mean (its Σ γ is ~0), typically
        // because that mean is a blend of two colors while a neighbor matches each of them
        // exactly. In place of the spec's "sum == 0 → uniform over R_k" guard, the kernel is
        // re-seeded with the color of the input pixel under its center, which it then matches
        // exactly and competes for on spatial terms alone.
        let (ry, rx) = (hi as f64 / ho as f64, wi as f64 / wo as f64);
        for k in 0..m {
            if !(has_pixels[k] && !ghost[k] && weight_sum[k] < starve_below) {
                continue;
            }
            let seed_y = floor_index(mu_y[k] * ry, hi);
            let seed_x = floor_index(mu_x[k] * rx, wi);
            let i = seed_y * wi + seed_x;
            if mask[i] {
                for c in 0..3 {
                    nu[3 * k + c] = channels[c * n + i];
                }
            }
        }

        // ------------------------------------------------------------- CONVERGENCE
        // DEVIATION: Section 6 — the spec's test (max |Δμ| and max |Δν| < kopf_tol, no σ
        // change) is never met on real content: border truncation and non-integer ratios
        // keep a few centroids moving by ~1e-3 to 5e-2 output pixels, and a handful of
        // kernels grow σ every iteration. Converged instead means: RMS |Δμ| over fed kernels
        // < 10·kopf_tol output pixels, RMS |Δν| (unit-cube color) < kopf_tol, and σ changed
        // on fewer than CHANGED_SIGMA_FRACTION of the kernels (those already at the cap
        // cannot change and do not count).
        let fed_k: Vec<usize> = (0..m).filter(|&k| fed[k]).collect();
        let (d_mu, d_nu) = if fed_k.is_empty() {
            (0.0, 0.0)
        } else {
            let dmu: Vec<f64> = fed_k
                .iter()
                .map(|&k| {
                    let (dx, dy) = (mu_x[k] - old_mu_x[k], mu_y[k] - old_mu_y[k]);
                    dx * dx + dy * dy
                })
                .collect();
            let dnu: Vec<f64> = fed_k
                .iter()
                .map(|&k| {
                    let d = |c: usize| {
                        let v = nu[3 * k + c] - old_nu[3 * k + c];
                        v * v
                    };
                    (d(0) + d(1)) + d(2)
                })
                .collect();
            (pairwise_mean(&dmu).sqrt(), pairwise_mean(&dnu).sqrt())
        };
        let sigma_changed = (0..m)
            .filter(|&k| sigma[k] != old_sigma[k] && old_sigma[k] < SIGMA_CAP)
            .count();
        log_iteration(
            iterations,
            max_iters,
            d_mu,
            d_nu,
            sigma_changed as f64 / m as f64,
        );
        if d_mu < 10.0 * tol
            && d_nu < tol
            && (sigma_changed as f64) < CHANGED_SIGMA_FRACTION * ho as f64 * wo as f64
        {
            converged = true;
            break;
        }
    }

    log::debug!(
        target: LOG_TARGET,
        "kopf {} after {iterations} iterations",
        if converged {
            "converged"
        } else {
            "stopped at the cap"
        }
    );

    // OUTPUT: small_mask[k] = Σ γ_k(i)·mask_i / Σ γ_k(i) >= 0.5.
    // DEVIATION: Section 6 — transparent pixels are excluded from R_k, so with the EM's own γ
    // that ratio is always 1. small_mask is instead the cell rule that defines the ghost
    // kernels: a kernel is opaque iff its own output cell is at least half opaque, the same
    // mask as the box filter. An earlier version measured the coverage with spatial-only
    // responsibilities; it kept kernels whose cell was as little as 19 % opaque (the ghosts'
    // problem above) and, for cells exactly half opaque, dropped kernels on some rows and not
    // others after they had claimed a line's pixels. Kernels that ended up starved take their
    // color from spatial-only responsibilities (the final μ, Σ without the color term).
    let small_mask: Vec<bool> = ghost.iter().map(|&g| !g).collect();
    let starved: Vec<bool> = (0..m)
        .map(|k| weight_sum[k] < starve_below && small_mask[k])
        .collect();
    if starved.iter().any(|&s| s) {
        let (inv_a, inv_b, inv_c) = inverse_cov(&cov);
        // Native: γ from the spatial term alone over every in-image slot (no pixel mask, no
        // ghost test), normalized per pixel; colors are then weighted by alpha (by the binary
        // mask when there is no weight).
        let pixel_mass: Vec<f64> = match &pixel_weight {
            Some(w) => w.clone(),
            None => mask.iter().map(|&b| b as u8 as f64).collect(),
        };
        let mut mass_w = vec![0.0f64; m];
        let mut color_w = vec![0.0f64; m * 3];
        kopf::spatial_refill(
            hi,
            wi,
            &Kernels {
                ho,
                wo,
                mu_x: &mu_x,
                mu_y: &mu_y,
                inv_a: &inv_a,
                inv_b: &inv_b,
                inv_c: &inv_c,
            },
            &channels,
            &pixel_mass,
            &mut mass_w,
            &mut color_w,
            threads,
        );
        for k in 0..m {
            if starved[k] && mass_w[k] > 0.0 {
                for c in 0..3 {
                    nu[3 * k + c] = color_w[3 * k + c] / mass_w[k];
                }
            }
        }
    }

    // from_unit_cube
    let mut small_lab = vec![0.0f64; m * 3];
    for k in 0..m {
        small_lab[3 * k] = nu[3 * k] * 100.0;
        small_lab[3 * k + 1] = nu[3 * k + 1] * 255.0 - 128.0;
        small_lab[3 * k + 2] = nu[3 * k + 2] * 255.0 - 128.0;
    }
    let sigma_capped = sigma.iter().filter(|&&s| s >= SIGMA_CAP).count() as i64;
    Downscaled {
        h: ho,
        w: wo,
        small_lab,
        small_mask,
        palette_lab: None,
        palette_lab_unsaturated: None,
        indices: None,
        mean_lab: None,
        stats: vec![
            ("iterations", StatValue::Int(iterations as i64)),
            ("converged", StatValue::Bool(converged)),
            ("sigma_capped", StatValue::Int(sigma_capped)),
        ],
    }
}

/// `np.clip(np.floor(v).astype(np.int64), 0, n - 1)`.
fn floor_index(v: f64, n: usize) -> usize {
    let f = v.floor();
    if f < 0.0 {
        0
    } else if f >= (n - 1) as f64 {
        n - 1
    } else {
        f as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_table_is_correctly_rounded() {
        // 1.1 = m · 2^-52 exactly; m^n as a big integer, rounded half to even to 53 bits.
        let m = (SIGMA_GROWTH.to_bits() & ((1 << 52) - 1)) | (1 << 52);
        let mut big: Vec<u64> = vec![1]; // little-endian limbs
        for (n, &bits) in GROWTH_POW.iter().enumerate() {
            let len = (big.len() - 1) * 64 + (64 - big.last().unwrap().leading_zeros() as usize);
            let shift = len.saturating_sub(53);
            let bit = |i: usize| (big[i / 64] >> (i % 64)) & 1;
            let mut q: u64 = (shift..len).rev().fold(0, |acc, i| (acc << 1) | bit(i));
            let half = shift > 0 && bit(shift - 1) == 1;
            let sticky = (0..shift.saturating_sub(1)).any(|i| bit(i) == 1);
            if half && (sticky || q & 1 == 1) {
                q += 1;
            }
            // value = q · 2^(shift − 52n); q <= 2^53 is exact in f64.
            let exp = shift as i64 - 52 * n as i64;
            let scale = f64::from_bits(((exp + 1023) as u64) << 52);
            let want = q as f64 * scale;
            assert_eq!(f64::from_bits(bits), want, "n = {n}");
            assert_eq!(growth_pow(n as u32), want);
            // big *= m
            let mut carry = 0u128;
            for limb in big.iter_mut() {
                let v = *limb as u128 * m as u128 + carry;
                *limb = v as u64;
                carry = v >> 64;
            }
            if carry > 0 {
                big.push(carry as u64);
            }
        }
        assert_eq!(growth_pow(1), 1.1);
        // libm's pow is 1 ulp off at n = 4: the reason for the table.
        assert_ne!(libm::pow(SIGMA_GROWTH, 4.0), growth_pow(4));
    }

    #[test]
    fn format_e2_matches_python() {
        // Expected strings from Python's '%.2e' % x.
        let cases = [
            (1.125, "1.12e+00"),
            (1.135, "1.14e+00"),
            (1.145, "1.15e+00"),
            (0.0, "0.00e+00"),
            (1.0e-300, "1.00e-300"),
            (9.995, "9.99e+00"),
            (9.9951, "1.00e+01"),
            (0.000123456, "1.23e-04"),
            (12345.0, "1.23e+04"),
            (2.5e-5, "2.50e-05"),
            (1e100, "1.00e+100"),
            (0.0005, "5.00e-04"),
            (0.000125, "1.25e-04"),
            (3.0517578125e-05, "3.05e-05"),
            (f64::INFINITY, "inf"),
        ];
        for (x, want) in cases {
            assert_eq!(format_e2(x), want, "{x}");
        }
    }

    #[test]
    fn percent_format_matches_python() {
        // Python's '%.1f' rounds the exact binary value half to even, as Rust does.
        let cases = [
            (0.25, "0.2"),
            (0.35, "0.3"),
            (0.05, "0.1"),
            (0.15, "0.1"),
            (12.25, "12.2"),
            (0.125, "0.1"),
            (0.0, "0.0"),
            (100.0, "100.0"),
            (2.34375, "2.3"),
            (0.390625, "0.4"),
        ];
        for (x, want) in cases {
            assert_eq!(format!("{x:.1}"), want, "{x}");
        }
    }

    #[test]
    fn eigh2_diagonal_and_sorted() {
        let (w, v) = eigh2(0.3, 0.0, 0.1);
        assert_eq!(w, [0.1, 0.3]);
        assert_eq!(v, [[0.0, 1.0], [1.0, 0.0]]);
        let (w, v) = eigh2(0.1, 0.0, 0.3);
        assert_eq!(w, [0.1, 0.3]);
        assert_eq!(v, [[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn eigh2_decomposes() {
        let (a, b, c) = (0.21, -0.07, 0.13);
        let (w, v) = eigh2(a, b, c);
        assert!(w[0] <= w[1]);
        for j in 0..2 {
            let (x, y) = (v[0][j], v[1][j]);
            assert!((a * x + b * y - w[j] * x).abs() < 1e-15);
            assert!((b * x + c * y - w[j] * y).abs() < 1e-15);
            assert!((x * x + y * y - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn clamp_keeps_eigenvalues_in_range() {
        let out = clamp_singular_values([1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0]);
        assert_eq!(out, [0.1, 0.0, 0.0, 0.1]);
    }

    #[test]
    fn hypot_of_unit_steps() {
        assert_eq!(hypot(1.0, 1.0), std::f64::consts::SQRT_2);
        assert_eq!(hypot(0.0, -1.0), 1.0);
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(0.0, 0.0), 0.0);
    }

    #[test]
    fn window_ranges_match_the_distance_test() {
        for (n_in, n_out) in [(64, 16), (50, 32), (5, 3), (300, 299)] {
            let scale = n_out as f64 / n_in as f64;
            for (k, &(lo, hi)) in window_ranges(n_in, n_out).iter().enumerate() {
                for i in 0..n_in {
                    let inside = ((i as f64 + 0.5) * scale - (k as f64 + 0.5)).abs() < 2.0;
                    assert_eq!(inside, (lo..hi).contains(&i));
                }
            }
        }
    }
}
