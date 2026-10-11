//! gerstner's control flow (`pixelforge.downscale.gerstner.run`) over the native loops of
//! [`crate::gerstner`]: the grid, the INITIALIZE pass, the iteration loop (one SLIC
//! assignment, one statistics pass, the centre smoothing, the superpixel bilateral, one
//! annealer step), the transparent-pixel attribution and the result.
//!
//! Ported operation for operation from the Python (Phase 5): sums over pixels are
//! sequential in row-major order like `np.bincount`, `coverage.sum()` is numpy's pairwise
//! sum, and the palette half lives in [`crate::annealer`] (where the replaced BLAS/LAPACK
//! routines are listed).

use crate::annealer::{PaletteAnnealer, Tunables};
use crate::config::Config;
use crate::downscale::{neighbor_mean4, Downscaled, StatValue};
use crate::gerstner::{bilateral_filter, slic_assign, superpixel_stats};

/// Grid cells searched around a pixel's home cell (safety bound).
pub const MAX_SEARCH_RADIUS: i64 = 6;
/// Iterations between INFO progress lines (DEBUG logs every one).
pub const PROGRESS_EVERY: i64 = 10;

/// Output cell of every input coordinate along one axis (`box.cell_index`).
fn cell_index(n_in: usize, n_out: usize) -> Vec<i64> {
    (0..n_in as i64)
        .map(|i| i * n_out as i64 / n_in as i64)
        .collect()
}

/// Geometry shared by the SLIC assignment passes (`gerstner._Grid`).
struct Grid {
    ho: usize,
    wo: usize,
    rx: f64,
    ry: f64,
    cx0: Vec<f64>,
    cy0: Vec<f64>,
    px: Vec<f64>,
    py: Vec<f64>,
    home_x: Vec<i64>,
    home_y: Vec<i64>,
}

impl Grid {
    fn new(hi: usize, wi: usize, ho: usize, wo: usize) -> Grid {
        let rx = wi as f64 / wo as f64;
        let ry = hi as f64 / ho as f64;
        let mut cx0 = Vec::with_capacity(ho * wo);
        let mut cy0 = Vec::with_capacity(ho * wo);
        for y in 0..ho {
            for x in 0..wo {
                cx0.push((x as f64 + 0.5) * rx);
                cy0.push((y as f64 + 0.5) * ry);
            }
        }
        Grid {
            ho,
            wo,
            rx,
            ry,
            cx0,
            cy0,
            px: (0..wi).map(|x| x as f64 + 0.5).collect(),
            py: (0..hi).map(|y| y as f64 + 0.5).collect(),
            home_x: cell_index(wi, wo),
            home_y: cell_index(hi, ho),
        }
    }

    /// Grid cells around the home cell that can hold a center inside the 2rx × 2ry window.
    ///
    /// A center that drifted D cells from its initial grid position can only be within the
    /// window of pixels whose home cell is < 1.5 + D cells away.
    fn search_radius(&self, cx: &[f64], cy: &[f64]) -> i64 {
        // np.abs(a - b).max() (NaN propagates, like numpy's maximum).
        let max_abs = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(
                    f64::NEG_INFINITY,
                    |m, v| if v > m || v.is_nan() { v } else { m },
                )
        };
        let dx = max_abs(cx, &self.cx0) / self.rx;
        let dy = max_abs(cy, &self.cy0) / self.ry;
        let drift = if dy > dx { dy } else { dx }; // Python's max(dx, dy)
        let reach = (1.5 + drift + 1e-9).floor();
        if reach >= MAX_SEARCH_RADIUS as f64 {
            MAX_SEARCH_RADIUS
        } else {
            reach as i64
        }
    }

    /// `_assign`: one modified-SLIC pass; `candidate_ok` None = every center may win.
    #[allow(clippy::too_many_arguments)]
    fn assign(
        &self,
        cx: &[f64],
        cy: &[f64],
        radius: i64,
        candidate_ok: Option<&[bool]>,
        color: Option<(&[f64], &[f64])>,
        coef: f64,
        windowed: bool,
    ) -> Vec<i64> {
        let center_x: Vec<f64> = match candidate_ok {
            // An infinitely distant center fails the window test and never wins.
            Some(ok) => cx
                .iter()
                .zip(ok)
                .map(|(&x, &ok)| if ok { x } else { f64::INFINITY })
                .collect(),
            None => cx.to_vec(),
        };
        slic_assign(
            &self.home_y,
            &self.home_x,
            &self.px,
            &self.py,
            &center_x,
            cy,
            self.ho,
            self.wo,
            radius,
            self.rx,
            self.ry,
            coef,
            windowed,
            color,
        )
    }
}

/// Move every value `fraction` of the way toward the mean of its 4-connected neighbours.
fn laplacian_smooth(grid: &[f64], h: usize, w: usize, fraction: f64) -> Vec<f64> {
    let mean = neighbor_mean4(grid, h, w);
    grid.iter()
        .zip(&mean)
        .map(|(&g, &m)| g + fraction * (m - g))
        .collect()
}

/// P(p_s) over the active superpixels from their mean pixel weight (alpha).
///
/// Uniform (exactly 1/N) when every superpixel is fully opaque, as in the spec; otherwise
/// proportional to the coverage, so a superpixel of semi-transparent fringe pixels holds
/// less palette mass. Superpixels lying entirely in the transparent region are not active.
pub fn palette_prior(coverage: &[f64]) -> Vec<f64> {
    // DEVIATION: Section 7 — the spec's uniform prior is weighted by alpha coverage. It is
    // the mean weight, not the total, so that equal-alpha superpixels of different pixel
    // counts keep the spec's equal mass (and an opaque input is bit-identical).
    if coverage.iter().all(|&c| c == 1.0) {
        return vec![1.0 / coverage.len() as f64; coverage.len()];
    }
    let total = crate::npsum::pairwise_sum(coverage);
    coverage.iter().map(|&c| c / total).collect()
}

/// Python's `'%.{precision}g' % x` (for the progress lines).
pub fn format_g(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let p = precision.max(1);
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    let sci = format!("{:.*e}", p - 1, x);
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("integer exponent");
    if exp < -4 || exp >= p as i32 {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa), exp.abs())
    } else {
        strip(&format!("{:.*}", (p as i32 - 1 - exp) as usize, x))
    }
}

/// The error `run` reports (the binding raises it as `PixelforgeError`).
#[derive(Debug, Clone, PartialEq)]
pub struct GerstnerError(pub String);

impl std::fmt::Display for GerstnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GerstnerError {}

fn flat_colors(colors: &[[f64; 3]], k: &[usize]) -> Vec<f64> {
    k.iter().flat_map(|&i| colors[i]).collect()
}

/// Joint superpixel + palette optimization (Gerstner et al. 2013), REQUIREMENTS.md
/// Section 7: `pixelforge.downscale.gerstner.run`.
///
/// The input `lab` (h, w, 3) is segmented into exactly out_w·out_h superpixels (one per
/// output pixel) while a K-colour palette is grown by mass-constrained deterministic
/// annealing ([`PaletteAnnealer`]). `mask` (h, w) marks the opaque pixels; `weight` (h, w),
/// the input alpha, scales how much each opaque pixel counts (None = 1 everywhere).
#[allow(clippy::too_many_arguments)]
pub fn run(
    lab: &[f64],
    mask: &[bool],
    h: usize,
    w: usize,
    out_w: usize,
    out_h: usize,
    config: &Config,
    weight: Option<&[f64]>,
) -> Result<Downscaled, GerstnerError> {
    run_with(
        lab,
        mask,
        h,
        w,
        out_w,
        out_h,
        config,
        weight,
        Tunables::default(),
    )
}

/// [`run`] with the annealer's tunable constants given explicitly.
#[allow(clippy::too_many_arguments)]
pub fn run_with(
    lab: &[f64],
    mask: &[bool],
    h: usize,
    w: usize,
    out_w: usize,
    out_h: usize,
    config: &Config,
    weight: Option<&[f64]>,
    tunables: Tunables,
) -> Result<Downscaled, GerstnerError> {
    let (hi, wi) = (h, w);
    assert!(
        lab.len() == hi * wi * 3 && mask.len() == hi * wi,
        "lab must be (h, w, 3) and mask (h, w)"
    );
    assert!(
        weight.is_none_or(|wt| wt.len() == hi * wi),
        "weight must be (h, w)"
    );
    assert!(out_w > 0 && out_h > 0, "the output size must be positive");
    if !mask.iter().any(|&m| m) {
        return Err(GerstnerError("no opaque pixels".into()));
    }
    let (wo, ho) = (out_w, out_h);
    let n_sp = wo * ho;
    let grid = Grid::new(hi, wi, ho, wo);
    let coef = config.g_m * (n_sp as f64 / (hi * wi) as f64).sqrt();

    // Opaque pixels in row-major order (np.nonzero(mask)) and their colours.
    let opaque: Vec<usize> = (0..hi * wi).filter(|&i| mask[i]).collect();
    let col: Vec<[f64; 3]> = opaque
        .iter()
        .map(|&i| [lab[3 * i], lab[3 * i + 1], lab[3 * i + 2]])
        .collect();
    let plane = hi * wi;
    let mut channels = vec![0.0_f64; 3 * plane]; // (3, hi, wi)
    for i in 0..plane {
        channels[i] = lab[3 * i];
        channels[plane + i] = lab[3 * i + 1];
        channels[2 * plane + i] = lab[3 * i + 2];
    }
    // Pixel weight (alpha) of the opaque pixels: superpixel means and centers are weighted by
    // it, and so is each superpixel's share of the palette prior. None when it is all ones.
    let pix_w: Option<Vec<f64>> = weight
        .map(|wt| opaque.iter().map(|&i| wt[i]).collect::<Vec<f64>>())
        .filter(|pw| !pw.iter().all(|&v| v == 1.0));
    // Inputs of the per-iteration statistics pass.
    let stats_w = if pix_w.is_some() { weight } else { None };

    // INITIALIZE: regular grid of centers, every pixel assigned to the nearest center.
    let mut cx = grid.cx0.clone();
    let mut cy = grid.cy0.clone();
    let mut count = vec![0_i64; n_sp];
    let mut mass = vec![0.0_f64; n_sp];
    let mut sums = vec![[0.0_f64; 3]; n_sp];
    // np.bincount: sequential per bin in pixel order.
    for (j, &i) in opaque.iter().enumerate() {
        let s = (grid.home_y[i / wi] as usize) * wo + grid.home_x[i % wi] as usize;
        count[s] += 1;
        match &pix_w {
            Some(pw) => {
                let wv = pw[j];
                mass[s] += wv;
                for c in 0..3 {
                    sums[s][c] += wv * col[j][c];
                }
            }
            None => {
                for c in 0..3 {
                    sums[s][c] += col[j][c];
                }
            }
        }
    }
    if pix_w.is_none() {
        for s in 0..n_sp {
            mass[s] = count[s] as f64;
        }
    }
    let mut has_color: Vec<bool> = count.iter().map(|&c| c > 0).collect();
    // Mean pixel weight of each superpixel (1 for an opaque input), kept for those that
    // lose all their pixels in an iteration, like their color.
    let mut coverage = vec![1.0_f64; n_sp];
    let mut mean = vec![0.0_f64; 3 * n_sp];
    for s in 0..n_sp {
        if has_color[s] {
            if pix_w.is_some() {
                coverage[s] = mass[s] / count[s] as f64;
            }
            for c in 0..3 {
                mean[3 * s + c] = sums[s][c] / mass[s];
            }
        }
    }

    let palette_size = usize::try_from(config.palette_size).expect("palette_size >= 2");
    let mut annealer = PaletteAnnealer::with_tunables(&col, palette_size, config, tunables);
    let mut k = vec![0usize; n_sp];
    let mut sp_color = flat_colors(&annealer.colors(), &k);
    let mut smooth = mean.clone();
    let mut iterations: i64 = 0;
    let mut hit_cap = true;
    let mut active_grid = vec![0.0_f64; n_sp];

    for _ in 0..config.g_max_iters.max(0) {
        // 1. SUPERPIXEL REFINEMENT (one modified-SLIC step).
        let radius = grid.search_radius(&cx, &cy);
        let assigned_to = grid.assign(
            &cx,
            &cy,
            radius,
            Some(&has_color),
            Some((&channels, &sp_color)),
            coef,
            true,
        );
        // Masked (transparent) input pixels are never assigned and never contribute. One
        // pass sums in pixel order, bitwise equal to np.bincount over the assigned pixels.
        let st = superpixel_stats(&assigned_to, mask, stats_w, &channels, hi, wi, n_sp);
        // Superpixels with zero pixels keep their previous center and color. Without
        // weights st.mass is the count (sums of 1.0), as `mass = count` in Python.
        for s in 0..n_sp {
            if st.count[s] > 0 {
                let m = st.mass[s];
                for c in 0..3 {
                    mean[3 * s + c] = st.sums[3 * s + c] / m;
                }
                cx[s] = st.sum_x[s] / m;
                cy[s] = st.sum_y[s] / m;
                if stats_w.is_some() {
                    coverage[s] = m / st.count[s] as f64;
                }
                has_color[s] = true;
            }
        }
        count = st.count;

        cx = laplacian_smooth(&cx, ho, wo, config.g_laplacian);
        cy = laplacian_smooth(&cy, ho, wo, config.g_laplacian);

        for (a, &has) in active_grid.iter_mut().zip(&has_color) {
            *a = if has { 1.0 } else { 0.0 };
        }
        smooth = bilateral_filter(
            &mean,
            &active_grid,
            ho,
            wo,
            config.g_bilateral_sigma_color,
            config.g_bilateral_sigma_spatial,
        );

        // 2-4. ASSOCIATE, REFINE, CONVERGENCE CHECK / EXPAND on the active superpixels.
        let active: Vec<usize> = (0..n_sp).filter(|&s| has_color[s]).collect();
        let points: Vec<[f64; 3]> = active
            .iter()
            .map(|&s| [smooth[3 * s], smooth[3 * s + 1], smooth[3 * s + 2]])
            .collect();
        let active_coverage: Vec<f64> = active.iter().map(|&s| coverage[s]).collect();
        annealer.step(&points, &palette_prior(&active_coverage));
        for (&s, &a) in active.iter().zip(&annealer.assign) {
            k[s] = a;
        }
        sp_color = flat_colors(&annealer.colors(), &k);
        iterations += 1;
        let level = if iterations % PROGRESS_EVERY == 0 {
            log::Level::Info
        } else {
            log::Level::Debug
        };
        // Read-only: logging never affects the result (and is not formatted when disabled).
        log::log!(
            target: "pixelforge::gerstner",
            level,
            "gerstner iter {}/{} T={} colors={}",
            iterations,
            config.g_max_iters,
            format_g(annealer.t, 3),
            annealer.n_colors()
        );
        if annealer.done {
            hit_cap = false;
            break;
        }
    }
    log::debug!(
        target: "pixelforge::gerstner",
        "gerstner {} after {} iterations",
        if hit_cap { "stopped at the cap" } else { "converged" },
        iterations
    );

    // POST: saturation, then indices / palette / mask.
    let unsaturated = annealer.colors();
    let beta = config.saturation_beta;
    let palette: Vec<[f64; 3]> = unsaturated
        .iter()
        .map(|c| [c[0], c[1] * beta, c[2] * beta])
        .collect();

    // DEVIATION: Section 7 — the spec does not say how transparent pixels count toward a
    // superpixel (and its prior is uniform over the superpixels holding opaque pixels).
    // small_mask[s] = opaque fraction of superpixel s >= 0.5. Transparent pixels are never
    // assigned during the optimization, so attribute each to its spatially nearest center.
    let mut transparent = vec![0_i64; n_sp];
    if !mask.iter().all(|&m| m) {
        let radius = grid.search_radius(&cx, &cy);
        let nearest = grid.assign(&cx, &cy, radius, None, None, 1.0, false);
        for (&s, &opaque) in nearest.iter().zip(mask) {
            if !opaque && s >= 0 {
                transparent[s as usize] += 1;
            }
        }
    }
    let small_mask: Vec<bool> = (0..n_sp)
        .map(|s| has_color[s] && count[s] * 2 >= count[s] + transparent[s])
        .collect();

    Ok(Downscaled {
        h: ho,
        w: wo,
        small_lab: flat_colors(&palette, &k),
        small_mask,
        palette_lab: Some(palette.iter().flatten().copied().collect()),
        palette_lab_unsaturated: Some(unsaturated.iter().flatten().copied().collect()),
        indices: Some(k.iter().map(|&i| i as i64).collect()),
        mean_lab: Some(smooth),
        stats: vec![
            ("iterations", StatValue::Int(iterations)),
            ("palette_size", StatValue::Int(palette.len() as i64)),
            ("hit_iteration_cap", StatValue::Bool(hit_cap)),
            ("final_temperature", StatValue::Float(annealer.t)),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Value;

    fn config(kw: &[(&str, Value)]) -> Config {
        let kw: Vec<(String, Value)> = kw.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        Config::new(&kw).unwrap()
    }

    fn halves(l: [f64; 2], a: [f64; 2]) -> Vec<f64> {
        let (h, w) = (32, 32);
        let mut lab = vec![0.0; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                let i = 3 * (y * w + x);
                let side = usize::from(x >= 16);
                lab[i] = l[side];
                lab[i + 1] = a[side];
            }
        }
        lab
    }

    #[test]
    fn format_g_matches_python() {
        assert_eq!(format_g(0.9251062652610845, 3), "0.925");
        assert_eq!(format_g(123.456, 3), "123");
        assert_eq!(format_g(1234.5, 3), "1.23e+03");
        assert_eq!(format_g(1.5e-5, 3), "1.5e-05");
        assert_eq!(format_g(0.0001, 3), "0.0001");
        assert_eq!(format_g(2.0, 3), "2");
        assert_eq!(format_g(0.0, 3), "0");
        assert_eq!(format_g(99.96, 3), "100");
        assert_eq!(format_g(999.6, 3), "1e+03");
    }

    #[test]
    fn two_halves_get_two_colours() {
        let lab = halves([30.0, 80.0], [40.0, -20.0]);
        let cfg = config(&[
            ("method", Value::Str("gerstner".into())),
            ("palette_size", Value::Int(2)),
            ("saturation_beta", Value::Float(1.0)),
        ]);
        let out = run(&lab, &[true; 1024], 32, 32, 8, 8, &cfg, None).unwrap();
        let idx = out.indices.unwrap();
        assert_eq!(out.palette_lab.unwrap().len(), 6);
        assert_ne!(idx[0], idx[7]);
        assert!(out.small_mask.iter().all(|&m| m));
        assert_eq!(out.stats[2], ("hit_iteration_cap", StatValue::Bool(false)));
    }

    #[test]
    fn stalled_pair_ends_annealing() {
        // tests/test_gerstner.py::test_stalled_pair_ends_annealing: with the spread cap
        // disabled the L 50 / L 54 pair never becomes ready; annealing must stall out
        // instead of running down to the 1e-3·T_final backstop.
        let lab = halves([50.0, 54.0], [0.0, 0.0]);
        let cfg = config(&[
            ("method", Value::Str("gerstner".into())),
            ("palette_size", Value::Int(2)),
        ]);
        let tunables = Tunables {
            split_spread: 1e9,
            ..Default::default()
        };
        let out = run_with(&lab, &[true; 1024], 32, 32, 8, 8, &cfg, None, tunables).unwrap();
        assert_eq!(out.stats[2], ("hit_iteration_cap", StatValue::Bool(false)));
        let StatValue::Float(t) = out.stats[3].1 else {
            panic!("final_temperature is a float")
        };
        assert!(t >= 1e-2 * cfg.g_T_final);
    }

    #[test]
    fn empty_mask_is_an_error() {
        let cfg = config(&[]);
        let err = run(&[0.0; 48], &[false; 16], 4, 4, 2, 2, &cfg, None).unwrap_err();
        assert_eq!(err.0, "no opaque pixels");
    }
}
