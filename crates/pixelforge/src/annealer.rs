//! The palette half of gerstner (mass-constrained deterministic annealing, Algorithm 1 of
//! Gerstner et al. 2013, docs/design/requirements.md Section 7): [`PaletteAnnealer`] and
//! [`principal_axis`], used by `gerstner::run` on superpixel colours and by `palette::mcda`
//! directly on pixel colours.
//!
//! Ported operation for operation from `pixelforge/downscale/gerstner.py`. The
//! reductions follow numpy's order (`npsum`): sums over the points (`axis=0`, `np.bincount`)
//! are sequential in point order, sums along a row (`axis=1`, `.sum()` of a 1-D array) are
//! numpy's pairwise sum. `exp` and `log` come from `libm`. Three routines numpy hands to
//! BLAS/LAPACK are replaced, so their results can differ from the Python version's in the
//! last bits (measured in tests/test_gerstner_run_native.py and CHANGELOG.md):
//! - `centered.T @ centered` (principal_axis) and `weighted.T @ points` (the REFINE step):
//!   sequential sums over the points in order (Accelerate/OpenBLAS block and vectorise them);
//! - `np.linalg.eigh` on the 3×3 covariance: a cyclic Jacobi eigensolver ([`eigh3`]), with
//!   the same sign normalisation (largest-magnitude component positive) and `v / ‖v‖`;
//! - `numpy.random.default_rng(seed).uniform(0.9, 1.1)`: a SplitMix64 stream seeded from
//!   `config.seed` (see [`SplitMix64`]); seeded outputs differ from the Python version's.

use crate::config::Config;
use crate::npsum;

/// Sub-cluster offset along the principal axis, LAB units.
pub const PERTURB_DELTA: f64 = 0.5;
/// A pair splits once it is this many times further apart than placed ...
pub const SPLIT_GROWTH: f64 = 3.0;
/// ... or once it is this many standard deviations of its cluster apart.
pub const SPLIT_SPREAD: f64 = 1.0;
/// Below g_T_final, stop once no pair grew over this many events.
pub const STALL_EVENTS: usize = 3;
/// Backstop: stop annealing below this fraction of g_T_final.
pub const EXHAUSTED_T_FRACTION: f64 = 1e-3;

/// The annealer's tunable constants. [`Tunables::default`] holds the constants above; the
/// Python binding passes `pixelforge.downscale.gerstner`'s module constants so tests that
/// patch them keep working.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tunables {
    pub perturb_delta: f64,
    pub split_growth: f64,
    pub split_spread: f64,
    pub exhausted_t_fraction: f64,
}

impl Default for Tunables {
    fn default() -> Self {
        Tunables {
            perturb_delta: PERTURB_DELTA,
            split_growth: SPLIT_GROWTH,
            split_spread: SPLIT_SPREAD,
            exhausted_t_fraction: EXHAUSTED_T_FRACTION,
        }
    }
}

/// The seeded perturbation's generator: SplitMix64 (Steele, Lea, Flood 2014) over the seed.
///
/// DEVIATION: the Python version drew from `numpy.random.default_rng(seed)` (PCG64 behind
/// numpy's SeedSequence). That stream is not reproduced: a seeded run stays deterministic and
/// still differs from the unseeded one, but its output differs from the Python version's for
/// the same seed.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [low, high): the top 53 bits as a double in [0, 1), scaled like numpy's
    /// `uniform` (`low + (high - low) * u`).
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        low + (high - low) * u
    }
}

/// Eigenvalues and eigenvectors of a symmetric 3×3 matrix by cyclic Jacobi rotations
/// (Numerical Recipes' `jacobi`): `(values, vectors)` with `vectors[i][j]` component i of the
/// eigenvector of `values[j]`, unsorted. Only +, −, ×, ÷ and `sqrt`, so the result is the
/// same on every platform. Replaces LAPACK's `dsyevd` (`np.linalg.eigh`), which may differ
/// in the last bits.
pub fn eigh3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut d = [a[0][0], a[1][1], a[2][2]];
    let mut b = d;
    let mut z = [0.0_f64; 3];
    fn rotate(
        a: &mut [[f64; 3]; 3],
        s: f64,
        tau: f64,
        (i, j): (usize, usize),
        (k, l): (usize, usize),
    ) {
        let g = a[i][j];
        let h = a[k][l];
        a[i][j] = g - s * (h + g * tau);
        a[k][l] = h + s * (g - h * tau);
    }
    for sweep in 1..=50 {
        let sm = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if sm == 0.0 {
            break;
        }
        let tresh = if sweep < 4 { 0.2 * sm / 9.0 } else { 0.0 };
        for p in 0..2 {
            for q in p + 1..3 {
                let g = 100.0 * a[p][q].abs();
                if sweep > 4 && d[p].abs() + g == d[p].abs() && d[q].abs() + g == d[q].abs() {
                    a[p][q] = 0.0;
                } else if a[p][q].abs() > tresh {
                    let mut h = d[q] - d[p];
                    let t = if h.abs() + g == h.abs() {
                        a[p][q] / h
                    } else {
                        let theta = 0.5 * h / a[p][q];
                        let t = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                        if theta < 0.0 {
                            -t
                        } else {
                            t
                        }
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    let tau = s / (1.0 + c);
                    h = t * a[p][q];
                    z[p] -= h;
                    z[q] += h;
                    d[p] -= h;
                    d[q] += h;
                    a[p][q] = 0.0;
                    for j in 0..p {
                        rotate(&mut a, s, tau, (j, p), (j, q));
                    }
                    for j in p + 1..q {
                        rotate(&mut a, s, tau, (p, j), (j, q));
                    }
                    for j in q + 1..3 {
                        rotate(&mut a, s, tau, (p, j), (q, j));
                    }
                    for j in 0..3 {
                        rotate(&mut v, s, tau, (j, p), (j, q));
                    }
                }
            }
        }
        for p in 0..3 {
            b[p] += z[p];
            d[p] = b[p];
            z[p] = 0.0;
        }
    }
    (d, v)
}

/// `points.mean(axis=0)`: per-channel sequential sums divided by n.
fn mean3(points: &[[f64; 3]]) -> [f64; 3] {
    let mut s = [0.0_f64; 3];
    for p in points {
        s[0] += p[0];
        s[1] += p[1];
        s[2] += p[2];
    }
    let n = points.len() as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

/// The 3×3 matrix `centered.T @ centered / n` with sequential sums over the points (numpy
/// calls BLAS here, whose summation order is not reproducible).
pub fn covariance(points: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mean = mean3(points);
    let mut s = [[0.0_f64; 3]; 3];
    for p in points {
        let c = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for i in 0..3 {
            for j in i..3 {
                s[i][j] += c[i] * c[j];
            }
        }
    }
    let n = points.len() as f64;
    let mut cov = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            cov[i][j] = s[i][j] / n;
            cov[j][i] = cov[i][j];
        }
    }
    cov
}

/// `exp` as the annealer computes it (`libm::exp`); exposed for parity tests.
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `log` as the annealer computes it (`libm::log`, `log(0) = -inf`); exposed for parity tests.
pub fn log(x: f64) -> f64 {
    libm::log(x)
}

/// Unit first principal component of the points and the variance along it.
///
/// Falls back to (1, 0, 0) with zero variance for fewer than 2 points or no spread. The sign
/// is normalised (largest-magnitude component positive, the first one on ties) so it does
/// not depend on the eigensolver.
pub fn principal_axis(points: &[[f64; 3]]) -> ([f64; 3], f64) {
    let fallback = ([1.0, 0.0, 0.0], 0.0);
    if points.len() < 2 {
        return fallback;
    }
    let (values, vectors) = eigh3(covariance(points));
    // The largest eigenvalue (`evals[-1]` of eigh's ascending order; the last on ties).
    let mut top = 0;
    for j in 1..3 {
        if values[j] >= values[top] {
            top = j;
        }
    }
    let variance = values[top];
    // `not variance > 1e-12`: NaN falls back too.
    if variance.is_nan() || variance <= 1e-12 {
        return fallback;
    }
    let mut v = [vectors[0][top], vectors[1][top], vectors[2][top]];
    let mut big = 0;
    for i in 1..3 {
        if v[i].abs() > v[big].abs() {
            big = i;
        }
    }
    if v[big] < 0.0 {
        v = [-v[0], -v[1], -v[2]];
    }
    let mut sq = 0.0;
    for x in v {
        sq += x * x;
    }
    let norm = sq.sqrt();
    ([v[0] / norm, v[1] / norm, v[2] / norm], variance)
}

/// The points whose `labels` entry equals `k`, in order (`points[labels == k]`).
fn select(points: &[[f64; 3]], labels: &[usize], k: usize) -> Vec<[f64; 3]> {
    points
        .iter()
        .zip(labels)
        .filter(|(_, &l)| l == k)
        .map(|(p, _)| *p)
        .collect()
}

/// ASSOCIATE / REFINE / CONVERGENCE CHECK / EXPAND of Algorithm 1.
///
/// While sub-clusters exist, cluster 2k and 2k+1 are the pair of palette colour k and the
/// displayed colour is their average.
#[derive(Clone, Debug)]
pub struct PaletteAnnealer {
    k: usize,
    /// Current temperature.
    pub t: f64,
    pub clusters: Vec<[f64; 3]>,
    /// Separation each sub-cluster pair was last placed at.
    pub placed: Vec<f64>,
    /// Separation of each pair at the last STALL_EVENTS convergence events, oldest first.
    pub history: Vec<[f64; STALL_EVENTS]>,
    pub prob: Vec<f64>,
    pub has_sub: bool,
    /// Palette index per point from the latest ASSOCIATE (empty before the first step).
    pub assign: Vec<usize>,
    pub done: bool,
    rng: Option<SplitMix64>,
    g_alpha: f64,
    g_t_final: f64,
    g_eps_palette: f64,
    g_eps_cluster: f64,
    tunables: Tunables,
}

impl PaletteAnnealer {
    /// `init_points` must not be empty; `palette_size` >= 1.
    pub fn new(init_points: &[[f64; 3]], palette_size: usize, config: &Config) -> Self {
        Self::with_tunables(init_points, palette_size, config, Tunables::default())
    }

    pub fn with_tunables(
        init_points: &[[f64; 3]],
        palette_size: usize,
        config: &Config,
        tunables: Tunables,
    ) -> Self {
        assert!(
            !init_points.is_empty(),
            "PaletteAnnealer needs at least one point"
        );
        assert!(palette_size >= 1, "palette_size must be >= 1");
        let mean = mean3(init_points);
        let (_, variance) = principal_axis(init_points);
        // DEVIATION: Section 7 — T0 = 1.1·2·σ (σ = standard deviation along the first
        // principal axis), not 1.1·2·σ². Rose's critical temperature Tc = 2λ holds for the
        // squared-distance kernel exp(−‖·‖²/T); the association here uses exp(−‖·‖/T), whose
        // Tc scales with the spread itself (for two colors at ±σ it is exactly σ). With the
        // variance a ΔE-100 checker started at T ≈ 6000 and spent 14 of 32 iterations cooling
        // before the first split. The floor only matters for a single-color input (σ = 0).
        let t0 = 1.1 * 2.0 * variance.sqrt();
        let t = if config.g_T_final > t0 {
            config.g_T_final
        } else {
            t0
        };
        PaletteAnnealer {
            k: palette_size,
            t,
            clusters: vec![mean, mean],
            placed: vec![0.0],
            history: vec![[0.0; STALL_EVENTS]],
            prob: vec![0.5, 0.5],
            has_sub: true,
            assign: Vec::new(),
            done: false,
            // The only permitted RNG use in the codebase (Section 3.1).
            rng: config.seed.map(|s| SplitMix64::new(s as u64)),
            g_alpha: config.g_alpha,
            g_t_final: config.g_T_final,
            g_eps_palette: config.g_eps_palette,
            g_eps_cluster: config.g_eps_cluster,
            tunables,
        }
    }

    /// The displayed palette: pair averages while sub-clusters exist.
    pub fn colors(&self) -> Vec<[f64; 3]> {
        if self.has_sub {
            self.clusters
                .chunks_exact(2)
                .map(|p| {
                    [
                        0.5 * (p[0][0] + p[1][0]),
                        0.5 * (p[0][1] + p[1][1]),
                        0.5 * (p[0][2] + p[1][2]),
                    ]
                })
                .collect()
        } else {
            self.clusters.clone()
        }
    }

    /// Number of displayed colours (`len(colors)`).
    pub fn n_colors(&self) -> usize {
        if self.has_sub {
            self.clusters.len() / 2
        } else {
            self.clusters.len()
        }
    }

    /// One iteration on the points with prior P(p_s) = weights.
    pub fn step(&mut self, points: &[[f64; 3]], weights: &[f64]) {
        assert_eq!(points.len(), weights.len(), "one weight per point");
        let m = self.clusters.len();
        let t = self.t; // divided by, not multiplied by a reciprocal (bitwise like numpy)
        let log_prob: Vec<f64> = self.prob.iter().map(|&p| libm::log(p)).collect();
        let mut sub_assign = Vec::with_capacity(points.len());
        let mut prob = vec![0.0_f64; m];
        let mut wsum = vec![[0.0_f64; 3]; m];
        let mut logp = vec![0.0_f64; m];
        for (p, &w) in points.iter().zip(weights) {
            // d2 = Σ_c (p_c − cluster_c)², c = 0, 1, 2; logp = log(prob) − sqrt(d2) / T.
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0;
            for (j, c) in self.clusters.iter().enumerate() {
                let mut d2 = 0.0;
                for ch in 0..3 {
                    let diff = p[ch] - c[ch];
                    d2 += diff * diff;
                }
                let v = log_prob[j] - d2.sqrt() / t;
                logp[j] = v;
                // logp.max(axis=1) and argmax (first maximum); NaN cannot occur.
                if j == 0 || v > best {
                    best = v;
                    arg = j;
                }
            }
            sub_assign.push(arg);
            for v in logp.iter_mut() {
                *v = libm::exp(*v - best);
            }
            let total = npsum::pairwise_sum(&logp);
            for (j, post) in logp.iter().enumerate() {
                let weighted = (post / total) * w;
                prob[j] += weighted;
                // weighted.T @ points as sequential sums over the points.
                wsum[j][0] += weighted * p[0];
                wsum[j][1] += weighted * p[1];
                wsum[j][2] += weighted * p[2];
            }
        }
        let mut refined = self.clusters.clone();
        for j in 0..m {
            if prob[j] > 0.0 {
                refined[j] = [
                    wsum[j][0] / prob[j],
                    wsum[j][1] / prob[j],
                    wsum[j][2] / prob[j],
                ];
            }
        }
        let moves: Vec<f64> = refined
            .iter()
            .zip(&self.clusters)
            .map(|(r, c)| {
                let mut s = 0.0;
                for ch in 0..3 {
                    let d = r[ch] - c[ch];
                    s += d * d;
                }
                s.sqrt()
            })
            .collect();
        let change = npsum::pairwise_sum(&moves);

        self.clusters = refined;
        self.prob = prob;
        self.assign = if self.has_sub {
            sub_assign.iter().map(|&s| s / 2).collect()
        } else {
            sub_assign.clone()
        };
        if change < self.g_eps_palette {
            self.cool_and_expand(points, &sub_assign);
        }
    }

    fn cool_and_expand(&mut self, points: &[[f64; 3]], sub_assign: &[usize]) {
        let tun = self.tunables;
        self.t *= self.g_alpha;
        if self.has_sub {
            let n_before = self.clusters.len() / 2;
            let first: Vec<[f64; 3]> = self.clusters.iter().step_by(2).copied().collect();
            let second: Vec<[f64; 3]> = self.clusters.iter().skip(1).step_by(2).copied().collect();
            let p_first: Vec<f64> = self.prob.iter().step_by(2).copied().collect();
            let p_second: Vec<f64> = self.prob.iter().skip(1).step_by(2).copied().collect();
            let separation: Vec<f64> = first
                .iter()
                .zip(&second)
                .map(|(f, s)| {
                    let mut acc = 0.0;
                    for ch in 0..3 {
                        let d = f[ch] - s[ch];
                        acc += d * d;
                    }
                    acc.sqrt()
                })
                .collect();
            // DEVIATION: Section 7 — a pair splits only once it is SPLIT_GROWTH times further
            // apart than the 2δ it was placed at (and beyond g_eps_cluster), and a pair that
            // is moving apart but not there yet is left alone instead of being re-centered.
            // The perturbation alone (1.0) already exceeds g_eps_cluster (0.25), and a
            // convergence event usually comes one iteration after it, so the spec's test
            // splits pairs that are still collapsing back together or drifting by a few
            // percent. Those "colors" coincide and, once K is reached and sub-clusters are
            // dropped, never separate: a sprite with one dominant color ended with 16
            // palette entries of which 4 were distinct. Requiring real growth alone is not
            // enough either: re-centering every pair at every event would undo the slow
            // divergence of a small cluster (a thin line holding 1.5% of the pixels never
            // split off at all), so growth has to be allowed to accumulate.
            //
            // The growth threshold is capped by the pair's own spread: a pair whose points
            // span only a few ΔE (two regions at L 50 and L 54 end up as a ramp after the
            // bilateral filter, halves ≈ 2.8 apart) can never reach 3·2δ, so it splits once
            // it is SPLIT_SPREAD standard deviations (along its principal axis) apart.
            let pair_of: Vec<usize> = sub_assign.iter().map(|&s| s / 2).collect();
            let spread: Vec<f64> = (0..n_before)
                .map(|k| principal_axis(&select(points, &pair_of, k)).1.sqrt())
                .collect();
            let growing: Vec<bool> = (0..n_before)
                .map(|k| separation[k] > self.placed[k])
                .collect();
            let ready: Vec<bool> = (0..n_before)
                .map(|k| {
                    let a = tun.split_growth * self.placed[k];
                    let b = tun.split_spread * spread[k];
                    let cap = if b < a { b } else { a }; // np.minimum (no NaN here)
                    growing[k] && separation[k] > self.g_eps_cluster && separation[k] > cap
                })
                .collect();
            let grown: Vec<f64> = (0..n_before)
                .map(|k| separation[k] - self.history[k][0])
                .collect();
            let history: Vec<[f64; STALL_EVENTS]> = (0..n_before)
                .map(|k| {
                    let mut h = [0.0; STALL_EVENTS];
                    h[..STALL_EVENTS - 1].copy_from_slice(&self.history[k][1..]);
                    h[STALL_EVENTS - 1] = separation[k];
                    h
                })
                .collect();
            // DEVIATION: Section 7 — when more pairs are ready than palette slots remain, the
            // slots go to the pairs whose split removes the most error (mass-weighted squared
            // separation), not to the lowest k. In ascending-k order the last slots went to
            // pairs a few ΔE apart while a pair 40 ΔE apart was left merged.
            let gain: Vec<f64> = (0..n_before)
                .map(|k| {
                    let total = p_first[k] + p_second[k];
                    let denom = if 1e-300 > total { 1e-300 } else { total };
                    p_first[k] * p_second[k] / denom * (separation[k] * separation[k])
                })
                .collect();
            let mut candidates: Vec<usize> = (0..n_before).filter(|&k| ready[k]).collect();
            // sorted(key=(-gain[k], k)): stable, larger gain first, then lower k.
            candidates.sort_by(|&a, &b| {
                (-gain[a])
                    .partial_cmp(&-gain[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let slots = self.k.saturating_sub(n_before);
            let mut chosen = vec![false; n_before];
            let mut any_chosen = false;
            for &k in candidates.iter().take(slots) {
                chosen[k] = true;
                any_chosen = true;
            }

            let mut n_colors = n_before;
            let mut colors = Vec::with_capacity(self.k);
            let mut probs = Vec::with_capacity(self.k);
            let mut extra_colors = Vec::new();
            let mut extra_probs = Vec::new();
            let mut remap = vec![0usize; 2 * n_before];
            for k in 0..n_before {
                if chosen[k] {
                    colors.push(first[k]);
                    probs.push(p_first[k]);
                    extra_colors.push(second[k]);
                    extra_probs.push(p_second[k]);
                    remap[2 * k] = k;
                    remap[2 * k + 1] = n_colors;
                    n_colors += 1;
                } else {
                    let (f, s) = (first[k], second[k]);
                    colors.push([
                        0.5 * (f[0] + s[0]),
                        0.5 * (f[1] + s[1]),
                        0.5 * (f[2] + s[2]),
                    ]);
                    probs.push(p_first[k] + p_second[k]);
                    remap[2 * k] = k;
                    remap[2 * k + 1] = k;
                }
            }
            colors.extend(extra_colors);
            probs.extend(extra_probs);
            self.assign = sub_assign.iter().map(|&s| remap[s]).collect();

            let keep: Vec<bool> = (0..n_colors)
                .map(|k| k < n_before && !chosen[k] && growing[k])
                .collect();

            // DEVIATION: Section 7 — the spec only breaks once K_current == K. An image with
            // fewer than K separable colors would then cool forever (T → 0) until
            // g_max_iters, so annealing also ends when the final temperature has been
            // reached and a convergence event splits no pair and leaves no pair whose
            // separation grew by more than g_eps_cluster over the last STALL_EVENTS events
            // (or, as a backstop, when T has fallen far below the final temperature); the
            // palette keeps K_current < K colors.
            let stalled = !(0..n_before).any(|k| keep[k] && grown[k] > self.g_eps_cluster);
            let exhausted = (self.t <= self.g_t_final && !any_chosen && stalled)
                || self.t < tun.exhausted_t_fraction * self.g_t_final;
            if n_colors == self.k || exhausted {
                self.clusters = colors;
                self.prob = probs;
                self.has_sub = false;
                if exhausted {
                    self.done = true;
                    return;
                }
            } else {
                let old_clusters = std::mem::take(&mut self.clusters);
                let old_prob = std::mem::take(&mut self.prob);
                let old_placed = std::mem::take(&mut self.placed);
                let mut clusters = vec![[0.0_f64; 3]; 2 * n_colors];
                let mut placed = vec![0.0_f64; n_colors];
                let mut new_history = vec![[0.0_f64; STALL_EVENTS]; n_colors];
                let mut prob = Vec::with_capacity(2 * n_colors);
                for &p in &probs {
                    let half = 0.5 * p;
                    prob.push(half);
                    prob.push(half);
                }
                for k in 0..n_colors {
                    if keep[k] {
                        // still moving apart: let the divergence accumulate
                        clusters[2 * k] = old_clusters[2 * k];
                        clusters[2 * k + 1] = old_clusters[2 * k + 1];
                        prob[2 * k] = old_prob[2 * k];
                        prob[2 * k + 1] = old_prob[2 * k + 1];
                        placed[k] = old_placed[k];
                        new_history[k] = history[k];
                        continue;
                    }
                    let (axis, _) = principal_axis(&select(points, &self.assign, k));
                    let mut delta = tun.perturb_delta;
                    if let Some(rng) = self.rng.as_mut() {
                        // The only permitted RNG use in the codebase (Section 3.1).
                        delta *= rng.uniform(0.9, 1.1);
                    }
                    let c = colors[k];
                    clusters[2 * k] = [
                        c[0] + delta * axis[0],
                        c[1] + delta * axis[1],
                        c[2] + delta * axis[2],
                    ];
                    clusters[2 * k + 1] = [
                        c[0] - delta * axis[0],
                        c[1] - delta * axis[1],
                        c[2] - delta * axis[2],
                    ];
                    placed[k] = 2.0 * delta;
                    new_history[k] = [2.0 * delta; STALL_EVENTS];
                }
                self.clusters = clusters;
                self.placed = placed;
                self.history = new_history;
                self.prob = prob;
            }
        }
        if self.t <= self.g_t_final && !self.has_sub {
            self.done = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(kw: &[(&str, crate::config::Value)]) -> Config {
        let kw: Vec<(String, crate::config::Value)> =
            kw.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        Config::new(&kw).unwrap()
    }

    #[test]
    fn eigh3_diagonalises() {
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]];
        let (d, v) = eigh3(m);
        for j in 0..3 {
            for i in 0..3 {
                let mv: f64 = (0..3).map(|k| m[i][k] * v[k][j]).sum();
                assert!((mv - d[j] * v[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn principal_axis_of_a_line() {
        let pts: Vec<[f64; 3]> = (0..10).map(|i| [0.0, -(i as f64), 0.0]).collect();
        let (axis, var) = principal_axis(&pts);
        assert_eq!(axis, [0.0, 1.0, 0.0]);
        assert!((var - 8.25).abs() < 1e-12);
        assert_eq!(principal_axis(&pts[..1]), ([1.0, 0.0, 0.0], 0.0));
        assert_eq!(
            principal_axis(&[[1.0, 2.0, 3.0]; 5]),
            ([1.0, 0.0, 0.0], 0.0)
        );
    }

    #[test]
    fn splitmix_is_deterministic_and_in_range() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..1000 {
            let x = a.uniform(0.9, 1.1);
            assert_eq!(x, b.uniform(0.9, 1.1));
            assert!((0.9..1.1).contains(&x));
        }
        // Reference value of SplitMix64(0)'s first output.
        assert_eq!(SplitMix64::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn two_clusters_are_found() {
        let cfg = config(&[]);
        let mut pts = vec![[20.0, 0.0, 0.0]; 50];
        pts.extend(vec![[80.0, 10.0, -10.0]; 50]);
        let w = vec![1.0 / 100.0; 100];
        let mut a = PaletteAnnealer::new(&pts, 2, &cfg);
        for _ in 0..cfg.g_max_iters {
            a.step(&pts, &w);
            if a.done {
                break;
            }
        }
        let colors = a.colors();
        assert_eq!(colors.len(), 2);
        assert_ne!(a.assign[0], a.assign[99]);
        for c in colors {
            assert!((c[0] - 20.0).abs() < 1e-6 || (c[0] - 80.0).abs() < 1e-6);
        }
    }
}
