"""Joint superpixel + palette optimization (Gerstner et al. 2013), REQUIREMENTS.md Section 7.

The input is segmented into exactly wo·ho superpixels (one per output pixel) while a K-color
palette is grown by mass-constrained deterministic annealing (MCDA). `PaletteAnnealer` is the
palette half on its own; `palette.mcda` reuses it directly on pixel colors.

The two per-pixel hot loops, the SLIC assignment pass and the bilateral filter, run in the
native extension (`pixelforge._native`, rust/src/gerstner.rs); the numpy versions they replace
are kept in `_reference/gerstner.py` for tests and benchmarks.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .. import _native
from ..errors import PixelforgeError
from . import Downscaled, neighbor_mean4
from .box import cell_index

PERTURB_DELTA = 0.5      # sub-cluster offset along the principal axis, LAB units
SPLIT_GROWTH = 3.0       # a pair splits once it is this many times further apart than placed
SPLIT_SPREAD = 1.0       # ... or once it is this many standard deviations of its cluster apart
STALL_EVENTS = 3         # below g_T_final, stop once no pair grew over this many events
EXHAUSTED_T_FRACTION = 1e-3   # backstop: stop annealing below this fraction of g_T_final
_MAX_SEARCH_RADIUS = 6   # grid cells searched around a pixel's home cell (safety bound)


log = logging.getLogger(__name__)
PROGRESS_EVERY = 10      # iterations between INFO progress lines (DEBUG logs every one)


def principal_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit first principal component of (n, 3) points and the variance along it.

    Falls back to (1, 0, 0) with zero variance for fewer than 2 points or no spread. The sign
    is normalized (largest-magnitude component positive) so it does not depend on LAPACK.
    """
    fallback = np.array([1.0, 0.0, 0.0])
    if len(points) < 2:
        return fallback, 0.0
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / len(points)
    evals, evecs = np.linalg.eigh(cov)
    variance = float(evals[-1])
    if not variance > 1e-12:
        return fallback, 0.0
    v = evecs[:, -1]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v / np.linalg.norm(v), variance


class PaletteAnnealer:
    """ASSOCIATE / REFINE / CONVERGENCE CHECK / EXPAND of Algorithm 1.

    While sub-clusters exist, cluster 2k and 2k+1 are the pair of palette color k and the
    displayed color is their average.
    """

    def __init__(self, init_points: np.ndarray, palette_size: int, config):
        self.K = int(palette_size)
        self.config = config
        mean = init_points.mean(axis=0)
        _, variance = principal_axis(init_points)
        # DEVIATION: Section 7 — T0 = 1.1·2·σ (σ = standard deviation along the first
        # principal axis), not 1.1·2·σ². Rose's critical temperature Tc = 2λ holds for the
        # squared-distance kernel exp(−‖·‖²/T); the association here uses exp(−‖·‖/T), whose
        # Tc scales with the spread itself (for two colors at ±σ it is exactly σ). With the
        # variance a ΔE-100 checker started at T ≈ 6000 and spent 14 of 32 iterations cooling
        # before the first split. The floor only matters for a single-color input (σ = 0).
        self.T = max(1.1 * 2.0 * math.sqrt(variance), config.g_T_final)
        self.clusters = np.stack([mean, mean])
        self.placed = np.zeros(1)    # separation each sub-cluster pair was last placed at
        # Separation of each pair at the last STALL_EVENTS convergence events, oldest first.
        self.history = np.zeros((1, STALL_EVENTS))
        self.prob = np.array([0.5, 0.5])
        self.has_sub = True
        self.assign = None       # palette index per point from the latest ASSOCIATE
        self.done = False
        self.rng = np.random.default_rng(config.seed) if config.seed is not None else None

    @property
    def colors(self) -> np.ndarray:
        if self.has_sub:
            return 0.5 * (self.clusters[0::2] + self.clusters[1::2])
        return self.clusters

    def step(self, points: np.ndarray, weights: np.ndarray) -> None:
        """One iteration on (n, 3) points with prior P(p_s) = weights."""
        clusters = self.clusters
        d2 = np.zeros((len(points), len(clusters)))
        for c in range(3):
            diff = points[:, c, None] - clusters[None, :, c]
            d2 += diff * diff
        with np.errstate(divide="ignore"):
            logp = np.log(self.prob)[None, :] - np.sqrt(d2) / self.T
        logp -= logp.max(axis=1, keepdims=True)
        sub_assign = np.argmax(logp, axis=1)   # ties → lowest index
        post = np.exp(logp)
        post /= post.sum(axis=1, keepdims=True)

        weighted = post * weights[:, None]
        prob = weighted.sum(axis=0)
        refined = clusters.copy()
        alive = prob > 0
        refined[alive] = (weighted.T @ points)[alive] / prob[alive, None]
        change = float(np.sqrt(((refined - clusters) ** 2).sum(axis=1)).sum())

        self.clusters, self.prob = refined, prob
        self.assign = sub_assign // 2 if self.has_sub else sub_assign
        if change < self.config.g_eps_palette:
            self._cool_and_expand(points, sub_assign)

    def _cool_and_expand(self, points: np.ndarray, sub_assign: np.ndarray) -> None:
        cfg = self.config
        self.T *= cfg.g_alpha
        if self.has_sub:
            n_before = len(self.clusters) // 2
            first, second = self.clusters[0::2], self.clusters[1::2]
            p_first, p_second = self.prob[0::2], self.prob[1::2]
            separation = np.sqrt(((first - second) ** 2).sum(axis=1))
            # DEVIATION: Section 7 — a pair splits only once it is SPLIT_GROWTH times further
            # apart than the 2δ it was placed at (and beyond g_eps_cluster), and a pair that
            # is moving apart but not there yet is left alone instead of being re-centered.
            # The perturbation alone (1.0) already exceeds g_eps_cluster (0.25), and a
            # convergence event usually comes one iteration after it, so the spec's test
            # splits pairs that are still collapsing back together or drifting by a few
            # percent. Those "colors" coincide and, once K is reached and sub-clusters are
            # dropped, never separate: a sprite with one dominant color ended with 16
            # palette entries of which 4 were distinct. Requiring real growth alone is not
            # enough either: re-centering every pair at every event would undo the slow
            # divergence of a small cluster (a thin line holding 1.5% of the pixels never
            # split off at all), so growth has to be allowed to accumulate.
            #
            # The growth threshold is capped by the pair's own spread: a pair whose points
            # span only a few ΔE (two regions at L 50 and L 54 end up as a ramp after the
            # bilateral filter, halves ≈ 2.8 apart) can never reach 3·2δ, so it splits once
            # it is SPLIT_SPREAD standard deviations (along its principal axis) apart.
            spread = np.array([math.sqrt(principal_axis(points[sub_assign // 2 == k])[1])
                               for k in range(n_before)])
            growing = separation > self.placed
            ready = growing & (separation > cfg.g_eps_cluster) & (
                separation > np.minimum(SPLIT_GROWTH * self.placed, SPLIT_SPREAD * spread))
            grown = separation - self.history[:, 0]
            history = np.concatenate([self.history[:, 1:], separation[:, None]], axis=1)
            # DEVIATION: Section 7 — when more pairs are ready than palette slots remain, the
            # slots go to the pairs whose split removes the most error (mass-weighted squared
            # separation), not to the lowest k. In ascending-k order the last slots went to
            # pairs a few ΔE apart while a pair 40 ΔE apart was left merged.
            gain = p_first * p_second / np.maximum(p_first + p_second, 1e-300) * separation ** 2
            candidates = sorted(np.nonzero(ready)[0].tolist(), key=lambda k: (-gain[k], k))
            chosen = set(candidates[:self.K - n_before])

            n_colors = n_before
            colors, probs, extra_colors, extra_probs = [], [], [], []
            remap = np.empty(2 * n_before, dtype=np.int64)
            for k in range(n_before):
                if k in chosen:
                    colors.append(first[k])
                    probs.append(p_first[k])
                    extra_colors.append(second[k])
                    extra_probs.append(p_second[k])
                    remap[2 * k], remap[2 * k + 1] = k, n_colors
                    n_colors += 1
                else:
                    colors.append(0.5 * (first[k] + second[k]))
                    probs.append(p_first[k] + p_second[k])
                    remap[2 * k] = remap[2 * k + 1] = k
            colors = np.array(colors + extra_colors)
            probs = np.array(probs + extra_probs)
            self.assign = remap[sub_assign]

            keep = [k < n_before and k not in chosen and bool(growing[k])
                    for k in range(n_colors)]

            # DEVIATION: Section 7 — the spec only breaks once K_current == K. An image with
            # fewer than K separable colors would then cool forever (T → 0) until
            # g_max_iters, so annealing also ends when the final temperature has been
            # reached and a convergence event splits no pair and leaves no pair whose
            # separation grew by more than g_eps_cluster over the last STALL_EVENTS events
            # (or, as a backstop, when T has fallen far below the final temperature); the
            # palette keeps K_current < K colors.
            stalled = not any(keep[k] and grown[k] > cfg.g_eps_cluster
                              for k in range(n_before))
            exhausted = ((self.T <= cfg.g_T_final and not chosen and stalled)
                         or self.T < EXHAUSTED_T_FRACTION * cfg.g_T_final)
            if n_colors == self.K or exhausted:
                self.clusters, self.prob, self.has_sub = colors, probs, False
                if exhausted:
                    self.done = True
                    return
            else:
                old_clusters, old_prob, old_placed = self.clusters, self.prob, self.placed
                self.clusters = np.empty((2 * n_colors, 3))
                self.placed = np.empty(n_colors)
                self.history = np.empty((n_colors, STALL_EVENTS))
                self.prob = np.repeat(0.5 * probs, 2)
                for k in range(n_colors):
                    if keep[k]:
                        # still moving apart: let the divergence accumulate
                        self.clusters[2 * k:2 * k + 2] = old_clusters[2 * k:2 * k + 2]
                        self.prob[2 * k:2 * k + 2] = old_prob[2 * k:2 * k + 2]
                        self.placed[k] = old_placed[k]
                        self.history[k] = history[k]
                        continue
                    axis, _ = principal_axis(points[self.assign == k])
                    delta = PERTURB_DELTA
                    if self.rng is not None:
                        # The only permitted RNG use in the codebase (Section 3.1).
                        delta *= self.rng.uniform(0.9, 1.1)
                    self.clusters[2 * k] = colors[k] + delta * axis
                    self.clusters[2 * k + 1] = colors[k] - delta * axis
                    self.placed[k] = 2.0 * delta
                    self.history[k] = 2.0 * delta
        if self.T <= cfg.g_T_final and not self.has_sub:
            self.done = True


def _laplacian_smooth(grid: np.ndarray, fraction: float) -> np.ndarray:
    """Move every value `fraction` of the way toward the mean of its 4-connected neighbors."""
    return grid + fraction * (neighbor_mean4(grid) - grid)


def bilateral_filter(image: np.ndarray, active: np.ndarray, sigma_color: float,
                     sigma_spatial: float) -> np.ndarray:
    """Bilateral filter of an (H, W, 3) LAB image; pixels with active == False are excluded.

    radius = ceil(2·σ_s), zero padding, weight = exp(−ΔE²/(2σ_c²) − |d|²/(2σ_s²))·active;
    pixels whose weights sum to 0 keep their value. Native (`_native.bilateral_filter`).
    """
    return _native.bilateral_filter(np.ascontiguousarray(image, dtype=np.float64),
                                    np.ascontiguousarray(active), float(sigma_color),
                                    float(sigma_spatial))


class _Grid:
    """Geometry shared by the SLIC assignment passes."""

    def __init__(self, hi: int, wi: int, ho: int, wo: int):
        self.hi, self.wi, self.ho, self.wo = hi, wi, ho, wo
        self.rx, self.ry = wi / wo, hi / ho
        self.cx0 = np.tile((np.arange(wo) + 0.5) * self.rx, ho)
        self.cy0 = np.repeat((np.arange(ho) + 0.5) * self.ry, wo)
        self.px = np.arange(wi) + 0.5            # pixel centers
        self.py = np.arange(hi) + 0.5
        # Initial-grid cell containing each pixel column / row.
        self.home_x = cell_index(wi, wo)
        self.home_y = cell_index(hi, ho)

    def search_radius(self, cx: np.ndarray, cy: np.ndarray) -> int:
        """Grid cells around the home cell that can hold a center inside the 2rx × 2ry window.

        A center that drifted D cells from its initial grid position can only be within the
        window of pixels whose home cell is < 1.5 + D cells away.
        """
        drift = max(float(np.abs(cx - self.cx0).max()) / self.rx,
                    float(np.abs(cy - self.cy0).max()) / self.ry)
        return min(int(math.floor(1.5 + drift + 1e-9)), _MAX_SEARCH_RADIUS)


def _assign(grid: _Grid, cx, cy, radius: int, candidate_ok=None, channels=None, sp_color=None,
            coef: float = 1.0, windowed: bool = True) -> np.ndarray:
    """One modified-SLIC assignment pass over the whole (hi, wi) pixel grid.

    d(p_i, p_s) = ||c_i − c_k(s)|| + coef·||(x_i, y_i) − (x_s, y_s)||, over superpixels whose
    center lies within the 2rx × 2ry window of the pixel. Candidates are the superpixels of the
    grid cells around the pixel's home cell, visited in ascending s with a strict '<', so ties
    resolve to the lowest s. (Offsets falling off the grid are clipped to the border cell,
    which merely revisits a candidate.) With channels=None only the spatial term is used.
    `channels` is the (3, hi, wi) LAB image. Returns the superpixel index per pixel, -1 where
    no candidate qualified. Native (`_native.slic_assign`), bitwise equal to the numpy
    reference in `_reference/gerstner.py`.
    """
    ho, wo = grid.ho, grid.wo
    center_x = cx.reshape(ho, wo)
    if candidate_ok is not None:
        # An infinitely distant center fails the window test and never wins.
        center_x = np.where(candidate_ok.reshape(ho, wo), center_x, np.inf)
    if channels is not None:
        channels = np.ascontiguousarray(channels, dtype=np.float64)
        sp_color = np.ascontiguousarray(sp_color, dtype=np.float64).reshape(ho, wo, 3)
    return _native.slic_assign(grid.home_y, grid.home_x, grid.px, grid.py,
                               np.ascontiguousarray(center_x, dtype=np.float64),
                               np.ascontiguousarray(cy, dtype=np.float64).reshape(ho, wo),
                               int(radius), float(grid.rx), float(grid.ry), float(coef),
                               bool(windowed), channels, sp_color)


def _weighted(values: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    return values if w is None else w * values


def palette_prior(coverage: np.ndarray) -> np.ndarray:
    """P(p_s) over the active superpixels from their mean pixel weight (alpha).

    Uniform (exactly 1/N) when every superpixel is fully opaque, as in the spec; otherwise
    proportional to the coverage, so a superpixel of semi-transparent fringe pixels holds
    less palette mass. Superpixels lying entirely in the transparent region are not active.
    """
    # DEVIATION: Section 7 — the spec's uniform prior is weighted by alpha coverage. It is
    # the mean weight, not the total, so that equal-alpha superpixels of different pixel
    # counts keep the spec's equal mass (and an opaque input is bit-identical).
    if np.all(coverage == 1.0):
        return np.full(len(coverage), 1.0 / len(coverage))
    return coverage / coverage.sum()


def run(lab: np.ndarray, mask: np.ndarray, out_width: int, out_height: int, config,
        weight: np.ndarray | None = None) -> Downscaled:
    if not mask.any():
        raise PixelforgeError("no opaque pixels")
    hi, wi = mask.shape
    wo, ho = int(out_width), int(out_height)
    n_sp = wo * ho
    grid = _Grid(hi, wi, ho, wo)
    coef = config.g_m * math.sqrt(n_sp / (hi * wi))

    ys, xs = np.nonzero(mask)
    px, py = xs + 0.5, ys + 0.5
    col = np.ascontiguousarray(lab[ys, xs], dtype=np.float64)
    channels = np.ascontiguousarray(np.moveaxis(lab, -1, 0), dtype=np.float64)   # (3, hi, wi)
    all_opaque = bool(mask.all())
    # Pixel weight (alpha) of the opaque pixels: superpixel means and centers are weighted by
    # it, and so is each superpixel's share of the palette prior. None when it is all ones.
    pix_w = None
    if weight is not None:
        pix_w = np.asarray(weight, dtype=np.float64)[ys, xs]
        if np.all(pix_w == 1.0):
            pix_w = None

    # INITIALIZE: regular grid of centers, every pixel assigned to the nearest center.
    cx, cy = grid.cx0.copy(), grid.cy0.copy()
    assign = grid.home_y[ys] * wo + grid.home_x[xs]
    count = np.bincount(assign, minlength=n_sp)
    has_color = count > 0
    mass = count if pix_w is None else np.bincount(assign, weights=pix_w, minlength=n_sp)
    # Mean pixel weight of each superpixel (1 for an opaque input), kept for those that
    # lose all their pixels in an iteration, like their color.
    coverage = np.ones(n_sp)
    if pix_w is not None:
        coverage[has_color] = mass[has_color] / count[has_color]
    mean = np.zeros((n_sp, 3))
    for c in range(3):
        sums = np.bincount(assign, weights=_weighted(col[:, c], pix_w), minlength=n_sp)
        mean[has_color, c] = sums[has_color] / mass[has_color]

    annealer = PaletteAnnealer(col, config.palette_size, config)
    k = np.zeros(n_sp, dtype=np.int64)
    sp_color = annealer.colors[k]
    smooth = mean.reshape(ho, wo, 3)
    iterations = 0
    hit_cap = True

    for _ in range(config.g_max_iters):
        # 1. SUPERPIXEL REFINEMENT (one modified-SLIC step).
        radius = grid.search_radius(cx, cy)
        assigned_to = _assign(grid, cx, cy, radius, candidate_ok=has_color, channels=channels,
                              sp_color=sp_color, coef=coef)
        # Masked (transparent) input pixels are never assigned and never contribute.
        assign = assigned_to.ravel() if all_opaque else assigned_to[mask]
        assigned = assign >= 0
        a_w = pix_w
        if assigned.all():
            a, a_px, a_py, a_col = assign, px, py, col
        else:
            a, a_px, a_py, a_col = assign[assigned], px[assigned], py[assigned], col[assigned]
            if pix_w is not None:
                a_w = pix_w[assigned]
        count = np.bincount(a, minlength=n_sp)
        got = count > 0   # superpixels with zero pixels keep their previous center and color
        mass = count if a_w is None else np.bincount(a, weights=a_w, minlength=n_sp)
        for c in range(3):
            sums = np.bincount(a, weights=_weighted(a_col[:, c], a_w), minlength=n_sp)
            mean[got, c] = sums[got] / mass[got]
        cx[got] = np.bincount(a, weights=_weighted(a_px, a_w), minlength=n_sp)[got] / mass[got]
        cy[got] = np.bincount(a, weights=_weighted(a_py, a_w), minlength=n_sp)[got] / mass[got]
        if a_w is not None:
            coverage[got] = mass[got] / count[got]
        has_color |= got

        cx = _laplacian_smooth(cx.reshape(ho, wo), config.g_laplacian).ravel()
        cy = _laplacian_smooth(cy.reshape(ho, wo), config.g_laplacian).ravel()

        active_grid = has_color.reshape(ho, wo)
        smooth = bilateral_filter(mean.reshape(ho, wo, 3), active_grid,
                                  config.g_bilateral_sigma_color,
                                  config.g_bilateral_sigma_spatial)

        # 2-4. ASSOCIATE, REFINE, CONVERGENCE CHECK / EXPAND on the active superpixels.
        active = np.nonzero(has_color)[0]
        points = smooth.reshape(-1, 3)[active]
        annealer.step(points, palette_prior(coverage[active]))
        k[active] = annealer.assign
        sp_color = annealer.colors[k]
        iterations += 1
        level = logging.INFO if iterations % PROGRESS_EVERY == 0 else logging.DEBUG
        if log.isEnabledFor(level):   # read-only: logging never affects the result
            log.log(level, "gerstner iter %d/%d T=%.3g colors=%d", iterations,
                    config.g_max_iters, annealer.T, len(annealer.colors))
        if annealer.done:
            hit_cap = False
            break
    log.debug("gerstner %s after %d iterations",
              "stopped at the cap" if hit_cap else "converged", iterations)

    # POST: saturation, then indices / palette / mask.
    unsaturated = annealer.colors.copy()
    palette = unsaturated.copy()
    palette[:, 1:] *= config.saturation_beta

    # DEVIATION: Section 7 — the spec does not say how transparent pixels count toward a
    # superpixel (and its prior is uniform over the superpixels holding opaque pixels).
    # small_mask[s] = opaque fraction of superpixel s >= 0.5. Transparent pixels are never
    # assigned during the optimization, so attribute each to its spatially nearest center.
    transparent = np.zeros(n_sp, dtype=np.int64)
    if not mask.all():
        nearest = _assign(grid, cx, cy, grid.search_radius(cx, cy), windowed=False)[~mask]
        transparent = np.bincount(nearest[nearest >= 0], minlength=n_sp)
    small_mask = has_color & (count * 2 >= count + transparent)

    return Downscaled(
        small_lab=palette[k].reshape(ho, wo, 3),
        small_mask=small_mask.reshape(ho, wo),
        palette_lab=palette,
        palette_lab_unsaturated=unsaturated,
        indices=k.reshape(ho, wo).copy(),
        mean_lab=smooth,
        stats={"iterations": iterations, "palette_size": int(len(palette)),
               "hit_iteration_cap": bool(hit_cap), "final_temperature": float(annealer.T)},
    )
