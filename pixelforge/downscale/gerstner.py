"""Joint superpixel + palette optimization (Gerstner et al. 2013), REQUIREMENTS.md Section 7.

The input is segmented into exactly wo·ho superpixels (one per output pixel) while a K-color
palette is grown by mass-constrained deterministic annealing (MCDA). `PaletteAnnealer` is the
palette half on its own; `palette.mcda` reuses it directly on pixel colors.
"""

from __future__ import annotations

import math

import numpy as np

from . import Downscaled

PERTURB_DELTA = 0.5      # sub-cluster offset along the principal axis, LAB units
SPLIT_GROWTH = 3.0       # a pair splits once it is this many times further apart than placed
EXHAUSTED_T_FRACTION = 1e-3   # backstop: stop annealing below this fraction of g_T_final
_MAX_SEARCH_RADIUS = 6   # grid cells searched around a pixel's home cell (safety bound)


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
        # T = 1.1·Tc with Tc = 2·(variance along the first principal axis). The floor only
        # matters for a single-color input, where Tc is 0.
        self.T = max(1.1 * 2.0 * variance, config.g_T_final)
        self.clusters = np.stack([mean, mean])
        self.placed = np.zeros(1)    # separation each sub-cluster pair was last placed at
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
            growing = separation > self.placed
            ready = growing & (separation > cfg.g_eps_cluster) & (
                separation > SPLIT_GROWTH * self.placed)
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
            # reached and a convergence event leaves no pair split or still moving apart
            # (or, as a backstop, when T has fallen far below the final temperature); the
            # palette keeps K_current < K colors.
            exhausted = ((self.T <= cfg.g_T_final and not chosen and not any(keep))
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
                self.prob = np.repeat(0.5 * probs, 2)
                for k in range(n_colors):
                    if keep[k]:
                        # still moving apart: let the divergence accumulate
                        self.clusters[2 * k:2 * k + 2] = old_clusters[2 * k:2 * k + 2]
                        self.prob[2 * k:2 * k + 2] = old_prob[2 * k:2 * k + 2]
                        self.placed[k] = old_placed[k]
                        continue
                    axis, _ = principal_axis(points[self.assign == k])
                    delta = PERTURB_DELTA
                    if self.rng is not None:
                        # The only permitted RNG use in the codebase (Section 3.1).
                        delta *= self.rng.uniform(0.9, 1.1)
                    self.clusters[2 * k] = colors[k] + delta * axis
                    self.clusters[2 * k + 1] = colors[k] - delta * axis
                    self.placed[k] = 2.0 * delta
        if self.T <= cfg.g_T_final and not self.has_sub:
            self.done = True


def _laplacian_smooth(grid: np.ndarray, fraction: float) -> np.ndarray:
    """Move every value `fraction` of the way toward the mean of its 4-connected neighbors."""
    total = np.zeros_like(grid)
    count = np.zeros_like(grid)
    total[1:] += grid[:-1]
    count[1:] += 1
    total[:-1] += grid[1:]
    count[:-1] += 1
    total[:, 1:] += grid[:, :-1]
    count[:, 1:] += 1
    total[:, :-1] += grid[:, 1:]
    count[:, :-1] += 1
    target = np.where(count > 0, total / np.maximum(count, 1), grid)
    return grid + fraction * (target - grid)


def bilateral_filter(image: np.ndarray, active: np.ndarray, sigma_color: float,
                     sigma_spatial: float) -> np.ndarray:
    """Bilateral filter of an (H, W, 3) LAB image; pixels with active == False are excluded."""
    h, w = active.shape
    radius = int(math.ceil(2.0 * sigma_spatial))
    padded = np.zeros((h + 2 * radius, w + 2 * radius, 3))
    padded[radius:radius + h, radius:radius + w] = image
    padded_active = np.zeros((h + 2 * radius, w + 2 * radius))
    padded_active[radius:radius + h, radius:radius + w] = active
    num = np.zeros((h, w, 3))
    den = np.zeros((h, w))
    inv_color = 1.0 / (2.0 * sigma_color * sigma_color)
    inv_spatial = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys = slice(radius + dy, radius + dy + h)
            xs = slice(radius + dx, radius + dx + w)
            neighbor = padded[ys, xs]
            diff = neighbor - image
            d2 = (diff * diff).sum(axis=-1)
            weight = np.exp(-d2 * inv_color - (dx * dx + dy * dy) * inv_spatial)
            weight *= padded_active[ys, xs]
            num += weight[..., None] * neighbor
            den += weight
    out = image.copy()
    ok = den > 0
    out[ok] = num[ok] / den[ok][:, None]
    return out


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
        self.home_x = (np.arange(wi, dtype=np.int64) * wo) // wi
        self.home_y = (np.arange(hi, dtype=np.int64) * ho) // hi

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
    Returns the superpixel index per pixel, -1 where no candidate qualified.
    """
    hi, wi, ho, wo = grid.hi, grid.wi, grid.ho, grid.wo
    rx, ry = grid.rx, grid.ry
    center_x = cx.reshape(ho, wo)
    center_y = cy.reshape(ho, wo)
    if candidate_ok is not None:
        # An infinitely distant center fails the window test and never wins.
        center_x = np.where(candidate_ok.reshape(ho, wo), center_x, np.inf)
    planes = [center_x, center_y]
    if channels is not None:
        planes += [np.ascontiguousarray(sp_color[:, c]).reshape(ho, wo) for c in range(3)]
    cell_y, cell_x = np.arange(ho)[:, None], np.arange(wo)[None, :]

    best_d = np.full((hi, wi), np.inf)
    best_s = np.full((hi, wi), -1, dtype=np.int64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1, x0, x1 = 0, hi, 0, wi
            if windowed:
                # Superpixel (sy, sx) is a candidate at this offset for the pixels of home cell
                # (sy − dy, sx − dx). Restrict the pass to the pixel rectangle covering the
                # cells whose candidate's center can lie inside the window at all.
                home_y, home_x = cell_y - dy, cell_x - dx
                reach = ((home_y >= 0) & (home_y < ho) & (home_x >= 0) & (home_x < wo)
                         & (center_x >= (home_x - 1) * rx - 1) & (center_x <= (home_x + 2) * rx + 1)
                         & (center_y >= (home_y - 1) * ry - 1) & (center_y <= (home_y + 2) * ry + 1))
                if not reach.any():
                    continue
                sy, sx = np.nonzero(reach)
                y0 = int(np.searchsorted(grid.home_y, sy.min() - dy, side="left"))
                y1 = int(np.searchsorted(grid.home_y, sy.max() - dy, side="right"))
                x0 = int(np.searchsorted(grid.home_x, sx.min() - dx, side="left"))
                x1 = int(np.searchsorted(grid.home_x, sx.max() - dx, side="right"))
            gy = np.clip(grid.home_y[y0:y1] + dy, 0, ho - 1)
            gx = np.clip(grid.home_x[x0:x1] + dx, 0, wo - 1)
            repeats = np.bincount(gx, minlength=wo)   # gx is non-decreasing: runs per cell

            def spread(plane):
                """(ho, wo) per-superpixel values → the candidate's value at every pixel."""
                return np.repeat(plane[gy], repeats, axis=1)

            ddx = spread(planes[0])
            np.subtract(grid.px[None, x0:x1], ddx, out=ddx)
            ddy = spread(planes[1])
            np.subtract(grid.py[y0:y1, None], ddy, out=ddy)
            d = ddx * ddx
            d += ddy * ddy
            np.sqrt(d, out=d)
            d *= coef
            if channels is not None:
                acc = None
                for c in range(3):
                    diff = spread(planes[2 + c])
                    np.subtract(channels[c][y0:y1, x0:x1], diff, out=diff)
                    np.multiply(diff, diff, out=diff)
                    if acc is None:
                        acc = diff
                    else:
                        acc += diff
                np.sqrt(acc, out=acc)
                d += acc
            sub_d, sub_s = best_d[y0:y1, x0:x1], best_s[y0:y1, x0:x1]
            better = d < sub_d
            if windowed:
                np.abs(ddx, out=ddx)
                better &= ddx <= rx
                np.abs(ddy, out=ddy)
                better &= ddy <= ry
            np.copyto(sub_d, d, where=better)
            np.copyto(sub_s, (gy * wo)[:, None] + gx[None, :], where=better)
    return best_s


def run(lab: np.ndarray, mask: np.ndarray, out_width: int, out_height: int, config) -> Downscaled:
    hi, wi = mask.shape
    wo, ho = int(out_width), int(out_height)
    n_sp = wo * ho
    grid = _Grid(hi, wi, ho, wo)
    coef = config.g_m * math.sqrt(n_sp / (hi * wi))

    ys, xs = np.nonzero(mask)
    px, py = xs + 0.5, ys + 0.5
    col = np.ascontiguousarray(lab[ys, xs], dtype=np.float64)
    channels = [np.ascontiguousarray(lab[..., c], dtype=np.float64) for c in range(3)]
    all_opaque = bool(mask.all())

    # INITIALIZE: regular grid of centers, every pixel assigned to the nearest center.
    cx, cy = grid.cx0.copy(), grid.cy0.copy()
    assign = grid.home_y[ys] * wo + grid.home_x[xs]
    count = np.bincount(assign, minlength=n_sp)
    has_color = count > 0
    mean = np.zeros((n_sp, 3))
    for c in range(3):
        sums = np.bincount(assign, weights=col[:, c], minlength=n_sp)
        mean[has_color, c] = sums[has_color] / count[has_color]

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
        if assigned.all():
            a, a_px, a_py, a_col = assign, px, py, col
        else:
            a, a_px, a_py, a_col = assign[assigned], px[assigned], py[assigned], col[assigned]
        count = np.bincount(a, minlength=n_sp)
        got = count > 0   # superpixels with zero pixels keep their previous center and color
        for c in range(3):
            sums = np.bincount(a, weights=a_col[:, c], minlength=n_sp)
            mean[got, c] = sums[got] / count[got]
        cx[got] = np.bincount(a, weights=a_px, minlength=n_sp)[got] / count[got]
        cy[got] = np.bincount(a, weights=a_py, minlength=n_sp)[got] / count[got]
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
        # P(p_s) is uniform over the superpixels that hold opaque pixels (1/N for an opaque
        # input); superpixels lying entirely in the transparent region have no color.
        annealer.step(points, np.full(len(active), 1.0 / len(active)))
        k[active] = annealer.assign
        sp_color = annealer.colors[k]
        iterations += 1
        if annealer.done:
            hit_cap = False
            break

    # POST: saturation, then indices / palette / mask.
    palette = annealer.colors.copy()
    palette[:, 1:] *= config.saturation_beta

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
        indices=k.reshape(ho, wo).copy(),
        mean_lab=smooth,
        stats={"iterations": iterations, "palette_size": int(len(palette)),
               "hit_iteration_cap": bool(hit_cap), "final_temperature": float(annealer.T)},
    )
