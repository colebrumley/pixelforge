"""Content-adaptive downscaling (Kopf, Shamir, Peers 2013), REQUIREMENTS.md Section 6.

A constrained EM over one anisotropic Gaussian kernel per output pixel. All spatial quantities
are in output-pixel units. Arrays are laid out "pixel-major": every input pixel is covered by
at most n_slots × n_slots kernels (those whose R_k window contains it), so per-kernel/per-pixel
weights live in arrays of shape (slots, hi, wi). Per-pixel normalization is then a reduction
over the slot axis, and per-kernel sums are block sums (np.add.reduceat) because the pixels
that see a given kernel in a given slot form a contiguous block.
"""

from __future__ import annotations

import math

import numpy as np

from . import Downscaled

RK_HALF_WIDTH = 2.0          # R_k half-width in output units
SIGMA_INIT = 1e-4
SIGMA_CAP = 1.0
SIGMA_GROWTH = 1.1
COV_INIT = 1.0 / 3.0
SINGULAR_MIN, SINGULAR_MAX = 0.05, 0.1
CLAMP_BOX = 0.25
DIRECTIONAL_VARIANCE_MAX = 0.2
OVERLAP_MIN = 0.08
ORIENTATION_MAX_DEGREES = 25.0
# A kernel holding less than this fraction of a cell's worth of pixels counts as starved.
STARVED_FRACTION = 0.02
# Convergence allows σ to still change on fewer than this fraction of the kernels. On noisy
# content 0.5–1 % of the kernels keep growing σ by 1.1× per iteration long after μ and ν
# have settled (σ starts at 1e-4, so reaching the cap takes ~100 steps), and at 0.1 % no
# real image converged within 50 iterations. At 2 % the result differs from the
# 50-iteration one by a mean ΔE below 0.1 on the noisy fixtures.
CHANGED_SIGMA_FRACTION = 0.02

_NEIGHBORS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def to_unit_cube(lab: np.ndarray) -> np.ndarray:
    out = np.empty(lab.shape, dtype=np.float64)
    out[..., 0] = lab[..., 0] / 100.0
    out[..., 1] = (lab[..., 1] + 128.0) / 255.0
    out[..., 2] = (lab[..., 2] + 128.0) / 255.0
    return out


def from_unit_cube(cube: np.ndarray) -> np.ndarray:
    out = np.empty(cube.shape, dtype=np.float64)
    out[..., 0] = cube[..., 0] * 100.0
    out[..., 1] = cube[..., 1] * 255.0 - 128.0
    out[..., 2] = cube[..., 2] * 255.0 - 128.0
    return out


class _Axis:
    """Window bookkeeping along one image axis."""

    def __init__(self, n_in: int, n_out: int):
        self.n_in, self.n_out = n_in, n_out
        self.pos = (np.arange(n_in) + 0.5) * (n_out / n_in)   # pixel positions, output units
        self.n_slots = int(math.ceil(2 * RK_HALF_WIDTH))
        # Kernel k (center k + 0.5) covers a pixel iff |pos − (k + 0.5)| < half-width; `base`
        # is the first such k, and slot j refers to kernel base + j.
        self.base = np.floor(self.pos - 0.5 - RK_HALF_WIDTH).astype(np.int64) + 1
        kern = self.base[None, :] + np.arange(self.n_slots)[:, None]
        self.valid = ((kern >= 0) & (kern < n_out)
                      & (np.abs(self.pos[None, :] - (kern + 0.5)) < RK_HALF_WIDTH))
        self.kern = np.clip(kern, 0, n_out - 1)
        # Pixels each kernel's window would span on an unbounded pixel lattice.
        lattice = (np.arange(-n_in, 2 * n_in) + 0.5) * (n_out / n_in)
        self.window = np.count_nonzero(
            np.abs(lattice[None, :] - (np.arange(n_out)[:, None] + 0.5)) < RK_HALF_WIDTH, axis=1)
        self.repeats = [np.bincount(k, minlength=n_out) for k in self.kern]
        # Runs of equal `base`: within a slot, each run is the block of one kernel.
        self.starts = np.concatenate([[0], np.nonzero(np.diff(self.base))[0] + 1])
        run_kernel = self.base[self.starts][None, :] + np.arange(self.n_slots)[:, None]
        self.sel = []
        for j in range(self.n_slots):
            runs = np.nonzero((run_kernel[j] >= 0) & (run_kernel[j] < n_out))[0]
            self.sel.append((runs, run_kernel[j][runs]))


class _Layout:
    def __init__(self, hi: int, wi: int, ho: int, wo: int):
        self.y, self.x = _Axis(hi, ho), _Axis(wi, wo)
        self.ns = self.y.n_slots
        self.slots = [(jy, jx) for jy in range(self.ns) for jx in range(self.ns)]

    def expand(self, per_kernel: np.ndarray, jy: int, jx: int) -> np.ndarray:
        """(ho, wo) per-kernel values → (hi, wi): the value of each pixel's slot kernel."""
        # kern[jx] is non-decreasing, so repeating each column by its run length is the same
        # as np.take along axis 1, only several times faster.
        return np.repeat(np.take(per_kernel, self.y.kern[jy], axis=0), self.x.repeats[jx], axis=1)

    def reduce(self, per_pixel: np.ndarray, jy: int, jx: int, out: np.ndarray,
               ufunc=np.add) -> None:
        """Fold a slot's (hi, wi) values into per-kernel `out` (ho, wo) with `ufunc`."""
        blocks = ufunc.reduceat(ufunc.reduceat(per_pixel, self.x.starts, axis=1),
                                self.y.starts, axis=0)
        rows, ky = self.y.sel[jy]
        cols, kx = self.x.sel[jx]
        target = np.ix_(ky, kx)
        out[target] = ufunc(out[target], blocks[np.ix_(rows, cols)])

    def slot_valid(self, jy: int, jx: int, mask: np.ndarray | None) -> np.ndarray:
        valid = self.y.valid[jy][:, None] & self.x.valid[jx][None, :]
        return valid if mask is None else valid & mask


def _inverse_cov(cov: np.ndarray):
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    det = a * d - b * b
    return d / det, -b / det, a / det


def _clamp_singular_values(cov: np.ndarray) -> np.ndarray:
    # Σ is symmetric positive semi-definite, so its SVD is its eigendecomposition.
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, SINGULAR_MIN, SINGULAR_MAX)
    return (evecs * evals[..., None, :]) @ np.swapaxes(evecs, -1, -2)


def _neighbor_mean4(grid: np.ndarray) -> np.ndarray:
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
    return np.where(count > 0, total / np.maximum(count, 1), grid)


def _shift(grid: np.ndarray, dy: int, dx: int, fill=0.0) -> np.ndarray:
    """out[ky, kx] = grid[ky + dy, kx + dx], `fill` outside the grid."""
    h, w = grid.shape
    out = np.full_like(grid, fill)
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    yd = slice(max(0, dy), min(h, h + dy))
    xd = slice(max(0, dx), min(w, w + dx))
    out[ys, xs] = grid[yd, xd]
    return out


def run(lab: np.ndarray, mask: np.ndarray, out_width: int, out_height: int, config) -> Downscaled:
    hi, wi = mask.shape
    wo, ho = int(out_width), int(out_height)
    layout = _Layout(hi, wi, ho, wo)
    slots, ns = layout.slots, layout.ns
    pos_x, pos_y = layout.x.pos[None, :], layout.y.pos[:, None]
    cube = to_unit_cube(lab)
    channels = [np.ascontiguousarray(cube[..., c]) for c in range(3)]
    pixel_mask = None if mask.all() else mask

    # INITIALIZE
    grid_x, grid_y = np.meshgrid(np.arange(wo) + 0.5, np.arange(ho) + 0.5)
    mu_x, mu_y = grid_x.copy(), grid_y.copy()
    cov = np.zeros((ho, wo, 2, 2))
    cov[..., 0, 0] = cov[..., 1, 1] = COV_INIT
    nu = np.full((ho, wo, 3), 0.5)
    sigma = np.full((ho, wo), SIGMA_INIT)

    gamma = np.empty((len(slots), hi, wi))
    # Per-slot work buffers, reused across slots and iterations. The offsets p_i − μ_k are
    # recomputed where they are needed instead of being stored for every slot.
    dx, dy, quad, work, work2 = (np.empty((hi, wi)) for _ in range(5))
    positive = np.empty((hi, wi), dtype=bool)
    weight_sum = np.zeros((ho, wo))
    iterations = 0
    converged = False
    cos_limit = math.cos(math.radians(ORIENTATION_MAX_DEGREES))
    starve_below = STARVED_FRACTION * (hi / ho) * (wi / wo)
    has_pixels = np.zeros((ho, wo))
    in_image = np.zeros((ho, wo))
    for jy, jx in slots:
        layout.reduce(layout.slot_valid(jy, jx, pixel_mask).astype(np.float64), jy, jx, has_pixels)
        layout.reduce(layout.slot_valid(jy, jx, None).astype(np.float64), jy, jx, in_image)
    # A kernel is truncated when its R_k window holds transparent pixels or reaches past the
    # image border (fewer in-image pixels than the window spans); see the shape constraints.
    truncated = (has_pixels < in_image) | (in_image < np.outer(layout.y.window, layout.x.window))
    has_pixels = has_pixels > 0

    def offsets(cx: np.ndarray, cy: np.ndarray, jy: int, jx: int) -> None:
        """dx, dy ← p_i − c_k for each pixel's slot kernel."""
        np.subtract(pos_x, layout.expand(cx, jy, jx), out=dx)
        np.subtract(pos_y, layout.expand(cy, jy, jx), out=dy)

    def spatial_quad(ia, ib, ic, jy: int, jx: int) -> None:
        """quad ← (p_i − μ_k)ᵀ Σ_k⁻¹ (p_i − μ_k), from dx, dy."""
        np.multiply(layout.expand(ia, jy, jx), dx, out=quad)
        np.multiply(quad, dx, out=quad)
        np.multiply(layout.expand(ib, jy, jx), 2.0, out=work)
        np.multiply(work, dx, out=work)
        np.multiply(work, dy, out=work)
        np.add(quad, work, out=quad)
        np.multiply(layout.expand(ic, jy, jx), dy, out=work)
        np.multiply(work, dy, out=work)
        np.add(quad, work, out=quad)

    for _ in range(config.kopf_max_iters):
        iterations += 1
        # ------------------------------------------------------------------ E-STEP
        inv_a, inv_b, inv_c = _inverse_cov(cov)
        inv_color = 1.0 / (2.0 * sigma * sigma)
        for s, (jy, jx) in enumerate(slots):
            offsets(mu_x, mu_y, jy, jx)
            spatial_quad(inv_a, inv_b, inv_c, jy, jx)
            dist = work2
            for c in range(3):
                np.subtract(channels[c], layout.expand(nu[..., c], jy, jx), out=work)
                work *= work
                if c == 0:
                    dist[...] = work
                else:
                    dist += work
            logw = gamma[s]
            np.multiply(quad, -0.5, out=logw)
            dist *= layout.expand(inv_color, jy, jx)
            logw -= dist
            logw[~layout.slot_valid(jy, jx, pixel_mask)] = -np.inf

        # DEVIATION: Section 6 — the per-kernel normalization w_k(i) /= Σ_{i∈R_k} w_k(i) is
        # skipped. It rescales every kernel relative to its own best-fitting pixels, so a
        # kernel sitting on a thin feature (which covers < 50% of its cell) can never become
        # more selective for the feature than for the background, and the darkest kernel in
        # reach takes all feature pixels from its neighbors: the 2-px line of
        # test_thin_line_stays_connected comes out as isolated dark blobs with gaps. Without
        # it the kernels compete on absolute bilateral fit and latch onto the line. Weights
        # stay in log space so the tiny initial σ cannot underflow; the spec's "sum == 0 →
        # uniform over R_k" guard is replaced by the re-seeding of starved kernels below.
        # Normalize per pixel: γ_k(i) = w_k(i) / Σ_n w_n(i), from the log-weights.
        pixel_peak = gamma.max(axis=0)
        pixel_peak[~np.isfinite(pixel_peak)] = 0.0
        gamma -= pixel_peak
        np.exp(gamma, out=gamma)
        pixel_sum = gamma.sum(axis=0)
        pixel_sum[pixel_sum == 0] = 1.0    # guard div-by-0 → γ = 0
        gamma /= pixel_sum

        # ------------------------------------------------------------------ M-STEP
        sums = {name: np.zeros((ho, wo)) for name in
                ("w", "x", "y", "xx", "xy", "yy", "c0", "c1", "c2")}
        for s, (jy, jx) in enumerate(slots):
            g = gamma[s]
            offsets(mu_x, mu_y, jy, jx)
            gx, gy = np.multiply(g, dx, out=quad), np.multiply(g, dy, out=work)
            layout.reduce(g, jy, jx, sums["w"])
            layout.reduce(gx, jy, jx, sums["x"])
            layout.reduce(gy, jy, jx, sums["y"])
            layout.reduce(np.multiply(gx, dx, out=work2), jy, jx, sums["xx"])
            layout.reduce(np.multiply(gx, dy, out=work2), jy, jx, sums["xy"])
            layout.reduce(np.multiply(gy, dy, out=work2), jy, jx, sums["yy"])
            for c in range(3):
                layout.reduce(np.multiply(g, channels[c], out=work2), jy, jx, sums[f"c{c}"])
        weight_sum = sums["w"]
        fed = weight_sum > 1e-12        # starved kernels keep their previous parameters
        wsafe = np.where(fed, weight_sum, 1.0)
        old_mu_x, old_mu_y, old_nu, old_sigma = mu_x, mu_y, nu, sigma
        # Σ_k uses the OLD μ_k: dx, dy above are offsets from it.
        new_cov = np.empty_like(cov)
        new_cov[..., 0, 0] = sums["xx"] / wsafe
        new_cov[..., 0, 1] = new_cov[..., 1, 0] = sums["xy"] / wsafe
        new_cov[..., 1, 1] = sums["yy"] / wsafe
        cov = np.where(fed[..., None, None], new_cov, cov)
        m_mu_x = np.where(fed, mu_x + sums["x"] / wsafe, mu_x)
        m_mu_y = np.where(fed, mu_y + sums["y"] / wsafe, mu_y)
        nu = np.where(fed[..., None],
                      np.stack([sums[f"c{c}"] / wsafe for c in range(3)], axis=-1), nu)

        # ------------------------------------------------------------------ C-STEP
        # Spatial: Laplacian smoothing, then clamp to a box around the grid position.
        # DEVIATION: Section 6 — the neighbor mean is taken over displacements from the grid
        # position, not raw positions. The two are identical for interior kernels; for border
        # kernels (fewer than 4 neighbors) the raw mean lies inside the image and would drag
        # them inward every iteration, a disturbance that then diffuses across the whole grid
        # and keeps |Δμ| above kopf_tol.
        off_x, off_y = m_mu_x - grid_x, m_mu_y - grid_y
        mu_x = grid_x + np.clip(0.5 * off_x + 0.5 * _neighbor_mean4(off_x), -CLAMP_BOX, CLAMP_BOX)
        mu_y = grid_y + np.clip(0.5 * off_y + 0.5 * _neighbor_mean4(off_y), -CLAMP_BOX, CLAMP_BOX)
        cov = _clamp_singular_values(cov)

        # Shape constraints, with (p_i − μ_k) relative to the constrained μ_k.
        move_x, move_y = mu_x - old_mu_x, mu_y - old_mu_y
        r = math.sqrt(0.5)
        axes = {(0, 1): (1.0, 0.0), (1, 0): (0.0, 1.0), (1, 1): (r, r), (-1, 1): (r, -r)}
        plus = {axis: np.zeros((ho, wo)) for axis in axes}
        both = {axis: np.zeros((ho, wo)) for axis in axes}
        for s, (jy, jx) in enumerate(slots):
            # (p_i − old μ_k) − (μ_k − old μ_k), in this order so that the result does not
            # depend on whether the offsets were stored or are recomputed.
            offsets(old_mu_x, old_mu_y, jy, jx)
            dx -= layout.expand(move_x, jy, jx)
            dy -= layout.expand(move_y, jy, jx)
            for axis, (ex, ey) in axes.items():
                u, t = np.multiply(dx, ex, out=quad), work
                u += np.multiply(dy, ey, out=work2)
                np.multiply(gamma[s], u, out=t)
                t *= u
                layout.reduce(t, jy, jx, both[axis])
                np.greater(u, 0, out=positive)
                np.logical_not(positive, out=positive)
                t[positive] = 0.0
                layout.reduce(t, jy, jx, plus[axis])
        toward = {}
        for (ndy, ndx) in axes:
            toward[(ndy, ndx)] = plus[(ndy, ndx)]
            toward[(-ndy, -ndx)] = both[(ndy, ndx)] - plus[(ndy, ndx)]

        overlap = {}
        for ndy, ndx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            f = np.zeros((ho, wo))
            for s, (jy, jx) in enumerate(slots):
                ny, nx = jy + ndy, jx + ndx
                if 0 <= ny < ns and 0 <= nx < ns:
                    layout.reduce(gamma[s] * gamma[ny * ns + nx], jy, jx, f)
            overlap[(ndy, ndx)] = f
            overlap[(-ndy, -ndx)] = _shift(f, -ndy, -ndx)   # f is symmetric in (k, n)

        # DEVIATION: Section 6 — s and f are divided by Σ_i γ_k(i). As literal sums they scale
        # with rx·ry (≈ the number of input pixels per kernel), so the thresholds 0.2 and
        # 0.08 would fire for every kernel on every iteration and σ would saturate at its
        # cap, turning the method into a box filter. Normalized, s is a directional variance
        # in output units² and f a mean overlap, which is what the thresholds are scaled for.
        #
        # DEVIATION: Section 6 — the edge orientation o = Σ_{i∈R_k} ∇(γ_k / (γ_k + γ_n)) is
        # replaced by the direction between the two kernels' M-step centroids. The summed
        # gradient telescopes to the ratio on the border of the R_k window, where both γ are
        # (numerically) zero, so the literal estimate is noise: it fired for thousands of
        # kernels per iteration on a plain checkerboard and steadily blurred a result that was
        # exact after five iterations. The integral it stands for is the net normal of the
        # boundary between the two kernels' pixel sets, which is what the centroid direction
        # measures robustly.
        #
        # DEVIATION: Section 6 — both tests are skipped for kernels whose R_k window is
        # truncated by transparent pixels or the image border, and the orientation test is
        # skipped for pairs that share no pixels (f == 0: there is no edge between them).
        # A truncated window gives the kernel a lopsided footprint, so the one-sided variance
        # toward the cut side and the centroid direction are set by the cut, not by the
        # content, and growing σ never changes that geometry: on circle_alpha 256→32 the same
        # ~340 kernels in a 2–4-kernel band along the silhouette fired on every iteration,
        # 248 ended at SIGMA_CAP, and those kernels lost their color selectivity (a dark line
        # 4 px inside the silhouette came out fainter than with the box filter).
        fires = np.zeros((ho, wo), dtype=np.int64)
        whole = fed & ~truncated
        for ndy, ndx in _NEIGHBORS8:
            exists = _shift(np.ones((ho, wo), dtype=bool), ndy, ndx, fill=False)
            pair = exists & whole & _shift(~truncated, ndy, ndx, fill=False)
            s_dir = toward[(ndy, ndx)] / wsafe
            f = overlap[(ndy, ndx)] / wsafe
            norm = math.hypot(ndx, ndy)
            d_x, d_y = ndx / norm, ndy / norm
            o_x = _shift(m_mu_x, ndy, ndx) - m_mu_x
            o_y = _shift(m_mu_y, ndy, ndx) - m_mu_y
            o_len = np.hypot(o_x, o_y)
            cos_angle = (d_x * o_x + d_y * o_y) / np.where(o_len > 0, o_len, 1.0)
            misaligned = (o_len > 0) & (cos_angle < cos_limit)
            fire = ((exists & whole & (s_dir > DIRECTIONAL_VARIANCE_MAX))
                    | (pair & (f > 0) & (f < OVERLAP_MIN) & misaligned))
            fires += fire                               # σ_k *= 1.1
            fires += _shift(fire, -ndy, -ndx, fill=False)  # σ_n *= 1.1
        sigma = np.minimum(sigma * SIGMA_GROWTH ** fires, SIGMA_CAP)

        # Starved kernels: no pixel prefers the kernel's color mean (its Σ γ is ~0), typically
        # because that mean is a blend of two colors while a neighbor matches each of them
        # exactly. In place of the spec's "sum == 0 → uniform over R_k" guard, the kernel is
        # re-seeded with the color of the input pixel under its center, which it then matches
        # exactly and competes for on spatial terms alone.
        starved = has_pixels & (weight_sum < starve_below)
        if starved.any():
            seed_y = np.clip(np.floor(mu_y[starved] * (hi / ho)).astype(np.int64), 0, hi - 1)
            seed_x = np.clip(np.floor(mu_x[starved] * (wi / wo)).astype(np.int64), 0, wi - 1)
            usable = mask[seed_y, seed_x]
            ky, kx = np.nonzero(starved)
            nu[ky[usable], kx[usable]] = cube[seed_y[usable], seed_x[usable]]

        # ------------------------------------------------------------- CONVERGENCE
        # DEVIATION: Section 6 — the spec's test (max |Δμ| and max |Δν| < kopf_tol, no σ
        # change) is never met on real content: border truncation and non-integer ratios
        # keep a few centroids moving by ~1e-3 to 5e-2 output pixels, and a handful of
        # kernels grow σ every iteration. Converged instead means: RMS |Δμ| over fed kernels
        # < 10·kopf_tol output pixels, RMS |Δν| (unit-cube color) < kopf_tol, and σ changed
        # on fewer than CHANGED_SIGMA_FRACTION of the kernels (those already at the cap
        # cannot change and do not count).
        if fed.any():
            d_mu = math.sqrt(float(np.mean(((mu_x - old_mu_x) ** 2
                                             + (mu_y - old_mu_y) ** 2)[fed])))
            d_nu = math.sqrt(float(np.mean(((nu - old_nu) ** 2).sum(axis=-1)[fed])))
        else:
            d_mu = d_nu = 0.0
        sigma_changed = int(np.count_nonzero((sigma != old_sigma) & (old_sigma < SIGMA_CAP)))
        if (d_mu < 10.0 * config.kopf_tol and d_nu < config.kopf_tol
                and sigma_changed < CHANGED_SIGMA_FRACTION * ho * wo):
            converged = True
            break

    # OUTPUT: small_mask[k] = Σ γ_k(i)·mask_i / Σ γ_k(i) >= 0.5.
    # DEVIATION: Section 6 — transparent pixels are excluded from R_k, so with the EM's own γ
    # that ratio is always 1. The coverage is measured with spatial-only responsibilities
    # (the final μ, Σ without the color term) over all pixels instead; kernels that ended up
    # starved take their color from the same weights.
    starved = weight_sum < starve_below
    small_mask = ~starved
    if pixel_mask is not None or starved.any():
        inv_a, inv_b, inv_c = _inverse_cov(cov)
        for s, (jy, jx) in enumerate(slots):
            offsets(mu_x, mu_y, jy, jx)
            spatial_quad(inv_a, inv_b, inv_c, jy, jx)
            logw = gamma[s]
            np.multiply(quad, -0.5, out=logw)
            logw[~layout.slot_valid(jy, jx, None)] = -np.inf
        pixel_peak = gamma.max(axis=0)
        pixel_peak[~np.isfinite(pixel_peak)] = 0.0
        gamma -= pixel_peak
        np.exp(gamma, out=gamma)
        pixel_sum = gamma.sum(axis=0)
        pixel_sum[pixel_sum == 0] = 1.0
        gamma /= pixel_sum
        opaque = mask.astype(np.float64)
        all_w, opaque_w = np.zeros((ho, wo)), np.zeros((ho, wo))
        color_w = [np.zeros((ho, wo)) for _ in range(3)]
        for s, (jy, jx) in enumerate(slots):
            layout.reduce(gamma[s], jy, jx, all_w)
            g = gamma[s] * opaque
            layout.reduce(g, jy, jx, opaque_w)
            for c in range(3):
                layout.reduce(g * channels[c], jy, jx, color_w[c])
        small_mask = (opaque_w > 0) & (opaque_w * 2 >= all_w)
        refill = starved & (opaque_w > 0)
        for c in range(3):
            nu[refill, c] = color_w[c][refill] / opaque_w[refill]

    return Downscaled(small_lab=from_unit_cube(nu), small_mask=small_mask,
                      stats={"iterations": iterations, "converged": converged,
                             "sigma_capped": int(np.count_nonzero(sigma >= SIGMA_CAP))})
