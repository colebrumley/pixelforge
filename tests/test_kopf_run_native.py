"""`_native.kopf_run` (crates/pixelforge/src/kopf_run.rs) against the Python control flow it
replaced.

The reference `ref_run` is the `kopf.run` that `pixelforge/downscale/kopf.py` held before it
delegated to the native code, copied here verbatim over the native per-pixel loops (which
tests/test_kopf_native.py checks against the numpy loops). The port reproduces numpy's
operations in order; the routines numpy delegates to LAPACK/BLAS/libm are reproduced as
numpy's macOS (Accelerate) build computes them:

- `np.linalg.eigh` on 2×2 (`dsyevd` → `dsteqr` → `dlaev2`) and the reconstruction
  `(V · w) @ Vᵀ`, with the fused multiply-adds Accelerate performs;
- `np.hypot` as `sqrt(fma(small, small, big²))`;
- `1.1 ** fires` from a table of correctly rounded powers.

So with an Accelerate numpy the whole run is bitwise the reference; with another LAPACK
(OpenBLAS on Linux) the eigen step and hypot differ in the last bits, so there the reference
takes the native eigen step and the comparison is to a tolerance.
"""

import logging
import math

import numpy as np
import pytest

from pixelforge import Config, _native
from pixelforge.downscale import Downscaled, kopf, neighbor_mean4
from pixelforge.downscale.box import cell_index


def _numpy_lapack() -> str:
    try:
        deps = np.show_config(mode="dicts")["Build Dependencies"]
        return str(deps["lapack"]["name"]).lower()
    except Exception:   # pragma: no cover - older numpy
        return "unknown"


ACCELERATE = _numpy_lapack() == "accelerate"
FIXTURES = ["checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color"]


# ------------------------------------------------------------------ reference (Python)

RK_HALF_WIDTH = 2.0
SIGMA_INIT = 1e-4
SIGMA_CAP = 1.0
SIGMA_GROWTH = 1.1
COV_INIT = 1.0 / 3.0
SINGULAR_MIN, SINGULAR_MAX = 0.05, 0.1
CLAMP_BOX = 0.25
DIRECTIONAL_VARIANCE_MAX = 0.2
OVERLAP_MIN = 0.08
ORIENTATION_MAX_DEGREES = 25.0
STARVED_FRACTION = 0.02
CHANGED_SIGMA_FRACTION = 0.02
PROGRESS_EVERY = 10
_NEIGHBORS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ref_log = logging.getLogger("pixelforge.downscale.kopf")


def to_unit_cube(lab):
    out = np.empty(lab.shape, dtype=np.float64)
    out[..., 0] = lab[..., 0] / 100.0
    out[..., 1] = (lab[..., 1] + 128.0) / 255.0
    out[..., 2] = (lab[..., 2] + 128.0) / 255.0
    return out


def from_unit_cube(cube):
    out = np.empty(cube.shape, dtype=np.float64)
    out[..., 0] = cube[..., 0] * 100.0
    out[..., 1] = cube[..., 1] * 255.0 - 128.0
    out[..., 2] = cube[..., 2] * 255.0 - 128.0
    return out


def _window(n_in, n_out):
    pos = (np.arange(n_in) + 0.5) * (n_out / n_in)
    return np.abs(pos[None, :] - (np.arange(n_out)[:, None] + 0.5)) < RK_HALF_WIDTH


def _window_span(n_in, n_out):
    lattice = (np.arange(-n_in, 2 * n_in) + 0.5) * (n_out / n_in)
    return np.count_nonzero(
        np.abs(lattice[None, :] - (np.arange(n_out)[:, None] + 0.5)) < RK_HALF_WIDTH, axis=1)


def _inverse_cov(cov):
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    det = a * d - b * b
    return d / det, -b / det, a / det


def numpy_clamp_singular_values(cov):
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, SINGULAR_MIN, SINGULAR_MAX)
    return (evecs * evals[..., None, :]) @ np.swapaxes(evecs, -1, -2)


def native_clamp_singular_values(cov):
    shape = cov.shape
    return _native.kopf_clamp_singular_values(
        np.ascontiguousarray(cov.reshape(-1, 2, 2))).reshape(shape)


def _shift(grid, dy, dx, fill=0.0):
    h, w = grid.shape
    out = np.full_like(grid, fill)
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    yd = slice(max(0, dy), min(h, h + dy))
    xd = slice(max(0, dx), min(w, w + dx))
    out[ys, xs] = grid[yd, xd]
    return out


def _log_iteration(it, max_iters, d_mu, d_nu, changed):
    level = logging.INFO if it % PROGRESS_EVERY == 0 else logging.DEBUG
    if ref_log.isEnabledFor(level):
        ref_log.log(level, "kopf iter %d/%d rms dmu=%.2e dnu=%.2e sigma_changed=%.1f%%",
                    it, max_iters, d_mu, d_nu, 100.0 * changed)


def ref_run(lab, mask, out_width, out_height, config, weight=None,
            clamp=numpy_clamp_singular_values):
    """The Python kopf.run (comments trimmed); `clamp` swaps the eigen step."""
    hi, wi = mask.shape
    wo, ho = int(out_width), int(out_height)
    cube = to_unit_cube(lab)
    channels = np.ascontiguousarray(np.moveaxis(cube, -1, 0))
    mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
    pixel_weight = None
    if weight is not None and not np.all(weight[mask] == 1.0):
        pixel_weight = np.ascontiguousarray(np.where(mask, weight, 0.0), dtype=np.float64)

    grid_x, grid_y = np.meshgrid(np.arange(wo) + 0.5, np.arange(ho) + 0.5)
    mu_x, mu_y = grid_x.copy(), grid_y.copy()
    cov = np.zeros((ho, wo, 2, 2))
    cov[..., 0, 0] = cov[..., 1, 1] = COV_INIT
    nu = np.full((ho, wo, 3), 0.5)
    sigma = np.full((ho, wo), SIGMA_INIT)

    gamma = np.empty((hi, wi, 16))
    sums = np.empty((ho, wo, 9))
    plus, both, overlap_sums = (np.empty((4, ho, wo)) for _ in range(3))
    weight_sum = np.zeros((ho, wo))
    iterations = 0
    converged = False
    cos_limit = math.cos(math.radians(ORIENTATION_MAX_DEGREES))
    starve_below = STARVED_FRACTION * (hi / ho) * (wi / wo)
    win_y = _window(hi, ho).astype(np.float64)
    win_x = _window(wi, wo).astype(np.float64)
    has_pixels = win_y @ mask.astype(np.float64) @ win_x.T
    in_image = np.outer(win_y.sum(axis=1), win_x.sum(axis=1))
    truncated = (has_pixels < in_image) | (in_image < np.outer(_window_span(hi, ho),
                                                               _window_span(wi, wo)))
    has_pixels = has_pixels > 0
    cell = (cell_index(hi, ho)[:, None] * wo + cell_index(wi, wo)[None, :]).ravel()
    cell_total = np.bincount(cell, minlength=ho * wo).reshape(ho, wo)
    cell_opaque = np.bincount(cell[mask.ravel()], minlength=ho * wo).reshape(ho, wo)
    ghost = cell_opaque * 2 < cell_total
    ghost_u8 = ghost.astype(np.uint8)

    for _ in range(config.kopf_max_iters):
        iterations += 1
        inv_a, inv_b, inv_c = _inverse_cov(cov)
        inv_color = 1.0 / (2.0 * sigma * sigma)
        _native.kopf_em_step(channels, mask_u8, pixel_weight, mu_x, mu_y, inv_a, inv_b, inv_c,
                             nu, inv_color, ghost_u8, gamma, sums)

        weight_sum = sums[..., 0].copy()
        fed = weight_sum > 1e-12
        wsafe = np.where(fed, weight_sum, 1.0)
        old_mu_x, old_mu_y, old_nu, old_sigma = mu_x, mu_y, nu, sigma
        new_cov = np.empty_like(cov)
        new_cov[..., 0, 0] = sums[..., 3] / wsafe
        new_cov[..., 0, 1] = new_cov[..., 1, 0] = sums[..., 4] / wsafe
        new_cov[..., 1, 1] = sums[..., 5] / wsafe
        cov = np.where(fed[..., None, None], new_cov, cov)
        m_mu_x = np.where(fed, mu_x + sums[..., 1] / wsafe, mu_x)
        m_mu_y = np.where(fed, mu_y + sums[..., 2] / wsafe, mu_y)
        nu = np.where(fed[..., None],
                      np.stack([sums[..., 6 + c] / wsafe for c in range(3)], axis=-1), nu)

        off_x, off_y = m_mu_x - grid_x, m_mu_y - grid_y
        mu_x = grid_x + np.clip(0.5 * off_x + 0.5 * neighbor_mean4(off_x), -CLAMP_BOX, CLAMP_BOX)
        mu_y = grid_y + np.clip(0.5 * off_y + 0.5 * neighbor_mean4(off_y), -CLAMP_BOX, CLAMP_BOX)
        cov = clamp(cov)

        move_x, move_y = mu_x - old_mu_x, mu_y - old_mu_y
        _native.kopf_c_step(gamma, old_mu_x, old_mu_y, move_x, move_y, plus, both, overlap_sums)
        axes = ((0, 1), (1, 0), (1, 1), (-1, 1))
        toward = {}
        for a, (ndy, ndx) in enumerate(axes):
            toward[(ndy, ndx)] = plus[a]
            toward[(-ndy, -ndx)] = both[a] - plus[a]

        overlap = {}
        for d, (ndy, ndx) in enumerate(((0, 1), (1, 0), (1, 1), (1, -1))):
            f = overlap_sums[d]
            overlap[(ndy, ndx)] = f
            overlap[(-ndy, -ndx)] = _shift(f, -ndy, -ndx)

        fires = np.zeros((ho, wo), dtype=np.int64)
        shaped = ~truncated & ~ghost
        whole = fed & shaped
        for ndy, ndx in _NEIGHBORS8:
            exists = _shift(np.ones((ho, wo), dtype=bool), ndy, ndx, fill=False)
            pair = exists & whole & _shift(shaped, ndy, ndx, fill=False)
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
            fires += fire
            fires += _shift(fire, -ndy, -ndx, fill=False)
        sigma = np.minimum(sigma * SIGMA_GROWTH ** fires, SIGMA_CAP)

        starved = has_pixels & ~ghost & (weight_sum < starve_below)
        if starved.any():
            seed_y = np.clip(np.floor(mu_y[starved] * (hi / ho)).astype(np.int64), 0, hi - 1)
            seed_x = np.clip(np.floor(mu_x[starved] * (wi / wo)).astype(np.int64), 0, wi - 1)
            usable = mask[seed_y, seed_x]
            ky, kx = np.nonzero(starved)
            nu[ky[usable], kx[usable]] = cube[seed_y[usable], seed_x[usable]]

        if fed.any():
            d_mu = math.sqrt(float(np.mean(((mu_x - old_mu_x) ** 2
                                             + (mu_y - old_mu_y) ** 2)[fed])))
            d_nu = math.sqrt(float(np.mean(((nu - old_nu) ** 2).sum(axis=-1)[fed])))
        else:
            d_mu = d_nu = 0.0
        sigma_changed = int(np.count_nonzero((sigma != old_sigma) & (old_sigma < SIGMA_CAP)))
        _log_iteration(iterations, config.kopf_max_iters, d_mu, d_nu, sigma_changed / (ho * wo))
        if (d_mu < 10.0 * config.kopf_tol and d_nu < config.kopf_tol
                and sigma_changed < CHANGED_SIGMA_FRACTION * ho * wo):
            converged = True
            break

    ref_log.debug("kopf %s after %d iterations",
                  "converged" if converged else "stopped at the cap", iterations)

    small_mask = ~ghost
    starved = (weight_sum < starve_below) & small_mask
    if starved.any():
        inv_a, inv_b, inv_c = _inverse_cov(cov)
        pixel_mass = mask.astype(np.float64) if pixel_weight is None else pixel_weight
        mass_w = np.empty((ho, wo))
        color_w = np.empty((ho, wo, 3))
        _native.kopf_spatial_refill(channels, pixel_mass, mu_x, mu_y, inv_a, inv_b, inv_c,
                                    mass_w, color_w)
        refill = starved & (mass_w > 0)
        for c in range(3):
            nu[refill, c] = color_w[refill, c] / mass_w[refill]

    return Downscaled(small_lab=from_unit_cube(nu), small_mask=small_mask,
                      stats={"iterations": iterations, "converged": converged,
                             "sigma_capped": int(np.count_nonzero(sigma >= SIGMA_CAP))})


# ------------------------------------------------------------------ comparison

def compare(got, want, label):
    """Report and return (max |Δ| LAB, differing small_mask cells)."""
    delta = float(np.abs(got.small_lab - want.small_lab).max())
    mask_diff = int(np.count_nonzero(got.small_mask != want.small_mask))
    print(f"{label}: max|dLAB| {delta:.3e}, small_mask cells differing {mask_diff}, "
          f"stats {got.stats} vs {want.stats}")
    return delta, mask_diff


def reference(lab, mask, wo, ho, config, weight=None):
    """The reference with numpy's eigen step where that is Accelerate (bitwise), else with
    the native one (numpy's LAPACK then differs in the last bits; see the module doc)."""
    clamp = numpy_clamp_singular_values if ACCELERATE else native_clamp_singular_values
    return ref_run(lab, mask, wo, ho, config, weight, clamp=clamp)


def assert_matches(got, want, label):
    delta, mask_diff = compare(got, want, label)
    assert mask_diff == 0
    assert got.small_lab.shape == want.small_lab.shape
    if ACCELERATE:
        assert got.small_lab.tobytes() == want.small_lab.tobytes(), label
        assert got.stats == want.stats
    else:
        # np.hypot is the platform's: only near-ties of the orientation test can differ.
        assert delta < 1e-6, label
        assert got.stats["converged"] == want.stats["converged"]


# ------------------------------------------------------------------ the eigen step

def random_spd(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 0.4, n)
    c = rng.uniform(0.0, 0.4, n)
    b = rng.uniform(-1, 1, n) * np.sqrt(a * c) * 0.999
    b[:50] = 0.0                                   # diagonal
    c[50:100] = a[50:100]                          # equal diagonal
    b[100:150] = rng.uniform(-1e-20, 1e-20, 50)    # negligible off-diagonal
    a[150:160] = c[150:160] = 1.0 / 3.0            # kopf's initial covariance
    b[150:160] = 0.0
    a[160:170] = b[160:170] = c[160:170] = 0.0     # a kernel fed by one pixel at its centre
    cov = np.empty((n, 2, 2))
    cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 0], cov[:, 1, 1] = a, b, b, c
    return cov


@pytest.mark.parametrize("seed", [0, 1])
def test_eigh_matches_numpy(seed):
    cov = random_spd(10_000, seed)
    want_w, want_v = np.linalg.eigh(cov)
    got_w, got_v = _native.kopf_eigh(cov)
    w_diff = int(np.count_nonzero(got_w != want_w))
    v_diff = int(np.count_nonzero(got_v != want_v))
    print(f"eigh: {w_diff} eigenvalues and {v_diff} eigenvector entries differ "
          f"(max {np.abs(got_w - want_w).max():.2e}, {np.abs(got_v - want_v).max():.2e})")
    if ACCELERATE:
        assert w_diff == 0 and v_diff == 0
    else:
        np.testing.assert_allclose(got_w, want_w, rtol=0, atol=1e-15)
        np.testing.assert_allclose(got_v, want_v, rtol=0, atol=1e-14)


def test_eigh_reads_the_lower_triangle():
    cov = random_spd(1000, 3)
    cov[:, 0, 1] += 0.01                           # only the upper triangle changes
    want_w, want_v = np.linalg.eigh(cov)
    got_w, got_v = _native.kopf_eigh(cov)
    np.testing.assert_allclose(got_w, want_w, rtol=0, atol=1e-15)
    np.testing.assert_allclose(got_v, want_v, rtol=0, atol=1e-14)


@pytest.mark.parametrize("seed", [0, 1])
def test_clamp_matches_numpy(seed):
    cov = random_spd(10_000, seed)
    want = numpy_clamp_singular_values(cov)
    got = native_clamp_singular_values(cov)
    print(f"clamp: {np.count_nonzero(got != want)} entries differ "
          f"(max {np.abs(got - want).max():.2e})")
    if ACCELERATE:
        assert got.tobytes() == want.tobytes()
    else:
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


@pytest.mark.skipif(not ACCELERATE, reason="the platform pow need not be correctly rounded")
def test_numpy_sigma_growth_powers_are_correctly_rounded():
    # The native table holds the correctly rounded powers (checked in kopf_run.rs).
    from fractions import Fraction

    for n in range(17):
        assert SIGMA_GROWTH ** np.int64(n) == float(Fraction(SIGMA_GROWTH) ** n)


# ------------------------------------------------------------------ whole runs

@pytest.mark.parametrize("preset", ["sprite", "background"])
@pytest.mark.parametrize("name", FIXTURES)
def test_run_matches_reference_32px(preprocessed, name, preset):
    pre, config = preprocessed(name, preset=preset, method="kopf", out_width=32, out_height=32,
                               kopf_max_iters=10)
    for weight in (None, pre.weight):
        got = kopf.run(pre.lab, pre.mask, 32, 32, config, weight=weight)
        want = reference(pre.lab, pre.mask, 32, 32, config, weight)
        assert_matches(got, want, f"{name}/{preset}/32px/{'w' if weight is not None else '-'}")


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["sprite", "background"])
@pytest.mark.parametrize("name", FIXTURES)
def test_run_matches_reference_preset_size(preprocessed, name, preset):
    pre, config = preprocessed(name, preset=preset, method="kopf")
    wo, ho = pre.target_width, pre.target_height
    got = kopf.run(pre.lab, pre.mask, wo, ho, config, weight=pre.weight)
    want = reference(pre.lab, pre.mask, wo, ho, config, pre.weight)
    assert_matches(got, want, f"{name}/{preset}/{wo}x{ho}")


@pytest.mark.parametrize("seed", range(4))
def test_run_matches_reference_random(seed):
    rng = np.random.default_rng(seed)
    hi, wi = rng.integers(20, 70, 2)
    ho, wo = rng.integers(3, 17, 2)
    lab = np.stack([rng.uniform(0, 100, (hi, wi)), rng.uniform(-60, 60, (hi, wi)),
                    rng.uniform(-60, 60, (hi, wi))], axis=-1)
    mask = rng.random((hi, wi)) < 0.8
    mask[: hi // 4, : wi // 3] = False
    weight = np.where(mask, rng.uniform(0.3, 1.0, (hi, wi)), 0.0)
    config = Config(method="kopf", kopf_max_iters=8)
    got = kopf.run(lab, mask, int(wo), int(ho), config, weight=weight)
    want = reference(lab, mask, int(wo), int(ho), config, weight)
    assert_matches(got, want, f"random {seed}")


def test_reference_with_native_eigen_step_is_bitwise(preprocessed):
    # Isolation: the reference with the native eigen step must equal the native run exactly
    # on every platform; any remaining difference would be a logic error, not LAPACK.
    pre, config = preprocessed("noisy_gradient", preset="background", method="kopf",
                               out_width=32, out_height=32, kopf_max_iters=10)
    got = kopf.run(pre.lab, pre.mask, 32, 32, config, weight=pre.weight)
    want = ref_run(pre.lab, pre.mask, 32, 32, config, pre.weight,
                   clamp=native_clamp_singular_values)
    if ACCELERATE:
        assert got.small_lab.tobytes() == want.small_lab.tobytes()
    else:
        # np.hypot is the platform's: only near-ties of the orientation test can differ.
        assert np.abs(got.small_lab - want.small_lab).max() < 1e-6
    assert np.array_equal(got.small_mask, want.small_mask)


# ------------------------------------------------------------------ logging and arguments

def test_progress_lines_match_python(preprocessed, caplog):
    pre, config = preprocessed("noisy_gradient", method="kopf", out_width=16, out_height=16,
                               kopf_max_iters=12, kopf_tol=1e-12)
    with caplog.at_level("DEBUG", logger="pixelforge"):
        kopf.run(pre.lab, pre.mask, 16, 16, config)
        native = [(r.levelno, r.getMessage()) for r in caplog.records
                  if r.name == "pixelforge.kopf"]
        caplog.clear()
        ref_run(pre.lab, pre.mask, 16, 16, config)
        python = [(r.levelno, r.getMessage()) for r in caplog.records
                  if r.name == "pixelforge.downscale.kopf"]
    assert native == python
    assert native[9][0] == logging.INFO and native[0][0] == logging.DEBUG
    assert native[-1] == (logging.DEBUG, "kopf stopped at the cap after 12 iterations")


def test_argument_errors():
    lab, mask, cfg = np.zeros((8, 8, 3)), np.ones((8, 8), bool), Config().to_dict()
    with pytest.raises(ValueError, match="kopf_run: lab must be a float64"):
        _native.kopf_run(lab.astype(np.float32), mask, 2, 2, cfg)
    with pytest.raises(ValueError, match="kopf_run: weight must be a float64"):
        _native.kopf_run(lab, mask, 2, 2, cfg, np.ones((8, 8), np.float32))
    with pytest.raises(ValueError, match="at least 1x1"):
        _native.kopf_run(lab, mask, 2, 0, cfg)
    with pytest.raises(ValueError, match=r"cov must have shape \(N, 2, 2\)"):
        _native.kopf_eigh(np.zeros((3, 2, 3)))
