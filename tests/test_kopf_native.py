"""The native kopf kernels (pixelforge._native.kopf_*) against the numpy reference.

The reference quantities are computed with the loops of pixelforge/downscale/_reference/kopf.py
(its `_Layout` expand/reduce), lifted out of `run` so that each native function can be fed
exactly the same state: the iteration-1 state, and a seeded perturbation of it with off-grid
μ, anisotropic Σ, varied ν and σ, and a non-trivial pixel weight.
"""

import math

import numpy as np
import pytest

from pixelforge import _native
from pixelforge.downscale import kopf
from pixelforge.downscale._reference import kopf as ref

RTOL = 1e-12
SUM_NAMES = ("w", "x", "y", "xx", "xy", "yy", "c0", "c1", "c2")


# ------------------------------------------------------------------ reference computations

def ref_gamma(layout, channels, pixel_mask, pixel_weight, mu_x, mu_y, inv_a, inv_b, inv_c,
              nu=None, inv_color=None, ghost=None):
    """γ (16, hi, wi) exactly as the reference E-step (color term iff nu is given)."""
    hi, wi = channels[0].shape
    pos_x, pos_y = layout.x.pos[None, :], layout.y.pos[:, None]
    gamma = np.empty((len(layout.slots), hi, wi))
    for s, (jy, jx) in enumerate(layout.slots):
        dx = pos_x - layout.expand(mu_x, jy, jx)
        dy = pos_y - layout.expand(mu_y, jy, jx)
        quad = layout.expand(inv_a, jy, jx) * dx * dx
        quad += layout.expand(inv_b, jy, jx) * 2.0 * dx * dy
        quad += layout.expand(inv_c, jy, jx) * dy * dy
        logw = gamma[s]
        np.multiply(quad, -0.5, out=logw)
        if nu is not None:
            dist = sum((channels[c] - layout.expand(nu[..., c], jy, jx)) ** 2 for c in range(3))
            logw -= dist * layout.expand(inv_color, jy, jx)
        logw[~layout.slot_valid(jy, jx, pixel_mask)] = -np.inf
        if ghost is not None and ghost.any():
            logw[layout.expand(ghost, jy, jx)] = -np.inf
    peak = gamma.max(axis=0)
    peak[~np.isfinite(peak)] = 0.0
    gamma -= peak
    np.exp(gamma, out=gamma)
    total = gamma.sum(axis=0)
    total[total == 0] = 1.0
    gamma /= total
    if pixel_weight is not None:
        gamma *= pixel_weight
    return gamma


def ref_em_sums(layout, gamma, channels, mu_x, mu_y):
    ho, wo = mu_x.shape
    pos_x, pos_y = layout.x.pos[None, :], layout.y.pos[:, None]
    sums = np.zeros((ho, wo, 9))
    for s, (jy, jx) in enumerate(layout.slots):
        g = gamma[s]
        dx = pos_x - layout.expand(mu_x, jy, jx)
        dy = pos_y - layout.expand(mu_y, jy, jx)
        gx, gy = g * dx, g * dy
        terms = (g, gx, gy, gx * dx, gx * dy, gy * dy, g * channels[0], g * channels[1],
                 g * channels[2])
        for q, t in enumerate(terms):
            out = np.zeros((ho, wo))
            layout.reduce(t, jy, jx, out)
            sums[..., q] += out
    return sums


def ref_c_step(layout, gamma, old_mu_x, old_mu_y, move_x, move_y):
    ho, wo = old_mu_x.shape
    ns = layout.ns
    pos_x, pos_y = layout.x.pos[None, :], layout.y.pos[:, None]
    r = math.sqrt(0.5)
    axes = [(1.0, 0.0), (0.0, 1.0), (r, r), (r, -r)]
    plus, both, overlap = (np.zeros((4, ho, wo)) for _ in range(3))
    for s, (jy, jx) in enumerate(layout.slots):
        dx = (pos_x - layout.expand(old_mu_x, jy, jx)) - layout.expand(move_x, jy, jx)
        dy = (pos_y - layout.expand(old_mu_y, jy, jx)) - layout.expand(move_y, jy, jx)
        for a, (ex, ey) in enumerate(axes):
            u = dx * ex + dy * ey
            t = gamma[s] * u * u
            layout.reduce(t, jy, jx, both[a])
            layout.reduce(np.where(u > 0, t, 0.0), jy, jx, plus[a])
        for d, (ndy, ndx) in enumerate(((0, 1), (1, 0), (1, 1), (1, -1))):
            ny, nx = jy + ndy, jx + ndx
            if 0 <= ny < ns and 0 <= nx < ns:
                layout.reduce(gamma[s] * gamma[ny * ns + nx], jy, jx, overlap[d])
    return plus, both, overlap


def ref_refill(layout, channels, pixel_mass, mu_x, mu_y, inv_a, inv_b, inv_c):
    ho, wo = mu_x.shape
    gamma = ref_gamma(layout, channels, None, None, mu_x, mu_y, inv_a, inv_b, inv_c)
    mass = np.zeros((ho, wo))
    color = np.zeros((ho, wo, 3))
    for s, (jy, jx) in enumerate(layout.slots):
        g = gamma[s] * pixel_mass
        layout.reduce(g, jy, jx, mass)
        for c in range(3):
            out = np.zeros((ho, wo))
            layout.reduce(g * channels[c], jy, jx, out)
            color[..., c] += out
    return mass, color


# ------------------------------------------------------------------ states

def make_state(pre, perturbed: bool):
    """Inputs of one kopf iteration on a preprocessed fixture, as kopf.run prepares them."""
    mask = pre.mask
    hi, wi = mask.shape
    wo, ho = pre.target_width, pre.target_height
    cube = kopf.to_unit_cube(pre.lab)
    grid_x, grid_y = np.meshgrid(np.arange(wo) + 0.5, np.arange(ho) + 0.5)
    rng = np.random.default_rng(7)
    mu_x, mu_y = grid_x.copy(), grid_y.copy()
    cov = np.zeros((ho, wo, 2, 2))
    cov[..., 0, 0] = cov[..., 1, 1] = kopf.COV_INIT
    nu = np.full((ho, wo, 3), 0.5)
    sigma = np.full((ho, wo), kopf.SIGMA_INIT)
    pixel_weight = None
    if perturbed:
        mu_x = mu_x + rng.uniform(-0.25, 0.25, (ho, wo))
        mu_y = mu_y + rng.uniform(-0.25, 0.25, (ho, wo))
        cov[..., 0, 0] = rng.uniform(0.05, 0.1, (ho, wo))
        cov[..., 1, 1] = rng.uniform(0.05, 0.1, (ho, wo))
        cov[..., 0, 1] = cov[..., 1, 0] = rng.uniform(-0.02, 0.02, (ho, wo))
        nu = nu + rng.uniform(-0.3, 0.3, (ho, wo, 3))
        sigma = rng.uniform(0.01, 1.0, (ho, wo))
        pixel_weight = np.where(mask, rng.uniform(0.2, 1.0, (hi, wi)), 0.0)
    cell = (kopf.cell_index(hi, ho)[:, None] * wo + kopf.cell_index(wi, wo)[None, :]).ravel()
    ghost = (np.bincount(cell[mask.ravel()], minlength=ho * wo) * 2
             < np.bincount(cell, minlength=ho * wo)).reshape(ho, wo)
    inv_a, inv_b, inv_c = kopf._inverse_cov(cov)
    # The C-step's μ move: the constrained μ relative to the old one, seeded.
    move_x = rng.uniform(-0.05, 0.05, (ho, wo))
    move_y = rng.uniform(-0.05, 0.05, (ho, wo))
    return dict(
        layout=ref._Layout(hi, wi, ho, wo),
        channels=np.ascontiguousarray(np.moveaxis(cube, -1, 0)),
        mask=mask, mask_u8=mask.astype(np.uint8),
        pixel_weight=pixel_weight, mu_x=mu_x, mu_y=mu_y, inv_a=inv_a, inv_b=inv_b, inv_c=inv_c,
        nu=nu, inv_color=1.0 / (2.0 * sigma * sigma), ghost=ghost,
        move_x=move_x, move_y=move_y)


def native_em(st):
    _, hi, wi = st["channels"].shape
    ho, wo = st["mu_x"].shape
    gamma = np.empty((hi, wi, 16))
    sums = np.empty((ho, wo, 9))
    _native.kopf_em_step(st["channels"], st["mask_u8"], st["pixel_weight"], st["mu_x"],
                         st["mu_y"], st["inv_a"], st["inv_b"], st["inv_c"], st["nu"],
                         st["inv_color"], st["ghost"].astype(np.uint8), gamma, sums)
    return gamma, sums


def native_c(st, gamma):
    ho, wo = st["mu_x"].shape
    plus, both, overlap = (np.empty((4, ho, wo)) for _ in range(3))
    _native.kopf_c_step(gamma, st["mu_x"], st["mu_y"], st["move_x"], st["move_y"],
                        plus, both, overlap)
    return plus, both, overlap


def native_refill(st):
    ho, wo = st["mu_x"].shape
    mass, color = np.empty((ho, wo)), np.empty((ho, wo, 3))
    _native.kopf_spatial_refill(st["channels"], pixel_mass(st), st["mu_x"], st["mu_y"],
                                st["inv_a"], st["inv_b"], st["inv_c"], mass, color)
    return mass, color


def pixel_mass(st):
    w = st["pixel_weight"]
    return st["mask"].astype(np.float64) if w is None else w


def assert_close(name, got, want):
    scale = np.abs(want).max()
    err = np.abs(got - want).max()
    assert err <= RTOL * max(scale, 1e-300), f"{name}: max|Δ| {err:.3e}, max|ref| {scale:.3e}"


CASES = [(name, preset, perturbed)
         for name in ("circle_alpha", "noisy_gradient")
         for preset in ("sprite", "background")
         for perturbed in (False, True)]


@pytest.fixture(scope="module")
def states(fixture_path):
    from pixelforge import Config, io, preprocess

    cache = {}

    def get(name, preset, perturbed):
        key = (name, preset, perturbed)
        if key not in cache:
            config = Config(preset=preset, method="kopf")
            loaded = io.load(fixture_path(name))
            pre = preprocess.run(loaded.rgb, loaded.alpha, config)
            cache[key] = make_state(pre, perturbed)
        return cache[key]

    return get


@pytest.mark.parametrize("name, preset, perturbed", CASES)
def test_em_step_matches_reference(states, name, preset, perturbed):
    st = states(name, preset, perturbed)
    layout = st["layout"]
    want_gamma = ref_gamma(layout, st["channels"], None if st["mask"].all() else st["mask"],
                           st["pixel_weight"], st["mu_x"], st["mu_y"], st["inv_a"],
                           st["inv_b"], st["inv_c"], st["nu"], st["inv_color"], st["ghost"])
    want_sums = ref_em_sums(layout, want_gamma, st["channels"], st["mu_x"], st["mu_y"])
    gamma, sums = native_em(st)
    assert np.abs(np.moveaxis(gamma, -1, 0) - want_gamma).max() < RTOL
    for q, qname in enumerate(SUM_NAMES):
        assert_close(qname, sums[..., q], want_sums[..., q])
    assert np.array_equal(sums[..., 0] > 1e-12, want_sums[..., 0] > 1e-12)   # fed kernels
    assert (sums[..., 0] > 1e-12).any()


@pytest.mark.parametrize("name, preset, perturbed", CASES)
def test_c_step_matches_reference(states, name, preset, perturbed):
    st = states(name, preset, perturbed)
    gamma, _ = native_em(st)
    want = ref_c_step(st["layout"], np.moveaxis(gamma, -1, 0), st["mu_x"], st["mu_y"],
                      st["move_x"], st["move_y"])
    got = native_c(st, gamma)
    for qname, g, w in zip(("plus", "both", "overlap"), got, want):
        for a in range(4):
            assert_close(f"{qname}[{a}]", g[a], w[a])
        assert np.abs(w).max() > 0


@pytest.mark.parametrize("name, preset, perturbed", CASES)
def test_spatial_refill_matches_reference(states, name, preset, perturbed):
    st = states(name, preset, perturbed)
    want_mass, want_color = ref_refill(st["layout"], st["channels"], pixel_mass(st), st["mu_x"],
                                       st["mu_y"], st["inv_a"], st["inv_b"], st["inv_c"])
    mass, color = native_refill(st)
    assert_close("mass", mass, want_mass)
    for c in range(3):
        assert_close(f"color[{c}]", color[..., c], want_color[..., c])
    assert np.array_equal(mass > 0, want_mass > 0)


# ------------------------------------------------------------------ thread-count invariance

@pytest.mark.parametrize("name, preset", [("circle_alpha", "sprite"),
                                          ("noisy_gradient", "background")])
def test_kernels_identical_for_any_thread_count(states, monkeypatch, name, preset):
    st = states(name, preset, True)
    assert st["channels"].shape[1] > 2 * 32    # several row bands
    results = []
    for threads in (1, 2, 3, 8):
        monkeypatch.setenv("PIXELFORGE_THREADS", str(threads))
        gamma, sums = native_em(st)
        results.append((gamma, sums, *native_c(st, gamma), *native_refill(st)))
    for other in results[1:]:
        for a, b in zip(results[0], other):
            assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("name", ["circle_alpha", "noisy_gradient"])
def test_run_identical_for_any_thread_count(preprocessed, monkeypatch, name):
    pre, config = preprocessed(name, method="kopf", out_width=32, out_height=32,
                               kopf_max_iters=6)
    outs = []
    for threads in (1, 2, 3, 8):
        monkeypatch.setenv("PIXELFORGE_THREADS", str(threads))
        outs.append(kopf.run(pre.lab, pre.mask, 32, 32, config, weight=pre.weight))
    for out in outs[1:]:
        assert out.small_lab.tobytes() == outs[0].small_lab.tobytes()
        assert np.array_equal(out.small_mask, outs[0].small_mask)
        assert out.stats == outs[0].stats


@pytest.mark.parametrize("value", ["0", "-2", "many", ""])
def test_invalid_thread_setting_falls_back(states, monkeypatch, value):
    st = states("circle_alpha", "sprite", False)
    monkeypatch.setenv("PIXELFORGE_THREADS", "1")
    want = native_em(st)[1]
    monkeypatch.setenv("PIXELFORGE_THREADS", value)
    assert native_em(st)[1].tobytes() == want.tobytes()


# ------------------------------------------------------------------ input validation

def em_args(hi=40, wi=36, ho=10, wo=9):
    z = np.zeros
    return dict(channels=z((3, hi, wi)), mask=np.ones((hi, wi), np.uint8), pixel_weight=None,
                mu_x=z((ho, wo)), mu_y=z((ho, wo)), inv_a=np.ones((ho, wo)), inv_b=z((ho, wo)),
                inv_c=np.ones((ho, wo)), nu=z((ho, wo, 3)), inv_color=np.ones((ho, wo)),
                ghost=z((ho, wo), np.uint8), gamma_out=z((hi, wi, 16)), sums_out=z((ho, wo, 9)))


def call_em(args):
    return _native.kopf_em_step(*args.values())


def test_em_step_accepts_valid_arguments():
    args = em_args()
    call_em(args)
    # Every pixel is covered by at least one kernel, so γ sums to 1 per pixel.
    assert np.allclose(args["gamma_out"].sum(axis=-1), 1.0)
    assert np.isclose(args["sums_out"][..., 0].sum(), 40 * 36)


@pytest.mark.parametrize("field, value, match", [
    ("channels", np.zeros((3, 40, 36), np.float32), "channels must be a float64"),
    ("channels", np.zeros((4, 40, 36)), r"channels must have shape \(3, hi, wi\)"),
    ("channels", [[0.0]], "channels must be a float64 numpy.ndarray"),
    ("mask", np.ones((40, 36), bool), "mask must be a uint8"),
    ("mask", np.ones((40, 35), np.uint8), r"mask must have shape \[40, 36\]"),
    ("pixel_weight", np.ones((40, 36), np.float32), "pixel_weight must be a float64"),
    ("mu_y", np.zeros((9, 10)).T, "mu_y must be C-contiguous"),
    ("nu", np.zeros((10, 9)), r"nu must have shape \[10, 9, 3\]"),
    ("ghost", np.zeros((10, 9)), "ghost must be a uint8"),
    ("sums_out", np.zeros((10, 9, 8)), r"sums_out must have shape \[10, 9, 9\]"),
    ("gamma_out", np.zeros((40, 36, 16))[:, ::-1], "gamma_out must be C-contiguous"),
    ("gamma_out", np.zeros((16, 40, 36)), r"gamma_out must have shape \[40, 36, 16\]"),
])
def test_em_step_rejects_bad_arguments(field, value, match):
    args = em_args()
    args[field] = value
    with pytest.raises(ValueError, match=match):
        call_em(args)


def test_rejects_read_only_and_aliased_outputs():
    args = em_args()
    args["sums_out"].flags.writeable = False
    with pytest.raises(ValueError, match="cannot write sums_out"):
        call_em(args)
    ho, wo = 10, 9
    a = np.zeros((4, ho, wo))
    with pytest.raises(ValueError, match="cannot write both_out"):
        _native.kopf_c_step(np.zeros((40, 36, 16)), *(np.zeros((ho, wo)) for _ in range(4)),
                            a, a, np.zeros((4, ho, wo)))


def test_c_step_and_refill_reject_bad_shapes():
    ho, wo = 10, 9
    k = [np.zeros((ho, wo)) for _ in range(4)]
    with pytest.raises(ValueError, match=r"gamma must have shape \(hi, wi, 16\)"):
        _native.kopf_c_step(np.zeros((40, 36, 9)), *k, *(np.zeros((4, ho, wo)) for _ in range(3)))
    with pytest.raises(ValueError, match=r"overlap_out must have shape \[4, 10, 9\]"):
        _native.kopf_c_step(np.zeros((40, 36, 16)), *k, np.zeros((4, ho, wo)),
                            np.zeros((4, ho, wo)), np.zeros((3, ho, wo)))
    with pytest.raises(ValueError, match="pixel_mass must be a float64"):
        _native.kopf_spatial_refill(np.zeros((3, 40, 36)), np.ones((40, 36), bool),
                                    *(np.ones((ho, wo)) for _ in range(5)), np.zeros((ho, wo)),
                                    np.zeros((ho, wo, 3)))
    with pytest.raises(ValueError, match=r"color_out must have shape \[10, 9, 3\]"):
        _native.kopf_spatial_refill(np.zeros((3, 40, 36)), np.ones((40, 36)),
                                    *(np.ones((ho, wo)) for _ in range(5)), np.zeros((ho, wo)),
                                    np.zeros((ho, wo)))
