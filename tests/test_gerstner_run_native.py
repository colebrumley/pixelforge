"""The native gerstner stage (`_native.gerstner_run`, `_native.PaletteAnnealer`,
`_native.mcda`) against the Python control flow it replaced (tests/gerstner_reference.py,
over the same native loops).

gerstner is a float stage (docs/rust-migration-plan.md, Decision 3): the control flow is
ported operation for operation, but three numpy routines are replaced and may differ in the
last bits: the BLAS products `centered.T @ centered` (principal_axis) and
`weighted.T @ points` (REFINE), now sequential sums, and LAPACK's `eigh`, now a Jacobi
solver; `exp`/`log` are libm's. A last-bit change in the principal axis or the variance moves
T0 and the perturbation directions, so the iteration path can diverge on near-ties. Every
case records whether it is identical, the differing indices and the palette's max |Δ|
(`-rP` prints them); the assertions bound the differences.
"""

import numpy as np
import pytest

from pixelforge import Config, _native, palette
from pixelforge.downscale import gerstner
from tests import gerstner_reference as reference

FIXTURES = ["checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color"]


def compare(native, ref) -> dict:
    """Per-field comparison of two Downscaled results."""
    same_shape = native.palette_lab.shape == ref.palette_lab.shape
    report = {
        "identical": (same_shape
                      and np.array_equal(native.indices, ref.indices)
                      and np.array_equal(native.palette_lab, ref.palette_lab)
                      and np.array_equal(native.palette_lab_unsaturated,
                                         ref.palette_lab_unsaturated)
                      and np.array_equal(native.small_mask, ref.small_mask)
                      and np.array_equal(native.mean_lab, ref.mean_lab)
                      and np.array_equal(native.small_lab, ref.small_lab)
                      and native.stats == ref.stats),
        "indices_differ": int((native.indices != ref.indices).sum()),
        "mask_differ": int((native.small_mask != ref.small_mask).sum()),
        "palette_max_abs": (float(np.abs(native.palette_lab - ref.palette_lab).max())
                            if same_shape else float("inf")),
        "mean_max_abs": float(np.abs(native.mean_lab - ref.mean_lab).max()),
        "stats_native": native.stats,
        "stats_ref": ref.stats,
    }
    return report


def run_both(preprocessed, name, **overrides):
    pre, config = preprocessed(name, method="gerstner", **overrides)
    args = (pre.lab, pre.mask, pre.target_width, pre.target_height, config)
    native = gerstner.run(*args, weight=pre.weight)
    ref = reference.run(*args, weight=pre.weight)
    assert native.indices.shape == ref.indices.shape
    assert native.indices.dtype == ref.indices.dtype == np.int64
    assert native.small_mask.dtype == bool and native.small_lab.shape == ref.small_lab.shape
    assert list(native.stats) == list(ref.stats)
    return native, ref


def check(report, record_property, n_pixels):
    record_property("parity", {k: v for k, v in report.items()})
    print(report)
    if report["identical"]:
        return
    # A diverged iteration path: same palette size, few moved indices, nearby colors.
    assert report["stats_native"]["palette_size"] == report["stats_ref"]["palette_size"]
    assert report["indices_differ"] <= max(4, n_pixels // 20), report
    assert report["palette_max_abs"] < 5.0, report


@pytest.mark.parametrize("k", [4, 8, 16])
@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("name", FIXTURES)
def test_run_matches_reference(preprocessed, record_property, name, size, k):
    native, ref = run_both(preprocessed, name, out_width=size, out_height=size,
                           palette_size=k)
    check(compare(native, ref), record_property, size * size)


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["sprite", "background"])
@pytest.mark.parametrize("name", FIXTURES)
def test_run_matches_reference_at_preset_size(preprocessed, record_property, name, preset):
    native, ref = run_both(preprocessed, name, preset=preset)
    check(compare(native, ref), record_property, native.indices.size)


def _libm(a, log):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return _native.libm_exp_log(a.ravel(), log).reshape(a.shape)


def _step_with_native_routines(self, points, weights):
    """reference.PaletteAnnealer.step with the three replaced routines swapped for what the
    native annealer computes: libm's exp/log and sequential sums for `weighted.T @ points`.
    Everything else is the reference's code."""
    clusters = self.clusters
    d2 = np.zeros((len(points), len(clusters)))
    for c in range(3):
        diff = points[:, c, None] - clusters[None, :, c]
        d2 += diff * diff
    logp = _libm(self.prob, True)[None, :] - np.sqrt(d2) / self.T
    logp -= logp.max(axis=1, keepdims=True)
    sub_assign = np.argmax(logp, axis=1)
    post = _libm(logp, False)
    post /= post.sum(axis=1, keepdims=True)
    weighted = post * weights[:, None]
    prob = weighted.sum(axis=0)
    refined = clusters.copy()
    alive = prob > 0
    products = (weighted[:, :, None] * points[:, None, :]).sum(axis=0)   # sequential
    refined[alive] = products[alive] / prob[alive, None]
    change = float(np.sqrt(((refined - clusters) ** 2).sum(axis=1)).sum())
    self.clusters, self.prob = refined, prob
    self.assign = sub_assign // 2 if self.has_sub else sub_assign
    if change < self.config.g_eps_palette:
        self._cool_and_expand(points, sub_assign)


@pytest.mark.parametrize("k", [4, 16])
@pytest.mark.parametrize("name", FIXTURES)
def test_differences_come_only_from_the_replaced_routines(preprocessed, monkeypatch, name, k):
    # With principal_axis (BLAS product + LAPACK eigh), the REFINE product and exp/log
    # swapped for the native ones, the reference's control flow gives the native result bit
    # for bit: every other difference would be a logic error.
    monkeypatch.setattr(reference, "principal_axis", gerstner.principal_axis)
    monkeypatch.setattr(reference.PaletteAnnealer, "step", _step_with_native_routines)
    native, ref = run_both(preprocessed, name, out_width=32, out_height=32, palette_size=k)
    assert compare(native, ref)["identical"]


@pytest.mark.parametrize("seed", [None, 3])
def test_weighted_input_with_swapped_routines_is_bitwise(monkeypatch, seed):
    # Semi-transparent weights and a seed: the same, except the seeded generator, which
    # differs by design (SplitMix64, not numpy's PCG64), so it is swapped too.
    monkeypatch.setattr(reference, "principal_axis", gerstner.principal_axis)
    monkeypatch.setattr(reference.PaletteAnnealer, "step", _step_with_native_routines)
    rng = np.random.default_rng(4)
    lab = np.zeros((40, 40, 3))
    lab[..., 0] = 30.0 + 40.0 * (np.arange(40) >= 20)[None, :]
    lab[..., 1] = rng.normal(0.0, 3.0, (40, 40))
    lab[:20, :, 2] = 25.0
    mask = np.ones((40, 40), dtype=bool)
    mask[:4] = False
    weight = np.where(mask, 1.0, 0.0)
    weight[4:10] = 0.4
    config = Config(method="gerstner", palette_size=6, seed=seed)
    if seed is not None:
        draws = iter(_splitmix_uniforms(seed, 1000))

        class Generator:
            def uniform(self, low, high):
                assert (low, high) == (0.9, 1.1)
                return next(draws)

        monkeypatch.setattr(reference.np.random, "default_rng", lambda s: Generator())
    native = gerstner.run(lab, mask, 10, 10, config, weight=weight)
    ref = reference.run(lab, mask, 10, 10, config, weight=weight)
    assert compare(native, ref)["identical"]


def _splitmix_uniforms(seed, n):
    """The native annealer's perturbation draws: SplitMix64, top 53 bits, 0.9 + 0.2·u."""
    mask = (1 << 64) - 1
    state = seed
    out = []
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        out.append(0.9 + (1.1 - 0.9) * ((z >> 11) * (1.0 / (1 << 53))))
    return out


def test_seed_is_deterministic_and_used(preprocessed):
    pre, config = preprocessed("gradient6", method="gerstner", out_width=16, out_height=16,
                               palette_size=6, seed=7)
    args = (pre.lab, pre.mask, 16, 16)
    a = gerstner.run(*args, config)
    b = gerstner.run(*args, config)
    assert np.array_equal(a.palette_lab, b.palette_lab) and np.array_equal(a.indices, b.indices)
    unseeded = gerstner.run(*args, config.replace(seed=None))
    other = gerstner.run(*args, config.replace(seed=8))
    # The jitter moves the palette (seed 7 vs unseeded, 7 vs 8).
    assert not np.array_equal(a.palette_lab, unseeded.palette_lab)
    assert not np.array_equal(a.palette_lab, other.palette_lab)


# --- one annealer step on identical inputs ----------------------------------------------------

def _annealer_inputs(preprocessed, name, iterations):
    """Points, weights and annealer state of a real reference run after `iterations` steps."""
    pre, config = preprocessed(name, method="gerstner", out_width=32, out_height=32,
                               palette_size=8)
    captured = []
    step = reference.PaletteAnnealer.step

    def record(self, points, weights):
        if len(captured) < iterations:
            captured.append((points.copy(), weights.copy(), self.T, self.clusters.copy(),
                             self.prob.copy(), self.placed.copy(), self.history.copy(),
                             self.has_sub))
        step(self, points, weights)

    reference.PaletteAnnealer.step = record
    try:
        reference.run(pre.lab, pre.mask, 32, 32, config, weight=pre.weight)
    finally:
        reference.PaletteAnnealer.step = step
    return captured, config


def _step_both(state, config):
    points, weights, t, clusters, prob, placed, history, has_sub = state
    ref = reference.PaletteAnnealer(points, config.palette_size, config)
    nat = _native.PaletteAnnealer(points, config.palette_size, config.to_dict())
    for a in (ref, nat):
        a.T, a.clusters, a.prob = t, clusters.copy(), prob.copy()
        a.placed, a.history, a.has_sub = placed.copy(), history.copy(), has_sub
    ref.step(points, weights)
    nat.step(points, weights)
    return ref, nat


@pytest.mark.parametrize("name", ["noisy_gradient", "checker_tiles", "circle_alpha"])
def test_one_step_matches_reference(preprocessed, record_property, name):
    states, config = _annealer_inputs(preprocessed, name, 40)
    worst = {"clusters": 0.0, "prob": 0.0, "assign": 0, "T": 0.0}
    for state in states:
        ref, nat = _step_both(state, config)
        assert nat.has_sub == ref.has_sub and nat.done == ref.done
        assert nat.clusters.shape == ref.clusters.shape
        worst["clusters"] = max(worst["clusters"],
                                float(np.abs(nat.clusters - ref.clusters).max()))
        worst["prob"] = max(worst["prob"], float(np.abs(nat.prob - ref.prob).max()))
        worst["assign"] = max(worst["assign"], int((nat.assign != ref.assign).sum()))
        worst["T"] = max(worst["T"], abs(nat.T - ref.T))
    record_property("step_max_deviation", worst)
    print(name, len(states), "steps", worst)
    assert worst["clusters"] < 1e-9 and worst["prob"] < 1e-12 and worst["assign"] == 0


def test_principal_axis_matches_numpy():
    rng = np.random.default_rng(5)
    worst_axis = worst_var = 0.0
    for n in (2, 3, 10, 100, 1000):
        for _ in range(20):
            pts = rng.normal(size=(n, 3)) * np.array([30.0, 10.0, 5.0]) + 50.0
            axis, var = gerstner.principal_axis(pts)
            ref_axis, ref_var = reference.principal_axis(pts)
            worst_axis = max(worst_axis, float(np.abs(axis - ref_axis).max()))
            worst_var = max(worst_var, abs(var - ref_var) / ref_var)
    print("principal_axis max |Δaxis|", worst_axis, "max rel Δvariance", worst_var)
    assert worst_axis < 1e-9 and worst_var < 1e-12
    for degenerate in (np.zeros((0, 3)), np.ones((1, 3)), np.tile([1.0, 2.0, 3.0], (5, 1))):
        axis, var = gerstner.principal_axis(degenerate)
        assert np.array_equal(axis, [1.0, 0.0, 0.0]) and var == 0.0


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_mcda_matches_reference(k):
    rng = np.random.default_rng(k)
    centers = rng.random((6, 3)) * np.array([100.0, 120.0, 120.0]) - np.array([0, 60, 60])
    pixels = centers[rng.integers(0, 6, 600)] + rng.normal(0.0, 2.0, (600, 3))
    config = Config()
    got = palette.mcda(pixels, k, config)
    want = reference.mcda(pixels, k, config)
    assert got.shape == want.shape
    print("mcda K", k, "max |Δ|", float(np.abs(got - want).max()))
    assert np.abs(got - want).max() < 1e-6


def test_mcda_rejects_empty_and_bad_k():
    with pytest.raises(ValueError, match="at least one pixel"):
        palette.mcda(np.zeros((0, 3)), 4, Config())
    with pytest.raises(ValueError, match="K >= 1"):
        _native.mcda(np.zeros((3, 3)), 0, Config().to_dict())


def test_logging_progress_lines(preprocessed, caplog):
    pre, config = preprocessed("gradient6", method="gerstner", out_width=16, out_height=16,
                               palette_size=6)
    with caplog.at_level("DEBUG", logger="pixelforge"):
        out = gerstner.run(pre.lab, pre.mask, 16, 16, config)
    lines = [r for r in caplog.records if r.name == "pixelforge.gerstner"]
    n = out.stats["iterations"]
    assert [r.getMessage().split(" T=")[0] for r in lines[:n]] == [
        f"gerstner iter {i}/{config.g_max_iters}" for i in range(1, n + 1)]
    assert lines[-1].getMessage() == f"gerstner converged after {n} iterations"
    levels = [r.levelname for r in lines[:n]]
    assert all(lv == ("INFO" if (i + 1) % 10 == 0 else "DEBUG") for i, lv in enumerate(levels))
    t = float(lines[n - 1].getMessage().split("T=")[1].split()[0])
    assert f"{out.stats['final_temperature']:.3g}" == lines[n - 1].getMessage().split("T=")[1].split()[0]
    assert t > 0
