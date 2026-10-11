"""Native gerstner kernels (`_native.slic_assign`, `_native.bilateral_filter`) against the
numpy reference in `pixelforge/downscale/_reference/gerstner.py`."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pixelforge import _native
from pixelforge.downscale import gerstner
from pixelforge.downscale._reference import gerstner as reference

PRESETS = {"sprite": {}, "background": {"out_width": 64, "out_height": 64}}


def _recorded_assign_calls(preprocessed, name, preset, monkeypatch, iterations=4):
    """Run the reference gerstner for a few iterations and record every `_assign` call with
    its real intermediate state (centers, candidate mask, superpixel colors)."""
    pre, config = preprocessed(name, preset=preset, method="gerstner",
                               g_max_iters=iterations, **PRESETS[preset])
    calls = []
    original = reference._assign

    def record(grid, cx, cy, radius, **kwargs):
        result = original(grid, cx, cy, radius, **kwargs)
        calls.append((grid, cx.copy(), cy.copy(), radius,
                      {k: (v.copy() if isinstance(v, np.ndarray) else v)
                       for k, v in kwargs.items()}, result))
        return result

    monkeypatch.setattr(reference, "_assign", record)
    reference.run(pre.lab, pre.mask, pre.target_width, pre.target_height, config,
                  weight=pre.weight)
    assert calls
    return calls


def _native_assign(grid, cx, cy, radius, kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("channels") is not None:
        kwargs["channels"] = np.stack(kwargs["channels"])
    return gerstner._assign(grid, cx, cy, radius, **kwargs)


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("name", ["circle_alpha", "noisy_gradient"])
def test_slic_assign_is_bitwise_equal_to_reference(preprocessed, monkeypatch, name, preset):
    calls = _recorded_assign_calls(preprocessed, name, preset, monkeypatch)
    windowed = [c for c in calls if c[4].get("windowed", True)]
    assert windowed and all(c[4].get("channels") is not None for c in windowed)
    for grid, cx, cy, radius, kwargs, expected in calls:
        got = _native_assign(grid, cx, cy, radius, kwargs)
        assert got.dtype == np.int64 and got.shape == expected.shape
        assert np.array_equal(got, expected)
    # Spatial-only pass (windowed=False, as for the transparent pixels) on the last centers.
    grid, cx, cy, radius = calls[-1][:4]
    expected = reference._assign(grid, cx, cy, radius, windowed=False)
    assert np.array_equal(gerstner._assign(grid, cx, cy, radius, windowed=False), expected)


def test_slic_assign_perturbed_state_and_ties():
    # Drifted centers, unusable candidates, a coarse integer grid that produces exact ties.
    rng = np.random.default_rng(3)
    for hi, wi, ho, wo in [(97, 83, 13, 11), (40, 40, 8, 8), (30, 50, 30, 10)]:
        grid = gerstner._Grid(hi, wi, ho, wo)
        cx = grid.cx0 + rng.normal(0.0, 0.6 * grid.rx, ho * wo)
        cy = grid.cy0 + rng.normal(0.0, 0.6 * grid.ry, ho * wo)
        ok = rng.random(ho * wo) > 0.15
        lab = np.round(rng.random((hi, wi, 3)) * 4.0) * 10.0     # few levels → ties
        sp = np.round(rng.random((ho * wo, 3)) * 4.0) * 10.0
        channels = [np.ascontiguousarray(lab[..., c]) for c in range(3)]
        radius = grid.search_radius(cx, cy)
        for coef in (0.0, 0.37, 1.0):
            with np.errstate(invalid="ignore"):   # the reference computes inf · 0 for coef 0
                expected = reference._assign(grid, cx, cy, radius, candidate_ok=ok,
                                             channels=channels, sp_color=sp, coef=coef)
            got = gerstner._assign(grid, cx, cy, radius, candidate_ok=ok,
                                   channels=np.stack(channels), sp_color=sp, coef=coef)
            assert np.array_equal(got, expected)
        for r in (0, 1, radius):
            assert np.array_equal(gerstner._assign(grid, cx, cy, r, windowed=False),
                                  reference._assign(grid, cx, cy, r, windowed=False))


def test_slic_assign_returns_minus_one_without_candidates():
    grid = gerstner._Grid(16, 16, 4, 4)
    ok = np.zeros(16, dtype=bool)
    channels = np.zeros((3, 16, 16))
    sp = np.zeros((16, 3))
    got = gerstner._assign(grid, grid.cx0, grid.cy0, 1, candidate_ok=ok, channels=channels,
                           sp_color=sp)
    assert (got == -1).all()


@pytest.mark.parametrize("sigma_spatial", [0.4, 1.0, 1.5])   # radius 1, 2, 3
def test_bilateral_filter_matches_reference(sigma_spatial):
    rng = np.random.default_rng(11)
    h, w = 41, 37
    image = rng.random((h, w, 3)) * np.array([100.0, 160.0, 160.0]) - np.array([0.0, 80.0, 80.0])
    active = rng.random((h, w)) > 0.3
    active[5:9, 5:9] = False   # a block whose interior has inactive neighbors only (r = 1)
    expected = reference.bilateral_filter(image, active, 12.0, sigma_spatial)
    got = gerstner.bilateral_filter(image, active, 12.0, sigma_spatial)
    assert got.shape == image.shape and got.dtype == np.float64
    assert np.abs(got - expected).max() < 1e-9
    as_float = _native.bilateral_filter(image, active.astype(np.float64), 12.0, sigma_spatial)
    assert np.array_equal(as_float, got)
    as_u8 = _native.bilateral_filter(image, active.astype(np.uint8), 12.0, sigma_spatial)
    assert np.array_equal(as_u8, got)


def test_bilateral_filter_keeps_pixels_without_weight():
    rng = np.random.default_rng(5)
    image = rng.random((12, 10, 3)) * 50.0
    active = np.zeros((12, 10), dtype=bool)
    active[0, 0] = True
    got = gerstner.bilateral_filter(image, active, 12.0, 0.4)   # radius 1
    reached = np.zeros((12, 10), dtype=bool)
    reached[:2, :2] = True
    assert np.array_equal(got[~reached], image[~reached])
    assert np.array_equal(got[reached], np.broadcast_to(image[0, 0], (4, 3)))
    nothing = gerstner.bilateral_filter(image, np.zeros((12, 10), dtype=bool), 12.0, 1.0)
    assert np.array_equal(nothing, image)


def _assign_args(hi=8, wi=8, ho=2, wo=2):
    grid = gerstner._Grid(hi, wi, ho, wo)
    return dict(home_y=grid.home_y, home_x=grid.home_x, px=grid.px, py=grid.py,
                center_x=grid.cx0.reshape(ho, wo), center_y=grid.cy0.reshape(ho, wo),
                radius=1, rx=grid.rx, ry=grid.ry, coef=1.0, windowed=True,
                channels=np.zeros((3, hi, wi)), sp_color=np.zeros((ho, wo, 3)))


def test_slic_assign_valid_args_run():
    assert _native.slic_assign(**_assign_args()).shape == (8, 8)


@pytest.mark.parametrize("change, error", [
    (dict(px=np.zeros(7)), ValueError),
    (dict(center_y=np.zeros((2, 3))), ValueError),
    (dict(channels=np.zeros((3, 8, 7))), ValueError),
    (dict(sp_color=np.zeros((2, 2, 4))), ValueError),
    (dict(sp_color=None), ValueError),
    (dict(radius=-1), ValueError),
    (dict(home_y=np.full(8, 2, dtype=np.int64)), ValueError),    # outside 0..ho
    (dict(home_x=np.full(8, -1, dtype=np.int64)), ValueError),
    (dict(center_x=np.zeros((0, 2)), center_y=np.zeros((0, 2))), ValueError),
    (dict(channels=np.zeros((3, 16, 8))[:, ::2]), ValueError),   # not C-contiguous
    (dict(center_x=np.zeros((2, 4))[:, ::2]), ValueError),
    (dict(home_y=np.zeros(8, dtype=np.int32)), TypeError),
    (dict(px=np.zeros(8, dtype=np.float32)), TypeError),
])
def test_slic_assign_rejects_bad_input(change, error):
    with pytest.raises(error):
        _native.slic_assign(**{**_assign_args(), **change})


@pytest.mark.parametrize("image, active, sigmas", [
    (np.zeros((4, 4, 4)), np.ones((4, 4)), (12.0, 1.0)),
    (np.zeros((4, 4, 3)), np.ones((4, 5)), (12.0, 1.0)),
    (np.zeros((4, 8, 3))[:, ::2], np.ones((4, 4)), (12.0, 1.0)),
    (np.zeros((4, 4, 3)), np.ones((4, 8))[:, ::2], (12.0, 1.0)),
    (np.zeros((4, 4, 3)), np.ones((4, 4)), (12.0, 0.0)),
    (np.zeros((4, 4, 3)), np.ones((4, 4)), (0.0, 1.0)),
    (np.zeros((4, 4, 3)), np.ones((4, 4)), (12.0, float("nan"))),
])
def test_bilateral_filter_rejects_bad_input(image, active, sigmas):
    with pytest.raises(ValueError):
        _native.bilateral_filter(image, active, *sigmas)


def test_bilateral_filter_rejects_wrong_dtype():
    with pytest.raises(TypeError):
        _native.bilateral_filter(np.zeros((4, 4, 3), dtype=np.float32), np.ones((4, 4)),
                                 12.0, 1.0)
    with pytest.raises(TypeError):
        _native.bilateral_filter(np.zeros((4, 4, 3)), np.ones((4, 4), dtype=np.int64),
                                 12.0, 1.0)


# --- thread-count invariance ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = """
import hashlib, sys
import numpy as np
from pixelforge import Config, io, preprocess
from pixelforge.downscale import gerstner
config = Config(preset=sys.argv[2], method="gerstner", out_width=48, out_height=48)
loaded = io.load(sys.argv[1])
pre = preprocess.run(loaded.rgb, loaded.alpha, config)
out = gerstner.run(pre.lab, pre.mask, pre.target_width, pre.target_height, config,
                   weight=pre.weight)
h = hashlib.sha256()
for a in (out.indices, out.small_mask, out.palette_lab, out.mean_lab):
    h.update(np.ascontiguousarray(a).tobytes())
print(h.hexdigest(), out.stats["iterations"])
"""


def _digest(path, preset, threads):
    env = dict(os.environ, PIXELFORGE_THREADS=str(threads))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    done = subprocess.run([sys.executable, "-c", _SCRIPT, str(path), preset], env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip()


@pytest.mark.parametrize("name, preset", [("circle_alpha", "sprite"),
                                          ("noisy_gradient", "background")])
def test_gerstner_output_does_not_depend_on_thread_count(fixture_path, name, preset):
    digests = {threads: _digest(fixture_path(name), preset, threads) for threads in (1, 3, 8)}
    assert len(set(digests.values())) == 1, digests


def test_native_kernels_do_not_depend_on_thread_count(monkeypatch):
    rng = np.random.default_rng(9)
    hi, wi, ho, wo = 203, 157, 29, 23   # many bands, a ragged last band
    grid = gerstner._Grid(hi, wi, ho, wo)
    cx = grid.cx0 + rng.normal(0.0, 1.0, ho * wo)
    cy = grid.cy0 + rng.normal(0.0, 1.0, ho * wo)
    channels = rng.random((3, hi, wi)) * 100.0
    sp = rng.random((ho * wo, 3)) * 100.0
    image = rng.random((hi, wi, 3)) * 100.0
    active = rng.random((hi, wi)) > 0.2
    results = []
    for threads in ("1", "2", "3", "8", "0", "junk"):   # 0 / junk fall back to the default
        monkeypatch.setenv("PIXELFORGE_THREADS", threads)
        results.append((gerstner._assign(grid, cx, cy, 3, channels=channels, sp_color=sp,
                                         coef=0.5),
                        gerstner.bilateral_filter(image, active, 12.0, 1.5)))
    for assign, filtered in results[1:]:
        assert np.array_equal(assign, results[0][0])
        assert np.array_equal(filtered, results[0][1])


def test_reference_shares_the_tunable_constants():
    for name in ("PERTURB_DELTA", "SPLIT_GROWTH", "SPLIT_SPREAD", "STALL_EVENTS",
                 "EXHAUSTED_T_FRACTION", "_MAX_SEARCH_RADIUS"):
        assert getattr(gerstner, name) == getattr(reference, name)
