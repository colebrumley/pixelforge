"""The native enhance pass and its helpers (pixelforge._native: percentile, enhance_levels,
subject_brightness_from, enhance) against the unchanged numpy functions in
`pixelforge.preprocess`.

Percentiles and levels are bitwise. `enhance` differs from numpy in the last bits only:
its Gaussian blurs (scipy's FMA contraction and exp's last bit, see test_filters_native.py)
and `libm::expm1`.
"""

from pathlib import Path

import numpy as np
import pytest

from pixelforge import Config, _native, io, preprocess

FIXTURES = ("checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color")
PORTRAIT = Path("/Users/cole/Documents/lib/pixelforge/.claude/worktrees/16-bit-sprite-photo-"
                "5897f3/out/sample/portrait.png")
ROOT = Path(__file__).resolve().parent / "fixtures"


def same_bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def to8(x):
    return np.round(np.asarray(x) * 255.0)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 100, 101, 1000, 4097])
def test_percentile_matches_numpy_bitwise(n):
    rng = np.random.default_rng(n)
    qs = [0.0, 1.0, 99.0, 50.0, 33.3, 100.0, 12.5, 99.99]
    for values in (rng.random(n), np.round(rng.random(n) * 8) / 8, -rng.random(n) * 1e3):
        want = np.percentile(values, qs)
        got = _native.percentile(values, qs)
        assert same_bits(got, want), (n, list(got), list(want))


def test_enhance_levels_matches_python():
    rng = np.random.default_rng(1)
    values = rng.random(500)
    for kw in ({}, {"enhance_black": 0.1}, {"enhance_white": 0.8},
               {"enhance_black": 0.2, "enhance_white": 0.3}):
        config = Config(**kw)
        assert _native.enhance_levels(values, config.to_dict()) == \
            preprocess.enhance_levels(values, config)


def test_stretch_matches_python_bitwise():
    rng = np.random.default_rng(3)
    values = rng.random(300)
    for lo, hi in ((0.1, 0.9), (0.4, 0.5), (0.45, 0.45 + 1e-7), (-0.2, 1.2), (0.0, 1.0),
                   (0.3, 0.5), (0.2, 0.2 + 0.2), (0.6, 0.3)):
        want = preprocess._stretch(values, lo, hi)
        assert same_bits(_native.enhance_stretch(values, lo, hi), want), (lo, hi)


@pytest.mark.parametrize("mode", ["nearest", "wrap"])
def test_gaussian_running_mean_path_is_bitwise(mode):
    """From BLUR_BOX_SIGMA up, `_gaussian` is three uniform filters per axis: no exp, no
    multiply-add, so bitwise; also when the axis is shorter than the widths."""
    rng = np.random.default_rng(int(mode == "wrap"))
    for shape in ((40, 33), (7, 120), (3, 3), (1, 50)):
        a = rng.random(shape)
        for sigma in ((4.0, 4.0), (4.5, 9.3), (16.0, 5.0), (31.7, 64.0)):
            want = preprocess._gaussian(a, sigma, mode)
            got = _native.enhance_gaussian(a, sigma, mode)
            assert same_bits(got, want), (shape, sigma)
        # Below the threshold: the Gaussian kernel (last-bit differences only).
        want = preprocess._gaussian(a, (1.5, 3.9), mode)
        assert float(np.abs(_native.enhance_gaussian(a, (1.5, 3.9), mode) - want).max()) <= 1e-12


def _subject(rgb, alpha, config):
    rgb, mask, weight, _, _ = preprocess._subject(rgb, alpha, config)
    return (np.ascontiguousarray(rgb), np.ascontiguousarray(mask),
            np.ascontiguousarray(weight))


def _cases():
    """(name, rgb, mask, weight) as `run` hands them to `enhance` (before any repeat or
    crop): every fixture with and without background keying, the 4× pre-reduced portrait."""
    out = []
    for name in FIXTURES:
        loaded = io.load(ROOT / f"{name}.png")
        for key_bg in (False, True):
            rgb, mask, weight = _subject(loaded.rgb, loaded.alpha, Config(key_bg=key_bg))
            out.append((f"{name}{'+key' if key_bg else ''}", rgb, mask, weight))
    if PORTRAIT.is_file():
        loaded = io.load(PORTRAIT)
        rgb, mask, weight = _subject(loaded.rgb, loaded.alpha, Config())
        small, small_mask, small_weight = preprocess.prereduce(rgb, mask, 4, weight)
        out.append(("portrait/4", np.ascontiguousarray(small),
                    np.ascontiguousarray(small_mask), np.ascontiguousarray(small_weight)))
    return out


CASES = _cases()


def test_subject_brightness_matches_python():
    for name, rgb, mask, weight in CASES:
        want = preprocess._level_pixels(rgb.max(axis=-1), mask, weight)
        assert same_bits(_native.subject_brightness_from(rgb, mask, weight), want), name
    # No mostly-opaque pixel: every masked pixel counts.
    rgb = np.random.default_rng(0).random((5, 6, 3))
    mask = np.ones((5, 6), dtype=bool)
    weight = np.full((5, 6), 0.2)
    assert same_bits(_native.subject_brightness_from(rgb, mask, weight),
                     rgb.max(axis=-1).ravel())


@pytest.mark.parametrize("enhance", [0.5, 1.0, 4.0])
@pytest.mark.parametrize("ink", [0.0, 0.6])
@pytest.mark.parametrize("radius", [1.0, 2.0, 8.0])
def test_enhance_matches_python(enhance, ink, radius):
    config = Config(enhance=enhance, ink=ink, enhance_radius=radius)
    d = config.to_dict()
    worst, flips, checked = 0.0, 0, 0
    for _name, rgb, mask, weight in CASES:
        h, w = mask.shape
        # About 64 output pixels across: radius 8 then blurs with sigma >= 4 input pixels
        # (the running-mean path) on every case.
        cell = (h / 64, w / 64)
        for wrap in (False, True):
            for wt in (None, weight):
                want = preprocess.enhance(rgb, mask, cell, config, wt, wrap=wrap)
                got = _native.enhance(rgb, mask, cell, d, wt, wrap)
                worst = max(worst, float(np.abs(got - want).max()))
                flips += int((to8(got) != to8(want)).sum())
                checked += 1
    assert checked == 4 * len(CASES)
    # Measured: at most 4.2e-15 (ink on), 1.2e-15 (ink off); no 8-bit flips.
    assert flips == 0
    assert worst <= 1e-9, worst


def test_enhance_cell_sizes_and_levels():
    """Small cells (Gaussian path on both axes), unequal axes, explicit levels, a narrow
    range (the capped gain), and enhance off with ink on."""
    name, rgb, mask, weight = CASES[0]
    for kw, cell in (({"enhance": 1.0}, (1.0, 1.0)),
                     ({"enhance": 1.0}, (0.5, 7.0)),
                     ({"enhance": 0.7, "enhance_black": 0.3, "enhance_white": 0.45}, (2.0, 2.0)),
                     ({"enhance": 2.0, "enhance_black": 0.0, "enhance_white": 1.0}, (3.0, 3.0)),
                     ({"ink": 1.0}, (2.5, 2.5)),
                     ({}, (2.0, 2.0))):
        config = Config(**kw)
        want = preprocess.enhance(rgb, mask, cell, config, weight)
        got = _native.enhance(rgb, mask, cell, config.to_dict(), weight, False)
        assert float(np.abs(got - want).max()) <= 1e-9, kw


def test_enhance_flat_subject_and_transparent_pixels():
    rgb = np.full((12, 10, 3), 0.4)
    rgb[3:5, 3:5] = 0.1
    mask = np.ones((12, 10), dtype=bool)
    mask[:, :2] = False
    config = Config(enhance=1.0, ink=0.5)
    want = preprocess.enhance(rgb, mask, (1.5, 1.5), config)
    got = _native.enhance(rgb, mask, (1.5, 1.5), config.to_dict(), None, False)
    assert float(np.abs(got - want).max()) <= 1e-9
    assert same_bits(got[:, :2], rgb[:, :2])          # transparent pixels untouched


def test_argument_errors():
    rgb = np.zeros((4, 5, 3))
    d = Config(enhance=1.0).to_dict()
    with pytest.raises(ValueError, match="mask must have the image's shape"):
        _native.enhance(rgb, np.ones((5, 4), dtype=bool), (1.0, 1.0), d, None, False)
    with pytest.raises(ValueError, match="weight must be a float64"):
        _native.enhance(rgb, np.ones((4, 5), dtype=bool), (1.0, 1.0), d,
                        np.ones((4, 5), dtype=np.float32), False)
    with pytest.raises(ValueError, match=r"values must have shape \(N,\)"):
        _native.percentile(np.zeros((2, 2)), [50.0])
