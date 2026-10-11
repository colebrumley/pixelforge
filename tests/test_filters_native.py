"""The native filters (pixelforge._native: uniform_filter1d, gaussian_filter1d, median_filter3,
denoise) against scipy.ndimage, scikit-image and the former `preprocess.denoise`
(tests/preprocess_reference.py).

Uniform and median filters are bitwise. The Gaussian filter and the bilateral differ in the
last bits: scipy's and scikit-image's macOS arm64 wheels contract `acc += a * b` into a fused
multiply-add in parts of their loops (the Rust port does not: no FMA, plan Decision 6), and the
kernels/lookup tables use `libm::exp` where numpy uses the platform `exp`. In the bilateral,
skimage's fused `dist += t * t` can also move a colour distance across a lookup-table bin
boundary, which changes that neighbour's weight by one bin step: rare (4 pixels of the
1024×1024 portrait) and at most a few 1e-6 (see test_bilateral_portrait).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from skimage.restoration import denoise_bilateral

from pixelforge import Config, _native, io
from tests import preprocess_reference as pyref

FIXTURES = ("checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color")
SHAPES = [(1, 1), (1, 2), (2, 1), (3, 3), (1, 9), (9, 1), (4, 7), (17, 23), (40, 5), (33, 64)]


def same_bits(a, b):
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def to8(x):
    return np.round(np.asarray(x) * 255.0)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", ["nearest", "wrap"])
def test_uniform_filter1d_matches_scipy_bitwise(shape, mode):
    rng = np.random.default_rng([shape[0], shape[1], int(mode == "wrap")])
    a = rng.random(shape) * 4 - 1
    for axis in (0, 1):
        # Sizes beyond the line length included: the extension then wraps or repeats more
        # than once.
        for size in (1, 2, 3, 4, 5, 9, 10, 15, 31, 80):
            want = ndimage.uniform_filter1d(a, size, axis=axis, mode=mode)
            got = _native.uniform_filter1d(a, size, axis, mode)
            assert same_bits(got, want), (shape, mode, axis, size)


@pytest.mark.parametrize("mode", ["nearest", "wrap"])
def test_gaussian_filter1d_matches_scipy(mode):
    rng = np.random.default_rng(11)
    worst = 0.0
    for shape in SHAPES:
        a = rng.random(shape)
        for axis in (0, 1):
            for sigma in (0.1, 0.5, 0.8, 1.5, 1.6, 2.0, 2.7, 3.2, 3.9, 3.99):
                want = ndimage.gaussian_filter1d(a, sigma, axis=axis, mode=mode)
                got = _native.gaussian_filter1d(a, sigma, axis, mode)
                worst = max(worst, float(np.abs(got - want).max()))
    # Measured: 2.2e-16 (scipy's FMA contraction and, at sigma 3.9, exp's last bit).
    assert worst <= 1e-12


def test_gaussian_kernel_matches_scipy_except_exp_last_bits():
    from scipy.ndimage._filters import _gaussian_kernel1d

    for sigma in (0.5, 1.5, 1.6, 2.7, 3.9):
        want = _gaussian_kernel1d(sigma, 0, int(4.0 * sigma + 0.5))
        got = _native.gaussian_kernel1d(sigma)
        assert got.shape == want.shape
        # libm's exp and numpy's agree on all but a few kernel values (sigma 3.9: 4 of 33).
        np.testing.assert_allclose(got, want, rtol=1e-15, atol=0)


@pytest.mark.parametrize("shape", SHAPES)
def test_median_filter3_matches_scipy_bitwise(shape):
    rng = np.random.default_rng(sum(shape))
    a = rng.random(shape)
    a[0, 0] = a[-1, -1]                      # duplicate values
    want = ndimage.median_filter(a, size=3, mode="nearest")
    assert same_bits(_native.median_filter3(a), want)
    q = np.round(a * 4) / 4                  # many ties
    assert same_bits(_native.median_filter3(q), ndimage.median_filter(q, size=3, mode="nearest"))


def _crops():
    """(name, rgb) inputs as `run` feeds `denoise`: transparent pixels filled first."""
    root = Path(__file__).resolve().parent / "fixtures"
    out = []
    for name in FIXTURES:
        loaded = io.load(root / f"{name}.png")
        mask = loaded.alpha >= 128
        rgb = pyref._fill_transparent(loaded.rgb, mask)
        out.append((name, np.ascontiguousarray(rgb[40:104, 50:114])))
    rng = np.random.default_rng(7)
    out.append(("random", rng.random((37, 45, 3))))
    out.append(("random_dark", rng.random((20, 30, 3)) * 0.05))
    flat = np.full((9, 11, 3), 0.25)
    out.append(("flat", flat))
    return out


def test_denoise_none_and_median_are_bitwise():
    for name, rgb in _crops():
        for kind in ("none", "median"):
            config = Config(denoise=kind)
            want = pyref.denoise(rgb, config)
            got = _native.denoise(rgb, config.to_dict())
            assert same_bits(got, np.asarray(want, dtype=np.float64)), (name, kind)


@pytest.mark.parametrize("sigmas", [(0.08, 2.0), (0.08, 0.5), (0.2, 1.0), (0.03, 3.0),
                                    (0.5, 16.0), (1.0, 0.1)])
def test_denoise_bilateral_matches_skimage(sigmas):
    sc, ss = sigmas
    config = Config(denoise="bilateral", denoise_sigma_color=sc, denoise_sigma_spatial=ss)
    worst, flips = 0.0, 0
    for _name, rgb in _crops():
        if ss >= 16 and rgb.shape[0] > 40:
            rgb = rgb[:32, :32].copy()        # the 97×97 window: keep skimage quick
        want = pyref.denoise(rgb, config)
        got = _native.denoise(rgb, config.to_dict())
        assert got.shape == want.shape
        worst = max(worst, float(np.abs(got - want).max()))
        flips += int((to8(got) != to8(want)).sum())
    # Measured: at most 7.8e-16 (sigma_spatial 2 and 16), no 8-bit flips.
    assert worst <= 1e-9, worst
    assert flips == 0


PORTRAIT = Path("/Users/cole/Documents/lib/pixelforge/.claude/worktrees/16-bit-sprite-photo-"
                "5897f3/out/sample/portrait.png")


@pytest.mark.slow
@pytest.mark.skipif(not PORTRAIT.is_file(), reason="sample portrait not available")
def test_bilateral_portrait():
    rgb = np.ascontiguousarray(io.load(PORTRAIT).rgb)
    config = Config()
    want = pyref.denoise(rgb, config)
    got = _native.denoise(rgb, config.to_dict())
    dev = np.abs(got - want).max(axis=-1)
    # Measured: 4 pixels above 1e-12 (colour-bin flips from skimage's fused distance), the
    # worst 5.4e-6; everywhere else at most 7.8e-16; no 8-bit value changes.
    assert int((dev > 1e-12).sum()) <= 20
    assert float(dev.max()) <= 1e-4
    assert int((to8(got) != to8(want)).sum()) == 0


def test_bilateral_direct_call_matches_skimage():
    """`denoise` is skimage's `denoise_bilateral(..., mode="edge")` clipped to [0, 1]."""
    rng = np.random.default_rng(2)
    rgb = rng.random((24, 19, 3))
    want = np.clip(denoise_bilateral(rgb, sigma_color=0.1, sigma_spatial=1.5,
                                     channel_axis=-1, mode="edge"), 0, 1)
    got = _native.denoise(rgb, Config(denoise_sigma_color=0.1,
                                      denoise_sigma_spatial=1.5).to_dict())
    assert float(np.abs(got - want).max()) <= 1e-9


def test_bilateral_is_thread_count_independent(monkeypatch):
    rng = np.random.default_rng(4)
    rgb = rng.random((70, 33, 3))
    d = Config().to_dict()
    outs = []
    for t in ("1", "2", "3", "8"):
        monkeypatch.setenv("PIXELFORGE_THREADS", t)
        outs.append(_native.denoise(rgb, d))
    assert all(same_bits(o, outs[0]) for o in outs)


def test_argument_errors():
    a = np.zeros((3, 4))
    with pytest.raises(ValueError, match="mode must be"):
        _native.uniform_filter1d(a, 3, 0, "reflect")
    with pytest.raises(ValueError, match="axis must be 0 or 1"):
        _native.gaussian_filter1d(a, 1.0, 2, "nearest")
    with pytest.raises(ValueError, match="size must be at least 1"):
        _native.uniform_filter1d(a, 0, 0, "nearest")
    with pytest.raises(ValueError, match=r"arr must have shape \(H, W\)"):
        _native.median_filter3(np.zeros(5))
    with pytest.raises(ValueError, match="rgb must be a float64"):
        _native.denoise(np.zeros((2, 2, 3), dtype=np.float32), Config().to_dict())
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.denoise(np.zeros((4, 4, 3))[:, ::2], Config().to_dict())
