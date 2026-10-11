"""The native quantize stage (pixelforge.quantize -> _native) against the former numpy/scipy
implementation, copied here as the reference. Everything is compared bit for bit."""

import numpy as np
import pytest
from scipy import ndimage

from pixelforge import Config, _native, color, quantize
from pixelforge.config import PRESETS

FIXTURES = ("checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color")
SHAPES = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 9), (9, 1), (3, 3), (4, 7), (8, 8), (17, 23),
          (64, 5), (33, 64)]


# ------------------------------------------------------------------------- reference

def ref_bayer(n):
    m = np.array([[0, 2], [3, 1]], dtype=np.int64)
    while len(m) < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m / float(n * n) - 0.5


def ref_local_std(lab, mask=None):
    lab = np.asarray(lab, dtype=np.float64)
    if mask is None or mask.all():
        variance = np.zeros(lab.shape[:2])
        for c in range(3):
            mean = ndimage.uniform_filter(lab[..., c], size=3, mode="nearest")
            mean_sq = ndimage.uniform_filter(lab[..., c] * lab[..., c], size=3, mode="nearest")
            variance += np.maximum(mean_sq - mean * mean, 0.0)
        return np.sqrt(variance)
    weight = mask.astype(np.float64)
    count = ndimage.uniform_filter(weight, size=3, mode="nearest")
    safe = np.where(mask, np.maximum(count, 1e-12), 1.0)
    variance = np.zeros(lab.shape[:2])
    for c in range(3):
        x = np.where(mask, lab[..., c], 0.0)
        mean = ndimage.uniform_filter(x, size=3, mode="nearest") / safe
        mean_sq = ndimage.uniform_filter(x * x, size=3, mode="nearest") / safe
        variance += np.maximum(mean_sq - mean * mean, 0.0)
    return np.where(mask, np.sqrt(variance), 0.0)


def ref_dither_offsets(small_lab, config, mask=None):
    if config.dither == "none" or config.dither_strength == 0:
        return None
    h, w = small_lab.shape[:2]
    n = 8 if config.dither == "bayer8" else 4
    matrix = ref_bayer(n)
    threshold = matrix[np.arange(h)[:, None] % n, np.arange(w)[None, :] % n]
    amount = config.dither_strength * quantize.DITHER_L_AMPLITUDE * threshold
    if config.dither == "auto":
        std = ref_local_std(small_lab, mask)
        amount = np.where(std < config.dither_variance_threshold, amount, 0.0)
    return np.stack([amount, 0.5 * amount, 0.5 * amount], axis=-1)


def ref_run(small_lab, small_mask, palette_lab, config):
    lab = np.asarray(small_lab, dtype=np.float64)
    offsets = ref_dither_offsets(lab, config, small_mask)
    if offsets is not None:
        lab = lab + offsets
    indices = np.full(small_mask.shape, -1, dtype=np.int64)
    indices[small_mask] = color.nearest_index(lab[small_mask], palette_lab)
    return indices


# --------------------------------------------------------------------------- helpers

def random_lab(rng, shape):
    lab = np.empty(shape + (3,))
    lab[..., 0] = rng.uniform(0, 100, shape)
    lab[..., 1:] = rng.uniform(-90, 90, shape + (2,))
    return lab


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        a.view(np.uint64) if a.dtype == np.float64 else a,
        b.view(np.uint64) if b.dtype == np.float64 else b)


def masks(rng, shape):
    yield None
    yield np.ones(shape, dtype=bool)
    yield rng.random(shape) < 0.6
    sparse = np.zeros(shape, dtype=bool)
    sparse[rng.integers(shape[0]), rng.integers(shape[1])] = True
    yield sparse
    yield np.zeros(shape, dtype=bool)


def fixture_cases(preprocessed):
    for name in FIXTURES:
        for preset in PRESETS:
            pre, config = preprocessed(name, preset=preset)
            yield name, preset, pre, config


# ----------------------------------------------------------------------------- tests

@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_bayer(n):
    assert same(quantize.bayer(n), ref_bayer(n))


@pytest.mark.parametrize("shape", SHAPES)
def test_uniform_filter3_matches_scipy(shape):
    rng = np.random.default_rng(shape[0] * 100 + shape[1])
    for x in (rng.random(shape), rng.normal(0, 1e3, shape), np.full(shape, 0.1),
              rng.integers(0, 2, shape).astype(np.float64)):
        assert same(_native.uniform_filter3(x), ndimage.uniform_filter(x, size=3,
                                                                       mode="nearest"))


@pytest.mark.parametrize("shape", SHAPES)
def test_local_std_random(shape):
    rng = np.random.default_rng(7 + shape[0] * 31 + shape[1])
    lab = random_lab(rng, shape)
    for mask in masks(rng, shape):
        assert same(quantize.local_std(lab, mask), ref_local_std(lab, mask)), mask


def test_local_std_flat_and_duplicate_values():
    rng = np.random.default_rng(3)
    lab = np.tile([50.0, 10.0, -20.0], (12, 9, 1))
    lab[4:7, 2:5] = [70.1, -3.3, 0.7]
    for mask in masks(rng, (12, 9)):
        assert same(quantize.local_std(lab, mask), ref_local_std(lab, mask))


def test_local_std_fixtures(preprocessed):
    for name, preset, pre, _ in fixture_cases(preprocessed):
        for mask in (None, pre.mask, np.ones_like(pre.mask)):
            assert same(quantize.local_std(pre.lab, mask), ref_local_std(pre.lab, mask)), (
                name, preset)


DITHER_CONFIGS = [
    dict(dither="none"),
    dict(dither="bayer4", dither_strength=0.0),
    dict(dither="bayer4"),
    dict(dither="bayer8", dither_strength=0.37),
    dict(dither="auto"),
    dict(dither="auto", dither_strength=1.0, dither_variance_threshold=25.0),
    dict(dither="auto", dither_variance_threshold=0.0),
]


@pytest.mark.parametrize("overrides", DITHER_CONFIGS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (8, 8), (13, 21)])
def test_dither_offsets(shape, overrides):
    config = Config(**overrides)
    rng = np.random.default_rng(11)
    lab = random_lab(rng, shape)
    lab[: shape[0] // 2] = [40.0, 5.0, 5.0]       # a flat part for "auto"
    for mask in masks(rng, shape):
        got = quantize.dither_offsets(lab, config, mask)
        want = ref_dither_offsets(lab, config, mask)
        assert (got is None) == (want is None)
        if want is not None:
            assert same(got, want)


@pytest.mark.parametrize("k", [1, 2, 7, 8, 9, 16, 32, 256])
def test_run_random(k):
    rng = np.random.default_rng(k)
    palette_lab = random_lab(rng, (k,))
    palette_lab[k // 2] = palette_lab[0]          # a duplicate: ties go to the lowest index
    for overrides in DITHER_CONFIGS:
        config = Config(**overrides)
        for shape in [(1, 1), (1, 7), (7, 1), (16, 16), (29, 31)]:
            lab = random_lab(rng, shape)
            for mask in masks(rng, shape):
                if mask is None:
                    continue
                got = quantize.run(lab, mask, palette_lab, config)
                assert same(got, ref_run(lab, mask, palette_lab, config)), (overrides, shape)


def test_run_fixtures(preprocessed):
    rng = np.random.default_rng(5)
    for name, preset, pre, config in fixture_cases(preprocessed):
        for k in (2, 16):
            palette_lab = pre.lab[pre.mask][rng.choice(int(pre.mask.sum()), k)]
            for overrides in DITHER_CONFIGS:
                c = config.replace(**overrides)
                assert same(quantize.run(pre.lab, pre.mask, palette_lab, c),
                            ref_run(pre.lab, pre.mask, palette_lab, c)), (name, preset)


def test_argument_errors():
    with pytest.raises(ValueError, match=r"local_std: lab must have shape \(H, W, 3\)"):
        _native.local_std(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="local_std: mask must have the image's shape"):
        _native.local_std(np.zeros((4, 4, 3)), np.ones((4, 5), dtype=bool))
    with pytest.raises(ValueError, match="local_std: mask must be a bool numpy.ndarray"):
        _native.local_std(np.zeros((4, 4, 3)), np.ones((4, 4)))
    with pytest.raises(ValueError, match="quantize_run: palette_lab must not be empty"):
        _native.quantize_run(np.zeros((2, 2, 3)), np.ones((2, 2), dtype=bool),
                             np.zeros((0, 3)), Config().to_dict())
    # No opaque pixel: an empty palette is fine.
    out = quantize.run(np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=bool), np.zeros((0, 3)),
                       Config())
    assert same(out, np.full((2, 2), -1, dtype=np.int64))
