"""The native masks (crates/pixelforge/src/mask.rs) reproduce pixelforge/preprocess.py and the
scipy routines it calls, bit for bit: the exact distance transform with scipy's feature
indices, transparent filling, 4-connected labelling, 8-neighbour dilation, flat-background
keying with fringe unmixing, and the subject mask, weight and crop box. The former Python
functions, kept in tests/preprocess_reference.py, are the reference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from pixelforge import Config, _native, io
from pixelforge.errors import ConfigError, PixelforgeError
from tests import preprocess_reference as pyref

ROOT = Path(__file__).resolve().parent.parent
# samples/ is not committed (.gitignore); PIXELFORGE_SAMPLES points elsewhere, e.g. at the
# main checkout's samples/ from a worktree. Without it the sample tests are skipped.
SAMPLES_DIR = Path(os.environ.get("PIXELFORGE_SAMPLES", ROOT / "samples"))
SAMPLES = sorted(SAMPLES_DIR.glob("*.png"))
FIXTURES = ("checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color")
EIGHT = np.ones((3, 3), dtype=bool)


def same(a, b):
    """Bitwise equality of two arrays (shape, dtype kind and every value's bits)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return np.array_equal(a, b)


def scipy_edt(mask):
    dist, (rows, cols) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return dist, rows, cols


def assert_edt(mask):
    dist, rows, cols = _native.edt(np.ascontiguousarray(mask))
    want_d, want_r, want_c = scipy_edt(mask)
    assert rows.dtype == np.int64 and cols.dtype == np.int64
    assert np.array_equal(rows, want_r) and np.array_equal(cols, want_c)
    assert same(dist, want_d)


def random_masks(n=200, seed=20261011):
    rng = np.random.default_rng(seed)
    out = [np.ones((1, 1), bool), np.zeros((1, 1), bool), np.ones((5, 7), bool),
           np.zeros((6, 4), bool), np.ones((1, 9), bool), np.zeros((9, 1), bool)]
    while len(out) < n:
        h, w = int(rng.integers(1, 71)), int(rng.integers(1, 46))
        density = float(rng.uniform(0.02, 0.98))
        out.append(rng.random((h, w)) < density)
    return out


def fixture(name):
    return io.load(ROOT / "tests" / "fixtures" / f"{name}.png")


def sample_ids():
    if not SAMPLES:
        return [pytest.param(None, marks=pytest.mark.skip(reason=f"no samples in {SAMPLES_DIR}"))]
    return [p.stem for p in SAMPLES]


def sample(name):
    return io.load(SAMPLES_DIR / f"{name}.png")


# ------------------------------------------------------------------ distance transform

def test_edt_matches_the_oracle_and_scipy():
    data = json.loads((ROOT / "tests" / "oracle" / "edt_indices.json").read_text())
    assert len(data["cases"]) == 8
    for case in data["cases"]:
        mask = np.array([[c == "1" for c in row] for row in case["mask"]])
        dist, rows, cols = _native.edt(mask)
        assert np.array_equal(rows, np.array(case["indices_row"])), case["name"]
        assert np.array_equal(cols, np.array(case["indices_col"])), case["name"]
        assert same(dist, np.array(case["distances"], dtype=np.float64)), case["name"]
        assert_edt(mask)


def test_edt_random_masks():
    for mask in random_masks():
        assert_edt(mask)


def test_edt_with_many_ties():
    # Regular grids of features make every background pixel equidistant to several.
    for step in (2, 3, 4, 5):
        for shape in ((17, 23), (24, 24), (31, 9)):
            mask = np.zeros(shape, dtype=bool)
            mask[::step, ::step] = True
            assert_edt(mask)
            assert_edt(~mask)
            ys, xs = np.indices(shape)
            assert_edt((ys + xs) % (2 * step) == 0)    # diagonal stripes
            assert_edt((ys * 2 + xs * 3) % (3 * step) == 1)


@pytest.mark.parametrize("name", FIXTURES)
def test_edt_fixtures(name):
    loaded = fixture(name)
    for threshold in (1, 128, 255):
        assert_edt(loaded.alpha >= threshold)


@pytest.mark.parametrize("name", sample_ids())
def test_edt_samples(name):
    loaded = sample(name)
    assert_edt(loaded.alpha >= 128)


def test_edt_rejects_bad_arguments():
    with pytest.raises(ValueError, match="edt: mask must be a bool"):
        _native.edt(np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"must have shape \(H, W\)"):
        _native.edt(np.zeros((3, 3, 1), dtype=bool))
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.edt(np.zeros((4, 4), dtype=bool)[:, ::2])


# ------------------------------------------------------------------ fill, label, dilation

def test_fill_transparent_random():
    rng = np.random.default_rng(7)
    for mask in random_masks(120, seed=8):
        if not mask.any():
            continue
        rgb = rng.random(mask.shape + (3,))
        assert same(_native.fill_transparent(rgb, mask), pyref._fill_transparent(rgb, mask))


@pytest.mark.parametrize("name", FIXTURES)
def test_fill_transparent_fixtures(name):
    loaded = fixture(name)
    mask = loaded.alpha >= 128
    if mask.any():
        assert same(_native.fill_transparent(loaded.rgb, mask),
                    pyref._fill_transparent(loaded.rgb, mask))


@pytest.mark.parametrize("name", sample_ids())
def test_fill_transparent_samples(name):
    loaded = sample(name)
    mask = loaded.alpha >= 128
    assert same(_native.fill_transparent(loaded.rgb, mask),
                pyref._fill_transparent(loaded.rgb, mask))


def same_partition(native, n_native, reference, n_reference):
    """Equal regions up to numbering."""
    if n_native != n_reference or not np.array_equal(native > 0, reference > 0):
        return False
    pairs = np.unique(np.stack([native[native > 0], reference[reference > 0]]), axis=1)
    return pairs.shape[1] == n_native


def test_label4_random():
    for mask in random_masks(150, seed=9):
        labels, n = _native.label4(mask)
        want, n_want = ndimage.label(mask)
        assert labels.dtype == np.int64
        assert same_partition(labels, n, want, n_want)
        # Raster-order numbering, like scipy's.
        assert np.array_equal(labels, want)


def test_binary_dilation_random():
    for mask in random_masks(120, seed=10):
        for iterations in (1, 2, 4, 7):
            want = ndimage.binary_dilation(mask, structure=EIGHT, iterations=iterations)
            assert np.array_equal(_native.binary_dilation8(mask, iterations), want)
        want = ndimage.binary_dilation(mask, structure=EIGHT, iterations=0)
        assert np.array_equal(_native.binary_dilation8(mask, 0), want)


# ------------------------------------------------------------------ keying

def composite(rgb, alpha, backdrop):
    a = alpha[..., None] / 255.0
    return rgb * a + np.asarray(backdrop, dtype=np.float64) * (1.0 - a)


def assert_flat_background(rgb, tolerance, fringe):
    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    got = _native.flat_background(rgb, tolerance, fringe)
    want = pyref.flat_background(rgb, tolerance, fringe)
    if want is None:
        assert got is None
    else:
        assert got is not None and np.array_equal(got, want)
    return want


BACKDROPS = [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.2, 0.8, 0.3), (1.0, 0.0, 1.0),
             (0.5, 0.5, 0.5), (0.13, 0.27, 0.91)]


def test_flat_background_circle_composites():
    loaded = fixture("circle_alpha")
    keyed = 0
    for backdrop in BACKDROPS:
        rgb = composite(loaded.rgb, loaded.alpha, backdrop)
        for tolerance in (0.0, 0.02, 0.05, 0.2):
            for fringe in range(5):
                keyed += assert_flat_background(rgb, tolerance, fringe) is not None
    assert keyed > 50


@pytest.mark.slow
@pytest.mark.parametrize("name", sample_ids())
def test_flat_background_samples(name):
    loaded = sample(name)
    for backdrop in BACKDROPS[:3]:
        rgb = composite(loaded.rgb, loaded.alpha, backdrop)
        for tolerance in (0.02, 0.1):
            for fringe in (0, 2, 4):
                assert_flat_background(rgb, tolerance, fringe)
    # The opaque sample as it is (keyed only if it has a flat border).
    for fringe in (0, 3):
        assert_flat_background(loaded.rgb, 0.02, fringe)


@pytest.mark.parametrize("name", FIXTURES)
def test_flat_background_fixtures(name):
    loaded = fixture(name)
    for fringe in (0, 2):
        assert_flat_background(loaded.rgb, 0.02, fringe)
        assert_flat_background(composite(loaded.rgb, loaded.alpha, (1, 1, 1)), 0.02, fringe)


def test_flat_background_degenerate_shapes():
    rng = np.random.default_rng(11)
    for shape in ((1, 1), (1, 5), (6, 1), (2, 2), (2, 9), (3, 3)):
        for _ in range(5):
            rgb = np.where(rng.random(shape + (1,)) < 0.8, 1.0, rng.random(shape + (3,)))
            for fringe in (0, 2):
                assert_flat_background(rgb, 0.02, fringe)


def test_flat_background_even_border_median():
    # An even border count whose middle pair differs: the reference is their mean.
    rgb = np.ones((4, 5, 3))
    rgb[0, :2] = 0.7
    rgb[1:3, 1:4] = (0.1, 0.4, 0.2)
    for tolerance in (0.0, 0.2, 0.31):
        assert_flat_background(rgb, tolerance, 2)


def test_unmix_fringe_directly():
    rng = np.random.default_rng(12)
    loaded = fixture("circle_alpha")
    for backdrop in BACKDROPS:
        rgb = composite(loaded.rgb, loaded.alpha, backdrop)
        reference = np.asarray(backdrop, dtype=np.float64)
        background = np.abs(rgb - reference).max(axis=-1) <= 0.02
        for passes in range(5):
            got = _native.unmix_fringe(rgb, background, tuple(reference), passes)
            want = pyref._unmix_fringe(rgb, background, reference, passes)
            assert np.array_equal(got, want)
    for _ in range(20):
        h, w = (int(v) for v in rng.integers(2, 30, 2))
        rgb = rng.random((h, w, 3))
        background = rng.random((h, w)) < rng.uniform(0.1, 0.9)
        reference = rng.random(3)
        for passes in (1, 3):
            got = _native.unmix_fringe(rgb, background, tuple(reference), passes)
            assert np.array_equal(got, pyref._unmix_fringe(rgb, background, reference,
                                                                passes))


# ------------------------------------------------------------------ subject

SUBJECT_CONFIGS = [
    {},
    {"preset": "background"},
    {"key_bg": False},
    {"key_bg_fringe": 0},
    {"key_bg_fringe": 4, "key_bg_tolerance": 0.1},
    {"alpha_threshold": 0},
    {"alpha_threshold": 1},
    {"alpha_threshold": 255},
    {"crop_to_alpha": False},
]


def assert_subject(rgb, alpha, kwargs):
    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    alpha = np.ascontiguousarray(alpha, dtype=np.uint8)
    config = Config(**kwargs)
    try:
        _, mask, weight, keyed, box = pyref._subject(rgb, alpha, config)
    except PixelforgeError as exc:
        with pytest.raises(PixelforgeError, match=str(exc)):
            _native.subject(rgb, alpha, kwargs)
        return None
    got = _native.subject(rgb, alpha, kwargs)
    assert np.array_equal(got[0], mask)
    assert same(got[1], weight)
    assert got[2] == keyed
    assert got[3] == tuple(int(v) for v in box)
    return keyed


@pytest.mark.parametrize("name", FIXTURES)
def test_subject_fixtures(name):
    loaded = fixture(name)
    for kwargs in SUBJECT_CONFIGS:
        assert_subject(loaded.rgb, loaded.alpha, kwargs)
        opaque = np.full_like(loaded.alpha, 255)
        assert_subject(composite(loaded.rgb, loaded.alpha, (1, 1, 1)), opaque, kwargs)


@pytest.mark.slow
@pytest.mark.parametrize("name", sample_ids())
def test_subject_samples(name):
    loaded = sample(name)
    keyed = 0
    for kwargs in SUBJECT_CONFIGS:
        assert_subject(loaded.rgb, loaded.alpha, kwargs)
        opaque = np.full_like(loaded.alpha, 255)
        for backdrop in BACKDROPS[:2]:
            keyed += bool(assert_subject(composite(loaded.rgb, loaded.alpha, backdrop), opaque,
                                         kwargs))


def test_subject_errors():
    rgb = np.zeros((4, 4, 3))
    with pytest.raises(PixelforgeError, match="fully transparent after alpha thresholding"):
        _native.subject(rgb, np.zeros((4, 4), dtype=np.uint8), {})
    with pytest.raises(ConfigError, match="alpha_threshold"):
        _native.subject(rgb, np.zeros((4, 4), dtype=np.uint8), {"alpha_threshold": 300})
    with pytest.raises(ValueError, match="same height and width"):
        _native.subject(rgb, np.zeros((4, 5), dtype=np.uint8), {})
