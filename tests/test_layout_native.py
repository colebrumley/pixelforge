"""The native geometry (crates/pixelforge/src/layout.rs) reproduces pixelforge/preprocess.py
bit for bit: outline margin, fit mode, resolve_layout with the tileset fit and its warning
text, fit_crop, the integer repeat and its limit, the box pre-reduction (numpy's summation
order) and the seamless wrap padding. The Python functions are the reference until the
integration step switches preprocess.py over."""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from pixelforge import Config, _native, io, preprocess
from pixelforge.errors import ConfigError, PixelforgeError

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ("checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color")


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return a.tobytes() == b.tobytes()


# ------------------------------------------------------------------ layout

SIZES = [(1, 1), (8, 8), (13, 7), (7, 13), (64, 64), (100, 50), (50, 100), (160, 96),
         (300, 200), (200, 300), (1000, 3), (3, 1000), (1023, 769), (4096, 4095)]

GRID = {
    "preset": ["sprite", "background"],
    "out_width": [None, 8, 48, 64, 100],
    "out_height": [None, 16, 48, 96],
    "scale": [None, 0.5, 1.0, 3.0, 7.5],
    "canvas": [None, "48x32", "64x64", "16x96"],
    "tileset": [False, True],
    "fit": ["pad", "stretch", "crop"],
    "seamless": [False, True],
    "outline": ["none", "dark"],
    "tile_size": [8, 16],
}


def valid_configs():
    out = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        kwargs = {k: v for k, v in zip(keys, values) if v is not None}
        try:
            Config(**kwargs)
        except ConfigError:
            continue
        out.append(kwargs)
    return out


VALID = valid_configs()


def python_layout(w, h, config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = preprocess.resolve_layout(w, h, config)
        except ConfigError as exc:
            return ("error", str(exc)), []
    return result, [str(c.message) for c in caught if c.category is UserWarning]


def native_layout(w, h, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = _native.resolve_layout(w, h, kwargs)
        except ConfigError as exc:
            return ("error", str(exc)), []
    return result, [str(c.message) for c in caught if c.category is UserWarning]


def test_the_grid_is_large():
    assert len(VALID) > 500


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_resolve_layout_grid(size):
    w, h = size
    warned = errors = 0
    for kwargs in VALID:
        config = Config(**kwargs)
        want, want_warnings = python_layout(w, h, config)
        got, got_warnings = native_layout(w, h, kwargs)
        assert got == want, kwargs
        assert got_warnings == want_warnings, kwargs
        warned += bool(want_warnings)
        errors += want[0] == "error"
        if want[0] != "error":
            assert _native.outline_margin(kwargs) == preprocess.outline_margin(config)
            assert _native.fit_mode(kwargs) == preprocess.fit_mode(config)
    if size not in ((1, 1), (8, 8), (64, 64), (4096, 4095)):
        assert warned > 0


def test_resolve_layout_examples_and_warning():
    sprite = {"canvas": "48x32"}
    assert _native.resolve_layout(100, 50, sprite) == (48, 32, (46, 23, 0, 3))
    pad = {"preset": "background", "out_width": 64}
    assert _native.resolve_layout(160, 96, pad) == (64, 48, (64, 38, 0, 5))
    with pytest.warns(UserWarning, match="aspect ratio from 1.667"):
        _native.resolve_layout(160, 96, {**pad, "fit": "stretch"})
    with pytest.raises(ConfigError, match=r"tile_size \(512\) exceeds the longest output edge"):
        _native.resolve_layout(300, 200, {"preset": "background", "tile_size": 512})


def test_fit_crop_grid():
    for w, h in SIZES:
        for ow, oh in itertools.product((1, 8, 16, 48, 64, 100), repeat=2):
            assert _native.fit_crop(w, h, ow, oh) == preprocess.fit_crop(w, h, ow, oh)


def test_prereduce_factor_grid():
    for h, w in [(1, 1), (64, 64), (100, 1000), (1024, 768), (4096, 17)]:
        for oh, ow in [(1, 1), (8, 8), (16, 64), (48, 32)]:
            for ratio in (0, 1, 2, 8, 50):
                assert _native.prereduce_factor(h, w, oh, ow, ratio) == \
                    preprocess.prereduce_factor(h, w, oh, ow, ratio)


# ------------------------------------------------------------------ pre-reduction

def assert_prereduce(rgb, mask, factor, weight):
    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    want = preprocess.prereduce(rgb, mask, factor, weight)
    got = _native.prereduce(rgb, mask, factor, weight)
    assert len(got) == len(want)
    assert same(got[0], want[0]), (rgb.shape, factor)
    assert np.array_equal(got[1], want[1])
    if weight is not None:
        assert same(got[2], want[2]), (rgb.shape, factor)


def test_prereduce_random():
    rng = np.random.default_rng(20261012)
    for factor in range(1, 10):
        shapes = [(factor, factor), (factor + 1, 2 * factor - 1), (3 * factor, factor),
                  (2 * factor + 1, 5 * factor + 2), (40, 33), (17, 64)]
        for h, w in shapes:
            if h < factor or w < factor:
                continue
            for density in (0.1, 0.5, 0.9, 1.0):
                mask = rng.random((h, w)) < density
                if not mask.any():
                    continue
                rgb = rng.random((h, w, 3))
                weight = np.where(mask, rng.random((h, w)), 0.0)
                assert_prereduce(rgb, mask, factor, None)
                assert_prereduce(rgb, mask, factor, weight)
                alpha = rng.integers(0, 256, (h, w)).astype(np.float64)
                assert_prereduce(rgb, mask, factor, np.where(mask, np.maximum(alpha, 1) / 255, 0.0))
    # numpy's sums start from +0.0: a block of -0.0 averages to +0.0.
    zeros = np.full((4, 4), -0.0)
    assert_prereduce(np.full((4, 4, 3), -0.0), np.ones((4, 4), bool), 2, zeros)
    assert_prereduce(np.full((2, 2, 3), -0.0), np.ones((2, 2), bool), 2, zeros[:2, :2].copy())


@pytest.mark.parametrize("name", FIXTURES)
def test_prereduce_fixtures(name):
    loaded = io.load(ROOT / "tests" / "fixtures" / f"{name}.png")
    mask = loaded.alpha >= 128
    if not mask.any():
        return
    weight = np.where(mask, np.maximum(loaded.alpha, 1).astype(np.float64) / 255.0, 0.0)
    for factor in range(1, 10):
        assert_prereduce(loaded.rgb, mask, factor, weight)


# ------------------------------------------------------------------ repeat, padding

def test_repeat_matches_numpy():
    rng = np.random.default_rng(13)
    for h, w in [(1, 1), (1, 7), (5, 3), (9, 12)]:
        rgb = rng.random((h, w, 3))
        mask = rng.random((h, w)) < 0.5
        weight = rng.random((h, w))
        for factor in (1, 2, 3, 7):
            got = _native.repeat(rgb, mask, weight, factor)
            for g, a in zip(got, (rgb, mask, weight)):
                assert same(g, np.repeat(np.repeat(a, factor, axis=0), factor, axis=1))


def test_repeat_factor_and_limit():
    for h, w in [(1, 1), (3, 5), (10, 20), (64, 64), (100, 7)]:
        for ow, oh in [(8, 8), (48, 32), (64, 64), (256, 100)]:
            want = max(math.ceil(ow / w), math.ceil(oh / h))
            if h * w * want * want <= io.MAX_INPUT_PIXELS:
                assert _native.repeat_factor(h, w, ow, oh) == max(want, 1)
    rgb = np.zeros((1, 2000, 3))
    alpha = np.full((1, 2000), 255, dtype=np.uint8)
    config = Config(out_width=2000, out_height=2000, key_bg=False, denoise="none")
    with pytest.raises(PixelforgeError) as python:
        preprocess.run(rgb, alpha, config)
    with pytest.raises(PixelforgeError) as native:
        _native.repeat_factor(1, 2000, 2000, 2000)
    assert str(native.value) == str(python.value)


def test_wrap_pad_matches_numpy():
    rng = np.random.default_rng(14)
    for h, w in [(1, 1), (2, 3), (5, 4), (7, 9)]:
        rgb = rng.random((h, w, 3))
        mask = rng.random((h, w)) < 0.5
        weight = rng.random((h, w))
        for py, px in [(0, 0), (1, 2), (3, 3), (h + 2, 2 * w + 1), (5 * h, 4 * w + 3)]:
            got = _native.wrap_pad(rgb, mask, weight, py, px)
            assert same(got[0], np.pad(rgb, ((py, py), (px, px), (0, 0)), mode="wrap"))
            assert same(got[1], np.pad(mask, ((py, py), (px, px)), mode="wrap"))
            assert same(got[2], np.pad(weight, ((py, py), (px, px)), mode="wrap"))


def test_seamless_pads():
    for h, w in [(8, 8), (100, 37), (1024, 768)]:
        for ih, iw in [(8, 8), (16, 32), (63, 17)]:
            want = (math.ceil(preprocess.SEAMLESS_PAD_OUT * h / ih),
                    math.ceil(preprocess.SEAMLESS_PAD_OUT * w / iw))
            assert _native.seamless_pads(h, w, ih, iw) == want
