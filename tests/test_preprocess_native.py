"""The native preprocessing stage (pixelforge.preprocess.run -> _native.preprocess_run, from
crates/pixelforge/src/preprocess.rs) against the former numpy/scipy/scikit-image version,
kept in tests/preprocess_reference.py.

Everything but the LAB is compared bit for bit: mask, weight, sizes, layout, fit, the
pre-reduction factor and keying flag, error classes and messages, warning texts. The LAB is
bitwise as well unless a float filter ran: `enhance` (scipy's Gaussian, see
test_enhance_native.py) or the bilateral denoise (see test_filters_native.py), where it agrees
to within the last bits, plus the rare bilateral colour-bin flip.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from pixelforge import Config, io, preprocess
from pixelforge.errors import ConfigError, PixelforgeError
from tests import preprocess_reference as pyref

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ("checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color")
SAMPLES_DIR = Path(os.environ.get("PIXELFORGE_SAMPLES", ROOT / "samples"))
SAMPLES = sorted(SAMPLES_DIR.glob("*.png"))

# Configs that walk every branch of run(): keying, crop, canvas, fit modes, repeat,
# pre-reduction, enhance (with and without wrap), the three denoisers, alpha_threshold 0.
CONFIGS = [
    dict(preset="sprite"),
    dict(preset="background"),
    dict(preset="sprite", denoise="none"),
    dict(preset="sprite", denoise="median", key_bg_fringe=0),
    dict(preset="sprite", canvas="48x32", denoise="none"),
    dict(preset="sprite", out_width=200, out_height=150, denoise="median"),       # repeat
    dict(preset="sprite", out_width=8, out_height=8, prereduce_max_ratio=2,
         denoise="none"),                                                         # prereduce
    dict(preset="background", out_width=64, fit="crop", denoise="none"),
    dict(preset="background", out_width=64, fit="pad", denoise="median"),
    dict(preset="background", out_width=48, out_height=48, tileset=False, seamless=True,
         denoise="none", enhance=1.0, ink=0.5),
    dict(preset="background", seamless=True, denoise="median"),
    dict(preset="sprite", enhance=1.0, ink=0.6, denoise="none"),
    dict(preset="sprite", enhance=0.5, enhance_black=0.1, enhance_white=0.9, denoise="none",
         alpha_threshold=0),
    dict(preset="background", scale=3, denoise="none", ink=1.0),
]


def _lab_is_exact(kwargs):
    config = Config(**kwargs)
    return config.denoise != "bilateral" and config.enhance <= 0 and config.ink <= 0


def _run(module, rgb, alpha, config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pre = module.run(rgb, alpha, config)
        except (ConfigError, PixelforgeError) as exc:
            return (type(exc), str(exc)), []
    return pre, [str(c.message) for c in caught if c.category is UserWarning]


def _compare(rgb, alpha, kwargs, label):
    config = Config(**kwargs)
    want, want_warnings = _run(pyref, rgb, alpha, config)
    got, got_warnings = _run(preprocess, rgb, alpha, config)
    assert got_warnings == want_warnings, label
    if isinstance(want, tuple):
        assert got == want, label
        return None
    for field in ("out_width", "out_height", "pad_out", "background_keyed", "outline_margin",
                  "layout", "fit", "prereduce_factor"):
        assert getattr(got, field) == getattr(want, field), (label, field)
    assert type(got.out_width) is int and type(got.prereduce_factor) is int
    assert got.mask.dtype == bool and np.array_equal(got.mask, want.mask), label
    assert got.weight.dtype == np.float64 and np.array_equal(got.weight, want.weight), label
    assert got.lab.dtype == np.float64 and got.lab.shape == want.lab.shape, label
    assert got.lab.flags.c_contiguous and got.mask.flags.c_contiguous
    assert (got.inner_width, got.inner_height, got.canvas_pad()) == \
        (want.inner_width, want.inner_height, want.canvas_pad())
    diff = float(np.abs(got.lab - want.lab).max()) if got.lab.size else 0.0
    if _lab_is_exact(kwargs):
        assert diff == 0.0, (label, diff)
    else:
        assert diff < 1e-3, (label, diff)
    return diff


@pytest.mark.parametrize("name", FIXTURES)
def test_run_matches_reference_on_fixtures(fixture_path, name):
    loaded = io.load(fixture_path(name))
    for kwargs in CONFIGS:
        _compare(loaded.rgb, loaded.alpha, kwargs, (name, kwargs))


def test_float_stages_agree_to_the_last_bits(fixture_path):
    """Away from the bilateral's bin flips the LAB differs by rounding only."""
    loaded = io.load(fixture_path("noisy_gradient"))
    for kwargs in (dict(preset="background"), dict(preset="sprite", enhance=1.0, ink=0.6,
                                                   denoise="none")):
        diff = _compare(loaded.rgb, loaded.alpha, kwargs, kwargs)
        assert diff < 1e-10, (kwargs, diff)


def test_synthetic_inputs():
    rng = np.random.default_rng(20261011)
    for h, w in [(1, 1), (1, 40), (40, 1), (3, 5), (17, 23), (64, 48), (130, 70)]:
        rgb = rng.random((h, w, 3))
        for alpha in (np.full((h, w), 255, dtype=np.uint8),
                      rng.integers(0, 256, (h, w)).astype(np.uint8),
                      np.where(rng.random((h, w)) < 0.2, 255, 0).astype(np.uint8)):
            for kwargs in CONFIGS:
                _compare(rgb, alpha, kwargs, ((h, w), kwargs))


def test_flat_backgrounds_are_keyed_alike():
    yy, xx = np.mgrid[:96, :80]
    rgb = np.ones((96, 80, 3))
    disc = np.hypot(yy - 47.5, xx - 39.5)
    coverage = np.clip(30.5 - disc, 0.0, 1.0)[..., None]
    rgb = coverage * np.array([0.85, 0.0, 0.07]) + (1.0 - coverage) * rgb
    rgb[40:50, 30:50] = 1.0                       # enclosed by the subject: kept
    alpha = np.full((96, 80), 255, dtype=np.uint8)
    for kwargs in (dict(preset="sprite"), dict(preset="sprite", key_bg_fringe=0),
                   dict(preset="sprite", crop_to_alpha=False, denoise="none"),
                   dict(preset="sprite", fit="crop", tileset=True, denoise="none")):
        _compare(rgb, alpha, kwargs, kwargs)
        assert preprocess.subject_bbox(rgb, alpha, Config(**kwargs)) == \
            pyref.subject_bbox(rgb, alpha, Config(**kwargs))
        assert np.array_equal(preprocess.subject_brightness(rgb, alpha, Config(**kwargs)),
                              pyref.subject_brightness(rgb, alpha, Config(**kwargs)))


def test_fit_crop_of_an_opaque_image_prereduces_alike():
    """fit="crop" leaves a column slice of the input; with nothing transparent the reference
    pre-reduces (and enhances) that strided view directly."""
    rgb = np.random.default_rng(5).random((300, 520, 3))
    alpha = np.full((300, 520), 255, dtype=np.uint8)
    for kwargs in (dict(preset="background", out_width=64, fit="crop", denoise="none"),
                   dict(preset="background", out_width=32, fit="crop", denoise="none",
                        prereduce_max_ratio=2),
                   dict(preset="background", out_width=32, fit="crop", denoise="none",
                        prereduce_max_ratio=2, enhance=1.0, ink=0.5)):
        _compare(rgb, alpha, kwargs, kwargs)


def test_errors_and_warnings_match():
    rgb = np.full((96, 160, 3), 0.5)
    transparent = np.zeros((96, 160), dtype=np.uint8)
    edges = transparent.copy()
    edges[:, 0] = edges[:, -1] = 255
    cases = [
        (transparent, dict(preset="sprite")),                                  # fully transparent
        (edges, dict(preset="background", out_width=64, fit="crop")),          # crop leaves none
        (np.full((96, 160), 255, dtype=np.uint8),
         dict(preset="background", tile_size=512)),                            # ConfigError
        (np.full((96, 160), 255, dtype=np.uint8),
         dict(preset="background", out_width=64, fit="stretch", denoise="none")),  # warning
    ]
    for alpha, kwargs in cases:
        _compare(rgb, alpha, kwargs, kwargs)
    tiny = np.full((1, 2000, 3), 0.5)
    _compare(tiny, np.full((1, 2000), 255, dtype=np.uint8),
             dict(out_width=2000, out_height=2000, key_bg=False, denoise="none"), "repeat")
    with pytest.raises(PixelforgeError, match="fully transparent"):
        preprocess.subject_bbox(rgb, transparent, Config())
    with pytest.raises(PixelforgeError, match="fully transparent"):
        preprocess.subject_brightness(rgb, transparent, Config())


def test_remove_bg_replaces_the_alpha_before_the_native_stage(monkeypatch):
    rgb = np.random.default_rng(3).random((40, 30, 3))
    alpha = np.full((40, 30), 255, dtype=np.uint8)
    matte = np.zeros((40, 30), dtype=np.uint8)
    matte[10:30, 5:20] = 255
    calls = []

    def fake_remove_background(rgb_arg, alpha_arg):
        calls.append(alpha_arg)
        return matte

    monkeypatch.setattr(preprocess, "remove_background", fake_remove_background)
    config = Config(preset="sprite", remove_bg=True, denoise="none")
    pre = preprocess.run(rgb, alpha, config)
    assert len(calls) == 1 and np.array_equal(calls[0], alpha)
    want = pyref.run(rgb, matte, config.replace(remove_bg=False))
    assert np.array_equal(pre.mask, want.mask) and np.array_equal(pre.lab, want.lab)
    assert preprocess.subject_bbox(rgb, alpha, config) == (9, 31, 4, 21)
    assert len(preprocess.subject_brightness(rgb, alpha, config)) == 20 * 15


def test_non_contiguous_and_other_dtypes_are_accepted(fixture_path):
    loaded = io.load(fixture_path("circle_alpha"))
    config = Config(preset="sprite", denoise="median")
    rgb32 = np.asfortranarray(loaded.rgb.astype(np.float32))
    got = preprocess.run(rgb32, loaded.alpha.astype(np.int64), config)
    want = pyref.run(rgb32, loaded.alpha.astype(np.int64), config)
    assert np.array_equal(got.mask, want.mask) and np.array_equal(got.lab, want.lab)


@pytest.mark.skipif(not SAMPLES, reason="samples/ not available (set PIXELFORGE_SAMPLES)")
@pytest.mark.parametrize("path", SAMPLES, ids=[p.stem for p in SAMPLES])
def test_run_matches_reference_on_samples(path):
    loaded = io.load(path)
    for kwargs in (dict(preset="sprite"), dict(preset="background"),
                   dict(preset="sprite", enhance=1.0, ink=0.6)):
        _compare(loaded.rgb, loaded.alpha, kwargs, (path.name, kwargs))
