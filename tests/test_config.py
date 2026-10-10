import json

import pytest

from pixelforge import PRESETS, Config


def test_defaults_then_preset_then_explicit():
    sprite = Config()
    assert (sprite.preset, sprite.method, sprite.dither, sprite.outline) == \
        ("sprite", "gerstner", "none", "auto")
    background = Config(preset="background")
    for key, value in PRESETS["background"].items():
        assert getattr(background, key) == value
    assert Config(preset="background", method="box", dither="bayer8").method == "box"
    assert isinstance(sprite.remove_bg, bool)


def test_hash_is_stable_and_sensitive():
    a, b = Config(palette_size=12), Config(palette_size=12)
    assert a.hash() == b.hash() and len(a.hash()) == 64
    assert a.hash() != Config(palette_size=13).hash()
    assert Config(saturation_beta=1).hash() == Config(saturation_beta=1.0).hash()
    canonical = a.canonical_json()
    assert " " not in canonical and list(json.loads(canonical)) == sorted(json.loads(canonical))
    assert Config(**json.loads(canonical)) == a


@pytest.mark.parametrize("kwargs", [
    dict(out_width=4), dict(out_height=7), dict(palette_size=1), dict(palette_size=257),
    dict(method="nearest"), dict(g_alpha=1.0), dict(g_alpha=0.0), dict(preset="tiles"),
    dict(tileset=True, out_width=40, out_height=32), dict(outline="red"), dict(nonsense=1),
    dict(palette_size="16"), dict(dither="floyd"),
])
def test_validation_errors(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("name, limit", [
    ("out_width", 4096), ("out_height", 4096), ("scale_preview", 64),
    ("denoise_sigma_spatial", 16), ("kopf_max_iters", 1000), ("g_max_iters", 10000),
    ("saturation_beta", 5), ("tile_size", 512), ("seed", 2**63 - 1),
])
def test_upper_bounds(name, limit):
    Config(**{name: limit})
    with pytest.raises(ValueError, match=name):
        Config(**{name: limit + 1})


@pytest.mark.parametrize("kwargs", [
    dict(seed=-1), dict(dither_variance_threshold=-0.1), dict(tile_dedupe_tolerance=-1.0),
    dict(key_bg_tolerance=-0.01), dict(g_m=0.0), dict(kopf_tol=0.0), dict(seed=2**64),
    dict(orphan_min_region=-2**63 - 1),
])
def test_lower_and_range_bounds(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("name", ["saturation_beta", "g_m", "kopf_tol", "dither_strength",
                                  "dither_variance_threshold", "tile_dedupe_tolerance",
                                  "key_bg_tolerance", "g_T_final"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        Config(**{name: value})


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_values_for_int_fields_are_rejected(value):
    with pytest.raises(ValueError, match="kopf_max_iters"):
        Config(kopf_max_iters=value)


def test_outline_is_normalised_to_lowercase():
    upper, lower = Config(outline="#AABBCC"), Config(outline="#aabbcc")
    assert upper.outline == "#aabbcc" and upper.hash() == lower.hash()
    for bad in ("#aabbcc\n", "#aabbccdd", "aabbcc"):
        with pytest.raises(ValueError, match="outline"):
            Config(outline=bad)
