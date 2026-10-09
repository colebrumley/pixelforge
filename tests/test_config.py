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
