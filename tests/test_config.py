import dataclasses
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


def test_replace_reapplies_changed_preset():
    background = Config(preset="sprite").replace(preset="background")
    assert background == Config(preset="background")
    for key in ("method", "tileset", "outline", "palette_size"):
        assert getattr(background, key) == PRESETS["background"][key]
    kept = Config(preset="sprite", palette_size=12).replace(preset="background")
    assert kept.palette_size == 12 and kept.method == "kopf" and kept.tileset
    # Fields set through replace() are explicit in the result and survive a later preset change.
    chained = Config().replace(method="box").replace(preset="background")
    assert chained.method == "box" and chained.tileset


def test_replace_keeps_everything_else():
    cfg = Config(preset="background", palette_size=12, dither="bayer8", seed=3)
    boxed = cfg.replace(method="box")
    assert boxed.method == "box"
    assert {k: v for k, v in boxed.to_dict().items() if k != "method"} == \
        {k: v for k, v in cfg.to_dict().items() if k != "method"}


def test_round_trip_hashable_and_frozen():
    cfg = Config(preset="background", palette_size=12)
    again = Config(**cfg.to_dict())
    assert again == cfg and hash(again) == hash(cfg) and again.hash() == cfg.hash()
    assert "_explicit" not in cfg.to_dict() and "_explicit" not in cfg.canonical_json()
    assert "_explicit" not in Config.field_names()
    assert len({cfg, again, Config()}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.method = "box"


def test_cli_warns_when_json_shadows_preset(tmp_path, capsys):
    from pixelforge.cli import build_config

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"preset": "sprite", "method": "gerstner", "palette_size": 16}))
    config = build_config(str(path), {"preset": "background", "palette_size": 8})
    assert (config.preset, config.method, config.palette_size) == ("background", "gerstner", 8)
    err = capsys.readouterr().err
    assert "warning" in err and "method" in err and "palette_size" not in err
    build_config(str(path), {"palette_size": 8})
    assert capsys.readouterr().err == ""
