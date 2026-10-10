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


@pytest.mark.parametrize("kwargs", [
    dict(scale=0), dict(scale=-1.5), dict(scale=4097), dict(scale=float("nan")),
    dict(canvas="64"), dict(canvas="64x"), dict(canvas="64X64"), dict(canvas=" 64x64"),
    dict(canvas="7x64"), dict(canvas="64x4097"), dict(canvas="６４x64"), dict(canvas=64),
    dict(scale=2, out_width=32), dict(canvas="64x64", out_height=64),
    dict(preset="background", canvas="40x32"),          # tileset: tile_size must divide it
])
def test_scale_and_canvas_validation(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_scale_and_canvas_accepted():
    cfg = Config(scale=2, canvas="48x32")
    assert cfg.scale == 2.0 and cfg.canvas_size == (48, 32)
    assert Config(scale=4096).scale == 4096 and Config(canvas="4096x8").canvas_size == (4096, 8)
    assert Config(preset="background", canvas="64x32").canvas_size == (64, 32)
    assert hash(cfg) == hash(Config(scale=2.0, canvas="48x32"))
    assert Config(**json.loads(cfg.canonical_json())) == cfg
    assert Config().canvas_size is None


def test_config_error_is_value_error_and_pixelforge_error():
    import pixelforge
    from pixelforge import ConfigError, PixelforgeError

    assert issubclass(ConfigError, ValueError) and issubclass(ConfigError, PixelforgeError)
    with pytest.raises(ConfigError, match="palette_size"):
        Config(palette_size=1)
    with pytest.raises(ValueError, match="unknown config field"):
        Config(nope=1)
    for name in ("Config", "run", "Result", "PixelforgeError", "ConfigError", "__version__"):
        assert name in pixelforge.__all__ and hasattr(pixelforge, name)


def test_negative_seed_rejected():
    from pixelforge import ConfigError

    with pytest.raises(ConfigError, match="seed must be >= 0"):
        Config(seed=-1)
