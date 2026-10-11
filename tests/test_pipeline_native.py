"""The native pipeline (crates/pixelforge/src/pipeline.rs through `pixelforge.pipeline`)
against the former Python orchestration (tests/pipeline_reference.py, over the same native
stage modules): every Result field bitwise, `stats` without `timings`, the warnings, and the
files `Result.save` writes (text files byte for byte, `_meta.json` without `environment` and
`stats.timings`, PNGs decoded: mode, size, palette, transparency, pixels and text)."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import make_fixtures   # scripts/ is on sys.path (conftest.py)
import numpy as np
import pytest
from PIL import Image

from pixelforge import Config, io, pipeline
from pixelforge.errors import ConfigError, PixelforgeError
from tests import pipeline_reference as reference

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "tests" / "oracle" / "outputs_small"
METHODS = ("box", "kopf", "gerstner")
FAST = dict(denoise="none", kopf_max_iters=10)


def _call(fn, *args):
    """(result, warning texts, exception) of fn(*args)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result, error = fn(*args), None
        except (ConfigError, PixelforgeError, ValueError) as exc:
            result, error = None, exc
    return result, [str(w.message) for w in caught], error


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _stats(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k != "timings"}


def _decoded(path: Path) -> dict:
    with Image.open(path) as im:
        im.load()
        return {"mode": im.mode, "size": im.size, "palette": im.getpalette(),
                "transparency": im.info.get("transparency"), "text": dict(im.text),
                "pixels": np.asarray(im).tobytes()}


def _meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    del meta["environment"], meta["stats"]["timings"]
    return meta


def _host_free(text: str) -> str:
    """_meta.json text without its `environment` and `stats.timings` objects (flat)."""
    out, n = re.subn(r'\n *"(environment|timings)": \{[^{}]*\},?', "", text)
    assert n == 2
    return out


def _compare_saved(got: dict, want: dict) -> None:
    assert list(got) == list(want)
    for name in want:
        a, b = Path(got[name]), Path(want[name])
        assert a.name == b.name
        if a.suffix == ".png":
            assert _decoded(a) == _decoded(b), name
        elif name == "meta":
            assert _meta(a) == _meta(b)
            # Same layout: the texts agree once the two host-dependent objects are cut out.
            assert _host_free(a.read_text()) == _host_free(b.read_text())
        else:
            assert a.read_bytes() == b.read_bytes(), name


def check(image, config: Config, tmp_path: Path, run_loaded: bool = False) -> None:
    if run_loaded:
        loaded = io.load(image) if isinstance(image, (str, Path)) else io.from_image(image)
        got, got_warn, got_err = _call(pipeline.run_loaded, loaded, config)
        want, want_warn, want_err = _call(reference.run_loaded, loaded, config)
    else:
        got, got_warn, got_err = _call(pipeline.run, image, config)
        want, want_warn, want_err = _call(reference.run, image, config)
    assert got_warn == want_warn
    if want_err is not None:
        assert type(got_err) is type(want_err) and str(got_err) == str(want_err)
        return
    assert got_err is None, got_err
    for name in ("image", "palette", "indices", "palette_lab", "tileset", "tileset_indices"):
        assert _same(getattr(got, name), getattr(want, name)), name
    assert got.tilemap == want.tilemap
    assert got.input_sha256 == want.input_sha256
    assert got.config is config
    assert _stats(got.stats) == _stats(want.stats)
    assert list(got.stats) == list(want.stats)                   # key order
    assert list(got.stats["timings"]) == list(want.stats["timings"])
    assert got.png_text() == want.png_text()
    saved = got.save(tmp_path / "native" / "out")
    expected = want.save(tmp_path / "python" / "out")
    _compare_saved(saved, expected)
    assert {k: Path(v) for k, v in saved.items()} == \
        pipeline.output_paths(tmp_path / "native" / "out", got.tilemap is not None)


def _fixture(name: str) -> Path:
    return ROOT / "tests" / "fixtures" / f"{name}.png"


def _size(preset: str) -> dict:
    return {"out_width": 32, "out_height": 32} if preset == "background" else {"out_width": 32}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("preset", ["sprite", "background"])
@pytest.mark.parametrize("name", make_fixtures.FIXTURES)
def test_fixtures(tmp_path, name, preset, method):
    check(_fixture(name), Config(preset=preset, method=method, **_size(preset), **FAST),
          tmp_path)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", ["circle_alpha", "noisy_gradient"])
def test_bilateral(tmp_path, name, method):
    check(_fixture(name), Config(preset="sprite", method=method, out_width=32,
                                 denoise="bilateral", kopf_max_iters=10), tmp_path)


@pytest.mark.parametrize("prefix", ["two_color_sprite", "checker_tiles_background"])
def test_oracle_configs(tmp_path, prefix):
    spec = json.loads((ORACLE / f"{prefix}_config.json").read_text())
    check(ROOT / spec["fixture"], Config(**spec["kwargs"]), tmp_path)


@pytest.mark.parametrize("method", METHODS)
def test_seamless_tileset(tmp_path, method):
    config = Config(preset="background", method=method, seamless=True, tileset=True,
                    tile_size=8, out_width=32, out_height=32, **FAST)
    check(_fixture("seamless_src"), config, tmp_path)


@pytest.mark.parametrize("preset", ["sprite", "background"])
def test_transparent_index_first(tmp_path, preset):
    config = Config(preset=preset, method="box", transparent_index="first", **_size(preset),
                    **FAST)
    check(_fixture("circle_alpha"), config, tmp_path)


@pytest.mark.parametrize("method", METHODS)
def test_named_palette(tmp_path, method):
    config = Config(preset="sprite", method=method, palette_name="pico8", out_width=32,
                    dither="bayer4", **FAST)
    check(_fixture("gradient6"), config, tmp_path)


def test_inline_palette(tmp_path):
    config = Config(preset="background", method="kopf", out_width=32, out_height=32,
                    palette_name="hex:000000,ffffff,ff0000,00ff00", **FAST)
    check(_fixture("gradient6"), config, tmp_path)


def test_palette_file(tmp_path):
    path = tmp_path / "four.hex"
    path.write_text("101010\nf0f0f0\nc03030\n3030c0\n")
    config = Config(preset="sprite", method="box", out_width=32, palette_name=str(path),
                    **FAST)
    config.hash()                               # the Python Config caches the palette lines
    check(_fixture("circle_alpha"), config, tmp_path)


@pytest.mark.parametrize("fit", ["crop", "pad", "stretch"])
def test_fit(tmp_path, fit):
    config = Config(preset="background", method="box", fit=fit, tile_size=16, out_height=32,
                    **FAST)
    check(_fixture("gradient6"), config, tmp_path)


@pytest.mark.parametrize("method", METHODS)
def test_canvas(tmp_path, method):
    config = Config(preset="sprite", method=method, canvas="40x24", **FAST)
    check(_fixture("circle_alpha"), config, tmp_path)


@pytest.mark.parametrize("dither", ["bayer8", "auto"])
def test_gerstner_dither(tmp_path, dither):
    config = Config(preset="sprite", method="gerstner", dither=dither, dither_strength=1.0,
                    out_width=32, **FAST)
    check(_fixture("gradient6"), config, tmp_path)


def test_in_memory_inputs(tmp_path):
    rgba = make_fixtures.circle_alpha()[::2, ::2]
    config = Config(preset="sprite", method="box", out_width=24, **FAST)
    check(rgba, config, tmp_path / "u8")
    check(rgba.astype(np.float64) / 255.0, config, tmp_path / "f64")
    check(rgba[..., :3].copy(), config, tmp_path / "rgb")
    check(Image.fromarray(rgba, mode="RGBA"), config, tmp_path / "pil")
    check(rgba, config, tmp_path / "loaded", run_loaded=True)
    check(_fixture("two_color"), config, tmp_path / "loaded_path", run_loaded=True)


def test_errors_match(tmp_path):
    # No opaque pixel left: the same PixelforgeError from both.
    clear = np.zeros((16, 16, 4), dtype=np.uint8)
    check(clear, Config(preset="sprite", method="box", out_width=8, **FAST), tmp_path)
    with pytest.raises(PixelforgeError, match="input budget"):
        pipeline.run(_fixture("two_color"), Config(**FAST), max_pixels=10)
    with pytest.raises(FileNotFoundError):
        pipeline.run(tmp_path / "missing.png", Config(**FAST))


def test_environment_and_paths(tmp_path):
    env = pipeline.environment()
    assert set(env) == {"python", "numpy", "scipy", "scikit_image", "pillow", "platform",
                        "threads"}
    assert env["platform"] == reference.environment()["platform"]
    native = pipeline._native.environment()
    assert set(native) == {"platform", "threads", "native"}
    assert native["native"] == pipeline._native.native_version()
    for prefix in ("out/hero", "./hero", "a//./b/", tmp_path / "x"):
        for tileset in (False, True):
            assert pipeline.output_paths(prefix, tileset) == \
                reference.output_paths(prefix, tileset)
    for kwargs in ({}, {"method": "gerstner"}, {"method": "gerstner", "palette_name": "pico8"},
                   {"method": "gerstner", "palette_source": "median_cut"}, {"method": "box"}):
        config = Config(**kwargs)
        assert pipeline.uses_gerstner_palette(config) == reference.uses_gerstner_palette(config)
