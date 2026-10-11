"""Smoke tests of the Python binding (the library itself is tested by `cargo test`)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pixelforge
from pixelforge import Config, ConfigError, PixelforgeError, run

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "two_color.png"
FAST = {"method": "box", "out_width": 16, "out_height": 16}
FAST_FLAGS = ["--method", "box", "--out-width", "16", "--out-height", "16"]
SAVED = ("image", "preview", "palette", "palette_hex", "meta")


def _binary() -> Path | None:
    path = Path(os.environ.get("PIXELFORGE_BIN") or ROOT / "target" / "release" / "pixelforge")
    return path if path.is_file() else None


def test_import_and_version():
    assert isinstance(pixelforge.__version__, str) and pixelforge.__version__.count(".") == 2
    assert pixelforge.thread_count() >= 1
    assert issubclass(ConfigError, PixelforgeError) and issubclass(ConfigError, ValueError)


def test_config_defaults_presets_and_api():
    c = Config()
    assert c.preset == "sprite" and c.method == "gerstner" and c.palette_size == 16
    assert c.key_bg is True                   # the sprite preset's override
    assert Config(preset="background").key_bg is False
    assert set(pixelforge.PRESETS) == {"sprite", "background"}
    with pytest.raises(TypeError):
        pixelforge.PRESETS["sprite"]["key_bg"] = False
    assert list(c.to_dict()) == Config.field_names()
    assert json.loads(c.canonical_json()) == c.to_dict()
    assert len(c.hash()) == 64 and c.hash() == Config().hash()
    assert c == Config() and hash(c) == hash(Config()) and c != Config(palette_size=8)
    assert c.replace(preset="background") == Config(preset="background")
    assert c.provenance()["version"] == pixelforge.__version__
    assert Config(canvas="40x30").canvas_size == (40, 30)
    with pytest.raises(AttributeError):
        c.palette_size = 8
    Config(palette_name="nes").validate_palette()


def test_config_validation_error_message():
    with pytest.raises(ConfigError, match=r"^unknown config field\(s\): bogus$"):
        Config(bogus=1)
    with pytest.raises(ValueError, match="preset must be one of"):
        Config(preset="tiny")
    with pytest.raises(ConfigError):
        Config(palette_name="missing.hex").validate_palette()


def test_run_and_save(tmp_path):
    res = run(FIXTURE, Config(**FAST))
    assert res.image.shape == (16, 16, 4) and res.image.dtype == np.uint8
    assert res.indices.shape == (16, 16) and res.indices.dtype == np.int64
    assert res.palette.shape[1] == 3 and res.palette_lab.shape == res.palette.shape
    assert "timings" in res.stats and res.tileset is None and res.tilemap is None
    paths = res.save(tmp_path / "out")
    assert tuple(paths) == SAVED
    assert all(Path(p).is_file() for p in paths.values())
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["out.png", "out_preview.png", "out_palette.json", "out_palette.hex", "out_meta.json"])
    meta = json.loads(Path(paths["meta"]).read_text())
    assert meta["config_hash"] == res.config.hash() == res.stats["config_hash"]


def test_config_hash_and_png_match_the_cli(tmp_path):
    binary = _binary()
    if binary is None:
        pytest.skip("no target/release/pixelforge (cargo build --release -p pixelforge)")
    subprocess.run([binary, "convert", FIXTURE, *FAST_FLAGS, "-o", tmp_path / "cli"],
                   check=True, capture_output=True)
    paths = run(FIXTURE, Config(**FAST)).save(tmp_path / "py" / "two_color")
    cli_meta = json.loads((tmp_path / "cli" / "two_color_meta.json").read_text())
    py_meta = json.loads(Path(paths["meta"]).read_text())
    assert py_meta["config_hash"] == cli_meta["config_hash"]
    assert Path(paths["image"]).read_bytes() == (tmp_path / "cli" / "two_color.png").read_bytes()


def test_numpy_array_input():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[:, 16:, :3] = (200, 40, 40)
    res = run(rgba, Config(**FAST, denoise="none"))
    assert res.image.shape == (16, 16, 4)
    assert len(res.input_sha256) == 64
    same = run(rgba.astype(np.float64) / 255.0, Config(**FAST, denoise="none"))
    assert same.input_sha256 == res.input_sha256
    np.testing.assert_array_equal(same.indices, res.indices)
    with pytest.raises(PixelforgeError):
        run(np.zeros((4, 4), dtype=np.uint8), Config(**FAST))


def test_threads_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXELFORGE_THREADS", "3")
    assert pixelforge.thread_count() == 3
    paths = run(FIXTURE, Config(**FAST)).save(tmp_path / "t")
    meta = json.loads(Path(paths["meta"]).read_text())
    assert meta["environment"]["threads"] == 3
