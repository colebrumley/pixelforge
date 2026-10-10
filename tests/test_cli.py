import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from pixelforge import io
from pixelforge.cli import cli

FAST = ["--method", "box", "--out-width", "16", "--out-height", "16"]


def _invoke(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_convert_into_input_dir_refused(fixture_path, tmp_path):
    src = tmp_path / "two_color.png"
    shutil.copy(fixture_path("two_color"), src)
    before = src.read_bytes()
    result = _invoke("convert", src, "-o", tmp_path, *FAST)
    assert result.exit_code == 2, result.output
    assert "would overwrite input" in result.output and str(src) in result.output
    assert src.read_bytes() == before
    assert not (tmp_path / "two_color_meta.json").exists()


def test_convert_force_overwrites(fixture_path, tmp_path):
    src = tmp_path / "two_color.png"
    shutil.copy(fixture_path("two_color"), src)
    before = src.read_bytes()
    result = _invoke("convert", src, "-o", tmp_path, "--force", *FAST)
    assert result.exit_code == 0, result.output
    assert src.read_bytes() != before
    assert (tmp_path / "two_color_meta.json").is_file()


def test_batch_into_input_dir_refused(fixture_path, tmp_path):
    shutil.copy(fixture_path("two_color"), tmp_path / "a.png")
    result = _invoke("batch", tmp_path, "-o", tmp_path, *FAST)
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "shared_palette.hex").exists()


def test_batch_stem_collision_refused(fixture_path, tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    shutil.copy(fixture_path("two_color"), src / "a.png")
    Image.open(fixture_path("two_color")).convert("RGB").save(src / "a.bmp")
    result = _invoke("batch", src, "-o", out, *FAST)
    assert result.exit_code == 2, result.output
    assert str(src / "a.bmp") in result.output and str(src / "a.png") in result.output
    assert not out.exists()
    result = _invoke("batch", src, "-o", out, "--force", *FAST)
    assert result.exit_code == 0, result.output
    assert (out / "a.png").is_file()


def test_batch_skips_own_outputs(fixture_path, tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    shutil.copy(fixture_path("two_color"), src / "a.png")
    shutil.copy(fixture_path("circle_alpha"), src / "a_preview.png")
    result = _invoke("batch", src, "-o", out, *FAST)
    assert result.exit_code == 0, result.output
    assert f"skipping {src / 'a_preview.png'}" in result.output
    assert not (out / "a_preview_preview.png").exists()
    assert (out / "a.png").is_file()


def _frame(path, size=(64, 64), radius=None, rect=None):
    """An RGBA frame with an opaque disk (radius) or rectangle (w, h) centered on it."""
    w, h = size
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    if radius is not None:
        opaque = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    else:
        opaque = (abs(xx - cx) < rect[0] / 2) & (abs(yy - cy) < rect[1] / 2)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = (200, 60, 40)
    rgba[..., 1] = np.linspace(40, 200, w).astype(np.uint8)[None, :]
    rgba[..., 3] = opaque * 255
    Image.fromarray(rgba, mode="RGBA").save(path)


def _opaque_box(path):
    """(height, width, y0, x0) of the opaque pixels of an output PNG."""
    alpha = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha)
    return ys.max() - ys.min() + 1, xs.max() - xs.min() + 1, ys.min(), xs.min()


def _batch(tmp_path, *args):
    result = _invoke("batch", tmp_path / "in", "-o", tmp_path / "out", "--method", "box",
                     "--palette-size", "4", *args)
    assert result.exit_code == 0, result.output
    return [json.loads(line) for line in result.output.splitlines()]


def test_batch_frames_share_one_scale_and_canvas(tmp_path):
    (tmp_path / "in").mkdir()
    _frame(tmp_path / "in" / "big.png", radius=24)
    _frame(tmp_path / "in" / "small.png", radius=12)
    lines = _batch(tmp_path)
    assert lines[0]["canvas"] == "64x64" and lines[0]["scale"] > 0
    assert "shared_palette" in lines[0]
    big, small = (Image.open(tmp_path / "out" / f"{n}.png") for n in ("big", "small"))
    assert big.size == small.size == (64, 64)
    bh, bw, _, _ = _opaque_box(tmp_path / "out" / "big.png")
    sh, sw, _, _ = _opaque_box(tmp_path / "out" / "small.png")
    assert abs(bh - 2 * sh) <= 2 and abs(bw - 2 * sw) <= 2
    for line in lines[1:]:
        meta = json.loads(Path(line["outputs"]["meta"]).read_text())
        assert (meta["config"]["scale"], meta["config"]["canvas"]) == \
            (lines[0]["scale"], lines[0]["canvas"])


def test_batch_wide_frame_sets_the_shared_scale(tmp_path):
    (tmp_path / "in").mkdir()
    _frame(tmp_path / "in" / "a_wide.png", size=(128, 64), rect=(112, 56))
    _frame(tmp_path / "in" / "b.png", rect=(56, 56))
    _frame(tmp_path / "in" / "c.png", rect=(56, 28))
    lines = _batch(tmp_path)
    assert lines[0]["scale"] == pytest.approx(114 / 62)      # 112 + 1 px crop ring per side
    sizes = {Image.open(tmp_path / "out" / f"{n}.png").size for n in ("a_wide", "b", "c")}
    assert sizes == {(64, 34)}                              # round(58 / scale) + 2
    # Subject sizes without the 1-px outline ring on each side.
    wide, square, flat = ([v - 2 for v in _opaque_box(tmp_path / "out" / f"{n}.png")[:2]]
                          for n in ("a_wide", "b", "c"))
    assert abs(wide[1] - 2 * square[1]) <= 2 and abs(square[0] - wide[0]) <= 1
    assert abs(square[0] - 2 * flat[0]) <= 2 and square[1] == flat[1]


def test_batch_explicit_scale_is_used_as_given(tmp_path):
    (tmp_path / "in").mkdir()
    _frame(tmp_path / "in" / "a.png", radius=24)
    lines = _batch(tmp_path, "--scale", "2", "--no-crop-to-alpha")
    assert set(lines[0]) == {"shared_palette"}
    assert Image.open(tmp_path / "out" / "a.png").size == (32, 32)


def test_canvas_pads_and_centers(tmp_path):
    (tmp_path / "in").mkdir()
    _frame(tmp_path / "in" / "a.png", rect=(40, 20))
    _batch(tmp_path, "--canvas", "48x48")
    out = tmp_path / "out" / "a.png"
    assert Image.open(out).size == (48, 48)
    h, w, y0, x0 = _opaque_box(out)
    assert w >= 46 and h < 30
    assert abs(y0 - (48 - (y0 + h))) <= 1 and abs(x0 - (48 - (x0 + w))) <= 1


def test_remove_bg_without_rembg_is_config_error(fixture_path, tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_rembg(name, *args, **kwargs):
        if name == "rembg":
            raise ImportError("no rembg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rembg)
    result = _invoke("convert", fixture_path("two_color"), "-o", tmp_path, "--remove-bg", *FAST)
    assert result.exit_code == 2, result.output
    assert "config error" in result.output and "rembg" in result.output


def test_runtime_error_exit_code_1(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    result = _invoke("convert", bad, "-o", tmp_path / "out", *FAST)
    assert result.exit_code == 1, result.output
    assert "not a supported image" in result.output


def test_debug_prints_traceback(tmp_path, monkeypatch):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    monkeypatch.delenv("PIXELFORGE_DEBUG", raising=False)
    plain = _invoke("convert", bad, "-o", tmp_path / "out", *FAST)
    assert plain.exit_code == 1 and "Traceback" not in plain.output
    debug = _invoke("--debug", "convert", bad, "-o", tmp_path / "out", *FAST)
    assert debug.exit_code == 1, debug.output
    assert "Traceback" in debug.output and "not a supported image" in debug.output
    config = _invoke("--debug", "convert", bad, "-o", tmp_path / "out", "--palette-size", "1")
    assert config.exit_code == 2 and "Traceback" in config.output
    monkeypatch.setenv("PIXELFORGE_DEBUG", "1")
    env = _invoke("convert", bad, "-o", tmp_path / "out", *FAST)
    assert env.exit_code == 1 and "Traceback" in env.output


def test_batch_metadata_does_not_depend_on_outdir(tmp_path):
    (tmp_path / "in").mkdir()
    _frame(tmp_path / "in" / "a.png", radius=12)
    runs = []
    for name in ("one", "two"):
        out = tmp_path / name / "deeper"
        result = _invoke("batch", tmp_path / "in", "-o", out, *FAST, "--palette-size", "4")
        assert result.exit_code == 0, result.output
        text = io.read_png_text(out / "a.png")
        assert str(out) not in json.dumps(text)
        meta = json.loads((out / "a_meta.json").read_text())
        assert meta["palette"] == (out / "shared_palette.hex").read_text().split()
        assert text["pixelforge:palette"].split("\n") == meta["palette"]
        assert text["pixelforge:palette_sha256"] == meta["palette_sha256"]
        runs.append((text, (out / "a.png").read_bytes(), meta["config_hash"]))
    assert runs[0] == runs[1]
