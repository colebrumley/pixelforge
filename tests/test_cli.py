import shutil

from click.testing import CliRunner
from PIL import Image

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
