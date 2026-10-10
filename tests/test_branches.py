"""Branches the module tests do not reach on their own: palette sources, PNG fallbacks,
denoisers, optional dependencies, seamless gerstner, empty batches, dithering, tile sizes."""

import importlib.util

import make_fixtures   # scripts/ is on sys.path (conftest.py)
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from pixelforge import Config, ConfigError, color, io, palette, preprocess, run, tiles
from pixelforge.cli import cli

FAST = dict(denoise="none", key_bg=False, out_width=16, out_height=16)
RAW = dict(outline="none", remove_orphans=False, fix_jaggies=False)   # no post-processing


def _small(name: str, step: int) -> np.ndarray:
    """A fixture subsampled to at most 64 px (uint8 RGB or RGBA array)."""
    image = getattr(make_fixtures, name)()[::step, ::step]
    assert max(image.shape[:2]) <= 64
    return image


@pytest.mark.parametrize("method", ["box", "kopf"])
def test_mcda_palette_source_with_box_and_kopf(method, monkeypatch):
    calls = []
    real = palette.mcda

    def spy(pixels, k, config):
        calls.append(k)
        return real(pixels, k, config)

    monkeypatch.setattr(palette, "mcda", spy)
    image = _small("gradient6", 4)
    config = Config(method=method, palette_source="mcda", palette_size=6, kopf_max_iters=10,
                    **FAST, **RAW)
    result = run(image, config)
    assert calls == [6]
    assert 2 <= len(result.palette) <= 6 and result.indices.shape == (16, 16)
    assert (result.indices >= 0).all()
    median = run(image, config.replace(palette_source="median_cut"))
    assert len(calls) == 1                     # median_cut does not go through mcda
    assert not np.array_equal(median.palette, result.palette)


@pytest.mark.parametrize("n_colors, mode", [(255, "P"), (256, "RGBA")])
def test_save_png_falls_back_to_rgba_at_256_colors_plus_transparency(tmp_path, n_colors,
                                                                    mode):
    palette_rgb8 = np.array([[i, 255 - i, (7 * i) % 256] for i in range(n_colors)],
                            dtype=np.uint8)
    indices = np.arange(16 * 16).reshape(16, 16) % n_colors
    indices[0, :4] = -1                                          # some transparency
    path = tmp_path / "out.png"
    io.save_png(path, indices, palette_rgb8, {"k": "v"})
    with Image.open(path) as im:
        assert im.mode == mode
        assert im.text == {"k": "v"}
        decoded = np.asarray(im.convert("RGBA"))
    expected = io.indices_to_rgba(indices, palette_rgb8)
    assert np.array_equal(decoded[..., 3], expected[..., 3])
    opaque = expected[..., 3] == 255
    assert np.array_equal(decoded[opaque], expected[opaque])
    # Without transparency 256 colors still fit a palette PNG.
    io.save_png(path, np.abs(indices), palette_rgb8)
    with Image.open(path) as im:
        assert im.mode == "P"


def test_median_denoise():
    image = _small("noisy_gradient", 4)
    loaded = io.from_image(image)
    none = preprocess.run(loaded.rgb, loaded.alpha, Config(**FAST))
    median = preprocess.run(loaded.rgb, loaded.alpha, Config(**{**FAST, "denoise": "median"}))
    assert median.lab.shape == none.lab.shape
    assert not np.allclose(median.lab, none.lab)
    # A 3×3 median removes a single outlier pixel.
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    flat[10, 10] = 255
    spot = io.from_image(flat)
    out = preprocess.run(spot.rgb, spot.alpha, Config(**{**FAST, "denoise": "median"}))
    assert np.ptp(out.lab[..., 0]) < 1e-9
    result = run(image, Config(method="box", **{**FAST, "denoise": "median"}))
    assert result.indices.shape == (16, 16)


def test_remove_bg_missing_dependency_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_rembg(name, *args, **kwargs):
        if name == "rembg":
            raise ImportError("No module named 'rembg'")
        return real_import(name, *args, **kwargs)

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(builtins, "__import__", no_rembg)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "rembg" else real_find_spec(name, *a))
    from pixelforge.config import rembg_available

    assert rembg_available() is False
    rgb, alpha = np.full((8, 8, 3), 0.5), np.full((8, 8), 255, dtype=np.uint8)
    with pytest.raises(ConfigError, match=r"--remove-bg requires the optional 'rembg' "
                                          r"dependency; install it with: pip install "
                                          r"'pixelforge\[bg\]'"):
        preprocess.remove_background(rgb, alpha)
    with pytest.raises(ConfigError, match="rembg"):
        run(np.zeros((16, 16, 3), dtype=np.uint8), Config(remove_bg=True, **FAST))


def test_seamless_with_gerstner():
    image = _small("seamless_src", 2)                            # 64 × 64, period 64
    config = Config(preset="background", method="gerstner", tileset=False, seamless=True,
                    denoise="none", out_width=16, out_height=16)
    result = run(image, config)
    assert result.indices.shape == (16, 16) and (result.indices >= 0).all()
    lab = result.palette_lab[result.indices]
    interior = color.delta_e(lab[:, 1:-2], lab[:, 2:-1]).mean()
    assert interior > 0
    assert color.delta_e(lab[:, 0], lab[:, -1]).mean() < 3 * interior
    assert color.delta_e(lab[0, :], lab[-1, :]).mean() < 3 * interior
    plain = run(image, config.replace(seamless=False))
    assert plain.indices.shape == (16, 16)


def test_batch_on_empty_directory(tmp_path):
    empty = tmp_path / "frames"
    empty.mkdir()
    (empty / "notes.txt").write_text("not an image")
    result = CliRunner().invoke(cli, ["batch", str(empty), "-o", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert f"no images found in {empty}" in result.output
    assert not (tmp_path / "out").exists()


def _gray_ramp() -> np.ndarray:
    ramp = np.round(np.linspace(30, 220, 64)).astype(np.uint8)
    return np.repeat(np.tile(ramp, (64, 1))[..., None], 3, axis=2)


def _dither_config(**overrides):
    return Config(method="box", palette_size=4, palette_source="median_cut", **FAST,
                  **RAW, **overrides)


def test_bayer8_dither():
    image = _gray_ramp()
    plain = run(image, _dither_config(dither="none"))
    bayer8 = run(image, _dither_config(dither="bayer8", dither_strength=1.0))
    assert np.array_equal(plain.palette, bayer8.palette)       # dithering only re-indexes
    assert not np.array_equal(plain.indices, bayer8.indices)
    from pixelforge import quantize

    offsets = quantize.dither_offsets(np.zeros((16, 16, 3)), _dither_config(dither="bayer8"))
    assert np.array_equal(offsets[:8, :8], offsets[8:, 8:])     # period 8
    assert not np.array_equal(offsets[:4, :4], offsets[4:8, 4:8])
    assert len(np.unique(offsets[:8, :8, 0])) == 64


@pytest.mark.parametrize("dither", ["bayer4", "bayer8", "auto"])
def test_zero_dither_strength_is_no_dither(dither):
    image = _gray_ramp()
    plain = run(image, _dither_config(dither="none"))
    zero = run(image, _dither_config(dither=dither, dither_strength=0.0))
    assert np.array_equal(plain.indices, zero.indices)
    assert np.array_equal(plain.palette, zero.palette)


@pytest.mark.parametrize("shape", [(16, 24), (24, 16), (20, 20)])
def test_tiles_extract_rejects_sizes_not_divisible_by_the_tile(shape):
    idx = np.zeros(shape, dtype=np.int64)
    with pytest.raises(ValueError, match=rf"tile_size \(16\) must divide the output size "
                                         rf"\({shape[1]}x{shape[0]}\)"):
        tiles.extract(idx, np.zeros((1, 3)), Config(tile_size=16))
