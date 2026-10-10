import json

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from pixelforge import Config, color, io, palette, quantize, run
from pixelforge.cli import cli
from pixelforge.pipeline import run_loaded


def _last_json(output: str) -> dict:
    return json.loads(output.strip().splitlines()[-1])


def test_cli_convert_sprite(fixture_path, tmp_path):
    result = CliRunner().invoke(cli, ["convert", str(fixture_path("circle_alpha")), "-o",
                                      str(tmp_path), "--preset", "sprite"])
    assert result.exit_code == 0, result.output
    payload = _last_json(result.output)
    names = ["circle_alpha.png", "circle_alpha_preview.png", "circle_alpha_palette.json",
             "circle_alpha_palette.hex", "circle_alpha_meta.json"]
    for name in names:
        assert (tmp_path / name).is_file(), name
    assert sorted(payload["outputs"].values()) == sorted(str(tmp_path / n) for n in names)

    config = Config(preset="sprite")
    text = io.read_png_text(tmp_path / "circle_alpha.png")
    assert text["pixelforge:config"] == config.canonical_json()
    assert text["pixelforge:input_sha256"] == io.sha256_file(fixture_path("circle_alpha"))

    rgba = np.asarray(Image.open(tmp_path / "circle_alpha.png").convert("RGBA"))
    opaque = rgba[rgba[..., 3] > 0][:, :3]
    assert 2 <= len(np.unique(opaque, axis=0)) <= config.palette_size + 1
    assert (rgba[..., 3] == 0).any()
    assert max(rgba.shape[:2]) == 64                # 64 px sprite, outline drawn inside

    preview = Image.open(tmp_path / "circle_alpha_preview.png")
    assert preview.size == (rgba.shape[1] * 8, rgba.shape[0] * 8)
    meta = json.loads((tmp_path / "circle_alpha_meta.json").read_text())
    assert meta["config_hash"] == config.hash() == payload["stats"]["config_hash"]
    palette = json.loads((tmp_path / "circle_alpha_palette.json").read_text())
    assert len(palette) == payload["stats"]["final_palette_size"]
    assert set(palette[0]) == {"hex", "lab", "used"}
    assert sum(e["used"] for e in palette) == meta["stats"]["colors_used"]
    assert meta["stats"]["colors_unused"] == len(palette) - meta["stats"]["colors_used"]


def test_cli_exit_codes(fixture_path, tmp_path):
    runner = CliRunner()
    bad = runner.invoke(cli, ["convert", str(fixture_path("two_color")), "-o", str(tmp_path),
                              "--palette-size", "1"])
    assert bad.exit_code == 2
    missing = runner.invoke(cli, ["convert", str(tmp_path / "nope.png"), "-o", str(tmp_path)])
    assert missing.exit_code == 1
    listing = runner.invoke(cli, ["palettes"])
    assert listing.exit_code == 0 and "pico8" in listing.output


def test_cli_config_file_and_flag_override(fixture_path, tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"method": "box", "palette_size": 4, "out_height": 16}))
    result = CliRunner().invoke(cli, ["convert", str(fixture_path("gradient6")), "-o",
                                      str(tmp_path), "--config", str(config_path),
                                      "--palette-size", "6", "--outline", "none"])
    assert result.exit_code == 0, result.output
    used = json.loads(io.read_png_text(tmp_path / "gradient6.png")["pixelforge:config"])
    assert (used["method"], used["palette_size"], used["out_height"]) == ("box", 6, 16)
    assert Image.open(tmp_path / "gradient6.png").size == (16, 16)


def test_cli_batch_shares_one_palette(fixture_path, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("gradient6", "noisy_gradient"):
        (frames / f"{name}.png").write_bytes(fixture_path(name).read_bytes())
    out = tmp_path / "out"
    result = CliRunner().invoke(cli, ["batch", str(frames), "-o", str(out), "--method", "box",
                                      "--out-height", "16", "--palette-size", "8",
                                      "--outline", "none"])
    assert result.exit_code == 0, result.output
    shared = (out / "shared_palette.hex").read_text().split()
    assert len(shared) == 8
    for name in ("gradient6", "noisy_gradient"):
        assert (out / f"{name}_palette.hex").read_text().split() == shared


def test_cli_compare(fixture_path, tmp_path):
    result = CliRunner().invoke(cli, ["compare", str(fixture_path("two_color")), "-o",
                                      str(tmp_path), "--out-height", "8", "--palette-size", "2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "two_color_compare.png").is_file()
    assert set(_last_json(result.output)["methods"]) == {"box", "kopf", "gerstner"}


def test_library_api(fixture_path, tmp_path):
    config = Config(preset="sprite", method="gerstner", palette_size=12, out_height=48)
    result = run(fixture_path("circle_alpha"), config)
    assert result.image.dtype == np.uint8 and result.image.shape[2] == 4
    assert result.palette.dtype == np.uint8 and result.palette.shape[1] == 3
    assert result.indices.shape == result.image.shape[:2] and result.indices.min() == -1
    assert set(result.stats) >= {"iterations", "timings", "final_palette_size", "config_hash"}
    paths = result.save(tmp_path / "out" / "hero")
    assert (tmp_path / "out" / "hero.png").is_file() and len(paths) == 5


def test_named_palette_and_dither(fixture_path):
    base = dict(preset="background", method="box", palette_name="pico8", out_width=32,
                out_height=32, tileset=False, remove_orphans=False)
    plain = run(fixture_path("gradient6"), Config(dither="none", **base))
    dithered = run(fixture_path("gradient6"), Config(dither="bayer4", dither_strength=1.0, **base))
    for result in (plain, dithered):
        # A named hardware palette is used as-is: no ramp shift, no saturation.
        assert [color.rgb8_to_hex(c)[1:] for c in result.palette] == \
            palette.parse_hex_lines("pico8")
        assert len(np.unique(result.indices)) >= 6
    assert (plain.indices != dithered.indices).any()
    assert plain.stats["final_palette_size"] == 16


def _gameboy_convert(src, outdir, palette_name="gameboy"):
    result = CliRunner().invoke(cli, ["convert", str(src), "-o", str(outdir), "--preset",
                                      "sprite", "--method", "box", "--out-height", "16",
                                      "--palette-name", str(palette_name)])
    assert result.exit_code == 0, result.output
    return _last_json(result.output)


def test_named_palette_outline_stays_in_palette(fixture_path, tmp_path):
    payload = _gameboy_convert(fixture_path("circle_alpha"), tmp_path)
    gameboy = ["#" + h for h in palette.parse_hex_lines("gameboy")]
    entries = json.loads((tmp_path / "circle_alpha_palette.json").read_text())
    assert [e["hex"] for e in entries] == gameboy
    rgba = np.asarray(Image.open(tmp_path / "circle_alpha.png").convert("RGBA"))
    used = {color.rgb8_to_hex(c) for c in rgba[rgba[..., 3] > 0][:, :3]}
    assert used and used <= set(gameboy)
    with Image.open(tmp_path / "circle_alpha.png") as png:
        indices = np.unique(np.asarray(png)[rgba[..., 3] > 0])
    assert all(0 <= i < len(gameboy) for i in indices.tolist())
    stats = payload["stats"]
    assert stats["final_palette_size"] == 4
    assert isinstance(stats["outline_index"], int) and 0 <= stats["outline_index"] < 4


def test_palette_hex_round_trip_is_stable(fixture_path, tmp_path):
    _gameboy_convert(fixture_path("circle_alpha"), tmp_path / "a")
    first = tmp_path / "a" / "circle_alpha_palette.hex"
    _gameboy_convert(fixture_path("circle_alpha"), tmp_path / "b", palette_name=first)
    assert (tmp_path / "b" / "circle_alpha_palette.hex").read_text() == first.read_text()


def test_outline_index_in_stats(fixture_path):
    base = dict(preset="sprite", method="box", out_height=16)
    result = run(fixture_path("circle_alpha"), Config(**base))
    assert result.stats["outline_index"] == len(result.palette) - 1
    json.dumps(result.stats)
    plain = run(fixture_path("circle_alpha"), Config(outline="none", **base))
    assert plain.stats["outline_index"] is None


def test_cli_rejects_non_finite_json_and_oversize_config(fixture_path, tmp_path):
    runner = CliRunner()
    image = str(fixture_path("two_color"))
    for literal in ("NaN", "Infinity", "-Infinity"):
        config_path = tmp_path / "cfg.json"
        config_path.write_text(f'{{"saturation_beta": {literal}}}')
        result = runner.invoke(cli, ["convert", image, "-o", str(tmp_path),
                                     "--config", str(config_path)])
        assert result.exit_code == 2 and literal in result.output, literal
    big = tmp_path / "big.json"
    big.write_text('{"palette_size": 4' + " " * (1024 * 1024) + "}")
    result = runner.invoke(cli, ["convert", image, "-o", str(tmp_path), "--config", str(big)])
    assert result.exit_code == 2 and "larger than" in result.output
    result = runner.invoke(cli, ["convert", image, "-o", str(tmp_path),
                                 "--config", str(tmp_path)])
    assert result.exit_code != 0


def test_cli_max_input_pixels(fixture_path, tmp_path):
    result = CliRunner().invoke(cli, ["convert", str(fixture_path("two_color")), "-o",
                                      str(tmp_path), "--max-input-pixels", "100"])
    assert result.exit_code == 1 and "input budget of 100 pixels" in result.output


GERSTNER_DITHER = dict(method="gerstner", out_width=24, out_height=24, palette_size=8,
                       denoise="none", key_bg=False, outline="none", remove_orphans=False,
                       fix_jaggies=False)


def _opaque(rgb: np.ndarray) -> io.Loaded:
    rgba = np.full(rgb.shape[:2] + (4,), 255, dtype=np.uint8)
    rgba[..., :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return io.from_rgba(rgba, "test")


def test_gerstner_dither_changes_smooth_gradient():
    y, x = np.mgrid[0:64, 0:64] / 63.0
    # Palette entries are far enough apart that dithering the palette colors themselves (the
    # old behavior) changed no pixel at all here.
    loaded = _opaque(np.stack([40 + 200 * x, 30 + 75 * (x + y), 20 + 100 * y], axis=-1))
    plain = run_loaded(loaded, Config(dither="none", **GERSTNER_DITHER))
    dithered = run_loaded(loaded, Config(dither="bayer4", dither_strength=1.0, **GERSTNER_DITHER))
    opaque = plain.indices >= 0
    assert np.array_equal(opaque, dithered.indices >= 0)
    assert (plain.indices != dithered.indices)[opaque].mean() > 0.05
    assert dithered.indices[opaque].min() >= 0
    assert dithered.indices.max() < len(dithered.palette)
    # Same palette (β applied exactly once); only the index image changes.
    assert np.array_equal(plain.palette, dithered.palette)


def test_gerstner_auto_dither_leaves_hard_edge_alone():
    rgb = np.empty((64, 64, 3))
    rgb[:, :32] = (200, 40, 40)
    rgb[:, 32:] = (40, 40, 200)
    loaded = _opaque(rgb)
    plain = run_loaded(loaded, Config(dither="none", **GERSTNER_DITHER))
    auto = run_loaded(loaded, Config(dither="auto", dither_strength=1.0, **GERSTNER_DITHER))
    edge = slice(10, 14)    # output columns on both sides of the red/blue boundary at x = 12
    assert len(np.unique(plain.indices[:, edge])) == 2
    assert np.array_equal(plain.indices[:, edge], auto.indices[:, edge])


def test_local_std_ignores_transparent_cells():
    lab = np.zeros((5, 5, 3))
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    lab[mask, 0] = 50.0               # flat opaque 3×3 block; transparent cells hold L = 0
    assert quantize.local_std(lab)[1, 1] > 20          # unmasked: the silhouette looks like detail
    std = quantize.local_std(lab, mask)
    assert std[mask].max() < 1e-3 and std[~mask].max() == 0.0
    lab[2, 2, 0] = 80.0               # real detail inside the opaque block is still seen
    assert quantize.local_std(lab, mask)[2, 2] > 5


def _opaque_square(size=64):
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = (200, 60, 30)
    rgba[size // 4:, :, :3] = (40, 120, 200)
    rgba[..., 3] = 255
    return io.from_rgba(rgba, "square")


def _has_outline_ring(result):
    ring = result.stats["outline_index"]
    border = np.concatenate([result.indices[0, 1:-1], result.indices[-1, 1:-1],
                             result.indices[1:-1, 0], result.indices[1:-1, -1]])
    return ring is not None and (border == ring).all()


def test_canvas_is_exactly_the_requested_size_with_outline():
    from pixelforge.pipeline import run_loaded
    base = dict(preset="sprite", method="box", key_bg=False)
    result = run_loaded(_opaque_square(), Config(out_width=32, out_height=32, **base))
    assert result.indices.shape == (32, 32)
    assert _has_outline_ring(result)
    assert (result.indices[1:-1, 1:-1] >= 0).all()
    assert result.stats["outline_margin"] == 1 and result.stats["outline_clipped"] is False
    only_height = run_loaded(_opaque_square(), Config(out_height=24, **base))
    assert only_height.indices.shape == (24, 24)
    assert _has_outline_ring(only_height)


def test_canvas_is_exactly_the_requested_size_without_outline():
    from pixelforge.pipeline import run_loaded
    base = dict(preset="sprite", method="box", key_bg=False, outline="none")
    result = run_loaded(_opaque_square(), Config(out_width=32, out_height=32, **base))
    assert result.indices.shape == (32, 32) and (result.indices >= 0).all()
    assert result.stats["outline_margin"] == 0 and result.stats["outline_index"] is None
    assert result.stats["outline_clipped"] is False


def test_default_sprite_longest_edge_is_exactly_64(fixture_path):
    result = run(fixture_path("circle_alpha"), Config(preset="sprite", method="box"))
    assert max(result.indices.shape) == 64
    assert result.stats["outline_margin"] == 1


def test_tileset_outline_is_clipped_not_padded():
    from pixelforge.pipeline import run_loaded
    config = Config(preset="sprite", method="box", key_bg=False, tileset=True, tile_size=8,
                    out_width=32, out_height=32)
    result = run_loaded(_opaque_square(), config)
    assert result.indices.shape == (32, 32)
    assert result.stats["outline_margin"] == 0 and result.stats["outline_clipped"] is True


def _sprite_rgba() -> np.ndarray:
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[3:13, 3:13] = (200, 60, 40, 255)
    rgba[5:9, 5:11] = (40, 90, 210, 255)
    rgba[10:12, 4:8] = (250, 240, 200, 255)
    return rgba


def _sprite_config(**overrides) -> Config:
    return Config(method="box", out_width=16, out_height=16, crop_to_alpha=False,
                  key_bg=False, remove_bg=False, denoise="none", dither="none",
                  outline="none", remove_orphans=False, fix_jaggies=False, **overrides)


def test_transparent_index_first(tmp_path):
    result = run_loaded(io.from_rgba(_sprite_rgba()), _sprite_config(transparent_index="first"))
    assert (result.indices == -1).any() and result.indices.min() == -1
    result.save(tmp_path / "s")
    with Image.open(tmp_path / "s.png") as im:
        assert im.mode == "P" and im.info["transparency"] == 0
        assert np.array_equal(np.asarray(im), result.indices + 1)
        assert np.array_equal(np.asarray(im.convert("RGBA")), result.image)
    entries = json.loads((tmp_path / "s_palette.json").read_text())
    assert entries[0] == {"hex": None, "transparent": True, "used": True}
    assert len(entries) == len(result.palette) + 1
    lines = (tmp_path / "s_palette.hex").read_text().splitlines()
    assert lines[0] == palette.TRANSPARENT_HEX_LINE and len(lines) == len(result.palette) + 1
    assert np.array_equal(palette.load_palette(str(tmp_path / "s_palette.hex")), result.palette)

    with pytest.raises(ValueError, match="transparent_index"):
        Config(transparent_index="middle")


def test_transparent_first_hex_round_trips_named_palette(fixture_path, tmp_path):
    runner = CliRunner()
    args = ["--method", "box", "--out-height", "16", "--transparent-index", "first"]
    first = runner.invoke(cli, ["convert", str(fixture_path("circle_alpha")), "-o",
                                str(tmp_path / "a"), "--palette-name", "gameboy", *args])
    assert first.exit_code == 0, first.output
    hex_path = tmp_path / "a" / "circle_alpha_palette.hex"
    assert hex_path.read_text().splitlines()[0] == "; transparent"
    assert len(palette.load_palette(str(hex_path))) == 4
    again = runner.invoke(cli, ["convert", str(fixture_path("circle_alpha")), "-o",
                                str(tmp_path / "b"), "--palette-name", str(hex_path), *args])
    assert again.exit_code == 0, again.output
    assert _last_json(again.output)["stats"]["final_palette_size"] == 4
    assert (tmp_path / "b" / "circle_alpha_palette.hex").read_text() == hex_path.read_text()


def test_unused_named_palette_entries_are_kept_and_flagged(fixture_path, tmp_path):
    config = Config(method="box", out_height=16, palette_name="pico8", dither="none")
    result = run(fixture_path("two_color"), config)
    result.save(tmp_path / "t")
    entries = json.loads((tmp_path / "t_palette.json").read_text())
    assert len(entries) == 16
    used = sorted(int(i) for i in np.unique(result.indices[result.indices >= 0]))
    assert [i for i, e in enumerate(entries) if e["used"]] == used
    meta = json.loads((tmp_path / "t_meta.json").read_text())
    assert meta["stats"]["colors_unused"] == 16 - len(used) > 0
