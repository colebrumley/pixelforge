import json

import numpy as np
from click.testing import CliRunner
from PIL import Image

from pixelforge import Config, color, io, palette, run
from pixelforge.cli import cli


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
    assert max(rgba.shape[:2]) in (64, 66)          # 64 px sprite (+1 px outline per side)

    preview = Image.open(tmp_path / "circle_alpha_preview.png")
    assert preview.size == (rgba.shape[1] * 8, rgba.shape[0] * 8)
    meta = json.loads((tmp_path / "circle_alpha_meta.json").read_text())
    assert meta["config_hash"] == config.hash() == payload["stats"]["config_hash"]
    palette = json.loads((tmp_path / "circle_alpha_palette.json").read_text())
    assert len(palette) == payload["stats"]["final_palette_size"]
    assert set(palette[0]) == {"hex", "lab"}


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
