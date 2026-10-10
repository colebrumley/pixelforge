import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from pixelforge import Config
from pixelforge.cli import cli

METHODS = ("box", "kopf", "gerstner")
# Preset flags. The background preset is run at 64×64 (4 × 4 tiles) to keep the suite quick.
PRESETS = {"sprite": [], "background": ["--out-width", "64", "--out-height", "64"]}
FIXTURE_NAMES = ("line_diag", "circle_alpha", "noisy_gradient")   # fixtures 1, 4, 6


def _convert(path, outdir, preset, method):
    args = ["convert", str(path), "-o", str(outdir), "--preset", preset, "--method", method]
    result = CliRunner().invoke(cli, args + PRESETS[preset])
    assert result.exit_code == 0, result.output
    return {p.name: p.read_bytes() for p in sorted(outdir.glob("*.png"))}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_cli_output_is_byte_identical(fixture_path, tmp_path, name, preset, method):
    first = _convert(fixture_path(name), tmp_path / "a", preset, method)
    second = _convert(fixture_path(name), tmp_path / "b", preset, method)
    expected = {f"{name}.png", f"{name}_preview.png"}
    if preset == "background":
        expected.add(f"{name}_tileset.png")
    assert set(first) == expected
    assert first == second


def test_config_hash_is_stable_across_runs():
    for preset in PRESETS:
        hashes = {Config(preset=preset, method=m).hash() for m in METHODS for _ in range(3)}
        assert len(hashes) == len(METHODS)
    assert Config().hash() == Config(**Config().to_dict()).hash()


def _decoded(png_path):
    """(mode, palette, pixels) of a PNG; the text chunks (which embed the config) excluded."""
    from PIL import Image

    with Image.open(png_path) as im:
        return im.mode, im.getpalette(), im.info.get("transparency"), im.tobytes()


def _convert_with_seed(path, outdir, method, seed):
    args = ["convert", str(path), "-o", str(outdir), "--method", method, "--out-height", "16",
            "--palette-size", "4", "--denoise", "none"]
    if seed is not None:
        args += ["--seed", str(seed)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    stem = Path(path).stem
    return {"png": _decoded(outdir / f"{stem}.png"),
            "preview": _decoded(outdir / f"{stem}_preview.png"),
            "palette": (outdir / f"{stem}_palette.json").read_bytes(),
            "palette_hex": (outdir / f"{stem}_palette.hex").read_bytes()}


@pytest.mark.parametrize("method", METHODS)
def test_seed_only_affects_gerstner(fixture_path, tmp_path, method, record_property):
    """seed feeds only gerstner's optional MCDA jitter: box and kopf ignore it entirely.

    The PNG bytes differ by the embedded config (seed is a field), so pixels, palette and
    transparency are compared after decoding, plus the palette files byte for byte.
    """
    path = fixture_path("gradient6")
    seeded = _convert_with_seed(path, tmp_path / "seeded", method, 7)
    unseeded = _convert_with_seed(path, tmp_path / "unseeded", method, None)
    if method == "gerstner":
        # The jitter may or may not change the result; only record which.
        record_property("gerstner_seed_changes_output", seeded != unseeded)
    else:
        assert seeded == unseeded
        again = _convert_with_seed(path, tmp_path / "again", method, 12345)
        assert again == unseeded


# --- across process boundaries ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
_ENVIRONMENTS = (
    {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
     "MKL_NUM_THREADS": "1"},
    {"PYTHONHASHSEED": "12345", "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
     "MKL_NUM_THREADS": "4"},
)
# Sprite (no tileset) and background (tileset + tilemap .json/.tmj/.csv), both at 16 px.
_SUBPROCESS_PRESETS = {"sprite": ["--out-width", "16"],
                       "background": ["--out-width", "16", "--out-height", "16",
                                      "--tile-size", "8"]}


def _convert_in_subprocess(path, outdir, method, preset, env_overrides):
    env = dict(os.environ, **env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    args = [sys.executable, "-m", "pixelforge", "convert", str(path), "-o", str(outdir),
            "--method", method, "--preset", preset, *_SUBPROCESS_PRESETS[preset]]
    done = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    outputs = {}
    for p in sorted(outdir.iterdir()):
        data = p.read_bytes()
        if p.name.endswith("_meta.json"):
            meta = json.loads(data)
            del meta["stats"]["timings"], meta["environment"]   # wall clock and host only
            data = json.dumps(meta, sort_keys=True).encode()
        outputs[p.name] = data
    return outputs


@pytest.mark.parametrize("preset", sorted(_SUBPROCESS_PRESETS))
@pytest.mark.parametrize("method", METHODS)
def test_outputs_identical_across_processes_and_environments(fixture_path, tmp_path, method,
                                                              preset):
    """Fresh interpreters with different hash seeds and BLAS/OpenMP thread counts agree on
    every output file, apart from the timings and environment recorded in _meta.json."""
    runs = [_convert_in_subprocess(fixture_path("two_color"), tmp_path / str(i), method,
                                   preset, env)
            for i, env in enumerate(_ENVIRONMENTS)]
    expected = {"two_color.png", "two_color_preview.png", "two_color_palette.json",
                "two_color_palette.hex", "two_color_meta.json"}
    if preset == "background":
        expected |= {"two_color_tileset.png", "two_color_tilemap.json", "two_color.tmj",
                     "two_color_tilemap.csv"}
    assert set(runs[0]) == expected and set(runs[1]) == expected
    for name in sorted(expected):
        assert runs[0][name] == runs[1][name], f"{name} differs between processes"
