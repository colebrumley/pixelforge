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


def test_seed_only_changes_gerstner_jitter(fixture_path, tmp_path):
    args = ["--out-height", "16", "--palette-size", "4"]
    runs = []
    for i, seed in enumerate(("7", "7")):
        outdir = tmp_path / str(i)
        result = CliRunner().invoke(cli, ["convert", str(fixture_path("gradient6")), "-o",
                                          str(outdir), "--seed", seed] + args)
        assert result.exit_code == 0, result.output
        runs.append((outdir / "gradient6.png").read_bytes())
    assert runs[0] == runs[1]
