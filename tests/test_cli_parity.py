"""The Rust `pixelforge` program against the Python CLI (Phase 7 parity).

Both CLIs run the same command into separate output directories; then PNG pixels, palette,
transparency and text chunks must decode equal, `_palette.json`, `_palette.hex`,
`_tilemap.*`, `.tmj` and `shared_palette.hex` must be byte-identical, `_meta.json` equal
without `environment` and `stats.timings`, the stdout JSON lines equal after the same drops
(output directories masked), and the `-v` stderr lines equal with the stage timings masked.
`compare`'s proof sheets are labelled in different fonts, so only their cells are compared.
The `--help` texts must list the same options with the same defaults, and are recorded in
tests/cli_help/ (`python scripts/cli_help.py`).

The Rust binary is ``$PIXELFORGE_BIN`` when set, else ``cargo run --quiet --bin pixelforge``
from the workspace; the tests skip when neither is available.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cli_help
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
HELP_RECORDS = ROOT / "tests" / "cli_help"
FIXTURE_NAMES = sorted(p.stem for p in FIXTURES.glob("*.png"))
TIMING = re.compile(r"\b\d+\.\d\d s\b")
BYTE_IDENTICAL = ("_palette.json", "_palette.hex", "_tilemap.json", "_tilemap.csv", ".tmj",
                  "shared_palette.hex")


def _rust_command() -> list[str] | None:
    explicit = os.environ.get("PIXELFORGE_BIN")
    if explicit:
        return [explicit] if Path(explicit).is_file() else None
    cargo = shutil.which("cargo")
    if cargo is None or not (ROOT / "Cargo.toml").is_file():
        return None
    return [cargo, "run", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"),
            "-p", "pixelforge", "--bin", "pixelforge", "--"]


@pytest.fixture(scope="module")
def rust():
    command = _rust_command()
    if command is None:
        pytest.skip("no Rust pixelforge binary (set PIXELFORGE_BIN, or install cargo)")
    done = subprocess.run(command + ["--version"], capture_output=True, text=True,
                          timeout=1800)
    if done.returncode != 0:
        pytest.skip(f"the Rust pixelforge binary does not run: {done.stderr.strip()[-500:]}")
    assert done.stdout == "pixelforge, version 0.1.0\n"
    return command


def _run(command: list[str], args: list[str], outdir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env.pop("PIXELFORGE_DEBUG", None)
    return subprocess.run(command + ["-v", *args, "-o", str(outdir)], capture_output=True,
                          text=True, env=env, timeout=600)


def _png(path: Path):
    with Image.open(path) as im:
        return (im.mode, im.size, im.getpalette(), im.info.get("transparency"), im.tobytes(),
                dict(im.text))


def _meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    del meta["environment"], meta["stats"]["timings"]
    return meta


def _stdout(text: str, outdir: Path) -> list[dict]:
    lines = []
    for line in text.replace(str(outdir), "OUT").splitlines():
        payload = json.loads(line)
        # convert / batch: "stats"; compare: per method, or per proof-sheet cell.
        nested = [*payload.get("methods", {}).values(), *payload.get("cells", [])]
        for entry in (payload, *nested):
            entry.get("stats", {}).pop("timings", None)
        lines.append(payload)
    return lines


def _stderr(text: str, outdir: Path) -> list[str]:
    return TIMING.sub("T s", text.replace(str(outdir), "OUT")).splitlines()


def _assert_same(rust_cmd, args, tmp_path):
    """Run both programs; same exit code, files, stdout and stderr. Returns the file names."""
    py_out, rs_out = tmp_path / "py", tmp_path / "rs"
    py = _run([sys.executable, "-m", "pixelforge"], args, py_out)
    rs = _run(rust_cmd, args, rs_out)
    assert rs.returncode == py.returncode, py.stderr + rs.stderr
    names = sorted(p.name for p in py_out.iterdir()) if py_out.exists() else []
    assert (sorted(p.name for p in rs_out.iterdir()) if rs_out.exists() else []) == names
    for name in names:
        a, b = py_out / name, rs_out / name
        if name.endswith(".png"):
            assert _png(a) == _png(b), name
        elif name.endswith("_meta.json"):
            assert _meta(a) == _meta(b), name
        else:
            assert name.endswith(BYTE_IDENTICAL), name
            assert a.read_bytes() == b.read_bytes(), name
    assert _stdout(py.stdout, py_out) == _stdout(rs.stdout, rs_out)
    assert _stderr(py.stderr, py_out) == _stderr(rs.stderr, rs_out)
    return names


_QUICK = {("circle_alpha", "sprite"), ("two_color", "background")}
_BOX_CASES = [pytest.param(name, preset, marks=() if (name, preset) in _QUICK
                           else pytest.mark.slow)
              for name in FIXTURE_NAMES for preset in ("sprite", "background")]


@pytest.mark.parametrize("name, preset", _BOX_CASES)
def test_convert_box_matches_python(rust, tmp_path, name, preset):
    names = _assert_same(rust, ["convert", str(FIXTURES / f"{name}.png"), "--preset", preset,
                                "--method", "box", "--out-width", "32", "--out-height", "32"],
                         tmp_path)
    if (name, preset) == ("line_diag", "sprite"):
        # Keyed, the 2-px line covers no 32-px cell by half: both fail the same way.
        assert names == []
        return
    assert f"{name}.png" in names
    if preset == "background":
        assert f"{name}.tmj" in names


@pytest.mark.slow
def test_convert_kopf_matches_python(rust, tmp_path):
    _assert_same(rust, ["convert", str(FIXTURES / "circle_alpha.png"), "--method", "kopf",
                        "--out-width", "32", "--out-height", "32", "--kopf-max-iters", "10"],
                 tmp_path)


@pytest.mark.slow
def test_convert_gerstner_matches_python(rust, tmp_path):
    _assert_same(rust, ["convert", str(FIXTURES / "noisy_gradient.png"), "--preset",
                        "background", "--method", "gerstner", "--out-width", "32",
                        "--out-height", "32", "--palette-size", "8"], tmp_path)


def _frames(tmp_path) -> Path:
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("circle_alpha", "gradient6", "noisy_gradient"):
        shutil.copy(FIXTURES / f"{name}.png", frames / f"{name}.png")
    shutil.copy(FIXTURES / "two_color.png", frames / "two_color_preview.png")   # skipped
    return frames


@pytest.mark.parametrize("extra", [
    ["--method", "box"],
    pytest.param(["--method", "box", "--enhance", "1", "--preset", "background",
                  "--out-width", "32", "--out-height", "32"], marks=pytest.mark.slow),
    pytest.param(["--method", "gerstner", "--out-height", "16", "--palette-size", "6"],
                 marks=pytest.mark.slow),
], ids=["box", "enhance-background", "gerstner"])
def test_batch_matches_python(rust, tmp_path, extra):
    frames = _frames(tmp_path)
    names = _assert_same(rust, ["batch", str(frames), *extra], tmp_path)
    assert "shared_palette.hex" in names and "noisy_gradient.png" in names


def test_config_errors_match_python(rust, tmp_path):
    """Usage of --config: the same message (and exit code 2) from both programs."""
    image = str(FIXTURES / "two_color.png")
    cases = {"bad.json": '{"method": "box",\n "seed" 1}', "nan.json": '{"enhance": NaN}',
             "list.json": '[1, 2]', "type.json": '{"palette_size": [1, {"a": 2.5}]}',
             "unknown.json": '{"zeta": 1, "alpha": 2}', "big.json": '{"seed": 1e400}'}
    for name, text in cases.items():
        path = tmp_path / name
        path.write_text(text)
        outputs = []
        for command in ([sys.executable, "-m", "pixelforge"], rust):
            done = subprocess.run(command + ["convert", image, "-o", str(tmp_path / "out"),
                                             "--config", str(path)],
                                  capture_output=True, text=True,
                                  env={**os.environ, "PYTHONPATH": str(ROOT)})
            outputs.append((done.returncode, done.stderr))
        assert outputs[0] == outputs[1], name
        assert outputs[0][0] == 2 and outputs[0][1].startswith("config error: ")


def test_compare_matches_python(rust, tmp_path):
    """Plain compare: every method's outputs and the side-by-side sheet (no labels) decode
    equal, and so do the JSON line and the -v lines."""
    names = _assert_same(rust, ["compare", str(FIXTURES / "two_color.png"), "--out-width", "16",
                                "--out-height", "16"], tmp_path)
    assert "two_color_compare.png" in names
    assert {f"two_color_{m}.png" for m in ("box", "kopf", "gerstner")} <= set(names)


def _cell_blocks(path: Path) -> list[tuple[int, int, bytes]]:
    """The cells of a proof sheet, row by row, each cropped to its opaque pixels: the runs of
    opaque rows and columns right of the row-label bar and below the column-label bar (the
    gaps between cells are transparent; each cell's subject must be one piece). The bars are
    measured on the sheet, so the label font does not matter.
    """
    with Image.open(path) as im:
        assert im.mode == "RGBA"
        sheet = np.asarray(im)
    bar = np.array([40, 40, 40, 255], dtype=np.uint8)
    # The column-label bar spans the full width; its last column is never under text.
    top = int(np.argmin((sheet[:, -1] == bar).all(axis=1)))
    # The row-label bar spans the full height; its last row is never under text.
    left = int(np.argmin((sheet[-1] == bar).all(axis=1)))
    assert top > 0 and left > 0
    opaque = sheet[top:, left:, 3] > 0

    def runs(mask: np.ndarray) -> list[tuple[int, int]]:
        edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    blocks = []
    for y0, y1 in runs(opaque.any(axis=1)):
        for x0, x1 in runs(opaque[y0:y1].any(axis=0)):
            cell = sheet[top + y0:top + y1, left + x0:left + x1]
            blocks.append((int(y1 - y0), int(x1 - x0), cell.tobytes()))
    return blocks


@pytest.mark.parametrize("extra", [["--prepass"], ["--sweep", "palette-size=4,8"]],
                         ids=["prepass", "sweep"])
def test_compare_proof_sheet_matches_python(rust, tmp_path, extra):
    """Proof sheets: the same JSON line (cells with flags, configs and stats without timings)
    and -v lines. The label fonts differ (Pillow's default font, a bitmap font in Rust), so the
    sheets differ in size, but their cells must be the same pixels in the same order."""
    args = ["compare", str(FIXTURES / "two_color.png"), "--method", "box", "--out-width", "16",
            "--out-height", "16", *extra]
    py_out, rs_out = tmp_path / "py", tmp_path / "rs"
    py = _run([sys.executable, "-m", "pixelforge"], args, py_out)
    rs = _run(rust, args, rs_out)
    assert py.returncode == rs.returncode == 0, py.stderr + rs.stderr
    assert sorted(p.name for p in py_out.iterdir()) == ["two_color_compare.png"]
    assert sorted(p.name for p in rs_out.iterdir()) == ["two_color_compare.png"]

    [py_json], [rs_json] = _stdout(py.stdout, py_out), _stdout(rs.stdout, rs_out)
    assert py_json == rs_json
    assert py_json["compare"] == "OUT/two_color_compare.png"
    assert py_json["cells"][0]["flags"].startswith("--method box ")
    py_err, rs_err = _stderr(py.stderr, py_out), _stderr(rs.stderr, rs_out)
    assert py_err == rs_err
    assert any(line.startswith("pixelforge: cell box / ") for line in rs_err)
    py_cells = _cell_blocks(py_out / "two_color_compare.png")
    rs_cells = _cell_blocks(rs_out / "two_color_compare.png")
    assert len(py_cells) == len(rs_cells) == len(py_json["cells"])
    for i, (a, b) in enumerate(zip(py_cells, rs_cells)):
        assert a == b, f"cell {i} ({py_json['cells'][i]['column']})"


@pytest.mark.parametrize("command", cli_help.COMMANDS)
def test_help_options_and_defaults_match_python(rust, command):
    """Every option of the click help is in the Rust help, with the same [default: ...] text,
    and the Rust help has no other options. The layouts differ (see scripts/cli_help.py); the
    negative halves of on/off pairs are hidden in the Rust help, which names them once."""
    py_help, rs_help = cli_help.python_help(command), cli_help.rust_help(rust, command)
    py, rs = cli_help.options(py_help), cli_help.options(rs_help)
    assert py, py_help
    hidden = {name for name in py if name.startswith("--no-") and name not in rs}
    for name in hidden:
        positive = "--" + name.removeprefix("--no-")
        assert positive in py and positive in rs, name
        assert py[name] == py[positive], name
    if hidden:
        assert "also has a --no- form" in rs_help
    assert set(rs) == set(py) - hidden
    assert {name: rs[name] for name in rs} == {name: py[name] for name in rs}


@pytest.mark.parametrize("command", cli_help.COMMANDS)
def test_help_records_are_current(rust, command):
    """tests/cli_help/ holds both helps as they are now (`python scripts/cli_help.py`)."""
    assert (HELP_RECORDS / f"python-{command}.txt").read_text() == cli_help.python_help(command)
    assert (HELP_RECORDS / f"rust-{command}.txt").read_text() == cli_help.rust_help(rust,
                                                                                    command)


@pytest.mark.parametrize("args", [
    ["--sweep", "nope=1"], ["--sweep", "method=box,kopf"], ["--sweep", "palette-size=4,,8"],
    ["--sweep", "palette-size=four"], ["--sweep", "key-bg=maybe"],
    ["--sweep", "palette-size=4,999"], ["--sweep", "palette-size=4", "--prepass"],
    ["--sweep", "palette-size=4", "--sweep", "dither=none"],
    ["--sweep", "palette-name=pico8,missing_palette.hex"],
], ids=lambda a: " ".join(a))
def test_compare_errors_match_python(rust, tmp_path, args):
    """compare's config errors: the same message and exit code 2, and nothing written."""
    outputs = []
    for name, command in (("py", [sys.executable, "-m", "pixelforge"]), ("rs", rust)):
        outdir = tmp_path / name
        done = subprocess.run(command + ["compare", str(FIXTURES / "two_color.png"), "-o",
                                         str(outdir), *args],
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT)})
        outputs.append((done.returncode, done.stderr.replace(str(outdir), "OUT")))
        assert not outdir.exists()
    assert outputs[0] == outputs[1]
    assert outputs[0][0] == 2 and outputs[0][1].startswith("config error: ")
