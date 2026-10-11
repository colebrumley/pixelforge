"""scripts/bench.py and scripts/snapshot_outputs.py on one fast case (two_color, box)."""

import json
import shutil
import subprocess
import sys

import numpy as np

import bench
import snapshot_outputs

CASE_ARGS = ["--inputs", "two_color", "--methods", "box", "--presets", "sprite"]
SNAPSHOT_SCRIPT = bench.ROOT / "scripts" / "snapshot_outputs.py"


def test_bench_writes_json(tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert bench.main([*CASE_ARGS, "--no-warmup", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    for key in ("git_sha", "git_dirty", "package_tree", "timestamp", "machine", "python",
                "numpy", "scipy", "env", "repeat", "cases"):
        assert key in report
    (case,) = report["cases"]
    assert (case["label"], case["preset"], case["method"]) == ("two_color", "sprite", "box")
    assert case["total_s"] > 0 and "downscale" in case["timings"]
    assert case["iterations"] == 1 and case["output_size"] == [64, 64]
    assert "two_color" in capsys.readouterr().out

    assert bench.main([*CASE_ARGS, "--no-warmup", "--no-write", "--compare", str(out)]) == 0
    assert "speedup" in capsys.readouterr().out


def test_snapshot_identical_then_one_pixel_changed(tmp_path, capsys):
    snap = tmp_path / "snap"
    assert snapshot_outputs.main(["save", str(snap), *CASE_ARGS]) == 0
    manifest = json.loads((snap / "manifest.json").read_text())
    (entry,) = manifest["cases"].values()
    assert entry["config_hash"] and manifest["git_sha"]

    assert snapshot_outputs.main(["compare", str(snap)]) == 0
    assert "all identical" in capsys.readouterr().out

    # Repaint one opaque pixel with a different palette color.
    path = snap / entry["file"]
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    indices, palette = arrays["indices"], arrays["palette"]
    y, x = np.argwhere(indices >= 0)[0]
    other = next(k for k in range(len(palette))
                 if (palette[k] != palette[indices[y, x]]).any())
    indices[y, x] = other
    np.savez_compressed(path, **arrays)

    code, (report,) = snapshot_outputs.compare(snap)
    assert code == 1
    assert not report["identical"] and report["diff_pixels"] == 1
    assert report["max_delta_e"] > 0 and report["palette_identical"]
    assert report["config_hash_match"]
    assert snapshot_outputs.compare(snap, tolerance_pixels=1)[0] == 0


def test_same_stem_inputs_get_distinct_keys(tmp_path, fixture_path, capsys):
    a, b = tmp_path / "a" / "two_color.png", tmp_path / "b" / "two_color.png"
    for path in (a, b):
        path.parent.mkdir()
        shutil.copy(fixture_path("two_color"), path)
    unique = fixture_path("line_diag")
    cases = bench.build_cases([str(a), str(b), str(unique)], ["sprite"], ["box"])
    keys = bench.case_keys(cases)
    assert len(set(keys)) == 3
    assert keys[0].startswith("two_color-") and keys[1].startswith("two_color-")
    assert keys[2] == "line_diag/sprite/box"  # unique stems keep their plain key

    snap = tmp_path / "snap"
    args = ["--inputs", str(a), str(b), "--methods", "box", "--presets", "sprite"]
    assert snapshot_outputs.main(["save", str(snap), *args]) == 0
    manifest = json.loads((snap / "manifest.json").read_text())
    assert {e["input"] for e in manifest["cases"].values()} == {str(a), str(b)}
    assert len({e["file"] for e in manifest["cases"].values()}) == 2
    code, reports = snapshot_outputs.compare(snap)
    assert code == 0 and len(reports) == 2 and all(r["identical"] for r in reports)


def test_snapshot_relative_input_resolves_from_any_cwd(tmp_path):
    snap = tmp_path / "snap"
    rel = "tests/fixtures/two_color.png"
    run = lambda *args, cwd: subprocess.run(  # noqa: E731
        [sys.executable, str(SNAPSHOT_SCRIPT), *args], cwd=cwd, capture_output=True,
        text=True)
    saved = run("save", str(snap), "--inputs", rel, "--methods", "box", "--presets", "sprite",
                cwd=bench.ROOT)
    assert saved.returncode == 0, saved.stderr
    (entry,) = json.loads((snap / "manifest.json").read_text())["cases"].values()
    assert entry["input"] == rel  # repository-relative, not cwd-relative
    other = tmp_path / "elsewhere"
    other.mkdir()
    compared = run("compare", str(snap), cwd=other)
    assert compared.returncode == 0, compared.stdout + compared.stderr
    assert "all identical" in compared.stdout

    # An input outside the repository given relative to the cwd is stored absolute.
    shutil.copy(bench.ROOT / rel, other / "outside.png")
    snap2 = tmp_path / "snap2"
    saved = run("save", str(snap2), "--inputs", "outside.png", "--methods", "box",
                "--presets", "sprite", cwd=other)
    assert saved.returncode == 0, saved.stderr
    (entry,) = json.loads((snap2 / "manifest.json").read_text())["cases"].values()
    assert entry["input"] == str((other / "outside.png").resolve())
    compared = run("compare", str(snap2), cwd=bench.ROOT)
    assert compared.returncode == 0, compared.stdout + compared.stderr


def test_snapshot_reordered_palette_is_pixels_identical_only(tmp_path, capsys):
    snap = tmp_path / "snap"
    assert snapshot_outputs.main(["save", str(snap), *CASE_ARGS]) == 0
    (entry,) = json.loads((snap / "manifest.json").read_text())["cases"].values()
    path = snap / entry["file"]
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    palette, indices = arrays["palette"], arrays["indices"]
    assert len(palette) >= 2
    order = np.arange(len(palette))[::-1]  # new palette[k] = old palette[order[k]]
    inverse = np.argsort(order)
    arrays["palette"] = palette[order]
    arrays["palette_lab"] = arrays["palette_lab"][order]
    arrays["indices"] = np.where(indices >= 0, inverse[np.clip(indices, 0, None)], indices)
    np.savez_compressed(path, **arrays)

    code, (report,) = snapshot_outputs.compare(snap)
    assert code == 1
    assert report["pixels_identical"] and report["diff_pixels"] == 0
    assert not report["identical"] and not report["palette_identical"]
    out = capsys.readouterr().out
    assert "pixels same" in out and "render the same pixels but are not identical" in out
