"""scripts/bench.py and scripts/snapshot_outputs.py on one fast case (two_color, box)."""

import json

import numpy as np

import bench
import snapshot_outputs

CASE_ARGS = ["--inputs", "two_color", "--methods", "box", "--presets", "sprite"]


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
