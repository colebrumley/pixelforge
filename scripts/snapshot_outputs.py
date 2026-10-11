"""Behavior baseline: save the pipeline's outputs per case, and compare a later run against them.

  python scripts/snapshot_outputs.py save [DIR] [--inputs ...] [--methods ...] [--presets ...]
                                         [--case INPUT:PRESET ...]
  python scripts/snapshot_outputs.py compare [DIR] [--tolerance-pixels N]

``save`` runs the same case matrix as scripts/bench.py and writes, per case, an .npz with
``indices``, ``palette`` (uint8) and ``palette_lab``, plus DIR/manifest.json with the git sha
and each case's input path and ``Config.hash()``. Input paths inside the repository are stored
relative to the repository root (and resolved against it by ``compare``, whatever the working
directory); other inputs are stored as absolute paths. ``compare`` re-runs every case in the
manifest and reports whether the output is identical, whether the rendered pixels are
identical, how many pixels differ, the mean and max CIE76 ΔE over the differing pixels that
are opaque in both, whether the palettes are identical, and whether the config hash still
matches (if not, the comparison is not like-for-like).

"identical" means the same indices and palette, i.e. the same indexed PNG; "pixels identical"
means the same rendered colors and transparency, which a reordered palette with remapped
indices also satisfies. Exit code 0 if every case is identical, or with --tolerance-pixels
N > 0 if every case has at most N differing pixels; 1 otherwise. DIR defaults to
bench/snapshots/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench  # noqa: E402

ROOT = bench.ROOT
DEFAULT_DIR = ROOT / "bench" / "snapshots"
MANIFEST = "manifest.json"
IDENTICAL_HELP = ('"identical" = same indices and palette, i.e. the same PNG; '
                  '"pixels identical" = same rendered colors.')


def case_filename(key: str) -> str:
    return key.replace("/", "__") + ".npz"


def run_case(path: Path, preset: str, method: str):
    from pixelforge import run

    config = bench.case_config(preset, method)
    return config, run(path, config)


def manifest_path(path: Path) -> str:
    """Repository-relative for inputs under ROOT, absolute otherwise."""
    return bench.display_path(path)


def resolve_manifest_path(stored: str) -> Path:
    """Inverse of manifest_path: relative paths are relative to ROOT, not the cwd."""
    path = Path(stored)
    return path if path.is_absolute() else ROOT / path


def save(directory: Path, cases) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {**bench.git_info(), "cases": {}}
    for (path, preset, method), key in zip(cases, bench.case_keys(cases)):
        print(f"saving {key} ...", file=sys.stderr, flush=True)
        config, result = run_case(path, preset, method)
        np.savez_compressed(directory / case_filename(key), indices=result.indices,
                            palette=np.asarray(result.palette, dtype=np.uint8),
                            palette_lab=result.palette_lab)
        manifest["cases"][key] = {"input": manifest_path(path), "preset": preset, "method": method,
                                  "config_hash": config.hash(), "file": case_filename(key)}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pixel_lab(indices: np.ndarray, palette: np.ndarray) -> np.ndarray:
    from pixelforge.color import rgb8_to_lab

    return rgb8_to_lab(palette)[np.clip(indices, 0, None)]


def compare_arrays(old: dict, indices: np.ndarray, palette: np.ndarray) -> dict:
    """Differences between a stored snapshot and a new result (indices + uint8 palette)."""
    from pixelforge.color import delta_e

    old_idx, old_pal = old["indices"], np.asarray(old["palette"], dtype=np.uint8)
    palette = np.asarray(palette, dtype=np.uint8)
    palette_identical = old_pal.shape == palette.shape and bool((old_pal == palette).all())
    if old_idx.shape != indices.shape:
        return {"identical": False, "pixels_identical": False, "shape_mismatch": True,
                "old_shape": list(old_idx.shape), "new_shape": list(indices.shape),
                "diff_pixels": int(max(old_idx.size, indices.size)),
                "mean_delta_e": None, "max_delta_e": None, "alpha_diff_pixels": None,
                "palette_identical": palette_identical}
    # Count differences in rendered colors, not raw indices, so a reordered palette with
    # remapped indices shows 0 differing pixels (pixels_identical) while ``identical`` stays
    # False: the indexed PNG differs.
    old_opaque, new_opaque = old_idx >= 0, indices >= 0
    old_rgb = old_pal[np.clip(old_idx, 0, None)]
    new_rgb = palette[np.clip(indices, 0, None)]
    color_diff = (old_rgb != new_rgb).any(axis=-1) & old_opaque & new_opaque
    alpha_diff = old_opaque != new_opaque
    diff = color_diff | alpha_diff
    if color_diff.any():
        de = delta_e(pixel_lab(old_idx, old_pal)[color_diff],
                     pixel_lab(indices, palette)[color_diff])
        mean_de, max_de = float(de.mean()), float(de.max())
    else:
        mean_de = max_de = 0.0
    n_diff = int(diff.sum())
    return {"identical": n_diff == 0 and palette_identical
            and bool((old_idx == indices).all()),
            "pixels_identical": n_diff == 0,
            "shape_mismatch": False, "diff_pixels": n_diff,
            "alpha_diff_pixels": int(alpha_diff.sum()),
            "mean_delta_e": mean_de, "max_delta_e": max_de,
            "palette_identical": palette_identical}


def compare(directory: Path, tolerance_pixels: int = 0) -> tuple[int, list[dict]]:
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(f"no snapshot manifest at {manifest_path}; run `save` first")
    manifest = json.loads(manifest_path.read_text())
    reports = []
    for key, entry in manifest["cases"].items():
        print(f"comparing {key} ...", file=sys.stderr, flush=True)
        config, result = run_case(resolve_manifest_path(entry["input"]), entry["preset"], entry["method"])
        with np.load(directory / entry["file"]) as data:
            old = {name: data[name] for name in data.files}
        report = compare_arrays(old, result.indices, result.palette)
        report["key"] = key
        report["config_hash_match"] = config.hash() == entry["config_hash"]
        reports.append(report)

    rows = []
    for r in reports:
        de = ("-" if r["mean_delta_e"] is None
              else f"{r['mean_delta_e']:.2f} / {r['max_delta_e']:.2f}")
        diff = str(r["diff_pixels"]) + (" (shape)" if r["shape_mismatch"] else "")
        rows.append([r["key"], "yes" if r["identical"] else "NO",
                     "yes" if r["pixels_identical"] else "no", diff, de,
                     "yes" if r["palette_identical"] else "no",
                     "yes" if r["config_hash_match"] else "NO (not like-for-like)"])
    print(f"snapshot {directory} (git {manifest.get('git_sha', '?')[:7]}"
          f"{', dirty' if manifest.get('git_dirty') else ''}) vs "
          f"{bench.git_info()['git_sha'][:7]}")
    print(bench.format_table(rows, ["case", "identical", "pixels same", "diff px",
                                    "ΔE mean / max",
                                    "palette same", "config hash same"]))
    if any(not r["config_hash_match"] for r in reports):
        print("note: config hash changed for some cases; those comparisons are not "
              "like-for-like (the configuration itself differs).")
    # With N = 0 any difference fails (a changed palette included); with N > 0 a case fails
    # only if more than N pixels differ.
    failing = [r["key"] for r in reports if not r["identical"] and (
        tolerance_pixels == 0 or r["shape_mismatch"] or r["diff_pixels"] > tolerance_pixels)]
    reordered = [r["key"] for r in reports if r["pixels_identical"] and not r["identical"]]
    if reordered:
        print(f"{len(reordered)} case(s) render the same pixels but are not identical "
              f"(indices or palette differ, so the PNG differs): {', '.join(reordered)}")
    if not failing and all(r["identical"] for r in reports):
        print("all identical")
    elif not failing:
        print(f"all within tolerance ({tolerance_pixels} differing pixels per case)")
    else:
        print(f"{len(failing)} of {len(reports)} case(s) differ beyond tolerance "
              f"({tolerance_pixels} px): {', '.join(failing)}")
    return (1 if failing else 0), reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog=IDENTICAL_HELP)
    sub = parser.add_subparsers(dest="command", required=True)
    p_save = sub.add_parser("save", help="run the case matrix and store the outputs")
    p_save.add_argument("dir", nargs="?", type=Path, default=DEFAULT_DIR)
    bench.add_case_arguments(p_save)
    p_cmp = sub.add_parser(
        "compare", help="re-run the stored cases and compare",
        description=IDENTICAL_HELP)
    p_cmp.add_argument("dir", nargs="?", type=Path, default=DEFAULT_DIR)
    p_cmp.add_argument("--tolerance-pixels", type=int, default=0, metavar="N",
                       help="up to N differing pixels per case still exit 0 (default: 0)")
    args = parser.parse_args(argv)

    if args.command == "save":
        manifest = save(args.dir, bench.cases_from_args(args))
        print(f"saved {len(manifest['cases'])} case(s) to {args.dir}")
        return 0
    code, _ = compare(args.dir, args.tolerance_pixels)
    return code


if __name__ == "__main__":
    sys.exit(main())
