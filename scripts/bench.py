"""Performance baseline: run the pipeline over a case matrix and record per-stage timings.

Each case is (input, preset, method), run in-process with ``pipeline.run`` and the preset's
defaults. Prints a table and writes bench/<short git sha>.json (git sha, dirty flag, platform,
library versions, thread environment variables, and the per-case numbers).

Usage:
  python scripts/bench.py [--inputs NAME_OR_PATH ...] [--methods ...] [--presets ...]
                          [--case INPUT:PRESET ...] [--repeat N] [--out PATH] [--no-write]
  python scripts/bench.py --compare bench/OLD.json [same options]

Inputs are fixture names (tests/fixtures/<name>.png) or paths. Every input in --inputs runs
with every preset in --presets; each --case INPUT:PRESET adds one input with one preset. All
inputs run every method in --methods. --repeat N keeps the minimum total time of N runs (and
the per-stage timings of that run). Before measuring, every preset/method runs once untimed on
a tiny image so lazy imports are not charged to the first case (--no-warmup to skip).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pixelforge.config import METHODS, PRESETS  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "fixtures"
DEFAULT_INPUTS = ("line_diag", "circle_alpha", "noisy_gradient", "gradient6")
THREAD_ENV = ("PIXELFORGE_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
STAGES = ("load", "preprocess", "downscale", "palette_quantize", "postprocess", "tiles")


def resolve_input(name: str) -> Path:
    """A fixture name (tests/fixtures/<name>.png) or a path to an image."""
    fixture = FIXTURE_DIR / f"{name}.png"
    if fixture.is_file():
        return fixture
    path = Path(name).expanduser()
    if not path.is_file():
        raise SystemExit(f"input not found: {name} (neither a fixture name nor a file)")
    return path


def input_label(path: Path) -> str:
    return path.stem


def build_cases(inputs, presets, methods, extra_cases=()) -> list[tuple[Path, str, str]]:
    """[(input path, preset, method)] in a fixed order: --inputs × presets, then --case."""
    for preset in presets:
        if preset not in PRESETS:
            raise SystemExit(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    pairs = [(resolve_input(name), preset) for name in inputs for preset in presets]
    for spec in extra_cases:
        name, sep, preset = spec.rpartition(":")
        if not sep or preset not in PRESETS:
            raise SystemExit(f"--case expects INPUT:PRESET with a preset in "
                             f"{', '.join(PRESETS)}, got {spec!r}")
        pairs.append((resolve_input(name), preset))
    return [(path, preset, method) for path, preset in pairs for method in methods]


def case_key(path: Path, preset: str, method: str) -> str:
    return f"{input_label(path)}/{preset}/{method}"


def case_config(preset: str, method: str):
    from pixelforge import Config

    return Config(preset=preset, method=method)


def add_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS),
                        metavar="NAME_OR_PATH",
                        help="fixture names or image paths (default: %(default)s)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        help="downscale methods (default: %(default)s)")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS),
                        help="presets applied to every --inputs entry (default: %(default)s)")
    parser.add_argument("--case", action="append", default=[], metavar="INPUT:PRESET",
                        help="add one input with one preset (all methods); repeatable")


def cases_from_args(args) -> list[tuple[Path, str, str]]:
    return build_cases(args.inputs, args.presets, args.methods, args.case)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_info() -> dict:
    """HEAD, whether tracked files are modified, and the same for the package alone: the
    pixelforge/ tree hash identifies the measured code even when only scripts/ changed."""
    sha = git("rev-parse", "HEAD") or "unknown"
    status = git("status", "--porcelain", "--untracked-files=no")
    package_status = git("status", "--porcelain", "--", "pixelforge")
    return {"git_sha": sha, "git_dirty": bool(status),
            "package_tree": git("rev-parse", "HEAD:pixelforge") or "unknown",
            "package_dirty": bool(package_status)}


def environment() -> dict:
    import numpy
    import scipy

    return {"machine": platform.machine(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "env": {name: os.environ[name] for name in THREAD_ENV if name in os.environ}}


def warm_up(methods) -> None:
    """One untimed run per preset and method on a tiny in-memory image, so lazy imports
    (skimage, scipy submodules) are not charged to the first measured case."""
    import numpy as np

    from pixelforge import Config, run

    yy, xx = np.mgrid[0:48, 0:48]
    rgb = np.stack([xx * 5, yy * 5, (xx + yy) * 2], axis=-1).astype(np.uint8)
    for preset in PRESETS:
        for method in methods:
            run(rgb, Config(preset=preset, method=method, out_width=16, out_height=16))


def run_case(path: Path, preset: str, method: str, repeat: int) -> dict:
    from pixelforge import run

    best = None
    for _ in range(max(1, repeat)):
        config = case_config(preset, method)
        start = time.perf_counter()
        result = run(path, config)
        total = time.perf_counter() - start
        if best is None or total < best[0]:
            best = (total, result)
    total, result = best
    stats = result.stats
    method_stats = stats.get("method", {})
    return {"input": str(path), "label": input_label(path), "preset": preset, "method": method,
            "config_hash": result.config.hash(), "total_s": round(total, 4),
            "timings": dict(stats["timings"]), "iterations": int(stats["iterations"]),
            "method_stats": {k: v for k, v in method_stats.items()
                             if isinstance(v, (bool, int, float, str))},
            "output_size": [int(result.indices.shape[1]), int(result.indices.shape[0])],
            "palette_size": int(len(result.palette)), "repeat": max(1, repeat)}


def format_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) if i < 3 else str(c).rjust(w)
                       for i, (c, w) in enumerate(zip(row, widths)))
             for row in [header, *rows]]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def results_table(cases: list[dict]) -> str:
    rows = [[c["label"], c["preset"], c["method"], f"{c['total_s']:.3f}",
             f"{c['timings'].get('downscale', 0.0):.3f}", str(c["iterations"]),
             "x".join(map(str, c["output_size"]))] for c in cases]
    return format_table(rows, ["input", "preset", "method", "total s", "downscale s", "iters",
                               "size"])


def compare_table(old: dict, new: dict) -> str:
    old_cases = {case_key(Path(c["label"]), c["preset"], c["method"]): c for c in old["cases"]}
    rows = []
    for c in new["cases"]:
        key = case_key(Path(c["label"]), c["preset"], c["method"])
        o = old_cases.get(key)
        if o is None:
            rows.append([c["label"], c["preset"], c["method"], "-", f"{c['total_s']:.3f}", "-",
                         "-", f"{c['timings'].get('downscale', 0.0):.3f}", "-", ""])
            continue
        ot, nt = o["total_s"], c["total_s"]
        od, nd = o["timings"].get("downscale", 0.0), c["timings"].get("downscale", 0.0)
        note = "" if o.get("config_hash") == c.get("config_hash") else "config differs"
        rows.append([c["label"], c["preset"], c["method"], f"{ot:.3f}", f"{nt:.3f}",
                     f"{ot / nt:.2f}x" if nt > 0 else "-", f"{od:.3f}", f"{nd:.3f}",
                     f"{od / nd:.2f}x" if nd > 0 else "-", note])
    return format_table(rows, ["input", "preset", "method", "old total", "new total", "speedup",
                               "old down", "new down", "speedup", "note"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_case_arguments(parser)
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per case; the minimum total is reported (default: 1)")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON output path (default: bench/<short git sha>.json)")
    parser.add_argument("--no-write", action="store_true", help="do not write the JSON file")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the untimed warm-up run per preset and method")
    parser.add_argument("--compare", type=Path, default=None, metavar="OLD.json",
                        help="also print old vs new total and downscale seconds per case")
    args = parser.parse_args(argv)

    cases = cases_from_args(args)
    old = json.loads(args.compare.read_text()) if args.compare else None
    report = {"timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
              **git_info(), **environment(), "repeat": max(1, args.repeat),
              "warmup": not args.no_warmup, "cases": []}
    if not args.no_warmup:
        warm_up(args.methods)
    for path, preset, method in cases:
        print(f"running {case_key(path, preset, method)} ...", file=sys.stderr, flush=True)
        report["cases"].append(run_case(path, preset, method, args.repeat))

    print(results_table(report["cases"]))
    if old is not None:
        print()
        print(f"compare: {args.compare} ({old.get('git_sha', '?')[:7]}) -> "
              f"{report['git_sha'][:7]}{' (dirty)' if report['git_dirty'] else ''}")
        print(compare_table(old, report))
    if not args.no_write:
        out = args.out or ROOT / "bench" / f"{report['git_sha'][:7]}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
