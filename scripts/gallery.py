"""Comparison gallery: box / kopf / gerstner × sprite / background on every fixture.

Runs all three methods with both presets on every image in tests/fixtures plus every PNG in an
optional samples/ directory and writes gallery/index.html: one row per image and preset with
the input and the three results (the *_preview.png files, i.e. upscaled by scale_preview),
with timings and iteration counts under each cell.

Usage: python scripts/gallery.py [--out gallery] [--samples samples] [--quick]
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_fixtures  # noqa: E402

from pixelforge import Config, run  # noqa: E402
from pixelforge.config import METHODS, PRESETS  # noqa: E402

STYLE = """
:root { color-scheme: light dark; --bg: #f4f1ea; --fg: #23201c; --muted: #6d665c;
        --card: #fffdf8; --line: #d9d2c4; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #17161a; --fg: #ebe7df; --muted: #9a94a3; --card: #211f26; --line: #37333f; } }
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
h1 { font-size: 20px; margin: 0 0 4px; }
p.lead { color: var(--muted); margin: 0 0 20px; max-width: 80ch; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; }
th, td { border: 1px solid var(--line); padding: 10px; vertical-align: top; text-align: left;
         background: var(--card); }
th { font-weight: 600; white-space: nowrap; }
td.name { white-space: nowrap; }
img { display: block; width: 256px; height: auto; image-rendering: pixelated;
      background: repeating-conic-gradient(#8883 0% 25%, transparent 0% 50%) 0 0 / 16px 16px; }
small { display: block; margin-top: 6px; color: var(--muted); }
"""


def collect_inputs(samples_dir: Path) -> list[Path]:
    fixtures_dir = ROOT / "tests" / "fixtures"
    if any(not (fixtures_dir / f"{name}.png").is_file() for name in make_fixtures.FIXTURES):
        make_fixtures.main(fixtures_dir)
    inputs = [fixtures_dir / f"{name}.png" for name in make_fixtures.FIXTURES]
    if samples_dir.is_dir():
        inputs += sorted(samples_dir.glob("*.png"))
    return inputs


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "gallery"))
    parser.add_argument("--samples", default=str(ROOT / "samples"))
    parser.add_argument("--quick", action="store_true",
                        help="run the background preset at 64 px instead of 256 px")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    image_dir = out_dir / "img"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in collect_inputs(Path(args.samples)):
        input_copy = image_dir / f"{path.stem}_input.png"
        shutil.copyfile(path, input_copy)
        for preset in PRESETS:
            cells = []
            for method in METHODS:
                overrides = {"out_height": 64, "out_width": 64} if (
                    args.quick and preset == "background") else {}
                if path.parent.name == "fixtures":
                    # The fixtures are test patterns, not sprites: keep their backgrounds.
                    overrides["key_bg"] = False
                config = Config(preset=preset, method=method, **overrides)
                start = time.perf_counter()
                result = run(path, config)
                seconds = time.perf_counter() - start
                outputs = result.save(image_dir / f"{path.stem}_{preset}_{method}")
                height, width = result.indices.shape
                caption = (f"{width}×{height} · {result.stats['colors_used']} colors · "
                           f"{seconds:.2f} s · {result.stats['iterations']} iterations")
                cells.append((Path(outputs["preview"]).name, caption))
                print(f"{path.stem:>16} {preset:>10} {method:>8}  {caption}", flush=True)
            rows.append((path.stem, preset, input_copy.name, cells))

    parts = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             "<title>pixelforge gallery</title>", f"<style>{STYLE}</style></head><body>",
             "<h1>pixelforge gallery</h1>",
             '<p class="lead">Every fixture converted with the three downscalers under both '
             "presets. Results are the <code>*_preview.png</code> files (nearest-neighbor "
             "upscale by <code>scale_preview</code>), displayed at a fixed width; open an image "
             "for full size.</p>",
             '<div class="scroll"><table><thead><tr><th>image · preset</th><th>input</th>']
    parts += [f"<th>{html.escape(m)}</th>" for m in METHODS]
    parts.append("</tr></thead><tbody>")
    for stem, preset, input_name, cells in rows:
        parts.append(f'<tr><td class="name">{html.escape(stem)}<small>{preset}</small></td>')
        parts.append(f'<td><a href="img/{html.escape(input_name)}">'
                     f'<img src="img/{html.escape(input_name)}" alt="{html.escape(stem)} input">'
                     "</a></td>")
        for (preview, caption), method in zip(cells, METHODS):
            parts.append(f'<td><a href="img/{html.escape(preview)}">'
                         f'<img src="img/{html.escape(preview)}" loading="lazy" '
                         f'alt="{html.escape(stem)} {preset} {method}"></a>'
                         f"<small>{html.escape(caption)}</small></td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div></body></html>")
    index = out_dir / "index.html"
    index.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {index}")
    return index


if __name__ == "__main__":
    main()
