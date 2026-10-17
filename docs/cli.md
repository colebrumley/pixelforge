# Command line

`pixelforge convert --help` (also `batch` and `compare`) lists each flag with its
description, per-preset default and allowed values. This page covers how options combine,
what gets written, and how to handle hard inputs.

```bash
pixelforge convert hero.png --preset sprite --method gerstner --palette-size 12
pixelforge convert scene.png --preset background           # writes tileset + tilemap
pixelforge batch frames/ --preset sprite                    # one shared palette for all
pixelforge compare hero.png -o out                          # box, kopf, gerstner side by side
pixelforge compare photo.png --prepass                      # proof sheet: methods × pre-pass
pixelforge compare hero.png --sweep palette-size=8,12,16    # proof sheet: methods × one option
pixelforge palettes                                         # list bundled palettes
pixelforge convert hero.png --preset sprite --edit          # convert, then touch up
pixelforge edit out/hero.png                                # edit an earlier output
```

The editor commands are covered in the [editor guide](editor.md).

## Options and config files

Every config field is a `--kebab-case` flag. Booleans take `--flag/--no-flag`. `g_T_final`
is `--g-t-final`, and the old `--g-T-final` still works.

`--config file.json` loads a config. Precedence is defaults ← preset ← JSON ← flags.

## Output files

`convert` writes to `OUTDIR` (default `out/`):

| file | content |
| --- | --- |
| `NAME.png` | native-resolution result (palette-indexed PNG) |
| `NAME_preview.png` | nearest-neighbor upscale |
| `NAME_palette.json`, `NAME_palette.hex` | the palette (LAB rounded to 6 decimals); `.hex` is reusable via `--palette-name`; `.json` entries carry `"used"` (unused entries are kept, so named palettes stay complete) |
| `NAME_meta.json` | stats, timings, config, config hash, input hash, palette, environment (`platform`, `threads`, and `native`, the pixelforge version; not hashed) |
| `NAME_tileset.png`, `NAME_tilemap.json` | background preset only; tiles merge when mean ΔE < `tile_dedupe_tolerance` and every pixel's ΔE < 10 (flips included) |
| `NAME.tmj` | background preset only: Tiled JSON map (one `background` layer, flips in the GID high bits, tileset embedded and pointing at `NAME_tileset.png`) |
| `NAME_tilemap.csv` | background preset only: one line per tile row of GIDs (1-based, 0 = empty, no flip bits) |

The editor adds `NAME_edits.json` when it saves (see [the edit log](editor.md#the-edit-log)).

`--tileset-columns N` sets the tileset sheet width in tiles. The default, 0, means
`ceil(sqrt(tiles))`.

`--transparent-index` sets where transparency sits in the PNG palette:

- `last` (the default) puts it after the colors.
- `first` puts it at index 0 and shifts every color by one, as Aseprite and console tools
  expect. It also adds a leading `{"hex": null, "transparent": true}` entry to
  `_palette.json` and a `; transparent` comment line to `_palette.hex`.

The library's `indices` always use −1 for transparent.

`convert`, `batch` and `compare` refuse (exit 2) to write an output over an input file.
`batch` also refuses inputs that share a name (`a.png`, `a.bmp`). `--force` overrides both.
`batch` skips `*_preview`, `*_tileset` and `*_compare` images.

## Output size

`--scale N` fixes the downscale ratio: the output is the cropped input divided by N per
axis. `--canvas WxH` fits the subject inside a fixed canvas, centered, outline included.
Neither combines with `--out-width/--out-height`.

Unless `--scale`, `--canvas` or both `--out-*` flags are given, `batch` derives one scale
from the largest frame's subject and gives every frame the same canvas. Animation frames
cropped to their alpha then keep a constant size. The first JSON line reports `scale` and
`canvas`.

With a tileset, a derived size must be a multiple of `--tile-size`. `--fit` chooses how:

- `pad` (the default) keeps the aspect and pads with transparent pixels, centered.
- `stretch` scales to fill, and warns if the aspect changes by more than 2 %.
- `crop` crops the input, centered.

Inputs much larger than the output are box-reduced to 8–16× the output size before denoising
(`--prereduce-max-ratio`, 0 disables).

## Palettes

`--palette-name` takes a bundled name, a `.hex`/`.gpl` path, or inline colors
(`hex:ff0000,00ff00,…`). `batch` writes its shared palette to `shared_palette.hex` and hands
it to every frame inline, so the frames' metadata carries the colors, not the output path.

## Backdrops and transparency

The sprite preset keys out a flat opaque backdrop without rembg (`--no-key-bg` disables
it). That includes the anti-aliased fringe where the subject blends into the backdrop
(`key_bg_fringe`).

`key_bg` only keys a backdrop that covers at least 90 % of the image border within
`--key-bg-tolerance`. Two kinds of input fail that test: a mottled studio backdrop is not
flat, and a head-and-shoulders shot cut off by the bottom edge has too little backdrop on
the border. In both cases the backdrop stays in the picture and takes part in the levels
stretch. To get a sprite with a transparent background, cut the subject out first with
`--remove-bg` or an input with an alpha channel.

Pixels at or above `alpha_threshold` are opaque. Each still counts in the downscaled colors
in proportion to its alpha, so a semi-transparent fringe pulls its neighbors less.

Orphan removal keeps high-contrast single pixels such as 1-px eyes and highlights. A stray
pixel is merged only when it is within ΔE `orphan_max_delta` (default 25) of its
replacement.

## Photos: `--enhance` and `--ink`

Photos have soft gradients and no lines, so features smaller than an output pixel (eyes, a
mouth, a chain) get averaged away. `--enhance` exaggerates them before downscaling. It
stretches the subject's brightness to the full range and adds local contrast at
`--enhance-radius` (2) output pixels. Try `0.5` to `1`. Higher values (up to 4) turn dark
areas such as hair solid black. `--ink 0.6` also darkens thin dark features into lines, and
it adds blots on some images.

The right values depend on the photo, so pick them from a [proof sheet](#proof-sheets). Both
options are off by default and in every preset. Sources that already have outlines and full
contrast (illustrations, pixel art) lose their mid-tone shading to them.

Colors keep their hue: a pixel is brightened only as far as its brightest channel allows.
The stretch maps the subject's 1st and 99th brightness percentiles to black and white. It
never amplifies by more than 5×, so a nearly flat subject stays nearly flat.
`--enhance-black` / `--enhance-white` set the two levels by hand. `batch` measures one pair
of levels over all frames (reported in its first JSON line), so a flash in one frame does not
change the rest of it. `--seamless` tiles stay seamless.

A source photo converts best when:

- the subject fills the frame (a feature needs 3–4 output pixels to survive),
- the light is flat and frontal,
- the backdrop is one flat color that differs from hair and skin (see
  [Backdrops and transparency](#backdrops-and-transparency)), and
- adjacent parts differ in tone.

## Proof sheets

`compare --prepass` and `compare --sweep FIELD=V1,V2,...` write one labelled sheet
(`INPUT_compare.png`) instead of each method's outputs: a row per method, a column per
setting.

- `--prepass` uses four columns, from no pre-pass to `--enhance 1 --ink 0.6`.
- `--sweep` takes one option other than `--method`, given once. Use `on`/`off` for booleans
  and `none` to unset. Values are split on commas, so inline `hex:` palettes cannot be
  swept.
- `--method` limits the sheet to that row.

The JSON result lists, per cell, its `flags` and its full `config`:

- The `flags` cover only the method and the swept setting. Add them to a `convert` call with
  the same base flags and `--config` file that `compare` was given.
- The `config` is the whole cell. Save it as JSON and pass it with `--config` to reproduce
  the cell on its own. It is also the only way to replay a cell that unsets a value with
  `none`, which no flag can do.

Previews are scaled down from `--scale-preview` when a sheet would exceed 32 million pixels.
Every cell is a full run, so a 3 × 4 sheet with kopf and gerstner takes minutes at large
sizes. Pick settings at a small `--out-height`, then convert at full size.

## Logging and errors

Runs take from under a second (box) to a minute or more (kopf and gerstner at large sizes).
They print nothing until the JSON result unless you ask:

- `pixelforge -v convert ...` logs each stage with its duration to stderr.
- `-vv` logs every downscaler iteration.
- `-q` logs only errors.

stdout carries only the JSON results. Library callers get the same messages through the
`log` crate (targets under `pixelforge`).

Errors print one line. The exit code is 2 for configuration problems (bad flags, unknown
palette, `--remove-bg` without rembg) and 1 for anything else. `pixelforge --debug ...` or
`PIXELFORGE_DEBUG=1` prints the full error chain instead.

## Limits

Inputs larger than 24 million pixels (`width × height`) are refused before decoding.
`--max-input-pixels N` raises or lowers the budget (the `max_pixels` argument of the
library's `run`). Only PNG, JPEG, GIF, WEBP, BMP and TIFF are read, and only regular files.
Numeric config fields have upper bounds (for example output edges ≤ 4096, `--scale-preview`
≤ 64), and non-finite numbers are rejected.
