# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `--enhance` (with `--enhance-radius`) and `--ink`: an opt-in pre-pass before downscaling
  for photos. `enhance` stretches the subject's luminance and adds local contrast at about
  two output pixels, so eyes, mouths and other sub-pixel features survive; `ink` darkens thin
  dark features. Off by default and in every preset, so output pixels are unchanged; every
  `config_hash` changes because `Config` has three new fields.
- `compare --prepass` and `compare --sweep FIELD=V1,V2,...` write a labelled proof sheet
  (methods × settings) and report the `convert` flags for each cell; `--method` limits the
  sheet to one row. Plain `compare` is unchanged.
- README: a photo sample (`docs/sample_photo.png`, from an AI-generated portrait) and guidance
  on source photos.

### Fixed

- `convert`, `batch` and `compare` refuse (exit 2) to write an output over an input file, and
  `batch` refuses inputs that share a stem; `--force` overrides both. `batch` skips
  `*_preview`, `*_tileset` and `*_compare` images.
- With a named palette (`--palette-name`) the outline snaps to the nearest palette entry
  instead of adding a colour outside the hardware palette; a `_palette.hex` fed back through
  `--palette-name` reproduces itself. `stats.outline_index` names the entry used.
- `Config.replace(preset=...)` re-applies the new preset to every field the caller did not
  set explicitly; previously the old preset's values were kept under the new label.
- The output canvas is exactly the requested size: the outline ring is reserved inside it
  (`stats.outline_margin`) instead of growing the image by 2 px; with `tileset=True` a
  clipped outline is reported as `stats.outline_clipped`.
- Dithering with gerstner (the default method) now dithers the smoothed superpixel means;
  it was a no-op because the per-pixel values already sat on palette colours. `auto` dither
  ignores transparent cells when measuring local detail, so silhouette edges dither too.
- `batch` derives one scale from the largest frame's subject and gives every frame the same
  canvas, so animation frames keep a constant size; `--scale N` and `--canvas WxH` are new.
- Tile de-duplication requires both a mean ΔE below `tile_dedupe_tolerance` and every
  pixel's ΔE below 10, and merges into the nearest tile; sparse details (stars, highlights)
  are no longer erased or stamped into every cell. `stats.tiles_rerender_px_changed` reports
  what the re-render changed. Tiling is 50–100× faster on large outputs.
- Orphan removal keeps high-contrast single pixels (eyes, highlights): a region is merged
  only when its ΔE to the replacement is below `orphan_max_delta` (25), and passes stop on
  convergence instead of oscillating.
- Gerstner/MCDA splits palette colours a few ΔE apart (the 3× growth rule collapsed them),
  stops annealing when no pair grows instead of cooling to 1e-3, and starts at a temperature
  scaled to the colour spread instead of its variance (about 40 % fewer iterations).
- `key_bg` also keys out the anti-aliased fringe around a flat backdrop (`key_bg_fringe`
  passes of coverage unmixing), so the halo no longer takes palette entries.
- Kopf converges: the |Δμ| < 1e-3 criterion was unreachable even on a solid colour, so
  every run hit `kopf_max_iters`; RMS criteria on Δμ, Δν and the σ-change fraction replace it.
- 16-bit grayscale inputs are rescaled instead of clipped to white; EXIF orientation is
  applied; animations use the first frame with a warning.

### Added

- MIT `LICENSE` and package metadata (license expression, author, classifiers, project URLs).
- `MANIFEST.in` so the sdist ships tests, fixtures, scripts, DESIGN.md and REQUIREMENTS.md.
- Ruff configuration and `ruff` in the `dev` dependency group.
- GitHub Actions CI: lint job plus tests on Ubuntu (Python 3.11/3.12/3.13) and macOS arm64 (3.12).
- This changelog.
- `fit` (`--fit pad|stretch|crop`) for tileset sizes; the default `pad` keeps the aspect ratio
  instead of stretching to a tile multiple, the longest edge never exceeds the target, and a
  `tile_size` above it is an error.

- `Config.provenance()`; PNG `tEXt` chunks `pixelforge:palette_sha256` and `pixelforge:palette`.
- `_meta.json` fields `palette_sha256`, `palette`, `environment` and, with `--remove-bg`,
  `remove_bg_versions`.
- Inline palettes: `--palette-name hex:rrggbb,rrggbb,...`.
- Logging: a `pixelforge` logger (silent by default) with one INFO line per pipeline stage and
  kopf/gerstner iteration progress (DEBUG each, INFO every 10th).
- CLI `-v/--verbose` (`-v` stages, `-vv` iterations) and `-q/--quiet`; messages go to stderr
  as `pixelforge: ...`, stdout stays JSON-only.
- A warning when `key_bg` finds no flat background on an opaque input, suggesting
  `--remove-bg` or an alpha channel.
- `--help` text, per-preset defaults and allowed values for every flag (`config.FIELD_HELP`).
- Tests: determinism across fresh processes with varied `PYTHONHASHSEED` and BLAS/OpenMP
  threads, comparing every output file; golden pixel hashes in `tests/golden.json`
  (regenerate with `scripts/update_golden.py` after an intentional output change); coverage
  of mcda with box/kopf, the RGBA PNG fallback, median denoise, missing rembg, seamless
  gerstner, empty batch directories, bayer8 and zero dither strength, tile size errors and
  every config validation message; a seed test that can fail; committed fixtures checked
  against `scripts/make_fixtures.py`.
- A `slow` pytest marker on the full-size kopf cases; `pytest -m "not slow"` skips them.

### Changed

- Inputs much larger than the output are box-reduced by an integer factor to at most
  `prereduce_max_ratio` (8) times the output size before denoising and downscaling (nearly
  lossless at these ratios); kopf drops its per-slot offset arrays and reuses buffers. Kopf
  1024²→64² went from ≈200 s and 550 MiB to ≈27 s and 67 MiB here.
- Tilesets also write `NAME.tmj` (Tiled JSON with GID flip bits and an embedded tileset) and
  `NAME_tilemap.csv`; the sheet is `tileset_columns` wide (default `ceil(sqrt(n))`, was 16);
  `transparent_index="first"` puts the transparent entry at PNG index 0; `_palette.json`
  marks `"used"` entries and `stats.colors_unused` counts the rest.
- Image loading only enables the PNG, JPEG, GIF, WEBP, BMP and TIFF decoders, refuses
  non-regular files, hashes inputs in a stream, and enforces a pixel budget
  (`--max-input-pixels`, default 24 Mpx); Pillow's decompression-bomb warning is an error.
- `Config` has upper bounds on sizes, iteration caps, `scale_preview`, `denoise_sigma_spatial`
  and `saturation_beta`, rejects NaN/Infinity (also in `--config` JSON), and lowercases
  `outline`. Palette files must be `.hex`/`.gpl`, under 64 KiB; errors name the line number,
  never the line.
- `--g-T-final` is now `--g-t-final`; the old spelling remains as a hidden alias.
- The test suite fails with a pointer to `scripts/make_fixtures.py` when a fixture is missing
  instead of regenerating it inside the repository.
- gerstner uses the shared `downscale.neighbor_mean4` and `box.cell_index` helpers (output
  unchanged); the remaining spec deviations carry `# DEVIATION:` comments.

- Downscalers weight colors by input alpha; semi-transparent fringes pull less (opaque
  inputs are unchanged).
- `Config.hash()` covers the palette's colors, so editing a palette file changes the hash.
- `batch` passes its shared palette to frames inline; frame metadata no longer holds `--outdir`.
- The sprite preset no longer enables `remove_bg` when rembg is installed; use `--remove-bg`.
- `_palette.json` LAB values are rounded to 6 decimals.
- The package version is single-sourced from `pixelforge/version.py`.
- README: byte identity across machines requires `git clone && uv sync --locked`.
- Kopf no longer grows σ for kernels cut by the alpha silhouette or the image border, which
  blurred features near sprite edges below the box filter's sharpness, and kernels whose
  cell is mostly transparent no longer claim (and then drop) pixels near the silhouette, which
  erased lines 2–3 px inside it. Kopf's alpha mask is now the box filter's cell rule.

### Removed

- Duplicate `dev` optional extra; use the `dev` dependency group (`uv sync`).

### Library API

- `pixelforge.run` accepts a path, a `PIL.Image.Image` or a numpy (H, W, 3|4) uint8 / [0, 1]
  float array; in-memory inputs hash shape plus RGBA bytes for `input_sha256`.
- `ConfigError` (subclass of `PixelforgeError` and `ValueError`) for every config validation
  failure; `PixelforgeError`, `ConfigError` and `run_loaded` are exported from `pixelforge`.
- An unknown `palette_name` fails in `Config(...)`; an unreadable palette file fails at the
  start of `run`, before preprocessing.
- `--remove-bg` without rembg is a config error (exit 2).
- `PRESETS` and `PRESET_LONGEST_EDGE` are read-only mappings.
- `box.run` and `gerstner.run` raise `PixelforgeError` on an all-transparent mask.
- `--debug` / `PIXELFORGE_DEBUG=1` prints the full traceback on CLI errors.


## [0.1.0]

- Initial prototype: box, kopf and gerstner downscalers, palette and post-processing pipeline,
  tileset/tilemap extraction, bundled hardware palettes, `pixelforge` CLI.
