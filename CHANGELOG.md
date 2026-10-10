# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

- `--g-T-final` is now `--g-t-final`; the old spelling remains as a hidden alias.

- Downscalers weight colors by input alpha; semi-transparent fringes pull less (opaque
  inputs are unchanged).
- `Config.hash()` covers the palette's colors, so editing a palette file changes the hash.
- `batch` passes its shared palette to frames inline; frame metadata no longer holds `--outdir`.
- The sprite preset no longer enables `remove_bg` when rembg is installed; use `--remove-bg`.
- `_palette.json` LAB values are rounded to 6 decimals.
- The package version is single-sourced from `pixelforge/version.py`.
- README: byte identity across machines requires `git clone && uv sync --locked`.
- Kopf no longer grows σ for kernels cut by the alpha silhouette or the image border, which
  blurred features near sprite edges below the box filter's sharpness.

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

<!-- Later commits on this branch are appended here by the final integrator. -->

## [0.1.0]

- Initial prototype: box, kopf and gerstner downscalers, palette and post-processing pipeline,
  tileset/tilemap extraction, bundled hardware palettes, `pixelforge` CLI.
