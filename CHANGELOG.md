# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The Config in Rust (`crates/pixelforge/src/config/`, Phase 2 of
  `docs/rust-migration-plan.md`): construction from presets and explicit values, Python's
  coercion and validation with byte-identical error messages, `replace`, the palette
  loaders (bundled palettes embedded, `.hex`/`.gpl` files, inline `hex:`), and a canonical
  JSON writer reproducing `json.dumps(sort_keys=True, separators=(",", ":"))` with
  Python's float `repr`, so `canonical_json()` and `hash()` match the Python Config byte
  for byte. Rust tests cover the whole oracle corpus, all 104 recorded error messages and
  10,000 float reprs; `tests/test_config_native.py` compares the new `_native.config_*`
  functions and `palette_hex_lines` with the live Python `Config`. The Python `Config` is
  still the one the pipeline uses.
- The package now builds a Rust extension, `pixelforge._native`, with maturin (binding
  crate `crates/pixelforge-py`), so installing from the repo requires a stable Rust
  toolchain alongside Python 3.11+. The extension holds the kopf and gerstner per-pixel loops
  (see Changed).
- `scripts/bench.py` (per-stage timings per input/preset/method, JSON with git sha and
  environment, `--compare OLD.json`) and `scripts/snapshot_outputs.py` (`save`/`compare` of
  output pixels and palettes with differing-pixel counts and ΔE). `bench/0edd462.json` is the
  pure-Python baseline on an Apple M1 Max for the Rust port (`docs/rust-port-plan.md`).
- `tests/oracle/`, recorded by `scripts/record_oracles.py`: Python reference outputs, as
  plain JSON, for the Rust migration (`docs/rust-migration-plan.md`): config canonical JSON,
  hashes and error messages, sRGB↔LAB, `distance_transform_edt` indices, skimage's
  bilateral denoise, scipy's 1-D filters, percentiles, the `repr` of 10,000 floats and two
  small pipeline runs' text outputs. `tests/test_oracle.py` re-records it and requires
  byte-identical files.
- `--enhance` (with `--enhance-radius`, `--enhance-black`, `--enhance-white`) and `--ink`: an
  opt-in pre-pass before downscaling for photos. `enhance` stretches the subject's brightness
  and adds local contrast at about two output pixels, so eyes, mouths and other sub-pixel
  features survive; `ink` darkens thin dark features. Hue is kept, the stretch gain is capped,
  seamless tiles stay seamless and `batch` shares one pair of levels between frames. Off by
  default and in every preset, so output pixels are unchanged; every `config_hash` changes
  because `Config` has five new fields.
- `compare --prepass` and `compare --sweep FIELD=V1,V2,...` write a labelled proof sheet
  (methods × settings) and report the `convert` flags for each cell; `--method` limits the
  sheet to one row. Each cell also reports its full config. Plain `compare` is unchanged.
- README: a photo sample (`docs/sample_photo.png`, from an AI-generated portrait) and guidance
  on source photos.
- MIT `LICENSE` and package metadata (license expression, author, classifiers, project URLs).
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
- The Config field table in Rust (`crates/pixelforge/src/config.rs`): every field's name,
  type, default, help text, choices and upper bound, the presets and the per-preset longest
  edge, transcribed from `pixelforge/config.py`. It is the single source the Rust `Config`,
  JSON and CLI will be built from; `_native.config_fields()` and `_native.config_presets()`
  expose it and `tests/test_field_table.py` fails on any drift from the Python tables.

### Changed

- Post-processing and tiles are native (`crates/pixelforge/src/postprocess.rs` and
  `tiles.rs`, migration plan Phase 3): saturation, orphan removal (own 8-connected
  labelling instead of `scipy.ndimage.label`), jaggies, the outline, tile extraction with its
  near-duplicate search, reconstruction, the tileset sheet, Tiled `.tmj` and CSV. Palette
  ramps still run in Python before the native call. Bitwise identical to the numpy version
  (`tests/test_postprocess_native.py`, `tests/test_tiles_native.py` keep it as the
  reference); the snapshot matrix and goldens are unchanged.
- Colour conversion is native (`crates/pixelforge/src/color.rs`, migration plan Phase 2):
  `pixelforge.color`'s `rgb_to_lab`, `lab_to_rgb`, `rgb8_to_lab`, `lab_to_rgb8` and
  `nearest_index` call `pixelforge._native` instead of scikit-image and scipy's `cKDTree`.
  The conversions are scikit-image 0.26's formulas with `libm`'s `pow` and `cbrt` (bit-identical
  on every platform) and agree with skimage to 1.1e-13 in LAB; `lab_to_rgb8` rounds half to
  even like `np.round`. The nearest colour is a brute-force CIE76 search whose ties go to the
  lowest palette index (cKDTree's tie-break was unspecified). Snapshot compare against the
  previous version on the 48-case matrix: 1 case changed, `noisy_gradient/background/gerstner`,
  10 of 65,536 pixels (all ΔE 4.03, palette unchanged); no other case moved and the goldens
  are unchanged. The cause is the last-bit LAB difference: jittering skimage's own LAB by one
  ulp moves the same 10 pixels, so this case sits on a near-tie in gerstner, not a formula
  difference.
- The Rust code is a Cargo workspace at the repo root (`Cargo.toml`, one `Cargo.lock`,
  `target/`) with two crates: `crates/pixelforge`, a plain Rust library without Python (the
  kopf and gerstner kernels and `parallel`, taking slices), and `crates/pixelforge-py`, the
  PyO3 binding built as `pixelforge._native` that checks arguments and calls the library.
  It replaces the single crate that was in `rust`. Same Python-facing functions, signatures
  and error messages; no output changed (`scripts/snapshot_outputs.py compare`: every
  fixture case identical).
- kopf's per-pixel loops run natively (`pixelforge._native`,
  `crates/pixelforge/src/kopf.rs`): the E-step fused with the M-step sums, the C-step's
  directional and overlap sums, and the final spatial-only recoloring of starved kernels.
  The per-kernel algebra, constraints and convergence stay in Python. kopf is much faster,
  and multithreaded (timings: TODO, measure on an idle machine). The numpy implementation is
  kept in `pixelforge/downscale/_reference/kopf.py` for tests and benchmarks only.
- New environment variable `PIXELFORGE_THREADS`: worker threads for the native kernels
  (default: the available parallelism). It affects speed only; results are bitwise
  identical for any thread count (fixed 32-row bands merged in a fixed order). `_meta.json`
  records the thread count used as `environment.threads` (recorded, never hashed, like the
  library versions); `pixelforge._native.thread_count()` returns it.
- The native sums accumulate in a different order than numpy's block sums and use `libm`'s
  `exp`, so kopf's internal values differ from the numpy version at ~1e-15 relative. No
  output changed: every case of `scripts/snapshot_outputs.py compare` against the
  pure-Python baseline is pixel-identical, and `scripts/update_golden.py` regenerated
  `tests/golden.json` unchanged.
- gerstner: the SLIC assignment pass and the superpixel bilateral filter run in the native
  extension (`_native.slic_assign`, `_native.bilateral_filter`,
  `crates/pixelforge/src/gerstner.rs`) on `PIXELFORGE_THREADS` threads (default: available
  parallelism; the thread count never changes the output). The palette annealer, Laplacian
  smoothing and control flow stay in Python; the numpy kernels are kept in
  `pixelforge/downscale/_reference/gerstner.py` for tests and benchmarks. The assignment is
  bitwise identical to numpy; the bilateral filter uses `libm::exp`, which differs from
  numpy's `exp` at rounding level, so a few output pixels can change (10 of 65,536 on
  noisy_gradient at the background size, none on the other fixtures). The goldens were
  regenerated and are unchanged. Timings: TODO, measure on an idle machine.
- gerstner: the per-iteration superpixel statistics (pixel count, mass, colour sums and
  centre sums, seven `np.bincount` calls) are one native pass, `_native.superpixel_stats`.
  It sums each superpixel in pixel order like `np.bincount`, so the results are bitwise
  identical: every case of `scripts/snapshot_outputs.py compare` is unchanged and the goldens
  did not change. The pass is single-threaded on purpose (per-band partial sums would change
  the rounding). Timings: TODO, measure on an idle machine.

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
- Kopf folds per-kernel block sums into place through slices instead of `np.ix_` gathers
  and scatters (output bitwise unchanged): portrait background 39.4 s → 35.2 s here.
- gerstner's bilateral filter computes the symmetric colour/spatial weight once per ± offset
  pair and shifts it for the mirror offset (output bitwise unchanged): 67 → 50 ms per call
  on 256×256.

- Downscalers weight colors by input alpha; semi-transparent fringes pull less (opaque
  inputs are unchanged).
- `Config.hash()` covers the palette's colors, so editing a palette file changes the hash.
- `batch` passes its shared palette to frames inline; frame metadata no longer holds `--outdir`.
- The sprite preset no longer enables `remove_bg` when rembg is installed; use `--remove-bg`.
- `_palette.json` LAB values are rounded to 6 decimals.
- The package version lives in `pixelforge/version.py`, mirrored in `pyproject.toml` and
  the root `Cargo.toml` (`[workspace.package]`); `tests/test_version.py` keeps the three equal.
- README: byte identity across machines requires `git clone && uv sync --locked`.
- Kopf no longer grows σ for kernels cut by the alpha silhouette or the image border, which
  blurred features near sprite edges below the box filter's sharpness, and kernels whose
  cell is mostly transparent no longer claim (and then drop) pixels near the silhouette, which
  erased lines 2–3 px inside it. Kopf's alpha mask is now the box filter's cell rule.

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
