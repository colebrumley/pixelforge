# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-11

pixelforge is now a single Rust program, installed with `cargo install --git`, with no
Python at runtime; the Python library API lives on as an optional binding. The port was done
stage by stage against the last Python version (9f53516, which still reported 0.1.0), and
its outputs are pixel-identical on the whole 48-case snapshot matrix and every golden case,
except one gerstner case that sits on a near-tie (10 of 65,536 pixels) and outputs from JPEG
inputs (a different decoder). Every `config_hash` changed with the version.
`docs/rust-migration-plan.md` records the port phase by phase. This release also carries
the Python improvements made since the 0.1.0 prototype (the photo pre-pass, proof sheets,
Tiled output, the algorithm fixes below), which do change outputs against 0.1.0.

### Added

- The `pixelforge` program in Rust (`crates/pixelforge`: the library, and the CLI behind the
  default `cli` feature). `cargo install --git https://github.com/colebrumley/pixelforge
  pixelforge` builds it with only a Rust toolchain; the first build takes a minute or two.
- The Rust library API: `pipeline::run(Input, &Config, max_pixels)` for a path or an
  in-memory `(H, W, 3|4)` u8 or `[0, 1]` f64 array (hashed as shape plus RGBA bytes),
  `run_loaded`, `Output::save(prefix)`, `output_paths`, `environment()`; `Config::new`,
  `replace`, `canonical_json`, `hash`, `provenance`, `validate_palette`; `config::PRESETS`;
  `Error::Config` (exit 2 in the CLI), `Error::Pixelforge` and `Error::Io` (exit 1).
  Progress goes through the `log` crate.
- The optional Python binding `crates/pixelforge-py` (PyO3 + maturin, numpy only; Pillow
  optional for PIL inputs): `Config`, `run`, `run_loaded`, `load`, `from_image`, `Result`
  with `save` (the files `pixelforge convert` writes, byte for byte), `PRESETS`,
  `PixelforgeError`, `ConfigError`, `__version__`. Install with `pip install
  'git+https://github.com/colebrumley/pixelforge#subdirectory=crates/pixelforge-py'`; its smoke
  tests run in CI.
- `PIXELFORGE_THREADS`: worker threads (default: the available parallelism). Threaded work
  uses a fixed partition merged in a fixed order, so results are bitwise identical for any
  thread count. `_meta.json` records it as `environment.threads` (never hashed).
- The `xtask` developer tooling, `cargo xtask fixtures|golden|bench|snapshot|gallery|help-record`
  (built with the release profile). `fixtures` draws the seven fixtures pixel-identical to the
  committed PNGs and `--check` compares them; `golden` regenerates `tests/golden.json`
  (`--check` reports drift); `bench` times a case matrix per stage and writes
  `bench/<git sha>.json` in the Python schema (`python`/`numpy`/`scipy` as `null`, plus
  `rustc` and `pixelforge`), and `--compare` also reads the Python-written files;
  `snapshot save|compare` reports per case whether outputs are identical, the differing
  pixels and their ΔE (snapshots are JSON now; directories saved by the old Python script
  must be re-saved); `gallery` writes the comparison page; `help-record` records each
  command's `--help` in `tests/cli_help/`, which the CLI tests compare with the live output.
  `bench/0edd462.json` (pure Python) and `bench/9f53516.json` (Python with the Rust hot
  loops) are the baselines.
- The Rust test suite: 434 tests (3 ignored as slow), replacing the Python suite, with a
  file-by-file mapping in `crates/pixelforge/tests/README.md`: per-module behavioural
  tests, CLI integration tests (`assert_cmd`), determinism in-process and across fresh
  processes at 1, 2, 3 and 8 threads, `tests/golden.json` consumed by `golden.rs`, and
  `tests/oracle/` (1.7 MB of reference outputs recorded from Python: config canonical JSON,
  hashes and all 104 error messages, sRGB↔LAB, distance-transform indices, scipy's filters,
  skimage's bilateral, percentiles, 10,000 float `repr`s, two small pipeline runs).
  `determinism_guard.rs` fails the build on any platform transcendental function, `powi`,
  `mul_add` or `.round()` in the sources.
- `--enhance` (with `--enhance-radius`, `--enhance-black`, `--enhance-white`) and `--ink`: an
  opt-in pre-pass before downscaling for photos. `enhance` stretches the subject's brightness
  and adds local contrast at about two output pixels, so eyes, mouths and other sub-pixel
  features survive; `ink` darkens thin dark features. Hue is kept, the stretch gain is capped,
  seamless tiles stay seamless and `batch` shares one pair of levels between frames. Off by
  default and in every preset.
- `compare --prepass` and `compare --sweep FIELD=V1,V2,...` write a labelled proof sheet
  (methods × settings) and report the `convert` flags and the full config of each cell;
  `--method` limits the sheet to one row.
- `fit` (`--fit pad|stretch|crop`) for tileset sizes; the default `pad` keeps the aspect ratio
  instead of stretching to a tile multiple, the longest edge never exceeds the target, and a
  `tile_size` above it is an error.
- Tilesets also write `NAME.tmj` (Tiled JSON with GID flip bits and an embedded tileset) and
  `NAME_tilemap.csv`; `--tileset-columns` (default `ceil(sqrt(n))`, was 16);
  `--transparent-index first` puts the transparent entry at PNG index 0; `_palette.json`
  marks `"used"` entries and `stats.colors_unused` counts the rest.
- `--scale N` and `--canvas WxH`.
- `Config::provenance()`; PNG `tEXt` chunks `pixelforge:palette_sha256` and
  `pixelforge:palette`; `_meta.json` fields `palette_sha256`, `palette`, `environment` and,
  with `--remove-bg`, `remove_bg_versions`.
- Inline palettes: `--palette-name hex:rrggbb,rrggbb,...`.
- Logging: one INFO line per pipeline stage and kopf/gerstner iteration progress (DEBUG each,
  INFO every 10th); CLI `-v/--verbose` (`-v` stages, `-vv` iterations) and `-q/--quiet`;
  messages go to stderr as `pixelforge: ...`, stdout stays JSON-only.
- A warning when `key_bg` finds no flat background on an opaque input, suggesting
  `--remove-bg` or an alpha channel.
- `--help` text, per-preset defaults and allowed values for every flag, generated from one
  field table that also drives the config, its JSON and `--sweep`.
- `--debug` / `PIXELFORGE_DEBUG=1` prints the full error chain on CLI errors.
- README: a photo sample (`docs/sample_photo.png`, from an AI-generated portrait) and
  guidance on source photos.
- MIT `LICENSE`, GitHub Actions CI (Rust lint, tests on Linux x86-64 and macOS arm64 with the
  slow group, `xtask golden --check` and `fixtures --check`, a smoke `convert`, and the
  binding's smoke tests on Linux) and this changelog.

### Changed

- The CLI is the Rust program, with the Python CLI's commands, flags, outputs, stdout JSON,
  exit codes and messages: `--config` is read by a port of CPython's JSON scanner (same
  values, `NaN`/`Infinity` rejection and syntax error messages), and the canonical config
  JSON reproduces Python's `json.dumps` and float `repr` byte for byte. Deliberate
  differences: `--help` is laid out by clap (option names and defaults are the same), usage
  errors use clap's wording, and library warnings (multi-frame input, the stretch aspect)
  print once each as `pixelforge: warning: …` and respect `-q`.
- Image decoding uses the pure-Rust `image` crate (and `png`, `gif`, `tiff`, `miniz_oxide`
  directly), reproducing Pillow 12's `open` + `exif_transpose` + 8-bit conversion: PNG in
  every bit depth and colour type, GIF, BMP, TIFF, lossless WEBP and EXIF/XMP orientations
  1–8 are bit-identical, lossy WEBP matched on every file tested. JPEG (zune-jpeg instead of
  libjpeg-turbo) differs: at quality 90 with 4:2:0 chroma by at most 7 levels on 0–53 % of
  pixels, with 4:4:4 by at most 3 on ≤ 1.6 %, greyscale by at most 1 on ≤ 0.6 %. Pillow's
  decompression-bomb limit is kept. PNGs are written 8-bit indexed (Pillow wrote 1/2/4-bit
  for small palettes): the bytes differ, decoded pixels, palette, transparency and text do
  not.
- Numerics: every routine ported formula for formula with `libm` transcendentals and
  half-to-even rounding, so results are the same on every platform. Bit-identical to the
  Python version: config, box, kopf (LAPACK's 2×2 eigensolver path reproduced with the two
  fused multiply-adds Accelerate performs, as `libm::fma`), quantization, median cut (numpy's
  pairwise summation reproduced), post-processing, tiles, masks with scipy's exact distance
  transform (ties included), layout, pre-reduction, uniform and median filters, percentiles.
  Within rounding: sRGB↔LAB (5.7e-14 from scikit-image; moved 10 of 65,536 pixels, ΔE 4.03,
  on `noisy_gradient`/background/gerstner, a near-tie in the annealer), gerstner's BLAS
  products now sequential sums and its 3×3 eigenvectors a Jacobi solver (palettes within
  3.3e-11, no index changed), hue ramps (7e-14), the Gaussian blur and the bilateral denoise
  (a few ulps from FMA-free scipy/scikit-image; the macOS wheels fuse multiply-adds, which
  moves 4 pixels of a 1024² photo by up to 5.4e-6 before rounding; no 8-bit value changed).
- Seeded gerstner runs draw their perturbation from SplitMix64 instead of numpy's PCG64:
  still deterministic and different from the unseeded run, but not the Python version's
  output for the same seed.
- `compare`'s proof-sheet labels use an embedded CC0 5×9 bitmap font instead of Pillow's
  default font, so label pixels and the sheet size differ by a few pixels; every cell is
  identical.
- `--remove-bg` runs the external `rembg` command (`rembg i IN OUT`, found on `PATH`;
  `pipx install 'rembg[cli]'`) instead of the rembg Python package; a missing command is
  the same config error (exit 2), and `_meta.json` records `rembg --version` with
  `onnxruntime: null`.
- `_meta.json`'s `environment` is `platform`, `threads` and `native` (the pixelforge version)
  instead of the Python library versions.
- Version 0.2.0, single-sourced from the workspace `Cargo.toml` (`pixelforge --version`,
  `_meta.json`, the config hash; the binding's `pyproject.toml` repeats it and a test checks).
  Every `config_hash` changed; no `pixels_sha256` in `tests/golden.json` did.
- Speed (whole runs on a loaded 10-core M1 Max, background preset): a 1024² photo to 256²
  with box 1.64 → 0.32 s, kopf 33.9 → 1.78 s, gerstner 31.0 → 3.80 s against pure Python
  (DESIGN.md has the table). Preprocessing of that photo 2.37 → 0.26 s (the bilateral denoise
  is threaded: 0.40 s on one thread against scikit-image's 1.39 s). Tiling is 50–100× faster
  on large outputs.
- Inputs much larger than the output are box-reduced by an integer factor to at most
  `prereduce_max_ratio` (8) times the output size before denoising and downscaling (nearly
  lossless at these ratios); with kopf's leaner buffers, kopf 1024² → 64² went from ≈ 200 s
  and 550 MiB to ≈ 27 s and 67 MiB in Python, before the port.
- Image loading only accepts PNG, JPEG, GIF, WEBP, BMP and TIFF, refuses non-regular files,
  hashes inputs in a stream, and enforces a pixel budget checked from the header
  (`--max-input-pixels`, default 24 Mpx).
- `Config` has upper bounds on sizes, iteration caps, `scale_preview`,
  `denoise_sigma_spatial` and `saturation_beta`, rejects NaN/Infinity (also in `--config`
  JSON), and lowercases `outline`. Palette files must be `.hex`/`.gpl`, under 64 KiB; errors
  name the line number, never the line. An unknown `palette_name` fails when the config is
  built; an unreadable palette file fails at the start of `run`, before preprocessing.
- `--g-T-final` is now `--g-t-final`; the old spelling remains as a hidden alias.
- Downscalers weight colors by input alpha; semi-transparent fringes pull less (opaque inputs
  are unchanged).
- The config hash covers the palette's colors, so editing a palette file changes it.
- `batch` passes its shared palette to frames inline; frame metadata no longer holds
  `--outdir`.
- The sprite preset no longer enables `remove_bg` when rembg is installed; use `--remove-bg`.
- `_palette.json` LAB values are rounded to 6 decimals.
- Kopf no longer grows σ for kernels cut by the alpha silhouette or the image border, which
  blurred features near sprite edges below the box filter's sharpness, and kernels whose cell
  is mostly transparent no longer claim (and then drop) pixels near the silhouette, which
  erased lines 2–3 px inside it. Kopf's alpha mask is now the box filter's cell rule.
- The binding's `Config` wraps the Rust one: a list given as `preset` is a `ConfigError`
  (Python raised `TypeError`), and numpy scalar field values are no longer accepted.

### Removed

- The Python package `pixelforge/` and its Python CLI, the Python test suite, the Python
  scripts (`scripts/*.py`, replaced by `cargo xtask`), `pyproject.toml` and `uv.lock`; numpy,
  scipy, scikit-image, Pillow and click as dependencies, and the `bg` extra (rembg is now an
  external command). The Python binding has its own `pyproject.toml` in
  `crates/pixelforge-py`. The two scripts that recorded `tests/oracle/` and the Unicode
  tables are kept next to their outputs as documentation.

### Fixed

- `convert`, `batch` and `compare` refuse (exit 2) to write an output over an input file, and
  `batch` refuses inputs that share a stem; `--force` overrides both. `batch` skips
  `*_preview`, `*_tileset` and `*_compare` images.
- With a named palette (`--palette-name`) the outline snaps to the nearest palette entry
  instead of adding a colour outside the hardware palette; a `_palette.hex` fed back through
  `--palette-name` reproduces itself. `stats.outline_index` names the entry used.
- `Config::replace` with a new preset re-applies it to every field the caller did not set
  explicitly; previously the old preset's values were kept under the new label.
- The output canvas is exactly the requested size: the outline ring is reserved inside it
  (`stats.outline_margin`) instead of growing the image by 2 px; with `tileset=True` a
  clipped outline is reported as `stats.outline_clipped`.
- Dithering with gerstner (the default method) now dithers the smoothed superpixel means;
  it was a no-op because the per-pixel values already sat on palette colours. `auto` dither
  ignores transparent cells when measuring local detail, so silhouette edges dither too.
- `batch` derives one scale from the largest frame's subject and gives every frame the same
  canvas, so animation frames keep a constant size.
- Tile de-duplication requires both a mean ΔE below `tile_dedupe_tolerance` and every
  pixel's ΔE below 10, and merges into the nearest tile; sparse details (stars, highlights)
  are no longer erased or stamped into every cell. `stats.tiles_rerender_px_changed` reports
  what the re-render changed.
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
- `--remove-bg` without rembg is a config error (exit 2); box and gerstner report an error
  on an all-transparent mask.
- Writing a `.hex` palette with values outside 0..255 is an error instead of a malformed
  line.

## [0.1.0]

- Initial prototype: box, kopf and gerstner downscalers, palette and post-processing pipeline,
  tileset/tilemap extraction, bundled hardware palettes, `pixelforge` CLI.
