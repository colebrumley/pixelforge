# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The `pixelforge` command line in Rust (Phase 7, part A of `docs/rust-migration-plan.md`):
  `crates/pixelforge/src/main.rs` and `src/cli/` (feature `cli`, on by default; the Python
  binding builds the library without it, so clap never reaches the extension) port
  `convert`, `batch` and `palettes` from `pixelforge/cli.py`, with `compare` present as a
  stub until its proof sheet is ported. The options are generated from the field table with
  clap 4.5 (`--flag`/`--no-flag` pairs where the last one wins, the hidden `--g-T-final`,
  per-preset defaults in `--help`, click's `int()`/`float()` conversions and `click.Path`
  checks, usage errors exit 2, `--version` prints `pixelforge, version 0.1.0`); `--config`
  is read by a port of CPython 3.11's JSON scanner, so values, `NaN`/`Infinity` rejection and
  every syntax error message are Python's; the collision check resolves paths like
  `Path.resolve()`; config errors exit 2 and other errors 1 with the same messages, `--debug`
  / `PIXELFORGE_DEBUG` print the error chain. Library `UserWarning`s (multi-frame input, the
  stretch aspect) are printed once each as `pixelforge: warning: …` and, unlike Python's
  `warnings`, respect `-q`. `crates/pixelforge/tests/cli.rs` ports the CLI cases of
  `test_cli.py`, `test_pipeline.py` and `test_determinism.py` (fresh processes at 1, 2, 3 and
  8 threads); `tests/test_cli_parity.py` runs both CLIs on every fixture (box, plus kopf,
  gerstner and three `batch` runs) and finds equal PNG pixels and text, byte-identical
  palette, tilemap and Tiled files, equal `_meta.json`, stdout JSON and `-v` stderr (timings
  and environment aside).
- Preprocessing filters and the enhance pass in Rust (Phase 4 of
  `docs/rust-migration-plan.md`): `filters.rs` ports
  scipy's `uniform_filter1d` (any size, `nearest`/`wrap`), `gaussian_filter1d`, the 3×3
  `median_filter` and scikit-image 0.26's `denoise_bilateral` (`mode="edge"`, its lookup
  tables as written, including the spatial table's `-win_size // 2` grid; threaded per pixel
  band, 0.40 s at 1 thread vs skimage's 1.39 s on a 1024² image); `enhance.rs` ports
  `np.percentile`, the levels, `_stretch`, `_gaussian`, `_blur` and `enhance`. Uniform and
  median filters, percentiles, `_stretch` and the running-mean blur are bitwise equal to
  numpy/scipy; the Gaussian filter (≤ 2.2e-16), the bilateral (≤ 7.8e-16 except where
  skimage's FMA-contracted colour distance crosses a lookup bin: 4 pixels of the 1024²
  portrait, ≤ 5.4e-6) and `enhance` (≤ 4.2e-15) differ in the last bits only, because the
  macOS arm64 scipy/skimage wheels fuse multiply-adds and numpy's `exp`/`expm1` are not
  `libm`'s; no 8-bit value changes. Exposed as `_native.uniform_filter1d`,
  `gaussian_filter1d`, `median_filter3`, `denoise`, `percentile`, `enhance_levels`,
  `enhance` and `subject_brightness_from`, tested in `tests/test_filters_native.py` and
  `tests/test_enhance_native.py` and, in Rust, against the oracle files.
- Preprocessing's masks and geometry in Rust (Phase 4, part A of
  `docs/rust-migration-plan.md`): `crates/pixelforge/src/mask.rs` (scipy's exact Euclidean
  feature transform, `NI_EuclideanFeatureTransform` reproduced pass for pass so the opaque
  pixel each transparent one copies from is scipy's, ties included; transparent filling,
  4-connected labelling, 8-neighbour dilation, flat-background keying with fringe
  unmixing, the subject mask, weight and crop box) and `crates/pixelforge/src/layout.rs`
  (outline margin, `resolve_layout` with the tileset fit and its warning, `fit_crop`, the
  integer repeat and its size limit, the box pre-reduction with numpy's summation order,
  the seamless wrap padding). Bitwise identical to the Python functions on the EDT
  oracle, 200 random masks, the fixtures and samples, and 1,728 configs × 14 input sizes;
  exposed as `_native.edt`, `subject`, `resolve_layout`, `prereduce` and friends for
  `tests/test_mask_native.py` and `tests/test_layout_native.py`.
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

- The pipeline is native (migration plan Phase 6, part B): `pixelforge.run` / `run_loaded`
  call `crates/pixelforge/src/pipeline.rs` (`pipeline::run`, `run_loaded`) through
  `_native.pipeline_run` / `pipeline_run_loaded`, and `Result.save` writes every output file
  from Rust (`Output::save`, `_native.result_save`); `output_paths` and
  `uses_gerstner_palette` are native too. Outputs are unchanged: every Result field, the
  stats (key order included), the log lines and every saved file match the former Python
  orchestration (kept as `tests/pipeline_reference.py`; `tests/test_pipeline_native.py`),
  the 48-case snapshot matrix is identical and the goldens are unchanged, and the new Rust
  test `crates/pixelforge/tests/pipeline_oracle.rs` reproduces `tests/oracle/outputs_small`
  with no Python involved. `remove_bg` in the Rust library runs an external `rembg i IN OUT`
  command (Decision 5; a missing command is the same `ConfigError`, and `_meta.json`
  records `rembg --version` with `onnxruntime: null`); the Python package still uses the
  `rembg` package and passes the matted alpha on (`alpha_matted`), recording the package
  versions as before. `_meta.json`'s `environment` from the Python package keeps its keys;
  the Rust library's own `environment()` has `platform` (spelled as Python's
  `sys.platform` + `platform.machine()`), `threads` and `native` (the crate version).
  Tests that spied on `palette.mcda`, `downscale.get`, `postprocess.run` or `tiles.extract`
  through the pipeline now observe them in `tests/pipeline_reference.py` (or the stage log
  line), since the native pipeline no longer calls the Python stage modules.
- Image I/O is native (migration plan Phase 6, part A): `pixelforge.io`'s `load`,
  `open_regular`, `sha256_file`, `from_image` (arrays; PIL images are converted in Python as
  before, then hashed natively), `from_rgba`, `indices_to_rgba`, `upscale_nearest`,
  `save_png` and `read_png_text` call `crates/pixelforge/src/io.rs` through `_native.io_*`;
  Pillow is no longer imported by `pixelforge.io` except for PIL-image inputs. Decoding
  reproduces Pillow 12's `open` + `exif_transpose` + `_to_8bit` + `convert("RGBA")`
  bitwise (pixels and sha256, checked against the old module, kept as
  `tests/io_reference.py`, in `tests/test_io_native.py`) for PNG in every bit depth and
  colour type (interlaced, tRNS palettes and colour keys compared as Pillow compares them,
  16-bit RGB/RGBA/grey+alpha reduced to the high byte as Pillow's `;16B` unpackers do,
  16-bit grey rescaled by 1/257), GIF (first frame on Pillow's canvas), BMP, TIFF (1/8/16/32-bit
  grey, palette, LA, RGB, RGBA incl. associated alpha, CMYK; raw, LZW, deflate, packbits),
  lossless WEBP, and EXIF/XMP orientations 1–8; frame counts come from headers only. Lossy
  WEBP decodes identically to libwebp on the fixtures and samples. JPEG (zune-jpeg vs
  libjpeg-turbo) does not: at quality 90 with 4:2:0 chroma the fixtures and samples differ
  by at most 7 levels on 0–53% of pixels (2 on `checker_tiles`, 7 on `town_painted`), with
  4:4:4 chroma by at most 3 on ≤ 1.6%, greyscale by at most 1 on ≤ 0.6%. Errors keep their
  classes and messages (budget, Pillow's decompression-bomb limit, non-regular files,
  unsupported content, `OSError`s); undecodable files keep the `could not decode image (`
  prefix with the Rust decoder's reason. `save_png` writes 8-bit indexed PNGs (Pillow wrote
  1/2/4-bit for small palettes): the bytes differ, the decoded mode, palette, transparency,
  pixels and text are the same. `MAX_INPUT_PIXELS` moved from `layout.rs` to `io.rs`, and
  `palette.rs` uses `io::open_regular`. New library dependencies (exact pins): `image`
  0.25.10 (no default features; png, jpeg, gif, webp, bmp, tiff), and the decoders it
  already pulls in, called directly where `image` hides what Pillow exposes: `png` 0.18.1,
  `gif` 0.14.2, `tiff` 0.11.3, `miniz_oxide` 0.8.9.
- The box and kopf downscalers are native (migration plan Phase 5, part A):
  `pixelforge.downscale.box.run` calls `_native.box_run`
  (`crates/pixelforge/src/boxfilter.rs`) and `kopf.run` calls `_native.kopf_run`
  (`crates/pixelforge/src/kopf_run.rs`: initialisation, window/truncation/ghost bookkeeping,
  M-step and C-step algebra, shape constraints, starved re-seeding, convergence, the final
  refill, over the existing per-pixel loops). Both are bitwise identical to the Python
  versions: the 48-case snapshot matrix (fixtures, samples, portrait, knight; box, kopf,
  gerstner × sprite/background) is identical and every golden passes unchanged. kopf
  reproduces numpy's macOS (Accelerate) routines exactly: `np.linalg.eigh` on 2×2 is
  LAPACK's `dsteqr`/`dlaev2` path with the two fused multiply-adds Accelerate performs
  (bitwise on 20,000 random SPD matrices), the `(V·w) @ Vᵀ` reconstruction fuses as its
  BLAS does, `np.hypot` is `sqrt(fma(small, small, big²))` (bitwise on 200,000 samples;
  `libm::hypot` differs on 10 %), and `1.1 ** fires` is a table of correctly rounded
  powers (`libm::pow` is 1 ulp off at n = 4). The fused multiply-adds are `libm::fma`, so
  results are the same on every platform; on numpy builds with another LAPACK the old
  Python kopf differed from this in the last bits. kopf's progress lines now come from the
  `pixelforge.kopf` logger (was `pixelforge.downscale.kopf`), same text. Downscale time on
  the 1024² portrait (background preset, kopf): 2.61 s → 1.68 s on a loaded machine (load
  average 16–22). Parity tests: `tests/test_box_native.py`, `tests/test_kopf_run_native.py`.
- gerstner's control flow and the MCDA palette are native (migration plan Phase 5, part B):
  `crates/pixelforge/src/annealer.rs` (`PaletteAnnealer`, `principal_axis`) and
  `gerstner_run.rs` (the grid, INITIALIZE, the iteration loop over the existing SLIC,
  statistics and bilateral loops, the transparent attribution); `pixelforge.downscale.gerstner.run`
  calls `_native.gerstner_run`, `palette.mcda` calls `_native.mcda`, and the Rust
  `palette::build` handles the mcda source. Ported operation for operation; three routines
  numpy hands to BLAS/LAPACK are replaced: `centered.T @ centered` and `weighted.T @ points`
  are sequential sums (Accelerate's differed from them on 89% and 35% of the entries, up
  to 217 ulp on the products), `np.linalg.eigh` is a cyclic Jacobi solver (principal axis
  within 2.8e-13, variance within 3.8e-13 relative), and `exp`/`log` are `libm`'s (1 ulp
  from numpy's on ~10% of arguments). On the fixtures at 16/32 px × K 4/8/16 and the preset
  sizes every index and mask is identical and palettes move by at most 3.3e-11; with those
  routines swapped into the old Python code the results are bit for bit identical
  (`tests/test_gerstner_run_native.py`). Snapshot compare against 1e69f8f: all 48 cases
  identical, `tests/golden.json` unchanged. A seeded run now draws its perturbation from
  SplitMix64 instead of numpy's PCG64 stream: it stays deterministic and still differs from
  the unseeded run, but seeded outputs differ from the Python version's for the same seed.
  Progress lines go to the `pixelforge.gerstner` logger (was `pixelforge.downscale.gerstner`).
  Downscale stage on the portrait, background preset (`stats.timings.downscale`, one run
  each, load average 21–29): 5.21 s → 4.49 s; one annealer step on 16,384 points × 64
  clusters: 12.9 ms (numpy) → 7.9 ms. The former Python control flow is kept as
  `tests/gerstner_reference.py`.
- Preprocessing is native (`crates/pixelforge/src/preprocess.rs`, migration plan Phase 4):
  `pixelforge.preprocess.run` calls `_native.preprocess_run`, which assembles the masks,
  layout, enhance pass, filters and LAB conversion in Rust, and `subject_bbox`,
  `subject_brightness` and the module's other functions are thin calls into `_native`;
  `pixelforge/preprocess.py` no longer imports scipy or scikit-image. Reproduced bitwise:
  the alpha mask, flat-background keying with fringe unmixing, the exact distance transform
  (scipy's feature indices, ties included), crop, layout and `fit`, integer repeat, box
  pre-reduction, seamless padding, uniform and median filters, and percentiles. The
  Gaussian blurs of `enhance` and the bilateral denoise differ in the last bits: the macOS
  arm64 wheels of scipy and scikit-image fuse multiply-adds in the Gaussian correlation and
  the bilateral accumulation, and this port does not (plan Decision 6), so it also matches
  FMA-free Linux builds. Snapshot compare against 5a35744: all 48 cases
  identical (no pixel, palette or hash moved), so `tests/golden.json` is unchanged.
  Preprocess stage on the 1024² portrait, background preset (`stats.timings.preprocess`,
  one run each): 2.37 s → 0.26 s; with `enhance=1.0, ink=0.6`: 2.54 s → 0.56 s (10-core
  M1 Max, load average 8 before and 19–23 after). `remove_bg` still runs rembg from Python
  before the native stage. The former implementation is kept as
  `tests/preprocess_reference.py` for the parity tests, and `scripts/record_oracles.py`
  records the bilateral oracle on scipy and scikit-image directly.
- Post-processing and tiles are native (`crates/pixelforge/src/postprocess.rs` and
  `tiles.rs`, migration plan Phase 3): saturation, orphan removal (own 8-connected
  labelling instead of `scipy.ndimage.label`), jaggies, the outline, tile extraction with its
  near-duplicate search, reconstruction, the tileset sheet, Tiled `.tmj` and CSV. Palette
  ramps still run in Python before the native call. Bitwise identical to the numpy version
  (`tests/test_postprocess_native.py`, `tests/test_tiles_native.py` keep it as the
  reference); the snapshot matrix and goldens are unchanged.
- Quantization and palette building are native (`crates/pixelforge/src/quantize.rs` and
  `palette.rs`, migration plan Phase 3): `quantize.bayer`, `local_std`, `dither_offsets` and
  `run`, and `palette.median_cut`, `regularize_ramps`, `build` (except the mcda source, still
  Python) and the text of `write_hex` call `pixelforge._native`; `quantize.py` no longer uses
  `scipy.ndimage` (scipy's `uniform_filter(size=3, mode="nearest")` is reproduced with its
  running-sum update). Median cut, the Bayer matrices, `local_std`, the dither offsets and
  the quantized indices are bitwise identical to the numpy version
  (`tests/test_quantize_native.py`, `tests/test_palette_native.py`). `regularize_ramps` uses
  `libm`'s `hypot`/`atan2`/`cos`/`sin`, which differ from numpy's by at most one ulp on a few
  percent of arguments: moved colours' a and b can differ by up to 7e-14 (bitwise equal with
  the libm values substituted into the numpy formula). Snapshot matrix: 48 of 48 cases
  identical; goldens unchanged. `write_hex` now raises ValueError for a palette with values
  outside 0..255 instead of writing a malformed line.
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
