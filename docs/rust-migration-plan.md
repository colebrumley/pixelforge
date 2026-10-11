# Moving the rest of pixelforge to Rust

Status: planning. Branch `claude/app-rust-migration-plan-b00727`, based on `claude/rust-port`
(918abef), which already holds the per-pixel hot loops in `rust/` (`pixelforge._native`).
This document covers everything that is still Python: config, I/O, preprocessing, the
downscalers' control flow, palette, quantization, post-processing, tiles, the pipeline, the
CLI, the dev scripts and the test suite. [docs/rust-port-plan.md](rust-port-plan.md) is the
plan this continues; its decisions stay in force unless amended below.

## 1. Goal and end state

One Rust workspace that builds a single `pixelforge` binary with no Python at runtime:

- `cargo install --git https://github.com/colebrumley/pixelforge pixelforge` installs the
  CLI (`convert`, `batch`, `compare`, `palettes`), with the same flags, outputs, JSON, exit
  codes and messages as today.
- The library is a Rust crate (`pixelforge::run(input, &Config) -> Result<Output>`); the
  Python library API survives as an optional PyO3 binding crate (Decision 2).
- The determinism contract is unchanged: same input + same config = byte-identical PNGs, on
  every machine, for any thread count; `tests/golden.json` and the cross-platform CI matrix
  stay the proof.
- Behaviour is the current Python behaviour, deviations and all. Every `# DEVIATION:` note
  moves with its code. Output pixels may move where a numerical routine is replaced
  (Section 4); every such move is measured and recorded once, never silently.

What is being removed: numpy, scipy, scikit-image, Pillow and click as runtime dependencies;
the Python package `pixelforge/` and `pyproject.toml` as the product (kept only inside the
optional binding crate); `uv` as the way to run tests and scripts.

## 2. Decisions

Each decision names the recommended option first; alternatives are listed so the choice can
be reversed before the phase that depends on it starts.

1. **Strangler migration through the existing binding, bottom-up.** Each stage is ported
   into the Rust library and immediately called from the Python pipeline through
   `pixelforge._native`, so the full Python test suite (576 tests) and the snapshot/golden
   tools gate every phase. The Python pipeline shrinks phase by phase; the CLI is ported
   last; the tests are ported after the CLI and Python is then removed. Alternative: write
   the Rust app side by side and compare at the end. Rejected: differences would be found
   months of work later with no way to tell which stage caused them.
2. **Python binding kept as an optional crate.** `crates/pixelforge-py` (PyO3 + maturin, the
   present `_native` grown into a binding of the whole library: `Config`, `run`, `Result`
   with numpy arrays) stays buildable with `uv sync` for the library users the README
   documents. It is not needed to install or run the CLI. Whether to drop it is decided at
   the end of Phase 8 with the port complete; the default is to keep it.
3. **Numerics are ported formula for formula; bits may move once per routine.** The
   previous plan's Decision 4 applies to every replaced numerical routine (colour
   conversion, bilateral denoise, Gaussian/uniform filters, percentile, eigenvectors, the
   palette annealer's linear algebra): the Rust code computes the same quantity with the same
   formula, the result is compared against the Python stage on the snapshot matrix, the
   differing-pixel count and ΔE are recorded in CHANGELOG.md, and the goldens are
   regenerated once per phase. Stages whose logic is exact (integer, comparisons on values
   both sides computed identically) must be bitwise identical to Python; Section 4 lists
   which is which.
4. **skimage's bilateral denoise is ported, not replaced.** `denoise_bilateral` is the
   default `denoise` in both presets, so its behaviour is product behaviour. Port the
   algorithm of scikit-image 0.26's `_denoise_bilateral` (Gaussian spatial weights, the
   colour-distance lookup table with its bin count, `mode="edge"` padding) so outputs stay
   within rounding of today's. Alternative: switch preprocessing to the exact bilateral
   filter already in `rust/src/gerstner.rs`. That is the "later, separately decided" item of
   the previous plan and remains a product decision for after the port.
5. **`--remove-bg` runs an external `rembg` command.** AI matting is opt-in, outside the
   determinism contract and needs an ONNX runtime plus a 170 MB model. The Rust CLI looks
   for a `rembg` executable (`rembg i INPUT OUTPUT`), reports its version in `_meta.json`
   (`remove_bg_versions`) and gives today's "install rembg" config error when it is missing.
   Alternatives: port matting with the `ort` crate (a project of its own), or drop the flag.
6. **Cross-platform bit identity is extended to every float routine.** All transcendental
   functions come from the pure-Rust `libm` crate (`exp`, `expm1`, `log`, `pow`, `cbrt`,
   `sin`, `cos`, `atan2`, `hypot`); `f64::sqrt` is exact and allowed; rounding to integers
   uses `f64::round_ties_even` wherever Python used `np.round` or `round()` (half to even);
   no FMA contraction, no `mul_add`, no `target-cpu=native`, no `powi`; an explicit
   `libm::fma` is allowed where a reference implementation is known to fuse (it is a pure,
   correctly rounded software operation, identical everywhere); sorts are stable with
   explicit keys; no `HashMap` iteration in anything that produces a number or an order.
   These rules go into DESIGN.md's determinism contract and the crate README.
7. **Canonical JSON must match Python byte for byte.** `Config.canonical_json()` is embedded
   in every PNG and hashed into `config_hash`. The Rust writer reproduces
   `json.dumps(sort_keys=True, separators=(",", ":"))`: sorted keys, no whitespace,
   `ensure_ascii` escaping, and Python's float `repr` (shortest round-trip digits, exponent
   form below 1e-4 and from 1e16, `1.0` not `1`). A corpus of configs and their Python
   canonical JSON and hashes is recorded in Phase 1 and tested forever.
8. **One field table drives Config, JSON, the CLI and `--sweep`.** A static table (name,
   type, default, per-preset overrides, help text, choices, bounds) replaces the dataclass
   introspection click relied on. A test asserts every field has a flag, a help line and a
   JSON key, and that the `--help` output lists per-preset defaults as today.
9. **Dev tooling moves to an `xtask` crate.** `make_fixtures`, `update_golden`, `bench`,
   `snapshot_outputs` and `gallery` become `cargo xtask fixtures|golden|bench|snapshot|gallery`
   so CI needs no Python once Phase 8 lands. Alternative: hidden `pixelforge dev …`
   subcommands; rejected to keep the product binary small and its `--help` clean.
10. **Distribution stays "from the repo".** `cargo install --git …` replaces
    `uv tool install git+…`; the README's one-liner for rustup stays and the Python
    requirement goes. Prebuilt binaries on GitHub Releases are a later, separate addition.
11. **Arrays.** Plain slices with explicit `(height, width)` dimensions everywhere, as the
    hot loops already do (amended in Phase 3: `ndarray` was planned but the ported stages did
    not need it, and the library stays free of array crates). No other numerical crate (no
    nalgebra, no rayon): the 2×2 and 3×3 eigenproblems are closed-form and Jacobi, and
    parallelism stays `parallel::run_bands`.
12. **Version 0.2.0 at the end**, single-sourced from the workspace `Cargo.toml`
    (`pixelforge --version`, `_meta.json`, `Config.hash()`); the bump changes every
    config hash, which is why the final golden regeneration happens anyway.

## 3. Inventory: what moves

Production Python today (`pixelforge/`, without `_reference/`): about 3,900 lines. Rust
today: 1,517 lines (`kopf.rs`, `gerstner.rs`, `parallel.rs`, `lib.rs`). Tests: 576
(5,800 lines). Scripts: 800 lines.

| Module | Python lines | Replaces | Parity class (Section 4) | Phase |
|---|---|---|---|---|
| `config.py` | 424 | dataclass, `json`, `hashlib` | exact (JSON bytes, hash, every error message) | 2 |
| `color.py` | 65 | `skimage.color.rgb2lab/lab2rgb`, `cKDTree` | float (LAB), exact (nearest, hex) | 2 |
| `palette.py` | 265 | numpy sorts/cumsum, file parsing, `atan2/cos/sin` | exact (parsing, write_hex, median_cut split choice) / float (ramps) | 3, 5 (mcda) |
| `quantize.py` | 77 | `ndimage.uniform_filter`, Bayer | exact (bayer, indices) / float (local_std) | 3 |
| `postprocess.py` | 298 | `ndimage.label`, numpy set ops | exact | 3 |
| `tiles.py` | 206 | numpy, `dict` digests | exact | 3 |
| `preprocess.py` | 597 | `ndimage.label/binary_dilation/distance_transform_edt/median_filter/gaussian_filter1d/uniform_filter1d`, `np.percentile/median`, `skimage.restoration.denoise_bilateral` | exact (mask, crop, repeat, prereduce, layout) / float (enhance, denoise, LAB) | 4 |
| `downscale/box.py` | 49 | `np.bincount` | float (sums; reproducible bitwise with row-major order) | 5 |
| `downscale/kopf.py` | 347 | `np.linalg.eigh` (2×2), array algebra | float | 5 |
| `downscale/gerstner.py` | 429 | `np.linalg.eigh` (3×3), `np.random.default_rng`, matrix products | float | 5 |
| `downscale/__init__.py` | 43 | — | exact | 5 |
| `io.py` | 215 | Pillow decode/encode, `os.open`, `hashlib` | exact (PNG pixels, tEXt, PLTE/tRNS, sha256) | 6 |
| `pipeline.py` | 274 | `json`, `importlib.metadata`, `time` | exact (files, stats, JSON layout) | 6 |
| `cli.py` | 612 | click, Pillow `ImageDraw/ImageFont` | exact (flags, JSON lines, exit codes, messages) / labels: deterministic, not identical | 7 |
| `errors.py`, `version.py`, `__init__.py`, `__main__.py` | 32 | — | exact | 6, 7 |
| `scripts/*.py` | 803 | — | — | 8 |
| `tests/*.py` | 5,800 | pytest, `CliRunner`, subprocess | — | 8 |

Runtime dependencies after the port: `clap`, `serde`/`serde_json` (reading JSON; writing is
the custom canonical writer), `sha2`, `libm`, `ndarray`, `image` (PNG, JPEG, GIF, WEBP, BMP,
TIFF decoders and the PNG encoder), `log`, `thiserror`. Dev: `assert_cmd`, `predicates`.
Binding: `pyo3`, `numpy`.

## 4. Requirements

### 4.1 Functional parity

- Every Config field, preset, default, bound, choice list and validation message in
  `config.py` (including `_coerce`'s rules: bools are not ints, ints must be exact and fit
  64 bits, floats finite, `outline` lowercased, unknown fields rejected, `replace()`
  re-applying a changed preset to non-explicit fields).
- Every output file and its layout: `NAME.png`, `_preview.png`, `_palette.json` (indent 2,
  LAB rounded to 6 decimals, `"used"`, the leading transparent entry for
  `transparent_index=first`), `_palette.hex` (`; transparent` comment line), `_meta.json`
  (sorted keys, indent 2, `environment` with the Rust toolchain and crate versions replacing
  the Python library versions, `threads`), `_tileset.png`, `_tilemap.json` (compact),
  `.tmj` (indent 1, Tiled 1.10 schema, GID flip bits), `_tilemap.csv`.
- PNG text chunks `pixelforge:config`, `pixelforge:input_sha256`, `pixelforge:palette_sha256`,
  `pixelforge:palette`, written sorted by key; palette-indexed PNGs with the transparent
  entry placed per `transparent_index`; RGBA fallback above 256 entries.
- Input handling: only PNG, JPEG, GIF, WEBP, BMP, TIFF; regular files only (open, then
  `fstat`; `O_NONBLOCK` on Unix so a FIFO cannot hang the open); the pixel budget checked
  from the header before any pixel is decoded; EXIF orientation applied; 16-bit greyscale
  rescaled by 1/257 and rounded, not clipped; first frame of animations with a warning;
  in-memory inputs (`Input::Rgba8`, `Input::RgbaF64`) hashed as `rgba8 HxW\n` + bytes.
- All CLI behaviour in `cli.py`: the flag per field with `--no-` negation for bools, the
  hidden alias `--g-T-final`, `--config` (regular file, 1 MiB cap, UTF-8, no NaN/Infinity
  constants, the "preset does not override" warning), `--max-input-pixels`, `--force` and
  the input-collision check, `-v/-vv/-q`, `--debug`/`PIXELFORGE_DEBUG`, exit 2 for config
  errors and 1 otherwise with `config error: …`/`error: …` on stderr, JSON lines on
  stdout, warnings prefixed `pixelforge: `. `batch`: image suffix filter, skipping
  `_preview/_tileset/_compare` stems, stem-collision refusal, shared scale and canvas, shared
  levels, shared palette written as `shared_palette.hex` and passed inline. `compare`:
  side-by-side sheet, `--prepass`, `--sweep` with its parsing rules and the per-cell
  `flags` and `config` report, the preview-scale cap. `palettes`.
- Logging: a stderr logger with the `pixelforge: ` prefix, WARNING by default, INFO per
  stage with `-v`, DEBUG per iteration with `-vv`, errors only with `-q`; logging never
  reads a value that is not already computed (it can never change a result).
- The library: `pixelforge::run(Input, &Config) -> Result<Output, Error>`,
  `run_loaded`, `Output::save(prefix)`, `output_paths`, `Config::hash/canonical_json/
  provenance/replace/validate_palette`, `PRESETS`, bundled palettes embedded with
  `include_str!`. Errors: `Error::Config` (exit 2) and `Error::Pixelforge` (exit 1), the
  same messages.

### 4.2 Numerics and determinism

- Decision 6 in full. `round_ties_even` is a specific trap: numpy rounds 2.5 to 2, Rust's
  `round` gives 3; `lab_to_rgb8`, `_to_8bit`, `from_image`, `resolve_layout` and the
  pre-reduction all round.
- Exact (bitwise against Python) stages, verified on the snapshot matrix and on synthetic
  cases through the binding before the Python code is deleted: PNG decoding, mask and
  keying topology (`ndimage.label` regions are sets; numbering may differ, outcomes may
  not), crop, integer repeat, box pre-reduction means (same summation order), layout and
  `fit`, Bayer matrices, nearest-colour indices (brute force, ties → lowest index; cKDTree's
  tie-break may differ, which the snapshot compare will show as isolated pixels and is
  accepted), orphan removal, jaggies, outline, tiles, tilemap, PNG pixels and text chunks,
  every JSON file except `_meta.json`'s `environment` and `timings`.
- Exact distance transform. `scipy.ndimage.distance_transform_edt(return_indices=True)`
  decides which opaque pixel a transparent one copies its colour from (`_fill_transparent`,
  `_unmix_fringe`), and ties are common on a pixel grid. Implement the same algorithm
  (Maurer, Qi, Raghavan 2003, the dimension order and sweep scipy's
  `NI_EuclideanFeatureTransform` uses) and verify the returned indices bitwise against
  scipy on the fixtures and on random masks before Phase 4 closes.
- Float stages, replaced routine for routine with the formulas noted: sRGB↔LAB (skimage's
  D65 constants, `pow(…, 2.4)`, `cbrt`), `np.percentile` (linear interpolation),
  `np.median` (mean of the middle pair), `gaussian_filter1d` (truncate 4σ, normalised
  kernel, `nearest`/`wrap`), `uniform_filter1d` (scipy's origin convention), 3×3 `median_filter`
  with `nearest`, skimage's bilateral (Decision 4), `expm1`, the 2×2 symmetric
  eigendecomposition for kopf's covariance clamp (closed form), the 3×3 for
  `principal_axis` (Jacobi; sign normalised as today), the annealer's association in log
  space, and `np.bincount` sums in row-major pixel order (already the rule for
  `superpixel_stats`).
- Threading: `PIXELFORGE_THREADS` unchanged. Any newly threaded routine must be per-pixel
  independent (the preprocess bilateral qualifies) or use fixed bands merged in band order.
  A test asserts byte-identical outputs for thread counts 1, 2, 3 and 8 on every fixture.
- The seeded RNG: `numpy.random.default_rng(seed).uniform(0.9, 1.1)` is the only RNG use.
  Reproducing numpy's PCG64 stream bit for bit is possible (PCG64 with numpy's SeedSequence
  and its `uniform` construction) but not required: `seed` already changes output by design.
  Requirement: a seeded run is deterministic and differs from the unseeded one; document
  that seeded outputs differ from the Python version's.

### 4.3 Performance

- No stage slower than today's Python on `scripts/bench.py`'s matrix (fixtures plus
  `samples/knight_sprite.png` and the portrait). Baselines: `bench/0edd462.json`
  (pure Python) and the `bench/<sha>.json` recorded on `claude/rust-port` in Phase 0.
- Targets for the ported stages on an M1 Max, 1024² input: preprocess with bilateral
  denoise under 0.5 s (skimage: 1.45 s), everything outside `downscale` under 0.2 s in
  total. Memory: peak RSS at or below the Python version on the 1024² cases.

### 4.4 Packaging, CI, docs

- Workspace layout: `Cargo.toml` (workspace) at the repo root; `crates/pixelforge` (library
  plus `src/main.rs`), `crates/pixelforge-py` (cdylib `pixelforge._native`, later the whole
  binding), `xtask/`. `rust/` is moved, not copied; `pyproject.toml` keeps maturin pointing
  at the binding crate until Phase 8, then moves into `crates/pixelforge-py/`.
- CI: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` on
  ubuntu and macOS arm64 (the cross-platform golden check), the binding built and its
  smoke tests run on one matrix entry while the binding exists. No Python in CI after
  Phase 8 except for the binding job.
- Docs: README (install, `cargo install`, no Python requirement, library section shows
  Rust first and the binding second), DESIGN.md (the native layer becomes the whole
  program; the determinism contract gains Decision 6; the timings table), CHANGELOG.md
  (one entry per phase with differing-pixel counts), REQUIREMENTS.md gets a note that the
  non-goal "No Rust" was superseded. `rust/README.md` becomes `crates/pixelforge/README.md`.

### 4.5 Tests

- During Phases 2–7 the Python suite is the gate: all 576 tests pass at the end of every
  phase, with goldens regenerated where the phase's numerics changed and the diff reviewed
  with `scripts/snapshot_outputs.py compare`.
- Oracle fixtures recorded in Phase 1, while Python is available, and kept in
  `tests/oracle/`: canonical JSON and hash for a corpus of configs (every preset, every
  field at a non-default value, inline and file palettes, non-ASCII palette paths, floats
  that exercise the repr rules); 1,000 lattice colours with skimage's LAB to 1e-12;
  `distance_transform_edt` indices for the fixture masks and random masks; skimage
  bilateral output for the fixtures at the preset sigmas; `np.percentile` and
  `gaussian_filter1d`/`uniform_filter1d` outputs for a few arrays; the Python `repr` of
  10,000 floats. Small (under 2 MB in total), checked in, loaded by Rust unit tests.
- After Phase 8 the Rust suite must cover what the Python suite covers: module unit tests
  per crate, CLI integration tests with `assert_cmd` (every `test_cli.py` and
  `test_pipeline.py` case), `tests/golden.json` consumed by a Rust test, determinism across
  fresh processes and thread counts, the fixtures check, the slow marker as an ignored test
  group (`cargo test -- --ignored` in CI). Test count parity is checked by a one-time
  mapping table in the Phase 8 PR, not by number.

## 5. Phases

Each phase is one task for an Opus 5.5 subagent at medium effort; the Fable session
reviews, runs the acceptance checks and regenerates goldens. A phase ends with green CI, a
CHANGELOG entry and, where outputs moved, the snapshot report pasted into the PR.

### Phase 0 — Baselines (no code)

- `scripts/bench.py` and `scripts/snapshot_outputs.py save` on `claude/rust-port` for the
  fixture matrix plus `samples/*.png` (six inputs) and the portrait; the snapshot directory
  is the reference for every later `compare`.
- Record the current test runtime (`pytest -q`, `-m "not slow"`) for the parity target.

Done 2026-10-10 on 9f53516: `bench/snapshots/` holds 48 cases (4 fixtures × 2 presets × 3
methods, the knight and portrait inputs of the old baseline, and the six `samples/*.png` with
their natural preset); `bench/9f53516.json` is the bench (`--compare bench/0edd462.json`:
portrait kopf 33.9 → 4.5 s, gerstner 31.0 → 6.8 s, knight kopf 8.1 → 0.7 s) taken with a load
average of 12–24 on the 10-core M1 Max, so it is an upper bound, not the idle-machine
number DESIGN.md still owes. Test suite: 576 passed in 54 s; `-m "not slow"` 566 in 54 s
(the same machine, same load).

### Phase 1 — Workspace and oracles

- Workspace layout of Section 4.4. The PyO3 wrappers in `kopf.rs`/`gerstner.rs` split from
  their pure functions (`em_step`, `c_step`, `spatial_refill`, the slic/bilateral/stats
  bodies) so the library crate has no `pyo3` dependency; the binding crate re-exports them
  as today's `_native` functions. `uv sync --locked` still builds `pixelforge._native`.
- `tests/oracle/` recorded (Section 4.5) with the Python script that produced it kept in
  `xtask`'s history for reference.
- The field table (Decision 8) written in Rust from `config.py`'s dataclass, FIELD_HELP,
  FIELD_CHOICES, UPPER_BOUNDS and PRESETS, with a Python test comparing it field by field.

Acceptance: 576 tests pass; snapshot compare reports every case identical; CI green.

Done 2026-10-10 (merged as 4157bba and the workspace merge after it): `crates/pixelforge`
(library, `libm` only), `crates/pixelforge-py` (binding, unchanged `_native` API), release
profile at the workspace root, CI and docs repointed; `tests/oracle/` (1.7 MB, 13 data files,
recorded by `scripts/record_oracles.py`, reproduced byte for byte by `tests/test_oracle.py`);
the field table in `crates/pixelforge/src/config.rs` with `tests/test_field_table.py`. 638
tests pass; the 48 Phase 0 snapshot cases are identical; `cargo clippy -D warnings` clean.
Two `ConfigError` branches are unreachable and have no oracle case (`_coerce`'s
"unsupported field type" and the `seed` upper bound, which the 64-bit check fires first).

### Phase 2 — Config and colour

- `config`: `Config`, presets, `_explicit`/`replace`, validation with today's messages,
  canonical JSON writer (Decision 7), `hash`, `provenance`, `palette_hex_lines` with the
  palette parsers (`.hex`, `.gpl`, `hex:`; 64 KiB cap, UTF-8, 1–256 colours).
- `color`: rgb↔lab, `rgb8`/`hex` helpers, `delta_e`, `nearest_index` (brute force, lowest
  index on ties), `palette_distance_matrix`.
- Binding: `_native.config_canonical_json(dict)`, `_native.config_errors(dict)`,
  `_native.rgb_to_lab`, `_native.lab_to_rgb8`, `_native.nearest_index`. The Python
  `color.py` calls the native versions; Python `Config` stays but a test asserts the
  native canonical JSON, hash and error messages agree on the oracle corpus and on every
  `test_config.py` case.

Acceptance: oracle tests pass; snapshot compare lists the pixels moved by the LAB
conversion (expected: isolated 8-bit rounding flips, mean ΔE under 0.5); goldens
regenerated (#1).

Done 2026-10-10 (merges of the colour branch and a5e361b Config). Colour: `crates/pixelforge/src/color.rs`
reproduces skimage 0.26 to 5.7e-14 in LAB (scipy's inverse matrix copied bit for bit),
`lab_to_rgb8` exact on the oracle, `nearest_index` brute force with lowest-index ties;
`pixelforge.color` now calls the binding. Snapshot compare against Phase 0: 47 of 48 cases
identical; `noisy_gradient/background/gerstner` moved 10 of 65,536 pixels (ΔE 4.03), traced to
a near-tie in the annealer that a one-ulp jitter of skimage's own LAB reproduces. Goldens did
not change, so no regeneration was needed (#1 is still available for a later phase). A guard
test (`crates/pixelforge/tests/determinism_guard.rs`) fails the build on any platform
transcendental or `.round()` in the Rust sources. Config: `config/model.rs`, `palette.rs`,
`pyrepr.rs` (Python float and str `repr`), `pyunicode.rs` (generated tables, Unicode 14);
all 103 corpus cases, 104 error messages and 10,000 float reprs match byte for byte;
`_native.config_*` compared with the live Python `Config` in `tests/test_config_native.py`.
Two Python-only behaviours are documented as not reproduced: a list given as `preset`
raises `TypeError` in Python, and numpy scalar values are accepted only by Python. 917
tests pass; `cargo tree -e normal` shows `libm` and `sha2`.

### Phase 3 — Quantize, palette, post-processing, tiles

- `quantize` (bayer, local_std with and without mask, dither_offsets, run), `palette`
  (median_cut with the error-driven split and lexsort order, regularize_ramps, write_hex,
  inline_name, resolve_source, build minus mcda), `postprocess` (8-connected labelling,
  `remove_orphans` with the frozen set and early exit, `fix_jaggies`, `add_outline`,
  `apply_saturation`, `run`), `tiles` (extract with the exact/near-duplicate search and
  its pruning, reconstruct, tileset_indices, tiled_map, tilemap_csv, gids).
- Binding exposes each stage; the Python modules become one-line calls.

Acceptance: postprocess, tiles and quantize indices bitwise identical to Python on the
snapshot matrix and on `test_postprocess.py`/`test_tiles.py` inputs; ramps and local_std
within 1e-9; goldens regenerated only if a pixel moved (not expected).

Done 2026-10-10 (merges be2b3ef and 2dacb1e). Preceded by `crates/pixelforge/src/npsum.rs`,
numpy's pairwise float64 summation reproduced bit for bit (sequential below 8 elements, eight
accumulators to 128, recursive split above; leading-axis reductions are sequential), verified
against `np.sum` on 620 arrays: median cut's box errors, ramp bins' mean L and the tile
de-duplication means all depend on it. `quantize.rs` (Bayer, scipy's `uniform_filter(size=3,
mode="nearest")` with its running-sum update, `local_std`, dither, `run`), `palette.rs`
(median cut, ramps, `build` minus mcda, hex text), `postprocess.rs` (own 8-connected
labelling; the merge decisions were shown to be independent of label numbering) and
`tiles.rs` (extraction with the pruning screens, reconstruction, sheet, Tiled map, CSV).
Everything bitwise identical to the numpy versions except `regularize_ramps`, whose moved
a/b values differ by up to 7e-14 because libm's `hypot`/`atan2`/`cos`/`sin` differ from
numpy's by one ulp on a few percent of arguments (bitwise equal once libm's values are
substituted into the numpy formula). Snapshot matrix 48 of 48 identical to the Phase 3
start; goldens unchanged. 1132 tests pass. Decision 11 amended: the stages use plain slices
with explicit dimensions, no `ndarray` dependency (the library still depends on `libm` and
`sha2` only). `preprocess.py` is now the only production module using scipy or skimage.

### Phase 4 — Preprocessing

- `preprocess`: `_subject` (keying with the border median, 4-connected labelling, fringe
  unmixing with the exact distance transform and dilations), crop, `resolve_layout`,
  `_fit_tiles`, `fit_crop`, repeat, `prereduce`, `enhance` (levels, `_stretch`,
  `_gaussian` with the running-mean approximation above σ 4, `_blur`, ink), `denoise`
  (median; skimage bilateral port per Decision 4), seamless padding, `Preprocessed` with
  `canvas_pad`, `subject_bbox`, `subject_brightness`, `enhance_levels`; `remove_background`
  via the external `rembg` command (Decision 5).
- Binding exposes `preprocess.run` and the three helpers `batch` uses.

Acceptance: masks, crops, layouts and pre-reduced images bitwise identical; EDT indices
bitwise identical on the oracle masks; bilateral within 1e-6 of skimage on the oracle
cases; snapshot report; goldens regenerated (#2). Bench: preprocess time on the portrait.

Done 2026-10-11 (merges 3235039 filters, 5a35744 masking and layout, cdae0c7 integration).
`mask.rs` reproduces scipy's `distance_transform_edt` indices bit for bit (dimension-by-
dimension Maurer passes; a site leaves the envelope only on scipy's strict inequality, so
the lower index along the pass wins ties), plus keying, fringe unmixing, 4-connected
labelling and dilation; `layout.rs` the sizes, `fit`, pre-reduction (whose numpy block
means turned out to be sequential for RGB and row-wise pairwise for weights, probed
empirically), repeat and wrap padding; `filters.rs` scipy's `uniform_filter1d` (bitwise,
any size and mode), `gaussian_filter1d` (2.2e-16), `median_filter` (bitwise) and skimage
0.26's bilateral as written, including its oversized spatial table and integer `dist_scale`
(3.3e-16 on the oracle; 4 pixels at 5.4e-6 on the full portrait); `enhance.rs` the whole
pass with `np.percentile` bitwise (4.2e-15 overall). The macOS scipy and skimage wheels fuse
multiply-adds in the Gaussian correlation and the bilateral accumulation and colour
distance; this port does not (Decision 6), so it matches FMA-free builds instead of this
Mac's wheels. `preprocess.rs::run` composes the stages; `pixelforge.preprocess` delegates,
with rembg still applied from Python (the external command waits for PNG I/O in Phase 6).
Snapshot matrix 48 of 48 identical to the Phase 4 start, goldens unchanged (so golden
regeneration #2 was not needed either). Preprocess on the portrait: 2.37 → 0.26 s
(background preset), 2.54 → 0.56 s with enhance 1.0 and ink 0.6, under load. 1325 tests
pass; no production module imports scipy or scikit-image any more (both stay installed for
the test references and the oracle recorder until Phase 8).

### Phase 5 — Downscalers' control flow

- `box::run`; `kopf::run` (initialisation, window and truncation bookkeeping, ghost cells,
  the M-step and C-step algebra, `_shift`, `neighbor_mean4`, the shape-constraint loop,
  starved re-seeding, convergence, the final refill) over the existing native loops;
  `gerstner::run` (`_Grid`, `PaletteAnnealer` with `principal_axis`, the split/stall
  logic, the seeded perturbation, `palette_prior`, `_laplacian_smooth`, the iteration loop,
  the transparent-pixel attribution) over `slic_assign`/`bilateral_filter`/
  `superpixel_stats`; `palette::mcda` on the annealer; `Downscaled`.
- Binding exposes `downscale(method, lab, mask, w, h, config, weight)`.

Acceptance: `test_kopf.py`, `test_gerstner.py`, `test_box.py`, `test_palette.py` pass
unchanged (behavioural tests); snapshot report (expected: a few pixels on the gerstner
cases from eigenvector and summation-order differences in the annealer); goldens
regenerated (#3). Bench: kopf and gerstner wall time at 1 and 8 threads.

Done 2026-10-11 (merges 99c5a08 box and kopf, cb15d34 gerstner). `boxfilter.rs` is bitwise.
`kopf_run.rs` is bitwise identical to the Python control flow on every fixture at every
size: LAPACK's 2×2 path (`dsteqr` → `dlaev2`) was reproduced including the two fused
multiply-adds Accelerate performs in it (`libm::fma`, correctly rounded and identical on
every platform), `1.1 ** n` as a table of the correctly rounded powers (libm's `pow` is off
by one ulp at n = 4), and `np.hypot` as `sqrt(fma(small, small, big²))`. `annealer.rs` and
`gerstner_run.rs` reproduce the annealer and the loop; BLAS's `centered.T @ centered` and
`weighted.T @ points` cannot be reproduced (Accelerate matches neither sequential nor
pairwise order), so the principal axis moves by ≤ 2.8e-13 and the palettes by ≤ 3.3e-11,
which never changed an index, a mask cell or an 8-bit value; substituting the Rust
values for those three routines into the Python reference makes all 42 fixture cases
bitwise identical, so the ported logic itself is exact. The seeded perturbation uses a
SplitMix64 draw instead of numpy's PCG64 (seeded outputs differ from the Python version's,
still deterministic). Snapshot matrix 48 of 48 identical; goldens unchanged (regeneration
#3 was not needed). Downscale on the portrait under load: kopf 2.61 → 1.68 s, gerstner
5.21 → 4.49 s. 1482 tests pass. Progress lines now come from the `pixelforge.kopf` and
`pixelforge.gerstner` loggers through the `log` bridge (`crates/pixelforge-py/src/pylog.rs`).
Decision 6 clarified: `libm::fma` is allowed as an explicit, correctly rounded operation
(what is forbidden is compiler contraction, `mul_add`, and platform libm); Phase 4's filters
stay unfused because there the fused variant would match only this Mac's wheels, while in
kopf it is what makes the 2×2 eigensolver match LAPACK bit for bit.

### Phase 6 — I/O, pipeline, library API

- `io`: decode (format allow-list, regular-file check, header budget, EXIF, 16-bit, first
  frame, sha256), `from_image` for arrays, `indices_to_rgba`, `upscale_nearest`,
  `save_png` (palette/tRNS/tEXt, RGBA fallback), `read_png_text`.
- `pipeline`: `run`, `run_loaded`, stats, timings, `environment()`, `Output::save`,
  `output_paths`, `uses_gerstner_palette`, the stage log lines. `Error` types.
- Binding: `pixelforge.run`, `run_loaded`, `Result` become thin wrappers over the Rust
  pipeline returning numpy views; `pixelforge/*.py` stage modules are deleted except
  `cli.py`, `__init__.py`, `version.py`.

Acceptance: all 576 tests pass against the Rust pipeline; snapshot identical to Phase 5;
PNG files decode to the same pixels and carry the same text chunks (bytes differ from
Pillow's and that is fine); `test_determinism.py` across processes passes.

Done 2026-10-11 (merges 1da8cfd I/O, 1d3a7b4 pipeline), preceded by `pyjson.rs` (Python's
`json.dumps` layouts and `round(x, n)`, verified against Python). `io.rs` decodes every
allowed format with the `image` crate (0.25.10, no default features) plus `gif`, `tiff` and
`miniz_oxide` directly where `image` hides what Pillow exposes; bit-identical to Pillow on
every fixture, sample and synthetic mode (Pillow takes the high byte of 16-bit RGB, and
rescales only integer greyscale by 1/257), lossless and lossy WEBP identical, JPEG within 7
levels on up to half the pixels (chroma upsampling; accepted); frame counts read from headers
only; Pillow's decompression-bomb limit kept; indexed PNGs written with the `png` crate (PLTE,
tRNS, sorted tEXt). `pipeline.rs` orchestrates the stages and writes every output file
byte-identically (except `_meta.json`'s environment and timings); `environment()` records
platform, threads and the crate version, with the Python wrapper adding its library versions
until Phase 8; `remove_bg` runs an external `rembg i` command on the Rust side and the rembg
package on the Python side (`alpha_matted` hands the result over). First end-to-end
Rust-only test: `crates/pixelforge/tests/pipeline_oracle.rs` against `tests/oracle/outputs_small`.
Snapshot matrix 48 of 48 identical; goldens unchanged. 1817 tests pass. Python left in the
package: `cli.py` (612 lines), the `Config` class (424, parity-tested against its Rust twin),
and thin wrappers (`pipeline.py` 154, `io.py` 139, the stage modules).

### Phase 7 — CLI

- `clap` builder from the field table: `convert`, `batch`, `compare`, `palettes`,
  `--version`, every flag and behaviour in Section 4.1. Proof-sheet labels drawn with an
  embedded 6×10 bitmap font (deterministic, no font files).
- A Rust integration suite (`crates/pixelforge/tests/cli.rs`) ports every `test_cli.py`,
  `test_pipeline.py` CLI case and `test_determinism.py` subprocess case to `assert_cmd`.
- The Python `cli.py` is deleted; the binding keeps no CLI.

Acceptance: both suites green; `pixelforge convert|batch|compare` on the snapshot matrix
produce byte-identical PNG pixels, JSON files and stdout JSON to the Phase 6 Python CLI
(stderr messages compared by a fixture of expected lines); `--help` for each command
diffed against the click output with only the usage header differing.

### Phase 8 — Tests, tooling, docs, retirement

- Remaining Python tests ported to Rust unit tests (config, color, palette, postprocess,
  tiles, preprocess, io, box/kopf/gerstner behavioural tests, version, fixtures). Golden
  test reads `tests/golden.json`; `xtask golden` regenerates it; `xtask fixtures`,
  `xtask bench` (same JSON schema as `bench/<sha>.json` so `--compare` works across the
  port), `xtask snapshot save|compare`, `xtask gallery`.
- CI rewritten (Section 4.4). `pyproject.toml`, `uv.lock`, `tests/*.py`, `scripts/*.py`,
  `pixelforge/downscale/_reference/` removed; the binding crate gets its own
  `pyproject.toml` and a dozen smoke tests. Decision 2 confirmed or the crate removed.
- README, DESIGN.md, CHANGELOG.md, REQUIREMENTS.md note, version 0.2.0, final golden
  regeneration with the version bump.

Acceptance: `cargo test` green on both CI platforms with identical goldens; a fresh
`cargo install --git <repo>@<branch>` on a machine with only rustup works and converts the
samples; the README commands run as written.

## 6. Risks

- **Pixel drift from float replacements.** Every phase that touches floats reports the
  differing pixels; the review rejects a phase whose drift is not explained by a listed
  routine (an unexplained difference means a logic error, not rounding).
- **Tie-breaking.** cKDTree ties (accepted, visible in the snapshot), EDT ties (not
  accepted: same algorithm, verified bitwise), `np.unique`/`lexsort`/`argmin` ties (all
  lowest-index; Rust must use stable sorts and first-minimum scans).
- **Half-to-even rounding.** One stray `f64::round` moves pixels everywhere. Clippy cannot
  catch it; a grep test in CI forbids `.round()` outside an allow-listed helper.
- **Decoder differences.** PNG decoding is lossless and must be identical. JPEG and lossy
  WEBP decode within ±1 of Pillow's libjpeg/libwebp output, so outputs from such inputs
  differ from the Python version's. Documented; within the Rust build they are
  deterministic across platforms (pure-Rust decoders).
- **Canonical JSON.** A formatting mismatch silently changes every hash. The oracle corpus
  catches it; the writer is one function with no serde involvement.
- **skimage bilateral port.** The Cython source must be read, not guessed; the oracle
  outputs at the preset sigmas and at σ_spatial 16 (the bound) are the check.
- **Scope creep.** No algorithm changes during the port, including the ones already queued
  (exact bilateral in preprocess, gerstner's convergence on smooth gradients). They are
  listed in [rust-port-plan.md](rust-port-plan.md) under "Later, separately decided" and
  stay there.
- **Python binding bit-rot.** If Decision 2 keeps it, its tests run in CI on one platform;
  if nobody uses it within a release, drop it.

## 7. Open questions

1. Decision 5: external `rembg` command, `ort` port, or drop `--remove-bg`?
2. Decision 2: keep the Python binding after Phase 8?
3. Windows: nothing today targets it (no CI, `O_NONBLOCK`). Add a Windows CI entry with
   the port, or state "unsupported" in the README?
