# pixelforge (Rust library)

The pixelforge library and the `pixelforge` program (`src/main.rs`, `src/cli/`, behind the
default `cli` feature): plain Rust, no Python (`libm`, `sha2`, `log`, for image I/O
`image`, `png`, `gif`, `tiff`, `miniz_oxide`, and `clap` for the CLI). It was ported from
the Python implementation stage by stage; the module
documentation cites the Python files each module came from (`pixelforge/*.py`,
`tests/test_*.py`), which were removed in 0.2.0 and are in the git history up to 1fff901.
The optional Python binding is `crates/pixelforge-py`.

Workspace layout (repo root):

- `Cargo.toml`: the workspace, the version (`[workspace.package]`, the single source;
  `crates/pixelforge-py/pyproject.toml` repeats it and `tests/version.rs` checks) and the
  release profile (profiles only take effect in the root manifest). One `Cargo.lock` and
  `target/` at the root.
- `crates/pixelforge`: this library and the CLI; `palettes/` holds the bundled palettes.
- `crates/pixelforge-py`: the optional Python binding (PyO3 + maturin, `cdylib`
  `pixelforge_native`, `#[pymodule] _native`, and the `pixelforge` Python package in
  `python/`); see its README.
- `xtask/`: developer tooling (`cargo xtask fixtures|golden|bench|snapshot|gallery|help-record`).

Modules:

- `src/kopf.rs`: `em_step` (fused E-step + M-step sums), `c_step` (directional and overlap
  sums) and `spatial_refill` (spatial-only recoloring of starved kernels). They reduce over bands of
  `BAND_ROWS` (32) rows merged in band order; the bands are part of the numerics.
  `src/kopf_run.rs` is the control flow around them.
- `src/gerstner.rs`: `slic_assign` (one modified-SLIC assignment pass, bitwise equal to the
  numpy `_assign`), `bilateral_filter` (the superpixel bilateral filter, `libm::exp`) and
  `superpixel_stats` (count, mass, colour and centre sums per superpixel, bitwise equal to
  the seven `np.bincount` calls it replaced). `slic_assign` and
  `bilateral_filter` are per-pixel independent and run over fixed 8-row bands on
  `PIXELFORGE_THREADS` threads (default: available parallelism); those bands are not part
  of the numerics. `superpixel_stats` is single-threaded on purpose: it sums in row-major
  pixel order like `np.bincount`, and per-band partial sums would change the rounding.
  `src/gerstner_run.rs` and `src/annealer.rs` are the rest of the method.
- `src/color.rs`: sRGB <-> CIELAB (`rgb_to_lab`, `lab_to_rgb`, `rgb8_to_lab`, `lab_to_rgb8`),
  scikit-image 0.26's formulas (D65/2° white, its `xyz_from_rgb` matrix and the bits of its
  inverse) with `libm::pow`/`libm::cbrt`; `delta_e`, `nearest_index` (brute-force CIE76,
  ties to the lowest index), `palette_distance_matrix`, `hex_to_rgb8`, `rgb8_to_hex`, all over
  slices of interleaved triples. `tests/color_oracle.rs` checks them against `tests/oracle/`
  (LAB within 1e-9, measured 6e-14; `lab_to_rgb8` exact).
- `src/parallel.rs`: `thread_count` and `run_bands` (see the determinism rules below).
- `src/config.rs`: the Config field table (name, type, default, help, choices, upper bound
  per field; the presets and their longest edge), the single source the Config, its JSON
  and the CLI flags are built from.
- `src/config/`: `Config`: `Config::new(&[(name, Value)])`
  (defaults ← preset ← explicit values, `_coerce`, `_validate` with Python's messages),
  `replace`, `to_dict`, `canonical_json`, `hash`, `provenance`, `palette_hex_lines`,
  `palette_sha256`, `validate_palette`. `Value` is the uncoerced runtime value (from CLI
  flags, a `--config` JSON file or a Python dict in the binding). `tests/config_oracle.rs`
  checks every case of `tests/oracle/config_corpus.json` and all 104 messages of
  `config_errors.json`.
- `src/palette.rs`: the bundled palettes (embedded from `palettes/*.hex` with
  `include_str!`), `parse_hex`, `parse_gpl`, `load_palette` (bundled name, `.hex`/`.gpl`
  path relative to the working directory, `hex:` inline; 1–256 colours; regular files of at
  most 64 KiB of UTF-8, opened `O_NONBLOCK` and checked with `fstat`), `inline_name`,
  `parse_hex_lines`. File paths and OS errors are reported as Python 3.11's `pathlib` and
  `OSError` print them.
- `src/io.rs`: image I/O: `open_regular` (the regular-file
  open used for every input and palette file), `sha256_file`, `load` / `decode_bytes`
  (PNG, JPEG, GIF, WEBP, BMP, TIFF to RGBA8 as Pillow 12 decodes them, EXIF orientation,
  the pixel budget checked from the header, first frame plus a warning text),
  `from_image_array`, `from_rgba`, `indices_to_rgba`, `upscale_nearest`, `save_png` /
  `encode_png` (indexed PNG with PLTE, tRNS and sorted tEXt, RGBA above 256 entries) and
  `read_png_text`. Everything but JPEG decoding was verified bitwise equal to the Pillow
  version; see the module documentation for the per-format rules.
- `src/pipeline.rs`: `run` / `run_loaded` (load, preprocess, downscale, palette and
  quantize, post-process, tiles), `Output::save` and the files it writes. The other stages:
  `preprocess.rs`, `layout.rs`, `mask.rs`, `filters.rs`, `enhance.rs`, `downscale.rs`,
  `boxfilter.rs`, `quantize.rs`, `postprocess.rs`, `tiles.rs`; `sheet.rs` draws the
  `compare` sheets.
- `src/npsum.rs`: numpy's float64 summation orders (pairwise `np.sum`, sequential
  leading-axis reductions), for the sums whose comparisons decide median cut, ramps and
  tile de-duplication. `src/pyjson.rs`: Python's `json.dumps` layouts and `round(x, n)` for
  the output JSON files.
- `src/pyrepr.rs`, `src/pyunicode.rs`: the Python text forms the above reproduce (see
  below). `pyunicode/tables.rs` was generated by `pyunicode/gen_tables.py` from Python
  3.11's `unicodedata` (Unicode 14.0.0).

Canonical JSON and Python's `repr` (Config hashes are embedded in every PNG, so these are
byte-exact):

- `Config::canonical_json()` is `json.dumps(config.to_dict(), sort_keys=True,
  separators=(",", ":"))`: keys sorted by code point, no whitespace, `true`/`false`/`null`,
  integers in decimal, strings with `ensure_ascii` escaping (`\"`, `\\`, `\n`, `\r`, `\t`,
  `\b`, `\f`; every other character outside U+0020..U+007E as lowercase `\uxxxx`,
  beyond the BMP as a surrogate pair).
- Floats are Python's `repr` (`pyrepr::float_repr`): the shortest digits that round-trip
  and, if two such strings are equally close, the one with the even last digit (Rust's
  `{:e}` gives the length, `{:.Ne}` the correctly rounded digits). Exponent form when the
  decimal exponent is below -4 or at least 16, with a sign and at least two digits (`1e-05`,
  `1e+16`, `1.5e+300`); otherwise positional with at least one digit after the point
  (`1.0`, `0.0001`, `9999999999999998.0`); `-0.0` keeps its sign. Checked against 10,000
  recorded Python reprs (`tests/oracle/float_repr.json`).
- Error messages use Python's `repr` of the offending value: `True`/`False`/`None`, float
  repr, `[1, 2]` for lists, and strings in single quotes (double quotes when the text has a
  `'` and no `"`), with `\\`, the quote, `\t`, `\n`, `\r` escaped, other control
  characters as `\xhh` and non-printable non-ASCII characters as `\xhh`, `\uhhhh` or
  `\Uhhhhhhhh` (`str.isprintable`, from the generated table).
- `hash()` is the SHA-256 of `canonical_json + "\n" + version`, plus
  `"\npalette:" + palette_sha256` with a palette; the version is the crate's
  (`CARGO_PKG_VERSION`, the workspace version).

Build: a stable Rust toolchain. At the repo root: `cargo build --release -p pixelforge`
(the program in `target/release/pixelforge`), `cargo test --workspace --all-features --
--include-ignored` (the full suite, as CI runs it), `cargo xtask golden --check`; lint with
`cargo fmt --all --check` and `cargo clippy --workspace --all-targets --all-features -- -D
warnings`. The Python binding is built with maturin (`pip install ./crates/pixelforge-py`;
see its README).

Determinism rules (outputs must be byte-identical on every machine):

- float64 only; no fast-math, no `target-cpu=native`, no compiler FMA contraction.
- Every transcendental function from the pure-Rust `libm` crate (`libm::exp`, `expm1`, `log`,
  `pow`, `cbrt`, `sin`, `cos`, `atan2`, `hypot`), never the `f64` methods (`f64::exp`,
  `powf`, `cbrt`, `ln`, ... call the platform libm, which differs in the last bit between
  macOS and Linux). `f64::sqrt` is IEEE-exact and allowed. No `powi` (its result depends on
  the compiler's expansion) and no `mul_add` (it falls back to the platform's C `fma`). An
  explicit `libm::fma` (pure Rust, correctly rounded, identical everywhere) is allowed where
  the reference implementation fused: `kopf_run.rs` uses it to reproduce LAPACK's 2×2
  eigensolver and `np.hypot` bit for bit. Shortcuts around a libm call must be exact
  (kopf skips `exp` below libm's own underflow threshold, where it returns exactly 0).
- Rounding to integers uses `f64::round_ties_even` wherever Python used `np.round` or `round()`
  (half to even: 2.5 -> 2). `f64::round` (half away from zero) is never used.
  `tests/determinism_guard.rs` fails if a source file of either crate contains `.round()`,
  `f64::exp(`, `.exp()`, `.powf(`, `.powi(`, `.cbrt()`, `.ln(`, `.log(`, `.sin(`, `.cos(`,
  `.atan2(` or `.mul_add(` outside a comment; nothing is allow-listed.
- The library functions take slices with explicit dimensions and assert their lengths;
  the Python binding converts and checks numpy arrays before passing them in.
- Every reduction runs in a fixed order. Parallelism only over a fixed partition that does not
  depend on the thread count, merged in partition order: no atomics in the numerics, no
  dynamic chunking. `parallel.rs` provides it: `run_bands` runs one closure per band of
  `BAND_ROWS` (32) input-pixel rows, the last band shorter; each band fills its own
  accumulator (only the kernel rows it touches) and returns it, and the caller merges the
  results in band order after every band has finished (kopf's three passes; the
  preprocessing bilateral denoise uses the same bands for per-pixel work). Threads pull band indices from an
  atomic counter, which decides only when a band runs, never what it computes. Per-pixel
  outputs (kopf's γ) are written into disjoint per-band slices and need no merge.
  `thread_count()` reads `PIXELFORGE_THREADS` (integer >= 1; anything else falls back to
  `std::thread::available_parallelism()`). `BAND_ROWS` is part of the numerics: changing it
  changes summation order, so treat it like an algorithm constant.
- No RNG and no hash-map iteration order in anything numeric.
