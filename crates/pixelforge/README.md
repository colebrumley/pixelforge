# pixelforge (Rust library)

The Rust library behind pixelforge: plain Rust, no Python dependency (only `libm`). It holds
the per-pixel hot loops of the downscalers and the Config field table; orchestration, config
handling and I/O are still in Python while the migration in `docs/rust-migration-plan.md`
proceeds. Python reaches it through the binding crate `crates/pixelforge-py` (PyO3 + the
`numpy` crate), built as the extension module `pixelforge._native`.

Workspace layout (repo root):

- `Cargo.toml`: the workspace, the shared version (`[workspace.package]`, kept equal to
  `pixelforge/version.py` and `pyproject.toml`) and the release profile (profiles only take
  effect in the root manifest). One `Cargo.lock` and `target/` at the root.
- `crates/pixelforge`: this library.
- `crates/pixelforge-py`: the binding (`cdylib` `pixelforge_native`, `#[pymodule] _native`).
  Every `#[pyfunction]` checks dtypes, shapes and contiguity, raises `ValueError` naming the
  argument, releases the GIL and calls the library.

Modules:

- `src/kopf.rs`: `em_step` (fused E-step + M-step sums), `c_step` (directional and overlap
  sums) and `spatial_refill` (spatial-only recoloring of starved kernels); in Python
  `kopf_em_step`, `kopf_c_step` and `kopf_spatial_refill`. They reduce over bands of
  `BAND_ROWS` (32) rows merged in band order; the bands are part of the numerics. The numpy
  versions are kept in `pixelforge/downscale/_reference/kopf.py`.
- `src/gerstner.rs`: `slic_assign` (one modified-SLIC assignment pass, bitwise equal to the
  numpy `_assign`), `bilateral_filter` (the superpixel bilateral filter, `libm::exp`) and
  `superpixel_stats` (count, mass, colour and centre sums per superpixel, bitwise equal to
  the seven `np.bincount` calls it replaces); same names in Python. `slic_assign` and
  `bilateral_filter` are per-pixel independent and run over fixed 8-row bands on
  `PIXELFORGE_THREADS` threads (default: available parallelism); those bands are not part
  of the numerics. `superpixel_stats` is single-threaded on purpose: it sums in row-major
  pixel order like `np.bincount`, and per-band partial sums would change the rounding. The
  numpy versions are kept in `pixelforge/downscale/_reference/gerstner.py`.
- `src/parallel.rs`: `thread_count` and `run_bands` (see the determinism rules below).
- `src/config.rs`: the Config field table (name, type, default, help, choices, upper bound
  per field; the presets and their longest edge), the single source the Rust Config, JSON
  and CLI will be built from. `tests/test_field_table.py` compares it with
  `pixelforge/config.py` through `_native.config_fields()` / `config_presets()`.

The binding's own functions: `native_version`, `sum_f64` (sequential sum, a smoke test) and
`thread_count` (the worker-thread count from `PIXELFORGE_THREADS`, recorded in
`_meta.json`).

Build: maturin is the Python package's build backend, so `uv sync --locked` (or any
`pip install`) compiles the binding and places `_native.abi3.so` inside `pixelforge/`. Needs
a stable Rust toolchain. At the repo root: `cargo check -p pixelforge` and
`cargo test -p pixelforge` need no Python; lint with `cargo fmt --all --check` and
`cargo clippy --workspace --all-targets -- -D warnings`. A bare `cargo build` of the binding
fails to link on macOS (Python symbols resolve at import time); use `uv sync` or
`cargo check`.

Determinism rules (outputs must be byte-identical on every machine):

- float64 only; no fast-math, no `target-cpu=native`, no `mul_add` unless used unconditionally.
- `exp` from the pure-Rust `libm` crate, never `f64::exp` (platform libm differs in the last bit).
  Shortcuts around it must be exact (kopf skips the call below libm's own underflow
  threshold, where it returns exactly 0).
- Python passes C-contiguous arrays of the exact dtype and shape; the binding checks them and
  raises `ValueError` naming the argument (see the helpers in
  `crates/pixelforge-py/src/kopf.rs`) rather than relying on pyo3's conversion error. The
  library functions take slices and assert their lengths.
- Every reduction runs in a fixed order. Parallelism only over a fixed partition that does not
  depend on the thread count, merged in partition order: no atomics in the numerics, no
  dynamic chunking. `parallel.rs` provides it: `run_bands` runs one closure per band of
  `BAND_ROWS` (32) input-pixel rows, the last band shorter; each band fills its own
  accumulator (only the kernel rows it touches) and returns it, and the caller merges the
  results in band order after every band has finished. Threads pull band indices from an
  atomic counter, which decides only when a band runs, never what it computes. Per-pixel
  outputs (kopf's γ) are written into disjoint per-band slices and need no merge.
  `thread_count()` reads `PIXELFORGE_THREADS` (integer >= 1; anything else falls back to
  `std::thread::available_parallelism()`). `BAND_ROWS` is part of the numerics: changing it
  changes summation order, so treat it like an algorithm constant.
- No RNG and no hash-map iteration order in anything numeric.
