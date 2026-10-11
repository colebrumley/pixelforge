# pixelforge_native

The Rust extension module `pixelforge._native` (PyO3 + the `numpy` crate). It holds the
per-pixel hot loops of the downscalers (so far kopf: `src/kopf.rs`); orchestration, config
and I/O stay in Python. See `docs/rust-port-plan.md`.

Functions:

- `src/gerstner.rs`: `slic_assign` (one modified-SLIC assignment pass, bitwise equal to the
  numpy `_assign`) and `bilateral_filter` (the superpixel bilateral filter, `libm::exp`).
  Both run over fixed row bands on `PIXELFORGE_THREADS` threads (default: available
  parallelism); the thread count never changes the output. The numpy versions they replace
  are kept in `pixelforge/downscale/_reference/gerstner.py` for tests and benchmarks.

Build: maturin is the package's build backend, so `uv sync --locked` (or any `pip install`)
compiles it and places `_native.abi3.so` inside `pixelforge/`. Needs a stable Rust
toolchain. Lint with `cargo fmt --all --check` and `cargo clippy --all-targets -- -D warnings`.
A bare `cargo build` fails to link on macOS (Python symbols resolve at import time); use
`uv sync` or `cargo check`.

Determinism rules (outputs must be byte-identical on every machine):

- float64 only; no fast-math, no `target-cpu=native`, no `mul_add` unless used unconditionally.
- `exp` from the pure-Rust `libm` crate, never `f64::exp` (platform libm differs in the last bit).
  Shortcuts around it must be exact (kopf skips the call below libm's own underflow
  threshold, where it returns exactly 0).
- Python passes C-contiguous arrays of the exact dtype and shape; check them and raise
  `ValueError` naming the argument (see the helpers in `kopf.rs`) rather than relying on
  pyo3's conversion error.
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
