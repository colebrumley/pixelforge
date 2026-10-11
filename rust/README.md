# pixelforge_native

The Rust extension module `pixelforge._native` (PyO3 + the `numpy` crate). It will hold the
per-pixel hot loops of the downscalers; orchestration, config and I/O stay in Python. See
`docs/rust-port-plan.md`.

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
- Every reduction runs in a fixed order. Parallelism only over a fixed partition that does not
  depend on the thread count, merged in partition order: no atomics, no dynamic chunking.
- No RNG and no hash-map iteration order in anything numeric.
