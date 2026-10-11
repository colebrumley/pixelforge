//! pixelforge: deterministic conversion of raster images into 16-bit-style pixel art.
//!
//! This crate is the Rust library, with no Python dependency. It holds the downscalers'
//! per-pixel kernels (`kopf`, `gerstner`), the thread-count-independent parallelism they use
//! (`parallel`), the Config with its field table and canonical JSON (`config`), the palette
//! file parsers (`palette`) and the Python text forms those reproduce byte for byte
//! (`pyrepr`, `pyunicode`). The Python package reaches it through the binding crate
//! `pixelforge-py` (module `pixelforge._native`).
//!
//! Determinism rules (see docs/rust-port-plan.md, Decision 3): float64 only, no fast-math,
//! no `target-cpu=native`, no `mul_add` unless used unconditionally, `exp` from the `libm`
//! crate, and reductions in a fixed order that does not depend on the thread count.

pub mod color;
pub mod config;
pub mod gerstner;
pub mod kopf;
pub mod palette;
pub mod parallel;
pub mod pyrepr;
pub mod pyunicode;
