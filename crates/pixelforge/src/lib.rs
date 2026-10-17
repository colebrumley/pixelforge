//! pixelforge: deterministic conversion of raster images into 16-bit-style pixel art.
//!
//! The whole application lives in this crate: `config` (the Config, its field table and the
//! canonical JSON), `io` (decoding, indexed PNG output), `preprocess` with `mask`, `layout`,
//! `filters` and `enhance`, the downscalers (`boxfilter`, `kopf` + `kopf_run`, `gerstner` +
//! `gerstner_run` + `annealer`), `palette`, `quantize`, `postprocess`, `tiles`, the
//! `pipeline` that orchestrates them and writes every output file, the pixel editor's core
//! (`editor`: document, ops, history, tools, edit log and replay; no GUI dependency), the
//! editor's window behind the `gui` feature (`gui`), and, behind the `cli` feature, the
//! `pixelforge` binary (`cli`, `sheet`). `npsum`, `pyjson`, `pyrepr` and
//! `pyunicode` reproduce the numpy and Python text forms the outputs were defined by, so
//! files stay byte-identical to the version this crate replaced. The optional Python
//! binding (`crates/pixelforge-py`) wraps `pipeline` and `config`.
//!
//! Determinism rules (docs/design/implementation.md, "Implementation"): float64 only, every
//! transcendental from the `libm` crate, `round_ties_even`, no fast-math or
//! `target-cpu=native`, no `mul_add` (an explicit `libm::fma` only where a reference
//! implementation fuses), and reductions in a fixed order that does not depend on the thread
//! count (`parallel`).

pub mod annealer;
pub mod boxfilter;
#[cfg(feature = "cli")]
pub mod cli;
pub mod color;
pub mod config;
pub mod downscale;
pub mod editor;
pub mod enhance;
pub mod filters;
pub mod gerstner;
pub mod gerstner_run;
#[cfg(feature = "gui")]
pub mod gui;
pub mod io;
pub mod kopf;
pub mod kopf_run;
pub mod layout;
pub mod mask;
pub mod npsum;
pub mod palette;
pub mod parallel;
pub mod pipeline;
pub mod postprocess;
pub mod preprocess;
pub mod pyjson;
pub mod pyrepr;
pub mod pyunicode;
pub mod quantize;
pub mod sheet;
pub mod tiles;
