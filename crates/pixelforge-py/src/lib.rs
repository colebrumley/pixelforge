//! Python binding of the pixelforge library, imported as `pixelforge._native` by the
//! package in `python/pixelforge/` (built with maturin; see README.md).
//!
//! The library (crates/pixelforge) does all the work; this crate converts arguments and
//! results, releases the GIL around the pipeline and raises the package's exceptions
//! (`pixelforge.ConfigError`, `pixelforge.PixelforgeError`, `OSError`). It exposes:
//!
//! - `NativeConfig` (config.rs): the Rust `Config`, wrapped by `pixelforge.Config`;
//!   `config_fields()` and `config_presets()`.
//! - `load`, `from_array` (io.rs): decode a file or convert an array to
//!   `(rgb, alpha, sha256)`; `MAX_INPUT_PIXELS`.
//! - `run_loaded` and `NativeOutput` (pipeline.rs): the pipeline and its result, which
//!   writes the output files (`save`).
//! - `native_version()`, `thread_count()`, and the `log` bridge to Python's `logging`
//!   (pylog.rs).

mod config;
mod io;
mod pipeline;
mod pylog;

use pyo3::prelude::*;

/// The crate version (the workspace version), `pixelforge.__version__`.
#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Worker threads the pipeline uses: `PIXELFORGE_THREADS` if it is an integer >= 1, else
/// the available parallelism. Affects speed only; recorded in `_meta.json`.
#[pyfunction]
fn thread_count() -> usize {
    pixelforge::parallel::thread_count()
}

/// `pixelforge.<class>(message)` (`ConfigError` or `PixelforgeError`, defined in the
/// package's `__init__.py`), or the import error if the package is not importable.
pub(crate) fn package_error(py: Python<'_>, class: &str, message: &str) -> PyErr {
    match py
        .import("pixelforge")
        .and_then(|m| m.getattr(class))
        .and_then(|cls| cls.call1((message,)))
    {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pylog::init(m.py());
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    config::register(m)?;
    io::register(m)?;
    pipeline::register(m)?;
    Ok(())
}
