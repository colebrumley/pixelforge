//! Forwards the library's `log` records to Python's `logging`.
//!
//! The Rust target `pixelforge::kopf` becomes the Python logger `pixelforge.kopf`, a child
//! of the `pixelforge` logger the CLI and the tests configure. The crate-wide maximum level
//! mirrors that logger's effective level: binding functions that run long loops call
//! `sync_level` on entry, so disabled levels cost nothing (no GIL, no formatting) and
//! `logging` never changes a result (the records only read values).

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::prelude::*;

struct PyLogger;

static LOGGER: PyLogger = PyLogger;

fn py_level(level: Level) -> u8 {
    match level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    }
}

impl Log for PyLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true // `log::max_level()` (kept in sync with Python) does the filtering
    }

    fn log(&self, record: &Record) {
        let name = record.target().replace("::", ".");
        let message = record.args().to_string();
        let level = py_level(record.level());
        // Re-attaches to the interpreter if the caller released the GIL around the loop.
        let _ = Python::attach(|py| -> PyResult<()> {
            let logger = py.import("logging")?.call_method1("getLogger", (name,))?;
            logger.call_method1("log", (level, message))?;
            Ok(())
        });
    }

    fn flush(&self) {}
}

/// Install the forwarder (once) and take the current Python level.
pub fn init(py: Python<'_>) {
    let _ = log::set_logger(&LOGGER);
    let _ = sync_level(py);
}

/// Set the crate-wide maximum level from the `pixelforge` logger's effective level.
pub fn sync_level(py: Python<'_>) -> PyResult<()> {
    let level: i64 = py
        .import("logging")?
        .call_method1("getLogger", ("pixelforge",))?
        .call_method0("getEffectiveLevel")?
        .extract()?;
    log::set_max_level(match level {
        l if l <= 10 => LevelFilter::Debug,
        l if l <= 20 => LevelFilter::Info,
        l if l <= 30 => LevelFilter::Warn,
        _ => LevelFilter::Error,
    });
    Ok(())
}

/// Emit one record from Rust (for `tests/test_logging_bridge.py`).
#[pyfunction]
fn log_probe(py: Python<'_>, target: &str, level: &str, message: &str) -> PyResult<()> {
    sync_level(py)?;
    let level = match level {
        "error" => Level::Error,
        "warning" => Level::Warn,
        "info" => Level::Info,
        "debug" => Level::Debug,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "log_probe: unknown level {other:?}"
            )))
        }
    };
    log::log!(target: target, level, "{message}");
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(log_probe, m)?)?;
    Ok(())
}
