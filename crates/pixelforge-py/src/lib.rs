//! Python binding of the pixelforge library, imported as `pixelforge._native`.
//!
//! The numerical code lives in the `pixelforge` crate (crates/pixelforge); this crate only
//! converts and checks arguments, releases the GIL and calls it. Determinism rules: see
//! crates/pixelforge/README.md.

mod color;
mod config;
mod filters;
mod gerstner;
mod kopf;
mod npsum;
mod postprocess;
mod quantize;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// The crate version; kept equal to `pixelforge.__version__`.
#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Sum a 1-D float64 array.
///
/// The sum runs sequentially in index order on purpose: a fixed summation order makes the
/// result bit-identical on every machine and equal to a plain Python loop. Do not replace it
/// with chunked, SIMD-lane or pairwise summation.
#[pyfunction]
fn sum_f64(arr: PyReadonlyArray1<'_, f64>) -> f64 {
    let view = arr.as_array();
    let mut total = 0.0_f64;
    for &x in view.iter() {
        total += x;
    }
    total
}

/// Worker threads the native kernels use: `PIXELFORGE_THREADS` if it is an integer >= 1,
/// else the available parallelism. Affects speed only; recorded in `_meta.json`.
#[pyfunction]
fn thread_count() -> usize {
    pixelforge::parallel::thread_count()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(sum_f64, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    gerstner::register(m)?;
    m.add_function(wrap_pyfunction!(kopf::kopf_em_step, m)?)?;
    m.add_function(wrap_pyfunction!(kopf::kopf_c_step, m)?)?;
    m.add_function(wrap_pyfunction!(kopf::kopf_spatial_refill, m)?)?;
    config::register(m)?;
    filters::register(m)?;
    color::register(m)?;
    npsum::register(m)?;
    postprocess::register(m)?;
    quantize::register(m)?;
    Ok(())
}
