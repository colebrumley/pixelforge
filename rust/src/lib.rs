//! Native extension for pixelforge, imported as `pixelforge._native`.
//!
//! Determinism rules (see docs/rust-port-plan.md, Decision 3): float64 only, no fast-math,
//! no `target-cpu=native`, no `mul_add` unless used unconditionally, `exp` from the `libm`
//! crate, and reductions in a fixed order that does not depend on the thread count.

mod kopf;
mod parallel;

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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(sum_f64, m)?)?;
    m.add_function(wrap_pyfunction!(kopf::kopf_em_step, m)?)?;
    m.add_function(wrap_pyfunction!(kopf::kopf_c_step, m)?)?;
    m.add_function(wrap_pyfunction!(kopf::kopf_spatial_refill, m)?)?;
    Ok(())
}
