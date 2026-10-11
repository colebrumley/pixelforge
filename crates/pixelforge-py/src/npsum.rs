//! numpy-compatible reductions, exposed so `tests/test_npsum.py` can compare them with
//! `np.sum` while the Python package exists.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// `np.sum(arr)` for a 1-D float64 array (any stride), bit for bit.
#[pyfunction]
fn np_pairwise_sum(arr: PyReadonlyArray1<'_, f64>) -> f64 {
    match arr.as_slice() {
        Ok(slice) => pixelforge::npsum::pairwise_sum(slice),
        Err(_) => {
            pixelforge::npsum::pairwise_sum(&arr.as_array().iter().copied().collect::<Vec<_>>())
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(np_pairwise_sum, m)?)?;
    Ok(())
}
