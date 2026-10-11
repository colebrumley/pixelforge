//! `run_loaded`: the whole pipeline on `(rgb, alpha, sha256)` with a `NativeConfig`, and
//! its result `NativeOutput` (numpy arrays and plain Python values; `save` writes the
//! output files exactly as `pixelforge convert` does).
//!
//! Errors keep the package's classes (`ConfigError`, `PixelforgeError`) and `OSError`; the
//! layout's aspect warning is a `UserWarning`. With `remove_bg`, the library runs the
//! external `rembg` command.

use std::path::PathBuf;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pixelforge::io::Loaded;
use pixelforge::pipeline::{self, Error, Output};
use pixelforge::pyjson::Json;

use crate::config::NativeConfig;
use crate::io::{io_error, warn};
use crate::package_error;

fn error_py(py: Python<'_>, e: Error) -> PyErr {
    match e {
        Error::Config(e) => package_error(py, "ConfigError", &e.0),
        Error::Pixelforge(e) => package_error(py, "PixelforgeError", &e.0),
        Error::Io(e) => io_error(py, e),
        Error::Value(e) => PyValueError::new_err(e),
    }
}

fn json_to_py<'py>(py: Python<'py>, v: &Json) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Json::Null => py.None().into_bound(py),
        Json::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Json::Int(i) => i.into_pyobject(py)?.into_any(),
        Json::Float(f) => f.into_pyobject(py)?.into_any(),
        Json::Str(s) => s.into_pyobject(py)?.into_any(),
        Json::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Json::Object(items) => {
            let dict = PyDict::new(py);
            for (k, item) in items {
                dict.set_item(k, json_to_py(py, item)?)?;
            }
            dict.into_any()
        }
    })
}

fn shaped<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    shape: &[usize],
    data: &[T],
) -> Bound<'py, PyArrayDyn<T>> {
    ArrayD::from_shape_vec(IxDyn(shape), data.to_vec())
        .expect("data length matches the shape")
        .into_pyarray(py)
}

/// The result of a run. The getters return fresh numpy arrays and Python values.
#[pyclass(name = "NativeOutput", module = "pixelforge._native", frozen)]
pub struct NativeOutput {
    output: Output,
}

#[pymethods]
impl NativeOutput {
    /// `(H, W, 4)` uint8 RGBA at native pixel-art resolution.
    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<u8>> {
        let (h, w) = (self.output.indices.h, self.output.indices.w);
        shaped(py, &[h, w, 4], &self.output.image)
    }

    /// `(K, 3)` uint8.
    #[getter]
    fn palette<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<u8>> {
        let flat: Vec<u8> = self.output.palette.iter().flatten().copied().collect();
        shaped(py, &[self.output.palette.len(), 3], &flat)
    }

    /// `(H, W)` int64 palette indices, -1 for transparent.
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<i64>> {
        let ix = &self.output.indices;
        shaped(py, &[ix.h, ix.w], &ix.indices)
    }

    /// `(K, 3)` float64 CIELAB of the palette.
    #[getter]
    fn palette_lab<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        shaped(
            py,
            &[self.output.palette.len(), 3],
            &self.output.palette_lab,
        )
    }

    #[getter]
    fn input_sha256(&self) -> &str {
        &self.output.input_sha256
    }

    /// The stats dict of `_meta.json` (keys in their written order).
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.output.stats)
    }

    /// The tileset sheet as `(h, w, 4)` uint8 RGBA, or None without a tileset.
    #[getter]
    fn tileset<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<u8>>> {
        let rgba = self.output.tileset.as_ref()?;
        let sheet = self.output.tileset_indices.as_ref()?;
        Some(shaped(py, &[sheet.h, sheet.w, 4], rgba))
    }

    /// The tileset sheet's `(h, w)` int64 indices, or None.
    #[getter]
    fn tileset_indices<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<i64>>> {
        let sheet = self.output.tileset_indices.as_ref()?;
        Some(shaped(py, &[sheet.h, sheet.w], &sheet.indices))
    }

    /// The tilemap dict of `_tilemap.json`, or None.
    #[getter]
    fn tilemap<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.output
            .tilemap
            .as_ref()
            .map(|m| json_to_py(py, &pipeline::tilemap_json(m)))
            .transpose()
    }

    /// Write `<prefix>.png`, `_preview.png`, `_palette.json`, `_palette.hex`, `_meta.json`
    /// and, with a tileset, `_tileset.png`, `_tilemap.json`, `.tmj` and `_tilemap.csv`.
    /// Returns `{name: path}` (str) in write order.
    fn save<'py>(&self, py: Python<'py>, prefix: PathBuf) -> PyResult<Bound<'py, PyDict>> {
        let paths = py
            .detach(|| self.output.save(&prefix))
            .map_err(|e| error_py(py, e))?;
        let d = PyDict::new(py);
        for (name, path) in paths {
            d.set_item(name, path.into_os_string())?;
        }
        Ok(d)
    }
}

/// The pipeline on an already loaded image: `rgb` (H, W, 3) float64 sRGB in [0, 1],
/// `alpha` (H, W) uint8, both C-contiguous; `timings` the stage times recorded so far
/// (e.g. `{"load": seconds}`), kept first in `stats["timings"]`.
#[pyfunction]
#[pyo3(signature = (rgb, alpha, sha256, config, timings=None))]
fn run_loaded(
    py: Python<'_>,
    rgb: PyReadonlyArray3<'_, f64>,
    alpha: PyReadonlyArray2<'_, u8>,
    sha256: String,
    config: &Bound<'_, NativeConfig>,
    timings: Option<&Bound<'_, PyDict>>,
) -> PyResult<NativeOutput> {
    crate::pylog::sync_level(py)?;
    let (rshape, ashape) = (rgb.shape().to_vec(), alpha.shape().to_vec());
    if rshape[2] != 3 || rshape[..2] != ashape[..] {
        return Err(PyValueError::new_err(format!(
            "run_loaded: rgb must be (H, W, 3) and alpha (H, W), got {rshape:?} and {ashape:?}"
        )));
    }
    if ashape[0] == 0 || ashape[1] == 0 {
        return Err(PyValueError::new_err(format!(
            "run_loaded: the image must not be empty, got {ashape:?}"
        )));
    }
    let loaded = Loaded {
        rgb: rgb
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("run_loaded: rgb: {e}")))?
            .to_vec(),
        alpha: alpha
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("run_loaded: alpha: {e}")))?
            .to_vec(),
        h: ashape[0],
        w: ashape[1],
        sha256,
        warnings: Vec::new(),
    };
    let timings: Vec<(String, f64)> = match timings {
        None => Vec::new(),
        Some(t) => t
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
            .collect::<PyResult<_>>()?,
    };
    let config = config.get().config.clone();
    let output = py
        .detach(|| pipeline::run_loaded(&loaded, &config, timings, false))
        .map_err(|e| error_py(py, e))?;
    warn(py, &output.warnings, 2)?;
    Ok(NativeOutput { output })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeOutput>()?;
    m.add_function(wrap_pyfunction!(run_loaded, m)?)?;
    Ok(())
}
