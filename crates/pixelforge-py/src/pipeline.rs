//! Python wrappers of `pixelforge::pipeline`: the whole pipeline behind `pixelforge.run` /
//! `run_loaded`, `Result.save`, `output_paths`, `environment` and `uses_gerstner_palette`.
//!
//! The Config arrives as `config.to_dict()`, plus the Python Config's cached palette lines
//! when it has them (`palette_lines`), so both sides use one reading of a palette file.
//! Results are dicts of numpy arrays and plain Python values; `stats` and `tilemap` keep
//! their key order. Errors keep Python's classes (`ConfigError`, `PixelforgeError`,
//! `OSError`); the layout's aspect warning and the multi-frame notice are `UserWarning`s.

use std::ffi::CString;
use std::path::PathBuf;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pixelforge::config::Config;
use pixelforge::io::{self, Loaded};
use pixelforge::pipeline::{self, Error, IndexImage, MetaExtras, Output};
use pixelforge::pyjson::Json;

use crate::color::{shaped, slice};
use crate::config::build;
use crate::mask::{array, nonempty, pixelforge_error, same_size};
use crate::pyjson::to_json;

fn error_py(py: Python<'_>, e: Error) -> PyErr {
    match e {
        Error::Config(e) => pixelforge_error(py, "ConfigError", &e.0),
        Error::Pixelforge(e) => pixelforge_error(py, "PixelforgeError", &e.0),
        Error::Io(e) => crate::io::io_error(py, e),
        Error::Value(e) => PyValueError::new_err(e),
    }
}

/// `Config(**config)` with the caller's cached palette lines.
fn config_of(
    py: Python<'_>,
    config: &Bound<'_, PyDict>,
    palette_lines: Option<Vec<String>>,
) -> PyResult<Config> {
    let config = build(py, config)?;
    if let Some(lines) = palette_lines {
        config.seed_palette_hex_lines(lines);
    }
    Ok(config)
}

pub(crate) fn json_to_py<'py>(py: Python<'py>, v: &Json) -> PyResult<Bound<'py, PyAny>> {
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

fn warn(py: Python<'_>, texts: &[String]) -> PyResult<()> {
    for text in texts {
        let message =
            CString::new(text.as_str()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 1)?;
    }
    Ok(())
}

/// The `Output` as a dict: image (H, W, 4) uint8, palette (K, 3) uint8, indices (H, W)
/// int64, palette_lab (K, 3) float64, input_sha256, stats (dict), tileset / tilemap / tileset_indices
/// (None without a tileset), warnings (list of str).
fn output_to_py<'py>(py: Python<'py>, out: Output) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let (h, w) = (out.indices.h, out.indices.w);
    let k = out.palette.len();
    d.set_item("image", shaped(py, &[h, w, 4], out.image))?;
    d.set_item(
        "palette",
        shaped(py, &[k, 3], out.palette.into_iter().flatten().collect()),
    )?;
    d.set_item("indices", shaped(py, &[h, w], out.indices.indices))?;
    d.set_item("palette_lab", shaped(py, &[k, 3], out.palette_lab))?;
    d.set_item("input_sha256", out.input_sha256)?;
    d.set_item("stats", json_to_py(py, &out.stats)?)?;
    match (out.tileset, out.tileset_indices, out.tilemap) {
        (Some(rgba), Some(sheet), Some(map)) => {
            d.set_item("tileset", shaped(py, &[sheet.h, sheet.w, 4], rgba))?;
            d.set_item(
                "tileset_indices",
                shaped(py, &[sheet.h, sheet.w], sheet.indices),
            )?;
            d.set_item("tilemap", json_to_py(py, &pipeline::tilemap_json(&map))?)?;
        }
        _ => {
            d.set_item("tileset", py.None())?;
            d.set_item("tileset_indices", py.None())?;
            d.set_item("tilemap", py.None())?;
        }
    }
    d.set_item("warnings", out.warnings)?;
    Ok(d)
}

fn timings_of(timings: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, f64)>> {
    match timings {
        None => Ok(Vec::new()),
        Some(t) => t
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
            .collect(),
    }
}

fn run_and_convert<'py>(
    py: Python<'py>,
    loaded: &Loaded,
    config: &Config,
    timings: Vec<(String, f64)>,
    alpha_matted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let out = py
        .detach(|| pipeline::run_loaded(loaded, config, timings, alpha_matted))
        .map_err(|e| error_py(py, e))?;
    warn(py, &out.warnings)?;
    output_to_py(py, out)
}

/// `pipeline.run_loaded` on `(rgb (H, W, 3) float64, alpha (H, W) uint8, sha256)` with the
/// timings recorded so far. `alpha_matted`: the caller already applied `remove_bg` (else
/// the external `rembg` command runs).
#[pyfunction]
#[pyo3(signature = (rgb, alpha, sha256, config, timings=None, alpha_matted=false, palette_lines=None))]
#[allow(clippy::too_many_arguments)]
fn pipeline_run_loaded<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
    sha256: String,
    config: &Bound<'py, PyDict>,
    timings: Option<&Bound<'py, PyDict>>,
    alpha_matted: bool,
    palette_lines: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    const F: &str = "pipeline_run_loaded";
    crate::pylog::sync_level(py)?;
    let config = config_of(py, config, palette_lines)?;
    let (rshape, rarr) = array::<f64>(F, "rgb", "float64", rgb, true)?;
    let (ashape, aarr) = array::<u8>(F, "alpha", "uint8", alpha, false)?;
    same_size(F, &rshape, &ashape, ("rgb", "alpha"))?;
    nonempty(F, &ashape)?;
    let loaded = Loaded {
        rgb: slice(F, "rgb", &rarr)?.to_vec(),
        alpha: slice(F, "alpha", &aarr)?.to_vec(),
        h: ashape[0],
        w: ashape[1],
        sha256,
        warnings: Vec::new(),
    };
    let timings = timings_of(timings)?;
    run_and_convert(py, &loaded, &config, timings, alpha_matted)
}

/// `pipeline.run` for a path: load (timed as `load`; the multi-frame notice is a
/// `UserWarning` right away), then [`pipeline_run_loaded`].
#[pyfunction]
#[pyo3(signature = (path, config, max_pixels=None, palette_lines=None))]
fn pipeline_run<'py>(
    py: Python<'py>,
    path: PathBuf,
    config: &Bound<'py, PyDict>,
    max_pixels: Option<i64>,
    palette_lines: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    crate::pylog::sync_level(py)?;
    let config = config_of(py, config, palette_lines)?;
    let budget = match max_pixels {
        None => io::MAX_INPUT_PIXELS,
        Some(n) => u64::try_from(n).unwrap_or(0), // a negative budget refuses everything
    };
    let start = std::time::Instant::now();
    let loaded = py
        .detach(|| io::load(&path, budget))
        .map_err(|e| crate::io::io_error(py, e))?;
    let timings = vec![("load".to_string(), start.elapsed().as_secs_f64())];
    warn(py, &loaded.warnings)?;
    run_and_convert(py, &loaded, &config, timings, false)
}

fn index_image(arr: &PyReadonlyArray2<'_, i64>, name: &str) -> PyResult<IndexImage> {
    let shape = arr.shape();
    let data = arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("result_save: {name}: {e}")))?;
    Ok(IndexImage {
        indices: data.to_vec(),
        h: shape[0],
        w: shape[1],
    })
}

fn triples<T: numpy::Element + Copy>(
    arr: &PyReadonlyArray2<'_, T>,
    name: &str,
) -> PyResult<Vec<T>> {
    if arr.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "result_save: {name} must have shape (K, 3), got {:?}",
            arr.shape()
        )));
    }
    Ok(arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("result_save: {name}: {e}")))?
        .to_vec())
}

fn paths_to_py<'py>(
    py: Python<'py>,
    paths: Vec<(impl AsRef<str>, PathBuf)>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, path) in paths {
        d.set_item(name.as_ref(), path.into_os_string())?;
    }
    Ok(d)
}

/// `Result.save(prefix)`: rebuild the `Output` from the Result's fields and write every
/// file. `environment` (dict) replaces the native `environment()` in `_meta.json`, and
/// `remove_bg_versions` (dict) the external command's versions, when given. Returns
/// `{name: path}` (str) in write order.
#[pyfunction]
#[pyo3(signature = (prefix, indices, palette_rgb8, palette_lab, stats, config, input_sha256,
                    tileset_indices=None, tilemap=None, environment=None,
                    remove_bg_versions=None, palette_lines=None))]
#[allow(clippy::too_many_arguments)]
fn result_save<'py>(
    py: Python<'py>,
    prefix: PathBuf,
    indices: PyReadonlyArray2<'py, i64>,
    palette_rgb8: PyReadonlyArray2<'py, u8>,
    palette_lab: PyReadonlyArray2<'py, f64>,
    stats: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
    input_sha256: String,
    tileset_indices: Option<PyReadonlyArray2<'py, i64>>,
    tilemap: Option<&Bound<'py, PyAny>>,
    environment: Option<&Bound<'py, PyAny>>,
    remove_bg_versions: Option<&Bound<'py, PyAny>>,
    palette_lines: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = config_of(py, config, palette_lines)?;
    let palette: Vec<[u8; 3]> = triples(&palette_rgb8, "palette_rgb8")?
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let palette_lab = triples(&palette_lab, "palette_lab")?;
    if palette_lab.len() != palette.len() * 3 {
        return Err(PyValueError::new_err(
            "result_save: palette_rgb8 and palette_lab have different lengths",
        ));
    }
    let output = Output {
        image: Vec::new(), // not read by save
        palette,
        indices: index_image(&indices, "indices")?,
        config,
        input_sha256,
        palette_lab,
        stats: to_json(stats)?,
        tileset: None,
        tilemap: tilemap
            .map(|m| crate::postprocess::tilemap_from_py("result_save", m))
            .transpose()?,
        tileset_indices: tileset_indices
            .as_ref()
            .map(|a| index_image(a, "tileset_indices"))
            .transpose()?,
        warnings: Vec::new(),
    };
    let extras = MetaExtras {
        environment: environment.map(to_json).transpose()?,
        remove_bg_versions: remove_bg_versions.map(to_json).transpose()?,
    };
    let paths = py
        .detach(|| output.save_with(&prefix, &extras))
        .map_err(|e| error_py(py, e))?;
    paths_to_py(py, paths)
}

/// `pipeline.output_paths(prefix, tileset)`: `{name: path}` in write order.
#[pyfunction]
fn output_paths(py: Python<'_>, prefix: PathBuf, tileset: bool) -> PyResult<Bound<'_, PyDict>> {
    paths_to_py(py, pipeline::output_paths(&prefix, tileset))
}

/// The native `environment()`: platform, threads and the crate version (`native`).
#[pyfunction]
fn environment(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    json_to_py(py, &pipeline::environment())
}

/// `pipeline.uses_gerstner_palette(Config(**config))`.
#[pyfunction]
fn uses_gerstner_palette(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<bool> {
    Ok(pipeline::uses_gerstner_palette(&build(py, config)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pipeline_run_loaded, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_run, m)?)?;
    m.add_function(wrap_pyfunction!(result_save, m)?)?;
    m.add_function(wrap_pyfunction!(output_paths, m)?)?;
    m.add_function(wrap_pyfunction!(environment, m)?)?;
    m.add_function(wrap_pyfunction!(uses_gerstner_palette, m)?)?;
    Ok(())
}
