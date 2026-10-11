//! Python wrappers of `pixelforge::postprocess` and `pixelforge::tiles`: argument checking
//! (dtype and shape, each failure a `ValueError` naming the function and the argument), then
//! the library call with the GIL released. Arrays of any layout are accepted (copied).
//!
//! Index images are int64 `(H, W)`, palettes float64 `(K, 3)`, tile stacks int64
//! `(n, t, t)`; a tilemap is the `out_tilemap.json` dict (`tile_size`, `width_tiles`,
//! `height_tiles`, `tileset_columns`, `tiles` = `[[tile_id, flip_h, flip_v], ...]`).

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pixelforge::postprocess::{self, PostConfig, Processed};
use pixelforge::tiles::{self, TileConfig, Tilemap};

fn value_error(func: &str, msg: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{func}: {msg}"))
}

/// A numpy array argument of element type `T` and `ndim` dimensions: its shape and data in
/// C order.
fn array<T: numpy::Element + Copy>(
    func: &str,
    name: &str,
    dtype: &str,
    obj: &Bound<'_, PyAny>,
    ndim: usize,
) -> PyResult<(Vec<usize>, Vec<T>)> {
    let arr = obj.cast::<PyArrayDyn<T>>().map_err(|_| {
        let got = obj
            .getattr("dtype")
            .map(|d| d.to_string())
            .unwrap_or_else(|_| obj.get_type().to_string());
        value_error(
            func,
            format!("{name} must be a {dtype} numpy.ndarray, got {got}"),
        )
    })?;
    let shape = arr.shape().to_vec();
    if shape.len() != ndim {
        return Err(value_error(
            func,
            format!("{name} must be {ndim}-dimensional, got shape {shape:?}"),
        ));
    }
    let ro = arr
        .try_readonly()
        .map_err(|e| value_error(func, format!("cannot read {name}: {e}")))?;
    let data = ro.as_array().iter().copied().collect();
    Ok((shape, data))
}

/// An int64 `(H, W)` index image: `(data, h, w)`.
fn index_image(func: &str, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<i64>, usize, usize)> {
    let (shape, data) = array::<i64>(func, "indices", "int64", obj, 2)?;
    Ok((data, shape[0], shape[1]))
}

/// A float64 `(K, 3)` palette, checked against the image's indices.
fn palette(func: &str, obj: &Bound<'_, PyAny>, indices: &[i64]) -> PyResult<Vec<f64>> {
    let (shape, data) = array::<f64>(func, "palette_lab", "float64", obj, 2)?;
    if shape[1] != 3 {
        return Err(value_error(
            func,
            format!("palette_lab must have shape (K, 3), got {shape:?}"),
        ));
    }
    if let Some(&bad) = indices.iter().find(|&&v| v >= shape[0] as i64) {
        return Err(value_error(
            func,
            format!("index {bad} is outside the palette ({} entries)", shape[0]),
        ));
    }
    Ok(data)
}

fn image_out(py: Python<'_>, data: Vec<i64>, h: usize, w: usize) -> Bound<'_, PyArray2<i64>> {
    Array2::from_shape_vec((h, w), data)
        .expect("output length matches its shape")
        .into_pyarray(py)
}

fn palette_out(py: Python<'_>, data: Vec<f64>) -> Bound<'_, PyArray2<f64>> {
    let k = data.len() / 3;
    Array2::from_shape_vec((k, 3), data)
        .expect("palette length is a multiple of 3")
        .into_pyarray(py)
}

type Outlined<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f64>>,
    Option<usize>,
    bool,
);

fn outlined(py: Python<'_>, p: Processed) -> Outlined<'_> {
    (
        image_out(py, p.indices, p.height, p.width),
        palette_out(py, p.palette_lab),
        p.stats.outline_index,
        p.stats.outline_clipped,
    )
}

/// `postprocess.run` minus palette ramps (applied by the caller): returns `(indices,
/// palette_lab, outline_index | None, outline_clipped)`. `config` is `Config.to_dict()`.
#[pyfunction]
fn postprocess_run<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    palette_lab: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
    saturated: bool,
    fixed_palette: bool,
    outline_margin: i64,
) -> PyResult<Outlined<'py>> {
    const F: &str = "postprocess_run";
    let (idx, h, w) = index_image(F, indices)?;
    let pal = palette(F, palette_lab, &idx)?;
    let cfg = PostConfig::from(&crate::config::build(py, config)?);
    let out = py
        .detach(|| {
            postprocess::run(
                &idx,
                h,
                w,
                &pal,
                &cfg,
                saturated,
                fixed_palette,
                outline_margin,
            )
        })
        .map_err(|e| value_error(F, e))?;
    Ok(outlined(py, out))
}

/// `postprocess.remove_orphans`: int64 `(H, W)` -> int64 `(H, W)`.
#[pyfunction]
fn remove_orphans<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    palette_lab: &Bound<'py, PyAny>,
    min_region: i64,
    max_passes: i64,
    max_delta: f64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    const F: &str = "remove_orphans";
    let (idx, h, w) = index_image(F, indices)?;
    let pal = palette(F, palette_lab, &idx)?;
    let passes = max_passes.max(0) as usize;
    let out =
        py.detach(|| postprocess::remove_orphans(&idx, h, w, &pal, min_region, passes, max_delta));
    Ok(image_out(py, out, h, w))
}

/// `postprocess.fix_jaggies`: int64 `(H, W)` -> int64 `(H, W)`.
#[pyfunction]
fn fix_jaggies<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let (idx, h, w) = index_image("fix_jaggies", indices)?;
    let out = py.detach(|| postprocess::fix_jaggies(&idx, h, w));
    Ok(image_out(py, out, h, w))
}

/// `postprocess.add_outline`: returns `(indices, palette_lab, outline_index | None,
/// outline_clipped)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn add_outline<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    palette_lab: &Bound<'py, PyAny>,
    outline: &str,
    darken: f64,
    allow_pad: bool,
    fixed_palette: bool,
) -> PyResult<Outlined<'py>> {
    const F: &str = "add_outline";
    let (idx, h, w) = index_image(F, indices)?;
    let pal = palette(F, palette_lab, &idx)?;
    let out = py
        .detach(|| {
            postprocess::add_outline(&idx, h, w, &pal, outline, darken, allow_pad, fixed_palette)
        })
        .map_err(PyValueError::new_err)?;
    Ok(outlined(py, out))
}

// ------------------------------------------------------------------------------- tiles

fn tilemap_from_py(func: &str, obj: &Bound<'_, PyAny>) -> PyResult<Tilemap> {
    let get = |key: &str| -> PyResult<i64> {
        obj.get_item(key)
            .and_then(|v| v.extract::<i64>())
            .map_err(|e| value_error(func, format!("tilemap[{key:?}]: {e}")))
    };
    let mut entries = Vec::new();
    for item in obj
        .get_item("tiles")
        .map_err(|e| value_error(func, format!("tilemap[\"tiles\"]: {e}")))?
        .try_iter()?
    {
        let item = item?;
        let entry: Vec<i64> = item
            .extract()
            .map_err(|e| value_error(func, format!("tilemap entry {item}: {e}")))?;
        let [id, fh, fv] = entry[..] else {
            return Err(value_error(
                func,
                format!("tilemap entry {item} is not [tile_id, flip_h, flip_v]"),
            ));
        };
        entries.push([id, fh, fv]);
    }
    Ok(Tilemap {
        tile_size: get("tile_size")?,
        width_tiles: get("width_tiles")?,
        height_tiles: get("height_tiles")?,
        tileset_columns: get("tileset_columns")?,
        tiles: entries,
    })
}

fn tilemap_to_py<'py>(py: Python<'py>, m: &Tilemap) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("tile_size", m.tile_size)?;
    d.set_item("width_tiles", m.width_tiles)?;
    d.set_item("height_tiles", m.height_tiles)?;
    d.set_item("tileset_columns", m.tileset_columns)?;
    let entries = PyList::empty(py);
    for e in &m.tiles {
        entries.append(PyList::new(py, e)?)?;
    }
    d.set_item("tiles", entries)?;
    Ok(d)
}

/// A tile stack int64 `(n, t, t)`: `(data, t)`.
fn tile_stack(func: &str, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<i64>, usize)> {
    let (shape, data) = array::<i64>(func, "tiles", "int64", obj, 3)?;
    if shape[1] != shape[2] || shape[1] == 0 {
        return Err(value_error(
            func,
            format!("tiles must have shape (n, t, t), got {shape:?}"),
        ));
    }
    Ok((data, shape[1]))
}

/// `tiles.extract`: returns `(tiles int64 (n, t, t), tilemap dict)`. `config` is
/// `Config.to_dict()`. A tile size that does not divide the image is a `ValueError` with the
/// Python message.
#[pyfunction]
fn tiles_extract<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    palette_lab: &Bound<'py, PyAny>,
    config: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyArray3<i64>>, Bound<'py, PyDict>)> {
    const F: &str = "tiles_extract";
    let (idx, h, w) = index_image(F, indices)?;
    let pal = palette(F, palette_lab, &idx)?;
    let cfg = TileConfig::from(&crate::config::build(py, config)?);
    let out = py
        .detach(|| tiles::extract(&idx, h, w, &pal, &cfg))
        .map_err(PyValueError::new_err)?;
    let t = out.tile_size;
    let stack = Array3::from_shape_vec((out.n_tiles, t, t), out.tiles)
        .expect("tile stack length matches its shape")
        .into_pyarray(py);
    Ok((stack, tilemap_to_py(py, &out.tilemap)?))
}

/// `tiles.reconstruct`: the index image int64 `(H, W)` a tileset and tilemap draw.
#[pyfunction]
fn tiles_reconstruct<'py>(
    py: Python<'py>,
    tiles: &Bound<'py, PyAny>,
    tilemap: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    const F: &str = "tiles_reconstruct";
    let (data, _) = tile_stack(F, tiles)?;
    let map = tilemap_from_py(F, tilemap)?;
    let (out, h, w) = py
        .detach(|| tiles::reconstruct(&data, &map))
        .map_err(PyValueError::new_err)?;
    Ok(image_out(py, out, h, w))
}

/// `Tiles.tileset_indices`: the tileset sheet, `columns` tiles wide, last row padded with -1.
#[pyfunction]
fn tileset_indices<'py>(
    py: Python<'py>,
    tiles: &Bound<'py, PyAny>,
    columns: i64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    const F: &str = "tileset_indices";
    let (data, t) = tile_stack(F, tiles)?;
    if columns <= 0 {
        return Err(value_error(
            F,
            format!("columns must be positive, got {columns}"),
        ));
    }
    let (out, h, w) = py.detach(|| tiles::tileset_indices(&data, t, columns as usize));
    Ok(image_out(py, out, h, w))
}

/// `tiles.tileset_columns`.
#[pyfunction]
fn tileset_columns(n_tiles: usize, requested: i64) -> i64 {
    tiles::tileset_columns(n_tiles, requested)
}

/// `tiles.gids`: Tiled global tile ids, row-major (firstgid 1), flip flags when `flips`.
#[pyfunction]
fn tilemap_gids(tilemap: &Bound<'_, PyAny>, flips: bool) -> PyResult<Vec<i64>> {
    Ok(tiles::gids(
        &tilemap_from_py("tilemap_gids", tilemap)?,
        flips,
    ))
}

/// `tiles.tiled_map`: a Tiled 1.10 JSON map as a dict (keys in the Python version's order).
#[pyfunction]
fn tiled_map<'py>(
    py: Python<'py>,
    tilemap: &Bound<'py, PyAny>,
    n_tiles: i64,
    image: &str,
    image_size: (i64, i64),
    name: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let map = tilemap_from_py("tiled_map", tilemap)?;
    let m = tiles::tiled_map(&map, n_tiles, image, image_size, name);
    let ts = &m.tilesets[0];
    let tileset = PyDict::new(py);
    tileset.set_item("firstgid", ts.firstgid)?;
    tileset.set_item("name", &ts.name)?;
    tileset.set_item("image", &ts.image)?;
    tileset.set_item("imagewidth", ts.imagewidth)?;
    tileset.set_item("imageheight", ts.imageheight)?;
    tileset.set_item("columns", ts.columns)?;
    tileset.set_item("tilecount", ts.tilecount)?;
    tileset.set_item("tilewidth", ts.tilewidth)?;
    tileset.set_item("tileheight", ts.tileheight)?;
    tileset.set_item("margin", ts.margin)?;
    tileset.set_item("spacing", ts.spacing)?;
    let l = &m.layers[0];
    let layer = PyDict::new(py);
    layer.set_item("id", l.id)?;
    layer.set_item("name", l.name)?;
    layer.set_item("type", l.kind)?;
    layer.set_item("x", l.x)?;
    layer.set_item("y", l.y)?;
    layer.set_item("width", l.width)?;
    layer.set_item("height", l.height)?;
    layer.set_item("opacity", l.opacity)?;
    layer.set_item("visible", l.visible)?;
    layer.set_item("data", &l.data)?;
    let out = PyDict::new(py);
    out.set_item("type", m.kind)?;
    out.set_item("version", m.version)?;
    out.set_item("orientation", m.orientation)?;
    out.set_item("renderorder", m.renderorder)?;
    out.set_item("infinite", m.infinite)?;
    out.set_item("width", m.width)?;
    out.set_item("height", m.height)?;
    out.set_item("tilewidth", m.tilewidth)?;
    out.set_item("tileheight", m.tileheight)?;
    out.set_item("nextlayerid", m.nextlayerid)?;
    out.set_item("nextobjectid", m.nextobjectid)?;
    out.set_item("layers", PyList::new(py, [layer])?)?;
    out.set_item("tilesets", PyList::new(py, [tileset])?)?;
    Ok(out)
}

/// `tiles.tilemap_csv`: one line per tile row of comma-separated GIDs (no flip flags).
#[pyfunction]
fn tilemap_csv(tilemap: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(tiles::tilemap_csv(&tilemap_from_py(
        "tilemap_csv",
        tilemap,
    )?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(postprocess_run, m)?)?;
    m.add_function(wrap_pyfunction!(remove_orphans, m)?)?;
    m.add_function(wrap_pyfunction!(fix_jaggies, m)?)?;
    m.add_function(wrap_pyfunction!(add_outline, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_extract, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_indices, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_columns, m)?)?;
    m.add_function(wrap_pyfunction!(tilemap_gids, m)?)?;
    m.add_function(wrap_pyfunction!(tiled_map, m)?)?;
    m.add_function(wrap_pyfunction!(tilemap_csv, m)?)?;
    Ok(())
}
