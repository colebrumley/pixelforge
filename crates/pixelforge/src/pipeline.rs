//! Pipeline orchestration (`pixelforge/pipeline.py`, Section 5): load, preprocess, downscale,
//! palette and quantization, post-processing, tiles, and the output files.
//!
//! [`run`] / [`run_loaded`] return an [`Output`] whose fields, `stats` (key order included)
//! and log lines are the Python version's; [`Output::save`] writes the same files byte for
//! byte (the PNGs decode to the same pixels, palette, transparency and text; their bytes
//! differ from Pillow's). Stage INFO lines and the keying warning go to the `log` target
//! `pixelforge::pipeline` (the binding forwards it to the Python logger
//! `pixelforge.pipeline`).

use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use crate::color;
use crate::config::{Config, ConfigError, Value};
use crate::downscale::{Downscaled, StatValue};
use crate::io::{self, IoError, Loaded, TransparentIndex};
use crate::mask::PixelforgeError;
use crate::palette::{self, PaletteError};
use crate::parallel;
use crate::postprocess::{self, PostConfig};
use crate::preprocess::{self, Preprocessed};
use crate::pyjson::{dumps, py_round, Json, Layout};
use crate::quantize;
use crate::tiles::{self, TileConfig, Tilemap};
use crate::{boxfilter, gerstner_run, kopf_run};

/// Precision of the LAB values in `_palette.json`.
pub const LAB_DECIMALS: usize = 6;

/// The package version recorded in `_meta.json` and hashed into `config_hash`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const TARGET: &str = "pixelforge::pipeline";

// ---------------------------------------------------------------------------------------
// Errors

/// A pipeline failure, with Python's class and message: `Config` (`ConfigError`, exit 2),
/// `Pixelforge` (`PixelforgeError`, exit 1), `Io` (an `OSError` on a path, or a
/// `PixelforgeError` from `io`), `Value` (an internal inconsistency Python reported as
/// `ValueError`/`IndexError`; not reachable from a valid Config).
#[derive(Debug)]
pub enum Error {
    Config(ConfigError),
    Pixelforge(PixelforgeError),
    Io(IoError),
    Value(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(e) => e.fmt(f),
            Error::Pixelforge(e) => e.fmt(f),
            Error::Io(e) => e.fmt(f),
            Error::Value(e) => f.write_str(e),
        }
    }
}

impl std::error::Error for Error {}

impl From<ConfigError> for Error {
    fn from(e: ConfigError) -> Self {
        Error::Config(e)
    }
}

impl From<PixelforgeError> for Error {
    fn from(e: PixelforgeError) -> Self {
        Error::Pixelforge(e)
    }
}

impl From<IoError> for Error {
    fn from(e: IoError) -> Self {
        match e {
            IoError::Pixelforge(e) => Error::Pixelforge(e),
            e => Error::Io(e),
        }
    }
}

impl From<preprocess::Error> for Error {
    fn from(e: preprocess::Error) -> Self {
        match e {
            preprocess::Error::Config(e) => Error::Config(e),
            preprocess::Error::Pixelforge(e) => Error::Pixelforge(e),
        }
    }
}

impl From<PaletteError> for Error {
    // The palette is validated (`validate_palette`) before any stage, so a palette error here
    // is the file changing under a run; Python raised it as ValueError from the loader.
    fn from(e: PaletteError) -> Self {
        Error::Config(ConfigError(e.0))
    }
}

fn os_error(error: std::io::Error, path: &Path) -> Error {
    Error::Io(IoError::Os {
        error,
        path: path.display().to_string(),
    })
}

// ---------------------------------------------------------------------------------------
// Inputs and outputs

/// What [`run`] converts: an image file, or an in-memory `(H, W, 3)` / `(H, W, 4)` array of
/// uint8 or float in `[0, 1]` (`io.from_image`; hashed as `rgba8 HxW\n` + RGBA bytes).
pub enum Input<'a> {
    Path(&'a Path),
    Array {
        shape: &'a [usize],
        data: io::ArrayData<'a>,
    },
}

/// An index image (`< 0` transparent) with its size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexImage {
    pub indices: Vec<i64>,
    pub h: usize,
    pub w: usize,
}

/// `pipeline.Result`. `palette_lab` is `(K, 3)` flattened, `palette` its RGB8 rendering.
#[derive(Clone, Debug)]
pub struct Output {
    /// `(H, W, 4)` RGBA8 at native pixel-art resolution (`indices` drawn with `palette`).
    pub image: Vec<u8>,
    pub palette: Vec<[u8; 3]>,
    pub indices: IndexImage,
    pub config: Config,
    pub input_sha256: String,
    pub palette_lab: Vec<f64>,
    /// The `stats` dict, keys in Python's insertion order.
    pub stats: Json,
    /// The tileset sheet as RGBA8 (`tileset_indices` drawn with `palette`).
    pub tileset: Option<Vec<u8>>,
    pub tilemap: Option<Tilemap>,
    pub tileset_indices: Option<IndexImage>,
    /// The texts Python emitted as `UserWarning` (multi-frame input, the stretch aspect).
    pub warnings: Vec<String>,
}

/// How [`Output::save_with`] fills the host-dependent parts of `_meta.json`.
#[derive(Clone, Debug, Default)]
pub struct MetaExtras {
    /// `environment`; `None` = [`environment()`].
    pub environment: Option<Json>,
    /// `remove_bg_versions` (written only with `remove_bg`); `None` = the external
    /// command's ([`preprocess::rembg_versions`]).
    pub remove_bg_versions: Option<Json>,
}

// ---------------------------------------------------------------------------------------
// Small helpers

/// `palette.resolve_source(config) == "mcda"` with gerstner and no named palette: gerstner's
/// own converged palette and indices are the final ones.
pub fn uses_gerstner_palette(config: &Config) -> bool {
    config.method == "gerstner"
        && config.palette_name.is_none()
        && palette::resolve_source(config) == "mcda"
}

/// `_describe_method`: `", 31 iterations, converged"` for the downscale log line (`""` for
/// box).
fn describe_method(stats: &[(&'static str, StatValue)]) -> String {
    let get = |name: &str| stats.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
    let truthy = |v: StatValue| match v {
        StatValue::Bool(b) => b,
        StatValue::Int(i) => i != 0,
        StatValue::Float(f) => f != 0.0,
    };
    let state = if let Some(v) = get("converged") {
        if truthy(v) {
            "converged"
        } else {
            "hit the iteration cap"
        }
    } else if let Some(v) = get("hit_iteration_cap") {
        if truthy(v) {
            "hit the iteration cap"
        } else {
            "converged"
        }
    } else {
        return String::new();
    };
    let iterations = match get("iterations") {
        Some(StatValue::Int(i)) => i.to_string(),
        Some(StatValue::Bool(b)) => (if b { "True" } else { "False" }).to_string(),
        Some(StatValue::Float(f)) => crate::pyrepr::float_repr(f),
        None => "?".to_string(),
    };
    format!(", {iterations} iterations, {state}")
}

fn stat_json(v: StatValue) -> Json {
    match v {
        StatValue::Int(i) => Json::Int(i),
        StatValue::Bool(b) => Json::Bool(b),
        StatValue::Float(f) => Json::Float(f),
    }
}

/// A Config value as JSON (`asdict` values: None, bool, int, float, str).
pub fn value_json(v: &Value) -> Json {
    match v {
        Value::None => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Int(i) => Json::Int(*i),
        Value::Float(f) => Json::Float(*f),
        Value::Str(s) => Json::Str(s.clone()),
        Value::List(items) => Json::List(items.iter().map(value_json).collect()),
        Value::BigInt(s) | Value::Other(s) => Json::Str(s.clone()),
    }
}

/// The tilemap dict (`out_tilemap.json`), keys in Python's order.
pub fn tilemap_json(m: &Tilemap) -> Json {
    Json::Object(vec![
        ("tile_size".into(), Json::Int(m.tile_size)),
        ("width_tiles".into(), Json::Int(m.width_tiles)),
        ("height_tiles".into(), Json::Int(m.height_tiles)),
        ("tileset_columns".into(), Json::Int(m.tileset_columns)),
        (
            "tiles".into(),
            Json::List(
                m.tiles
                    .iter()
                    .map(|e| Json::List(e.iter().map(|&v| Json::Int(v)).collect()))
                    .collect(),
            ),
        ),
    ])
}

/// A Tiled map (`tiles.tiled_map`) as the JSON dict Python built, keys in its order.
pub fn tiled_map_json(m: &tiles::TiledMap) -> Json {
    let s = |v: &str| Json::Str(v.to_string());
    let ts = &m.tilesets[0];
    let tileset = Json::Object(vec![
        ("firstgid".into(), Json::Int(ts.firstgid)),
        ("name".into(), s(&ts.name)),
        ("image".into(), s(&ts.image)),
        ("imagewidth".into(), Json::Int(ts.imagewidth)),
        ("imageheight".into(), Json::Int(ts.imageheight)),
        ("columns".into(), Json::Int(ts.columns)),
        ("tilecount".into(), Json::Int(ts.tilecount)),
        ("tilewidth".into(), Json::Int(ts.tilewidth)),
        ("tileheight".into(), Json::Int(ts.tileheight)),
        ("margin".into(), Json::Int(ts.margin)),
        ("spacing".into(), Json::Int(ts.spacing)),
    ]);
    let l = &m.layers[0];
    let layer = Json::Object(vec![
        ("id".into(), Json::Int(l.id)),
        ("name".into(), s(l.name)),
        ("type".into(), s(l.kind)),
        ("x".into(), Json::Int(l.x)),
        ("y".into(), Json::Int(l.y)),
        ("width".into(), Json::Int(l.width)),
        ("height".into(), Json::Int(l.height)),
        ("opacity".into(), Json::Int(l.opacity)),
        ("visible".into(), Json::Bool(l.visible)),
        (
            "data".into(),
            Json::List(l.data.iter().map(|&v| Json::Int(v)).collect()),
        ),
    ]);
    Json::Object(vec![
        ("type".into(), s(m.kind)),
        ("version".into(), s(m.version)),
        ("orientation".into(), s(m.orientation)),
        ("renderorder".into(), s(m.renderorder)),
        ("infinite".into(), Json::Bool(m.infinite)),
        ("width".into(), Json::Int(m.width)),
        ("height".into(), Json::Int(m.height)),
        ("tilewidth".into(), Json::Int(m.tilewidth)),
        ("tileheight".into(), Json::Int(m.tileheight)),
        ("nextlayerid".into(), Json::Int(m.nextlayerid)),
        ("nextobjectid".into(), Json::Int(m.nextobjectid)),
        ("layers".into(), Json::List(vec![layer])),
        ("tilesets".into(), Json::List(vec![tileset])),
    ])
}

fn lab_triples(flat: &[f64]) -> Vec<[f64; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn rgb8_triples(flat: &[u8]) -> Vec<[u8; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// `a[p:-p, p:-p]` of an `(h, w, c)` row-major array.
fn crop_border<T: Copy>(a: &[T], h: usize, w: usize, c: usize, p: usize) -> Vec<T> {
    let mut out = Vec::with_capacity((h - 2 * p) * (w - 2 * p) * c);
    for y in p..h - p {
        out.extend_from_slice(&a[(y * w + p) * c..(y * w + w - p) * c]);
    }
    out
}

/// `np.pad(idx, ((top, bottom), (left, right)), constant_values=-1)`.
fn pad_indices(idx: &IndexImage, pad: ((i64, i64), (i64, i64))) -> IndexImage {
    let ((top, bottom), (left, right)) = pad;
    let (top, bottom, left, right) = (top as usize, bottom as usize, left as usize, right as usize);
    let (h, w) = (idx.h + top + bottom, idx.w + left + right);
    let mut out = vec![-1i64; h * w];
    for y in 0..idx.h {
        let row = (y + top) * w + left;
        out[row..row + idx.w].copy_from_slice(&idx.indices[y * idx.w..(y + 1) * idx.w]);
    }
    IndexImage { indices: out, h, w }
}

/// `len(np.unique(indices[indices >= 0]))`.
fn colors_used(indices: &[i64]) -> usize {
    let mut seen: Vec<i64> = indices.iter().copied().filter(|&i| i >= 0).collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

// ---------------------------------------------------------------------------------------
// Running

/// `pipeline.run`: load (or convert) one image and run the pipeline on it. For paths
/// `input_sha256` hashes the file bytes; for arrays it hashes shape and RGBA bytes.
/// `max_pixels` defaults to [`io::MAX_INPUT_PIXELS`] and applies to both kinds. The load
/// time is the first entry of `stats.timings`.
pub fn run(input: Input<'_>, config: &Config, max_pixels: Option<u64>) -> Result<Output, Error> {
    let start = Instant::now();
    let budget = max_pixels.unwrap_or(io::MAX_INPUT_PIXELS);
    let loaded = match input {
        Input::Path(path) => io::load(path, budget)?,
        Input::Array { shape, data } => io::from_image_array(shape, data, budget)?,
    };
    let timings = vec![("load".to_string(), start.elapsed().as_secs_f64())];
    let mut out = run_loaded(&loaded, config, timings, false)?;
    let mut warnings = loaded.warnings.clone();
    warnings.append(&mut out.warnings);
    out.warnings = warnings;
    Ok(out)
}

/// Add `seconds` to the timing `name` (appended if new, so the insertion order is the order
/// of first use; an existing entry, e.g. the caller's matting time under "preprocess", is
/// added to).
fn record(timings: &mut Vec<(String, f64)>, name: &str, seconds: f64) {
    match timings.iter_mut().find(|(k, _)| k == name) {
        Some(entry) => entry.1 += seconds,
        None => timings.push((name.to_string(), seconds)),
    }
}

fn downscale(pre: &Preprocessed, config: &Config) -> Result<Downscaled, Error> {
    let (tw, th) = (pre.target_width() as usize, pre.target_height() as usize);
    let weight = Some(&pre.weight[..]);
    match config.method.as_str() {
        "box" => Ok(boxfilter::run(
            &pre.lab, &pre.mask, pre.h, pre.w, tw, th, weight,
        )?),
        "kopf" => Ok(kopf_run::run(
            &pre.lab,
            &pre.mask,
            pre.h,
            pre.w,
            tw,
            th,
            config,
            weight,
            parallel::thread_count(),
        )),
        "gerstner" => gerstner_run::run(&pre.lab, &pre.mask, pre.h, pre.w, tw, th, config, weight)
            .map_err(|e| Error::Pixelforge(PixelforgeError(e.0))),
        other => Err(Error::Value(format!("unknown method {other:?}"))),
    }
}

/// `pipeline.run_loaded`: convert an already loaded image. `timings` are the entries
/// recorded before (e.g. `load`). `alpha_matted`: the caller already replaced the alpha for
/// `remove_bg`; otherwise, with `remove_bg`, the external `rembg` command mats it here
/// ([`preprocess::remove_background_external`]).
pub fn run_loaded(
    loaded: &Loaded,
    config: &Config,
    timings: Vec<(String, f64)>,
    alpha_matted: bool,
) -> Result<Output, Error> {
    config.validate_palette()?; // before any expensive stage
    let mut timings = timings;
    let (h, w) = (loaded.h, loaded.w);

    let t = Instant::now();
    let matted;
    let alpha: &[u8] = if config.remove_bg && !alpha_matted {
        matted = preprocess::remove_background_external(&loaded.rgb, &loaded.alpha, h, w)?;
        &matted
    } else {
        &loaded.alpha
    };
    let pre = preprocess::run(&loaded.rgb, alpha, h, w, config)?;
    record(&mut timings, "preprocess", t.elapsed().as_secs_f64());
    let mut warnings = pre.warnings.clone();
    if config.key_bg
        && !config.remove_bg
        && !pre.background_keyed
        && loaded
            .alpha
            .iter()
            .all(|&a| i64::from(a) >= config.alpha_threshold)
    {
        log::warn!(
            target: TARGET,
            "no flat background to key out; the whole image is the subject \
             (try --remove-bg, or an input with an alpha channel)"
        );
    }
    log::info!(
        target: TARGET,
        "preprocess {:.2} s ({}x{} -> {}x{}, prereduce {})",
        seconds(&timings, "preprocess"),
        w,
        h,
        pre.w,
        pre.h,
        pre.prereduce_factor
    );

    let t = Instant::now();
    let mut small = downscale(&pre, config)?;
    let pad_out = pre.pad_out() as usize;
    if pad_out > 0 {
        // Seamless mode: drop the wrapped context again.
        let (sh, sw, p) = (small.h, small.w, pad_out);
        small.small_lab = crop_border(&small.small_lab, sh, sw, 3, p);
        small.small_mask = crop_border(&small.small_mask, sh, sw, 1, p);
        if let Some(indices) = &small.indices {
            small.indices = Some(crop_border(indices, sh, sw, 1, p));
            small.mean_lab = small
                .mean_lab
                .as_ref()
                .map(|m| crop_border(m, sh, sw, 3, p));
        }
        small.h = sh - 2 * p;
        small.w = sw - 2 * p;
    }
    record(&mut timings, "downscale", t.elapsed().as_secs_f64());
    log::info!(
        target: TARGET,
        "downscale {} {:.2} s{}",
        config.method,
        seconds(&timings, "downscale"),
        describe_method(&small.stats)
    );
    if !small.small_mask.iter().any(|&m| m) {
        return Err(PixelforgeError("no opaque pixels left after downscaling".into()).into());
    }

    let t = Instant::now();
    let (sh, sw) = (small.h, small.w);
    let own_palette = uses_gerstner_palette(config);
    let (mut palette_lab, indices): (Vec<f64>, Vec<i64>) = if own_palette {
        let palette_lab = small
            .palette_lab
            .clone()
            .ok_or_else(|| Error::Value("gerstner returned no palette".into()))?;
        let own = small
            .indices
            .as_ref()
            .ok_or_else(|| Error::Value("gerstner returned no indices".into()))?;
        let mut indices: Vec<i64> = own
            .iter()
            .zip(&small.small_mask)
            .map(|(&i, &m)| if m { i } else { -1 })
            .collect();
        if config.dither != "none" {
            // DEVIATION: Section 8.2 — small_lab is the palette color per pixel, so dithering it
            // rarely moves a pixel. Dither the smoothed superpixel means against the annealer's
            // colors before β instead; palette_lab (index-aligned) already carries β once.
            let mean = small
                .mean_lab
                .as_ref()
                .ok_or_else(|| Error::Value("gerstner returned no mean colors".into()))?;
            let unsaturated = small
                .palette_lab_unsaturated
                .as_ref()
                .ok_or_else(|| Error::Value("gerstner returned no palette".into()))?;
            indices = quantize::run(mean, &small.small_mask, unsaturated, sh, sw, config);
        }
        (palette_lab, indices)
    } else {
        // box / kopf, or gerstner with an external palette: quantize the (mean) colors.
        // DEVIATION: Section 7 — with palette_name or palette_source="median_cut", gerstner's
        // smoothed superpixel means are quantized against that palette, not its own.
        let source = small.mean_lab.as_ref().unwrap_or(&small.small_lab);
        let opaque: Vec<[f64; 3]> = source
            .chunks_exact(3)
            .zip(&small.small_mask)
            .filter(|(_, &m)| m)
            .map(|(c, _)| [c[0], c[1], c[2]])
            .collect();
        let palette_lab: Vec<f64> = palette::build(&opaque, config)?
            .into_iter()
            .flatten()
            .collect();
        let indices = quantize::run(source, &small.small_mask, &palette_lab, sh, sw, config);
        (palette_lab, indices)
    };
    record(&mut timings, "palette_quantize", t.elapsed().as_secs_f64());
    log::info!(
        target: TARGET,
        "palette/quantize {:.2} s, {} colors ({})",
        seconds(&timings, "palette_quantize"),
        palette_lab.len() / 3,
        if config.palette_name.is_some() {
            "named"
        } else {
            palette::resolve_source(config)
        }
    );

    let t = Instant::now();
    let mut idx = IndexImage {
        indices,
        h: sh,
        w: sw,
    };
    let pad = pre.canvas_pad();
    if [pad.0 .0, pad.0 .1, pad.1 .0, pad.1 .1]
        .iter()
        .any(|&p| p != 0)
    {
        // The outline ring is drawn inside the requested canvas: the image was downscaled to
        // the inner size and gets a transparent margin here, so the size never changes later.
        // With `canvas` the same padding also centers the fitted image on the canvas.
        idx = pad_indices(&idx, pad);
    }
    let fixed_palette = config.palette_name.is_some();
    // DEVIATION: Section 8.3 — ramps and saturation are skipped for a named palette
    // (palette_name): hardware colors are fixed by definition and stay exactly as loaded.
    // Ramps are applied here, before the other passes (postprocess::run does the rest).
    if !fixed_palette && config.palette_ramps {
        palette_lab = palette::regularize_ramps(&lab_triples(&palette_lab))
            .into_iter()
            .flatten()
            .collect();
    }
    let post = postprocess::run(
        &idx.indices,
        idx.h,
        idx.w,
        &palette_lab,
        &PostConfig::from(config),
        own_palette,
        fixed_palette,
        pre.outline_margin(),
    )
    .map_err(Error::Value)?;
    let palette_lab = post.palette_lab;
    let post_stats = post.stats;
    let mut idx = IndexImage {
        indices: post.indices,
        h: post.height,
        w: post.width,
    };
    record(&mut timings, "postprocess", t.elapsed().as_secs_f64());
    log::info!(
        target: TARGET,
        "postprocess {:.2} s ({}x{})",
        seconds(&timings, "postprocess"),
        idx.w,
        idx.h
    );

    let mut tile_result = None;
    if config.tileset {
        let t = Instant::now();
        let extracted = tiles::extract(
            &idx.indices,
            idx.h,
            idx.w,
            &palette_lab,
            &TileConfig::from(config),
        )
        .map_err(Error::Value)?;
        // DEVIATION: Section 9 — near-duplicate tiles were merged, so the output image is
        // re-rendered from the tileset and is exactly what the tilemap draws.
        let (rendered, rh, rw) =
            tiles::reconstruct(&extracted.tiles, &extracted.tilemap).map_err(Error::Value)?;
        let changed = rendered
            .iter()
            .zip(&idx.indices)
            .filter(|(a, b)| a != b)
            .count();
        idx = IndexImage {
            indices: rendered,
            h: rh,
            w: rw,
        };
        record(&mut timings, "tiles", t.elapsed().as_secs_f64());
        log::info!(
            target: TARGET,
            "tiles {:.2} s, {} tiles",
            seconds(&timings, "tiles"),
            extracted.n_tiles
        );
        tile_result = Some((extracted, changed));
    }

    let palette_rgb8 = rgb8_triples(&color::lab_to_rgb8(&palette_lab));
    let k = palette_rgb8.len();
    let used = colors_used(&idx.indices);
    let mut stats: Vec<(String, Json)> = vec![
        (
            "iterations".into(),
            Json::Int(
                small
                    .stats
                    .iter()
                    .find(|(name, _)| *name == "iterations")
                    .map(|(_, v)| match *v {
                        StatValue::Int(i) => i,
                        StatValue::Bool(b) => i64::from(b),
                        StatValue::Float(f) => f as i64,
                    })
                    .unwrap_or(0),
            ),
        ),
        ("final_palette_size".into(), Json::Int(k as i64)),
        ("colors_used".into(), Json::Int(used as i64)),
        ("colors_unused".into(), Json::Int(k as i64 - used as i64)),
        ("config_hash".into(), Json::Str(config.hash()?)),
        ("background_keyed".into(), Json::Bool(pre.background_keyed)),
        (
            "fit".into(),
            pre.fit.as_deref().map_or(Json::Null, Json::str),
        ),
        (
            "content_size".into(),
            Json::List(vec![
                Json::Int(pre.inner_width()),
                Json::Int(pre.inner_height()),
            ]),
        ),
        (
            "outline_index".into(),
            post_stats
                .outline_index
                .map_or(Json::Null, |i| Json::Int(i as i64)),
        ),
        ("outline_margin".into(), Json::Int(pre.outline_margin())),
        (
            "outline_clipped".into(),
            Json::Bool(post_stats.outline_clipped),
        ),
        (
            "prereduce_factor".into(),
            Json::Int(pre.prereduce_factor as i64),
        ),
        (
            "method".into(),
            Json::Object(
                small
                    .stats
                    .iter()
                    .map(|(name, v)| (name.to_string(), stat_json(*v)))
                    .collect(),
            ),
        ),
        (
            "timings".into(),
            Json::Object(
                timings
                    .iter()
                    .map(|(name, s)| (name.clone(), Json::Float(py_round(*s, 4))))
                    .collect(),
            ),
        ),
    ];
    let image = io::indices_to_rgba(&idx.indices, &palette_rgb8).map_err(Error::Value)?;
    let mut out = Output {
        image,
        palette: palette_rgb8,
        indices: idx,
        config: config.clone(),
        input_sha256: loaded.sha256.clone(),
        palette_lab,
        stats: Json::Null,
        tileset: None,
        tilemap: None,
        tileset_indices: None,
        warnings: std::mem::take(&mut warnings),
    };
    if let Some((extracted, changed)) = tile_result {
        let columns = usize::try_from(extracted.tilemap.tileset_columns)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| Error::Value("tileset_columns must be positive".into()))?;
        let (sheet, th, tw) =
            tiles::tileset_indices(&extracted.tiles, extracted.tile_size, columns);
        out.tileset = Some(io::indices_to_rgba(&sheet, &out.palette).map_err(Error::Value)?);
        out.tileset_indices = Some(IndexImage {
            indices: sheet,
            h: th,
            w: tw,
        });
        out.tilemap = Some(extracted.tilemap);
        stats.push(("tiles".into(), Json::Int(extracted.n_tiles as i64)));
        stats.push((
            "tiles_rerender_px_changed".into(),
            Json::Int(changed as i64),
        ));
    }
    out.stats = Json::Object(stats);
    Ok(out)
}

fn seconds(timings: &[(String, f64)], name: &str) -> f64 {
    timings
        .iter()
        .find(|(k, _)| k == name)
        .map_or(0.0, |(_, s)| *s)
}

// ---------------------------------------------------------------------------------------
// Environment and output paths

/// Python's `f"{sys.platform} {platform.machine()}"` for this build: `std::env::consts::OS`
/// mapped to `sys.platform` (macos → darwin, windows → win32, the others unchanged:
/// linux, freebsd, ...) and `ARCH` to `platform.machine()` (arm64 on macOS for aarch64;
/// AMD64 / ARM64 on Windows; unchanged elsewhere: x86_64, aarch64, ...).
pub fn platform() -> String {
    use std::env::consts::{ARCH, OS};
    let os = match OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    };
    let arch = match (OS, ARCH) {
        ("macos", "aarch64") => "arm64",
        ("windows", "x86_64") => "AMD64",
        ("windows", "aarch64") => "ARM64",
        (_, other) => other,
    };
    format!("{os} {arch}")
}

/// `pipeline.environment()` of the native library: `platform` (see [`platform`]), `threads`
/// (the worker threads, which affect speed only) and `native` (this crate's version).
/// Recorded in `_meta.json`, never hashed.
pub fn environment() -> Json {
    Json::Object(vec![
        ("platform".into(), Json::Str(platform())),
        ("threads".into(), Json::Int(parallel::thread_count() as i64)),
        ("native".into(), Json::str(VERSION)),
    ])
}

/// A path as Python's `pathlib.PurePosixPath` normalises it: repeated separators and `.`
/// components dropped, no trailing separator; empty means `.`.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `{name: path}` of every file [`Output::save`] writes, in write order (`image`,
/// `preview`, `palette`, `palette_hex`, `meta`, then with a tileset `tileset`, `tilemap`,
/// `tilemap_tmj`, `tilemap_csv`). Paths are spelled as Python's `pathlib` spells them.
pub fn output_paths(prefix: &Path, tileset: bool) -> Vec<(&'static str, PathBuf)> {
    let prefix = normalize(prefix);
    let name = prefix
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let parent = prefix.parent().map(Path::to_path_buf).unwrap_or_default();
    let sibling = |suffix: &str| parent.join(format!("{name}{suffix}"));
    let mut paths = vec![
        ("image", sibling(".png")),
        ("preview", sibling("_preview.png")),
        // DEVIATION: Section 5 — _palette.hex is a fifth output, not in the spec.
        ("palette", sibling("_palette.json")),
        ("palette_hex", sibling("_palette.hex")),
        ("meta", sibling("_meta.json")),
    ];
    if tileset {
        paths.push(("tileset", sibling("_tileset.png")));
        paths.push(("tilemap", sibling("_tilemap.json")));
        paths.push(("tilemap_tmj", sibling(".tmj")));
        paths.push(("tilemap_csv", sibling("_tilemap.csv")));
    }
    paths
}

// ---------------------------------------------------------------------------------------
// Saving

fn write_text(path: &Path, text: &str) -> Result<(), Error> {
    std::fs::write(path, text).map_err(|e| os_error(e, path))
}

impl Output {
    /// `Result.png_text()`: the PNG text chunks (config, input hash, and with a named palette
    /// its hash and lines).
    pub fn png_text(&self) -> Result<Vec<(String, String)>, Error> {
        let mut text = vec![
            (
                io::META_CONFIG_KEY.to_string(),
                self.config.canonical_json(),
            ),
            (io::META_INPUT_KEY.to_string(), self.input_sha256.clone()),
        ];
        let provenance = self.config.provenance()?;
        if let Some(lines) = provenance.palette {
            text.push((
                io::META_PALETTE_SHA_KEY.to_string(),
                provenance.palette_sha256.unwrap_or_default(),
            ));
            text.push((io::META_PALETTE_KEY.to_string(), lines.join("\n")));
        }
        Ok(text)
    }

    /// `Result.save(prefix)` with [`environment()`] and the external command's matting
    /// versions.
    pub fn save(&self, prefix: &Path) -> Result<Vec<(String, PathBuf)>, Error> {
        self.save_with(prefix, &MetaExtras::default())
    }

    /// `_palette.json`: `json.dumps(entries, indent=2) + "\n"`.
    pub fn palette_json(&self) -> String {
        let mut used = vec![false; self.palette.len()];
        for &i in &self.indices.indices {
            if let Some(u) = usize::try_from(i).ok().and_then(|i| used.get_mut(i)) {
                *u = true;
            }
        }
        let mut entries: Vec<Json> = Vec::with_capacity(self.palette.len() + 1);
        if self.config.transparent_index == "first" {
            entries.push(Json::Object(vec![
                ("hex".into(), Json::Null),
                ("transparent".into(), Json::Bool(true)),
                (
                    "used".into(),
                    Json::Bool(self.indices.indices.iter().any(|&i| i < 0)),
                ),
            ]));
        }
        for ((rgb, lab), u) in self
            .palette
            .iter()
            .zip(self.palette_lab.chunks_exact(3))
            .zip(&used)
        {
            // LAB_DECIMALS: float64 LAB can differ at ~1e-12 across SIMD paths; the PNG
            // does not. `+ 0.0` turns -0.0 into 0.0, as in Python.
            entries.push(Json::Object(vec![
                ("hex".into(), Json::Str(color::rgb8_to_hex(*rgb))),
                (
                    "lab".into(),
                    Json::List(
                        lab.iter()
                            .map(|&v| Json::Float(py_round(v, LAB_DECIMALS) + 0.0))
                            .collect(),
                    ),
                ),
                ("used".into(), Json::Bool(*u)),
            ]));
        }
        dumps(&Json::List(entries), Layout::INDENT2) + "\n"
    }

    /// `_meta.json`: `json.dumps(meta, indent=2, sort_keys=True) + "\n"`.
    pub fn meta_json(&self, extras: &MetaExtras) -> Result<String, Error> {
        let provenance = self.config.provenance()?;
        let mut meta = vec![
            ("version".to_string(), Json::str(VERSION)),
            (
                "config".to_string(),
                Json::Object(
                    provenance
                        .config
                        .iter()
                        .map(|(k, v)| (k.to_string(), value_json(v)))
                        .collect(),
                ),
            ),
            ("config_hash".to_string(), Json::Str(self.config.hash()?)),
            (
                "input_sha256".to_string(),
                Json::Str(self.input_sha256.clone()),
            ),
            (
                "palette_sha256".to_string(),
                provenance.palette_sha256.map_or(Json::Null, Json::Str),
            ),
            (
                "palette".to_string(),
                provenance.palette.map_or(Json::Null, |lines| {
                    Json::List(lines.into_iter().map(Json::Str).collect())
                }),
            ),
            ("width".to_string(), Json::Int(self.indices.w as i64)),
            ("height".to_string(), Json::Int(self.indices.h as i64)),
            ("stats".to_string(), self.stats.clone()),
            (
                "environment".to_string(),
                extras.environment.clone().unwrap_or_else(environment),
            ),
        ];
        if self.config.remove_bg {
            // The matting model is outside the determinism contract; record what ran.
            let versions = extras.remove_bg_versions.clone().unwrap_or_else(|| {
                Json::Object(
                    preprocess::rembg_versions()
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v.map_or(Json::Null, Json::Str)))
                        .collect(),
                )
            });
            meta.push(("remove_bg_versions".to_string(), versions));
        }
        Ok(dumps(&Json::Object(meta), Layout::INDENT2_SORTED) + "\n")
    }

    /// `stats["tiles"]` (the tile count the .tmj records).
    fn n_tiles(&self) -> Result<i64, Error> {
        if let Json::Object(items) = &self.stats {
            if let Some((_, Json::Int(n))) = items.iter().find(|(k, _)| k == "tiles") {
                return Ok(*n);
            }
        }
        Err(Error::Value("stats has no integer 'tiles' entry".into()))
    }

    /// `Result.save(prefix)`: write `<prefix>.png`, `_preview.png`, `_palette.json`,
    /// `_palette.hex`, `_meta.json` and, with a tilemap, `_tileset.png`, `_tilemap.json`,
    /// `.tmj` (Tiled) and `_tilemap.csv`, creating the parent directory. Returns
    /// `[(name, path)]` in write order. Reads `indices`, `palette`, `palette_lab`,
    /// `config`, `input_sha256`, `stats`, `tilemap` and `tileset_indices` (not the RGBA
    /// images).
    pub fn save_with(
        &self,
        prefix: &Path,
        extras: &MetaExtras,
    ) -> Result<Vec<(String, PathBuf)>, Error> {
        let paths = output_paths(prefix, self.tilemap.is_some());
        let path = |name: &str| -> &Path {
            &paths
                .iter()
                .find(|(n, _)| *n == name)
                .expect("output_paths has every name")
                .1
        };
        let parent = normalize(prefix).parent().map(Path::to_path_buf);
        if let Some(parent) = parent.filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(&parent).map_err(|e| os_error(e, &parent))?;
        }
        let text = self.png_text()?;
        let layout = if self.config.transparent_index == "first" {
            TransparentIndex::First
        } else {
            TransparentIndex::Last
        };
        let (idx, h, w) = (&self.indices.indices, self.indices.h, self.indices.w);
        io::save_png(path("image"), idx, h, w, &self.palette, &text, 1, layout)?;
        let scale = usize::try_from(self.config.scale_preview).unwrap_or(1);
        io::save_png(
            path("preview"),
            idx,
            h,
            w,
            &self.palette,
            &text,
            scale,
            layout,
        )?;
        write_text(path("palette"), &self.palette_json())?;
        write_text(
            path("palette_hex"),
            &palette::hex_text(&self.palette, layout == TransparentIndex::First),
        )?;
        write_text(path("meta"), &self.meta_json(extras)?)?;
        if let Some(tilemap) = &self.tilemap {
            // With "last", an opaque image has no transparent entry while a tileset whose
            // last row is padded gets one appended; the K colors keep their indices.
            let sheet = self
                .tileset_indices
                .as_ref()
                .ok_or_else(|| Error::Value("a tilemap without tileset_indices".into()))?;
            io::save_png(
                path("tileset"),
                &sheet.indices,
                sheet.h,
                sheet.w,
                &self.palette,
                &text,
                1,
                layout,
            )?;
            write_text(
                path("tilemap"),
                &(dumps(&tilemap_json(tilemap), Layout::COMPACT) + "\n"),
            )?;
            let image_name = path("tileset")
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let map_name = normalize(prefix)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let tmj = tiles::tiled_map(
                tilemap,
                self.n_tiles()?,
                &image_name,
                (sheet.w as i64, sheet.h as i64),
                &map_name,
            );
            write_text(
                path("tilemap_tmj"),
                &(dumps(&tiled_map_json(&tmj), Layout::INDENT1) + "\n"),
            )?;
            write_text(path("tilemap_csv"), &tiles::tilemap_csv(tilemap))?;
        }
        Ok(paths
            .into_iter()
            .map(|(name, p)| (name.to_string(), p))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_is_python_spelled() {
        let p = platform();
        assert!(!p.contains("macos"));
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(p, "darwin arm64");
        }
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            assert_eq!(p, "linux x86_64");
        }
    }

    #[test]
    fn output_paths_as_pathlib() {
        let names = |p: &str, t: bool| -> Vec<String> {
            output_paths(Path::new(p), t)
                .into_iter()
                .map(|(_, p)| p.display().to_string())
                .collect()
        };
        assert_eq!(
            names("out/hero", false),
            [
                "out/hero.png",
                "out/hero_preview.png",
                "out/hero_palette.json",
                "out/hero_palette.hex",
                "out/hero_meta.json"
            ]
        );
        assert_eq!(names("./hero", false)[0], "hero.png");
        assert_eq!(names("a//./b/", true)[8], "a/b_tilemap.csv");
        assert_eq!(names("/tmp/x", true)[7], "/tmp/x.tmj");
    }

    #[test]
    fn crop_and_pad() {
        let a: Vec<i64> = (0..16).collect();
        assert_eq!(crop_border(&a, 4, 4, 1, 1), vec![5, 6, 9, 10]);
        let img = IndexImage {
            indices: vec![1, 2],
            h: 1,
            w: 2,
        };
        let p = pad_indices(&img, ((1, 0), (0, 1)));
        assert_eq!((p.h, p.w), (2, 3));
        assert_eq!(p.indices, vec![-1, -1, -1, 1, 2, -1]);
    }

    #[test]
    fn describe() {
        assert_eq!(describe_method(&[]), "");
        assert_eq!(
            describe_method(&[
                ("iterations", StatValue::Int(31)),
                ("converged", StatValue::Bool(true))
            ]),
            ", 31 iterations, converged"
        );
        assert_eq!(
            describe_method(&[
                ("iterations", StatValue::Int(5)),
                ("hit_iteration_cap", StatValue::Bool(true))
            ]),
            ", 5 iterations, hit the iteration cap"
        );
    }
}
