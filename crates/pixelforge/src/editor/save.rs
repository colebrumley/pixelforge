//! Saving (docs/editor.md, Section 3.10): the document becomes a `pipeline::Output` and is
//! written by `Output::save_with`, so every file is what `convert` writes, plus the edit log
//! `<prefix>_edits.json`.
//!
//! The PNGs keep the `pixelforge:config` and `pixelforge:input_sha256` text chunks and gain
//! `pixelforge:edits_sha256`. `stats` keeps the conversion's entries with the palette counts
//! updated and gains an `edited` object. With `config.tileset` the tileset is re-extracted
//! from the edited image and the image re-rendered from it, as `convert` does; the pixels
//! that re-rendering changed are `edited.tiles_rerender_px_changed`.

use std::path::{Path, PathBuf};

use super::log::EDITS_SUFFIX;
use super::{Document, EditLog};
use crate::config::{Config, ConfigError};
use crate::io;
use crate::pipeline::{self, Error, IndexImage, MetaExtras, Output, VERSION};
use crate::pyjson::Json;
use crate::tiles::{self, TileConfig, Tiles};

/// `<prefix>_edits.json`.
pub fn edits_path(prefix: &Path) -> PathBuf {
    let name = prefix
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    prefix.with_file_name(format!("{name}{EDITS_SUFFIX}"))
}

/// Every file [`save`] writes: `pipeline::output_paths`, then `edits`.
pub fn output_paths(doc: &Document, prefix: &Path) -> Vec<(&'static str, PathBuf)> {
    let mut paths = pipeline::output_paths(prefix, doc.origin.config.tileset);
    let image = paths[0].1.clone();
    let stem = image.with_extension("");
    paths.push(("edits", edits_path(&stem)));
    paths
}

/// Replace `key` in a JSON object in place, or append it.
fn set(pairs: &mut Vec<(String, Json)>, key: &str, value: Json) {
    match pairs.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => pairs.push((key.to_string(), value)),
    }
}

/// With `config.tileset`: the tileset extracted from the document and the image it draws
/// (what a save writes, and the tiles panel shows); `None` without a tileset.
pub fn rerender(doc: &Document) -> Result<Option<(Tiles, IndexImage)>, Error> {
    rerender_indices(&doc.indices, &doc.palette_lab, &doc.origin.config)
}

/// [`rerender`] of an index image and LAB palette under `config` (what the editor's worker
/// thread runs on a snapshot).
pub fn rerender_indices(
    idx: &IndexImage,
    palette_lab: &[f64],
    config: &Config,
) -> Result<Option<(Tiles, IndexImage)>, Error> {
    if !config.tileset {
        return Ok(None);
    }
    let extracted = tiles::extract(
        &idx.indices,
        idx.h,
        idx.w,
        palette_lab,
        &TileConfig::from(config),
    )
    .map_err(Error::Value)?;
    let (rendered, h, w) =
        tiles::reconstruct(&extracted.tiles, &extracted.tilemap).map_err(Error::Value)?;
    Ok(Some((
        extracted,
        IndexImage {
            indices: rendered,
            h,
            w,
        },
    )))
}

/// The document as the `Output` `convert` would have produced, with `log`'s hash in
/// `stats.edited`.
pub fn to_output(doc: &Document, log: &EditLog) -> Result<Output, Error> {
    let config = &doc.origin.config;
    let k = doc.palette.len();
    let mut idx = doc.indices.clone();
    let mut tileset = None;
    let mut rerender_changed = None;
    if let Some((extracted, rendered)) = rerender(doc)? {
        rerender_changed = Some(
            rendered
                .indices
                .iter()
                .zip(&idx.indices)
                .filter(|(a, b)| a != b)
                .count(),
        );
        idx = rendered;
        tileset = Some(extracted);
    }

    let mut used = vec![false; k];
    for &i in &idx.indices {
        if i >= 0 {
            used[i as usize] = true;
        }
    }
    let n_used = used.iter().filter(|&&u| u).count() as i64;
    let mut stats = match &doc.origin.stats {
        Json::Object(pairs) => pairs.clone(),
        _ => Vec::new(),
    };
    set(&mut stats, "final_palette_size", Json::Int(k as i64));
    set(&mut stats, "colors_used", Json::Int(n_used));
    set(&mut stats, "colors_unused", Json::Int(k as i64 - n_used));
    let mut edited = vec![
        ("ops".to_string(), Json::Int(log.ops.len() as i64)),
        (
            "pixels_changed".to_string(),
            Json::Int(doc.pixels_changed() as i64),
        ),
        (
            "palette_changed".to_string(),
            Json::Bool(doc.palette_changed()),
        ),
        ("edits_sha256".to_string(), Json::Str(log.sha256())),
        ("pixelforge".to_string(), Json::str(VERSION)),
    ];
    if let Some(n) = rerender_changed {
        edited.push(("tiles_rerender_px_changed".into(), Json::Int(n as i64)));
    }

    let image = io::indices_to_rgba(&idx.indices, &doc.palette).map_err(Error::Value)?;
    let mut out = Output {
        image,
        palette: doc.palette.clone(),
        indices: idx,
        config: config.clone(),
        input_sha256: doc.origin.input_sha256.clone(),
        palette_lab: doc.palette_lab.clone(),
        stats: Json::Null,
        tileset: None,
        tilemap: None,
        tileset_indices: None,
        warnings: Vec::new(),
    };
    if let Some(extracted) = tileset {
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
        set(&mut stats, "tiles", Json::Int(extracted.n_tiles as i64));
    }
    set(&mut stats, "edited", Json::Object(edited));
    out.stats = Json::Object(stats);
    Ok(out)
}

/// `path` as the file system resolves it (a missing file through its parent).
fn resolved(path: &Path) -> PathBuf {
    if let Ok(p) = std::fs::canonicalize(path) {
        return p;
    }
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => {
            let parent = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            std::fs::canonicalize(parent)
                .map(|p| p.join(name))
                .unwrap_or_else(|_| path.to_path_buf())
        }
        _ => path.to_path_buf(),
    }
}

/// What [`save_with`] wrote: `[(name, path)]` in write order (ending with `edits`), and the
/// stats it recorded in `_meta.json`.
#[derive(Clone, Debug)]
pub struct Saved {
    pub outputs: Vec<(String, PathBuf)>,
    pub stats: Json,
}

/// [`save_with`] with the default `_meta.json` extras.
pub fn save(doc: &Document, log: &EditLog, prefix: &Path) -> Result<Saved, Error> {
    save_with(doc, log, prefix, &MetaExtras::default())
}

/// Write every output file of `doc` under `prefix` and the edit log `log` (normally
/// [`EditLog::of`] the document and its history). Refuses (a config error) when an output
/// would overwrite the conversion's input file (`origin.input_path`).
pub fn save_with(
    doc: &Document,
    log: &EditLog,
    prefix: &Path,
    extras: &MetaExtras,
) -> Result<Saved, Error> {
    let paths = output_paths(doc, prefix);
    if let Some(input) = doc.origin.input_path.as_deref().filter(|p| p.is_file()) {
        let input_key = resolved(input);
        if let Some((_, hit)) = paths.iter().find(|(_, p)| resolved(p) == input_key) {
            return Err(Error::Config(ConfigError(format!(
                "{} would overwrite input {}",
                hit.display(),
                input.display()
            ))));
        }
    }
    let out = to_output(doc, log)?;
    let mut extras = extras.clone();
    extras
        .png_text
        .push((io::META_EDITS_KEY.to_string(), log.sha256()));
    let mut outputs = out.save_with(prefix, &extras)?;
    let (_, edits) = paths.last().expect("output_paths ends with the log");
    log.write(edits).map_err(|e| {
        Error::Io(io::IoError::Os {
            error: e,
            path: edits.display().to_string(),
        })
    })?;
    outputs.push(("edits".to_string(), edits.clone()));
    Ok(Saved {
        outputs,
        stats: out.stats,
    })
}
