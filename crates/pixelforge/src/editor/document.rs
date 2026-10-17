//! The edited document (docs/editor.md, Section 3.2): an index image (`-1` transparent) and
//! its palette, plus where it came from.
//!
//! Built from a conversion in memory ([`Document::from_output`]), `palette_lab` is the
//! pipeline's own. Built from a PNG on disk ([`Document::open`]), the palette is the PNG's
//! `PLTE` (without the transparent slot `tRNS` marks) and `palette_lab` is `rgb8_to_lab` of
//! it; the config and input hash come from the `pixelforge:*` text chunks and the stats from
//! the sibling `_meta.json` when there is one. A PNG that is not a pixelforge output opens
//! with a default config when it has at most 256 colors (R3).

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::log::{EditLog, EDITS_SUFFIX};
use super::{err, EditError, Op};
use crate::color;
use crate::config::{Config, Value};
use crate::io;
use crate::pipeline::{self, IndexImage, Output};
use crate::preprocess;
use crate::pyjson::loads::{self, Json as Loaded};
use crate::pyjson::Json;

/// The input image for the reference panel, as RGBA8. `factor` is set when it is aligned
/// with the output (output pixel `(x, y)` covers the `factor × factor` block at
/// `(factor * x, factor * y)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRef {
    pub rgba: Vec<u8>,
    pub h: usize,
    pub w: usize,
    pub factor: Option<usize>,
}

impl SourceRef {
    /// An image file as an unaligned reference (`edit --source` when the geometry that would
    /// align it with the output cannot be recovered, see [`aligned`](Self::aligned)).
    pub fn load(path: &Path) -> Result<SourceRef, EditError> {
        let bytes = io::read_regular(path).map_err(|e| EditError(e.to_string()))?;
        let name = path.display().to_string();
        let image =
            io::decode_bytes(&name, &bytes, io::MAX_INPUT_PIXELS).map_err(|e| EditError(e.0))?;
        Ok(SourceRef {
            rgba: image.rgba,
            h: image.h,
            w: image.w,
            factor: None,
        })
    }

    /// The input image aligned with an output of `w × h` converted with `config`, for
    /// `edit --source` (docs/editor.md, Section 3.8): `preprocess::run` with the output's
    /// own config recovers the crop and canvas layout the conversion used, and
    /// `pipeline::source_reference` renders it as `convert --edit` does. `Ok(None)` when the
    /// geometry cannot be recovered: `remove_bg` (the external matting is not run again),
    /// a config the input does not preprocess with, or an output size other than `w × h`.
    pub fn aligned(
        path: &Path,
        config: &Config,
        w: usize,
        h: usize,
    ) -> Result<Option<SourceRef>, EditError> {
        SourceRef::aligned_within(path, config, w, h, io::MAX_INPUT_PIXELS)
    }

    /// [`aligned`](Self::aligned) with the input loaded under `budget` pixels, as the
    /// conversion loaded it (`batch` passes its `--max-pixels`).
    pub fn aligned_within(
        path: &Path,
        config: &Config,
        w: usize,
        h: usize,
        budget: u64,
    ) -> Result<Option<SourceRef>, EditError> {
        if config.remove_bg {
            return Ok(None);
        }
        let loaded = io::load(path, budget).map_err(|e| EditError(e.to_string()))?;
        let Ok(pre) = preprocess::run(&loaded.rgb, &loaded.alpha, loaded.h, loaded.w, config)
        else {
            return Ok(None);
        };
        let g = &pre.geometry;
        if (g.out_width, g.out_height) != (w as i64, h as i64) {
            return Ok(None);
        }
        let factor = pipeline::source_reference_factor(&pre, w);
        Ok(Some(pipeline::source_reference(&loaded, &pre, factor)))
    }
}

/// What a document was opened from.
#[derive(Clone, Debug)]
pub struct Origin {
    pub config: Config,
    pub input_sha256: String,
    /// The conversion's stats, carried into `_meta.json`.
    pub stats: Json,
    /// The input image for the reference panel.
    pub source: Option<SourceRef>,
    /// The conversion's input file, when known: saving refuses to write over it.
    pub input_path: Option<PathBuf>,
    /// The PNG the document was read from.
    pub opened_from: Option<PathBuf>,
    /// The edits the opened PNG already carries (its `NAME_edits.json`, when that log's hash
    /// is the PNG's `pixelforge:edits_sha256` and it leads from the conversion to the PNG).
    /// A saved log starts with them, so it still replays on the conversion.
    pub prior_ops: Vec<Op>,
    /// The opened PNG names an edit log that is missing or does not match: a saved log then
    /// holds this session's edits only and does not replay on the conversion.
    pub prior_log_missing: bool,
    /// `config` is the conversion's own (a conversion in memory, or a PNG with the
    /// `pixelforge:config` chunk), not the default of a plain PNG.
    pub converted: bool,
    /// The input `source` can be rebuilt from, and the pixel budget it was loaded with.
    /// `batch --edit` sets it so that only the current frame holds its source (see
    /// [`Origin::load_source`] and [`Origin::drop_source`]).
    pub source_input: Option<(PathBuf, u64)>,
}

impl Origin {
    /// An origin with `config`, the input hash and stats, and nothing else known.
    pub fn new(config: Config, input_sha256: String, stats: Json) -> Origin {
        Origin {
            config,
            input_sha256,
            stats,
            source: None,
            input_path: None,
            opened_from: None,
            prior_ops: Vec::new(),
            prior_log_missing: false,
            converted: false,
            source_input: None,
        }
    }

    /// Build the aligned `source` from `source_input` when it is not held, for an output of
    /// `w × h`. A source that cannot be rebuilt (the input moved, or it no longer aligns)
    /// stays `None`; the reference panel then shows nothing.
    pub fn load_source(&mut self, w: usize, h: usize) {
        if self.source.is_some() {
            return;
        }
        if let Some((path, budget)) = &self.source_input {
            self.source = SourceRef::aligned_within(path, &self.config, w, h, *budget)
                .ok()
                .flatten();
        }
    }

    /// Release `source` when it can be rebuilt from `source_input`.
    pub fn drop_source(&mut self) {
        if self.source_input.is_some() {
            self.source = None;
        }
    }
}

/// An index image and its palette.
#[derive(Clone, Debug)]
pub struct Document {
    /// `-1` transparent, else an index into `palette`.
    pub indices: IndexImage,
    /// `K * 3` CIELAB, the source of truth for colors.
    pub palette_lab: Vec<f64>,
    /// `lab_to_rgb8(palette_lab)` as loaded, then as edited: what the PNG stores.
    pub palette: Vec<[u8; 3]>,
    /// The named palette's full color list (`palette_name` set): entries take only these.
    pub fixed: Option<Vec<[u8; 3]>>,
    pub origin: Origin,
    /// The conversion result, for the diff overlay and `edited.pixels_changed`.
    pub baseline: IndexImage,
    /// The conversion's palette (the colors `baseline` indexes).
    pub baseline_palette: Vec<[u8; 3]>,
}

fn open_error(path: &Path, e: impl std::fmt::Display) -> EditError {
    EditError(format!("{}: {e}", path.display()))
}

/// `palette_name`'s colors, or `None` without one.
fn fixed_colors(config: &Config) -> Result<Option<Vec<[u8; 3]>>, EditError> {
    let lines = config.palette_hex_lines().map_err(|e| EditError(e.0))?;
    lines
        .map(|lines| {
            lines
                .iter()
                .map(|l| color::hex_to_rgb8(l).map_err(EditError))
                .collect()
        })
        .transpose()
}

/// `len(np.unique(indices[indices >= 0]))`.
fn colors_used(indices: &[i64]) -> usize {
    let mut seen: Vec<i64> = indices.iter().copied().filter(|&i| i >= 0).collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

/// The stats of a document without a `_meta.json`: the palette counts and the config hash.
pub fn minimal_stats(config: &Config, indices: &[i64], k: usize) -> Result<Json, EditError> {
    let used = colors_used(indices);
    Ok(Json::Object(vec![
        ("final_palette_size".into(), Json::Int(k as i64)),
        ("colors_used".into(), Json::Int(used as i64)),
        ("colors_unused".into(), Json::Int(k as i64 - used as i64)),
        (
            "config_hash".into(),
            Json::Str(config.hash().map_err(|e| EditError(e.0))?),
        ),
    ]))
}

/// A value read by the JSON reader, as the writer's value (integers beyond i64 as text).
pub(crate) fn to_writer_json(v: &Loaded) -> Json {
    match v {
        Loaded::Null => Json::Null,
        Loaded::Bool(b) => Json::Bool(*b),
        Loaded::Int(Value::Int(i)) => Json::Int(*i),
        Loaded::Int(other) => Json::Str(other.repr()),
        Loaded::Float(x) => Json::Float(*x),
        Loaded::Str(s) => Json::Str(s.clone()),
        Loaded::List(items) => Json::List(items.iter().map(to_writer_json).collect()),
        Loaded::Object(pairs) => Json::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.clone(), to_writer_json(v)))
                .collect(),
        ),
    }
}

/// `<dir>/<stem><suffix>` next to `path`.
pub(crate) fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{stem}{suffix}"))
}

/// The config of a `pixelforge:config` text chunk (its canonical JSON).
fn parse_config(text: &str) -> Result<Config, EditError> {
    let bad = |m: String| EditError(format!("pixelforge:config: {m}"));
    let parsed = loads::loads(text).map_err(|e| match e {
        loads::JsonError::Value(m) | loads::JsonError::Recursion(m) => bad(m),
    })?;
    let Loaded::Object(pairs) = parsed else {
        return Err(bad("not a JSON object".into()));
    };
    let values: Vec<(String, Value)> = pairs
        .iter()
        .map(|(k, v)| (k.clone(), v.to_value()))
        .collect();
    Config::new(&values).map_err(|e| bad(e.0))
}

/// An indexed PNG's samples, `PLTE` and `tRNS` (no transformation applied).
struct RawIndexed {
    indices: Vec<usize>,
    h: usize,
    w: usize,
    plte: Vec<[u8; 3]>,
    trns: Vec<u8>,
}

/// The raw samples of an indexed PNG, or `None` for another color type.
fn decode_indexed(bytes: &[u8]) -> Result<Option<RawIndexed>, String> {
    let mut decoder = png::Decoder::new_with_limits(
        std::io::Cursor::new(bytes),
        png::Limits { bytes: usize::MAX },
    );
    decoder.set_transformations(png::Transformations::IDENTITY);
    let header = decoder.read_header_info().map_err(|e| e.to_string())?;
    if header.color_type != png::ColorType::Indexed {
        return Ok(None);
    }
    let (w, h) = (header.width as usize, header.height as usize);
    io::check_budget("PNG", w as u64, h as u64, io::MAX_INPUT_PIXELS).map_err(|e| e.0)?;
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "image too large".to_string())?;
    let mut buf = vec![0u8; size];
    let frame = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let info = reader.info();
    let depth = info.bit_depth as usize;
    let plte: Vec<[u8; 3]> = info
        .palette
        .as_deref()
        .unwrap_or_default()
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let trns = info.trns.as_deref().unwrap_or_default().to_vec();
    let mask = (1usize << depth) - 1;
    let mut indices = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &buf[y * frame.line_size..(y + 1) * frame.line_size];
        for x in 0..w {
            let bit = x * depth;
            let shift = 8 - depth - bit % 8;
            indices.push((usize::from(row[bit / 8]) >> shift) & mask);
        }
    }
    Ok(Some(RawIndexed {
        indices,
        h,
        w,
        plte,
        trns,
    }))
}

/// A pixelforge PNG's indices and palette, as `Output::save` laid them out: with
/// `transparent_index="first"` entry 0 is the transparent slot; with `"last"` a transparent
/// slot is the last `tRNS` entry, when its alpha is 0.
fn pixelforge_layout(raw: RawIndexed, first: bool) -> Result<(IndexImage, Vec<[u8; 3]>), String> {
    let (clear, palette): (Option<usize>, Vec<[u8; 3]>) = if first {
        (Some(0), raw.plte.get(1..).unwrap_or_default().to_vec())
    } else {
        match raw.trns.last() {
            Some(0) if raw.trns.len() <= raw.plte.len() => {
                let c = raw.trns.len() - 1;
                (Some(c), raw.plte[..c].to_vec())
            }
            _ => (None, raw.plte.clone()),
        }
    };
    let k = palette.len();
    let mut indices = Vec::with_capacity(raw.indices.len());
    for &p in &raw.indices {
        let i = match clear {
            Some(c) if p == c => -1,
            _ if first => p as i64 - 1,
            _ => p as i64,
        };
        if i >= k as i64 {
            return Err(format!("pixel index {p} is outside the palette"));
        }
        indices.push(i);
    }
    Ok((
        IndexImage {
            indices,
            h: raw.h,
            w: raw.w,
        },
        palette,
    ))
}

/// Any image as indices and a palette of its distinct colors in order of first appearance
/// (row-major); alpha below `alpha_threshold` is transparent. At most 256 colors.
fn distinct_colors(
    rgba: &[u8],
    h: usize,
    w: usize,
    alpha_threshold: i64,
) -> Result<(IndexImage, Vec<[u8; 3]>), String> {
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut lookup: std::collections::HashMap<[u8; 3], usize> = Default::default();
    let mut indices = Vec::with_capacity(h * w);
    for px in rgba.chunks_exact(4) {
        if i64::from(px[3]) < alpha_threshold {
            indices.push(-1);
            continue;
        }
        let rgb = [px[0], px[1], px[2]];
        let i = *lookup.entry(rgb).or_insert_with(|| {
            palette.push(rgb);
            palette.len() - 1
        });
        if palette.len() > 256 {
            return Err("has more than 256 colors; the editor opens indexed images only".into());
        }
        indices.push(i as i64);
    }
    Ok((IndexImage { indices, h, w }, palette))
}

impl Document {
    /// A document of `indices` and `palette` (RGB8; LAB is `rgb8_to_lab` of it). Errors for
    /// a size mismatch, an index outside the palette or an unreadable named palette.
    pub fn new(
        indices: IndexImage,
        palette: Vec<[u8; 3]>,
        origin: Origin,
    ) -> Result<Document, EditError> {
        let flat: Vec<u8> = palette.iter().flatten().copied().collect();
        let lab = color::rgb8_to_lab(&flat);
        Document::with_lab(indices, palette, lab, origin)
    }

    fn with_lab(
        indices: IndexImage,
        palette: Vec<[u8; 3]>,
        palette_lab: Vec<f64>,
        origin: Origin,
    ) -> Result<Document, EditError> {
        if indices.indices.len() != indices.h * indices.w {
            return err("the index image does not match its size");
        }
        if palette_lab.len() != palette.len() * 3 {
            return err("palette and LAB palette differ in length");
        }
        let k = palette.len() as i64;
        if let Some(i) = indices.indices.iter().find(|&&i| i < -1 || i >= k) {
            return err(format!("index {i} is outside the palette of {k} entries"));
        }
        let fixed = fixed_colors(&origin.config)?;
        Ok(Document {
            baseline: indices.clone(),
            baseline_palette: palette.clone(),
            indices,
            palette_lab,
            palette,
            fixed,
            origin,
        })
    }

    /// The document of a conversion in memory (`convert --edit`, `convert --edits`).
    pub fn from_output(out: &Output) -> Result<Document, EditError> {
        let mut origin = Origin::new(
            out.config.clone(),
            out.input_sha256.clone(),
            out.stats.clone(),
        );
        origin.converted = true;
        Document::with_lab(
            out.indices.clone(),
            out.palette.clone(),
            out.palette_lab.clone(),
            origin,
        )
    }

    /// Open a PNG: a pixelforge output (palette from `PLTE`/`tRNS`, config and input hash
    /// from the text chunks, stats from `<stem>_meta.json`, earlier edits from
    /// `<stem>_edits.json`), or any PNG with at most 256 colors (default config, the file's
    /// hash as input hash).
    pub fn open(path: &Path) -> Result<Document, EditError> {
        let bytes = io::read_regular(path).map_err(|e| EditError(e.to_string()))?;
        if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Err(open_error(path, "not a PNG file"));
        }
        let text = io::read_png_text(path).map_err(|e| EditError(e.to_string()))?;
        let chunk = |key: &str| {
            text.iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        };
        let converted = chunk(io::META_CONFIG_KEY).is_some();
        let (config, input_sha256, (indices, palette)) = match chunk(io::META_CONFIG_KEY) {
            Some(json) => {
                let config = parse_config(&json).map_err(|e| open_error(path, e))?;
                if let Some(lines) = chunk(io::META_PALETTE_KEY) {
                    // The palette as the conversion read it; the file may have changed or gone.
                    config.seed_palette_hex_lines(lines.split('\n').map(str::to_string).collect());
                }
                let first = config.transparent_index == "first";
                let image = match decode_indexed(&bytes).map_err(|e| open_error(path, e))? {
                    Some(raw) => pixelforge_layout(raw, first),
                    None => {
                        let rgba = io::decode_bytes("PNG", &bytes, io::MAX_INPUT_PIXELS)
                            .map_err(|e| open_error(path, e.0))?;
                        distinct_colors(&rgba.rgba, rgba.h, rgba.w, 1)
                    }
                }
                .map_err(|e| open_error(path, e))?;
                (config, chunk(io::META_INPUT_KEY).unwrap_or_default(), image)
            }
            None => {
                let config = Config::new(&[]).map_err(|e| EditError(e.0))?;
                let rgba = io::decode_bytes("PNG", &bytes, io::MAX_INPUT_PIXELS)
                    .map_err(|e| open_error(path, e.0))?;
                let image = distinct_colors(&rgba.rgba, rgba.h, rgba.w, config.alpha_threshold)
                    .map_err(|e| open_error(path, e))?;
                let sha = format!("{:x}", Sha256::digest(&bytes));
                (config, sha, image)
            }
        };
        let stats = match read_meta_stats(&sibling(path, "_meta.json")) {
            Some(stats) => stats,
            None => minimal_stats(&config, &indices.indices, palette.len())?,
        };
        let mut origin = Origin::new(config, input_sha256, stats);
        origin.opened_from = Some(path.to_path_buf());
        origin.converted = converted;
        let mut doc = Document::new(indices, palette, origin)?;
        if let Some(sha) = chunk(io::META_EDITS_KEY) {
            doc.attach_prior_log(&sibling(path, EDITS_SUFFIX), &sha);
        }
        Ok(doc)
    }

    /// Take the edits of `log_path` as this document's history before opening: the
    /// baseline becomes the conversion they were made on. Marks the log missing when it is
    /// absent, its hash is not `sha`, or it does not lead to this document.
    ///
    /// With a tileset the PNG is the image re-rendered from the tiles, which can differ from
    /// the edited pixels; the log is accepted when its result re-renders to the PNG, and the
    /// document becomes that result (the pixels the user edited, as before the save).
    fn attach_prior_log(&mut self, log_path: &Path, sha: &str) {
        let usable = EditLog::read(log_path).ok().filter(|log| {
            log.sha256() == sha && log.width == self.width() && log.height == self.height()
        });
        let Some(log) = usable else {
            self.origin.prior_log_missing = true;
            return;
        };
        // Undo the logged ops (each carries its old values), then check that replaying
        // them leads back here.
        let mut base = self.clone();
        let mut ops = log.ops.clone();
        for op in ops.iter_mut().rev() {
            if !base.can_revert(op) {
                self.origin.prior_log_missing = true;
                return;
            }
            base.fill_undo_lab(op);
            base.revert(op);
        }
        let start_palette = base.palette.clone();
        let start = base.indices.clone();
        // The old values come from the file: a baseline index outside the palette would
        // break the diff and the save, so such a log is treated as missing.
        let k0 = start_palette.len() as i64;
        if start.indices.iter().any(|&i| i < -1 || i >= k0) {
            self.origin.prior_log_missing = true;
            return;
        }
        let replayed = base.palette.len() == log.palette_len
            && log.ops.iter().all(|op| base.replay(op).is_ok())
            && base.palette == self.palette;
        let leads_here = replayed
            && (base.indices == self.indices
                || matches!(
                    super::save::rerender(&base),
                    Ok(Some((_, rendered))) if rendered == self.indices
                ));
        if !leads_here {
            self.origin.prior_log_missing = true;
            return;
        }
        self.indices = base.indices;
        self.baseline = start;
        self.baseline_palette = start_palette;
        self.origin.prior_ops = log.ops;
    }

    pub fn width(&self) -> usize {
        self.indices.w
    }

    pub fn height(&self) -> usize {
        self.indices.h
    }

    /// The index at `(x, y)` (`-1` transparent); panics outside the image.
    pub fn get(&self, x: usize, y: usize) -> i64 {
        assert!(
            x < self.indices.w && y < self.indices.h,
            "({x}, {y}) is outside the image"
        );
        self.indices.indices[y * self.indices.w + x]
    }

    /// The index at a pointer position, or `None` outside the image.
    pub fn get_at(&self, x: i64, y: i64) -> Option<i64> {
        let (x, y) = (usize::try_from(x).ok()?, usize::try_from(y).ok()?);
        (x < self.indices.w && y < self.indices.h).then(|| self.get(x, y))
    }

    /// Pixels per palette entry.
    pub fn counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.palette.len()];
        for &i in &self.indices.indices {
            if i >= 0 {
                counts[i as usize] += 1;
            }
        }
        counts
    }

    /// The image as RGBA8 (transparent pixels `0, 0, 0, 0`).
    pub fn rgba(&self) -> Vec<u8> {
        io::indices_to_rgba(&self.indices.indices, &self.palette).expect("indices are checked")
    }

    /// Whether pixel `(x, y)`'s color or transparency differs from the conversion result.
    pub fn differs_at(&self, x: usize, y: usize) -> bool {
        let p = y * self.indices.w + x;
        self.differs(self.indices.indices[p], self.baseline.indices[p])
    }

    fn differs(&self, now: i64, base: i64) -> bool {
        let color = |i: i64, palette: &[[u8; 3]]| (i >= 0).then(|| palette[i as usize]);
        color(now, &self.palette) != color(base, &self.baseline_palette)
    }

    /// Per pixel: its color or transparency differs from the conversion result.
    pub fn diff_mask(&self) -> Vec<bool> {
        self.indices
            .indices
            .iter()
            .zip(&self.baseline.indices)
            .map(|(&a, &b)| self.differs(a, b))
            .collect()
    }

    /// How many pixels differ from the conversion result ([`diff_mask`](Self::diff_mask)).
    pub fn pixels_changed(&self) -> usize {
        self.diff_mask().iter().filter(|&&d| d).count()
    }

    /// Whether the palette differs from the conversion's.
    pub fn palette_changed(&self) -> bool {
        self.palette != self.baseline_palette
    }

    /// Whether reverting `op` is possible here (the shape checks only; ops a log replayed
    /// are always revertible in reverse order).
    fn can_revert(&self, op: &Op) -> bool {
        let (w, h, k) = (self.indices.w, self.indices.h, self.palette.len());
        let inside = |c: &super::PixelChange| c.x < w && c.y < h;
        match op {
            Op::Pixels { changes, .. } => changes.iter().all(inside),
            Op::PaletteSet { index, .. } => *index < k,
            Op::PaletteAdd { .. } => k > 0,
            Op::PaletteRemove { index, remap, .. } => *index <= k && remap.iter().all(inside),
        }
    }

    /// LAB undo data for an op read from a log: `rgb8_to_lab` of the recorded colors.
    fn fill_undo_lab(&self, op: &mut Op) {
        let lab = |rgb: [u8; 3]| {
            let l = color::rgb8_to_lab(&rgb);
            [l[0], l[1], l[2]]
        };
        match op {
            Op::PaletteSet { old, old_lab, .. } => *old_lab = lab(*old),
            Op::PaletteRemove { rgb, lab: l, .. } => *l = lab(*rgb),
            _ => {}
        }
    }
}

/// `stats` of a `_meta.json`, when the file exists and has a `stats` object.
fn read_meta_stats(path: &Path) -> Option<Json> {
    let text = std::fs::read_to_string(path).ok()?;
    match loads::loads(&text).ok()? {
        Loaded::Object(pairs) => pairs
            .iter()
            .rev()
            .find(|(k, _)| k == "stats")
            .filter(|(_, v)| matches!(v, Loaded::Object(_)))
            .map(|(_, v)| to_writer_json(v)),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A document from rows of digits (`.` transparent) and a palette, default config.
    pub(crate) fn doc(rows: &[&str], palette: &[[u8; 3]]) -> Document {
        let h = rows.len();
        let w = rows[0].len();
        let indices = rows
            .iter()
            .flat_map(|r| {
                r.bytes().map(|b| match b {
                    b'.' => -1,
                    d => i64::from(d - b'0'),
                })
            })
            .collect();
        let config = Config::new(&[]).unwrap();
        let origin = Origin::new(config, String::new(), Json::Object(vec![]));
        Document::new(IndexImage { indices, h, w }, palette.to_vec(), origin).unwrap()
    }

    /// The rows of digits of a document (inverse of [`doc`]).
    pub(crate) fn rows(d: &Document) -> Vec<String> {
        d.indices
            .indices
            .chunks(d.width())
            .map(|r| {
                r.iter()
                    .map(|&i| {
                        if i < 0 {
                            '.'
                        } else {
                            char::from(b'0' + i as u8)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn indices_must_fit_the_palette() {
        let config = Config::new(&[]).unwrap();
        let origin = Origin::new(config, String::new(), Json::Null);
        let img = IndexImage {
            indices: vec![0, 2],
            h: 1,
            w: 2,
        };
        assert!(Document::new(img, vec![[0; 3], [1; 3]], origin).is_err());
    }

    #[test]
    fn distinct_colors_in_order_of_appearance() {
        let rgba = [9, 9, 9, 255, 1, 1, 1, 255, 0, 0, 0, 0, 9, 9, 9, 200];
        let (img, palette) = distinct_colors(&rgba, 2, 2, 128).unwrap();
        assert_eq!(img.indices, vec![0, 1, -1, 0]);
        assert_eq!(palette, vec![[9, 9, 9], [1, 1, 1]]);
    }

    #[test]
    fn diff_mask_compares_colors_not_indices() {
        let mut d = doc(&["01"], &[[0, 0, 0], [255, 255, 255]]);
        let mut op = Op::PaletteRemove {
            index: 0,
            rgb: [0; 3],
            lab: [0.0; 3],
            remap: vec![super::super::PixelChange {
                x: 0,
                y: 0,
                old: 0,
                new: 1,
            }],
        };
        d.apply(&mut op).unwrap();
        assert_eq!(d.indices.indices, vec![0, 0]);
        assert_eq!(d.diff_mask(), vec![true, false]);
        assert!(d.differs_at(0, 0) && !d.differs_at(1, 0));
        assert!(d.palette_changed());
    }
}
