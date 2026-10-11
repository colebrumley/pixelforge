//! `pixelforge batch INPUT_DIR`: every image of a directory with one shared palette, one
//! scale and canvas, and (with `--enhance` / `--ink`) one pair of levels, so animation
//! frames stay consistent.

use std::path::{Path, PathBuf};

use clap::{Arg, ArgMatches, Command};

use crate::cli::{
    args, build_config, check_collisions, config_args, config_args_from, emit, output_args,
    outputs_json, paths, show_warnings, subcommand, warn, CliError,
};
use crate::color;
use crate::config::{Config, Value, PRESET_LONGEST_EDGE};
use crate::enhance;
use crate::io::{self, Loaded};
use crate::layout::outline_margin;
use crate::palette;
use crate::pipeline::{self, Input};
use crate::postprocess;
use crate::preprocess;
use crate::pyjson::Json;

/// The input file types `batch` picks up (suffix, any case).
pub const IMAGE_SUFFIXES: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff",
];
/// Pixels sampled per image for the shared palette and levels (a fixed stride, not random).
pub const BATCH_PIXELS_PER_IMAGE: usize = 50_000;
pub const SHARED_PALETTE_NAME: &str = "shared_palette.hex";
/// Stems of pixelforge's own image outputs, skipped as inputs.
pub const OUTPUT_STEM_SUFFIXES: &[&str] = &["_preview", "_tileset", "_compare"];

const DOC: &str = "Convert a directory's images with a shared palette.

Every image in INPUT_DIR is converted with ONE shared palette and ONE scale
(and, with --enhance or --ink, ONE pair of levels), so animation frames stay
consistent.";

pub fn command() -> Command {
    let cmd = subcommand("batch", DOC).arg(
        Arg::new("input_dir")
            .value_name("INPUT_DIR")
            .required(true)
            .value_parser(args::dir_path()),
    );
    config_args(output_args(cmd))
}

/// `max(1, ceil(n / BATCH_PIXELS_PER_IMAGE))`.
fn stride(n: usize) -> usize {
    n.div_ceil(BATCH_PIXELS_PER_IMAGE).max(1)
}

/// `io.load` with the CLI's budget, the load's warnings shown, and the alpha replaced by the
/// external `rembg` with `remove_bg` (as `preprocess._inputs` did).
fn load_subject(path: &Path, config: &Config, budget: u64) -> Result<(Loaded, Vec<u8>), CliError> {
    let loaded = io::load(path, budget)?;
    show_warnings(&loaded.warnings);
    let alpha = if config.remove_bg {
        preprocess::remove_background_external(&loaded.rgb, &loaded.alpha, loaded.h, loaded.w)?
    } else {
        loaded.alpha.clone()
    };
    Ok((loaded, alpha))
}

/// `shared_palette`: one palette (RGB8) from the union of every image's preprocessed LAB
/// pixels, each image subsampled with a fixed stride, finished as the pipeline finishes a
/// built palette (ramps, saturation).
pub fn shared_palette(
    paths: &[PathBuf],
    config: &Config,
    budget: u64,
) -> Result<Vec<[u8; 3]>, CliError> {
    let mut samples: Vec<[f64; 3]> = Vec::new();
    for path in paths {
        let (loaded, alpha) = load_subject(path, config, budget)?;
        let pre = preprocess::run(&loaded.rgb, &alpha, loaded.h, loaded.w, config)?;
        show_warnings(&pre.warnings);
        let pixels: Vec<[f64; 3]> = pre
            .lab
            .chunks_exact(3)
            .zip(&pre.mask)
            .filter(|(_, &m)| m)
            .map(|(c, _)| [c[0], c[1], c[2]])
            .collect();
        let step = stride(pixels.len());
        samples.extend(pixels.into_iter().step_by(step));
    }
    let mut lab = palette::build(&samples, config).map_err(|e| CliError::runtime(e.0))?;
    // A named palette is used as-is by the pipeline, so finish it here once.
    if config.palette_ramps {
        lab = palette::regularize_ramps(&lab);
    }
    let mut flat: Vec<f64> = lab.iter().flatten().copied().collect();
    postprocess::apply_saturation(&mut flat, config.saturation_beta);
    Ok(color::lab_to_rgb8(&flat)
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect())
}

/// `shared_scale`: `config` with one `scale` and `canvas` for every frame, unless they are
/// pinned (`None` then).
///
/// The union (largest width, largest height) of the frames' subject boxes is mapped to the
/// preset's longest edge (or to the one given out_width/out_height), outline margin inside,
/// so every frame keeps the same input-to-output ratio and all share one canvas.
pub fn shared_scale(
    paths: &[PathBuf],
    config: &Config,
    budget: u64,
) -> Result<Option<Config>, CliError> {
    let pinned = config.out_width.is_some() && config.out_height.is_some();
    if config.scale.is_some() || config.canvas.is_some() || pinned {
        return Ok(None);
    }
    let (mut width, mut height) = (0usize, 0usize);
    for path in paths {
        let (loaded, alpha) = load_subject(path, config, budget)?;
        let (y0, y1, x0, x1) =
            preprocess::subject_bbox(&loaded.rgb, &alpha, loaded.h, loaded.w, config)?;
        width = width.max(x1 - x0);
        height = height.max(y1 - y0);
    }
    let m2 = 2 * outline_margin(config);
    let divide = |a: usize, b: i64| -> Result<f64, CliError> {
        if b == 0 {
            return Err(CliError::runtime("division by zero"));
        }
        Ok(a as f64 / b as f64)
    };
    let scale = if let Some(w) = config.out_width {
        divide(width, w - m2)?
    } else if let Some(h) = config.out_height {
        divide(height, h - m2)?
    } else {
        let edge = PRESET_LONGEST_EDGE
            .iter()
            .find(|(p, _)| *p == config.preset)
            .map(|(_, e)| *e)
            .expect("a valid preset");
        divide(width.max(height), edge - m2)?
    };
    if scale == 0.0 {
        return Err(CliError::runtime("float division by zero"));
    }
    let mut canvas: Vec<i64> = [width, height]
        .iter()
        .map(|&v| ((v as f64 / scale).round_ties_even() as i64 + m2).max(8))
        .collect();
    if config.tileset {
        let tile = config.tile_size as f64;
        canvas = canvas
            .iter()
            .map(|&v| (v as f64 / tile).ceil() as i64 * config.tile_size)
            .collect();
    }
    let changes = vec![
        ("out_width".to_string(), Value::None),
        ("out_height".to_string(), Value::None),
        ("scale".to_string(), Value::Float(scale)),
        (
            "canvas".to_string(),
            Value::Str(format!("{}x{}", canvas[0], canvas[1])),
        ),
    ];
    Ok(Some(config.replace(&changes)?))
}

/// `shared_levels`: `config` with one `enhance_black` / `enhance_white` for every frame,
/// unless both are set or neither `enhance` nor `ink` is on (`None` then).
///
/// The pre-pass otherwise measures each frame's own brightness range, so a flash in one frame
/// would change the brightness of everything else in it.
pub fn shared_levels(
    paths: &[PathBuf],
    config: &Config,
    budget: u64,
) -> Result<Option<Config>, CliError> {
    if (config.enhance <= 0.0 && config.ink <= 0.0)
        || (config.enhance_black.is_some() && config.enhance_white.is_some())
    {
        return Ok(None);
    }
    let mut samples: Vec<f64> = Vec::new();
    for path in paths {
        let (loaded, alpha) = load_subject(path, config, budget)?;
        let value =
            preprocess::subject_brightness(&loaded.rgb, &alpha, loaded.h, loaded.w, config)?;
        let step = stride(value.len());
        samples.extend(value.into_iter().step_by(step));
    }
    let (black, white) = enhance::enhance_levels(&samples, config);
    if black.partial_cmp(&white) != Some(std::cmp::Ordering::Less) {
        // One flat colour: nothing to stretch in any frame.
        return Ok(None);
    }
    let changes = vec![
        ("enhance_black".to_string(), Value::Float(black)),
        ("enhance_white".to_string(), Value::Float(white)),
    ];
    Ok(Some(config.replace(&changes)?))
}

/// The images of `input_dir` in name order (warning about and skipping our own outputs).
fn input_images(input_dir: &Path) -> Result<Vec<PathBuf>, CliError> {
    let entries = std::fs::read_dir(input_dir).map_err(|e| CliError::os(e, input_dir))?;
    let mut names = Vec::new();
    for entry in entries {
        names.push(entry.map_err(|e| CliError::os(e, input_dir))?.file_name());
    }
    names.sort();
    let mut images = Vec::new();
    for name in names {
        let p = input_dir.join(name);
        let suffix = paths::suffix(&p).to_lowercase();
        if !(p.is_file() && IMAGE_SUFFIXES.contains(&suffix.as_str())) {
            continue;
        }
        let stem = paths::stem(&p);
        if OUTPUT_STEM_SUFFIXES.iter().any(|s| stem.ends_with(s)) {
            warn(&format!(
                "skipping {}: looks like a pixelforge output",
                paths::display(&p)
            ));
            continue;
        }
        images.push(p);
    }
    Ok(images)
}

pub fn run(matches: &ArgMatches) -> Result<(), CliError> {
    let input_dir: &PathBuf = matches.get_one("input_dir").expect("required");
    let outdir: &PathBuf = matches.get_one("outdir").expect("has a default");
    let force = matches.get_flag("force");
    let given = config_args_from(matches);
    let budget = given.max_input_pixels.unwrap_or(io::MAX_INPUT_PIXELS);
    let mut config = build_config(given.config_path.as_deref(), &given.flags)?;
    let images = input_images(input_dir)?;
    if images.is_empty() {
        return Err(CliError::runtime(format!(
            "no images found in {}",
            input_dir.display()
        )));
    }
    // {stem: inputs}, in first-seen order.
    let mut by_stem: Vec<(String, Vec<&PathBuf>)> = Vec::new();
    for p in &images {
        let stem = paths::stem(p);
        match by_stem.iter_mut().find(|(s, _)| *s == stem) {
            Some((_, group)) => group.push(p),
            None => by_stem.push((stem, vec![p])),
        }
    }
    let clashes: Vec<String> = by_stem
        .iter()
        .filter(|(_, group)| group.len() > 1)
        .map(|(_, group)| {
            group
                .iter()
                .map(|p| paths::display(p))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect();
    if !clashes.is_empty() && !force {
        return Err(CliError::config(format!(
            "inputs share an output name (later file would win): {} (rename them, or --force)",
            clashes.join("; ")
        )));
    }
    let palette_path = outdir.join(SHARED_PALETTE_NAME);
    let mut outputs = Vec::new();
    if config.palette_name.is_none() {
        outputs.push(palette_path.clone());
    }
    for (stem, _) in &by_stem {
        outputs.extend(
            pipeline::output_paths(&outdir.join(stem), config.tileset)
                .into_iter()
                .map(|(_, p)| p),
        );
    }
    check_collisions(&images, &outputs, force)?;
    std::fs::create_dir_all(outdir).map_err(|e| CliError::os(e, outdir))?;
    let mut header: Vec<(String, Json)> = Vec::new();
    // DEVIATION: Section 10 — one scale and canvas for the whole set (not per frame), so that
    // animation frames cropped to their own alpha keep a constant size.
    if let Some(scaled) = shared_scale(&images, &config, budget)? {
        config = scaled;
        header.push(("scale".into(), Json::Float(config.scale.expect("set"))));
        header.push((
            "canvas".into(),
            Json::Str(config.canvas.clone().expect("set")),
        ));
    }
    if let Some(leveled) = shared_levels(&images, &config, budget)? {
        config = leveled;
        header.push((
            "enhance_black".into(),
            Json::Float(config.enhance_black.expect("set")),
        ));
        header.push((
            "enhance_white".into(),
            Json::Float(config.enhance_white.expect("set")),
        ));
    }
    if config.palette_name.is_none() {
        let shared = shared_palette(&images, &config, budget)?;
        std::fs::write(&palette_path, palette::hex_text(&shared, false))
            .map_err(|e| CliError::os(e, &palette_path))?;
        // The colours themselves, not the path: frame metadata must not depend on --outdir.
        config = config.replace(&[(
            "palette_name".to_string(),
            Value::Str(palette::inline_name(&shared)),
        )])?;
        header.push((
            "shared_palette".into(),
            Json::Str(paths::display(&palette_path)),
        ));
    }
    if !header.is_empty() {
        emit(&Json::Object(header))?;
    }
    for path in &images {
        let result = pipeline::run(Input::Path(path), &config, Some(budget))?;
        show_warnings(&result.warnings);
        let saved = result.save(&outdir.join(paths::stem(path)))?;
        emit(&Json::Object(vec![
            ("input".into(), Json::Str(paths::display(path))),
            ("outputs".into(), outputs_json(&saved)),
            ("stats".into(), result.stats.clone()),
        ]))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_is_a_fixed_ceiling() {
        assert_eq!(stride(0), 1);
        assert_eq!(stride(1), 1);
        assert_eq!(stride(50_000), 1);
        assert_eq!(stride(50_001), 2);
        assert_eq!(stride(150_000), 3);
    }
}
