//! `pixelforge compare INPUT`: box, kopf and gerstner side by side, or (with `--prepass` /
//! `--sweep`) a labelled proof sheet of every method at several settings.

use std::path::{Path, PathBuf};

use clap::{Arg, ArgAction, ArgMatches, Command};

use crate::cli::{
    args, build_config, check_collisions, config_args, config_args_from, emit, logging,
    output_args, outputs_json, paths, show_warnings, subcommand, warn, CliError, ConfigArgs,
};
use crate::config::{ConfigError, Value, METHODS};
use crate::io;
use crate::pipeline::{self, value_json, Input};
use crate::pyjson::Json;
use crate::sheet::{
    override_flags, parse_sweep, prepass_columns, preview_scale, proof_sheet, show, side_by_side,
    RgbaImage,
};

const DOC: &str = "Compare box, kopf and gerstner side by side.

Runs all three methods with the same config and writes INPUT_compare.png, plus
each method's own outputs.

With --prepass or --sweep it writes only a labelled proof sheet instead: one
row per method (only the given one with --method), one column per setting. The
JSON result lists, per cell, the flags to add to a `convert` call with the same
base options, and the cell's full config (usable on its own with --config).";

pub fn command() -> Command {
    let cmd = subcommand("compare", DOC).arg(
        Arg::new("input")
            .value_name("INPUT")
            .required(true)
            .value_parser(args::file_path()),
    );
    let cmd = output_args(cmd)
        .arg(
            Arg::new("prepass")
                .long("prepass")
                .action(ArgAction::SetTrue)
                .help("Proof sheet: every method at four --enhance/--ink settings."),
        )
        .arg(
            Arg::new("sweep")
                .long("sweep")
                .value_name("FIELD=V1,V2,...")
                .action(ArgAction::Append)
                .allow_hyphen_values(true)
                .help("Proof sheet: every method at each value of one option, e.g. palette-size=8,12,16."),
        );
    config_args(cmd)
}

pub fn run(matches: &ArgMatches) -> Result<(), CliError> {
    let input: &PathBuf = matches.get_one("input").expect("required");
    let outdir: &PathBuf = matches.get_one("outdir").expect("has a default");
    let force = matches.get_flag("force");
    let prepass = matches.get_flag("prepass");
    let sweeps: Vec<&String> = matches
        .get_many::<String>("sweep")
        .map(|v| v.collect())
        .unwrap_or_default();
    let given = config_args_from(matches);
    if sweeps.len() > 1 {
        return Err(ConfigError(
            "--sweep can be given once: a sheet has one row per method and one column per value"
                .into(),
        )
        .into());
    }
    let sweep = sweeps.first().copied();
    if prepass && sweep.is_some() {
        return Err(ConfigError("--prepass and --sweep cannot be combined".into()).into());
    }
    let stem = paths::stem(input);
    let sheet_path = outdir.join(format!("{stem}_compare.png"));
    if prepass || sweep.is_some() {
        let columns = match sweep {
            Some(text) => parse_sweep(text)?,
            None => prepass_columns(),
        };
        return proof(input, outdir, &sheet_path, &columns, &given, force);
    }
    methods_side_by_side(input, outdir, &stem, &sheet_path, given, force)
}

/// A result's image (native resolution) as an [`RgbaImage`].
fn image_of(result: &pipeline::Output) -> RgbaImage {
    RgbaImage::new(result.indices.h, result.indices.w, result.image.clone())
}

/// `io.upscale_nearest(im, factor)`.
fn upscale(im: &RgbaImage, factor: i64) -> RgbaImage {
    let factor = usize::try_from(factor).unwrap_or(1).max(1);
    RgbaImage::new(
        im.h * factor,
        im.w * factor,
        io::upscale_nearest(&im.data, im.h, im.w, 4, factor),
    )
}

/// `Image.fromarray(sheet, mode="RGBA").save(path)`.
fn save_sheet(path: &Path, sheet: &RgbaImage) -> Result<(), CliError> {
    io::save_rgba_png(path, &sheet.data, sheet.h, sheet.w)?;
    Ok(())
}

/// The proof sheet: one row per method (only `--method` when given), one column per entry
/// of `columns`; only the sheet is written, the JSON lists every cell.
fn proof(
    input: &Path,
    outdir: &Path,
    sheet_path: &Path,
    columns: &[Vec<(String, Value)>],
    given: &ConfigArgs,
    force: bool,
) -> Result<(), CliError> {
    let methods: Vec<String> = match given.flags.iter().find(|(k, _)| k == "method") {
        Some((_, Value::Str(m))) => vec![m.clone()],
        _ => METHODS.iter().map(|m| m.to_string()).collect(),
    };
    let base = build_config(given.config_path.as_deref(), &given.flags)?;
    check_collisions(&[input.to_path_buf()], &[sheet_path.to_path_buf()], force)?;
    // Every cell is validated before the first (slow) one runs.
    let with_method = |method: &str, column: &[(String, Value)]| {
        let mut changes = vec![("method".to_string(), Value::Str(method.to_string()))];
        changes.extend(column.iter().cloned());
        changes
    };
    let mut configs = Vec::with_capacity(methods.len());
    for method in &methods {
        let row = columns
            .iter()
            .map(|column| base.replace(&with_method(method, column)))
            .collect::<Result<Vec<_>, _>>()?;
        configs.push(row);
    }
    for config in &configs[0] {
        config.validate_palette()?;
    }
    let budget = given.max_input_pixels.unwrap_or(io::MAX_INPUT_PIXELS);
    let loaded = io::load(input, budget)?;
    show_warnings(&loaded.warnings);
    let labels: Vec<String> = columns
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|(name, value)| format!("{name}={}", show(value)))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let mut images = Vec::with_capacity(methods.len() * columns.len());
    let mut report = Vec::with_capacity(methods.len() * columns.len());
    for (method, row) in methods.iter().zip(&configs) {
        for ((column, label), config) in columns.iter().zip(&labels).zip(row) {
            log::info!(target: logging::TARGET, "cell {method} / {label}");
            let result = pipeline::run_loaded(&loaded, config, Vec::new(), false)?;
            show_warnings(&result.warnings);
            // Native size: upscaled once the total is known.
            images.push(image_of(&result));
            let config_json = config
                .to_dict()
                .iter()
                .map(|(k, v)| (k.to_string(), value_json(v)))
                .collect();
            report.push(Json::Object(vec![
                ("method".into(), Json::Str(method.clone())),
                ("column".into(), Json::Str(label.clone())),
                ("stats".into(), result.stats.clone()),
                (
                    "flags".into(),
                    Json::Str(override_flags(&with_method(method, column))),
                ),
                ("config".into(), Json::Object(config_json)),
            ]));
        }
    }
    let sizes: Vec<(usize, usize)> = images.iter().map(|im| (im.h, im.w)).collect();
    let wanted: Vec<i64> = configs.iter().flatten().map(|c| c.scale_preview).collect();
    let (scales, warning) = preview_scale(&sizes, &wanted);
    if let Some(text) = warning {
        warn(&text);
    }
    let cells: Vec<RgbaImage> = images
        .iter()
        .zip(&scales)
        .map(|(im, &s)| upscale(im, s))
        .collect();
    let previews: Vec<Vec<RgbaImage>> = cells.chunks(columns.len()).map(<[_]>::to_vec).collect();
    std::fs::create_dir_all(outdir).map_err(|e| CliError::os(e, outdir))?;
    save_sheet(sheet_path, &proof_sheet(&previews, &methods, &labels))?;
    emit(&Json::Object(vec![
        ("compare".into(), Json::Str(paths::display(sheet_path))),
        ("cells".into(), Json::List(report)),
    ]))
}

/// The three methods with the same config (`--method` ignored): each one's outputs and a
/// side-by-side sheet of their previews.
fn methods_side_by_side(
    input: &Path,
    outdir: &Path,
    stem: &str,
    sheet_path: &Path,
    mut given: ConfigArgs,
    force: bool,
) -> Result<(), CliError> {
    given.flags.retain(|(k, _)| k != "method");
    let base = build_config(given.config_path.as_deref(), &given.flags)?;
    let prefix = |method: &str| outdir.join(format!("{stem}_{method}"));
    let mut outputs = vec![sheet_path.to_path_buf()];
    for method in METHODS {
        outputs.extend(
            pipeline::output_paths(&prefix(method), base.tileset)
                .into_iter()
                .map(|(_, p)| p),
        );
    }
    check_collisions(&[input.to_path_buf()], &outputs, force)?;
    let mut previews = Vec::with_capacity(METHODS.len());
    let mut report = Vec::with_capacity(METHODS.len());
    for method in METHODS {
        let config = base.replace(&[("method".to_string(), Value::Str(method.to_string()))])?;
        let result = pipeline::run(Input::Path(input), &config, given.max_input_pixels)?;
        show_warnings(&result.warnings);
        let saved = result.save(&prefix(method))?;
        report.push((
            method.to_string(),
            Json::Object(vec![
                ("outputs".into(), outputs_json(&saved)),
                ("stats".into(), result.stats.clone()),
            ]),
        ));
        previews.push(upscale(&image_of(&result), config.scale_preview));
    }
    let gap = usize::try_from(base.scale_preview).unwrap_or(0);
    save_sheet(sheet_path, &side_by_side(&previews, gap))?;
    emit(&Json::Object(vec![
        ("compare".into(), Json::Str(paths::display(sheet_path))),
        ("methods".into(), Json::Object(report)),
    ]))
}
