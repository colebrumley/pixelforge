//! `pixelforge convert INPUT`: one image; with `--edits LOG`, the editor's edit log replayed
//! on the result before it is saved; with `--edit`, the editor opened on the saved result.

use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command};

use crate::cli::commands::edit;
use crate::cli::{
    args, build_config, check_collisions, config_args, config_args_from, emit, output_args,
    outputs_json, paths, show_warnings, subcommand, CliError,
};
use crate::editor::{self, Document, EditLog, Session};
use crate::pipeline::{self, Input};
use crate::pyjson::Json;

pub fn command() -> Command {
    let cmd = subcommand("convert", "Convert one image.")
        .arg(
            Arg::new("input")
                .value_name("INPUT")
                .required(true)
                .value_parser(args::file_path()),
        )
        .arg(
            Arg::new("edits")
                .long("edits")
                .value_name("FILE")
                .allow_hyphen_values(true)
                .value_parser(args::file_path())
                .help(
                    "Replay an edit log (NAME_edits.json, written by the editor) on the result \
                     before saving.",
                ),
        )
        .arg(args::flag(
            "edit",
            "edit",
            "Open the result in the editor after saving it (saves there rewrite the outputs).",
        ));
    config_args(output_args(cmd))
}

pub fn run(matches: &ArgMatches) -> Result<(), CliError> {
    let input: &PathBuf = matches.get_one("input").expect("required");
    let outdir: &PathBuf = matches.get_one("outdir").expect("has a default");
    let force = matches.get_flag("force");
    let edits: Option<&PathBuf> = matches.get_one("edits");
    let given = config_args_from(matches);
    let config = build_config(given.config_path.as_deref(), &given.flags)?;
    // Read before converting: a broken log fails fast.
    let log = edits
        .map(|path| EditLog::read(path).map_err(|e| CliError::config(e.0)))
        .transpose()?;
    let prefix = outdir.join(paths::stem(input));
    let mut outputs: Vec<PathBuf> = pipeline::output_paths(&prefix, config.tileset)
        .into_iter()
        .map(|(_, p)| p)
        .collect();
    if log.is_some() {
        outputs.push(editor::save::edits_path(&prefix));
    }
    check_collisions(std::slice::from_ref(input), &outputs, force)?;
    let open_editor = matches.get_flag("edit");
    if open_editor && !cfg!(feature = "gui") {
        return Err(edit::not_built());
    }
    let (result, loaded, pre) =
        pipeline::run_keeping(Input::Path(input), &config, given.max_input_pixels)?;
    show_warnings(&result.warnings);
    let doc = match log {
        None => {
            let saved = result.save(&prefix)?;
            emit(&Json::Object(vec![
                ("outputs".into(), outputs_json(&saved)),
                ("stats".into(), result.stats.clone()),
            ]))?;
            if !open_editor {
                return Ok(());
            }
            Document::from_output(&result).map_err(|e| CliError::config(e.0))?
        }
        Some(log) => {
            let mut doc = Document::from_output(&result).map_err(|e| CliError::config(e.0))?;
            log.replay(&mut doc).map_err(|e| CliError::config(e.0))?;
            // `check_collisions` (with --force) decided about the input already.
            let saved = editor::save::save(&doc, &log, &prefix)?;
            emit(&Json::Object(vec![
                ("outputs".into(), outputs_json(&saved.outputs)),
                ("stats".into(), saved.stats),
            ]))?;
            if !open_editor {
                return Ok(());
            }
            // The replayed edits stay part of every log the editor saves.
            doc.origin.prior_ops = log.ops;
            doc
        }
    };
    // The editor: the result as saved, with the input aligned for reference.
    let mut doc = doc;
    let factor = pipeline::source_reference_factor(&pre, result.indices.w);
    doc.origin.source = Some(pipeline::source_reference(&loaded, &pre, factor));
    if !force {
        doc.origin.input_path = Some(input.clone());
    }
    edit::open(Session::new(doc, prefix))
}
