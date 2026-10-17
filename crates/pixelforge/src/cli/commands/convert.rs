//! `pixelforge convert INPUT`: one image; with `--edits LOG`, the editor's edit log replayed
//! on the result before it is saved.

use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command};

use crate::cli::{
    args, build_config, check_collisions, config_args, config_args_from, emit, output_args,
    outputs_json, paths, show_warnings, subcommand, CliError,
};
use crate::editor::{self, Document, EditLog};
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
        );
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
    let result = pipeline::run(Input::Path(input), &config, given.max_input_pixels)?;
    show_warnings(&result.warnings);
    let Some(log) = log else {
        let saved = result.save(&prefix)?;
        return emit(&Json::Object(vec![
            ("outputs".into(), outputs_json(&saved)),
            ("stats".into(), result.stats.clone()),
        ]));
    };
    let mut doc = Document::from_output(&result).map_err(|e| CliError::config(e.0))?;
    log.replay(&mut doc).map_err(|e| CliError::config(e.0))?;
    // `check_collisions` (with --force) decided about the input already.
    let saved = editor::save::save(&doc, &log, &prefix)?;
    emit(&Json::Object(vec![
        ("outputs".into(), outputs_json(&saved.outputs)),
        ("stats".into(), saved.stats),
    ]))
}
