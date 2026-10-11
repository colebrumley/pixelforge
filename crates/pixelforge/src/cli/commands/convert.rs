//! `pixelforge convert INPUT`: one image.

use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command};

use crate::cli::{
    args, build_config, check_collisions, config_args, config_args_from, emit, output_args,
    outputs_json, paths, show_warnings, subcommand, CliError,
};
use crate::pipeline::{self, Input};
use crate::pyjson::Json;

pub fn command() -> Command {
    let cmd = subcommand("convert", "Convert one image.").arg(
        Arg::new("input")
            .value_name("INPUT")
            .required(true)
            .value_parser(args::file_path()),
    );
    config_args(output_args(cmd))
}

pub fn run(matches: &ArgMatches) -> Result<(), CliError> {
    let input: &PathBuf = matches.get_one("input").expect("required");
    let outdir: &PathBuf = matches.get_one("outdir").expect("has a default");
    let force = matches.get_flag("force");
    let given = config_args_from(matches);
    let config = build_config(given.config_path.as_deref(), &given.flags)?;
    let prefix = outdir.join(paths::stem(input));
    let outputs: Vec<PathBuf> = pipeline::output_paths(&prefix, config.tileset)
        .into_iter()
        .map(|(_, p)| p)
        .collect();
    check_collisions(std::slice::from_ref(input), &outputs, force)?;
    let result = pipeline::run(Input::Path(input), &config, given.max_input_pixels)?;
    show_warnings(&result.warnings);
    let saved = result.save(&prefix)?;
    emit(&Json::Object(vec![
        ("outputs".into(), outputs_json(&saved)),
        ("stats".into(), result.stats.clone()),
    ]))
}
