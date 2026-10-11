//! `pixelforge compare`: the option set of the Python command, so the command tree and its
//! `--help` are complete. Running it is not ported yet (the proof sheet lives in
//! `sheet.rs`); it reports an error and exits 1.

use std::path::PathBuf;

use clap::{Arg, ArgAction, ArgMatches, Command};

use crate::cli::{args, config_args, output_args, subcommand, CliError};

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
    let _input: Option<&PathBuf> = matches.get_one("input");
    Err(CliError::runtime("compare is not available yet"))
}
