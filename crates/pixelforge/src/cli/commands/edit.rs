//! `pixelforge edit OUTPUT.png`: open a pixelforge output (or any PNG with at most 256
//! colors) in the editor (docs/design/editor.md R1, R3), and the windows `convert --edit` (R2) and
//! `batch --edit` (R4) open. Without the `gui` feature the command exists but only checks
//! that the file opens, then says the editor is not built in.

use std::path::PathBuf;

use clap::{Arg, ArgMatches, Command};

use crate::cli::{args, subcommand, CliError};
use crate::editor::Session;

const DOC: &str = "Open a pixelforge output in the editor.

OUTPUT is a PNG written by convert (its palette, config and input hash
are read from the file), or any PNG with at most 256 colors (default
config). Saving writes every output file of convert under the same
name, plus NAME_edits.json (replay it with convert --edits).";

pub fn command() -> Command {
    subcommand("edit", DOC)
        .arg(
            Arg::new("output")
                .value_name("OUTPUT")
                .required(true)
                .value_parser(args::file_path()),
        )
        .arg(
            Arg::new("source")
                .long("source")
                .value_name("FILE")
                .allow_hyphen_values(true)
                .value_parser(args::file_path())
                .help("The original input image, shown for reference."),
        )
        .arg(
            Arg::new("outdir")
                .short('o')
                .long("outdir")
                .value_name("DIRECTORY")
                .allow_hyphen_values(true)
                .value_parser(args::dir_path())
                .help("Where saves go (default: OUTPUT's directory)."),
        )
}

/// The error of a program built without the editor.
pub fn not_built() -> CliError {
    CliError::runtime(
        "this pixelforge was built without the editor (the gui feature); \
         reinstall it with the default features",
    )
}

/// Open the editor window on `session`; returns when it is closed.
#[cfg(feature = "gui")]
pub fn open(session: Session) -> Result<(), CliError> {
    crate::gui::run(session)
        .map_err(|e| CliError::runtime(format!("cannot open the editor window: {e}")))
}

#[cfg(not(feature = "gui"))]
pub fn open(_session: Session) -> Result<(), CliError> {
    Err(not_built())
}

/// Open the editor window on the frames of a batch (`batch --edit`); returns when it is
/// closed.
#[cfg(feature = "gui")]
pub fn open_frames(sessions: Vec<Session>) -> Result<(), CliError> {
    crate::gui::run_frames(sessions)
        .map_err(|e| CliError::runtime(format!("cannot open the editor window: {e}")))
}

#[cfg(not(feature = "gui"))]
pub fn open_frames(_sessions: Vec<Session>) -> Result<(), CliError> {
    Err(not_built())
}

pub fn run(matches: &ArgMatches) -> Result<(), CliError> {
    let output: &PathBuf = matches.get_one("output").expect("required");
    let source: Option<&PathBuf> = matches.get_one("source");
    let outdir: Option<&PathBuf> = matches.get_one("outdir");
    // A file that does not open fails like an unreadable input of convert (exit 1).
    let session = Session::open(
        output,
        source.map(|p| p.as_path()),
        outdir.map(|p| p.as_path()),
    )
    .map_err(|e| CliError::runtime(e.0))?;
    open(session)
}
