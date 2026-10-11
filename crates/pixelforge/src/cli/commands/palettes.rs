//! `pixelforge palettes`: the bundled palettes and their sizes.

use std::io::Write;

use clap::{ArgMatches, Command};

use crate::cli::{subcommand, CliError};
use crate::palette;

pub fn command() -> Command {
    subcommand("palettes", "List the bundled palettes.")
}

pub fn run(_matches: &ArgMatches) -> Result<(), CliError> {
    let mut out = std::io::stdout().lock();
    for name in palette::bundled_palettes() {
        let colors = palette::load_palette(name).map_err(|e| CliError::runtime(e.0))?;
        writeln!(out, "{name}\t{} colors", colors.len())
            .map_err(|e| CliError::runtime(e.to_string()))?;
    }
    Ok(())
}
