//! `xtask help-record [--bin PATH] [OUTDIR]`: record the `--help` text of the `pixelforge`
//! group and of each command in `tests/cli_help/<command>.txt` (`<command>` = `group`
//! for the group itself). `crates/pixelforge/tests/cli.rs` asserts the live help equals
//! these records, so rerun this after changing a flag, a default or a help text, and review
//! the diff.
//!
//! The program is `--bin`, else `$PIXELFORGE_BIN`, else `target/release/pixelforge`, built
//! first with `cargo build --release -p pixelforge`. clap does not wrap the help, so the
//! records do not depend on the terminal width.

use std::path::{Path, PathBuf};
use std::process::Command as Process;

use clap::{Arg, ArgMatches, Command};

use crate::{fail, root, Fail};

/// The group (`group`) and every command, as the records name them.
pub const COMMANDS: &[&str] = &["group", "batch", "compare", "convert", "palettes"];

pub fn command() -> Command {
    Command::new("help-record")
        .about("Record the pixelforge --help texts in tests/cli_help/<command>.txt.")
        .long_about(
            "Record the --help text of the pixelforge group and of each command in \
             OUTDIR/<command>.txt (default OUTDIR: tests/cli_help). The program is \
             --bin, else $PIXELFORGE_BIN, else target/release/pixelforge, built first.",
        )
        .arg(
            Arg::new("bin")
                .long("bin")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf))
                .help("the pixelforge program to run (not rebuilt)"),
        )
        .arg(
            Arg::new("outdir")
                .value_name("OUTDIR")
                .value_parser(clap::value_parser!(PathBuf))
                .help("directory to write"),
        )
}

pub fn default_dir() -> PathBuf {
    root().join("tests").join("cli_help")
}

/// The arguments that print `command`'s help.
pub fn help_args(command: &str) -> Vec<&str> {
    if command == "group" {
        vec!["--help"]
    } else {
        vec![command, "--help"]
    }
}

/// `binary <help_args(command)>`'s stdout.
pub fn help_text(binary: &Path, command: &str) -> Result<String, Fail> {
    let out = Process::new(binary)
        .args(help_args(command))
        .env_remove("PIXELFORGE_DEBUG")
        .output()
        .map_err(|e| fail(1, format!("{}: {e}", binary.display())))?;
    if !out.status.success() {
        return Err(fail(
            1,
            format!(
                "{} {}: {}\n{}",
                binary.display(),
                help_args(command).join(" "),
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ),
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| fail(1, format!("non-UTF-8 help: {e}")))
}

/// `cargo build --release -p pixelforge`; the program's path.
fn build_release() -> Result<PathBuf, Fail> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Process::new(cargo)
        .args([
            "build",
            "--quiet",
            "--release",
            "-p",
            "pixelforge",
            "--bin",
            "pixelforge",
        ])
        .current_dir(root())
        .status()
        .map_err(|e| fail(1, format!("cargo build: {e}")))?;
    if !status.success() {
        return Err(fail(1, format!("cargo build: {status}")));
    }
    let exe = format!("pixelforge{}", std::env::consts::EXE_SUFFIX);
    Ok(root().join("target").join("release").join(exe))
}

pub fn main(m: &ArgMatches) -> Result<(), Fail> {
    let binary = match m.get_one::<PathBuf>("bin") {
        Some(p) => p.clone(),
        None => match std::env::var_os("PIXELFORGE_BIN") {
            Some(p) => PathBuf::from(p),
            None => build_release()?,
        },
    };
    let dir = m
        .get_one::<PathBuf>("outdir")
        .cloned()
        .unwrap_or_else(default_dir);
    std::fs::create_dir_all(&dir).map_err(|e| fail(1, format!("{}: {e}", dir.display())))?;
    for command in COMMANDS {
        let path = dir.join(format!("{command}.txt"));
        std::fs::write(&path, help_text(&binary, command)?)
            .map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
        println!("{}", path.display());
    }
    Ok(())
}
