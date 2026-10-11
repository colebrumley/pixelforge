//! The `xtask` command line: `fixtures`, `golden`, `bench`, `snapshot`, `gallery`.

use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Arg, ArgAction, Command};

use crate::{bench, fixtures, gallery, golden, snapshot, Fail};

pub fn command() -> Command {
    Command::new("xtask")
        .about("pixelforge developer tooling (run as `cargo xtask <command>`)")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("fixtures")
                .about("Draw the synthetic test images of tests/fixtures (no RNG).")
                .long_about(
                    "Draw the synthetic test images of tests/fixtures deterministically (no \
                     RNG). Without --check, write them to OUTDIR (default: tests/fixtures); the \
                     pixels equal the committed files', the PNG bytes may not (so do not \
                     overwrite the committed fixtures: every input_sha256 would change).",
                )
                .arg(
                    Arg::new("check")
                        .long("check")
                        .action(ArgAction::SetTrue)
                        .help("decode the committed fixtures and fail if any pixel differs"),
                )
                .arg(
                    Arg::new("outdir")
                        .value_name("OUTDIR")
                        .value_parser(clap::value_parser!(PathBuf))
                        .help("directory to write (or, with --check, to check)"),
                ),
        )
        .subcommand(
            Command::new("golden")
                .about("Regenerate tests/golden.json: SHA-256 of the pixels each method/preset produces.")
                .long_about(
                    "Regenerate tests/golden.json: SHA-256 of the pixels each method/preset \
                     produces.\n\nRun this only when an output change is intentional (an \
                     algorithm fix, a new default), then review and commit the diff together \
                     with the change.",
                )
                .arg(
                    Arg::new("check")
                        .long("check")
                        .action(ArgAction::SetTrue)
                        .help("exit 1 if tests/golden.json is out of date (write nothing)"),
                ),
        )
        .subcommand(bench::command().args_override_self(true))
        .subcommand(snapshot::command())
        .subcommand(gallery::command())
}

/// Run `xtask` with `args` (the program name first); the exit status.
pub fn run<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let matches = match command().try_get_matches_from(args) {
        Ok(m) => m,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    let result: Result<i32, Fail> = match matches.subcommand() {
        Some(("fixtures", m)) => {
            let dir = m
                .get_one::<PathBuf>("outdir")
                .cloned()
                .unwrap_or_else(fixtures::default_dir);
            if m.get_flag("check") {
                fixtures::check(&dir).map(|()| 0)
            } else {
                fixtures::write(&dir).map(|written| {
                    for path in written {
                        println!("{}", path.display());
                    }
                    0
                })
            }
        }
        Some(("golden", m)) => {
            golden::main(&golden::golden_path(), m.get_flag("check")).map(|()| 0)
        }
        Some(("bench", m)) => bench::main(m).map(|()| 0),
        Some(("snapshot", m)) => snapshot::main(m),
        Some(("gallery", m)) => gallery::main(m).map(|_| 0),
        _ => unreachable!("subcommand_required"),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{}", e.message);
            e.code
        }
    }
}
