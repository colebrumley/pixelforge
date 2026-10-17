//! The `pixelforge` command line (`pixelforge/cli.py`, Section 10): `convert`, `batch`,
//! `compare`, `palettes`.
//!
//! Behaviour, messages and exit codes are the Python CLI's: results go to stdout as JSON
//! lines (sorted keys, compact), progress (`-v`, `-vv`), warnings and errors go to stderr
//! prefixed `pixelforge: `; config errors exit 2 with `config error: …`, every other failure
//! exits 1 with `error: …` (the error's debug form instead with `--debug` or
//! `PIXELFORGE_DEBUG`), and usage errors exit 2.

mod args;
mod commands;
mod config_load;
mod logging;
mod paths;

use std::ffi::OsString;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Arg, ArgAction, ArgMatches, Command};

use crate::config::ConfigError;
use crate::io::{os_error_message, IoError};
use crate::pipeline;
/// The JSON reader (`json.loads`) lives in the library, which the edit log needs without the
/// CLI.
use crate::pyjson::loads as json;
use crate::pyjson::{dumps, Json, Layout};

pub use args::{config_args, config_args_from, field_help, flag_name, py_int, ConfigArgs};
pub use config_load::{build_config, read_config_text, MAX_CONFIG_FILE_BYTES};
pub use paths::{display as path_display, resolve as resolve_path, stem as path_stem};

/// The version `--version` prints (`pixelforge, version X`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------------------
// Errors

/// A failure of a command: a config error (exit 2) or any other error (exit 1), with the
/// message Python printed and, for `--debug`, the underlying error.
pub struct CliError {
    pub config: bool,
    pub message: String,
    pub source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl CliError {
    /// A config error (`ConfigError`): exit 2.
    pub fn config(message: impl Into<String>) -> CliError {
        CliError {
            config: true,
            message: message.into(),
            source: None,
        }
    }

    /// Any other failure: exit 1.
    pub fn runtime(message: impl Into<String>) -> CliError {
        CliError {
            config: false,
            message: message.into(),
            source: None,
        }
    }

    /// An `OSError` on `path`, as Python words it (`[Errno N] text: 'path'`).
    pub fn os(error: std::io::Error, path: &Path) -> CliError {
        CliError::runtime(os_error_message(&error, &paths::display(path))).with_source(error)
    }

    fn with_source(mut self, e: impl std::error::Error + Send + Sync + 'static) -> CliError {
        self.source = Some(Box::new(e));
        self
    }

    pub fn exit_code(&self) -> i32 {
        if self.config {
            2
        } else {
            1
        }
    }

    /// The stderr text of `handle_errors`: `{prefix}: {message}`, or with `debug` the
    /// error's debug form and source chain (Python printed the traceback).
    pub fn report(&self, debug: bool) -> String {
        let prefix = if self.config { "config error" } else { "error" };
        let message = if self.message.is_empty() {
            if self.config {
                "ConfigError"
            } else {
                "Error"
            }
        } else {
            &self.message
        };
        if !debug {
            return format!("{prefix}: {message}");
        }
        let mut out = format!("{prefix}: Traceback (error chain, outermost first):\n");
        match &self.source {
            Some(source) => {
                out.push_str(&format!("  {source:?}\n"));
                let mut next = source.source();
                while let Some(e) = next {
                    out.push_str(&format!("  caused by: {e:?}\n"));
                    next = e.source();
                }
            }
            None => out.push_str(&format!(
                "  {}Error({:?})\n",
                if self.config { "Config" } else { "Runtime" },
                self.message
            )),
        }
        out.push_str(message);
        out
    }
}

impl fmt::Debug for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report(true))
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<ConfigError> for CliError {
    fn from(e: ConfigError) -> Self {
        CliError::config(e.0.clone()).with_source(e)
    }
}

impl From<pipeline::Error> for CliError {
    fn from(e: pipeline::Error) -> Self {
        let config = matches!(e, pipeline::Error::Config(_));
        let message = e.to_string();
        CliError {
            config,
            message,
            source: Some(Box::new(e)),
        }
    }
}

impl From<IoError> for CliError {
    fn from(e: IoError) -> Self {
        pipeline::Error::from(e).into()
    }
}

impl From<crate::preprocess::Error> for CliError {
    fn from(e: crate::preprocess::Error) -> Self {
        pipeline::Error::from(e).into()
    }
}

impl From<crate::mask::PixelforgeError> for CliError {
    fn from(e: crate::mask::PixelforgeError) -> Self {
        pipeline::Error::from(e).into()
    }
}

// ---------------------------------------------------------------------------------------
// Output

/// `_emit`: one JSON line on stdout (`json.dumps(payload, sort_keys=True,
/// separators=(",", ":"))`), flushed.
pub fn emit(payload: &Json) -> Result<(), CliError> {
    let layout = Layout {
        indent: None,
        sort_keys: true,
        compact: true,
    };
    let line = dumps(payload, layout);
    let mut out = std::io::stdout().lock();
    writeln!(out, "{line}")
        .and_then(|()| out.flush())
        .map_err(|e| {
            // Python: `[Errno 32] Broken pipe` (an OSError without a filename).
            let text = match e.raw_os_error() {
                Some(code) => {
                    let full = std::io::Error::from_raw_os_error(code).to_string();
                    let suffix = format!(" (os error {code})");
                    format!(
                        "[Errno {code}] {}",
                        full.strip_suffix(&suffix).unwrap_or(&full)
                    )
                }
                None => e.to_string(),
            };
            CliError::runtime(text).with_source(e)
        })
}

/// `_warn`: a CLI warning on stderr (`pixelforge: {message}`), unless `-q`.
pub fn warn(message: &str) {
    log::warn!(target: logging::TARGET, "{message}");
}

/// The texts of the library's `UserWarning`s (multi-frame input, the stretch aspect), shown
/// once each per process as `pixelforge: warning: {text}` (Python printed them through its
/// `warnings` module, as `path:line: UserWarning: {text}`).
pub fn show_warnings(texts: &[String]) {
    use std::sync::Mutex;
    static SHOWN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut shown = SHOWN.lock().unwrap_or_else(|e| e.into_inner());
    for text in texts {
        if !shown.contains(text) {
            warn(&format!("warning: {text}"));
            shown.push(text.clone());
        }
    }
}

/// `{name: path}` of the files `Output::save` wrote, as a JSON object.
pub fn outputs_json(paths: &[(String, PathBuf)]) -> Json {
    Json::Object(
        paths
            .iter()
            .map(|(name, path)| (name.clone(), Json::Str(path.to_string_lossy().into_owned())))
            .collect(),
    )
}

/// `check_collisions`: a config error when an output path resolves to an input file (an
/// existing regular file), unless `force`.
pub fn check_collisions(
    inputs: &[PathBuf],
    outputs: &[PathBuf],
    force: bool,
) -> Result<(), CliError> {
    if force {
        return Ok(());
    }
    // {resolved input: its spelling}; a later input with the same target replaces the name.
    let mut resolved: Vec<(PathBuf, String)> = Vec::new();
    for p in inputs {
        if p.is_file() {
            let key = paths::resolve(p);
            let shown = paths::display(p);
            match resolved.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = shown,
                None => resolved.push((key, shown)),
            }
        }
    }
    let mut hits: Vec<String> = outputs
        .iter()
        .filter_map(|out| {
            let key = paths::resolve(out);
            resolved
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, input)| format!("{} would overwrite input {input}", paths::display(out)))
        })
        .collect();
    hits.sort();
    if hits.is_empty() {
        Ok(())
    } else {
        Err(CliError::config(format!(
            "{} (use another --outdir, or --force)",
            hits.join("; ")
        )))
    }
}

// ---------------------------------------------------------------------------------------
// The command tree

const ROOT_DOC: &str = "Deterministic conversion of images into 16-bit-style pixel art.

Results go to stdout (JSON lines); progress (-v), warnings and errors go to
stderr.";

/// `--help` only (click has no `-h`).
fn help_arg() -> Arg {
    Arg::new("help")
        .long("help")
        .action(ArgAction::HelpShort)
        .help("Show this message and exit.")
}

/// The help layout of click: usage, the docstring indented, then the options. The first
/// paragraph of `doc` is the one-line summary in the parent's command list.
fn documented(cmd: Command, doc: &str) -> Command {
    let summary = doc.split("\n\n").next().unwrap_or(doc).to_string();
    let indented: Vec<String> = doc
        .lines()
        .map(|l| {
            if l.is_empty() {
                String::new()
            } else {
                format!("  {l}")
            }
        })
        .collect();
    cmd.about(summary).help_template(format!(
        "{{usage-heading}} {{usage}}\n\n{}\n\n{{all-args}}{{after-help}}",
        indented.join("\n")
    ))
}

/// A subcommand with click's conventions: `--help` only, the last of a repeated option wins,
/// the docstring `doc` as its help text.
pub fn subcommand(name: &'static str, doc: &str) -> Command {
    let cmd = Command::new(name)
        .disable_help_flag(true)
        .args_override_self(true);
    documented(cmd, doc)
}

/// `-o/--outdir` and `--force`, shared by every converting command.
pub fn output_args(cmd: Command) -> Command {
    cmd.arg(
        Arg::new("outdir")
            .short('o')
            .long("outdir")
            .value_name("DIRECTORY")
            .default_value("out")
            .allow_hyphen_values(true)
            .value_parser(args::dir_path())
            .help("Output directory."),
    )
    .arg(args::flag(
        "force",
        "force",
        "Allow outputs that overwrite an input file.",
    ))
}

/// The full command tree.
pub fn command() -> Command {
    documented(Command::new("pixelforge"), ROOT_DOC)
        .override_usage("pixelforge [OPTIONS] COMMAND [ARGS]...")
        .disable_version_flag(true)
        .disable_help_flag(true)
        .disable_help_subcommand(true)
        .args_override_self(true)
        .subcommand_value_name("COMMAND")
        .arg(
            Arg::new("version")
                .long("version")
                .action(ArgAction::SetTrue)
                .help("Show the version and exit."),
        )
        .arg(args::flag(
            "debug",
            "debug",
            "Print full tracebacks on errors (also: PIXELFORGE_DEBUG=1).",
        ))
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(ArgAction::Count)
                .help("Progress on stderr: -v per stage, -vv every iteration."),
        )
        .arg(
            Arg::new("quiet")
                .short('q')
                .long("quiet")
                .action(ArgAction::SetTrue)
                .help("Only errors on stderr (no warnings or hints)."),
        )
        .arg(help_arg())
        .subcommand(commands::batch::command().arg(help_arg()))
        .subcommand(commands::compare::command().arg(help_arg()))
        .subcommand(commands::convert::command().arg(help_arg()))
        .subcommand(commands::palettes::command().arg(help_arg()))
}

/// click's eager `--version`: given among the group's options (before the command name)
/// and before any `--help`, it prints the version and exits.
fn wants_version(args: &[OsString]) -> bool {
    for arg in args.iter().skip(1) {
        let Some(a) = arg.to_str() else { return false };
        match a {
            "--version" => return true,
            "--help" | "--" => return false,
            _ if !a.starts_with('-') || a == "-" => return false,
            _ => {}
        }
    }
    false
}

fn debug_requested(matches: &ArgMatches) -> bool {
    matches.get_flag("debug")
        || std::env::var_os("PIXELFORGE_DEBUG").is_some_and(|v| !v.is_empty() && v != "0")
}

fn usage_error(message: &str) -> i32 {
    eprintln!(
        "Usage: pixelforge [OPTIONS] COMMAND [ARGS]...\nTry 'pixelforge --help' for help.\n\n\
         Error: {message}"
    );
    2
}

/// Run the CLI on `args` (program name first); returns the exit code.
pub fn run(args: Vec<OsString>) -> i32 {
    if wants_version(&args) {
        println!("pixelforge, version {VERSION}");
        return 0;
    }
    let mut cmd = command();
    if args.len() <= 1 {
        // click 8.2+: a group without arguments shows its help on stderr and exits 2.
        let _ = write!(std::io::stderr(), "{}", cmd.render_help());
        return 2;
    }
    let matches = match cmd.try_get_matches_from_mut(&args) {
        Ok(m) => m,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    logging::configure(matches.get_count("verbose"), matches.get_flag("quiet"));
    let debug = debug_requested(&matches);
    let Some((name, sub)) = matches.subcommand() else {
        return usage_error("Missing command.");
    };
    let result = match name {
        "convert" => commands::convert::run(sub),
        "batch" => commands::batch::run(sub),
        "compare" => commands::compare::run(sub),
        "palettes" => commands::palettes::run(sub),
        other => unreachable!("unknown subcommand {other}"),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{}", e.report(debug));
            e.exit_code()
        }
    }
}

/// The program: `run` on the process arguments, with a panic reported as an error (exit 1)
/// like any other failure.
pub fn main() -> i32 {
    let args: Vec<OsString> = std::env::args_os().collect();
    let debug_env = std::env::var_os("PIXELFORGE_DEBUG").is_some_and(|v| !v.is_empty() && v != "0");
    let debug_flag = args.iter().any(|a| a == "--debug");
    if !(debug_env || debug_flag) {
        // The message is printed below as `error: …`; the default hook would add a
        // "thread 'main' panicked at" line.
        std::panic::set_hook(Box::new(|_| {}));
    }
    match std::panic::catch_unwind(|| run(args)) {
        Ok(code) => code,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "internal error".into());
            eprintln!("error: {message}");
            1
        }
    }
}
