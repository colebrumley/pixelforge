//! pixelforge developer tooling (`cargo xtask …`, Decision 9 of docs/rust-migration-plan.md):
//! `fixtures`, `golden`, `bench`, `snapshot` and `gallery` (ports of the Python scripts
//! removed in 0.2.0, with their options, tables, files and exit codes) and `help-record`.
//!
//! Errors are reported as the scripts' `SystemExit(message)` was: the message on stderr and
//! exit status 1; usage errors exit 2 (clap, as argparse).

pub mod bench;
pub mod cli;
pub mod fixtures;
pub mod gallery;
pub mod golden;
pub mod help;
pub mod json;
pub mod snapshot;

use std::path::{Path, PathBuf};

/// A failure: the message for stderr and the exit status.
#[derive(Debug)]
pub struct Fail {
    pub code: i32,
    pub message: String,
}

pub fn fail(code: i32, message: impl Into<String>) -> Fail {
    Fail {
        code,
        message: message.into(),
    }
}

/// The repository root (the parent of `xtask/`).
pub fn root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let parent = manifest.parent().unwrap_or(manifest);
    std::fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf())
}

/// `Path.resolve()` (non-strict): absolute, symlinks resolved where the path exists.
pub fn resolve(path: &Path) -> PathBuf {
    if let Ok(p) = std::fs::canonicalize(path) {
        return p;
    }
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

/// `bench.display_path`: repository-relative for files in the checkout, absolute otherwise.
pub fn display_path(path: &Path) -> String {
    let path = resolve(path);
    match path.strip_prefix(root()) {
        Ok(rel) => rel.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

/// `Path(p).stem`: the file name without its last suffix (a leading dot is not a suffix).
pub fn stem(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => name[..i].to_string(),
        _ => name,
    }
}

/// `git <args>` in the repository root: stdout trimmed, or "" on any failure.
pub fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .current_dir(root())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}
