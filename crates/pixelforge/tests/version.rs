//! One version everywhere: the workspace `Cargo.toml` (the single source), the library's
//! `pipeline::VERSION` and `pixelforge --version`.

use std::path::{Path, PathBuf};

use assert_cmd::Command;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// The `version = "..."` line of the `[section]` table of a TOML file.
fn toml_version(path: &Path, section: &str) -> String {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .skip_while(|l| l.trim() != section)
        .skip(1)
        .take_while(|l| !l.starts_with('['))
        .find_map(|l| l.strip_prefix("version = \""))
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or_else(|| panic!("no version in {section} of {}", path.display()))
        .to_string()
}

#[test]
fn version_is_the_workspace_version() {
    let cargo = toml_version(&root().join("Cargo.toml"), "[workspace.package]");
    assert_eq!(cargo, VERSION);
    assert_eq!(pixelforge::pipeline::VERSION, VERSION);
}

#[test]
fn cli_version_matches_package() {
    let out = Command::cargo_bin("pixelforge")
        .unwrap()
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("pixelforge, version {VERSION}\n")
    );
}
