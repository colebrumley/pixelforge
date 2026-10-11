//! tests/test_version.py: one version everywhere. `test_version_is_single_sourced` (the
//! installed distribution metadata) is Python-only and not ported.

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
fn version_matches_pyproject_and_cargo() {
    let root = root();
    let source = std::fs::read_to_string(root.join("pixelforge/version.py")).unwrap();
    let source = source
        .lines()
        .find_map(|l| l.strip_prefix("__version__ = \""))
        .and_then(|v| v.strip_suffix('"'))
        .expect("__version__ in pixelforge/version.py");
    let project = toml_version(&root.join("pyproject.toml"), "[project]");
    let cargo = toml_version(&root.join("Cargo.toml"), "[workspace.package]");
    assert_eq!(cargo, VERSION);
    assert_eq!(project, source);
    assert_eq!(source, VERSION);
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
