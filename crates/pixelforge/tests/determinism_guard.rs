//! Determinism rules (crates/pixelforge/README.md, plan Decision 6): transcendental
//! functions come from the `libm` crate and rounding to integers is `round_ties_even`.
//! This test fails if any source file of the library or the binding calls the platform
//! versions or the half-away-from-zero `round` outside a comment. Nothing is allow-listed.

use std::path::{Path, PathBuf};

const FORBIDDEN: &[&str] = &[
    ".round()",
    "f64::exp(",
    ".exp()",
    ".powf(",
    ".powi(",
    ".cbrt()",
    ".ln(",
    ".log(",
    ".sin(",
    ".cos(",
    ".atan2(",
    ".mul_add(",
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

/// The line without its `//` comment (line, doc and inner-doc comments), and `""` for lines
/// inside a `/* ... */` block. A `//` inside a string literal also ends the scanned part,
/// which can only hide code, never invent a finding.
fn code_part<'a>(line: &'a str, in_block: &mut bool) -> &'a str {
    let mut s = line;
    if *in_block {
        match s.find("*/") {
            Some(end) => {
                *in_block = false;
                s = &s[end + 2..];
            }
            None => return "",
        }
    }
    if let Some(start) = s.find("/*") {
        if !s[start..].contains("*/") {
            *in_block = true;
        }
        s = &s[..start];
    }
    match s.find("//") {
        Some(i) => &s[..i],
        None => s,
    }
}

#[test]
fn no_platform_math_or_half_away_rounding() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    rust_files(&root.join("../pixelforge-py/src"), &mut files);
    assert!(files.len() >= 5, "found only {files:?}");
    let mut findings = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        let mut in_block = false;
        for (n, line) in text.lines().enumerate() {
            let code = code_part(line, &mut in_block);
            for pat in FORBIDDEN {
                if code.contains(pat) {
                    findings.push(format!(
                        "{}:{}: {pat}  {}",
                        file.display(),
                        n + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
    assert!(
        findings.is_empty(),
        "use libm::* / f64::round_ties_even instead (plan Decision 6):\n{}",
        findings.join("\n")
    );
}

#[test]
fn comment_stripping() {
    let mut b = false;
    assert_eq!(
        code_part("let y = x.round(); // ok", &mut b),
        "let y = x.round(); "
    );
    assert_eq!(code_part("/// uses .round() in docs", &mut b), "");
    assert_eq!(code_part("a /* start", &mut b), "a ");
    assert!(b);
    assert_eq!(code_part("x.exp() inside", &mut b), "");
    assert_eq!(code_part("end */ x.ln(2)", &mut b), " x.ln(2)");
    assert!(!b);
}
