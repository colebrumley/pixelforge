//! The parts of Python's `pathlib` the CLI's messages and checks depend on: the normalised
//! spelling of a path (`str(Path(p))`), `stem` and `suffix`, and `Path.resolve()`, which
//! (unlike `std::fs::canonicalize`) also resolves paths that do not exist yet.

use std::path::{Component, Path, PathBuf};

/// `Path(p)` as `pathlib` spells it: repeated separators and `.` components dropped, no
/// trailing separator, `.` for an empty path (a leading `//` is not special-cased).
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// `str(Path(p))`.
pub fn display(path: &Path) -> String {
    normalize(path).to_string_lossy().into_owned()
}

/// `Path(p).name`: the last component (`..` included), or "" for a root or `.`.
fn name(path: &Path) -> String {
    match normalize(path).components().next_back() {
        Some(Component::Normal(n)) => n.to_string_lossy().into_owned(),
        Some(Component::ParentDir) => "..".into(),
        _ => String::new(),
    }
}

/// The index of the suffix's dot in `name` (pathlib: a dot that is neither the first nor
/// the last character).
fn suffix_dot(name: &str) -> Option<usize> {
    let i = name.rfind('.')?;
    (i > 0 && i < name.len() - 1).then_some(i)
}

/// `Path(p).stem`.
pub fn stem(path: &Path) -> String {
    let name = name(path);
    match suffix_dot(&name) {
        Some(i) => name[..i].to_string(),
        None => name,
    }
}

/// `Path(p).suffix`.
pub fn suffix(path: &Path) -> String {
    let name = name(path);
    match suffix_dot(&name) {
        Some(i) => name[i..].to_string(),
        None => String::new(),
    }
}

/// `Path(p).resolve()` (non-strict, i.e. `os.path.realpath`): an absolute path with every
/// symlink of the existing part resolved; components that do not exist are appended as
/// they are, and `..` removes the last component of what has been resolved so far.
pub fn resolve(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut out = PathBuf::new();
    for c in absolute.components() {
        match c {
            Component::Prefix(_) | Component::RootDir => out.push(c.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => {
                let candidate = out.join(name);
                out = match std::fs::symlink_metadata(&candidate) {
                    // Exists (perhaps as a symlink): the canonical form of everything so far,
                    // or, for a dangling or looping link, the path as it is.
                    Ok(_) => std::fs::canonicalize(&candidate).unwrap_or(candidate),
                    Err(_) => candidate,
                };
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pathlib_spelling() {
        assert_eq!(display(Path::new("./out//a/")), "out/a");
        assert_eq!(display(Path::new("")), ".");
        assert_eq!(display(Path::new("./")), ".");
        assert_eq!(display(Path::new("/tmp/./x")), "/tmp/x");
        assert_eq!(display(Path::new("a/../b")), "a/../b");
    }

    #[test]
    fn stem_and_suffix() {
        for (p, s, x) in [
            ("a/b.png", "b", ".png"),
            ("b.tar.gz", "b.tar", ".gz"),
            (".hidden", ".hidden", ""),
            ("name.", "name.", ""),
            ("noext", "noext", ""),
            ("dir/x.PNG", "x", ".PNG"),
            ("..", "..", ""),
            ("a/.", "a", ""),
        ] {
            assert_eq!(stem(Path::new(p)), s, "{p}");
            assert_eq!(suffix(Path::new(p)), x, "{p}");
        }
    }

    #[test]
    fn resolve_is_lenient() {
        let tmp = std::env::temp_dir().join(format!("pf-resolve-{}", std::process::id()));
        let real = tmp.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let base = std::fs::canonicalize(&real).unwrap();
        // Existing prefix canonicalised, the rest appended lexically.
        assert_eq!(
            resolve(&real.join("missing/x.png")),
            base.join("missing/x.png")
        );
        // `..` after a missing component pops it again.
        assert_eq!(resolve(&real.join("missing/../y")), base.join("y"));
        assert_eq!(resolve(&real.join("./a/./b")), base.join("a/b"));
        #[cfg(unix)]
        {
            let link = tmp.join("link");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(&real, &link).unwrap();
            assert_eq!(resolve(&link.join("new.png")), base.join("new.png"));
            assert_eq!(resolve(&link), base);
        }
        // Relative paths are made absolute against the working directory.
        let cwd = std::fs::canonicalize(std::env::current_dir().unwrap()).unwrap();
        assert_eq!(resolve(Path::new("no/such/file")), cwd.join("no/such/file"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
