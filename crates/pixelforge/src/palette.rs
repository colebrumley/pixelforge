//! Hardware palettes: the bundled `.hex` palettes (embedded), the `.hex` / `.gpl` parsers,
//! inline `hex:rrggbb,...` palettes and file loading, ported from `pixelforge/palette.py`
//! with the same rules and messages.

use crate::pyrepr::{str_list_repr, str_repr};
use crate::pyunicode;
use std::fmt;
use std::io::Read;

/// The bundled palettes, sorted by name (`bundled_palettes()`), embedded from
/// `pixelforge/palettes/*.hex`.
pub const BUNDLED: &[(&str, &str)] = &[
    (
        "gameboy",
        include_str!("../../../pixelforge/palettes/gameboy.hex"),
    ),
    (
        "genesis",
        include_str!("../../../pixelforge/palettes/genesis.hex"),
    ),
    ("nes", include_str!("../../../pixelforge/palettes/nes.hex")),
    (
        "pico8",
        include_str!("../../../pixelforge/palettes/pico8.hex"),
    ),
    (
        "snes15",
        include_str!("../../../pixelforge/palettes/snes15.hex"),
    ),
];

pub const PALETTE_SUFFIXES: &[&str] = &[".hex", ".gpl"];
pub const MAX_PALETTE_FILE_BYTES: u64 = 64 * 1024;
pub const INLINE_PREFIX: &str = "hex:";

/// A palette that cannot be loaded. Python raises ValueError (bad name, size, encoding or
/// content) or OSError (the file cannot be read); the message is the same `str(exc)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteError(pub String);

impl fmt::Display for PaletteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PaletteError {}

/// The bundled palette names, sorted.
pub fn bundled_palettes() -> Vec<&'static str> {
    BUNDLED.iter().map(|(name, _)| *name).collect()
}

fn bundled_text(name: &str) -> Option<&'static str> {
    BUNDLED.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// `#?[0-9a-fA-F]{6}`, the whole line.
fn is_hex_line(line: &str) -> bool {
    let digits = line.strip_prefix('#').unwrap_or(line);
    digits.len() == 6 && digits.bytes().all(|b| b.is_ascii_hexdigit())
}

fn hex_to_rgb8(line: &str) -> [u8; 3] {
    let d = line.strip_prefix('#').unwrap_or(line);
    let channel = |i: usize| u8::from_str_radix(&d[i..i + 2], 16).expect("checked hex digits");
    [channel(0), channel(2), channel(4)]
}

/// `'rrggbb'`, lowercase.
pub fn rgb8_hex(c: [u8; 3]) -> String {
    format!("{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// One `rrggbb` or `#rrggbb` per line; `;` starts a comment, lines starting with `//` are
/// skipped. Errors name the line number only, never its content.
pub fn parse_hex(text: &str) -> Result<Vec<[u8; 3]>, String> {
    let mut colors = Vec::new();
    for (i, line) in pyunicode::splitlines(text).into_iter().enumerate() {
        let line = pyunicode::strip(line.split(';').next().unwrap_or(""));
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if !is_hex_line(line) {
            return Err(format!("line {}: not a hex color", i + 1));
        }
        colors.push(hex_to_rgb8(line));
    }
    Ok(colors)
}

/// GIMP palette: every line whose first three whitespace-separated items are digit strings
/// is a colour.
pub fn parse_gpl(text: &str) -> Result<Vec<[u8; 3]>, String> {
    let mut colors = Vec::new();
    for (i, line) in pyunicode::splitlines(text).into_iter().enumerate() {
        let parts = pyunicode::split_whitespace(line);
        if parts.len() < 3
            || !parts[..3]
                .iter()
                .all(|p| p.chars().all(pyunicode::is_digit))
        {
            continue;
        }
        let mut rgb = [0u64; 3];
        for (v, p) in rgb.iter_mut().zip(&parts[..3]) {
            // int() rejects digits that are not decimal ('²'), with this message.
            *v = pyunicode::int_of_digits(p).ok_or_else(|| {
                format!("invalid literal for int() with base 10: {}", str_repr(p))
            })?;
        }
        if rgb.iter().any(|&v| v > 255) {
            return Err(format!("line {}: color channel above 255", i + 1));
        }
        colors.push([rgb[0] as u8, rgb[1] as u8, rgb[2] as u8]);
    }
    Ok(colors)
}

/// A palette_name that carries the colours themselves: `hex:rrggbb,rrggbb,...`.
pub fn inline_name(colors: &[[u8; 3]]) -> String {
    let lines: Vec<String> = colors.iter().map(|&c| rgb8_hex(c)).collect();
    format!("{INLINE_PREFIX}{}", lines.join(","))
}

/// `str(pathlib.PurePosixPath(name))` (Python 3.11): `//` kept as a root only when the path
/// starts with exactly two slashes, empty and `.` components dropped, `.` for nothing.
pub(crate) fn pathlib_str(name: &str) -> String {
    let (root, rest) = if let Some(rest) = name.strip_prefix("//") {
        if rest.starts_with('/') {
            ("/", rest)
        } else {
            ("//", rest)
        }
    } else if let Some(rest) = name.strip_prefix('/') {
        ("/", rest)
    } else {
        ("", name)
    };
    let parts: Vec<&str> = rest
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    if root.is_empty() && parts.is_empty() {
        return ".".to_string();
    }
    format!("{root}{}", parts.join("/"))
}

/// `PurePosixPath(name).suffix` (Python 3.11): the final component's text from its last
/// `.`, unless that dot is the first or the last character.
pub(crate) fn pathlib_suffix(name: &str) -> &str {
    let last = name
        .split('/')
        .rfind(|p| !p.is_empty() && *p != ".")
        .unwrap_or("");
    match last.rfind('.') {
        Some(i) if i > 0 && i < last.len() - 1 => &last[i..],
        _ => "",
    }
}

/// `str(OSError)` for a failed call on `path`: `[Errno N] strerror: 'path'`.
fn os_error(err: &std::io::Error, path: &str) -> PaletteError {
    match err.raw_os_error() {
        Some(code) => {
            let text = std::io::Error::from_raw_os_error(code).to_string();
            let suffix = format!(" (os error {code})");
            let strerror = text.strip_suffix(&suffix).unwrap_or(&text);
            PaletteError(format!("[Errno {code}] {strerror}: {}", str_repr(path)))
        }
        None => PaletteError(err.to_string()),
    }
}

#[cfg(unix)]
mod errno {
    pub const ENOTDIR: i32 = 20;
    pub const EBADF: i32 = 9;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub const ELOOP: i32 = 40;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub const ELOOP: i32 = 62;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub const O_NONBLOCK: i32 = 0o4000;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub const O_NONBLOCK: i32 = 0x0004;
}

/// `Path(path).exists()` (Python 3.11): false for a missing file, ENOTDIR, EBADF, ELOOP and
/// paths the OS cannot take (an embedded NUL); any other failure is an error.
fn path_exists(path: &str) -> Result<bool, PaletteError> {
    match std::fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) if e.kind() == std::io::ErrorKind::InvalidInput && path.contains('\0') => Ok(false),
        #[cfg(unix)]
        Err(e)
            if matches!(
                e.raw_os_error(),
                Some(errno::ENOTDIR | errno::EBADF | errno::ELOOP)
            ) =>
        {
            Ok(false)
        }
        Err(e) => Err(os_error(&e, path)),
    }
}

/// `io.open_regular`: open for reading, refusing FIFOs, devices and directories (checked on
/// the opened descriptor; `O_NONBLOCK` so opening a FIFO cannot hang).
fn open_regular(path: &str) -> Result<std::fs::File, PaletteError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(errno::O_NONBLOCK);
    }
    let not_regular = || PaletteError(format!("{path} is not a regular file"));
    let file = match options.open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::IsADirectory => return Err(not_regular()),
        Err(e) => return Err(os_error(&e, path)),
    };
    let meta = file.metadata().map_err(|e| os_error(&e, path))?;
    if !meta.is_file() {
        return Err(not_regular());
    }
    // O_NONBLOCK has no effect on reads from a regular file, so it is left set.
    Ok(file)
}

/// `_read_palette_text`: a regular file of at most 64 KiB of UTF-8 text.
fn read_palette_text(path: &str) -> Result<String, PaletteError> {
    let file = open_regular(path)?;
    let mut data = Vec::new();
    file.take(MAX_PALETTE_FILE_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|e| os_error(&e, path))?;
    if data.len() as u64 > MAX_PALETTE_FILE_BYTES {
        return Err(PaletteError(format!(
            "palette file {path} is larger than {MAX_PALETTE_FILE_BYTES} bytes"
        )));
    }
    String::from_utf8(data)
        .map_err(|_| PaletteError(format!("palette file {path} is not a UTF-8 text file")))
}

fn check_count(colors: &[[u8; 3]], what: &str) -> Result<(), PaletteError> {
    if (1..=256).contains(&colors.len()) {
        Ok(())
    } else {
        Err(PaletteError(format!(
            "{what} must have between 1 and 256 colors, got {}",
            colors.len()
        )))
    }
}

/// Load a bundled palette by name, a `.hex` / `.gpl` file by path (relative to the working
/// directory) or an inline `hex:rrggbb,rrggbb,...` palette: 1 to 256 colours.
pub fn load_palette(name: &str) -> Result<Vec<[u8; 3]>, PaletteError> {
    if let Some(inline) = name.strip_prefix(INLINE_PREFIX) {
        let text = inline.split(',').collect::<Vec<_>>().join("\n");
        let colors = parse_hex(&text).map_err(|e| PaletteError(format!("inline palette: {e}")))?;
        check_count(&colors, "inline palette")?;
        return Ok(colors);
    }
    let (path, text, gpl) = if let Some(text) = bundled_text(name) {
        (format!("{name}.hex"), text.to_string(), false)
    } else {
        let path = pathlib_str(name);
        let suffix = pathlib_suffix(name).to_lowercase();
        if !PALETTE_SUFFIXES.contains(&suffix.as_str()) || !path_exists(&path)? {
            return Err(PaletteError(format!(
                "unknown palette {}: not one of {} and not a path to a .hex or .gpl file",
                str_repr(name),
                str_list_repr(&bundled_palettes())
            )));
        }
        let text = read_palette_text(&path)?;
        (path, text, suffix == ".gpl")
    };
    let parsed = if gpl {
        parse_gpl(&text)
    } else {
        parse_hex(&text)
    };
    let colors = parsed.map_err(|e| PaletteError(format!("palette file {path}: {e}")))?;
    check_count(&colors, &format!("palette {}", str_repr(name)))?;
    Ok(colors)
}

/// The colours of a palette as lowercase `rrggbb` strings.
pub fn parse_hex_lines(name: &str) -> Result<Vec<String>, PaletteError> {
    Ok(load_palette(name)?.into_iter().map(rgb8_hex).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_matches_the_package_directory() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pixelforge/palettes");
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| {
                let name = e.unwrap().file_name().into_string().unwrap();
                name.strip_suffix(".hex").map(str::to_string)
            })
            .collect();
        names.sort();
        assert_eq!(names, bundled_palettes());
        for (name, text) in BUNDLED {
            let on_disk = std::fs::read_to_string(format!("{dir}/{name}.hex")).unwrap();
            assert_eq!(&on_disk, text, "{name}");
            let n = parse_hex_lines(name).unwrap().len();
            assert!((1..=256).contains(&n), "{name}");
        }
    }

    #[test]
    fn hex_rules() {
        let colors = parse_hex(" FF0000 ; red\n// skip\n\n#00ff00\r\n0000Ff").unwrap();
        assert_eq!(colors, vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
        assert_eq!(
            parse_hex("ff0000\nzz").unwrap_err(),
            "line 2: not a hex color"
        );
        assert_eq!(
            parse_hex("\u{feff}ff0000").unwrap_err(),
            "line 1: not a hex color"
        );
        assert_eq!(
            parse_hex("##ff0000").unwrap_err(),
            "line 1: not a hex color"
        );
    }

    #[test]
    fn gpl_rules() {
        let text = "GIMP Palette\nName: x\n#\n  1 2 3 one\n255\t255 255\n\u{663} 1 1\n";
        assert_eq!(
            parse_gpl(text).unwrap(),
            vec![[1, 2, 3], [255, 255, 255], [3, 1, 1]]
        );
        assert_eq!(
            parse_gpl("1 2 300").unwrap_err(),
            "line 1: color channel above 255"
        );
        assert_eq!(
            parse_gpl("GIMP\n\u{b2} 1 1").unwrap_err(),
            "invalid literal for int() with base 10: '\u{b2}'"
        );
    }

    #[test]
    fn inline() {
        assert_eq!(
            parse_hex_lines("hex: ff0000 ;c,//x,#00FF00").unwrap(),
            ["ff0000", "00ff00"]
        );
        assert_eq!(inline_name(&[[1, 2, 3], [255, 0, 16]]), "hex:010203,ff0010");
        let err = |n: &str| load_palette(n).unwrap_err().0;
        assert_eq!(
            err("hex:"),
            "inline palette must have between 1 and 256 colors, got 0"
        );
        assert_eq!(
            err("hex:ff0000,zz"),
            "inline palette: line 2: not a hex color"
        );
        assert_eq!(
            err("hex:ff0000\rzz"),
            "inline palette: line 2: not a hex color"
        );
        let many = format!("hex:{}", vec!["000000"; 257].join(","));
        assert_eq!(
            err(&many),
            "inline palette must have between 1 and 256 colors, got 257"
        );
    }

    #[test]
    fn pathlib_rules() {
        assert_eq!(pathlib_str(".//bad.hex"), "bad.hex");
        assert_eq!(pathlib_str("///a/./b/"), "/a/b");
        assert_eq!(pathlib_str("//a"), "//a");
        assert_eq!(pathlib_str("a//b/../c.hex"), "a/b/../c.hex");
        assert_eq!(pathlib_str(""), ".");
        assert_eq!(pathlib_suffix("x/a.HEX"), ".HEX");
        assert_eq!(pathlib_suffix(".hex"), "");
        assert_eq!(pathlib_suffix("a."), "");
        assert_eq!(pathlib_suffix("a.hex/."), ".hex");
    }

    #[test]
    fn missing_and_unusable_files() {
        let unknown = |n: &str| {
            format!(
                "unknown palette {}: not one of ['gameboy', 'genesis', 'nes', 'pico8', \
                 'snes15'] and not a path to a .hex or .gpl file",
                str_repr(n)
            )
        };
        for name in ["missing-palette-file.hex", ".hex", "a\0.hex", "x.txt"] {
            assert_eq!(load_palette(name).unwrap_err().0, unknown(name));
        }
        let dir =
            std::env::temp_dir().join(format!("pixelforge-palette-{}.hex", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        assert_eq!(
            load_palette(d).unwrap_err().0,
            format!("{d} is not a regular file")
        );
        std::fs::remove_dir(&dir).unwrap();
    }
}
