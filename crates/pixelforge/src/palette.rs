//! Hardware palettes: the bundled `.hex` palettes (embedded), the `.hex` / `.gpl` parsers,
//! inline `hex:rrggbb,...` palettes and file loading, ported from `pixelforge/palette.py`
//! with the same rules and messages; median cut, hue ramps and the palette build
//! (`median_cut`, `regularize_ramps`, `build`), formula for formula. Median cut is bitwise
//! identical to the numpy version; the ramps use `libm`'s hypot/atan2/cos/sin, which differ
//! from numpy's platform functions by at most one ulp on a few percent of arguments.

use crate::pyrepr::{str_list_repr, str_repr};
use crate::pyunicode;
use std::fmt;
use std::io::Read;

/// The bundled palettes, sorted by name (`bundled_palettes()`), embedded from
/// `crates/pixelforge/palettes/*.hex`.
pub const BUNDLED: &[(&str, &str)] = &[
    ("gameboy", include_str!("../palettes/gameboy.hex")),
    ("genesis", include_str!("../palettes/genesis.hex")),
    ("nes", include_str!("../palettes/nes.hex")),
    ("pico8", include_str!("../palettes/pico8.hex")),
    ("snes15", include_str!("../palettes/snes15.hex")),
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
    PaletteError(crate::io::os_error_message(err, path))
}

#[cfg(unix)]
mod errno {
    pub const ENOTDIR: i32 = 20;
    pub const EBADF: i32 = 9;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub const ELOOP: i32 = 40;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub const ELOOP: i32 = 62;
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

/// `io.open_regular` (see `crate::io::open_regular`) with this module's error type.
fn open_regular(path: &str) -> Result<std::fs::File, PaletteError> {
    crate::io::open_regular(std::path::Path::new(path)).map_err(|e| match e {
        crate::io::IoError::Os { error, .. } => os_error(&error, path),
        crate::io::IoError::Pixelforge(e) => PaletteError(e.0),
    })
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

// ------------------------------------------------------------------------------ clustering
//
// Ported from `pixelforge/palette.py` formula for formula: numpy's reductions through
// `npsum` (`box.mean(axis=0)` is a sequential sum per column divided by the count,
// `.sum()` over a whole contiguous array is pairwise, `np.cumsum` is sequential), stable
// sorts with numpy's comparison semantics (`-0.0 == 0.0`, NaN last), first minimum and
// maximum for `argmin` / `argmax`, so median cut is bitwise identical to Python's.

/// Boxes spanning less than this (LAB units; ΔE ≈ 2.3 is one just-noticeable difference)
/// are not split.
pub const MEDIAN_CUT_MIN_RANGE: f64 = 2.3;

pub const RAMP_BIN_DEGREES: f64 = 30.0;
pub const RAMP_MAX_SHIFT_DEGREES: f64 = 6.0;
pub const RAMP_ACHROMATIC_CHROMA: f64 = 8.0;
/// L distance from the bin's mean L at which the hue shift reaches its 6° maximum.
pub const RAMP_FULL_SHIFT_L: f64 = 30.0;
/// Blue/purple.
pub const RAMP_DARK_HUE: f64 = 270.0;
/// Yellow.
pub const RAMP_LIGHT_HUE: f64 = 90.0;

/// The comment line `write_hex` puts first for a transparent-first palette; `parse_hex`
/// skips it.
pub const TRANSPARENT_HEX_LINE: &str = "; transparent";

/// numpy's sort order for floats: NaN after everything, `-0.0 == 0.0`.
fn np_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(&b).expect("not NaN"),
    }
}

/// `box.mean(axis=0)` for an `(n, 3)` box: rows added in order per column, divided by n.
fn mean_axis0(pixels: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for p in pixels {
        for c in 0..3 {
            sum[c] += p[c];
        }
    }
    let n = pixels.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Summed squared LAB distance of a box's pixels from the box mean:
/// `((box - mean) ** 2).sum()`, pairwise over the 3n values in row-major order.
fn box_error(pixels: &[[f64; 3]]) -> f64 {
    let mean = mean_axis0(pixels);
    let sq: Vec<f64> = pixels
        .iter()
        .flat_map(|p| {
            let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
            [d[0] * d[0], d[1] * d[1], d[2] * d[2]]
        })
        .collect();
    crate::npsum::pairwise_sum(&sq)
}

/// `box.max(axis=0) - box.min(axis=0)`, NaN propagating as numpy's reductions do.
fn box_ranges(pixels: &[[f64; 3]]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let (mut lo, mut hi) = (pixels[0][c], pixels[0][c]);
        for p in &pixels[1..] {
            let v = p[c];
            if !hi.is_nan() && (v.is_nan() || v > hi) {
                hi = v;
            }
            if !lo.is_nan() && (v.is_nan() || v < lo) {
                lo = v;
            }
        }
        *o = hi - lo;
    }
    out
}

/// Index of the first maximum (`np.argmax`: the first NaN counts as the maximum).
fn argmax3(v: &[f64; 3]) -> usize {
    let mut best = 0;
    for i in 0..3 {
        if v[i].is_nan() {
            return i;
        }
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// `_split_mask`: the pixels on the lower side of the cut that minimizes the summed squared
/// error of the two halves. The cut is only placed between two different values; ties → the
/// lowest cut.
fn split_mask(values: &[f64]) -> Vec<bool> {
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| np_cmp(*a, *b));
    let n = ordered.len();
    let total = crate::npsum::cumsum(&ordered);
    let squares: Vec<f64> = ordered.iter().map(|v| v * v).collect();
    let total_sq = crate::npsum::cumsum(&squares);
    let (last, last_sq) = (total[n - 1], total_sq[n - 1]);
    let mut best = 0usize;
    let mut best_err = 0.0_f64;
    for j in 0..n - 1 {
        let below = (j + 1) as f64;
        let above = (n - (j + 1)) as f64;
        let rest = last - total[j];
        let mut error = (total_sq[j] - total[j] * total[j] / below)
            + ((last_sq - total_sq[j]) - rest * rest / above);
        if ordered[j + 1] == ordered[j] {
            error = f64::INFINITY;
        }
        // np.argmin: the first minimum, or the first NaN.
        if j == 0 || (!best_err.is_nan() && (error.is_nan() || error < best_err)) {
            best = j;
            best_err = error;
        }
    }
    let cut = ordered[best];
    values.iter().map(|&v| v <= cut).collect()
}

/// Box-splitting palette in LAB (`palette.median_cut`): at most `k` colours (fewer if the
/// input has fewer). The box with the largest summed squared error (ties → the earliest) is
/// split along its axis of largest range at the error-minimizing cut; boxes whose ranges are
/// all at most [`MEDIAN_CUT_MIN_RANGE`] are not split. The colours are the box means, sorted
/// by L, then a, then b (`np.lexsort((b, a, L))`, stable).
pub fn median_cut(lab_pixels: &[[f64; 3]], k: i64) -> Result<Vec<[f64; 3]>, PaletteError> {
    if lab_pixels.is_empty() {
        return Err(PaletteError(
            "median_cut needs at least one pixel".to_string(),
        ));
    }
    let mut boxes: Vec<Vec<[f64; 3]>> = vec![lab_pixels.to_vec()];
    let mut errors = vec![box_error(lab_pixels)];
    let mut ranges = vec![box_ranges(lab_pixels)];
    while (boxes.len() as i64) < k {
        let mut order: Vec<usize> = (0..boxes.len()).collect();
        // sorted(range(len(boxes)), key=lambda i: (-errors[i], i))
        order.sort_by(|&i, &j| np_cmp(-errors[i], -errors[j]).then(i.cmp(&j)));
        let chosen = order.into_iter().find_map(|i| {
            let axis = argmax3(&ranges[i]);
            (ranges[i][axis] > MEDIAN_CUT_MIN_RANGE).then_some((i, axis))
        });
        let Some((i, axis)) = chosen else {
            break;
        };
        let values: Vec<f64> = boxes[i].iter().map(|p| p[axis]).collect();
        let lower = split_mask(&values);
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for (p, &l) in boxes[i].iter().zip(&lower) {
            if l {
                lo.push(*p);
            } else {
                hi.push(*p);
            }
        }
        errors.splice(i..=i, [box_error(&lo), box_error(&hi)]);
        ranges.splice(i..=i, [box_ranges(&lo), box_ranges(&hi)]);
        boxes.splice(i..=i, [lo, hi]);
    }
    let mut palette: Vec<[f64; 3]> = boxes.iter().map(|b| mean_axis0(b)).collect();
    palette.sort_by(|p, q| {
        np_cmp(p[0], q[0])
            .then(np_cmp(p[1], q[1]))
            .then(np_cmp(p[2], q[2]))
    });
    Ok(palette)
}

/// Python's float `%` (numpy's `remainder`): the result has the divisor's sign.
fn py_mod(x: f64, y: f64) -> f64 {
    let m = x % y;
    if m != 0.0 {
        if (y < 0.0) != (m < 0.0) {
            m + y
        } else {
            m
        }
    } else {
        0.0_f64.copysign(y)
    }
}

/// `np.minimum`: NaN propagates.
fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || a <= b {
        a
    } else {
        b
    }
}

/// `np.sign`.
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// Hue-shifted ramps (Section 8.1, `palette.regularize_ramps`): colours are binned by hue
/// angle (30° bins, achromatic ones separately); within a bin a colour's hue moves toward
/// 270° if darker than the bin's mean L and toward 90° if lighter, by
/// 6°·min(1, |L − mean L| / 30), never past the target. L and chroma are kept. Every
/// operation is numpy's; only `libm`'s transcendentals (Decision 6) can move the last bit of
/// a moved colour's a and b.
pub fn regularize_ramps(palette: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let deg = 180.0 / std::f64::consts::PI; // npy_rad2deg
    let rad = std::f64::consts::PI / 180.0; // npy_deg2rad
    let chroma: Vec<f64> = palette.iter().map(|p| libm::hypot(p[1], p[2])).collect();
    let hue: Vec<f64> = palette
        .iter()
        .map(|p| py_mod(libm::atan2(p[2], p[1]) * deg, 360.0))
        .collect();
    let bins: Vec<i64> = hue
        .iter()
        .zip(&chroma)
        .map(|(&h, &c)| {
            if c < RAMP_ACHROMATIC_CHROMA {
                -1
            } else {
                (h / RAMP_BIN_DEGREES).floor() as i64
            }
        })
        .collect();
    let mut unique = bins.clone();
    unique.sort_unstable();
    unique.dedup();
    let mut new_hue = hue.clone();
    for bin in unique {
        let members: Vec<usize> = (0..palette.len()).filter(|&i| bins[i] == bin).collect();
        let ls: Vec<f64> = members.iter().map(|&i| palette[i][0]).collect();
        let mean = crate::npsum::pairwise_mean(&ls);
        for &i in &members {
            let offset = palette[i][0] - mean;
            let amount = RAMP_MAX_SHIFT_DEGREES * np_min(1.0, offset.abs() / RAMP_FULL_SHIFT_L);
            let target = if offset < 0.0 {
                RAMP_DARK_HUE
            } else {
                RAMP_LIGHT_HUE
            };
            // Signed shortest rotation to the target in (-180, 180]; +180 when opposite.
            let mut to_target = py_mod(target - hue[i] + 180.0, 360.0) - 180.0;
            if to_target == -180.0 {
                to_target = 180.0;
            }
            new_hue[i] = hue[i] + np_sign(to_target) * np_min(amount, to_target.abs());
        }
    }
    palette
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if new_hue[i] != hue[i] {
                let r = new_hue[i] * rad;
                [p[0], chroma[i] * libm::cos(r), chroma[i] * libm::sin(r)]
            } else {
                *p
            }
        })
        .collect()
}

/// `palette.resolve_source`: `palette_source`, with "auto" meaning mcda for gerstner and
/// median_cut otherwise.
pub fn resolve_source(config: &crate::config::Config) -> &str {
    if config.palette_source != "auto" {
        return &config.palette_source;
    }
    if config.method == "gerstner" {
        "mcda"
    } else {
        "median_cut"
    }
}

/// The LAB colours of `palette_name`, from the Config's cached hex lines, or `None` without
/// a palette name.
pub fn build_named(config: &crate::config::Config) -> Result<Option<Vec<[f64; 3]>>, PaletteError> {
    let Some(lines) = config.palette_hex_lines()? else {
        return Ok(None);
    };
    let mut rgb8 = Vec::with_capacity(lines.len() * 3);
    for line in &lines {
        rgb8.extend_from_slice(&hex_to_rgb8(line));
    }
    let lab = crate::color::rgb8_to_lab(&rgb8);
    Ok(Some(
        lab.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
    ))
}

/// The palette half of Section 7 applied directly to pixel colors (no superpixels):
/// `palette.mcda`. The annealer runs on the pixels with the uniform prior 1/n for at most
/// `g_max_iters` steps. Error for no pixels or `k` < 1.
pub fn mcda(
    lab_pixels: &[[f64; 3]],
    k: i64,
    config: &crate::config::Config,
) -> Result<Vec<[f64; 3]>, PaletteError> {
    if lab_pixels.is_empty() {
        return Err(PaletteError("mcda needs at least one pixel".to_string()));
    }
    let k = usize::try_from(k)
        .ok()
        .filter(|&k| k >= 1)
        .ok_or_else(|| PaletteError(format!("mcda needs K >= 1, got {k}")))?;
    let mut annealer = crate::annealer::PaletteAnnealer::new(lab_pixels, k, config);
    let weights = vec![1.0 / lab_pixels.len() as f64; lab_pixels.len()];
    for _ in 0..config.g_max_iters.max(0) {
        annealer.step(lab_pixels, &weights);
        if annealer.done {
            break;
        }
    }
    Ok(annealer.colors())
}

/// Palette `(K, 3)` in LAB for the given pixels according to the config: the named palette,
/// else mcda or median cut (`resolve_source`).
pub fn build(
    lab_pixels: &[[f64; 3]],
    config: &crate::config::Config,
) -> Result<Vec<[f64; 3]>, PaletteError> {
    if let Some(named) = build_named(config)? {
        return Ok(named);
    }
    if resolve_source(config) == "mcda" {
        return mcda(lab_pixels, config.palette_size, config);
    }
    median_cut(lab_pixels, config.palette_size)
}

/// The text `write_hex` writes: one `rrggbb` per line, with a leading
/// [`TRANSPARENT_HEX_LINE`] when `transparent_first`, and a final newline.
pub fn hex_text(colors: &[[u8; 3]], transparent_first: bool) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(colors.len() + 1);
    if transparent_first {
        lines.push(TRANSPARENT_HEX_LINE.to_string());
    }
    lines.extend(colors.iter().map(|&c| rgb8_hex(c)));
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_matches_the_package_directory() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/palettes");
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

    #[test]
    fn median_cut_hand_checked() {
        assert_eq!(
            median_cut(&[], 4).unwrap_err().0,
            "median_cut needs at least one pixel"
        );
        // L = 0, 10, 10, 50: the cut between the two 10s is not allowed (inf); the cut above
        // them leaves 66.7 of squared error against 1066.7 below 0.
        let px = [
            [50.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ];
        let pal = median_cut(&px, 2).unwrap();
        assert_eq!(pal, vec![[20.0 / 3.0, 0.0, 0.0], [50.0, 0.0, 0.0]]);
        // All ranges at most 2.3: one box, the mean, however large k is.
        let pal = median_cut(&[[0.0, 0.0, 0.0], [1.0, 2.3, 0.0]], 8).unwrap();
        assert_eq!(pal, vec![[0.5, 1.15, 0.0]]);
        // k = 1 and k = 0 keep the single box.
        assert_eq!(median_cut(&px, 1).unwrap().len(), 1);
        assert_eq!(median_cut(&px, 0).unwrap().len(), 1);
        // Equal L: ordered by a, then b (np.lexsort((b, a, L))).
        let px = [[50.0, 30.0, 0.0], [50.0, -30.0, 9.0], [50.0, -30.0, -9.0]];
        let pal = median_cut(&px, 3).unwrap();
        assert_eq!(
            pal,
            vec![[50.0, -30.0, -9.0], [50.0, -30.0, 9.0], [50.0, 30.0, 0.0]]
        );
    }

    #[test]
    fn regularize_ramps_hand_checked() {
        let (cos6, sin6) = (0.9945218953682733, 0.10452846326765347);
        let palette = [
            [30.0, 20.0, 0.0],  // hue 0, darker than the bin mean 60: toward 270, -6 degrees
            [90.0, 20.0, 0.0],  // hue 0, lighter: toward 90, +6 degrees
            [50.0, 1.0, 1.0],   // achromatic, alone in its bin: unchanged
            [40.0, 0.0, -30.0], // hue 270, alone in its bin: unchanged
        ];
        let out = regularize_ramps(&palette);
        let close = |x: f64, y: f64| (x - y).abs() < 1e-12;
        assert_eq!(out[0][0], 30.0);
        assert!(
            close(out[0][1], 20.0 * cos6) && close(out[0][2], -20.0 * sin6),
            "{out:?}"
        );
        assert!(
            close(out[1][1], 20.0 * cos6) && close(out[1][2], 20.0 * sin6),
            "{out:?}"
        );
        assert_eq!(out[2], palette[2]);
        assert_eq!(out[3], palette[3]);
    }

    #[test]
    fn python_modulo_has_the_divisor_sign() {
        assert_eq!(py_mod(-90.0, 360.0), 270.0);
        assert_eq!(py_mod(-1e-20, 360.0), 360.0);
        assert_eq!(py_mod(720.0, 360.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(py_mod(-720.0, 360.0).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn hex_text_lines() {
        let colors = [[0, 255, 16], [171, 205, 239]];
        assert_eq!(hex_text(&colors, false), "00ff10\nabcdef\n");
        assert_eq!(hex_text(&colors, true), "; transparent\n00ff10\nabcdef\n");
        assert_eq!(hex_text(&[], false), "\n");
    }
}
