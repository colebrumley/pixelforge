//! Image input and output, ported from `pixelforge/io.py`: the regular-file open, the input
//! hash, decoding the six accepted formats to 8-bit RGBA exactly as Pillow 12 does
//! (`Image.open` + `ImageOps.exif_transpose` + `_to_8bit` + `convert("RGBA")`), the
//! in-memory array inputs, and the palette-indexed PNG writer with its text chunks.
//!
//! # Decoding
//!
//! [`load`] opens the file once ([`open_regular`]), reads it, hashes the bytes and decodes
//! the same bytes ([`decode_bytes`]). The format comes from the content, with Pillow's
//! signature tests; anything else is "not a supported image". Each format is read in two
//! steps: the header (failures here are "not a supported image", as when no Pillow plugin
//! accepts the file), then the pixel budget ([`check_budget`], after Pillow's own
//! decompression-bomb limit), then the pixels (failures are "could not decode image").
//! No pixel is decoded before the budget check, and the decoders' own allocation limits are
//! switched off: the budget is the only limit.
//!
//! - PNG: chunks are walked here (Pillow's rules for `tRNS`, text, `eXIf`, `acTL`/`fcTL`);
//!   the `png` crate inflates and unfilters the samples untransformed, and the conversion to
//!   RGBA reproduces Pillow's modes: 1/2/4-bit grey scaled to 8 bits, the grey and RGB
//!   colour keys compared as Pillow compares them (the raw key against the 8-bit value),
//!   16-bit grey rescaled by `_to_8bit` (its colour key dropped, as `Image.fromarray`
//!   drops it), 16-bit RGB, RGBA and grey+alpha reduced to their high byte (Pillow's
//!   `RGB;16B`, `RGBA;16B`, `LA;16B` unpackers), palettes with per-entry tRNS alpha.
//! - GIF: the `gif` crate decodes the first frame's indices; the canvas is the logical
//!   screen (grown to the frame, as Pillow grows it), filled with the transparent index or
//!   0, and the palette lookup and transparency follow Pillow's "P"/"L" handling.
//! - TIFF: the `tiff` crate decodes the first image's samples; Pillow's mode table decides
//!   how they become RGBA (palette TIFFs are read as grey indices and mapped through the
//!   `ColorMap` here, since the crate does not expand palettes).
//! - JPEG, WEBP, BMP: the `image` crate. JPEG and lossy WEBP are not bit-identical to
//!   libjpeg/libwebp (docs/rust-migration-plan.md, Section 6); everything else is.
//!
//! Palette lookups out of the palette's range give black, as in Pillow's "P" images.
//! EXIF orientation is read as `Image.getexif()` reads it (IFD0 tag 274 of the `eXIf` /
//! APP1 / `EXIF` chunk / TIFF IFD0, else the `tiff:Orientation` of the XMP packet) and
//! applied as `exif_transpose` applies it; values outside 2..=8 change nothing. Frames are
//! counted from headers only (PNG `acTL`, GIF image descriptors, WEBP `ANMF` chunks, TIFF
//! IFD chain, JPEG MPF index); a multi-frame input yields a warning text.
//!
//! # Writing
//!
//! [`save_png`] writes an index image as Pillow did: palette-indexed with `PLTE` (and a
//! `tRNS` whose last entry is the transparent one) when the palette fits in 256 entries,
//! else RGBA; text chunks sorted by key as `tEXt` (`iTXt` for values outside Latin-1). The
//! bytes differ from Pillow's (bit depth 8 always, another deflate); the decoded image,
//! palette, transparency and text are the same.

use std::fmt;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::mask::PixelforgeError;
use crate::pyrepr::str_repr;

/// PNG text key of the canonical config JSON.
pub const META_CONFIG_KEY: &str = "pixelforge:config";
/// PNG text key of the input's SHA-256.
pub const META_INPUT_KEY: &str = "pixelforge:input_sha256";
/// PNG text key of the palette file's SHA-256 (only when `palette_name` is set).
pub const META_PALETTE_SHA_KEY: &str = "pixelforge:palette_sha256";
/// PNG text key of the resolved palette, `rrggbb` lines joined by newlines.
pub const META_PALETTE_KEY: &str = "pixelforge:palette";

/// Default input pixel budget (width × height), checked before any pixel data is decoded.
pub const MAX_INPUT_PIXELS: u64 = 24_000_000;
/// The only formats [`load`] accepts, in Pillow's names.
pub const INPUT_FORMATS: [&str; 6] = ["PNG", "JPEG", "GIF", "WEBP", "BMP", "TIFF"];
/// Pillow's `Image.MAX_IMAGE_PIXELS`. Python turned its DecompressionBombWarning (above this)
/// and DecompressionBombError (above twice this) into errors, before its own budget check.
pub const PIL_MAX_IMAGE_PIXELS: u64 = 89_478_485;
/// Pillow's `PngImagePlugin.MAX_TEXT_CHUNK`: the largest decompressed zTXt / iTXt value.
pub const MAX_TEXT_CHUNK: usize = 1024 * 1024;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

// ---------------------------------------------------------------------------------------
// Errors

/// An I/O failure: an operating-system error on `path` (Python's `OSError`, raised with its
/// errno and the path), or a `PixelforgeError` with Python's message.
#[derive(Debug)]
pub enum IoError {
    Os { error: std::io::Error, path: String },
    Pixelforge(PixelforgeError),
}

impl IoError {
    fn os(error: std::io::Error, path: &Path) -> IoError {
        IoError::Os {
            error,
            path: path.display().to_string(),
        }
    }
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Os { error, path } => f.write_str(&os_error_message(error, path)),
            IoError::Pixelforge(e) => f.write_str(&e.0),
        }
    }
}

impl std::error::Error for IoError {}

impl From<PixelforgeError> for IoError {
    fn from(e: PixelforgeError) -> Self {
        IoError::Pixelforge(e)
    }
}

/// `str(OSError)` for a failure on `path`: `[Errno N] strerror: 'path'`, or the error's own
/// text when it carries no errno.
pub fn os_error_message(err: &std::io::Error, path: &str) -> String {
    match err.raw_os_error() {
        Some(code) => {
            let text = std::io::Error::from_raw_os_error(code).to_string();
            let suffix = format!(" (os error {code})");
            let strerror = text.strip_suffix(&suffix).unwrap_or(&text);
            format!("[Errno {code}] {strerror}: {}", str_repr(path))
        }
        None => err.to_string(),
    }
}

fn pf(message: String) -> PixelforgeError {
    PixelforgeError(message)
}

// ---------------------------------------------------------------------------------------
// Files and hashes

#[cfg(unix)]
mod flags {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub const O_NONBLOCK: i32 = 0o4000;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub const O_NONBLOCK: i32 = 0x0004;
}

/// `io.open_regular`: open for reading, refusing FIFOs, devices and directories.
///
/// `O_NONBLOCK` (Unix) keeps the open itself from hanging on a FIFO; the check runs on the
/// opened descriptor, so the file cannot be swapped between check and read. The flag has no
/// effect on reads from a regular file and is left set.
pub fn open_regular(path: &Path) -> Result<File, IoError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(flags::O_NONBLOCK);
    }
    let not_regular =
        || IoError::Pixelforge(pf(format!("{} is not a regular file", path.display())));
    let file = match options.open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::IsADirectory => return Err(not_regular()),
        Err(e) => return Err(IoError::os(e, path)),
    };
    let meta = file.metadata().map_err(|e| IoError::os(e, path))?;
    if !meta.is_file() {
        return Err(not_regular());
    }
    Ok(file)
}

fn read_regular(path: &Path) -> Result<Vec<u8>, IoError> {
    let mut file = open_regular(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| IoError::os(e, path))?;
    Ok(bytes)
}

fn hex_digest(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `io.sha256_file`: the SHA-256 of a regular file's bytes, as lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String, IoError> {
    let mut file = open_regular(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).map_err(|e| IoError::os(e, path))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_digest(hasher))
}

/// The provenance hash of an in-memory RGBA8 image: SHA-256 of `rgba8 HxW\n` and the bytes.
pub fn rgba8_digest(h: usize, w: usize, rgba: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("rgba8 {h}x{w}\n").as_bytes());
    hasher.update(rgba);
    hex_digest(hasher)
}

// ---------------------------------------------------------------------------------------
// Loaded images

/// A decoded input (`io.Loaded`): `rgb` is `h * w * 3` sRGB in [0, 1] (`v / 255`), `alpha`
/// is `h * w`, `sha256` the input file's hash (or [`rgba8_digest`] for arrays). `warnings`
/// holds the texts Python emitted as `UserWarning` (the multi-frame notice).
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub rgb: Vec<f64>,
    pub alpha: Vec<u8>,
    pub h: usize,
    pub w: usize,
    pub sha256: String,
    pub warnings: Vec<String>,
}

impl Loaded {
    /// `Loaded.alpha_mask`: alpha as a float mask in [0, 1].
    pub fn alpha_mask(&self) -> Vec<f64> {
        self.alpha.iter().map(|&a| f64::from(a) / 255.0).collect()
    }
}

/// `io.from_rgba`: an `h * w * 4` RGBA8 buffer as [`Loaded`].
pub fn from_rgba(rgba: &[u8], h: usize, w: usize, sha256: String) -> Loaded {
    assert_eq!(
        rgba.len(),
        h * w * 4,
        "from_rgba: rgba must be h * w * 4 bytes"
    );
    let mut rgb = Vec::with_capacity(h * w * 3);
    let mut alpha = Vec::with_capacity(h * w);
    for px in rgba.chunks_exact(4) {
        rgb.push(f64::from(px[0]) / 255.0);
        rgb.push(f64::from(px[1]) / 255.0);
        rgb.push(f64::from(px[2]) / 255.0);
        alpha.push(px[3]);
    }
    Loaded {
        rgb,
        alpha,
        h,
        w,
        sha256,
        warnings: Vec::new(),
    }
}

/// `_check_budget`: `PixelforgeError` when `width * height` exceeds `max_pixels`.
pub fn check_budget(
    name: &str,
    width: u64,
    height: u64,
    max_pixels: u64,
) -> Result<(), PixelforgeError> {
    let pixels = u128::from(width) * u128::from(height);
    if pixels > u128::from(max_pixels) {
        return Err(pf(format!(
            "{name}: {width}x{height} = {pixels} pixels exceeds the input budget of \
             {max_pixels} pixels (--max-input-pixels)"
        )));
    }
    Ok(())
}

/// Pillow's `_decompression_bomb_check`, which Python's `load` turned into an error.
fn check_bomb(name: &str, width: u64, height: u64) -> Result<(), PixelforgeError> {
    let pixels = u128::from(width.max(1)) * u128::from(height.max(1));
    let limit = u128::from(PIL_MAX_IMAGE_PIXELS);
    if pixels > 2 * limit {
        return Err(pf(format!(
            "{name}: Image size ({pixels} pixels) exceeds limit of {} pixels, could be \
             decompression bomb DOS attack.",
            2 * limit
        )));
    }
    if pixels > limit {
        return Err(pf(format!(
            "{name}: Image size ({pixels} pixels) exceeds limit of {limit} pixels, could be \
             decompression bomb DOS attack."
        )));
    }
    Ok(())
}

/// `io.load`: decode an image file to RGBA (see the module documentation).
///
/// Errors: `IoError::Os` when the file cannot be opened or read; `PixelforgeError` for a
/// non-regular file, unsupported or undecodable data, and above `max_pixels` pixels.
pub fn load(path: &Path, max_pixels: u64) -> Result<Loaded, IoError> {
    let bytes = read_regular(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hex_digest(hasher);
    let name = path.display().to_string();
    let decoded = decode_bytes(&name, &bytes, max_pixels)?;
    let mut loaded = from_rgba(&decoded.rgba, decoded.h, decoded.w, digest);
    loaded.warnings = decoded.warnings;
    Ok(loaded)
}

/// An image decoded to `h * w * 4` RGBA8, with the warning texts of the decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8 {
    pub h: usize,
    pub w: usize,
    pub rgba: Vec<u8>,
    pub warnings: Vec<String>,
}

/// The in-memory array input of [`from_image_array`]: its elements in C order, uint8 or
/// float (converted to float64), or the name of any other dtype (an error).
pub enum ArrayData<'a> {
    U8(&'a [u8]),
    F64(&'a [f64]),
    Other(&'a str),
}

/// Python's tuple repr of a shape: `(2, 4)`, `(5,)`, `()`.
fn shape_repr(shape: &[usize]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => {
            let parts: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    }
}

/// `io.from_image` for arrays: `(H, W, 3)` or `(H, W, 4)`, uint8 or float in [0, 1]
/// (`round_ties_even(x * 255)`), alpha 255 for three channels; the hash is
/// [`rgba8_digest`]. Errors with Python's messages, checked in Python's order (shape,
/// budget, dtype and range).
pub fn from_image_array(
    shape: &[usize],
    data: ArrayData<'_>,
    max_pixels: u64,
) -> Result<Loaded, PixelforgeError> {
    if shape.len() != 3 || !(shape[2] == 3 || shape[2] == 4) {
        return Err(pf(format!(
            "image array must have shape (H, W, 3) or (H, W, 4), got {}",
            shape_repr(shape)
        )));
    }
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    check_budget("image", w as u64, h as u64, max_pixels)?;
    let n = h * w * c;
    let values: Vec<u8> = match data {
        ArrayData::U8(v) => {
            assert_eq!(
                v.len(),
                n,
                "from_image_array: data does not match the shape"
            );
            v.to_vec()
        }
        ArrayData::F64(v) => {
            assert_eq!(
                v.len(),
                n,
                "from_image_array: data does not match the shape"
            );
            if !v.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0) {
                return Err(pf(
                    "float image arrays must be finite and in [0, 1]".to_string()
                ));
            }
            v.iter()
                .map(|&x| (x * 255.0).round_ties_even() as u8)
                .collect()
        }
        ArrayData::Other(dtype) => {
            return Err(pf(format!(
                "image array must be uint8 or float, got {dtype}"
            )));
        }
    };
    let rgba = if c == 4 {
        values
    } else {
        let mut out = Vec::with_capacity(h * w * 4);
        for px in values.chunks_exact(3) {
            out.extend_from_slice(px);
            out.push(255);
        }
        out
    };
    let digest = rgba8_digest(h, w, &rgba);
    Ok(from_rgba(&rgba, h, w, digest))
}

// ---------------------------------------------------------------------------------------
// Decoding

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Png,
    Jpeg,
    Gif,
    Webp,
    Bmp,
    Tiff,
}

/// Pillow's `_accept` signature tests of the six plugins.
fn detect(bytes: &[u8]) -> Option<Format> {
    const TIFF_PREFIXES: [&[u8]; 6] = [
        b"MM\x00\x2a",
        b"II\x2a\x00",
        b"MM\x2a\x00",
        b"II\x00\x2a",
        b"MM\x00\x2b",
        b"II\x2b\x00",
    ];
    if bytes.starts_with(PNG_SIGNATURE) {
        Some(Format::Png)
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some(Format::Jpeg)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some(Format::Gif)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some(Format::Webp)
    } else if bytes.starts_with(b"BM") {
        Some(Format::Bmp)
    } else if TIFF_PREFIXES.iter().any(|p| bytes.starts_with(p)) {
        Some(Format::Tiff)
    } else {
        None
    }
}

/// Why a decode failed: no plugin accepted the header, the size check, or the pixel data.
enum Fail {
    Unidentified,
    Check(PixelforgeError),
    Decode(String),
}

/// The size check run between the header and the pixels: Pillow's bomb limit, then the
/// budget.
type SizeCheck<'a> = dyn FnMut(u64, u64) -> Result<(), PixelforgeError> + 'a;

/// A decoded first frame before orientation: RGBA8, the frame count and the metadata the
/// orientation comes from.
struct Raw {
    w: usize,
    h: usize,
    rgba: Vec<u8>,
    frames: u64,
    meta: Meta,
}

/// Where `getexif()` finds the orientation: EXIF bytes (TIFF structure, optionally
/// prefixed with `Exif\0\0`), an IFD0 offset into the file itself (TIFF), and the XMP
/// fallbacks (`info["XML:com.adobe.xmp"]` text first, then `info["xmp"]` bytes).
#[derive(Default)]
struct Meta {
    exif: Option<Vec<u8>>,
    tiff_ifd0: Option<u64>,
    xmp_text: Option<String>,
    xmp_bytes: Option<Vec<u8>>,
}

/// Decode an image's bytes (the part of [`load`] after reading the file). `name` is used
/// in the messages and the warning texts.
pub fn decode_bytes(name: &str, bytes: &[u8], max_pixels: u64) -> Result<Rgba8, PixelforgeError> {
    let unsupported = || {
        pf(format!(
            "{name}: not a supported image (expected one of {})",
            INPUT_FORMATS.join(", ")
        ))
    };
    let Some(format) = detect(bytes) else {
        return Err(unsupported());
    };
    let mut check = |w: u64, h: u64| -> Result<(), PixelforgeError> {
        check_bomb(name, w, h)?;
        check_budget(name, w, h, max_pixels)
    };
    let result = match format {
        Format::Png => read_png(bytes, &mut check),
        Format::Gif => read_gif(bytes, &mut check),
        Format::Tiff => read_tiff(bytes, &mut check),
        Format::Jpeg => read_jpeg(bytes, &mut check),
        Format::Webp => read_webp(bytes, &mut check),
        Format::Bmp => read_bmp(bytes, &mut check),
    };
    let decode_error = |reason: String| pf(format!("{name}: could not decode image ({reason})"));
    let raw = match result {
        Ok(raw) => raw,
        Err(Fail::Unidentified) => return Err(unsupported()),
        Err(Fail::Check(e)) => return Err(e),
        Err(Fail::Decode(reason)) => return Err(decode_error(reason)),
    };
    let mut warnings = Vec::new();
    if raw.frames > 1 {
        warnings.push(format!("{name} has {} frames; using the first", raw.frames));
    }
    let orientation = orientation(bytes, &raw.meta).map_err(decode_error)?;
    let (h, w, rgba) = apply_orientation(raw.h, raw.w, &raw.rgba, orientation);
    Ok(Rgba8 {
        h,
        w,
        rgba,
        warnings,
    })
}

/// A palette entry; an index outside the palette is black, as in Pillow's "P" images.
fn palette_rgb(palette: &[u8], index: usize) -> [u8; 3] {
    match palette.get(index * 3..index * 3 + 3) {
        Some(c) => [c[0], c[1], c[2]],
        None => [0, 0, 0],
    }
}

/// `_to_8bit`: `round_ties_even(clip(v, 0, 65535) / 257)`.
fn to_8bit(v: i64) -> u8 {
    let clipped = v.clamp(0, 65535) as f64;
    (clipped / 257.0).round_ties_even() as u8
}

/// The samples of one packed row of `bits`-bit samples (1, 2, 4 or 8).
fn unpack_row(row: &[u8], bits: u8, count: usize) -> Vec<u8> {
    if bits == 8 {
        return row[..count].to_vec();
    }
    let per_byte = 8 / bits as usize;
    let mask = (1u16 << bits) as u8 - 1;
    (0..count)
        .map(|i| {
            let byte = row[i / per_byte];
            let shift = 8 - bits as usize * (i % per_byte + 1);
            (byte >> shift) & mask
        })
        .collect()
}

/// A grey sample of `bits` bits scaled to 8 bits as Pillow's `1`, `L;2`, `L;4` unpackers do.
fn scale_grey(v: u8, bits: u8) -> u8 {
    match bits {
        1 => {
            if v != 0 {
                255
            } else {
                0
            }
        }
        2 => v * 85,
        4 => v * 17,
        _ => v,
    }
}

fn be16(b: &[u8], i: usize) -> u16 {
    u16::from_be_bytes([b[i], b[i + 1]])
}

// --- PNG -------------------------------------------------------------------------------

/// The PNG chunks Pillow reads, with its rules: up to the first IDAT the chunk headers and
/// CRCs must be sound (else the file is not identified); after the image data the walk is
/// lenient and stops at IEND, at a broken chunk header or, for APNG metadata, at the next
/// `fcTL` (`stop_at_fctl`).
struct PngChunk<'a> {
    kind: [u8; 4],
    data: &'a [u8],
}

struct PngWalk<'a> {
    before_idat: Vec<PngChunk<'a>>,
    after_idat: Vec<PngChunk<'a>>,
    /// The after-IDAT walk ended inside a chunk's data (Pillow's "Truncated File Read").
    truncated_after: bool,
}

fn crc32(kind: &[u8], data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in kind.iter().chain(data) {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Before the image data: a short or invalid chunk header and a bad or missing CRC are
/// Pillow's SyntaxError / struct.error (no plugin accepts the file); chunk data cut short
/// is its OSError "Truncated File Read" (the file is accepted but cannot be decoded).
fn walk_png(bytes: &[u8], stop_at_fctl: bool) -> Result<PngWalk<'_>, Fail> {
    let mut pos = PNG_SIGNATURE.len();
    let header = |pos: usize| -> Option<(usize, [u8; 4])> {
        let h = bytes.get(pos..pos + 8)?;
        let len = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as usize;
        let kind = [h[4], h[5], h[6], h[7]];
        Some((len, kind))
    };
    let is_cid = |k: &[u8; 4]| k.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_');
    let mut before = Vec::new();
    loop {
        let (len, kind) = header(pos).ok_or(Fail::Unidentified)?;
        if !is_cid(&kind) {
            return Err(Fail::Unidentified);
        }
        let data_start = pos + 8;
        if &kind == b"IDAT" {
            // Pillow stops at the first IDAT without reading its CRC.
            break;
        }
        if &kind == b"IEND" {
            return Err(Fail::Unidentified);
        }
        let data = bytes
            .get(data_start..data_start.saturating_add(len))
            .ok_or_else(|| Fail::Decode("Truncated File Read".to_string()))?;
        let crc = bytes
            .get(data_start + len..data_start + len + 4)
            .ok_or(Fail::Unidentified)?;
        if u32::from_be_bytes([crc[0], crc[1], crc[2], crc[3]]) != crc32(&kind, data) {
            return Err(Fail::Unidentified);
        }
        before.push(PngChunk { kind, data });
        pos = data_start + len + 4;
    }
    // Skip the image data: the IDAT run.
    while let Some((len, kind)) = header(pos) {
        if &kind != b"IDAT" {
            break;
        }
        pos += 12 + len;
    }
    let mut after = Vec::new();
    let mut truncated_after = false;
    while let Some((len, kind)) = header(pos) {
        if !is_cid(&kind) || &kind == b"IEND" || (stop_at_fctl && &kind == b"fcTL") {
            break;
        }
        let data_start = pos + 8;
        match bytes.get(data_start..data_start.saturating_add(len)) {
            Some(data) => after.push(PngChunk { kind, data }),
            None => {
                truncated_after = true;
                break;
            }
        }
        pos = data_start + len + 4;
    }
    Ok(PngWalk {
        before_idat: before,
        after_idat: after,
        truncated_after,
    })
}

/// zlib-inflate a text value as `_safe_zlib_decompress`: at most [`MAX_TEXT_CHUNK`] bytes
/// (more is Pillow's ValueError), an incomplete stream yields what it decoded, corrupt data
/// is `None` (Python's `zlib.error`).
fn inflate_text(data: &[u8]) -> Result<Option<Vec<u8>>, String> {
    use miniz_oxide::inflate::{decompress_to_vec_zlib_with_limit, TINFLStatus};
    match decompress_to_vec_zlib_with_limit(data, MAX_TEXT_CHUNK) {
        Ok(v) => Ok(Some(v)),
        Err(e) => match e.status {
            TINFLStatus::HasMoreOutput => {
                Err("Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK".to_string())
            }
            TINFLStatus::FailedCannotMakeProgress => Ok(Some(e.output)),
            _ => Ok(None),
        },
    }
}

fn latin1(b: &[u8]) -> String {
    b.iter().map(|&c| char::from(c)).collect()
}

/// One text chunk as Pillow's `chunk_tEXt` / `chunk_zTXt` / `chunk_iTXt` reads it:
/// `(key, value)` or nothing; for `tEXt` the raw value bytes too (Pillow keeps
/// `info["exif"]` as bytes); for `iTXt` the raw value bytes of an XMP packet.
struct TextChunk {
    key: String,
    value: String,
    raw: Option<Vec<u8>>,
}

fn parse_text_chunk(chunk: &PngChunk<'_>) -> Result<Option<TextChunk>, String> {
    let split = |s: &[u8]| -> (Vec<u8>, Vec<u8>) {
        match s.iter().position(|&b| b == 0) {
            Some(i) => (s[..i].to_vec(), s[i + 1..].to_vec()),
            None => (s.to_vec(), Vec::new()),
        }
    };
    match &chunk.kind {
        b"tEXt" => {
            let (k, v) = split(chunk.data);
            if k.is_empty() {
                return Ok(None);
            }
            Ok(Some(TextChunk {
                key: latin1(&k),
                value: latin1(&v),
                raw: Some(v),
            }))
        }
        b"zTXt" => {
            let (k, v) = split(chunk.data);
            let method = v.first().copied().unwrap_or(0);
            if method != 0 {
                // Pillow raises SyntaxError here; treated as an undecodable file.
                return Err(format!("Unknown compression method {method} in zTXt chunk"));
            }
            let text = if v.is_empty() {
                Vec::new()
            } else {
                inflate_text(&v[1..])?.unwrap_or_default()
            };
            if k.is_empty() {
                return Ok(None);
            }
            Ok(Some(TextChunk {
                key: latin1(&k),
                value: latin1(&text),
                raw: None,
            }))
        }
        b"iTXt" => {
            let Some(i) = chunk.data.iter().position(|&b| b == 0) else {
                return Ok(None);
            };
            let k = &chunk.data[..i];
            let r = &chunk.data[i + 1..];
            if r.len() < 2 {
                return Ok(None);
            }
            let (cf, cm, r) = (r[0], r[1], &r[2..]);
            let mut parts = r.splitn(3, |&b| b == 0);
            let (Some(lang), Some(tk), Some(v)) = (parts.next(), parts.next(), parts.next()) else {
                return Ok(None);
            };
            let v = if cf != 0 {
                if cm != 0 {
                    return Ok(None);
                }
                match inflate_text(v)? {
                    Some(v) => v,
                    None => return Ok(None),
                }
            } else {
                v.to_vec()
            };
            let raw = (k == b"XML:com.adobe.xmp").then(|| v.clone());
            let (Ok(_), Ok(_), Ok(value)) = (
                std::str::from_utf8(lang),
                std::str::from_utf8(tk),
                String::from_utf8(v),
            ) else {
                // Pillow keeps info["xmp"] even when the text is not UTF-8.
                return Ok(raw.map(|raw| TextChunk {
                    key: String::new(),
                    value: String::new(),
                    raw: Some(raw),
                }));
            };
            Ok(Some(TextChunk {
                key: latin1(k),
                value,
                raw,
            }))
        }
        _ => Ok(None),
    }
}

/// Apply one chunk's metadata the way `PngStream` fills `im_info`.
fn png_meta_chunk(chunk: &PngChunk<'_>, meta: &mut Meta) -> Result<(), String> {
    if &chunk.kind == b"eXIf" {
        let mut exif = b"Exif\x00\x00".to_vec();
        exif.extend_from_slice(chunk.data);
        meta.exif = Some(exif);
        return Ok(());
    }
    let Some(t) = parse_text_chunk(chunk)? else {
        return Ok(());
    };
    if &chunk.kind == b"iTXt" {
        if let Some(raw) = &t.raw {
            meta.xmp_bytes = Some(raw.clone());
        }
        if t.key.is_empty() {
            return Ok(());
        }
    }
    match t.key.as_str() {
        "exif" if &chunk.kind == b"tEXt" => meta.exif = t.raw.clone(),
        "Raw profile type exif" => {
            // getexif(): bytes.fromhex of the lines after the third, joined.
            let hex: String = t.value.split('\n').skip(3).collect();
            meta.exif = Some(fromhex(&hex).map_err(|e| e.to_string())?);
        }
        "XML:com.adobe.xmp" => meta.xmp_text = Some(t.value.clone()),
        _ => {}
    }
    Ok(())
}

/// `bytes.fromhex`: pairs of hex digits, ASCII whitespace allowed between pairs.
fn fromhex(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let pair = b.get(i..i + 2).and_then(|p| {
            let hi = (p[0] as char).to_digit(16)?;
            let lo = (p[1] as char).to_digit(16)?;
            Some((hi * 16 + lo) as u8)
        });
        match pair {
            Some(v) => out.push(v),
            None => {
                return Err(format!(
                    "non-hexadecimal number found in fromhex() arg at position {i}"
                ))
            }
        }
        i += 2;
    }
    Ok(out)
}

struct PngHeader {
    w: u32,
    h: u32,
    bits: u8,
    color: u8,
}

fn read_png(bytes: &[u8], check: &mut SizeCheck<'_>) -> Result<Raw, Fail> {
    let walk = walk_png(bytes, true)?;
    let ihdr = walk
        .before_idat
        .iter()
        .find(|c| &c.kind == b"IHDR")
        .ok_or(Fail::Unidentified)?;
    if ihdr.data.len() < 13 {
        return Err(Fail::Decode("Truncated IHDR chunk".to_string()));
    }
    let d = ihdr.data;
    let header = PngHeader {
        w: u32::from_be_bytes([d[0], d[1], d[2], d[3]]),
        h: u32::from_be_bytes([d[4], d[5], d[6], d[7]]),
        bits: d[8],
        color: d[9],
    };
    let supported = matches!(
        (header.bits, header.color),
        (1 | 2 | 4 | 8 | 16, 0) | (8 | 16, 2) | (1 | 2 | 4 | 8, 3) | (8 | 16, 4) | (8 | 16, 6)
    );
    if !supported || d[11] != 0 {
        return Err(Fail::Unidentified);
    }
    // APNG: acTL before the image data (a second acTL or a bad count disables it); the IDAT
    // image is an extra frame when no fcTL precedes it.
    let mut n_frames: Option<u64> = None;
    let mut actl_seen = false;
    let mut fctl_before_idat = false;
    let mut palette: Vec<u8> = Vec::new();
    let mut trns: Option<&[u8]> = None;
    for c in &walk.before_idat {
        match &c.kind {
            b"acTL" => {
                if c.data.len() < 8 {
                    return Err(Fail::Decode(
                        "APNG contains truncated acTL chunk".to_string(),
                    ));
                }
                let n = u32::from_be_bytes([c.data[0], c.data[1], c.data[2], c.data[3]]);
                if actl_seen || n_frames.is_some() {
                    n_frames = None;
                } else if n != 0 && n <= 0x8000_0000 {
                    n_frames = Some(u64::from(n));
                }
                actl_seen = true;
            }
            b"fcTL" => fctl_before_idat = true,
            b"PLTE" => palette = c.data.to_vec(),
            b"tRNS" => trns = Some(c.data),
            _ => {}
        }
    }
    let frames = match n_frames {
        Some(n) if !fctl_before_idat => n + 1,
        Some(n) => n,
        None => 1,
    };
    check(u64::from(header.w), u64::from(header.h)).map_err(Fail::Check)?;

    let mut decoder =
        png::Decoder::new_with_limits(Cursor::new(bytes), png::Limits { bytes: usize::MAX });
    decoder.set_transformations(png::Transformations::IDENTITY);
    decoder.ignore_checksums(true);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let mut reader = decoder
        .read_info()
        .map_err(|e| Fail::Decode(e.to_string()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| Fail::Decode("image too large".to_string()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| Fail::Decode(e.to_string()))?;
    let (w, h) = (header.w as usize, header.h as usize);
    let rgba = png_to_rgba(&buf, info.line_size, w, h, &header, &palette, trns);

    // Metadata as Pillow sees it after load(): every chunk up to IEND (APNG: up to the next
    // fcTL), in file order.
    let mut meta = Meta::default();
    for c in walk.before_idat.iter().chain(&walk.after_idat) {
        png_meta_chunk(c, &mut meta).map_err(Fail::Decode)?;
    }
    if walk.truncated_after {
        return Err(Fail::Decode("Truncated File Read".to_string()));
    }
    Ok(Raw {
        w,
        h,
        rgba,
        frames,
        meta,
    })
}

/// Raw PNG samples to RGBA8 with Pillow's mode rules (see the module documentation).
fn png_to_rgba(
    buf: &[u8],
    line: usize,
    w: usize,
    h: usize,
    hd: &PngHeader,
    palette: &[u8],
    trns: Option<&[u8]>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 4);
    let trns16 = |i: usize| trns.filter(|t| t.len() >= i + 2).map(|t| be16(t, i));
    for y in 0..h {
        let row = &buf[y * line..(y + 1) * line];
        match (hd.color, hd.bits) {
            (0, 16) => {
                // I;16 -> _to_8bit; the colour key is lost with the mode change.
                for x in 0..w {
                    let g = to_8bit(i64::from(be16(row, 2 * x)));
                    out.extend_from_slice(&[g, g, g, 255]);
                }
            }
            (0, bits) => {
                // "1": the key is 255 or 0; "L": the raw key against the scaled value.
                let key = trns16(0).map(|k| {
                    if bits == 1 {
                        if k != 0 {
                            255
                        } else {
                            0
                        }
                    } else {
                        k
                    }
                });
                for v in unpack_row(row, bits, w) {
                    let g = scale_grey(v, bits);
                    let a = if key == Some(u16::from(g)) { 0 } else { 255 };
                    out.extend_from_slice(&[g, g, g, a]);
                }
            }
            (2, bits) => {
                let key = match (trns16(0), trns16(2), trns16(4)) {
                    (Some(r), Some(g), Some(b)) => Some([r, g, b]),
                    _ => None,
                };
                for x in 0..w {
                    let px = if bits == 16 {
                        [row[6 * x], row[6 * x + 2], row[6 * x + 4]]
                    } else {
                        [row[3 * x], row[3 * x + 1], row[3 * x + 2]]
                    };
                    let a = if key == Some(px.map(u16::from)) {
                        0
                    } else {
                        255
                    };
                    out.extend_from_slice(&[px[0], px[1], px[2], a]);
                }
            }
            (3, bits) => {
                let alphas = trns.unwrap_or(&[]);
                for i in unpack_row(row, bits, w) {
                    let i = usize::from(i);
                    let c = palette_rgb(palette, i);
                    let a = alphas.get(i).copied().unwrap_or(255);
                    out.extend_from_slice(&[c[0], c[1], c[2], a]);
                }
            }
            (4, bits) => {
                for x in 0..w {
                    let (g, a) = if bits == 16 {
                        (row[4 * x], row[4 * x + 2])
                    } else {
                        (row[2 * x], row[2 * x + 1])
                    };
                    out.extend_from_slice(&[g, g, g, a]);
                }
            }
            (_, bits) => {
                for x in 0..w {
                    let px = if bits == 16 {
                        [row[8 * x], row[8 * x + 2], row[8 * x + 4], row[8 * x + 6]]
                    } else {
                        [row[4 * x], row[4 * x + 1], row[4 * x + 2], row[4 * x + 3]]
                    };
                    out.extend_from_slice(&px);
                }
            }
        }
    }
    out
}

// --- GIF -------------------------------------------------------------------------------

/// Pillow's `n_frames` for GIF: image descriptors up to the trailer or the end of the data,
/// found by skipping blocks (no LZW decoding).
fn gif_frame_count(bytes: &[u8]) -> u64 {
    let mut pos = 13;
    if let Some(&flags) = bytes.get(10) {
        if flags & 128 != 0 {
            pos += 3 << ((flags & 7) + 1);
        }
    }
    let skip_sub_blocks = |mut pos: usize| -> usize {
        while let Some(&n) = bytes.get(pos) {
            pos += 1;
            if n == 0 {
                break;
            }
            pos += usize::from(n);
        }
        pos
    };
    let mut count = 0;
    while let Some(&b) = bytes.get(pos) {
        pos += 1;
        match b {
            b';' => break,
            b'!' => pos = skip_sub_blocks(pos + 1),
            b',' => {
                let Some(desc) = bytes.get(pos..pos + 9) else {
                    break;
                };
                pos += 9;
                if desc[8] & 128 != 0 {
                    pos += 3 << ((desc[8] & 7) + 1);
                }
                count += 1;
                pos = skip_sub_blocks(pos + 1);
            }
            _ => {}
        }
    }
    count.max(1)
}

fn read_gif(bytes: &[u8], check: &mut SizeCheck<'_>) -> Result<Raw, Fail> {
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::Indexed);
    options.set_memory_limit(gif::MemoryLimit::Unlimited);
    // The gif crate refuses a frame with no colour table even for indexed output; Pillow
    // reads it as grey ("L"). Decode a copy with a 256-entry global table inserted (the
    // indices are unchanged) and remember that there was none.
    let has_global = bytes.len() > 10 && bytes[10] & 0x80 != 0;
    let patched;
    let source: &[u8] = if has_global || bytes.len() < 13 {
        bytes
    } else {
        let mut copy = bytes[..13].to_vec();
        copy[10] |= 0x87;
        copy.extend((0..=255u8).flat_map(|i| [i, i, i]));
        copy.extend_from_slice(&bytes[13..]);
        patched = copy;
        &patched
    };
    let mut decoder = options
        .read_info(Cursor::new(source))
        .map_err(|_| Fail::Unidentified)?;
    let global: Option<Vec<u8>> = if has_global {
        decoder.global_palette().map(<[u8]>::to_vec)
    } else {
        None
    };
    let (sw, sh) = (usize::from(decoder.width()), usize::from(decoder.height()));
    let frame = match decoder.next_frame_info() {
        Ok(Some(f)) => f,
        Ok(None) => return Err(Fail::Decode("image not found in GIF frame".to_string())),
        Err(e) => return Err(Fail::Decode(e.to_string())),
    };
    let (left, top) = (usize::from(frame.left), usize::from(frame.top));
    let (fw, fh) = (usize::from(frame.width), usize::from(frame.height));
    let transparent = frame.transparent;
    let local = frame.palette.clone();
    let (w, h) = (sw.max(left + fw), sh.max(top + fh));
    check(w as u64, h as u64).map_err(Fail::Check)?;
    let mut indices = vec![0u8; fw * fh];
    decoder
        .read_into_buffer(&mut indices)
        .map_err(|e| Fail::Decode(e.to_string()))?;
    // Frame 0 is drawn on a canvas filled with the transparent index (else 0).
    let mut canvas = vec![transparent.unwrap_or(0); w * h];
    for y in 0..fh {
        let dst = (top + y) * w + left;
        canvas[dst..dst + fw].copy_from_slice(&indices[y * fw..(y + 1) * fw]);
    }
    // Pillow reads a frame without a palette, or whose palette is the grey ramp
    // (entry i == (i, i, i)), as mode "L": the index is the grey value.
    let palette: Vec<u8> = local.or(global).unwrap_or_default();
    let grey = palette
        .chunks_exact(3)
        .enumerate()
        .all(|(i, c)| c.iter().all(|&v| usize::from(v) == i));
    let mut rgba = Vec::with_capacity(w * h * 4);
    for &i in &canvas {
        let c = if grey {
            [i, i, i]
        } else {
            palette_rgb(&palette, usize::from(i))
        };
        let a = if transparent == Some(i) { 0 } else { 255 };
        rgba.extend_from_slice(&[c[0], c[1], c[2], a]);
    }
    Ok(Raw {
        w,
        h,
        rgba,
        frames: gif_frame_count(bytes),
        meta: Meta::default(),
    })
}

// --- TIFF structures (TIFF files, EXIF, MPF) --------------------------------------------

/// A TIFF-structured byte buffer (a TIFF file, an EXIF block, a JPEG MPF index).
struct TiffView<'a> {
    data: &'a [u8],
    le: bool,
    big: bool,
}

struct IfdEntry {
    tag: u16,
    typ: u16,
    count: u64,
    /// Offset of the value field inside `data` (the value, or the offset of the value).
    field: usize,
}

impl<'a> TiffView<'a> {
    /// The header: Pillow's six prefixes; returns the view and IFD0's offset.
    fn new(data: &'a [u8]) -> Option<(TiffView<'a>, u64)> {
        let le = match data.get(..2)? {
            b"II" => true,
            b"MM" => false,
            _ => return None,
        };
        let magic = data.get(2..4)?;
        let (classic, big) = if le {
            (
                magic == b"\x2a\x00" || magic == b"\x00\x2a",
                magic == b"\x2b\x00",
            )
        } else {
            (
                magic == b"\x00\x2a" || magic == b"\x2a\x00",
                magic == b"\x00\x2b",
            )
        };
        if !classic && !big {
            return None;
        }
        let view = TiffView { data, le, big };
        let first = if big {
            view.u64_at(8)?
        } else {
            u64::from(view.u32_at(4)?)
        };
        Some((view, first))
    }

    fn u16_at(&self, i: usize) -> Option<u16> {
        let b = self.data.get(i..i + 2)?;
        Some(if self.le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    }

    fn u32_at(&self, i: usize) -> Option<u32> {
        let b: [u8; 4] = self.data.get(i..i + 4)?.try_into().ok()?;
        Some(if self.le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn u64_at(&self, i: usize) -> Option<u64> {
        let b: [u8; 8] = self.data.get(i..i + 8)?.try_into().ok()?;
        Some(if self.le {
            u64::from_le_bytes(b)
        } else {
            u64::from_be_bytes(b)
        })
    }

    /// The entries of the IFD at `offset` (as many as are complete) and the next IFD's
    /// offset (`None` when the directory is truncated).
    fn ifd(&self, offset: u64) -> (Vec<IfdEntry>, Option<u64>) {
        let mut entries = Vec::new();
        let Ok(mut pos) = usize::try_from(offset) else {
            return (entries, None);
        };
        let (count, entry_size) = if self.big {
            match self.u64_at(pos) {
                Some(n) => (n, 20),
                None => return (entries, None),
            }
        } else {
            match self.u16_at(pos) {
                Some(n) => (u64::from(n), 12),
                None => return (entries, None),
            }
        };
        pos += if self.big { 8 } else { 2 };
        for _ in 0..count {
            if pos + entry_size > self.data.len() {
                return (entries, None);
            }
            let (Some(tag), Some(typ)) = (self.u16_at(pos), self.u16_at(pos + 2)) else {
                return (entries, None);
            };
            let count = if self.big {
                self.u64_at(pos + 4).unwrap_or(0)
            } else {
                u64::from(self.u32_at(pos + 4).unwrap_or(0))
            };
            let field = pos + if self.big { 12 } else { 8 };
            entries.push(IfdEntry {
                tag,
                typ,
                count,
                field,
            });
            pos += entry_size;
        }
        let next = if self.big {
            self.u64_at(pos)
        } else {
            self.u32_at(pos).map(u64::from)
        };
        (entries, next)
    }

    /// The bytes of an entry's value, inline or at its offset; `None` when out of range.
    fn value_bytes(&self, e: &IfdEntry) -> Option<&'a [u8]> {
        let unit: u64 = match e.typ {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 | 16 | 17 | 18 => 8,
            _ => return None,
        };
        let size = usize::try_from(unit.checked_mul(e.count)?).ok()?;
        let inline = if self.big { 8 } else { 4 };
        let start = if size <= inline {
            e.field
        } else if self.big {
            usize::try_from(self.u64_at(e.field)?).ok()?
        } else {
            usize::try_from(self.u32_at(e.field)?).ok()?
        };
        self.data.get(start..start.checked_add(size)?)
    }

    /// The integer values of an entry of an integer type.
    fn uints(&self, e: &IfdEntry) -> Option<Vec<i64>> {
        let b = self.value_bytes(e)?;
        let view = TiffView {
            data: b,
            le: self.le,
            big: self.big,
        };
        let n = usize::try_from(e.count).ok()?;
        (0..n)
            .map(|i| {
                Some(match e.typ {
                    1 | 7 => i64::from(b[i]),
                    6 => i64::from(b[i] as i8),
                    3 => i64::from(view.u16_at(2 * i)?),
                    8 => i64::from(view.u16_at(2 * i)? as i16),
                    4 => i64::from(view.u32_at(4 * i)?),
                    9 => i64::from(view.u32_at(4 * i)? as i32),
                    16 => i64::try_from(view.u64_at(8 * i)?).ok()?,
                    17 => view.u64_at(8 * i)? as i64,
                    _ => return None,
                })
            })
            .collect()
    }

    /// The single integer value of `tag` in the IFD at `offset` (the last entry with that
    /// tag wins, as in Pillow's dict); `None` when absent or not one integer.
    fn tag_uint(&self, offset: u64, tag: u16) -> Option<i64> {
        let (entries, _) = self.ifd(offset);
        let e = entries.iter().rev().find(|e| e.tag == tag)?;
        if e.count != 1 {
            return None;
        }
        self.uints(e)?.first().copied()
    }
}

/// Pillow's `Exif.load` + `get(Orientation)` on an EXIF block: `Ok(None)` when the block is
/// empty or has no integer orientation, `Err` when the header is not TIFF (Pillow's
/// SyntaxError, which made Python's load fail).
fn exif_orientation(exif: &[u8]) -> Result<Option<i64>, String> {
    let data = exif.strip_prefix(b"Exif\x00\x00").unwrap_or(exif);
    if data.is_empty() {
        return Ok(None);
    }
    match TiffView::new(data) {
        Some((view, ifd0)) if !view.big => Ok(view.tag_uint(ifd0, 274)),
        _ => Err("not a TIFF file (EXIF header not valid)".to_string()),
    }
}

/// `re.search(r'tiff:Orientation(="|>)([0-9])', xmp)`.
fn xmp_orientation(xmp: &[u8]) -> Option<i64> {
    let pat = b"tiff:Orientation";
    let mut start = 0;
    while let Some(p) = xmp[start..].windows(pat.len()).position(|w| w == pat) {
        let rest = &xmp[start + p + pat.len()..];
        let digit = if rest.starts_with(b"=\"") {
            rest.get(2)
        } else if rest.starts_with(b">") {
            rest.get(1)
        } else {
            None
        };
        if let Some(d) = digit.filter(|d| d.is_ascii_digit()) {
            return Some(i64::from(d - b'0'));
        }
        start += p + 1;
    }
    None
}

/// The EXIF orientation as `exif_transpose` reads it (1 when there is none).
fn orientation(bytes: &[u8], meta: &Meta) -> Result<i64, String> {
    let mut value = None;
    if let Some(exif) = &meta.exif {
        value = exif_orientation(exif)?;
    } else if let Some(ifd0) = meta.tiff_ifd0 {
        if let Some((view, _)) = TiffView::new(bytes) {
            value = view.tag_uint(ifd0, 274);
        }
    }
    if value.is_none() {
        match (&meta.xmp_text, &meta.xmp_bytes) {
            (Some(t), _) if !t.is_empty() => value = xmp_orientation(t.as_bytes()),
            (_, Some(b)) if !b.is_empty() => value = xmp_orientation(b),
            _ => {}
        }
    }
    Ok(value.unwrap_or(1))
}

/// `ImageOps.exif_transpose` on an RGBA8 image: orientations 2..=8 are Pillow's
/// FLIP_LEFT_RIGHT, ROTATE_180, FLIP_TOP_BOTTOM, TRANSPOSE, ROTATE_270, TRANSVERSE and
/// ROTATE_90; any other value leaves the image as it is. Returns `(h, w, rgba)`.
pub fn apply_orientation(
    h: usize,
    w: usize,
    rgba: &[u8],
    orientation: i64,
) -> (usize, usize, Vec<u8>) {
    if !(2..=8).contains(&orientation) {
        return (h, w, rgba.to_vec());
    }
    let swap = orientation >= 5;
    let (oh, ow) = if swap { (w, h) } else { (h, w) };
    let mut out = vec![0u8; rgba.len()];
    for y in 0..oh {
        for x in 0..ow {
            // The source pixel (sy, sx) shown at (y, x).
            let (sy, sx) = match orientation {
                2 => (y, w - 1 - x),
                3 => (h - 1 - y, w - 1 - x),
                4 => (h - 1 - y, x),
                5 => (x, y),
                6 => (h - 1 - x, y),
                7 => (h - 1 - x, w - 1 - y),
                _ => (x, w - 1 - y),
            };
            let s = (sy * w + sx) * 4;
            let d = (y * ow + x) * 4;
            out[d..d + 4].copy_from_slice(&rgba[s..s + 4]);
        }
    }
    (oh, ow, out)
}

// --- TIFF ------------------------------------------------------------------------------

/// Pillow's TIFF `n_frames`: the IFD chain from IFD0, stopping at a zero or repeated
/// offset (or a truncated directory).
fn tiff_frame_count(view: &TiffView<'_>, ifd0: u64) -> u64 {
    let mut seen = std::collections::BTreeSet::from([ifd0]);
    let mut offset = ifd0;
    loop {
        let (_, next) = view.ifd(offset);
        match next {
            Some(n) if n != 0 && seen.insert(n) => offset = n,
            _ => return seen.len() as u64,
        }
    }
}

/// Pillow's `cmyk2rgb`: `nk - nk * c / 255` with its rounding division.
fn cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
    let nk = 255 - i32::from(k);
    let muldiv255 = |a: i32, b: i32| {
        let t = a * b + 128;
        ((t >> 8) + t) >> 8
    };
    [c, m, y].map(|v| (nk - muldiv255(i32::from(v), nk)).clamp(0, 255) as u8)
}

/// Pillow's `RGBa` unpacker: premultiplied to straight alpha.
fn unpremultiply(px: [u8; 4]) -> [u8; 4] {
    let a = u32::from(px[3]);
    match a {
        0 => [0, 0, 0, 0],
        255 => px,
        _ => {
            let f = |v: u8| (u32::from(v) * 255 / a).min(255) as u8;
            [f(px[0]), f(px[1]), f(px[2]), px[3]]
        }
    }
}

fn read_tiff(bytes: &[u8], check: &mut SizeCheck<'_>) -> Result<Raw, Fail> {
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    use tiff::ColorType;

    let (view, ifd0) = TiffView::new(bytes).ok_or(Fail::Unidentified)?;
    let (entries, _) = view.ifd(ifd0);
    let find = |tag: u16| entries.iter().rev().find(|e| e.tag == tag);
    let photometric = find(262)
        .and_then(|e| view.uints(e))
        .and_then(|v| v.first().copied());
    let extra: Vec<i64> = find(338).and_then(|e| view.uints(e)).unwrap_or_default();
    let colormap: Option<Vec<i64>> = find(320).and_then(|e| view.uints(e));
    let xmp = find(700)
        .and_then(|e| view.value_bytes(e))
        .map(<[u8]>::to_vec);

    // The tiff crate does not read palette images: present the indices as grey samples
    // (PhotometricInterpretation 1) and map them through the ColorMap below.
    let palette_mode = photometric == Some(3);
    let patched;
    let source: &[u8] = if palette_mode {
        let e = find(262).filter(|e| e.typ == 3).ok_or(Fail::Unidentified)?;
        let mut copy = bytes.to_vec();
        let one = if view.le { [1, 0] } else { [0, 1] };
        copy[e.field..e.field + 2].copy_from_slice(&one);
        patched = copy;
        &patched
    } else {
        bytes
    };

    let mut decoder = Decoder::new(Cursor::new(source))
        .map_err(|_| Fail::Unidentified)?
        .with_limits(Limits::unlimited());
    let (w32, h32) = decoder.dimensions().map_err(|_| Fail::Unidentified)?;
    let color = decoder.colortype().map_err(|_| Fail::Unidentified)?;
    check(u64::from(w32), u64::from(h32)).map_err(Fail::Check)?;
    let (w, h) = (w32 as usize, h32 as usize);
    let image = decoder
        .read_image()
        .map_err(|e| Fail::Decode(e.to_string()))?;
    let unsupported = || Fail::Decode(format!("unsupported TIFF sample layout {color:?}"));
    let n = w * h;
    let samples = match &image {
        DecodingResult::U8(d) => d.len(),
        DecodingResult::U16(d) => d.len(),
        DecodingResult::I16(d) => d.len(),
        DecodingResult::U32(d) => d.len(),
        DecodingResult::I32(d) => d.len(),
        _ => 0,
    };
    let needed = match color {
        ColorType::Gray(bits) if bits < 8 => (w * bits as usize).div_ceil(8) * h,
        ColorType::Gray(_) => n,
        ColorType::Multiband { num_samples, .. } => n * usize::from(num_samples),
        ColorType::RGB(_) => n * 3,
        _ => n * 4,
    };
    if samples < needed {
        return Err(unsupported());
    }
    let mut rgba = Vec::with_capacity(n * 4);
    match (color, image) {
        (ColorType::Gray(bits @ (1 | 2 | 4 | 8)), DecodingResult::U8(data)) if palette_mode => {
            let map = colormap.ok_or_else(|| Fail::Decode("missing ColorMap".to_string()))?;
            let k = map.len() / 3;
            let palette: Vec<u8> = (0..k)
                .flat_map(|i| [map[i], map[k + i], map[2 * k + i]])
                .map(|v| (v / 256).clamp(0, 255) as u8)
                .collect();
            let line = (w * bits as usize).div_ceil(8);
            for y in 0..h {
                for i in unpack_row(&data[y * line..(y + 1) * line], bits, w) {
                    let c = palette_rgb(&palette, usize::from(i));
                    rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
                }
            }
        }
        (ColorType::Gray(bits @ (1 | 2 | 4 | 8)), DecodingResult::U8(data)) => {
            let line = (w * bits as usize).div_ceil(8);
            for y in 0..h {
                for v in unpack_row(&data[y * line..(y + 1) * line], bits, w) {
                    let g = scale_grey(v, bits);
                    rgba.extend_from_slice(&[g, g, g, 255]);
                }
            }
        }
        (ColorType::Gray(16), DecodingResult::U16(data)) => {
            for &v in &data[..n] {
                let g = to_8bit(i64::from(v));
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        (ColorType::Gray(16), DecodingResult::I16(data)) => {
            for &v in &data[..n] {
                let g = to_8bit(i64::from(v));
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        // Pillow's I;32N reads the unsigned samples as int32.
        (ColorType::Gray(32), DecodingResult::U32(data)) if view.le => {
            for &v in &data[..n] {
                let g = to_8bit(i64::from(v as i32));
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        (ColorType::Gray(32), DecodingResult::I32(data)) => {
            for &v in &data[..n] {
                let g = to_8bit(i64::from(v));
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        (
            ColorType::Multiband {
                bit_depth: 8,
                num_samples: 2,
            },
            DecodingResult::U8(data),
        ) if extra.first() == Some(&2) => {
            for px in data[..n * 2].chunks_exact(2) {
                rgba.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
            }
        }
        (ColorType::RGB(8), DecodingResult::U8(data)) => {
            for px in data[..n * 3].chunks_exact(3) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        (ColorType::RGB(16), DecodingResult::U16(data)) => {
            for px in data[..n * 3].chunks_exact(3) {
                rgba.extend_from_slice(&[
                    (px[0] >> 8) as u8,
                    (px[1] >> 8) as u8,
                    (px[2] >> 8) as u8,
                    255,
                ]);
            }
        }
        (ColorType::RGBA(bits @ (8 | 16)), image) => {
            let samples: Vec<u8> = match image {
                DecodingResult::U8(d) if bits == 8 => d,
                DecodingResult::U16(d) if bits == 16 => d.iter().map(|v| (v >> 8) as u8).collect(),
                _ => return Err(unsupported()),
            };
            let spp = samples.len() / n.max(1);
            if spp < 4 {
                return Err(unsupported());
            }
            for px in samples.chunks_exact(spp).take(n) {
                let px = [px[0], px[1], px[2], px[3]];
                let out = match extra.first() {
                    Some(0) => [px[0], px[1], px[2], 255],
                    Some(1) => unpremultiply(px),
                    _ => px,
                };
                rgba.extend_from_slice(&out);
            }
        }
        (ColorType::CMYK(8), DecodingResult::U8(data)) => {
            for px in data[..n * 4].chunks_exact(4) {
                let c = cmyk_to_rgb(px[0], px[1], px[2], px[3]);
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        _ => return Err(unsupported()),
    }
    let meta = Meta {
        tiff_ifd0: Some(ifd0),
        xmp_bytes: xmp,
        ..Meta::default()
    };
    Ok(Raw {
        w,
        h,
        rgba,
        frames: tiff_frame_count(&view, ifd0),
        meta,
    })
}

// --- JPEG, WEBP, BMP (the image crate) --------------------------------------------------

/// Decode with an `image` crate decoder: the header has been read; check the size, then
/// decode with no allocation limits (the budget was the limit) and expand to RGBA8.
fn decode_with<D: image::ImageDecoder>(
    mut decoder: D,
    check: &mut SizeCheck<'_>,
) -> Result<(usize, usize, Vec<u8>), Fail> {
    let (w, h) = decoder.dimensions();
    check(u64::from(w), u64::from(h)).map_err(Fail::Check)?;
    decoder
        .set_limits(image::Limits::no_limits())
        .map_err(|e| Fail::Decode(e.to_string()))?;
    let img =
        image::DynamicImage::from_decoder(decoder).map_err(|e| Fail::Decode(e.to_string()))?;
    let rgba: Vec<u8> = match img {
        image::DynamicImage::ImageLuma8(g) => {
            g.into_raw().iter().flat_map(|&v| [v, v, v, 255]).collect()
        }
        image::DynamicImage::ImageLumaA8(g) => g
            .into_raw()
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        image::DynamicImage::ImageRgb8(c) => c
            .into_raw()
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        image::DynamicImage::ImageRgba8(c) => c.into_raw(),
        other => {
            return Err(Fail::Decode(format!(
                "unsupported color type {:?}",
                other.color()
            )))
        }
    };
    Ok((w as usize, h as usize, rgba))
}

/// The JPEG markers before the scan: EXIF (APP1 `Exif\0\0`, later ones appended), XMP
/// (APP1), the MPF index (APP2) and whether an APP1 announces Ultra HDR.
fn jpeg_meta(bytes: &[u8]) -> (Meta, u64) {
    let mut meta = Meta::default();
    let mut mp: Option<&[u8]> = None;
    let mut hdr = false;
    let mut pos = 2;
    while pos + 4 <= bytes.len() {
        if bytes[pos] != 0xff {
            break;
        }
        let marker = bytes[pos + 1];
        if marker == 0xff {
            pos += 1;
            continue;
        }
        if marker == 0xda || marker == 0xd9 {
            break;
        }
        if (0xd0..=0xd7).contains(&marker) || marker == 0x01 {
            pos += 2;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]));
        if len < 2 {
            break;
        }
        let end = (pos + 2 + len).min(bytes.len());
        let s = &bytes[pos + 4..end];
        match marker {
            0xe1 if s.starts_with(b"Exif\x00\x00") => match &mut meta.exif {
                Some(exif) => exif.extend_from_slice(&s[6..]),
                None => meta.exif = Some(s.to_vec()),
            },
            0xe1 if s.starts_with(b"http://ns.adobe.com/xap/1.0/\x00") => {
                let i = s.iter().position(|&b| b == 0).unwrap_or(0);
                meta.xmp_bytes = Some(s[i + 1..].to_vec());
            }
            0xe2 if s.starts_with(b"MPF\x00") => mp = Some(&s[4..]),
            _ => {}
        }
        if marker == 0xe1 && s.windows(16).any(|w| w == b" hdrgm:Version=\"") {
            hdr = true;
        }
        pos = end;
    }
    // MPO: the MP index's NumberOfImages (with a full entry table), unless Ultra HDR.
    let frames = mp
        .and_then(|data| {
            let (view, ifd0) = TiffView::new(data)?;
            let n = view.tag_uint(ifd0, 0xb001)?;
            let (entries, _) = view.ifd(ifd0);
            let table = entries.iter().rev().find(|e| e.tag == 0xb002)?;
            let table_len = view.value_bytes(table)?.len() as i64;
            (n > 1 && table_len >= 16 * n && !hdr).then_some(n as u64)
        })
        .unwrap_or(1);
    (meta, frames)
}

fn read_jpeg(bytes: &[u8], check: &mut SizeCheck<'_>) -> Result<Raw, Fail> {
    let decoder = image::codecs::jpeg::JpegDecoder::new(Cursor::new(bytes))
        .map_err(|_| Fail::Unidentified)?;
    let (w, h, rgba) = decode_with(decoder, check)?;
    let (meta, frames) = jpeg_meta(bytes);
    Ok(Raw {
        w,
        h,
        rgba,
        frames,
        meta,
    })
}

/// WEBP RIFF chunks: the `ANMF` count (libwebp's frame count, 1 for a still image) and the
/// first `EXIF` and `XMP ` chunks.
fn webp_meta(bytes: &[u8]) -> (Meta, u64) {
    let mut meta = Meta::default();
    let mut frames = 0u64;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let kind = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let start = pos + 8;
        let end = start.saturating_add(len).min(bytes.len());
        let data = &bytes[start..end];
        match kind {
            b"ANMF" => frames += 1,
            b"EXIF" if meta.exif.is_none() => meta.exif = Some(data.to_vec()),
            b"XMP " if meta.xmp_bytes.is_none() => meta.xmp_bytes = Some(data.to_vec()),
            _ => {}
        }
        pos = start.saturating_add(len).saturating_add(len & 1);
    }
    (meta, frames.max(1))
}

fn read_webp(bytes: &[u8], check: &mut SizeCheck<'_>) -> Result<Raw, Fail> {
    let decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))
        .map_err(|_| Fail::Unidentified)?;
    let (w, h, rgba) = decode_with(decoder, check)?;
    let (meta, frames) = webp_meta(bytes);
    Ok(Raw {
        w,
        h,
        rgba,
        frames,
        meta,
    })
}

fn read_bmp(bytes: &[u8], check: &mut SizeCheck<'_>) -> Result<Raw, Fail> {
    let decoder =
        image::codecs::bmp::BmpDecoder::new(Cursor::new(bytes)).map_err(|_| Fail::Unidentified)?;
    let (w, h, rgba) = decode_with(decoder, check)?;
    Ok(Raw {
        w,
        h,
        rgba,
        frames: 1,
        meta: Meta::default(),
    })
}

// ---------------------------------------------------------------------------------------
// Index images and PNG output

/// `io.indices_to_rgba`: an index image (`< 0` transparent) and a palette of RGB8 triples
/// to RGBA8 (transparent pixels are `0, 0, 0, 0`). An index beyond the palette is an error
/// with numpy's IndexError message.
pub fn indices_to_rgba(indices: &[i64], palette: &[[u8; 3]]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(indices.len() * 4);
    for &i in indices {
        if i < 0 {
            out.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let c = usize::try_from(i)
                .ok()
                .and_then(|i| palette.get(i))
                .ok_or_else(|| {
                    format!(
                        "index {i} is out of bounds for axis 0 with size {}",
                        palette.len()
                    )
                })?;
            out.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    Ok(out)
}

/// `io.upscale_nearest`: repeat every pixel `factor` times along both axes. `data` is
/// `h * w * channels` elements; the result is `(h * factor) * (w * factor) * channels`.
pub fn upscale_nearest<T: Copy>(
    data: &[T],
    h: usize,
    w: usize,
    channels: usize,
    factor: usize,
) -> Vec<T> {
    assert_eq!(
        data.len(),
        h * w * channels,
        "upscale_nearest: data does not match the shape"
    );
    let ow = w * factor;
    let mut out = Vec::with_capacity(h * factor * ow * channels);
    for y in 0..h {
        let row_start = out.len();
        for x in 0..w {
            let px = &data[(y * w + x) * channels..(y * w + x + 1) * channels];
            for _ in 0..factor {
                out.extend_from_slice(px);
            }
        }
        for _ in 1..factor {
            out.extend_from_within(row_start..row_start + ow * channels);
        }
    }
    out
}

/// Where `save_png` puts the transparent palette entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransparentIndex {
    /// After the colours, only when some pixel is transparent.
    Last,
    /// Always at index 0, every colour shifted up by one (one PLTE for every image of a
    /// palette).
    First,
}

/// The chunk bytes of a tEXt (Latin-1) or, for values outside Latin-1, iTXt chunk, as
/// Pillow's `PngInfo.add_text` writes them. A key outside Latin-1 is an error.
fn text_chunk(key: &str, value: &str) -> Result<([u8; 4], Vec<u8>), String> {
    let to_latin1 = |s: &str| -> Option<Vec<u8>> {
        s.chars().map(|c| u8::try_from(u32::from(c)).ok()).collect()
    };
    let k = to_latin1(key).ok_or_else(|| {
        format!(
            "'latin-1' codec can't encode the PNG text key {}",
            str_repr(key)
        )
    })?;
    match to_latin1(value) {
        Some(v) => {
            let mut data = k;
            data.push(0);
            data.extend_from_slice(&v);
            Ok((*b"tEXt", data))
        }
        None => {
            let mut data = k;
            data.extend_from_slice(&[0, 0, 0, 0, 0]);
            data.extend_from_slice(value.as_bytes());
            Ok((*b"iTXt", data))
        }
    }
}

/// The PNG file `save_png` writes, as bytes (see [`save_png`]).
pub fn encode_png(
    indices: &[i64],
    h: usize,
    w: usize,
    palette: &[[u8; 3]],
    text: &[(String, String)],
    scale: usize,
    transparent_index: TransparentIndex,
) -> Result<Vec<u8>, String> {
    assert_eq!(indices.len(), h * w, "encode_png: indices must be h * w");
    let scale = scale.max(1);
    let up;
    let (indices, h, w) = if scale > 1 {
        up = upscale_nearest(indices, h, w, 1, scale);
        (&up[..], h * scale, w * scale)
    } else {
        (indices, h, w)
    };
    let first = transparent_index == TransparentIndex::First;
    let has_transparency = first || indices.iter().any(|&i| i < 0);
    let n_entries = palette.len() + usize::from(has_transparency);
    let width = u32::try_from(w).map_err(|_| "image too wide for PNG".to_string())?;
    let height = u32::try_from(h).map_err(|_| "image too tall for PNG".to_string())?;

    let mut keys: Vec<&(String, String)> = text.iter().collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0));
    let chunks: Vec<([u8; 4], Vec<u8>)> = keys
        .iter()
        .map(|(k, v)| text_chunk(k, v))
        .collect::<Result<_, _>>()?;

    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_depth(png::BitDepth::Eight);
        let data: Vec<u8> = if n_entries <= 256 {
            let mut plte: Vec<u8> = Vec::with_capacity(n_entries * 3);
            let clear;
            let data: Vec<u8> = if first {
                clear = 0usize;
                plte.extend_from_slice(&[0, 0, 0]);
                plte.extend(palette.iter().flatten());
                // -1 (transparent) becomes 0; uint8 wrap-around as numpy's astype.
                indices.iter().map(|&i| i.wrapping_add(1) as u8).collect()
            } else {
                clear = palette.len();
                plte.extend(palette.iter().flatten());
                if has_transparency {
                    plte.extend_from_slice(&[0, 0, 0]);
                }
                indices
                    .iter()
                    .map(|&i| if i < 0 { clear as u8 } else { i as u8 })
                    .collect()
            };
            encoder.set_color(png::ColorType::Indexed);
            encoder.set_palette(plte);
            if has_transparency {
                // Pillow's tRNS for an int transparency: 255 for every entry before it.
                let mut trns = vec![255u8; clear];
                trns.push(0);
                encoder.set_trns(trns);
            }
            data
        } else {
            encoder.set_color(png::ColorType::Rgba);
            indices_to_rgba(indices, palette)?
        };
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        for (kind, chunk) in &chunks {
            writer
                .write_chunk(png::chunk::ChunkType(*kind), chunk)
                .map_err(|e| e.to_string())?;
        }
        writer.write_image_data(&data).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// `io.save_png`: write an index image (`h * w`, `< 0` transparent) as PNG, scaled by
/// `scale` (nearest neighbour), creating the parent directory. Palette-indexed when the
/// palette plus a transparent entry fits in 256 entries, else RGBA. `text` is written
/// sorted by key. Errors: `IoError::Os` for the directory or the file, `PixelforgeError`
/// for a text key outside Latin-1 or an index beyond the palette in the RGBA fallback.
#[allow(clippy::too_many_arguments)]
pub fn save_png(
    path: &Path,
    indices: &[i64],
    h: usize,
    w: usize,
    palette: &[[u8; 3]],
    text: &[(String, String)],
    scale: usize,
    transparent_index: TransparentIndex,
) -> Result<(), IoError> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| IoError::os(e, parent))?;
    }
    let bytes = encode_png(indices, h, w, palette, text, scale, transparent_index)
        .map_err(|e| IoError::Pixelforge(pf(e)))?;
    std::fs::write(path, bytes).map_err(|e| IoError::os(e, path))
}

/// The PNG file of an RGBA8 image (`h * w * 4` bytes, row-major), 8-bit RGBA without text
/// chunks: the pixels Pillow's `Image.fromarray(a, mode="RGBA").save(path)` writes.
pub fn encode_rgba_png(rgba: &[u8], h: usize, w: usize) -> Result<Vec<u8>, String> {
    assert_eq!(
        rgba.len(),
        h * w * 4,
        "encode_rgba_png: data must be h * w * 4"
    );
    let width = u32::try_from(w).map_err(|_| "image too wide for PNG".to_string())?;
    let height = u32::try_from(h).map_err(|_| "image too tall for PNG".to_string())?;
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_color(png::ColorType::Rgba);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(rgba).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Write an RGBA8 image as a plain RGBA PNG ([`encode_rgba_png`]). Unlike [`save_png`] the
/// parent directory must exist (as for Pillow's `Image.save`). Errors: `IoError::Os` for
/// the file, `PixelforgeError` when the image cannot be encoded (e.g. an empty one).
pub fn save_rgba_png(path: &Path, rgba: &[u8], h: usize, w: usize) -> Result<(), IoError> {
    let bytes = encode_rgba_png(rgba, h, w).map_err(|e| IoError::Pixelforge(pf(e)))?;
    std::fs::write(path, bytes).map_err(|e| IoError::os(e, path))
}

/// `io.read_png_text`: a PNG file's text chunks (tEXt, zTXt, iTXt) as Pillow's `im.text`
/// has them, in file order (a repeated key appears once per chunk; the last one wins in a
/// dict). Any other content has no text (Pillow had none for the other formats). Errors:
/// `IoError::Os` when the file cannot be read; `PixelforgeError` for a PNG whose chunks are
/// broken or cut short, or a decompressed text value above [`MAX_TEXT_CHUNK`].
pub fn read_png_text(path: &Path) -> Result<Vec<(String, String)>, IoError> {
    let bytes = read_regular(path)?;
    if !bytes.starts_with(PNG_SIGNATURE) {
        return Ok(Vec::new());
    }
    let fail = |reason: &str| IoError::Pixelforge(pf(format!("{}: {reason}", path.display())));
    let walk = match walk_png(&bytes, false) {
        Ok(walk) => walk,
        Err(Fail::Decode(reason)) => return Err(fail(&reason)),
        Err(_) => return Err(fail("broken PNG file")),
    };
    if walk.truncated_after {
        return Err(fail("Truncated File Read"));
    }
    let mut out = Vec::new();
    for c in walk.before_idat.iter().chain(&walk.after_idat) {
        let parsed = parse_text_chunk(c).map_err(|e| fail(&e))?;
        if let Some(t) = parsed.filter(|t| !t.key.is_empty() || &c.kind != b"iTXt") {
            out.push((t.key, t.value));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("pixelforge-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn rgba_png_round_trip_without_text() {
        let path = tmp("rgba.png");
        let rgba: Vec<u8> = (0..2 * 3 * 4).map(|i| (i * 11) as u8).collect();
        save_rgba_png(&path, &rgba, 2, 3).unwrap();
        let decoder = png::Decoder::new(std::io::BufReader::new(File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (3, 2));
        assert_eq!(info.color_type, png::ColorType::Rgba);
        assert_eq!(&buf[..info.buffer_size()], &rgba[..]);
        assert!(read_png_text(&path).unwrap().is_empty());
        // An empty image cannot be written (Pillow refuses it too).
        assert!(matches!(
            save_rgba_png(&tmp("empty.png"), &[], 0, 3),
            Err(IoError::Pixelforge(_))
        ));
    }

    #[test]
    fn indexed_png_round_trip_with_text() {
        let path = tmp("roundtrip/a.png");
        let palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let indices = [0, 1, -1, 2, 2, 1];
        let text = vec![
            ("b".to_string(), "second".to_string()),
            ("a".to_string(), "first".to_string()),
        ];
        for ti in [TransparentIndex::Last, TransparentIndex::First] {
            save_png(&path, &indices, 2, 3, &palette, &text, 2, ti).unwrap();
            let loaded = load(&path, MAX_INPUT_PIXELS).unwrap();
            assert_eq!((loaded.h, loaded.w), (4, 6));
            let back: Vec<i64> = (0..loaded.h * loaded.w)
                .map(|p| {
                    if loaded.alpha[p] == 0 {
                        return -1;
                    }
                    let rgb: Vec<u8> = loaded.rgb[p * 3..p * 3 + 3]
                        .iter()
                        .map(|v| (v * 255.0).round_ties_even() as u8)
                        .collect();
                    palette.iter().position(|c| c[..] == rgb[..]).unwrap() as i64
                })
                .collect();
            let expected = upscale_nearest(&indices, 2, 3, 1, 2);
            assert_eq!(back, expected);
            let read = read_png_text(&path).unwrap();
            assert_eq!(
                read,
                vec![
                    ("a".to_string(), "first".to_string()),
                    ("b".to_string(), "second".to_string())
                ]
            );
        }
    }

    #[test]
    fn budget_is_checked_from_the_header() {
        // A PNG whose IHDR claims 30x20 and whose image data is missing entirely: only the
        // header is needed to refuse it.
        let mut bytes = PNG_SIGNATURE.to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&30u32.to_be_bytes());
        ihdr.extend_from_slice(&20u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&ihdr);
        bytes.extend_from_slice(&crc32(b"IHDR", &ihdr).to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(b"IDAT");
        let err = decode_bytes("big.png", &bytes, 599).unwrap_err();
        assert_eq!(
            err.0,
            "big.png: 30x20 = 600 pixels exceeds the input budget of 599 pixels \
             (--max-input-pixels)"
        );
        let err = decode_bytes("big.png", &bytes, 600).unwrap_err();
        assert!(
            err.0.starts_with("big.png: could not decode image ("),
            "{}",
            err.0
        );
    }

    #[test]
    fn orientation_matches_pillow_transposes() {
        // 2 rows x 3 columns, pixel value = 10 * row + column.
        let img: Vec<u8> = [0u8, 1, 2, 10, 11, 12]
            .iter()
            .flat_map(|&v| [v, v, v, 255])
            .collect();
        let grid = |o: i64| {
            let (h, w, out) = apply_orientation(2, 3, &img, o);
            let vals: Vec<u8> = out.chunks_exact(4).map(|p| p[0]).collect();
            (h, w, vals)
        };
        assert_eq!(grid(1), (2, 3, vec![0, 1, 2, 10, 11, 12]));
        assert_eq!(grid(9), (2, 3, vec![0, 1, 2, 10, 11, 12]));
        assert_eq!(grid(2), (2, 3, vec![2, 1, 0, 12, 11, 10]));
        assert_eq!(grid(3), (2, 3, vec![12, 11, 10, 2, 1, 0]));
        assert_eq!(grid(4), (2, 3, vec![10, 11, 12, 0, 1, 2]));
        assert_eq!(grid(5), (3, 2, vec![0, 10, 1, 11, 2, 12]));
        // ROTATE_270 (clockwise): the left column becomes the top row, read bottom-up.
        assert_eq!(grid(6), (3, 2, vec![10, 0, 11, 1, 12, 2]));
        assert_eq!(grid(7), (3, 2, vec![12, 2, 11, 1, 10, 0]));
        assert_eq!(grid(8), (3, 2, vec![2, 12, 1, 11, 0, 10]));
    }

    #[test]
    fn exif_orientation_is_read_from_ifd0() {
        // Little-endian TIFF header, IFD0 with one SHORT entry 274 = 6.
        let mut exif = b"Exif\x00\x00II\x2a\x00\x08\x00\x00\x00".to_vec();
        exif.extend_from_slice(&1u16.to_le_bytes());
        exif.extend_from_slice(&274u16.to_le_bytes());
        exif.extend_from_slice(&3u16.to_le_bytes());
        exif.extend_from_slice(&1u32.to_le_bytes());
        exif.extend_from_slice(&[6, 0, 0, 0]);
        exif.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(exif_orientation(&exif), Ok(Some(6)));
        assert_eq!(exif_orientation(b"Exif\x00\x00"), Ok(None));
        assert!(exif_orientation(b"Exif\x00\x00garbage!").is_err());
        assert_eq!(xmp_orientation(b"<x tiff:Orientation=\"8\"/>"), Some(8));
        assert_eq!(
            xmp_orientation(b"<tiff:Orientation>3</tiff:Orientation>"),
            Some(3)
        );
        assert_eq!(
            xmp_orientation(b"tiff:Orientation x tiff:Orientation>5"),
            Some(5)
        );
    }

    #[test]
    fn sixteen_bit_grey_is_rescaled_half_to_even() {
        assert_eq!(to_8bit(7710), 30);
        assert_eq!(to_8bit(64250), 250);
        assert_eq!(to_8bit(65535), 255);
        assert_eq!(to_8bit(-5), 0);
        assert_eq!(to_8bit(70000), 255);
        // 128.5 * 257 = 33024.5 is not an integer; 385.5 / 257 ... check a true tie:
        // v / 257 == k + 0.5 needs v = 257k + 128.5, never an integer, so ties cannot occur;
        // values just around the midpoint round to the nearer integer.
        assert_eq!(to_8bit(257 * 2 + 128), 2);
        assert_eq!(to_8bit(257 * 2 + 129), 3);
        let png = {
            let mut out = Vec::new();
            let mut enc = png::Encoder::new(&mut out, 2, 1);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Sixteen);
            let mut wr = enc.write_header().unwrap();
            let mut data = Vec::new();
            data.extend_from_slice(&7710u16.to_be_bytes());
            data.extend_from_slice(&64250u16.to_be_bytes());
            wr.write_image_data(&data).unwrap();
            wr.finish().unwrap();
            out
        };
        let img = decode_bytes("g16.png", &png, MAX_INPUT_PIXELS).unwrap();
        assert_eq!(img.rgba, vec![30, 30, 30, 255, 250, 250, 250, 255]);
    }

    #[test]
    fn array_inputs_round_half_to_even_and_hash_the_rgba() {
        let shape = [1, 2, 3];
        let data = [0.5 / 255.0, 1.5 / 255.0, 1.0, 0.0, 2.5 / 255.0, 0.5];
        let loaded = from_image_array(&shape, ArrayData::F64(&data), 100).unwrap();
        let rgba: Vec<u8> = vec![0, 2, 255, 255, 0, 2, 128, 255];
        assert_eq!(loaded.alpha, vec![255, 255]);
        assert_eq!(loaded.sha256, rgba8_digest(1, 2, &rgba));
        let err = from_image_array(&[2, 3], ArrayData::U8(&[0; 6]), 100).unwrap_err();
        assert_eq!(
            err.0,
            "image array must have shape (H, W, 3) or (H, W, 4), got (2, 3)"
        );
        let err = from_image_array(&shape, ArrayData::Other("int64"), 100).unwrap_err();
        assert_eq!(err.0, "image array must be uint8 or float, got int64");
    }
}
