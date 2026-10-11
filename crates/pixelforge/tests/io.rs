//! Port of tests/test_io.py (and test_branches.py's PNG palette fallback): refused inputs,
//! the pixel budget, FIFOs and directories, 16-bit samples, multi-frame input, EXIF
//! orientation. The input files Python built with Pillow are written here byte by byte
//! (PNG chunks with stored deflate blocks, a GIF with uncompressed-style LZW codes).

mod common;
use common::*;

use std::path::Path;

use pixelforge::io::{self, IoError, TransparentIndex};
use pixelforge::mask::PixelforgeError;
use pixelforge::preprocess;

// ---------------------------------------------------------------------------------------
// Hand-written PNG files

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A zlib stream of stored (uncompressed) deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = if data.is_empty() {
        vec![&[]]
    } else {
        data.chunks(0xFFFF).collect()
    };
    for (i, block) in blocks.iter().enumerate() {
        out.push(u8::from(i + 1 == blocks.len()));
        let len = block.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &v in data {
        a = (a + u32::from(v)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut body = kind.to_vec();
    body.extend_from_slice(data);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

/// A PNG file: IHDR (`width`, `height`, `depth`, `color_type`), the `extra` chunks, then
/// one IDAT of the unfiltered `rows` (each without its filter byte).
fn png_file(
    width: u32,
    height: u32,
    depth: u8,
    color_type: u8,
    extra: &[(&[u8; 4], Vec<u8>)],
    rows: &[Vec<u8>],
) -> Vec<u8> {
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[depth, color_type, 0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &ihdr);
    for (kind, data) in extra {
        png_chunk(&mut out, kind, data);
    }
    let mut raw = Vec::new();
    for row in rows {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    png_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    png_chunk(&mut out, b"IEND", &[]);
    out
}

// ---------------------------------------------------------------------------------------
// Hand-written GIF files

/// LZW image data (minimum code size 2) that sends a clear code before every pixel, so the
/// code width stays at 3 bits: valid for any decoder, no dictionary needed.
fn gif_lzw(pixels: &[u8]) -> Vec<u8> {
    let (clear, end) = (4u32, 5u32);
    let mut codes = Vec::new();
    for &p in pixels {
        codes.push(clear);
        codes.push(u32::from(p));
    }
    codes.push(end);
    let mut bytes = Vec::new();
    let (mut acc, mut n) = (0u32, 0u32);
    for c in codes {
        acc |= c << n;
        n += 3;
        while n >= 8 {
            bytes.push(acc as u8);
            acc >>= 8;
            n -= 8;
        }
    }
    if n > 0 {
        bytes.push(acc as u8);
    }
    let mut out = vec![2u8]; // minimum code size
    for block in bytes.chunks(255) {
        out.push(block.len() as u8);
        out.extend_from_slice(block);
    }
    out.push(0);
    out
}

/// A `w × h` GIF89a with a two-colour global table and one frame per entry of `frames`
/// (each frame a solid table index).
fn gif_file(w: u16, h: u16, table: [[u8; 3]; 2], frames: &[u8]) -> Vec<u8> {
    let mut out = b"GIF89a".to_vec();
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(&[0x80, 0, 0]); // global table of 2 entries
    for c in table {
        out.extend_from_slice(&c);
    }
    for &index in frames {
        out.push(0x2C);
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&w.to_le_bytes());
        out.extend_from_slice(&h.to_le_bytes());
        out.push(0);
        out.extend_from_slice(&gif_lzw(&vec![index; usize::from(w) * usize::from(h)]));
    }
    out.push(0x3B);
    out
}

fn message(err: IoError) -> String {
    err.to_string()
}

fn load_err(path: &Path, max_pixels: u64) -> String {
    match io::load(path, max_pixels) {
        Ok(_) => panic!("{} was loaded", path.display()),
        Err(e @ IoError::Pixelforge(_)) => message(e),
        Err(e) => panic!("not a PixelforgeError: {e}"),
    }
}

/// `np.round(loaded.rgb * 255)` per pixel.
fn rgb8(loaded: &io::Loaded) -> Vec<[u8; 3]> {
    loaded
        .rgb
        .chunks_exact(3)
        .map(|p| [0, 1, 2].map(|i| (p[i] * 255.0).round_ties_even() as u8))
        .collect()
}

// ---------------------------------------------------------------------------------------

#[test]
fn test_pixel_budget_is_checked_before_decoding() {
    let tmp = TempDir::new("budget");
    let path = tmp.join("big.png");
    let rgba = Image::filled(20, 30, &[0, 0, 0]).rgba();
    io::save_rgba_png(&path, &rgba, 20, 30).unwrap();
    let err = load_err(&path, 599);
    assert!(
        err.contains("600 pixels exceeds the input budget of 599"),
        "{err}"
    );
    let loaded = io::load(&path, 600).unwrap();
    assert_eq!(
        (loaded.h, loaded.w, loaded.rgb.len()),
        (20, 30, 20 * 30 * 3)
    );
}

#[test]
fn test_preprocessing_errors_still_importable_from_preprocess() {
    // Python: `preprocess.PixelforgeError is PixelforgeError`. Here: preprocessing and I/O
    // report the one error type (this compiles only if they do).
    let e = PixelforgeError("x".into());
    let from_preprocess: preprocess::Error = e.clone().into();
    let from_io: IoError = e.clone().into();
    assert!(matches!(from_preprocess, preprocess::Error::Pixelforge(ref p) if *p == e));
    assert!(matches!(from_io, IoError::Pixelforge(ref p) if *p == e));
}

#[test]
fn test_non_image_bytes_give_a_clean_error() {
    let tmp = TempDir::new("fake");
    let path = tmp.join("fake.png");
    std::fs::write(
        &path,
        b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n",
    )
    .unwrap();
    let err = load_err(&path, io::MAX_INPUT_PIXELS);
    assert!(err.contains("not a supported image"), "{err}");
}

#[test]
fn test_16_bit_grayscale_is_rescaled_not_clipped() {
    // Image.fromarray(uint16).save(): mode I;16, a 16-bit greyscale PNG.
    let values: [[u16; 2]; 2] = [[7710, 64250], [64250, 7710]];
    let rows: Vec<Vec<u8>> = values
        .iter()
        .map(|r| r.iter().flat_map(|v| v.to_be_bytes()).collect())
        .collect();
    let tmp = TempDir::new("gray16");
    let path = tmp.join("gray16.png");
    std::fs::write(&path, png_file(2, 2, 16, 0, &[], &rows)).unwrap();
    let loaded = io::load(&path, io::MAX_INPUT_PIXELS).unwrap();
    let px = rgb8(&loaded);
    assert_eq!(px[0], [30, 30, 30]);
    assert_eq!(px[1], [250, 250, 250]);
}

#[test]
fn test_exif_orientation_is_applied() {
    // 8 wide × 4 high, left half white; Pillow writes the EXIF (orientation 6, rotate 90°
    // clockwise to display) as an eXIf chunk holding the little-endian TIFF block.
    let rows: Vec<Vec<u8>> = (0..4)
        .map(|_| {
            (0..8)
                .flat_map(|x| if x < 4 { [255u8; 3] } else { [0u8; 3] })
                .collect()
        })
        .collect();
    let mut exif = b"II\x2a\x00\x08\x00\x00\x00".to_vec();
    exif.extend_from_slice(&1u16.to_le_bytes());
    exif.extend_from_slice(&0x0112u16.to_le_bytes());
    exif.extend_from_slice(&3u16.to_le_bytes());
    exif.extend_from_slice(&1u32.to_le_bytes());
    exif.extend_from_slice(&[6, 0, 0, 0]);
    exif.extend_from_slice(&0u32.to_le_bytes());
    let tmp = TempDir::new("exif");
    let path = tmp.join("rotated.png");
    std::fs::write(&path, png_file(8, 4, 8, 2, &[(b"eXIf", exif)], &rows)).unwrap();
    let loaded = io::load(&path, io::MAX_INPUT_PIXELS).unwrap();
    assert_eq!((loaded.h, loaded.w), (8, 4));
    let half = 4 * loaded.w * 3;
    let top = loaded.rgb[..half]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let bottom = loaded.rgb[half..]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(top == 1.0 && bottom == 0.0, "white on top: {top} {bottom}");
}

#[test]
fn test_multi_frame_uses_first_frame_with_a_warning() {
    let tmp = TempDir::new("anim");
    let path = tmp.join("anim.gif");
    std::fs::write(&path, gif_file(4, 4, [[255, 0, 0], [0, 0, 255]], &[0, 1])).unwrap();
    let loaded = io::load(&path, io::MAX_INPUT_PIXELS).unwrap();
    assert!(
        loaded.warnings.iter().any(|w| w.contains("2 frames")),
        "{:?}",
        loaded.warnings
    );
    assert!(loaded.rgb[0] > 0.9 && loaded.rgb[2] < 0.1);
}

#[test]
fn test_directory_is_refused() {
    let tmp = TempDir::new("dir");
    let err = load_err(tmp.path(), io::MAX_INPUT_PIXELS);
    assert!(err.contains("not a regular file"), "{err}");
}

#[cfg(unix)]
#[test]
fn test_fifo_is_refused_without_blocking() {
    let tmp = TempDir::new("fifo");
    let fifo = tmp.join("pipe.png");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(status.success());
    let err = load_err(&fifo, io::MAX_INPUT_PIXELS);
    assert!(err.contains("not a regular file"), "{err}");
    match io::sha256_file(&fifo) {
        Err(e @ IoError::Pixelforge(_)) => {
            let err = message(e);
            assert!(err.contains("not a regular file"), "{err}");
        }
        other => panic!("sha256_file on a FIFO: {other:?}"),
    }
}

#[test]
fn test_small_input_repeat_respects_the_budget() {
    let (h, w) = (1, 4000);
    let rgb = vec![0.5; h * w * 3];
    let alpha = vec![255u8; h * w];
    let config = config(&[
        ("preset", s("background")),
        ("out_width", int(512)),
        ("out_height", int(512)),
        ("tileset", b(false)),
        ("denoise", s("none")),
    ]);
    match preprocess::run(&rgb, &alpha, h, w, &config) {
        Err(preprocess::Error::Pixelforge(e)) => assert!(e.0.contains("too small"), "{}", e.0),
        Err(e) => panic!("not a PixelforgeError: {e}"),
        Ok(_) => panic!("accepted"),
    }
}

// From tests/test_branches.py.
#[test]
fn test_save_png_falls_back_to_rgba_at_256_colors_plus_transparency() {
    for (n_colors, color_type) in [(255usize, 3u8), (256, 6)] {
        // 3 = palette ("P"), 6 = RGBA.
        let palette: Vec<[u8; 3]> = (0..n_colors)
            .map(|i| [i as u8, (255 - i) as u8, ((7 * i) % 256) as u8])
            .collect();
        let (h, w) = (16, 16);
        let mut indices: Vec<i64> = (0..h * w).map(|i| (i % n_colors) as i64).collect();
        indices[..4].fill(-1); // some transparency
        let tmp = TempDir::new("save_png");
        let path = tmp.join("out.png");
        let text = vec![("k".to_string(), "v".to_string())];
        io::save_png(
            &path,
            &indices,
            h,
            w,
            &palette,
            &text,
            1,
            TransparentIndex::Last,
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[25], color_type, "{n_colors} colors");
        assert_eq!(io::read_png_text(&path).unwrap(), text);
        let loaded = io::load(&path, io::MAX_INPUT_PIXELS).unwrap();
        let decoded = rgb8(&loaded);
        let expected = io::indices_to_rgba(&indices, &palette).unwrap();
        for (i, e) in expected.chunks_exact(4).enumerate() {
            assert_eq!(loaded.alpha[i], e[3], "alpha at {i}");
            if e[3] == 255 {
                assert_eq!(decoded[i], [e[0], e[1], e[2]], "pixel {i}");
            }
        }
        // Without transparency 256 colors still fit a palette PNG.
        let opaque: Vec<i64> = indices.iter().map(|i| i.abs()).collect();
        io::save_png(
            &path,
            &opaque,
            h,
            w,
            &palette,
            &[],
            1,
            TransparentIndex::Last,
        )
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap()[25], 3, "{n_colors} colors");
    }
}
