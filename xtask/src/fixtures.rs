//! `xtask fixtures`: the synthetic test images of `tests/fixtures` (`scripts/make_fixtures.py`),
//! drawn deterministically with no RNG.
//!
//! Every formula follows numpy's evaluation order on float64 (`np.mgrid` integers promoted
//! to float where numpy promotes them); `np.round` is half to even and `np.sin`, `np.cos`,
//! `np.hypot` come from the portable `libm` crate. The pixels equal the committed PNGs'
//! (checked by `--check` and the tests); the PNG bytes differ from Pillow's encoder.

use std::path::{Path, PathBuf};

use crate::{fail, Fail};

/// The fixture names, in the order `make_fixtures.FIXTURES` lists them.
pub const FIXTURES: [&str; 7] = [
    "line_diag",
    "two_color",
    "gradient6",
    "circle_alpha",
    "checker_tiles",
    "noisy_gradient",
    "seamless_src",
];

/// A drawn fixture: `h * w * channels` bytes, row-major, 3 (RGB) or 4 (RGBA) channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub h: usize,
    pub w: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

/// `_to_u8`: `np.round(np.clip(v, 0, 255)).astype(np.uint8)` (half to even).
fn to_u8(v: f64) -> u8 {
    v.clamp(0.0, 255.0).round_ties_even() as u8
}

fn grid(h: usize, w: usize, channels: usize, mut pixel: impl FnMut(i64, i64) -> Vec<u8>) -> Image {
    let mut data = Vec::with_capacity(h * w * channels);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let px = pixel(y, x);
            debug_assert_eq!(px.len(), channels);
            data.extend_from_slice(&px);
        }
    }
    Image {
        h,
        w,
        channels,
        data,
    }
}

/// 256×256 white, 2-px black diagonal line corner to corner.
fn line_diag() -> Image {
    grid(256, 256, 3, |y, x| {
        if (x - y).abs() <= 1 {
            vec![0, 0, 0]
        } else {
            vec![255, 255, 255]
        }
    })
}

/// 128×128, left half pure red, right half pure blue.
fn two_color() -> Image {
    grid(128, 128, 3, |_, x| {
        if x < 64 {
            vec![255, 0, 0]
        } else {
            vec![0, 0, 255]
        }
    })
}

/// 256×256, six vertical bands, each a smooth dark-to-light gradient of one hue.
fn gradient6() -> Image {
    grid(256, 256, 3, |y, x| {
        let band = (x * 6 / 256).min(5);
        let hue = band as f64 / 6.0;
        let t = y as f64 / 255.0;
        let value = 0.25 + 0.75 * t;
        let saturation = 1.0 - 0.5 * t;
        let h6 = hue * 6.0;
        let sector = (h6.floor() as i64).rem_euclid(6);
        let f = h6 - h6.floor();
        let p = value * (1.0 - saturation);
        let q = value * (1.0 - saturation * f);
        let u = value * (1.0 - saturation * (1.0 - f));
        let rgb = match sector {
            0 => [value, u, p],
            1 => [q, value, p],
            2 => [p, value, u],
            3 => [p, q, value],
            4 => [u, p, value],
            _ => [value, p, q],
        };
        rgb.iter().map(|&c| to_u8(c * 255.0)).collect()
    })
}

/// 256×256 RGBA: orange disc (r=100) with a 4-px feathered alpha edge and a darker eye.
fn circle_alpha() -> Image {
    let (cx, cy) = (127.5, 127.5);
    let orange = [240.0, 140.0, 30.0];
    let dark = [110.0, 50.0, 10.0];
    grid(256, 256, 4, |y, x| {
        let (xf, yf) = (x as f64, y as f64);
        let dist = libm::hypot(xf - cx, yf - cy);
        let alpha = ((102.0 - dist) / 4.0).clamp(0.0, 1.0);
        let eye_dist = libm::hypot(xf - (cx - 32.0), yf - (cy - 28.0));
        let eye = (20.5 - eye_dist).clamp(0.0, 1.0);
        let mut px: Vec<u8> = (0..3)
            .map(|c| to_u8(orange[c] * (1.0 - eye) + dark[c] * eye))
            .collect();
        px.push(to_u8(alpha * 255.0));
        px
    })
}

/// 256×256 of 16-px tiles, each a 2×2 checker of 8-px blocks; odd tile rows are hflipped.
fn checker_tiles() -> Image {
    grid(256, 256, 3, |y, x| {
        let odd_row = (y / 16) % 2 == 1;
        let local_x = if odd_row { 15 - (x % 16) } else { x % 16 };
        if ((local_x / 8) + ((y % 16) / 8)) % 2 == 0 {
            vec![60, 90, 160]
        } else {
            vec![230, 200, 90]
        }
    })
}

/// 256×256 horizontal gray gradient plus a deterministic high-frequency pattern.
fn noisy_gradient() -> Image {
    grid(256, 256, 3, |y, x| {
        let gray = x as f64 + (libm::sin(37.0 * x as f64) + libm::cos(53.0 * y as f64)) * 8.0;
        vec![to_u8(gray); 3]
    })
}

/// 128×128 smooth 2-D sine color field with period 128, so it wraps.
fn seamless_src() -> Image {
    let w = 2.0 * std::f64::consts::PI / 128.0;
    grid(128, 128, 3, |y, x| {
        let rgb = [
            0.5 + 0.4 * libm::sin(w * x as f64),
            0.5 + 0.4 * libm::sin(w * y as f64),
            0.5 + 0.4 * libm::cos(w * (x + y) as f64),
        ];
        rgb.iter().map(|&c| to_u8(c * 255.0)).collect()
    })
}

/// Draw one fixture by name.
pub fn draw(name: &str) -> Option<Image> {
    Some(match name {
        "line_diag" => line_diag(),
        "two_color" => two_color(),
        "gradient6" => gradient6(),
        "circle_alpha" => circle_alpha(),
        "checker_tiles" => checker_tiles(),
        "noisy_gradient" => noisy_gradient(),
        "seamless_src" => seamless_src(),
        _ => return None,
    })
}

/// The PNG of an image (8-bit RGB or RGBA, no text chunks).
pub fn encode(image: &Image) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, image.w as u32, image.h as u32);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_color(if image.channels == 4 {
            png::ColorType::Rgba
        } else {
            png::ColorType::Rgb
        });
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer
            .write_image_data(&image.data)
            .map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Decode an 8-bit RGB or RGBA PNG as stored (no expansion), as Pillow's `np.asarray(im)`
/// gives it for those modes.
pub fn decode(path: &Path) -> Result<Image, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let channels = match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => 3,
        (png::ColorType::Rgba, png::BitDepth::Eight) => 4,
        (color, depth) => {
            return Err(format!(
                "{}: expected 8-bit RGB or RGBA, got {color:?} at {depth:?}",
                path.display()
            ))
        }
    };
    let (w, h) = (info.width as usize, info.height as usize);
    let mut data = Vec::with_capacity(h * w * channels);
    for row in buf[..info.buffer_size()].chunks(info.line_size) {
        data.extend_from_slice(&row[..w * channels]);
    }
    Ok(Image {
        h,
        w,
        channels,
        data,
    })
}

/// The differences between a committed fixture and the drawn one: empty when the mode,
/// size and every pixel agree; otherwise one line per difference (at most `limit` pixel
/// lines, then a count).
pub fn differences(committed: &Image, fresh: &Image, limit: usize) -> Vec<String> {
    if (committed.h, committed.w, committed.channels) != (fresh.h, fresh.w, fresh.channels) {
        return vec![format!(
            "shape {}x{}x{} != drawn {}x{}x{}",
            committed.h, committed.w, committed.channels, fresh.h, fresh.w, fresh.channels
        )];
    }
    let c = committed.channels;
    let mut lines = Vec::new();
    let mut count = 0;
    for (i, (a, b)) in committed
        .data
        .chunks_exact(c)
        .zip(fresh.data.chunks_exact(c))
        .enumerate()
    {
        if a != b {
            count += 1;
            if lines.len() < limit {
                lines.push(format!(
                    "pixel (y={}, x={}): committed {a:?}, drawn {b:?}",
                    i / committed.w,
                    i % committed.w
                ));
            }
        }
    }
    if count > lines.len() {
        lines.push(format!("... {count} differing pixels in all"));
    }
    lines
}

/// The committed fixture directory.
pub fn default_dir() -> PathBuf {
    crate::root().join("tests").join("fixtures")
}

/// `xtask fixtures --check`: every committed fixture exists and decodes to the drawn pixels.
pub fn check(dir: &Path) -> Result<(), Fail> {
    let mut problems = Vec::new();
    for name in FIXTURES {
        let path = dir.join(format!("{name}.png"));
        if !path.is_file() {
            problems.push(format!(
                "missing test fixture {name} in {}; restore the committed file, or regenerate \
                 it with `cargo xtask fixtures` (same pixels, but the PNG bytes and so every \
                 input_sha256 may differ)",
                dir.display()
            ));
            continue;
        }
        let committed = decode(&path).map_err(|e| fail(1, e))?;
        let fresh = draw(name).expect("listed fixture");
        for line in differences(&committed, &fresh, 20) {
            problems.push(format!("{}: {line}", path.display()));
        }
    }
    if problems.is_empty() {
        println!("{} fixtures match {}", FIXTURES.len(), dir.display());
        Ok(())
    } else {
        Err(fail(1, problems.join("\n")))
    }
}

/// `xtask fixtures [OUTDIR]`: write every fixture; returns the paths written.
pub fn write(dir: &Path) -> Result<Vec<PathBuf>, Fail> {
    std::fs::create_dir_all(dir).map_err(|e| fail(1, format!("{}: {e}", dir.display())))?;
    let mut written = Vec::new();
    for name in FIXTURES {
        let path = dir.join(format!("{name}.png"));
        let bytes = encode(&draw(name).expect("listed fixture")).map_err(|e| fail(1, e))?;
        std::fs::write(&path, bytes).map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
        written.push(path);
    }
    Ok(written)
}
