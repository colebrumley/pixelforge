//! The preprocessing stage (`pixelforge/preprocess.py`'s `run`), assembled from the masks
//! (`mask`), the geometry (`layout`), the enhance pass (`enhance`), the filters (`filters`)
//! and the colour conversion (`color`): alpha threshold and keying, crop, output size and
//! layout, `fit="crop"`, integer repeat, box pre-reduction, enhance, seamless wrap padding,
//! transparent fill, denoise and CIELAB.
//!
//! Images are row-major slices with explicit `(h, w)`: RGB `h * w * 3` in `[0, 1]`, alpha,
//! masks and weights `h * w`. Every step runs in Python's order with Python's arithmetic; the
//! exact steps (mask, keying, crop, layout, repeat, pre-reduction, padding, fill, median) are
//! bitwise identical, the float ones (enhance's Gaussian, the bilateral, LAB) are within the
//! last bits (see `filters`).
//!
//! `remove_bg` is not applied by `run`: it takes the alpha as given and the caller mats it
//! first. `pipeline::run_loaded` does so with [`remove_background_external`] (the `rembg`
//! command, docs/rust-migration-plan.md Decision 5) unless its caller says the alpha is
//! already matted (the Python wrapper runs the `rembg` package itself and passes the result).

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::color;
use crate::config::{Config, ConfigError};
use crate::enhance;
use crate::filters;
use crate::layout::{self, Geometry, Layout, SEAMLESS_PAD_OUT};
use crate::mask::{self, PixelforgeError};
use crate::parallel;

/// A preprocessing failure: an invalid configuration for this input (`ConfigError`, e.g. a
/// tile size above the derived output size) or a runtime failure (`PixelforgeError`), each
/// with Python's message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Config(ConfigError),
    Pixelforge(PixelforgeError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(e) => e.fmt(f),
            Error::Pixelforge(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<ConfigError> for Error {
    fn from(e: ConfigError) -> Self {
        Error::Config(e)
    }
}

impl From<PixelforgeError> for Error {
    fn from(e: PixelforgeError) -> Self {
        Error::Pixelforge(e)
    }
}

/// `preprocess.Preprocessed`: the image the downscalers work on and its output geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct Preprocessed {
    /// `h * w * 3` CIELAB.
    pub lab: Vec<f64>,
    /// `h * w`, true = opaque.
    pub mask: Vec<bool>,
    /// `h * w` in `[0, 1]`: how much each pixel counts in the downscalers' averages (its
    /// alpha / 255 inside the mask, 0 outside).
    pub weight: Vec<f64>,
    /// Size of `lab`, `mask` and `weight` (seamless padding included).
    pub h: usize,
    pub w: usize,
    /// Output size, wrap padding, outline margin and canvas layout.
    pub geometry: Geometry,
    /// A flat opaque background was made transparent.
    pub background_keyed: bool,
    /// `fit_mode(config)`: how the size was reconciled with `tile_size`.
    pub fit: Option<String>,
    /// The integer box pre-reduction applied to the input (1 = none).
    pub prereduce_factor: usize,
    /// The texts of the `UserWarning`s Python emits (from `resolve_layout`).
    pub warnings: Vec<String>,
}

impl Preprocessed {
    pub fn out_width(&self) -> i64 {
        self.geometry.out_width
    }

    pub fn out_height(&self) -> i64 {
        self.geometry.out_height
    }

    pub fn pad_out(&self) -> i64 {
        self.geometry.pad_out
    }

    pub fn outline_margin(&self) -> i64 {
        self.geometry.outline_margin
    }

    pub fn layout(&self) -> Option<Layout> {
        self.geometry.layout
    }

    /// Width the image itself is downscaled to (the outline margin is added after).
    pub fn inner_width(&self) -> i64 {
        self.geometry.inner_width()
    }

    pub fn inner_height(&self) -> i64 {
        self.geometry.inner_height()
    }

    /// Transparent `((top, bottom), (left, right))` padding from the inner size to the canvas,
    /// outline margin included.
    pub fn canvas_pad(&self) -> ((i64, i64), (i64, i64)) {
        self.geometry.canvas_pad()
    }

    pub fn target_width(&self) -> i64 {
        self.geometry.target_width()
    }

    pub fn target_height(&self) -> i64 {
        self.geometry.target_height()
    }
}

/// The RGB, mask and weight planes being carried through `run`, with their size.
struct Planes {
    rgb: Vec<f64>,
    mask: Vec<bool>,
    weight: Vec<f64>,
    h: usize,
    w: usize,
}

/// `a[y0:y1, x0:x1]` of the three planes of a `w`-wide image (a copy).
#[allow(clippy::too_many_arguments)]
fn crop_planes(
    rgb: &[f64],
    mask: &[bool],
    weight: &[f64],
    w: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> Planes {
    let (ch, cw) = (y1 - y0, x1 - x0);
    let mut out = Planes {
        rgb: Vec::with_capacity(ch * cw * 3),
        mask: Vec::with_capacity(ch * cw),
        weight: Vec::with_capacity(ch * cw),
        h: ch,
        w: cw,
    };
    for y in y0..y1 {
        let row = y * w;
        out.rgb
            .extend_from_slice(&rgb[(row + x0) * 3..(row + x1) * 3]);
        out.mask.extend_from_slice(&mask[row + x0..row + x1]);
        out.weight.extend_from_slice(&weight[row + x0..row + x1]);
    }
    out
}

/// `preprocess.subject_bbox`: the crop box `(y0, y1, x0, x1)` `run` would use (after keying,
/// before denoising).
pub fn subject_bbox(
    rgb: &[f64],
    alpha: &[u8],
    h: usize,
    w: usize,
    config: &Config,
) -> Result<(usize, usize, usize, usize), PixelforgeError> {
    Ok(mask::subject(rgb, alpha, h, w, config)?.crop)
}

/// `preprocess.subject_brightness`: the brightness (largest channel) of the subject's mostly
/// opaque pixels after keying, row-major, for levels shared between images.
pub fn subject_brightness(
    rgb: &[f64],
    alpha: &[u8],
    h: usize,
    w: usize,
    config: &Config,
) -> Result<Vec<f64>, PixelforgeError> {
    let s = mask::subject(rgb, alpha, h, w, config)?;
    Ok(enhance::subject_brightness(rgb, &s.mask, &s.weight))
}

/// `preprocess.denoise(rgb, config)` on an `(h, w, 3)` image: none, the 3×3 median per
/// channel, or skimage's bilateral (clipped to `[0, 1]`) on `thread_count()` threads, whose
/// result does not depend on the thread count. The bilateral sigmas are in the image's pixels.
pub fn denoise(rgb: Vec<f64>, h: usize, w: usize, config: &Config) -> Vec<f64> {
    match config.denoise.as_str() {
        "median" => filters::denoise_median(&rgb, h, w),
        "bilateral" => {
            // DEVIATION: Section 5 — mode="edge" instead of skimage's default zero padding,
            // which darkens the image border. Transparent pixels were filled with the nearest
            // opaque color by the caller for the same reason.
            filters::denoise_bilateral(
                &rgb,
                h,
                w,
                config.denoise_sigma_color,
                config.denoise_sigma_spatial,
                parallel::thread_count(),
            )
        }
        _ => rgb,
    }
}

/// The message of the `ConfigError` raised when `remove_bg` is set and the `rembg`
/// command is not on `PATH`.
pub const REMBG_MISSING: &str = "--remove-bg requires the rembg command on PATH; \
                                 install it with: pipx install 'rembg[cli]'";

/// The external matting command `remove_bg` runs: `rembg i INPUT OUTPUT`, found on `PATH`.
pub const REMBG_COMMAND: &str = "rembg";

/// A private temporary directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> std::io::Result<TempDir> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        loop {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = base.join(format!(
                "pixelforge-rembg-{}-{nanos}-{n}",
                std::process::id()
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(TempDir(dir)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An `h * w * 4` RGBA8 image as PNG bytes (8-bit RGBA, no text chunks).
fn encode_rgba_png(rgba: &[u8], h: usize, w: usize) -> Result<Vec<u8>, String> {
    let width = u32::try_from(w).map_err(|_| "image too wide for PNG".to_string())?;
    let height = u32::try_from(h).map_err(|_| "image too tall for PNG".to_string())?;
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(rgba).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// `preprocess.remove_background` through the external `rembg` command (Decision 5): the
/// image (`rgb` `h * w * 3` in `[0, 1]`, rounded half to even to 8 bits as Python's
/// `np.round`, with `alpha`) is written to a temporary RGBA PNG, `rembg i IN OUT` runs on it,
/// and the alpha channel of its output is returned (`h * w`). A missing `rembg` executable is
/// a `ConfigError` with Python's message; a failing command (its stderr is reported), an
/// unreadable output or one of another size is a `PixelforgeError`. AI matting is outside
/// the determinism contract.
pub fn remove_background_external(
    rgb: &[f64],
    alpha: &[u8],
    h: usize,
    w: usize,
) -> Result<Vec<u8>, Error> {
    assert_eq!(
        rgb.len(),
        h * w * 3,
        "remove_background_external: rgb length"
    );
    assert_eq!(
        alpha.len(),
        h * w,
        "remove_background_external: alpha length"
    );
    let fail = |msg: String| Error::Pixelforge(PixelforgeError(format!("--remove-bg: {msg}")));
    let mut rgba = Vec::with_capacity(h * w * 4);
    for (px, &a) in rgb.chunks_exact(3).zip(alpha) {
        rgba.extend(px.iter().map(|&v| (v * 255.0).round_ties_even() as u8));
        rgba.push(a);
    }
    let png = encode_rgba_png(&rgba, h, w).map_err(fail)?;
    let dir = TempDir::new().map_err(|e| fail(format!("temporary directory: {e}")))?;
    let (input, output) = (dir.path().join("input.png"), dir.path().join("output.png"));
    std::fs::write(&input, png).map_err(|e| fail(e.to_string()))?;
    let done = match Command::new(REMBG_COMMAND)
        .arg("i")
        .arg(&input)
        .arg(&output)
        .stdin(Stdio::null())
        .output()
    {
        Ok(done) => done,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ConfigError(REMBG_MISSING.to_string()).into());
        }
        Err(e) => return Err(fail(format!("could not run {REMBG_COMMAND}: {e}"))),
    };
    if !done.status.success() {
        let stderr = String::from_utf8_lossy(&done.stderr);
        return Err(fail(format!(
            "{REMBG_COMMAND} failed ({}): {}",
            done.status,
            stderr.trim()
        )));
    }
    let bytes = std::fs::read(&output)
        .map_err(|e| fail(format!("{REMBG_COMMAND} wrote no output: {e}")))?;
    let name = output.display().to_string();
    let matted = crate::io::decode_bytes(&name, &bytes, u64::MAX).map_err(|e| fail(e.0))?;
    if (matted.h, matted.w) != (h, w) {
        return Err(fail(format!(
            "{REMBG_COMMAND} returned a {}x{} image for a {w}x{h} input",
            matted.w, matted.h
        )));
    }
    Ok(matted.rgba.chunks_exact(4).map(|px| px[3]).collect())
}

/// What `_meta.json` records as `remove_bg_versions` for the external command: `rembg` is the
/// trimmed standard output of `rembg --version` (`None` when the command is missing, fails
/// or prints nothing); `onnxruntime` is always `None`, since the runtime inside the command
/// is not visible from outside. (The Python binding records the installed package versions
/// instead.)
pub fn rembg_versions() -> Vec<(&'static str, Option<String>)> {
    let rembg = Command::new(REMBG_COMMAND)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    vec![("rembg", rembg), ("onnxruntime", None)]
}

/// `preprocess.run(rgb, alpha, config)` with the alpha as given (`remove_bg` already
/// applied by the caller, e.g. with [`remove_background_external`]): the subject cropped,
/// sized and filtered for the downscalers.
pub fn run(
    rgb: &[f64],
    alpha: &[u8],
    h: usize,
    w: usize,
    config: &Config,
) -> Result<Preprocessed, Error> {
    assert_eq!(rgb.len(), h * w * 3, "preprocess::run: rgb length");
    assert_eq!(alpha.len(), h * w, "preprocess::run: alpha length");
    let subject = mask::subject(rgb, alpha, h, w, config)?;
    let background_keyed = subject.background_keyed;
    let (y0, y1, x0, x1) = subject.crop;
    let mut p = if (y0, y1, x0, x1) == (0, h, 0, w) {
        Planes {
            rgb: rgb.to_vec(),
            mask: subject.mask,
            weight: subject.weight,
            h,
            w,
        }
    } else {
        crop_planes(rgb, &subject.mask, &subject.weight, w, y0, y1, x0, x1)
    };

    let resolved = layout::resolve_layout(p.w as i64, p.h as i64, config)?;
    let (out_w, out_h, canvas) = (resolved.out_width, resolved.out_height, resolved.layout);
    let fit = layout::fit_mode(config).map(str::to_string);
    if fit.as_deref() == Some("crop") {
        // The downscalers always fill the whole target, so the cropped part of the canvas is
        // removed from the input instead (centered, to the canvas aspect).
        let (cy0, cy1, cx0, cx1) = layout::fit_crop(p.w as i64, p.h as i64, out_w, out_h);
        let (cy0, cy1, cx0, cx1) = (cy0 as usize, cy1 as usize, cx0 as usize, cx1 as usize);
        p = crop_planes(&p.rgb, &p.mask, &p.weight, p.w, cy0, cy1, cx0, cx1);
        if !p.mask.iter().any(|&m| m) {
            return Err(PixelforgeError("no opaque pixels left after fit='crop'".into()).into());
        }
    }
    let margin = layout::outline_margin(config);
    let (inner_w, inner_h) = match canvas {
        Some(l) => (l.0, l.1),
        None => (out_w - 2 * margin, out_h - 2 * margin),
    };

    // DEVIATION: Section 5 — inputs smaller than the output are repeated by an integer factor
    // first, so that every output pixel is backed by at least one input pixel in all three
    // downscalers (otherwise gerstner would leave superpixels without pixels).
    let factor = layout::repeat_factor(p.h, p.w, out_w, out_h)?;
    if factor > 1 {
        let (rgb, mask, weight) = layout::repeat(&p.rgb, &p.mask, &p.weight, p.h, p.w, factor);
        p = Planes {
            rgb,
            mask,
            weight,
            h: p.h * factor,
            w: p.w * factor,
        };
    }

    // DEVIATION: Section 5 — inputs far larger than the output are first box-reduced by an
    // integer factor (exact area mean), so that denoise and the downscalers run on less than
    // 2 × prereduce_max_ratio × the output resolution. Kopf's kernels span 4×4 output pixels,
    // so at these ratios the reduction is nearly lossless, and it bounds time and memory. Up
    // to factor − 1 trailing rows/columns that do not fill a block are dropped.
    let mut reduce_by = layout::prereduce_factor(
        p.h as i64,
        p.w as i64,
        out_h,
        out_w,
        config.prereduce_max_ratio,
    ) as usize;
    if reduce_by > 1 {
        let small = layout::prereduce(&p.rgb, &p.mask, Some(&p.weight), p.h, p.w, reduce_by);
        if small.mask.iter().any(|&m| m) {
            p = Planes {
                rgb: small.rgb,
                mask: small.mask,
                weight: small.weight.expect("prereduce was given a weight"),
                h: small.h,
                w: small.w,
            };
        } else {
            // A sparse subject would vanish: keep full resolution.
            reduce_by = 1;
        }
    }

    // Before denoising: the bilateral filter then sees the exaggerated edges and keeps them.
    // Before the seamless padding: the image is filtered as the periodic tile it is.
    if config.enhance > 0.0 || config.ink > 0.0 {
        let cell = (p.h as f64 / inner_h as f64, p.w as f64 / inner_w as f64);
        p.rgb = enhance::enhance(
            &p.rgb,
            &p.mask,
            p.h,
            p.w,
            cell,
            config,
            Some(&p.weight),
            config.seamless,
        );
    }

    let mut pad_out = 0;
    if config.seamless {
        pad_out = SEAMLESS_PAD_OUT;
        let (pad_y, pad_x) = layout::seamless_pads(p.h, p.w, inner_h, inner_w);
        let (rgb, mask, weight) =
            layout::wrap_pad(&p.rgb, &p.mask, &p.weight, p.h, p.w, pad_y, pad_x);
        p = Planes {
            rgb,
            mask,
            weight,
            h: p.h + 2 * pad_y,
            w: p.w + 2 * pad_x,
        };
    }

    let filled = if p.mask.iter().all(|&m| m) {
        p.rgb
    } else {
        mask::fill_transparent(&p.rgb, &p.mask, p.h, p.w)
    };
    let rgb = denoise(filled, p.h, p.w, config);
    let lab = color::rgb_to_lab(&rgb);
    Ok(Preprocessed {
        lab,
        mask: p.mask,
        weight: p.weight,
        h: p.h,
        w: p.w,
        geometry: Geometry {
            out_width: out_w,
            out_height: out_h,
            pad_out,
            outline_margin: margin,
            layout: canvas,
        },
        background_keyed,
        fit,
        prereduce_factor: reduce_by,
        warnings: resolved.warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Value;

    fn config(kwargs: &[(&str, Value)]) -> Config {
        let kw: Vec<(String, Value)> = kwargs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        Config::new(&kw).unwrap()
    }

    fn s(v: &str) -> Value {
        Value::Str(v.into())
    }

    /// A red block on a white backdrop, fully opaque.
    fn sprite_on_white(size: usize) -> (Vec<f64>, Vec<u8>) {
        let mut rgb = vec![1.0; size * size * 3];
        for y in 20..76 {
            for x in 30..66 {
                rgb[(y * size + x) * 3..(y * size + x) * 3 + 3].copy_from_slice(&[0.8, 0.2, 0.1]);
            }
        }
        (rgb, vec![255; size * size])
    }

    #[test]
    fn keyed_sprite_is_cropped_to_its_box_plus_one() {
        // tests/test_preprocess.py::test_flat_background_is_keyed_but_enclosed_regions_are_kept
        let (rgb, alpha) = sprite_on_white(96);
        let c = config(&[
            ("preset", s("sprite")),
            ("denoise", s("none")),
            ("out_height", Value::Int(16)),
        ]);
        let pre = run(&rgb, &alpha, 96, 96, &c).unwrap();
        assert!(pre.background_keyed);
        assert_eq!((pre.h, pre.w), (58, 38));
        assert!(!pre.mask[..pre.w].iter().any(|&m| m));
        assert_eq!(pre.lab.len(), pre.h * pre.w * 3);
        assert_eq!(pre.weight.len(), pre.h * pre.w);
        assert_eq!(pre.fit, None);
        assert_eq!(pre.outline_margin(), 1);
    }

    #[test]
    fn fully_transparent_input_is_an_error() {
        let c = config(&[("preset", s("sprite"))]);
        let err = run(&[0.5; 4 * 4 * 3], &[0; 16], 4, 4, &c).unwrap_err();
        assert_eq!(
            err,
            Error::Pixelforge(PixelforgeError(
                "image is fully transparent after alpha thresholding".into()
            ))
        );
    }

    #[test]
    fn small_inputs_are_repeated_and_large_ones_reduced() {
        let c = config(&[
            ("out_width", Value::Int(16)),
            ("out_height", Value::Int(16)),
            ("denoise", s("none")),
            ("crop_to_alpha", Value::Bool(false)),
            ("outline", s("none")),
            ("key_bg", Value::Bool(false)),
        ]);
        let pre = run(&[0.5; 8 * 8 * 3], &[255; 64], 8, 8, &c).unwrap();
        assert_eq!((pre.h, pre.w, pre.prereduce_factor), (16, 16, 1));
        let big = config(&[
            ("preset", s("sprite")),
            ("key_bg", Value::Bool(false)),
            ("crop_to_alpha", Value::Bool(false)),
            ("denoise", s("none")),
            ("out_width", Value::Int(16)),
            ("out_height", Value::Int(16)),
            ("prereduce_max_ratio", Value::Int(8)),
        ]);
        let pre = run(
            &vec![0.25; 256 * 256 * 3],
            &vec![255; 256 * 256],
            256,
            256,
            &big,
        )
        .unwrap();
        assert_eq!((pre.h, pre.w, pre.prereduce_factor), (128, 128, 2));
    }

    #[test]
    fn seamless_mode_pads_by_two_output_pixels() {
        let c = config(&[
            ("out_width", Value::Int(16)),
            ("out_height", Value::Int(16)),
            ("key_bg", Value::Bool(false)),
            ("seamless", Value::Bool(true)),
            ("denoise", s("none")),
        ]);
        let n = 40 * 24;
        let rgb: Vec<f64> = (0..n * 3).map(|i| (i % 7) as f64 / 7.0).collect();
        let pre = run(&rgb, &vec![255; n], 40, 24, &c).unwrap();
        assert_eq!(pre.pad_out(), 2);
        let (pad_y, pad_x) = layout::seamless_pads(40, 24, pre.inner_height(), pre.inner_width());
        assert_eq!((pre.h, pre.w), (40 + 2 * pad_y, 24 + 2 * pad_x));
        assert!(pre.weight.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn fit_crop_with_nothing_opaque_left_is_an_error() {
        // A 160×96 input whose only opaque pixels lie in the columns fit="crop" removes.
        let (h, w) = (96, 160);
        let mut alpha = vec![0u8; h * w];
        for y in 0..h {
            alpha[y * w] = 255;
            alpha[y * w + w - 1] = 255;
        }
        let c = config(&[
            ("preset", s("background")),
            ("out_width", Value::Int(64)),
            ("fit", s("crop")),
        ]);
        let err = run(&vec![0.5; h * w * 3], &alpha, h, w, &c).unwrap_err();
        assert_eq!(err.to_string(), "no opaque pixels left after fit='crop'");
    }

    #[test]
    fn subject_helpers_agree_with_run() {
        let (rgb, alpha) = sprite_on_white(96);
        let c = config(&[("preset", s("sprite"))]);
        assert_eq!(
            subject_bbox(&rgb, &alpha, 96, 96, &c).unwrap(),
            (19, 77, 29, 67)
        );
        let value = subject_brightness(&rgb, &alpha, 96, 96, &c).unwrap();
        assert_eq!(value.len(), 56 * 36);
        assert!(value.iter().all(|&v| v == 0.8));
    }
}
