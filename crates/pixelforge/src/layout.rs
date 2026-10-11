//! Output geometry and the input resampling that does not filter: the outline margin, the
//! output size and canvas layout (`resolve_layout` with the tileset fit), the `fit="crop"`
//! box, the integer repeat of small inputs, the box pre-reduction of large ones and the
//! seamless wrap padding. Ported from `pixelforge/preprocess.py`; every result is bitwise
//! identical to the Python version (docs/rust-migration-plan.md, Section 4.2).
//!
//! Python's `round()` is `f64::round_ties_even`, `//` on the non-negative sizes here is
//! integer division, `math.ceil(a / b)` is the ceiling of the float quotient. Images are
//! row-major slices with explicit `(h, w)`: masks and weights `h * w`, RGB `h * w * 3`.

use crate::config::{Config, ConfigError, PRESET_LONGEST_EDGE};
use crate::mask::{fill_transparent, PixelforgeError};
use crate::npsum::pairwise_sum;

/// Output pixels of wrapped context added on every side in seamless mode (Section 9: the
/// input is wrap-padded by 2·r input pixels, i.e. 2 output pixels).
// DEVIATION: Section 9 — the pad is fixed in output pixels (2·rx × 2·ry input pixels).
pub const SEAMLESS_PAD_OUT: i64 = 2;

/// `fit="stretch"` warns when it changes the aspect ratio by more than this fraction.
pub const STRETCH_WARN_ASPECT: f64 = 0.02;

/// `pixelforge.io.MAX_INPUT_PIXELS`: the largest image the pipeline accepts, also the bound
/// on the repeated input (defined in `io`).
pub use crate::io::MAX_INPUT_PIXELS;

/// The canvas layout `(inner width, inner height, left, top)`: the image's size and offset
/// inside the area left by the outline margin.
pub type Layout = (i64, i64, i64, i64);

/// `resolve_layout`'s result: the output size, the layout when the image is smaller than the
/// canvas, and the texts of the `UserWarning`s Python would emit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub out_width: i64,
    pub out_height: i64,
    pub layout: Option<Layout>,
    pub warnings: Vec<String>,
}

/// Output pixels per side reserved inside the canvas for the outline ring: 1 whenever an
/// outline is drawn and `tileset` is off, else 0.
pub fn outline_margin(config: &Config) -> i64 {
    if config.outline != "none" && !config.tileset {
        1
    } else {
        0
    }
}

/// How a derived size is made a multiple of `tile_size`: `config.fit`, or `"stretch"` in
/// seamless mode; `None` when nothing is derived (no tileset, a canvas, or both `out_width`
/// and `out_height` given).
pub fn fit_mode(config: &Config) -> Option<&str> {
    if !config.tileset
        || config.canvas.is_some()
        || (config.out_width.is_some() && config.out_height.is_some())
    {
        return None;
    }
    Some(if config.seamless {
        "stretch"
    } else {
        config.fit.as_str()
    })
}

/// Python's `round(x)` for the sizes here.
fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Python's `int / int` (true division) for operands below 2^53, where it is the
/// correctly rounded quotient, as is the float64 division of the converted operands.
fn div(a: i64, b: i64) -> f64 {
    a as f64 / b as f64
}

/// `resolve_dims`: the final output `(width, height)`, outline margin included.
pub fn resolve_dims(width: i64, height: i64, config: &Config) -> Result<(i64, i64), ConfigError> {
    let r = resolve_layout(width, height, config)?;
    Ok((r.out_width, r.out_height))
}

/// `resolve_layout(width, height, config)`: `(out_width, out_height, layout)`, the layout
/// being `None` unless the image is smaller than the canvas (`canvas`, or `fit="pad"` with a
/// tileset).
pub fn resolve_layout(width: i64, height: i64, config: &Config) -> Result<Resolved, ConfigError> {
    let m2 = 2 * outline_margin(config);
    if let Some((cw, ch)) = config.canvas_size() {
        let (aw, ah) = (cw - m2, ch - m2);
        let mut scaled = None;
        if let Some(scale) = config.scale {
            scaled = Some((
                pyround(width as f64 / scale).max(1),
                pyround(height as f64 / scale).max(1),
            ));
        }
        let (iw, ih) = match scaled {
            Some((iw, ih)) if iw <= aw && ih <= ah => (iw, ih),
            // Fit inside, preserving aspect; the longer relative side fills the canvas exactly.
            _ if width * ah >= height * aw => (aw, ah.min(pyround(div(aw * height, width)).max(1))),
            _ => (aw.min(pyround(div(ah * width, height)).max(1)), ah),
        };
        return Ok(Resolved {
            out_width: cw,
            out_height: ch,
            layout: Some((iw, ih, (aw - iw).div_euclid(2), (ah - ih).div_euclid(2))),
            warnings: Vec::new(),
        });
    }
    let (mut ow, mut oh) = (config.out_width, config.out_height);
    let mut fixed = None; // the axis whose size is a target: 0 = width, 1 = height
    if let Some(scale) = config.scale {
        ow = Some(pyround(width as f64 / scale).max(8));
        oh = Some(pyround(height as f64 / scale).max(8));
    } else {
        match (ow, oh) {
            (None, None) => {
                let longest = PRESET_LONGEST_EDGE
                    .iter()
                    .find(|(name, _)| *name == config.preset)
                    .map(|&(_, edge)| edge)
                    .expect("validated preset");
                if width >= height {
                    ow = Some(longest);
                    oh = Some((pyround(div((longest - m2) * height, width)) + m2).max(8));
                    fixed = Some(0);
                } else {
                    ow = Some((pyround(div((longest - m2) * width, height)) + m2).max(8));
                    oh = Some(longest);
                    fixed = Some(1);
                }
            }
            (None, Some(h)) => {
                ow = Some((pyround(div((h - m2) * width, height)) + m2).max(8));
                fixed = Some(1);
            }
            (Some(w), None) => {
                oh = Some((pyround(div((w - m2) * height, width)) + m2).max(8));
                fixed = Some(0);
            }
            (Some(_), Some(_)) => {}
        }
    }
    let (ow, oh) = (ow.expect("width resolved"), oh.expect("height resolved"));
    match fit_mode(config) {
        None => Ok(Resolved {
            out_width: ow,
            out_height: oh,
            layout: None,
            warnings: Vec::new(),
        }),
        Some(mode) => fit_tiles(width, height, (ow, oh), fixed, mode, config.tile_size),
    }
}

/// `_fit_tiles`: `resolve_layout` for a derived tileset size (the outline margin is 0 with
/// tilesets). The target axis is rounded down to a multiple of `t` and the other re-derived;
/// the canvas rounds each size up to a multiple.
fn fit_tiles(
    width: i64,
    height: i64,
    size: (i64, i64),
    fixed: Option<usize>,
    mode: &str,
    t: i64,
) -> Result<Resolved, ConfigError> {
    // DEVIATION: Section 5 step 2f rounds both dims up, which stretches the image and can exceed
    // the target edge; that is fit="stretch" now, and the default "pad" keeps the aspect.
    let longest = size.0.max(size.1);
    if t > longest {
        return Err(ConfigError(format!(
            "tile_size ({t}) exceeds the longest output edge ({longest}) derived for a \
             {width}x{height} input; lower tile_size or raise the output size"
        )));
    }
    let mut inner = [size.0, size.1];
    if let Some(fixed) = fixed {
        let dims = [width, height];
        inner[fixed] = (size_at(size, fixed) / t) * t;
        inner[1 - fixed] = pyround(div(inner[fixed] * dims[1 - fixed], dims[fixed])).max(1);
    }
    let cw = (div(inner[0], t)).ceil() as i64 * t;
    let ch = (div(inner[1], t)).ceil() as i64 * t;
    let mut warnings = Vec::new();
    if mode == "stretch" {
        let (source, result) = (div(width, height), div(cw, ch));
        if (result / source - 1.0).abs() > STRETCH_WARN_ASPECT {
            warnings.push(format!(
                "fit='stretch' changes the aspect ratio from {source:.3} ({width}x{height}) to \
                 {result:.3} ({cw}x{ch}) to fit tile_size={t}; use fit='pad' or 'crop' to keep it"
            ));
        }
    }
    let layout = if mode != "pad" || (inner[0], inner[1]) == (cw, ch) {
        None
    } else {
        let (iw, ih) = (inner[0], inner[1]);
        Some((iw, ih, (cw - iw).div_euclid(2), (ch - ih).div_euclid(2)))
    };
    Ok(Resolved {
        out_width: cw,
        out_height: ch,
        layout,
        warnings,
    })
}

fn size_at(size: (i64, i64), axis: usize) -> i64 {
    if axis == 0 {
        size.0
    } else {
        size.1
    }
}

/// `fit_crop`: the centred `(y0, y1, x0, x1)` of a `width`×`height` input with the aspect of
/// the output; the axis in excess is cropped, the other kept whole.
pub fn fit_crop(width: i64, height: i64, out_width: i64, out_height: i64) -> (i64, i64, i64, i64) {
    if width * out_height > height * out_width {
        let cw = pyround(div(height * out_width, out_height)).max(1);
        let x0 = (width - cw).div_euclid(2);
        return (0, height, x0, x0 + cw);
    }
    let ch = pyround(div(width * out_height, out_width)).max(1);
    let y0 = (height - ch).div_euclid(2);
    (y0, y0 + ch, 0, width)
}

/// The output geometry of a preprocessed image (`Preprocessed`'s size fields and helpers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Final output size, without the seamless padding.
    pub out_width: i64,
    pub out_height: i64,
    /// Output pixels of wrap padding per side (0 unless seamless).
    pub pad_out: i64,
    /// Output pixels per side reserved for the outline ring (0 or 1).
    pub outline_margin: i64,
    /// The canvas layout, when the image is smaller than the canvas.
    pub layout: Option<Layout>,
}

impl Geometry {
    /// Width the image itself is downscaled to (the outline margin is added after).
    pub fn inner_width(&self) -> i64 {
        match self.layout {
            Some(l) => l.0,
            None => self.out_width - 2 * self.outline_margin,
        }
    }

    pub fn inner_height(&self) -> i64 {
        match self.layout {
            Some(l) => l.1,
            None => self.out_height - 2 * self.outline_margin,
        }
    }

    /// Transparent `((top, bottom), (left, right))` padding from the inner size to the canvas,
    /// outline margin included.
    pub fn canvas_pad(&self) -> ((i64, i64), (i64, i64)) {
        let (left, top) = match self.layout {
            Some(l) => (l.2, l.3),
            None => (0, 0),
        };
        let (left, top) = (left + self.outline_margin, top + self.outline_margin);
        (
            (top, self.out_height - self.inner_height() - top),
            (left, self.out_width - self.inner_width() - left),
        )
    }

    pub fn target_width(&self) -> i64 {
        self.inner_width() + 2 * self.pad_out
    }

    pub fn target_height(&self) -> i64 {
        self.inner_height() + 2 * self.pad_out
    }
}

// ------------------------------------------------------------------ repeat

/// The integer factor `run` repeats a small `w`×`h` input by so every output pixel is backed
/// by an input pixel: `max(ceil(out_w / w), ceil(out_h / h))`. An error (Python's message) when
/// the repeated input would exceed `MAX_INPUT_PIXELS`.
// DEVIATION: Section 5 — inputs smaller than the output are repeated by an integer factor
// first, so that every output pixel is backed by at least one input pixel in all three
// downscalers (otherwise gerstner would leave superpixels without pixels).
pub fn repeat_factor(h: usize, w: usize, out_w: i64, out_h: i64) -> Result<usize, PixelforgeError> {
    let fx = (out_w as f64 / w as f64).ceil() as i64;
    let fy = (out_h as f64 / h as f64).ceil() as i64;
    let factor = fx.max(fy);
    if factor <= 1 {
        return Ok(1);
    }
    // Checked before allocating: a 2000×1 input with a square output would otherwise be
    // repeated into hundreds of millions of float64 pixels.
    let total = (h as u128) * (w as u128) * (factor as u128) * (factor as u128);
    if total > MAX_INPUT_PIXELS as u128 {
        return Err(PixelforgeError(format!(
            "a {w}x{h} input is too small for a {out_w}x{out_h} output: repeating it \
             {factor}x per axis would exceed {MAX_INPUT_PIXELS} pixels"
        )));
    }
    Ok(factor as usize)
}

fn repeat_plane<T: Copy>(data: &[T], h: usize, w: usize, ch: usize, f: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len() * f * f);
    for y in 0..h {
        let row = &data[y * w * ch..(y + 1) * w * ch];
        let start = out.len();
        for px in row.chunks_exact(ch) {
            for _ in 0..f {
                out.extend_from_slice(px);
            }
        }
        let end = out.len();
        for _ in 1..f {
            out.extend_from_within(start..end);
        }
    }
    out
}

/// `np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)` of the RGB, mask and weight.
pub fn repeat(
    rgb: &[f64],
    mask: &[bool],
    weight: &[f64],
    h: usize,
    w: usize,
    factor: usize,
) -> (Vec<f64>, Vec<bool>, Vec<f64>) {
    assert_eq!(rgb.len(), h * w * 3, "repeat: rgb length");
    assert_eq!(mask.len(), h * w, "repeat: mask length");
    assert_eq!(weight.len(), h * w, "repeat: weight length");
    (
        repeat_plane(rgb, h, w, 3, factor),
        repeat_plane(mask, h, w, 1, factor),
        repeat_plane(weight, h, w, 1, factor),
    )
}

// ------------------------------------------------------------------ pre-reduction

/// `prereduce_factor`: the largest integer box factor that keeps the input at least
/// `max_ratio` × the output on both axes (1 = none; `max_ratio` 0 disables it).
pub fn prereduce_factor(h: i64, w: i64, out_h: i64, out_w: i64, max_ratio: i64) -> i64 {
    if max_ratio <= 0 {
        return 1;
    }
    ((h / out_h).min(w / out_w) / max_ratio).max(1)
}

/// `prereduce`'s result: the reduced image (`h`×`w`), its mask, and the block-mean weight
/// when one was given.
#[derive(Clone, Debug, PartialEq)]
pub struct Prereduced {
    pub rgb: Vec<f64>,
    pub mask: Vec<bool>,
    pub weight: Option<Vec<f64>>,
    pub h: usize,
    pub w: usize,
}

/// `prereduce(rgb, mask, factor, weight)`: exact area means over `factor`×`factor` blocks,
/// trailing partial blocks dropped. Transparent pixels first take the nearest opaque colour; a
/// block is opaque when at least half of its pixels are; the weight's block mean is zeroed on
/// transparent blocks.
///
/// The sums follow numpy's reduction order for `.mean(axis=(1, 3))` of the reshaped arrays,
/// established empirically against numpy 2.4 (tests/test_layout_native.py): the RGB block
/// (5-D, channel innermost) is added left to right in row-major order; the weight block (4-D)
/// is, per block row, a pairwise sum of that row added to the running total, except with a
/// single block column (`w / factor == 1`), where numpy reduces the whole block as one
/// pairwise sum in row-major order. The same holds when Python's weight is a strided view
/// (the crop box), checked against numpy for blocks of up to 19×19 pixels.
// DEVIATION: Section 5 — inputs far larger than the output are first box-reduced by an
// integer factor (exact area mean), so that denoise and the downscalers run on less than
// 2 × prereduce_max_ratio × the output resolution. Kopf's kernels span 4×4 output pixels, so
// at these ratios the reduction is nearly lossless, and it bounds time and memory. Up to
// factor − 1 trailing rows/columns that do not fill a block are dropped.
pub fn prereduce(
    rgb: &[f64],
    mask: &[bool],
    weight: Option<&[f64]>,
    h: usize,
    w: usize,
    factor: usize,
) -> Prereduced {
    assert!(factor >= 1, "prereduce: factor must be >= 1");
    assert_eq!(rgb.len(), h * w * 3, "prereduce: rgb length");
    assert_eq!(mask.len(), h * w, "prereduce: mask length");
    let (hb, wb) = (h / factor, w / factor);
    let n = (factor * factor) as f64;
    let filled = fill_transparent(rgb, mask, h, w);
    let mut small = Vec::with_capacity(hb * wb * 3);
    let mut small_mask = Vec::with_capacity(hb * wb);
    for by in 0..hb {
        for bx in 0..wb {
            for c in 0..3 {
                // numpy starts these reductions from +0.0 (np.sum([-0.0]) is 0.0).
                let mut s = 0.0;
                for y in by * factor..(by + 1) * factor {
                    for x in bx * factor..(bx + 1) * factor {
                        s += filled[(y * w + x) * 3 + c];
                    }
                }
                small.push(s / n);
            }
            let mut count = 0usize;
            for y in by * factor..(by + 1) * factor {
                count += mask[y * w + bx * factor..y * w + (bx + 1) * factor]
                    .iter()
                    .filter(|&&m| m)
                    .count();
            }
            small_mask.push(count as f64 / n >= 0.5);
        }
    }
    let small_weight = weight.map(|weight| {
        assert_eq!(weight.len(), h * w, "prereduce: weight length");
        let mut out = Vec::with_capacity(hb * wb);
        let mut block = Vec::with_capacity(factor * factor);
        for by in 0..hb {
            for bx in 0..wb {
                let rows = (by * factor..(by + 1) * factor)
                    .map(|y| &weight[y * w + bx * factor..y * w + (bx + 1) * factor]);
                let s = if wb == 1 {
                    block.clear();
                    for row in rows {
                        block.extend_from_slice(row);
                    }
                    pairwise_sum(&block)
                } else {
                    let mut s = 0.0;
                    for row in rows {
                        s += pairwise_sum(row);
                    }
                    s
                };
                out.push(if small_mask[by * wb + bx] { s / n } else { 0.0 });
            }
        }
        out
    });
    Prereduced {
        rgb: small,
        mask: small_mask,
        weight: small_weight,
        h: hb,
        w: wb,
    }
}

// ------------------------------------------------------------------ seamless padding

/// The wrap padding `(pad_y, pad_x)` in input pixels for `SEAMLESS_PAD_OUT` output pixels:
/// `ceil(pad_out * h / inner_h)` and `ceil(pad_out * w / inner_w)`.
pub fn seamless_pads(h: usize, w: usize, inner_h: i64, inner_w: i64) -> (usize, usize) {
    let pad_y = ((SEAMLESS_PAD_OUT * h as i64) as f64 / inner_h as f64).ceil() as usize;
    let pad_x = ((SEAMLESS_PAD_OUT * w as i64) as f64 / inner_w as f64).ceil() as usize;
    (pad_y, pad_x)
}

fn wrap_plane<T: Copy>(data: &[T], h: usize, w: usize, ch: usize, py: usize, px: usize) -> Vec<T> {
    let (oh, ow) = (h + 2 * py, w + 2 * px);
    let mut out = Vec::with_capacity(oh * ow * ch);
    for y in 0..oh {
        let sy = (y + h - py % h) % h;
        for x in 0..ow {
            let sx = (x + w - px % w) % w;
            let i = (sy * w + sx) * ch;
            out.extend_from_slice(&data[i..i + ch]);
        }
    }
    out
}

/// `np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode="wrap")` of the RGB, mask and weight:
/// the periodic extension, also for pads larger than the image.
pub fn wrap_pad(
    rgb: &[f64],
    mask: &[bool],
    weight: &[f64],
    h: usize,
    w: usize,
    pad_y: usize,
    pad_x: usize,
) -> (Vec<f64>, Vec<bool>, Vec<f64>) {
    assert!(h > 0 && w > 0, "wrap_pad: empty image");
    assert_eq!(rgb.len(), h * w * 3, "wrap_pad: rgb length");
    assert_eq!(mask.len(), h * w, "wrap_pad: mask length");
    assert_eq!(weight.len(), h * w, "wrap_pad: weight length");
    (
        wrap_plane(rgb, h, w, 3, pad_y, pad_x),
        wrap_plane(mask, h, w, 1, pad_y, pad_x),
        wrap_plane(weight, h, w, 1, pad_y, pad_x),
    )
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

    fn layout(w: i64, h: i64, c: &Config) -> (i64, i64, Option<Layout>) {
        let r = resolve_layout(w, h, c).unwrap();
        (r.out_width, r.out_height, r.layout)
    }

    #[test]
    fn canvas_layouts() {
        // tests/test_preprocess.py::test_resolve_dims_with_canvas
        let sprite = config(&[("canvas", s("48x32"))]);
        assert_eq!(layout(100, 50, &sprite), (48, 32, Some((46, 23, 0, 3))));
        assert_eq!(layout(50, 100, &sprite), (48, 32, Some((15, 30, 15, 0))));
        let plain = config(&[("canvas", s("48x32")), ("outline", s("none"))]);
        assert_eq!(layout(100, 50, &plain), (48, 32, Some((48, 24, 0, 4))));
        let scaled = config(&[("canvas", s("48x32")), ("scale", Value::Float(1.0))]);
        assert_eq!(layout(20, 10, &scaled), (48, 32, Some((20, 10, 13, 10))));
        assert_eq!(layout(200, 10, &scaled), (48, 32, Some((46, 2, 0, 14))));
    }

    #[test]
    fn tileset_fits() {
        // tests/test_preprocess.py::test_tileset_fit_pad_keeps_aspect_and_pads_centered
        let pad = config(&[("preset", s("background")), ("out_width", Value::Int(64))]);
        assert_eq!(layout(160, 96, &pad), (64, 48, Some((64, 38, 0, 5))));
        for fit in ["stretch", "crop"] {
            let c = config(&[
                ("preset", s("background")),
                ("out_width", Value::Int(64)),
                ("fit", s(fit)),
            ]);
            let r = resolve_layout(160, 96, &c).unwrap();
            assert_eq!((r.out_width, r.out_height, r.layout), (64, 48, None));
            assert_eq!(r.warnings.len(), usize::from(fit == "stretch"));
        }
        assert_eq!(fit_crop(160, 96, 64, 48), (0, 96, 16, 144));
        assert_eq!(fit_crop(96, 160, 64, 48), (44, 116, 0, 96));
        let big = config(&[("preset", s("background")), ("tile_size", Value::Int(512))]);
        let err = resolve_layout(300, 200, &big).unwrap_err();
        assert!(
            err.0
                .starts_with("tile_size (512) exceeds the longest output edge (256)"),
            "{}",
            err.0
        );
    }

    #[test]
    fn stretch_warning_text() {
        let c = config(&[
            ("preset", s("background")),
            ("out_width", Value::Int(64)),
            ("fit", s("stretch")),
        ]);
        let r = resolve_layout(160, 96, &c).unwrap();
        assert_eq!(
            r.warnings,
            vec![
                "fit='stretch' changes the aspect ratio from 1.667 (160x96) to 1.333 (64x48) \
                 to fit tile_size=16; use fit='pad' or 'crop' to keep it"
                    .to_string()
            ]
        );
        // Python's '.3f' rounds exact ties to even, as Rust's formatting does.
        assert_eq!(format!("{:.3}", 0.0625), "0.062");
        assert_eq!(format!("{:.3}", 0.1875), "0.188");
    }

    #[test]
    fn geometry_helpers() {
        let g = Geometry {
            out_width: 48,
            out_height: 32,
            pad_out: 0,
            outline_margin: 1,
            layout: Some((46, 23, 0, 3)),
        };
        assert_eq!((g.inner_width(), g.inner_height()), (46, 23));
        assert_eq!(g.canvas_pad(), ((4, 5), (1, 1)));
        let s = Geometry {
            layout: None,
            pad_out: 2,
            ..g
        };
        assert_eq!((s.target_width(), s.target_height()), (50, 34));
        assert_eq!(s.canvas_pad(), ((1, 1), (1, 1)));
    }

    #[test]
    fn repeat_and_its_limit() {
        assert_eq!(repeat_factor(10, 20, 64, 32), Ok(4));
        assert_eq!(repeat_factor(100, 100, 64, 32), Ok(1));
        let err = repeat_factor(1, 2000, 2000, 2000).unwrap_err();
        assert_eq!(
            err.0,
            "a 2000x1 input is too small for a 2000x2000 output: repeating it 2000x per axis \
             would exceed 24000000 pixels"
        );
        let (rgb, mask, weight) = repeat(
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            &[true, false],
            &[1.0, 0.0],
            1,
            2,
            2,
        );
        assert_eq!(
            mask,
            vec![true, true, false, false, true, true, false, false]
        );
        assert_eq!(weight, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(&rgb[..6], &[0.0, 0.1, 0.2, 0.0, 0.1, 0.2]);
        assert_eq!(&rgb[12..15], &[0.0, 0.1, 0.2]);
    }

    #[test]
    fn prereduce_block_means() {
        // tests/test_preprocess.py::test_prereduce_factor_and_disable
        assert_eq!(prereduce_factor(1024, 1024, 64, 64, 8), 2);
        assert_eq!(prereduce_factor(1024, 1024, 64, 64, 0), 1);
        assert_eq!(prereduce_factor(100, 100, 64, 64, 8), 1);
        let (h, w) = (5, 4);
        let rgb: Vec<f64> = (0..h * w * 3).map(|i| i as f64).collect();
        let mask = vec![true; h * w];
        let weight = vec![0.5; h * w];
        let r = prereduce(&rgb, &mask, Some(&weight), h, w, 2);
        assert_eq!((r.h, r.w), (2, 2));
        // Block (0, 0): pixels 0, 1, 4, 5 → channel 0 values 0, 3, 12, 15.
        assert_eq!(r.rgb[0], 7.5);
        assert_eq!(r.weight, Some(vec![0.5; 4]));
        // Coverage below one half: transparent block, zero weight.
        let mut sparse = vec![false; h * w];
        sparse[0] = true;
        let r = prereduce(&rgb, &sparse, Some(&weight), h, w, 2);
        assert_eq!(r.mask, vec![false; 4]);
        assert_eq!(r.weight, Some(vec![0.0; 4]));
        assert_eq!(&r.rgb[..3], &[0.0, 1.0, 2.0]); // filled from pixel 0
    }

    #[test]
    fn wrap_padding_is_periodic() {
        let rgb: Vec<f64> = (0..6).map(|i| i as f64).collect(); // 1×2 RGB
        let (out, mask, _) = wrap_pad(&rgb, &[true, false], &[1.0, 0.0], 1, 2, 2, 3);
        assert_eq!(mask.len(), 5 * 8);
        // Row of 8: x - 3 mod 2 → 1, 0, 1, 0, 1, 0, 1, 0.
        assert_eq!(
            &mask[..8],
            &[false, true, false, true, false, true, false, true]
        );
        assert_eq!(&out[..3], &[3.0, 4.0, 5.0]);
        assert_eq!(seamless_pads(100, 50, 16, 16), (13, 7));
    }
}
