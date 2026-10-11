//! The `enhance` pre-pass of `pixelforge/preprocess.py` (levels stretch, local contrast,
//! ink), with its helpers `_level_pixels`, `enhance_levels` (`np.percentile`), `_stretch`,
//! `_gaussian` and `_blur`, ported formula for formula.
//!
//! Images are row-major `(h, w, 3)` float64 slices, masks and weights `(h, w)`. Every step
//! keeps numpy's operation order; the only transcendental functions are the Gaussian kernel's
//! `exp` (in `filters`) and `expm1`, both from `libm`, which with scipy's FMA-contracted
//! correlation (see `filters`) is where results may differ from numpy's in the last bits.
//! Percentiles, levels, `stretch` and the running-mean path of `gaussian` are bitwise.

use crate::config::Config;
use crate::filters::{self, clip, np_maximum, Mode};

/// Rec. 709 luma weights (`_LUMA`).
pub const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];
/// Percentiles of the subject's brightness mapped to black and white (`ENHANCE_LEVELS`).
pub const ENHANCE_LEVELS: [f64; 2] = [1.0, 99.0];
/// The levels gain is capped at 1 / this (`ENHANCE_MIN_RANGE`).
pub const ENHANCE_MIN_RANGE: f64 = 0.2;
/// The ink difference of Gaussians, in output pixels (`INK_SIGMAS`).
pub const INK_SIGMAS: [f64; 2] = [0.5, 1.6];
/// `INK_ONSET`, `INK_RANGE`.
pub const INK_ONSET: f64 = 0.02;
pub const INK_RANGE: f64 = 0.04;
/// From this sigma (input pixels) up a blur axis uses three running means (`BLUR_BOX_SIGMA`).
pub const BLUR_BOX_SIGMA: f64 = 4.0;

/// `rgb @ _LUMA` per pixel. numpy 2.4.6 on macOS arm64 computes this matrix-vector product
/// as `(r * 0.2126 + g * 0.7152) + b * 0.0722`, left to right (probed bitwise on random
/// arrays up to 301×257).
pub fn luma(rgb: &[f64]) -> Vec<f64> {
    rgb.chunks_exact(3)
        .map(|p| p[0] * LUMA[0] + p[1] * LUMA[1] + p[2] * LUMA[2])
        .collect()
}

/// `rgb.max(axis=-1)`: numpy's `maximum` reduction over the three channels.
pub fn peak(rgb: &[f64]) -> Vec<f64> {
    rgb.chunks_exact(3)
        .map(|p| np_maximum(np_maximum(p[0], p[1]), p[2]))
        .collect()
}

/// `_level_pixels(value, mask, weight)`: the values of the mostly opaque pixels
/// (`mask & (weight >= 0.5)`), or of every masked pixel when there are none, row-major.
pub fn level_pixels(value: &[f64], mask: &[bool], weight: &[f64]) -> Vec<f64> {
    let solid: Vec<bool> = mask
        .iter()
        .zip(weight)
        .map(|(&m, &wt)| m && wt >= 0.5)
        .collect();
    let select = if solid.iter().any(|&s| s) {
        &solid[..]
    } else {
        mask
    };
    value
        .iter()
        .zip(select)
        .filter(|(_, &s)| s)
        .map(|(&v, _)| v)
        .collect()
}

/// `subject_brightness` after `_subject`: the largest channel of the level pixels of an
/// `(h, w, 3)` image with its mask and weight.
pub fn subject_brightness(rgb: &[f64], mask: &[bool], weight: &[f64]) -> Vec<f64> {
    level_pixels(&peak(rgb), mask, weight)
}

/// numpy's `_lerp(a, b, t)`: `a + (b - a) * t`, or `b - (b - a) * (1 - t)` when `t >= 0.5`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `np.percentile(values, qs)` with the default linear method, bit for bit: per `q`,
/// `index = (n - 1) * (q / 100)`, the order statistics at `floor(index)` and the next one,
/// interpolated by numpy's `_lerp` with `gamma = index - floor(index)`. An index at or above
/// `n - 1` takes the largest value (numpy then has `gamma = index + 1` and both neighbours
/// equal). NaN anywhere gives NaN, as does an empty array.
pub fn percentile(values: &[f64], qs: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 || values.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; qs.len()];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    qs.iter()
        .map(|&q| {
            let quantile = q / 100.0;
            let index = (n - 1) as f64 * quantile;
            let (prev, next, gamma) = if index >= (n - 1) as f64 {
                (n - 1, n - 1, index - (-1.0))
            } else {
                let lo = index.floor();
                (lo as usize, lo as usize + 1, index - lo)
            };
            lerp(sorted[prev], sorted[next], gamma)
        })
        .collect()
}

/// `enhance_levels(value, config)`: `enhance_black` / `enhance_white` where set, else the
/// ENHANCE_LEVELS percentiles of `value`.
pub fn enhance_levels(value: &[f64], config: &Config) -> (f64, f64) {
    let p = percentile(value, &ENHANCE_LEVELS);
    (
        config.enhance_black.unwrap_or(p[0]),
        config.enhance_white.unwrap_or(p[1]),
    )
}

/// `_stretch(value, lo, hi)`: map `[lo, hi]` to `[0, 1]` with the gain capped at
/// 1 / ENHANCE_MIN_RANGE.
pub fn stretch(value: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let spread = hi - lo;
    if spread <= 1e-6 {
        return value.to_vec();
    }
    if spread >= 1.0 {
        return value
            .iter()
            .map(|&v| clip((v - lo) / spread, 0.0, 1.0))
            .collect();
    }
    // Python's max(spread, ENHANCE_MIN_RANGE): the first argument unless the second is larger.
    let floor = if ENHANCE_MIN_RANGE > spread {
        ENHANCE_MIN_RANGE
    } else {
        spread
    };
    let gain = 1.0 / floor;
    let new_lo = lo * (1.0 - spread * gain) / (1.0 - spread);
    value
        .iter()
        .map(|&v| clip(new_lo + gain * (v - lo), 0.0, 1.0))
        .collect()
}

/// The three running-mean widths `_gaussian` uses for one axis of length `len` at `sigma`
/// (>= BLUR_BOX_SIGMA), with Python's `int()`, `round()` (half to even) and `%`.
pub fn box_widths(sigma: f64, len: usize) -> [usize; 3] {
    let ideal = (12.0 * sigma * sigma / 3.0 + 1.0).sqrt();
    let ii = ideal as i64; // int(ideal): truncation (ideal > 0)
    let low = ii - (1 - ii.rem_euclid(2));
    let numerator = 12.0 * sigma * sigma - (3 * low * low) as f64 - (12 * low) as f64 - 9.0;
    let narrow = (numerator / (-4.0 * low as f64 - 4.0)).round_ties_even() as i64;
    let n = len as i64;
    let limit = n - (1 - n.rem_euclid(2));
    let mut out = [0usize; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let want = if (i as i64) < narrow { low } else { low + 2 };
        // Python's min(want, limit): the first unless the second is smaller.
        *o = if limit < want { limit } else { want } as usize;
    }
    out
}

/// `_gaussian(image, sigma, mode)` on an `(h, w)` array: per axis (0 with `sigma.0`, then 1
/// with `sigma.1`), `gaussian_filter1d` below BLUR_BOX_SIGMA, else three `uniform_filter1d`
/// passes of the `box_widths`.
pub fn gaussian(image: &[f64], h: usize, w: usize, sigma: (f64, f64), mode: Mode) -> Vec<f64> {
    let mut out = image.to_vec();
    for (axis, sg) in [(0usize, sigma.0), (1, sigma.1)] {
        if sg < BLUR_BOX_SIGMA {
            out = filters::gaussian_filter1d(&out, h, w, sg, axis, mode);
            continue;
        }
        let len = if axis == 0 { h } else { w };
        for width in box_widths(sg, len) {
            out = filters::uniform_filter1d(&out, h, w, width, axis, mode);
        }
    }
    out
}

/// `_blur(values, cover, sigma, wrap)`: the normalised convolution
/// `_gaussian(values * cover) / maximum(_gaussian(cover), 1e-6)`.
pub fn blur(
    values: &[f64],
    cover: &[f64],
    h: usize,
    w: usize,
    sigma: (f64, f64),
    wrap: bool,
) -> Vec<f64> {
    let mode = if wrap { Mode::Wrap } else { Mode::Nearest };
    let weighted: Vec<f64> = values.iter().zip(cover).map(|(&v, &c)| v * c).collect();
    let num = gaussian(&weighted, h, w, sigma, mode);
    let den = gaussian(cover, h, w, sigma, mode);
    num.iter()
        .zip(&den)
        .map(|(&a, &b)| a / np_maximum(b, 1e-6))
        .collect()
}

/// `preprocess.enhance(rgb, mask, cell, config, weight, wrap)` on an `(h, w, 3)` image:
/// levels stretch, local contrast (`config.enhance`, radius `config.enhance_radius`) and ink
/// (`config.ink`), on luminance, each pixel's RGB scaled by one factor. `cell` is the (y, x)
/// number of input pixels per output pixel; `weight` defaults to the mask.
#[allow(clippy::too_many_arguments)]
pub fn enhance(
    rgb: &[f64],
    mask: &[bool],
    h: usize,
    w: usize,
    cell: (f64, f64),
    config: &Config,
    weight: Option<&[f64]>,
    wrap: bool,
) -> Vec<f64> {
    let n = h * w;
    assert_eq!(rgb.len(), n * 3, "enhance: rgb length is not h * w * 3");
    assert_eq!(mask.len(), n, "enhance: mask length is not h * w");
    if let Some(wt) = weight {
        assert_eq!(wt.len(), n, "enhance: weight length is not h * w");
    }
    if config.enhance <= 0.0 && config.ink <= 0.0 {
        return rgb.to_vec();
    }
    let cover: Vec<f64> = match weight {
        None => mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect(),
        Some(wt) => mask
            .iter()
            .zip(wt)
            .map(|(&m, &v)| if m { v } else { 0.0 })
            .collect(),
    };
    let blur_at = |values: &[f64], sigma_out: f64| {
        blur(
            values,
            &cover,
            h,
            w,
            (sigma_out * cell.0, sigma_out * cell.1),
            wrap,
        )
    };

    let luma = luma(rgb);
    let peak = peak(rgb);
    let (lo, hi) = enhance_levels(&level_pixels(&peak, mask, &cover), config);
    let stretched_peak = stretch(&peak, lo, hi);
    let stretched: Vec<f64> = (0..n)
        .map(|i| luma[i] * stretched_peak[i] / np_maximum(peak[i], 1e-6))
        .collect();
    let mut out = luma.clone();
    if config.enhance > 0.0 {
        // Python's min(config.enhance, 1.0).
        let amount = if 1.0 < config.enhance {
            1.0
        } else {
            config.enhance
        };
        let base: Vec<f64> = (0..n)
            .map(|i| luma[i] + amount * (stretched[i] - luma[i]))
            .collect();
        let blurred = blur_at(&base, config.enhance_radius);
        out = (0..n)
            .map(|i| clip(base[i] + config.enhance * (base[i] - blurred[i]), 0.0, 1.0))
            .collect();
    }
    if config.ink > 0.0 {
        let wide = blur_at(&stretched, INK_SIGMAS[1]);
        let narrow = blur_at(&stretched, INK_SIGMAS[0]);
        for i in 0..n {
            let dog = wide[i] - narrow[i];
            out[i] *= 1.0 - config.ink * clip((dog - INK_ONSET) / INK_RANGE, 0.0, 1.0);
        }
    }
    let mut result = rgb.to_vec();
    for i in 0..n {
        let (l, p) = (luma[i], peak[i]);
        let ceiling = if p > 1e-6 {
            l / np_maximum(p, 1e-6)
        } else {
            1.0
        };
        let headroom = np_maximum(ceiling - l, 0.0);
        let rise = np_maximum(out[i] - l, 0.0);
        let eased = headroom * -libm::expm1(-rise / np_maximum(headroom, 1e-9));
        let o = if out[i] > l { l + eased } else { out[i] };
        let gain = o / np_maximum(l, 1e-6);
        if mask[i] {
            for c in 0..3 {
                result[3 * i + c] = clip(rgb[3 * i + c] * gain, 0.0, 1.0);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_switches_formula_at_one_half() {
        assert_eq!(lerp(1.0, 3.0, 0.25), 1.5);
        assert_eq!(lerp(1.0, 3.0, 0.75), 3.0 - 2.0 * 0.25);
    }

    #[test]
    fn percentile_endpoints() {
        let v = [3.0, 1.0, 2.0];
        assert_eq!(percentile(&v, &[0.0, 50.0, 100.0]), vec![1.0, 2.0, 3.0]);
        assert_eq!(percentile(&[5.0], &[1.0, 99.0]), vec![5.0, 5.0]);
        assert!(percentile(&[], &[1.0])[0].is_nan());
    }

    #[test]
    fn box_widths_match_python() {
        // _gaussian at sigma 8 on a long axis: ideal = sqrt(257) = 16.03, low = 15,
        // narrow = round((768 - 675 - 180 - 9) / -64) = round(1.5) = 2.
        assert_eq!(box_widths(8.0, 1000), [15, 15, 17]);
        // A short axis caps the width at the largest odd length within it.
        assert_eq!(box_widths(8.0, 10), [9, 9, 9]);
    }

    #[test]
    fn stretch_cases() {
        assert_eq!(stretch(&[0.3], 0.5, 0.5), vec![0.3]);
        assert_eq!(stretch(&[0.5], -0.5, 1.5), vec![0.5]);
        let s = stretch(&[0.4, 0.6], 0.4, 0.6);
        assert!(s[0] < s[1]);
    }
}
