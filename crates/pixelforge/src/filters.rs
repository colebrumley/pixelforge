//! scipy.ndimage's 1-D filters, its 3×3 median filter and scikit-image's bilateral denoise,
//! ported operation for operation for the preprocessing stage (`pixelforge/preprocess.py`,
//! docs/rust-migration-plan.md Decision 4 and Section 4.2).
//!
//! Arrays are row-major `(h, w)` slices (images `(h, w, 3)`). The sources reproduced:
//!
//! - scipy 1.17.1, `scipy/ndimage/src/ni_filters.c` (`NI_UniformFilter1D`, `NI_Correlate1D`)
//!   and `ni_support.c` (`NI_ExtendLine`), with `_filters.py` (`gaussian_filter1d`,
//!   `_gaussian_kernel1d`). Each line is extended by `size / 2 + origin` values before and
//!   `size - size / 2 - 1 - origin` after (origin 0 here), so an even-sized window covers
//!   `x[i - size/2 ..= i + size/2 - 1]`. `nearest` repeats the edge values; `wrap` is
//!   periodic (scipy's pointer walk reads values it has already written, which is the same as
//!   `x[k mod n]` for any extension length).
//! - scikit-image 0.26.0, `restoration/_denoise.py` (`denoise_bilateral`, the colour and
//!   spatial lookup tables) and `restoration/_denoise_cy.pyx` (`_denoise_bilateral`), with
//!   `_shared/interpolation.pxd` (`get_pixel3d`; mode "edge" clamps coordinates).
//!
//! The uniform and median filters involve no transcendental function and are bitwise equal
//! to scipy. The Gaussian kernel and the bilateral lookup tables use `libm::exp` where numpy
//! uses the platform's `exp`, so they may differ in the last bit of a kernel value. Every
//! product here is rounded before it is added, as the C and Cython sources are written; the
//! macOS arm64 wheels of scipy 1.17.1 and scikit-image 0.26.0 were compiled with FMA
//! contraction (scipy's symmetric correlation fuses the last `radius % 8` terms, the ones
//! nearest the centre, and skimage's bilateral fuses `total += value * weight`), so outputs
//! there differ from this port in the last bit or two (measured: at most 2.2e-16 for the
//! Gaussian filter, 7.8e-16 for the bilateral).

use crate::npsum;
use crate::parallel;

/// scipy.ndimage's boundary modes used by the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Repeat the edge value (`mode="nearest"`).
    Nearest,
    /// Periodic (`mode="wrap"`).
    Wrap,
}

impl Mode {
    /// `"nearest"` or `"wrap"`.
    pub fn parse(name: &str) -> Option<Mode> {
        match name {
            "nearest" => Some(Mode::Nearest),
            "wrap" => Some(Mode::Wrap),
            _ => None,
        }
    }
}

/// `np.clip(v, lo, hi)` (`minimum(maximum(v, lo), hi)`): NaN propagates.
pub(crate) fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    np_minimum(np_maximum(v, lo), hi)
}

/// `np.maximum(a, b)`: `a` when `a >= b` or `a` is NaN, else `b`.
pub(crate) fn np_maximum(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// `np.minimum(a, b)`: `a` when `a <= b` or `a` is NaN, else `b`.
pub(crate) fn np_minimum(a: f64, b: f64) -> f64 {
    if a <= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// `line` extended by `before` values in front and `after` behind, per `NI_ExtendLine`.
fn extend_line(line: &[f64], before: usize, after: usize, mode: Mode, buf: &mut Vec<f64>) {
    let n = line.len() as isize;
    buf.clear();
    for k in -(before as isize)..n + after as isize {
        let j = match mode {
            Mode::Nearest => k.clamp(0, n - 1),
            Mode::Wrap => k.rem_euclid(n),
        };
        buf.push(line[j as usize]);
    }
}

/// Apply `f(line, out)` to every line of an `(h, w)` array along `axis` (0: columns, 1: rows).
fn along_axis(
    values: &[f64],
    h: usize,
    w: usize,
    axis: usize,
    func: &str,
    mut f: impl FnMut(&[f64], &mut [f64]),
) -> Vec<f64> {
    assert_eq!(values.len(), h * w, "{func}: length is not h * w");
    let mut out = vec![0.0; h * w];
    match axis {
        0 => {
            let mut line = vec![0.0; h];
            let mut res = vec![0.0; h];
            for x in 0..w {
                for y in 0..h {
                    line[y] = values[y * w + x];
                }
                f(&line, &mut res);
                for y in 0..h {
                    out[y * w + x] = res[y];
                }
            }
        }
        1 => {
            if w > 0 {
                for (src, dst) in values.chunks_exact(w).zip(out.chunks_exact_mut(w)) {
                    f(src, dst);
                }
            }
        }
        _ => panic!("{func}: axis {axis} of a 2-D array"),
    }
    out
}

/// `NI_UniformFilter1D` on one line: a running sum over the extended line, each output the
/// sum divided by `size`.
fn uniform_line(line: &[f64], out: &mut [f64], size: usize, mode: Mode, buf: &mut Vec<f64>) {
    if line.is_empty() {
        return;
    }
    let size1 = size / 2;
    let size2 = size - size1 - 1;
    extend_line(line, size1, size2, mode, buf);
    let fs = size as f64;
    let mut tmp = 0.0;
    for &v in &buf[..size] {
        tmp += v;
    }
    out[0] = tmp / fs;
    for ll in 1..line.len() {
        tmp += buf[ll - 1 + size] - buf[ll - 1];
        out[ll] = tmp / fs;
    }
}

/// `scipy.ndimage.uniform_filter1d(values, size, axis=axis, mode=mode)` on an `(h, w)`
/// float64 array (`axis` 0 or 1, `size >= 1`, origin 0), bit for bit.
pub fn uniform_filter1d(
    values: &[f64],
    h: usize,
    w: usize,
    size: usize,
    axis: usize,
    mode: Mode,
) -> Vec<f64> {
    assert!(size >= 1, "uniform_filter1d: size must be at least 1");
    let mut buf = Vec::new();
    along_axis(values, h, w, axis, "uniform_filter1d", |line, out| {
        uniform_line(line, out, size, mode, &mut buf)
    })
}

/// scipy's `_gaussian_kernel1d(sigma, 0, radius)` with `radius = int(4.0 * sigma + 0.5)`:
/// `phi = exp(-0.5 / sigma² * x²)` over `x = -radius..=radius`, divided by its (pairwise)
/// sum.
pub fn gaussian_kernel1d(sigma: f64) -> Vec<f64> {
    let truncate = 4.0;
    // int(truncate * sd + 0.5): truncation toward zero (sigma >= 0 in the pipeline).
    let radius = (truncate * sigma + 0.5) as i64;
    let sigma2 = sigma * sigma;
    let c = -0.5 / sigma2;
    let phi: Vec<f64> = (-radius..=radius)
        .map(|x| libm::exp(c * (x * x) as f64))
        .collect();
    let total = npsum::pairwise_sum(&phi);
    phi.iter().map(|&p| p / total).collect()
}

/// `NI_Correlate1D` with a symmetric odd kernel (`fw` centred on `fw[r]`): per output,
/// `x[0] * fw[0]`, then `(x[j] + x[-j]) * fw[j]` added for `j = -r..-1` in that order.
fn correlate_symmetric_line(
    line: &[f64],
    out: &mut [f64],
    weights: &[f64],
    mode: Mode,
    buf: &mut Vec<f64>,
) {
    if line.is_empty() {
        return;
    }
    let r = weights.len() / 2;
    extend_line(line, r, r, mode, buf);
    for (ll, o) in out.iter_mut().enumerate() {
        let c = ll + r;
        let mut acc = buf[c] * weights[r];
        for k in (1..=r).rev() {
            // jj = -k: (iline[-k] + iline[k]) * fw[-k]
            acc += (buf[c - k] + buf[c + k]) * weights[r - k];
        }
        *o = acc;
    }
}

/// `NI_Correlate1D`'s general path (kernel not symmetric): `x[size2] * fw[size2]`, then
/// `x[j] * fw[j]` for `j = -size1..size2-1`.
fn correlate_general_line(
    line: &[f64],
    out: &mut [f64],
    weights: &[f64],
    mode: Mode,
    buf: &mut Vec<f64>,
) {
    if line.is_empty() {
        return;
    }
    let size1 = weights.len() / 2;
    let size2 = weights.len() - size1 - 1;
    extend_line(line, size1, size2, mode, buf);
    for (ll, o) in out.iter_mut().enumerate() {
        // iline[j] = buf[ll + size1 + j], fw[j] = weights[size1 + j]
        let mut acc = buf[ll + size1 + size2] * weights[size1 + size2];
        for i in 0..size1 + size2 {
            acc += buf[ll + i] * weights[i];
        }
        *o = acc;
    }
}

/// `scipy.ndimage.correlate1d(values, weights, axis, mode=mode)` (origin 0), with scipy's
/// symmetric-kernel detection (odd length, mirrored weights within `DBL_EPSILON`).
pub fn correlate1d(
    values: &[f64],
    h: usize,
    w: usize,
    weights: &[f64],
    axis: usize,
    mode: Mode,
) -> Vec<f64> {
    assert!(!weights.is_empty(), "correlate1d: no filter weights given");
    let n = weights.len();
    let size1 = n / 2;
    let symmetric = n % 2 == 1
        && (1..=n / 2).all(|i| (weights[size1 + i] - weights[size1 - i]).abs() <= f64::EPSILON);
    // The antisymmetric path is never reached by a Gaussian kernel; scipy would take it for
    // kernels with w[c + i] == -w[c - i], which the general path computes differently.
    let antisymmetric = !symmetric
        && n % 2 == 1
        && (1..=n / 2).all(|i| (weights[size1 + i] + weights[size1 - i]).abs() <= f64::EPSILON);
    assert!(
        !antisymmetric,
        "correlate1d: antisymmetric kernels are not supported"
    );
    let mut buf = Vec::new();
    along_axis(values, h, w, axis, "correlate1d", |line, out| {
        if symmetric {
            correlate_symmetric_line(line, out, weights, mode, &mut buf)
        } else {
            correlate_general_line(line, out, weights, mode, &mut buf)
        }
    })
}

/// `scipy.ndimage.gaussian_filter1d(values, sigma, axis=axis, mode=mode)` (truncate 4.0) on
/// an `(h, w)` float64 array.
pub fn gaussian_filter1d(
    values: &[f64],
    h: usize,
    w: usize,
    sigma: f64,
    axis: usize,
    mode: Mode,
) -> Vec<f64> {
    let mut weights = gaussian_kernel1d(sigma);
    // Correlation with the reversed kernel (it is symmetric, so this changes nothing).
    weights.reverse();
    correlate1d(values, h, w, &weights, axis, mode)
}

/// `scipy.ndimage.median_filter(values, size=3, mode="nearest")` on an `(h, w)` array: the
/// 5th smallest of the 9 edge-extended neighbours, an input value, so bit for bit. A 1-D
/// array is the `(1, n)` case (each of its 3 neighbours appears three times in the window).
pub fn median_filter3_nearest(values: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(
        values.len(),
        h * w,
        "median_filter3_nearest: length is not h * w"
    );
    let mut out = vec![0.0; h * w];
    let (hi, wi) = (h as isize, w as isize);
    for y in 0..hi {
        for x in 0..wi {
            let mut win = [0.0f64; 9];
            let mut k = 0;
            for dy in -1..=1 {
                let yy = (y + dy).clamp(0, hi - 1) as usize;
                for dx in -1..=1 {
                    let xx = (x + dx).clamp(0, wi - 1) as usize;
                    win[k] = values[yy * w + xx];
                    k += 1;
                }
            }
            win.sort_by(f64::total_cmp);
            out[y as usize * w + x as usize] = win[4];
        }
    }
    out
}

/// `preprocess.denoise` with `denoise="median"`: `median_filter3_nearest` per channel of an
/// `(h, w, 3)` image.
pub fn denoise_median(rgb: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(
        rgb.len(),
        h * w * 3,
        "denoise_median: length is not h * w * 3"
    );
    let mut out = vec![0.0; h * w * 3];
    for c in 0..3 {
        let channel: Vec<f64> = (0..h * w).map(|i| rgb[3 * i + c]).collect();
        let filtered = median_filter3_nearest(&channel, h, w);
        for (i, v) in filtered.into_iter().enumerate() {
            out[3 * i + c] = v;
        }
    }
    out
}

/// scikit-image's colour-distance bins (`denoise_bilateral(bins=10000)`).
pub const BILATERAL_BINS: usize = 10000;

/// skimage's `_gaussian_weight(array, sigma²)`: `exp(-0.5 * (x**2 / sigma²))`.
fn gaussian_weight(x: f64, sigma_squared: f64) -> f64 {
    libm::exp(-0.5 * (x * x / sigma_squared))
}

/// `_compute_color_lut(bins, sigma, max_value)`: the weights at
/// `np.linspace(0, max_value, bins, endpoint=False)`.
fn color_lut(bins: usize, sigma: f64, max_value: f64) -> Vec<f64> {
    let s2 = sigma * sigma;
    let div = bins as f64;
    let delta = max_value - 0.0;
    let step = delta / div;
    (0..bins)
        .map(|i| {
            // numpy's linspace: arange * step + start, or (arange / div) * delta when the
            // step underflows to zero.
            let v = if step == 0.0 {
                (i as f64 / div) * delta
            } else {
                i as f64 * step
            } + 0.0;
            gaussian_weight(v, s2)
        })
        .collect()
}

/// `_compute_spatial_lut(win_size, sigma)`, flattened. Its grid is
/// `np.arange(-win_size // 2, win_size // 2 + 1)`: for an odd `win_size = 2k + 1` that is
/// `-(k + 1)..=k`, so `2k + 2` points per axis, while `_denoise_bilateral` indexes the
/// flattened table as `kr * win_size + kc`. Reproduced as it is.
fn spatial_lut(win_size: usize, sigma: f64) -> Vec<f64> {
    let s2 = sigma * sigma;
    let ws = win_size as i64;
    let lo = (-ws).div_euclid(2);
    let hi = ws / 2 + 1;
    let mut out = Vec::new();
    for r in lo..hi {
        for c in lo..hi {
            let d = libm::hypot(r as f64, c as f64);
            out.push(gaussian_weight(d, s2));
        }
    }
    out
}

/// numpy's `image.std()` over all elements of a contiguous array: the pairwise mean, then the
/// pairwise mean of squared deviations, square-rooted.
fn np_std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = npsum::pairwise_sum(values) / n;
    let sq: Vec<f64> = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .collect();
    (npsum::pairwise_sum(&sq) / n).sqrt()
}

/// `np.min` / `np.max` over all values (NaN propagates, as numpy's reductions do).
fn np_min_max(values: &[f64]) -> (f64, f64) {
    let mut lo = values[0];
    let mut hi = values[0];
    for &v in &values[1..] {
        lo = np_minimum(lo, v);
        hi = np_maximum(hi, v);
    }
    (lo, hi)
}

/// `preprocess.denoise` with `denoise="bilateral"`: scikit-image 0.26's
/// `denoise_bilateral(rgb, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
/// channel_axis=-1, mode="edge")`, then `clip(0, 1)`, on an `(h, w, 3)` float64 image.
///
/// skimage's steps: `win_size = max(5, 2 * ceil(3 * sigma_spatial) + 1)`; a constant image is
/// returned as it is; `sigma_color` 0 means the image's standard deviation; the colour table
/// holds `bins` weights over `[0, max)`, the spatial table is described at `spatial_lut`.
/// For each pixel and each window offset (rows, then columns, edge-clamped) the Euclidean
/// RGB distance to the centre picks the colour bin `min(int(dist * dist_scale), bins - 1)`
/// with `dist_scale = bins / dims / max_value`; the product of the two weights accumulates
/// the neighbour's channels and the total weight, and the output is their quotient.
pub fn denoise_bilateral(
    rgb: &[f64],
    h: usize,
    w: usize,
    sigma_color: f64,
    sigma_spatial: f64,
    threads: usize,
) -> Vec<f64> {
    assert_eq!(
        rgb.len(),
        h * w * 3,
        "denoise_bilateral: length is not h * w * 3"
    );
    if rgb.is_empty() {
        return Vec::new();
    }
    let win_size = 5usize.max(2 * (3.0 * sigma_spatial).ceil() as usize + 1);
    let (min_value, mut max_value) = np_min_max(rgb);
    if min_value == max_value {
        return rgb.iter().map(|&v| clip(v, 0.0, 1.0)).collect();
    }
    // `sigma_color or image.std()`: 0.0 is falsy (NaN is not).
    let sigma_color = if sigma_color == 0.0 {
        np_std(rgb)
    } else {
        sigma_color
    };
    let color_lut = color_lut(BILATERAL_BINS, sigma_color, max_value);
    let range_lut = spatial_lut(win_size, sigma_spatial);
    let shifted: Vec<f64>;
    let image: &[f64] = if min_value < 0.0 {
        shifted = rgb.iter().map(|&v| v - min_value).collect();
        max_value -= min_value;
        &shifted
    } else {
        rgb
    };
    // Cython `bins / dims` on two Py_ssize_t under cdivision: C integer division.
    let dist_scale = (BILATERAL_BINS / 3) as f64 / max_value;
    let max_bin = BILATERAL_BINS - 1;
    let ext = (win_size - 1) / 2;

    // Edge-padded copy, so the window reads need no clamping (the same values get_pixel3d's
    // mode "edge" reads).
    let pw = w + 2 * ext;
    let ph = h + 2 * ext;
    let mut padded = vec![0.0; ph * pw * 3];
    for py in 0..ph {
        let y = py.saturating_sub(ext).min(h - 1);
        for px in 0..pw {
            let x = px.saturating_sub(ext).min(w - 1);
            let s = 3 * (y * w + x);
            let d = 3 * (py * pw + px);
            padded[d..d + 3].copy_from_slice(&image[s..s + 3]);
        }
    }

    // Every output pixel depends only on the input, so the bands only split the work: they
    // are not part of the numerics (unlike gerstner's per-band sums) and any thread count
    // gives the same bits.
    let bands: Vec<(usize, usize)> = (0..parallel::n_bands(h))
        .map(|b| parallel::band_rows(b, h))
        .collect();
    let results = parallel::run_bands(bands, threads, |_, (y0, y1)| {
        let mut out = Vec::with_capacity((y1 - y0) * w * 3);
        for r in y0..y1 {
            for c in 0..w {
                let centre_at = 3 * ((r + ext) * pw + c + ext);
                let centre = [
                    padded[centre_at],
                    padded[centre_at + 1],
                    padded[centre_at + 2],
                ];
                let mut total_weight = 0.0;
                let mut total = [0.0f64; 3];
                for kr in 0..win_size {
                    let row = (r + kr) * pw;
                    for kc in 0..win_size {
                        let at = 3 * (row + c + kc);
                        let v = [padded[at], padded[at + 1], padded[at + 2]];
                        let mut dist = 0.0;
                        for d in 0..3 {
                            let t = centre[d] - v[d];
                            dist += t * t;
                        }
                        let dist = dist.sqrt();
                        let range_weight = range_lut[kr * win_size + kc];
                        // <Py_ssize_t>(dist * dist_scale): truncation toward zero.
                        let bin = ((dist * dist_scale) as usize).min(max_bin);
                        let weight = range_weight * color_lut[bin];
                        for d in 0..3 {
                            total[d] += v[d] * weight;
                        }
                        total_weight += weight;
                    }
                }
                for t in total {
                    let mut v = t / total_weight;
                    if min_value < 0.0 {
                        v += min_value;
                    }
                    out.push(clip(v, 0.0, 1.0));
                }
            }
        }
        out
    });
    results.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_modes() {
        let mut buf = Vec::new();
        extend_line(&[1.0, 2.0, 3.0], 4, 5, Mode::Nearest, &mut buf);
        assert_eq!(
            buf,
            [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        );
        extend_line(&[1.0, 2.0, 3.0], 4, 5, Mode::Wrap, &mut buf);
        assert_eq!(
            buf,
            [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
        );
    }

    #[test]
    fn even_window_leans_left() {
        // size 2: out[i] = (x[i - 1] + x[i]) / 2.
        let out = uniform_filter1d(&[0.0, 2.0, 4.0], 1, 3, 2, 1, Mode::Nearest);
        assert_eq!(out, [0.0, 1.0, 3.0]);
    }

    #[test]
    fn gaussian_kernel_is_normalised_and_symmetric() {
        let k = gaussian_kernel1d(1.5);
        assert_eq!(k.len(), 13);
        for i in 0..6 {
            assert_eq!(k[i].to_bits(), k[12 - i].to_bits());
        }
        assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn spatial_lut_has_the_off_by_one_grid() {
        // win_size 5: grid -3..=2, 36 entries; the centre (0, 0) is at (3, 3).
        let lut = spatial_lut(5, 2.0);
        assert_eq!(lut.len(), 36);
        assert_eq!(lut[3 * 6 + 3], 1.0);
    }

    #[test]
    fn bilateral_is_thread_count_independent() {
        let (h, w) = (37, 29);
        let rgb: Vec<f64> = (0..h * w * 3)
            .map(|i| ((i * 7919) % 1000) as f64 / 999.0)
            .collect();
        let one = denoise_bilateral(&rgb, h, w, 0.08, 2.0, 1);
        for t in [2, 3, 8] {
            let many = denoise_bilateral(&rgb, h, w, 0.08, 2.0, t);
            assert!(one
                .iter()
                .zip(&many)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}
