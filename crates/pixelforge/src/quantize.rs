//! Map an image to a palette, with optional ordered dithering (Section 8.2), ported from
//! `pixelforge/quantize.py`. Every float is computed with Python's formula and operation
//! order, so the results are bitwise identical to the numpy/scipy version.
//!
//! Images are row-major slices with explicit `(h, w)`: a LAB image is `h * w * 3` values, a
//! mask `h * w` booleans (`true` = opaque).

use crate::color;
use crate::config::Config;

/// Typical L step between ramp colours; the threshold matrix perturbs L by up to ±half of it.
pub const DITHER_L_AMPLITUDE: f64 = 12.0;

/// The `size × size` Bayer threshold matrix (row-major) built by the recursive block
/// construction until it is at least `n` wide, normalised to `m / (n * n) - 0.5`. Returns the
/// matrix side and its values. `n` is 4 or 8 in the pipeline (a power of two).
pub fn bayer(n: usize) -> (usize, Vec<f64>) {
    let mut size = 2usize;
    let mut m: Vec<i64> = vec![0, 2, 3, 1];
    while size < n {
        let big = size * 2;
        let mut next = vec![0i64; big * big];
        for y in 0..size {
            for x in 0..size {
                let v = 4 * m[y * size + x];
                next[y * big + x] = v;
                next[y * big + x + size] = v + 2;
                next[(y + size) * big + x] = v + 3;
                next[(y + size) * big + x + size] = v + 1;
            }
        }
        m = next;
        size = big;
    }
    let denom = (n * n) as f64;
    (size, m.iter().map(|&v| v as f64 / denom - 0.5).collect())
}

/// scipy's `NI_UniformFilter1D` for size 3, mode "nearest", origin 0, on one line read with
/// `stride` from `src` and written to `dst`: a running sum over the edge-extended line,
/// `tmp = e[0] + e[1] + e[2]`, then `tmp += e[i + 2] - e[i - 1]`, each output `tmp / 3`.
fn uniform_line_3(src: &[f64], dst: &mut [f64], start: usize, len: usize, stride: usize) {
    if len == 0 {
        return;
    }
    let at = |i: isize| -> f64 {
        // Index into the line extended by one edge value on each side.
        let j = i.clamp(0, len as isize - 1) as usize;
        src[start + j * stride]
    };
    let mut tmp = 0.0;
    for i in -1..2 {
        tmp += at(i);
    }
    dst[start] = tmp / 3.0;
    for ll in 1..len as isize {
        tmp += at(ll + 1) - at(ll - 2);
        dst[start + ll as usize * stride] = tmp / 3.0;
    }
}

/// `scipy.ndimage.uniform_filter1d(values, size=3, axis=axis, mode="nearest")` on an
/// `(h, w)` float64 array (`axis` 0 or 1), bit for bit.
pub fn uniform_filter1d_3_nearest(values: &[f64], h: usize, w: usize, axis: usize) -> Vec<f64> {
    assert_eq!(
        values.len(),
        h * w,
        "uniform_filter1d_3_nearest: length is not h * w"
    );
    let mut out = vec![0.0; h * w];
    match axis {
        0 => (0..w).for_each(|x| uniform_line_3(values, &mut out, x, h, w)),
        1 => (0..h).for_each(|y| uniform_line_3(values, &mut out, y * w, w, 1)),
        _ => panic!("uniform_filter1d_3_nearest: axis {axis} of a 2-D array"),
    }
    out
}

/// `scipy.ndimage.uniform_filter(values, size=3, mode="nearest")` on an `(h, w)` float64
/// array, bit for bit: the 1-D filter along axis 0, then along axis 1 on that intermediate
/// (also along an axis of length 1, where `(3 * v) / 3` need not be `v`).
pub fn uniform_filter_3_nearest(values: &[f64], h: usize, w: usize) -> Vec<f64> {
    let mid = uniform_filter1d_3_nearest(values, h, w, 0);
    uniform_filter1d_3_nearest(&mid, h, w, 1)
}

/// `np.maximum(v, 0.0)`: NaN propagates.
fn max0(v: f64) -> f64 {
    if v.is_nan() || v >= 0.0 {
        v
    } else {
        0.0
    }
}

/// Local LAB standard deviation over a 3×3 window (root of the summed channel variances),
/// `(h, w)`. With a mask that is not all opaque, only opaque cells enter the window (masked
/// mean and variance) and transparent cells get 0.
pub fn local_std(lab: &[f64], mask: Option<&[bool]>, h: usize, w: usize) -> Vec<f64> {
    let n = h * w;
    assert_eq!(lab.len(), n * 3, "local_std: lab length is not h * w * 3");
    if let Some(m) = mask {
        assert_eq!(m.len(), n, "local_std: mask length is not h * w");
    }
    let channel = |c: usize, f: &dyn Fn(usize, f64) -> f64| -> Vec<f64> {
        (0..n).map(|i| f(i, lab[3 * i + c])).collect()
    };
    let mask = mask.filter(|m| !m.iter().all(|&b| b));
    let Some(mask) = mask else {
        let mut variance = vec![0.0; n];
        for c in 0..3 {
            let x = channel(c, &|_, v| v);
            let x2: Vec<f64> = x.iter().map(|&v| v * v).collect();
            let mean = uniform_filter_3_nearest(&x, h, w);
            let mean_sq = uniform_filter_3_nearest(&x2, h, w);
            for i in 0..n {
                variance[i] += max0(mean_sq[i] - mean[i] * mean[i]);
            }
        }
        return variance.iter().map(|v| v.sqrt()).collect();
    };
    let weight: Vec<f64> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let count = uniform_filter_3_nearest(&weight, h, w);
    // An opaque cell counts itself.
    let safe: Vec<f64> = (0..n)
        .map(|i| {
            if mask[i] {
                if count[i].is_nan() || count[i] >= 1e-12 {
                    count[i]
                } else {
                    1e-12
                }
            } else {
                1.0
            }
        })
        .collect();
    let mut variance = vec![0.0; n];
    for c in 0..3 {
        let x = channel(c, &|i, v| if mask[i] { v } else { 0.0 });
        let x2: Vec<f64> = x.iter().map(|&v| v * v).collect();
        let sum = uniform_filter_3_nearest(&x, h, w);
        let sum_sq = uniform_filter_3_nearest(&x2, h, w);
        for i in 0..n {
            let mean = sum[i] / safe[i];
            let mean_sq = sum_sq[i] / safe[i];
            variance[i] += max0(mean_sq - mean * mean);
        }
    }
    (0..n)
        .map(|i| if mask[i] { variance[i].sqrt() } else { 0.0 })
        .collect()
}

/// Per-pixel LAB offset `(h, w, 3)` from the ordered threshold matrix, or `None` when
/// dithering is off. `mask` keeps transparent cells out of the "auto" local-std window.
pub fn dither_offsets(
    small_lab: &[f64],
    mask: Option<&[bool]>,
    h: usize,
    w: usize,
    config: &Config,
) -> Option<Vec<f64>> {
    if config.dither == "none" || config.dither_strength == 0.0 {
        return None;
    }
    let n = if config.dither == "bayer8" { 8 } else { 4 };
    let (side, matrix) = bayer(n);
    let scale = config.dither_strength * DITHER_L_AMPLITUDE;
    let std = (config.dither == "auto").then(|| local_std(small_lab, mask, h, w));
    let mut out = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            let mut amount = scale * matrix[(y % n) * side + x % n];
            if let Some(std) = &std {
                // np.where(std < threshold, amount, 0.0): NaN is not below the threshold.
                let below = std[y * w + x] < config.dither_variance_threshold;
                if !below {
                    amount = 0.0;
                }
            }
            out.extend_from_slice(&[amount, 0.5 * amount, 0.5 * amount]);
        }
    }
    Some(out)
}

/// Index image `(h, w)`: the nearest palette colour (ties to the lowest index) of every
/// opaque pixel after the dither offsets, -1 where transparent.
pub fn run(
    small_lab: &[f64],
    small_mask: &[bool],
    palette_lab: &[f64],
    h: usize,
    w: usize,
    config: &Config,
) -> Vec<i64> {
    let n = h * w;
    assert_eq!(
        small_lab.len(),
        n * 3,
        "quantize::run: lab length is not h * w * 3"
    );
    assert_eq!(
        small_mask.len(),
        n,
        "quantize::run: mask length is not h * w"
    );
    let offsets = dither_offsets(small_lab, Some(small_mask), h, w, config);
    let mut opaque = Vec::new();
    for i in 0..n {
        if small_mask[i] {
            for c in 0..3 {
                let v = small_lab[3 * i + c];
                opaque.push(match &offsets {
                    Some(o) => v + o[3 * i + c],
                    None => v,
                });
            }
        }
    }
    let nearest = color::nearest_index(&opaque, palette_lab);
    let mut it = nearest.into_iter();
    small_mask
        .iter()
        .map(|&m| {
            if m {
                it.next().expect("one index per opaque pixel")
            } else {
                -1
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer4_matches_the_classic_matrix() {
        let (side, m) = bayer(4);
        assert_eq!(side, 4);
        let ints: Vec<i64> = m.iter().map(|v| ((v + 0.5) * 16.0) as i64).collect();
        assert_eq!(
            ints,
            vec![0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
        );
        assert_eq!(m[0], -0.5);
    }

    #[test]
    fn bayer8_is_a_permutation() {
        let (side, m) = bayer(8);
        assert_eq!(side, 8);
        let mut ints: Vec<i64> = m.iter().map(|v| ((v + 0.5) * 64.0) as i64).collect();
        assert_eq!(&ints[..4], &[0, 32, 8, 40]);
        ints.sort();
        assert_eq!(ints, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn uniform_filter_on_a_constant_is_the_constant_up_to_rounding() {
        let out = uniform_filter_3_nearest(&[2.0; 6], 2, 3);
        assert!(out.iter().all(|&v| v == 2.0));
        assert!(uniform_filter_3_nearest(&[], 0, 4).is_empty());
    }

    #[test]
    fn local_std_is_zero_on_flat_images_and_transparent_cells() {
        let lab = vec![50.0; 4 * 4 * 3];
        assert!(local_std(&lab, None, 4, 4).iter().all(|&v| v == 0.0));
        let mut mask = vec![true; 16];
        mask[5] = false;
        let mut lab = lab;
        lab[15] = 0.0; // the transparent cell's L
        let std = local_std(&lab, Some(&mask), 4, 4);
        assert!(std.iter().all(|&v| v == 0.0), "{std:?}");
    }
}
