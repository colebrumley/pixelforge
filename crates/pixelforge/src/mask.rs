//! Masks: the exact Euclidean feature transform, transparent-pixel filling, 4-connected
//! labelling, 8-neighbour dilation, flat-background keying with fringe unmixing, and the
//! subject mask, weight and crop box. Ported from `pixelforge/preprocess.py`
//! (`_fill_transparent`, `flat_background`, `_unmix_fringe`, `_subject`) and from the scipy
//! routines it calls; every function here is bitwise identical to the Python version.
//!
//! Images are row-major slices with explicit `(h, w)`: masks `h * w`, RGB `h * w * 3`.

use std::fmt;

use crate::config::Config;

/// A runtime failure with Python's `PixelforgeError` message, byte for byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PixelforgeError(pub String);

impl fmt::Display for PixelforgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PixelforgeError {}

/// A flat background is assumed when at least this fraction of the image's border pixels
/// share one colour (within `key_bg_tolerance`).
pub const KEY_BG_MIN_BORDER_FRACTION: f64 = 0.9;

/// Width in pixels of the widest anti-aliasing fringe `unmix_fringe` expects.
pub const FRINGE_DEPTH: usize = 3;

// ------------------------------------------------------------------ distance transform

/// The feature transform of scipy's `distance_transform_edt(input, return_indices=True)`
/// for a 2-D input whose zeros are `features[i]` (scipy transforms the *background*: its
/// features are the input's zero pixels). Returns, for every pixel, the row and column of
/// the feature pixel scipy assigns to it.
///
/// This is scipy 1.17's `NI_EuclideanFeatureTransform` (scipy/ndimage/src/ni_morphology.c;
/// Maurer, Qi and Raghavan 2003) for rank 2 with `sampling=None`: a 1-D pass down every
/// column (`_ComputeFT` at d = 0), then a pass along every row (d = 1) over the column
/// results, each a `_VoronoiFT` with the same float64 arithmetic in the same order. That fixes
/// the tie-breaking: in each pass a site is popped from the lower envelope only when the C
/// expression is strictly positive, and the query walk advances to the next site only when it
/// is strictly closer, so among equidistant candidates the one with the lower index along the
/// pass wins.
///
/// With no feature at all, scipy leaves row −1 and column 0 everywhere; so does this.
pub fn feature_transform(features: &[bool], h: usize, w: usize) -> (Vec<i32>, Vec<i32>) {
    assert_eq!(features.len(), h * w, "feature_transform: mask length");
    // scipy's `ft` starts as np.zeros; the column pass sets row = y on features and −1
    // elsewhere and column = x on features only.
    let mut rows = vec![0i32; h * w];
    let mut cols = vec![0i32; h * w];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if features[i] {
                rows[i] = y as i32;
                cols[i] = x as i32;
            } else {
                rows[i] = -1;
            }
        }
    }
    let mx = h.max(w);
    let mut f = vec![[0i64; 2]; mx];
    let mut g = vec![0usize; mx];
    // d = 0: down each column, coor[1] = x.
    for x in 0..w {
        voronoi_ft(
            &mut rows,
            &mut cols,
            |ii| ii * w + x,
            h,
            0,
            [0, x as i64],
            &mut f,
            &mut g,
        );
    }
    // d = 1: along each row, coor[0] = y.
    for y in 0..h {
        voronoi_ft(
            &mut rows,
            &mut cols,
            |ii| y * w + ii,
            w,
            1,
            [y as i64, 0],
            &mut f,
            &mut g,
        );
    }
    (rows, cols)
}

/// `_VoronoiFT` for rank 2 and no sampling, on the line of `len` pixels at `index(ii)`.
#[allow(clippy::too_many_arguments)]
fn voronoi_ft(
    rows: &mut [i32],
    cols: &mut [i32],
    index: impl Fn(usize) -> usize,
    len: usize,
    d: usize,
    coor: [i64; 2],
    f: &mut [[i64; 2]],
    g: &mut [usize],
) {
    let o = 1 - d;
    for (ii, site) in f[..len].iter_mut().enumerate() {
        let i = index(ii);
        *site = [rows[i] as i64, cols[i] as i64];
    }
    // `l` is C's signed index into g, −1 for an empty envelope.
    let mut l: isize = -1;
    for ii in 0..len {
        // C tests the first component (the row) whatever d is.
        if f[ii][0] >= 0 {
            let fd = f[ii][d] as f64;
            let tw = (f[ii][o] - coor[o]) as f64;
            let w_r = tw * tw;
            while l >= 1 {
                let idx1 = g[l as usize];
                let f1 = f[idx1][d] as f64;
                let idx2 = g[l as usize - 1];
                let a = f1 - f[idx2][d] as f64;
                let b = fd - f1;
                let c = a + b;
                let cc = coor[o] as f64;
                let tu = f[idx2][o] as f64 - cc;
                let tv = f[idx1][o] as f64 - cc;
                let u_r = tu * tu;
                let v_r = tv * tv;
                if c * v_r - b * u_r - a * w_r - a * b * c <= 0.0 {
                    break;
                }
                l -= 1;
            }
            l += 1;
            g[l as usize] = ii;
        }
    }
    let maxl = l;
    if maxl < 0 {
        return;
    }
    let delta = |site: [i64; 2], ii: usize| -> f64 {
        let mut s = 0.0;
        for (jj, &v) in site.iter().enumerate() {
            let t = if jj == d { v - ii as i64 } else { v - coor[jj] };
            let t = t as f64;
            s += t * t;
        }
        s
    };
    let mut l = 0usize;
    for ii in 0..len {
        let mut delta1 = delta(f[g[l]], ii);
        while (l as isize) < maxl {
            let delta2 = delta(f[g[l + 1]], ii);
            if delta1 <= delta2 {
                break;
            }
            delta1 = delta2;
            l += 1;
        }
        let site = f[g[l]];
        let i = index(ii);
        rows[i] = site[0] as i32;
        cols[i] = site[1] as i32;
    }
}

/// scipy's distances from the feature transform: `sqrt((row - y)² + (col - x)²)` in float64
/// (`ft - np.indices(...)`, squared, summed over the axis, `np.sqrt`).
pub fn feature_distances(rows: &[i32], cols: &[i32], h: usize, w: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let dy = (rows[i] - y as i32) as f64;
            let dx = (cols[i] - x as i32) as f64;
            out.push((dy * dy + dx * dx).sqrt());
        }
    }
    out
}

/// `distance_transform_edt(~mask, return_distances=True, return_indices=True)`: for every
/// pixel the distance to, and the row and column of, the `mask` pixel scipy picks (mask
/// pixels map to themselves).
pub fn edt(mask: &[bool], h: usize, w: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let (rows, cols) = feature_transform(mask, h, w);
    let dist = feature_distances(&rows, &cols, h, w);
    (dist, rows, cols)
}

/// numpy's reading of a feature index (`rgb[-1]` is the last row; only scipy's no-feature
/// result has −1).
fn wrap_index(i: i32, n: usize) -> usize {
    if i < 0 {
        (i + n as i32) as usize
    } else {
        i as usize
    }
}

/// `_fill_transparent`: every pixel takes the colour of the opaque pixel the feature
/// transform assigns to it (opaque pixels keep theirs); a fully opaque image is returned as is.
pub fn fill_transparent(rgb: &[f64], mask: &[bool], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(rgb.len(), h * w * 3, "fill_transparent: rgb length");
    assert_eq!(mask.len(), h * w, "fill_transparent: mask length");
    if mask.iter().all(|&m| m) {
        return rgb.to_vec();
    }
    let (rows, cols) = feature_transform(mask, h, w);
    let mut out = Vec::with_capacity(rgb.len());
    for (&r, &c) in rows.iter().zip(&cols) {
        let src = (wrap_index(r, h) * w + wrap_index(c, w)) * 3;
        out.extend_from_slice(&rgb[src..src + 3]);
    }
    out
}

// ------------------------------------------------------------------ labelling, dilation

/// `ndimage.label(mask)` with the default (4-connected) structure: labels 1..=n in raster
/// order of each component's first pixel, 0 off the mask, and n.
pub fn label4(mask: &[bool], h: usize, w: usize) -> (Vec<i64>, usize) {
    assert_eq!(mask.len(), h * w, "label4: mask length");
    let mut labels = vec![0i64; h * w];
    let mut n = 0usize;
    let mut stack = Vec::new();
    for start in 0..h * w {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n as i64;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (y, x) = (i / w, i % w);
            let mut visit = |j: usize| {
                if mask[j] && labels[j] == 0 {
                    labels[j] = n as i64;
                    stack.push(j);
                }
            };
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
        }
    }
    (labels, n)
}

/// `ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=iterations)`
/// (`border_value=0`): `iterations` 8-neighbour dilations, i.e. every pixel within Chebyshev
/// distance `iterations` of a mask pixel. `iterations == 0` is scipy's "until nothing
/// changes": the whole image when the mask has any pixel.
pub fn binary_dilation8(mask: &[bool], h: usize, w: usize, iterations: usize) -> Vec<bool> {
    assert_eq!(mask.len(), h * w, "binary_dilation8: mask length");
    let any = mask.iter().any(|&m| m);
    if !any {
        return vec![false; h * w];
    }
    let k = if iterations == 0 {
        h.max(w)
    } else {
        iterations
    };
    // Separable: a (2k+1)-wide window along each row, then along each column, from running
    // counts (the square structuring element iterated k times is the (2k+1)² square, and the
    // straight paths between two in-image pixels stay in the image).
    let window = |get: &dyn Fn(usize) -> bool, n: usize, out: &mut dyn FnMut(usize, bool)| {
        let mut prefix = vec![0usize; n + 1];
        for i in 0..n {
            prefix[i + 1] = prefix[i] + get(i) as usize;
        }
        for i in 0..n {
            let lo = i.saturating_sub(k);
            let hi = (i + k + 1).min(n);
            out(i, prefix[hi] > prefix[lo]);
        }
    };
    let mut horizontal = vec![false; h * w];
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        window(&|x| row[x], w, &mut |x, v| horizontal[y * w + x] = v);
    }
    let mut out = vec![false; h * w];
    for x in 0..w {
        window(&|y| horizontal[y * w + x], h, &mut |y, v| {
            out[y * w + x] = v
        });
    }
    out
}

// ------------------------------------------------------------------ keying

/// `max_c |rgb[c] - reference[c]|` of one pixel (`np.abs(p - reference).max(axis=-1)`).
fn max_abs_diff(p: &[f64], reference: &[f64; 3]) -> f64 {
    let mut m = (p[0] - reference[0]).abs();
    for c in 1..3 {
        let v = (p[c] - reference[c]).abs();
        if v > m {
            m = v;
        }
    }
    m
}

/// `np.median(values)` of a non-empty float64 sample: the middle value, or numpy's mean of
/// the middle pair (`(lo + hi) / 2`).
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// `flat_background(rgb, tolerance, fringe)`: the mask of a flat background connected to the
/// image border, or `None` if there is none (or it would cover the whole image).
pub fn flat_background(
    rgb: &[f64],
    h: usize,
    w: usize,
    tolerance: f64,
    fringe: usize,
) -> Option<Vec<bool>> {
    assert_eq!(rgb.len(), h * w * 3, "flat_background: rgb length");
    assert!(h > 0 && w > 0, "flat_background: empty image");
    // np.concatenate([rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]]), duplicates included.
    let mut border: Vec<usize> = Vec::with_capacity(2 * (h + w));
    border.extend(0..w);
    border.extend((h - 1) * w..h * w);
    for y in 1..h.saturating_sub(1) {
        border.push(y * w);
    }
    for y in 1..h.saturating_sub(1) {
        border.push(y * w + w - 1);
    }
    let mut reference = [0.0; 3];
    for (c, r) in reference.iter_mut().enumerate() {
        let mut values: Vec<f64> = border.iter().map(|&i| rgb[i * 3 + c]).collect();
        *r = median(&mut values);
    }
    let close_count = border
        .iter()
        .filter(|&&i| max_abs_diff(&rgb[i * 3..i * 3 + 3], &reference) <= tolerance)
        .count();
    if (close_count as f64) / (border.len() as f64) < KEY_BG_MIN_BORDER_FRACTION {
        return None;
    }
    let close: Vec<bool> = rgb
        .chunks_exact(3)
        .map(|p| max_abs_diff(p, &reference) <= tolerance)
        .collect();
    let (labels, n) = label4(&close, h, w);
    let mut on_border = vec![false; n + 1];
    for &i in &border {
        on_border[labels[i] as usize] = true;
    }
    on_border[0] = false;
    let background: Vec<bool> = labels.iter().map(|&l| on_border[l as usize]).collect();
    if background.iter().all(|&b| b) {
        return None;
    }
    let background = unmix_fringe(rgb, &background, h, w, &reference, fringe);
    if background.iter().all(|&b| b) {
        return None;
    }
    Some(background)
}

/// `_unmix_fringe`: grow `background` into the anti-aliased edge of the subject, up to
/// `passes` times (see the Python docstring).
pub fn unmix_fringe(
    rgb: &[f64],
    background: &[bool],
    h: usize,
    w: usize,
    reference: &[f64; 3],
    passes: usize,
) -> Vec<bool> {
    assert_eq!(rgb.len(), h * w * 3, "unmix_fringe: rgb length");
    assert_eq!(background.len(), h * w, "unmix_fringe: background length");
    let mut background = background.to_vec();
    for _ in 0..passes {
        let dilated = binary_dilation8(&background, h, w, 1);
        let edge: Vec<bool> = dilated
            .iter()
            .zip(&background)
            .map(|(&d, &b)| d && !b)
            .collect();
        let shallow: Vec<bool> = background
            .iter()
            .zip(&edge)
            .map(|(&b, &e)| !b && !e)
            .collect();
        if !edge.iter().any(|&e| e) || !shallow.iter().any(|&s| s) {
            break;
        }
        let edge_pixels: Vec<usize> = (0..h * w).filter(|&i| edge[i]).collect();
        let (near_r, near_c) = feature_transform(&shallow, h, w);
        let mut fg: Vec<usize> = edge_pixels
            .iter()
            .map(|&i| near_r[i] as usize * w + near_c[i] as usize)
            .collect();
        let deep: Vec<bool> = binary_dilation8(&background, h, w, FRINGE_DEPTH + 1)
            .iter()
            .map(|&d| !d)
            .collect();
        if deep.iter().any(|&d| d) {
            let (far_r, far_c) = feature_transform(&deep, h, w);
            let dist = feature_distances(&far_r, &far_c, h, w);
            for (k, &i) in edge_pixels.iter().enumerate() {
                if dist[i] <= (2 * FRINGE_DEPTH) as f64 {
                    fg[k] = far_r[i] as usize * w + far_c[i] as usize;
                }
            }
        }
        let keyed: Vec<usize> = edge_pixels
            .iter()
            .zip(&fg)
            .filter(|&(&i, &f)| {
                let mixed = max_abs_diff(&rgb[i * 3..i * 3 + 3], reference);
                let full = max_abs_diff(&rgb[f * 3..f * 3 + 3], reference);
                // A subject colour indistinguishable from the backdrop gives no coverage
                // estimate; such pixels are kept.
                full > 1e-6 && 2.0 * mixed < full
            })
            .map(|(&i, _)| i)
            .collect();
        if keyed.is_empty() {
            break;
        }
        for i in keyed {
            background[i] = true;
        }
    }
    background
}

// ------------------------------------------------------------------ subject

/// The subject of an image before any denoising (`preprocess._subject` without `remove_bg`,
/// which the caller applies to `alpha` first).
#[derive(Clone, Debug, PartialEq)]
pub struct Subject {
    /// `h * w`, true = opaque.
    pub mask: Vec<bool>,
    /// `h * w`: `max(alpha, 1) / 255` on the mask, 0 elsewhere.
    pub weight: Vec<f64>,
    /// A flat opaque background was made transparent.
    pub background_keyed: bool,
    /// The crop box `(y0, y1, x0, x1)`: the mask's bounding box plus one pixel, clamped; the
    /// whole image when `crop_to_alpha` is off or the mask is full.
    pub crop: (usize, usize, usize, usize),
}

/// `_subject(rgb, alpha, config)` minus `remove_bg`: the alpha threshold, keying of a flat
/// background when `key_bg` is set and the image is fully opaque, the weight and the crop box.
pub fn subject(
    rgb: &[f64],
    alpha: &[u8],
    h: usize,
    w: usize,
    config: &Config,
) -> Result<Subject, PixelforgeError> {
    assert_eq!(rgb.len(), h * w * 3, "subject: rgb length");
    assert_eq!(alpha.len(), h * w, "subject: alpha length");
    let threshold = config.alpha_threshold;
    let mut mask: Vec<bool> = alpha.iter().map(|&a| a as i64 >= threshold).collect();
    if !mask.iter().any(|&m| m) {
        return Err(PixelforgeError(
            "image is fully transparent after alpha thresholding".into(),
        ));
    }
    // Not in the spec: image generators return sprites on an opaque flat background, which
    // would otherwise take part in every statistic and swallow most of the palette.
    let mut background_keyed = false;
    if config.key_bg && mask.iter().all(|&m| m) {
        let fringe = usize::try_from(config.key_bg_fringe).unwrap_or(0);
        if let Some(background) = flat_background(rgb, h, w, config.key_bg_tolerance, fringe) {
            mask = background.iter().map(|&b| !b).collect();
            background_keyed = true;
        }
    }
    let weight: Vec<f64> = mask
        .iter()
        .zip(alpha)
        .map(|(&m, &a)| if m { a.max(1) as f64 / 255.0 } else { 0.0 })
        .collect();
    let mut crop = (0, h, 0, w);
    if config.crop_to_alpha && !mask.iter().all(|&m| m) {
        let (mut ymin, mut ymax, mut xmin, mut xmax) = (usize::MAX, 0, usize::MAX, 0);
        for y in 0..h {
            for x in 0..w {
                if mask[y * w + x] {
                    ymin = ymin.min(y);
                    ymax = ymax.max(y);
                    xmin = xmin.min(x);
                    xmax = xmax.max(x);
                }
            }
        }
        crop = (
            ymin.saturating_sub(1),
            (ymax + 2).min(h),
            xmin.saturating_sub(1),
            (xmax + 2).min(w),
        );
    }
    Ok(Subject {
        mask,
        weight,
        background_keyed,
        crop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Value;

    fn parse(rows: &[&str]) -> (Vec<bool>, usize, usize) {
        let h = rows.len();
        let w = rows[0].len();
        let mask = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| b == b'1'))
            .collect();
        (mask, h, w)
    }

    /// Brute force: the nearest feature distance (ties are scipy's business).
    fn brute_distance(mask: &[bool], h: usize, w: usize) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; h * w];
        for y in 0..h {
            for x in 0..w {
                for fy in 0..h {
                    for fx in 0..w {
                        if mask[fy * w + fx] {
                            let dy = fy as f64 - y as f64;
                            let dx = fx as f64 - x as f64;
                            let d = (dy * dy + dx * dx).sqrt();
                            if d < out[y * w + x] {
                                out[y * w + x] = d;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn edt_distances_are_exact() {
        let (mask, h, w) = parse(&[
            "0000000", "0100000", "0000000", "0000001", "0000000", "1000000",
        ]);
        let (dist, rows, cols) = edt(&mask, h, w);
        assert_eq!(dist, brute_distance(&mask, h, w));
        for i in 0..h * w {
            let f = rows[i] as usize * w + cols[i] as usize;
            assert!(mask[f]);
        }
    }

    #[test]
    fn edt_ties_go_to_the_lower_index_of_each_pass() {
        // Two features at equal distance along a column: (0, 1) and (2, 1) for (1, 1).
        let (mask, h, w) = parse(&["010", "000", "010"]);
        let (_, rows, cols) = edt(&mask, h, w);
        assert_eq!((rows[4], cols[4]), (0, 1));
        // Along a row: (1, 0) and (1, 2) for (1, 1).
        let (mask, h, w) = parse(&["000", "101", "000"]);
        let (_, rows, cols) = edt(&mask, h, w);
        assert_eq!((rows[4], cols[4]), (1, 0));
        // Diagonal tie (0, 0) vs (2, 2) for (1, 1): the column pass keeps both columns'
        // nearest rows, then the row pass prefers column 0.
        let (mask, h, w) = parse(&["100", "000", "001"]);
        let (_, rows, cols) = edt(&mask, h, w);
        assert_eq!((rows[4], cols[4]), (0, 0));
    }

    #[test]
    fn edt_without_features_is_scipys_minus_one() {
        let (rows, cols) = feature_transform(&[false; 6], 2, 3);
        assert_eq!(rows, vec![-1; 6]);
        assert_eq!(cols, vec![0; 6]);
    }

    #[test]
    fn fill_copies_the_nearest_opaque_colour() {
        let (mask, h, w) = parse(&["100", "000", "001"]);
        let mut rgb = vec![0.5; h * w * 3];
        rgb[0..3].copy_from_slice(&[1.0, 0.0, 0.0]);
        rgb[24..27].copy_from_slice(&[0.0, 0.0, 1.0]);
        let out = fill_transparent(&rgb, &mask, h, w);
        assert_eq!(&out[3..6], &[1.0, 0.0, 0.0]); // (0, 1)
        assert_eq!(&out[21..24], &[0.0, 0.0, 1.0]); // (2, 1)
        assert_eq!(&out[12..15], &[1.0, 0.0, 0.0]); // (1, 1): tie, (0, 0) wins
    }

    #[test]
    fn label4_is_four_connected() {
        let (mask, h, w) = parse(&["1100", "0010", "0011", "1000"]);
        let (labels, n) = label4(&mask, h, w);
        assert_eq!(n, 3);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[1], labels[6]); // diagonal only
        assert_eq!(labels[6], labels[10]);
        assert_eq!(labels[10], labels[11]);
        assert_eq!(labels[12], 3);
        assert_eq!(labels[2], 0);
    }

    #[test]
    fn dilation_is_chebyshev() {
        let (mask, h, w) = parse(&["0000000", "0000000", "0001000", "0000000", "0000000"]);
        let one = binary_dilation8(&mask, h, w, 1);
        let two = binary_dilation8(&mask, h, w, 2);
        for y in 0..h {
            for x in 0..w {
                let d = (y as i64 - 2).abs().max((x as i64 - 3).abs());
                assert_eq!(one[y * w + x], d <= 1);
                assert_eq!(two[y * w + x], d <= 2);
            }
        }
        assert_eq!(binary_dilation8(&mask, h, w, 0), vec![true; h * w]);
        assert_eq!(binary_dilation8(&[false; 4], 2, 2, 3), vec![false; 4]);
    }

    /// A 12×12 dark square on a white backdrop with a one-pixel 30 % / 70 % fringe.
    fn keyed_scene() -> (Vec<f64>, usize, usize) {
        let (h, w) = (20, 20);
        let mut rgb = vec![1.0; h * w * 3];
        for y in 3..17 {
            for x in 3..17 {
                let inner = (4..16).contains(&y) && (4..16).contains(&x);
                let v = if inner {
                    0.1
                } else if (y + x) % 2 == 0 {
                    0.37 // 70 % subject: kept
                } else {
                    0.73 // 30 % subject: keyed
                };
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[v, v, v]);
            }
        }
        (rgb, h, w)
    }

    #[test]
    fn keyed_backdrop_and_fringe() {
        let (rgb, h, w) = keyed_scene();
        let plain = flat_background(&rgb, h, w, 0.02, 0).unwrap();
        assert!(plain[0] && !plain[3 * w + 3] && !plain[10 * w + 10]);
        let unmixed = flat_background(&rgb, h, w, 0.02, 2).unwrap();
        for y in 3..17 {
            for x in 3..17 {
                let inner = (4..16).contains(&y) && (4..16).contains(&x);
                let mixed_light = !inner && (y + x) % 2 == 1;
                assert_eq!(unmixed[y * w + x], mixed_light, "({y}, {x})");
            }
        }
        // No flat border: nothing keyed.
        let mut noisy = rgb.clone();
        for (i, v) in noisy.iter_mut().enumerate() {
            *v = (i % 7) as f64 / 7.0;
        }
        assert_eq!(flat_background(&noisy, h, w, 0.02, 2), None);
        // A flat image is all background: None.
        assert_eq!(flat_background(&vec![0.5; 27], 3, 3, 0.0, 0), None);
    }

    fn config(kwargs: &[(&str, Value)]) -> Config {
        let kw: Vec<(String, Value)> = kwargs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        Config::new(&kw).unwrap()
    }

    #[test]
    fn subject_mask_weight_and_crop() {
        let (h, w) = (6, 7);
        let rgb = vec![0.5; h * w * 3];
        let mut alpha = vec![0u8; h * w];
        alpha[2 * w + 3] = 200;
        alpha[3 * w + 4] = 128;
        alpha[4 * w + 4] = 100;
        let s = subject(&rgb, &alpha, h, w, &config(&[])).unwrap();
        assert_eq!(s.crop, (1, 5, 2, 6));
        assert_eq!(s.weight[2 * w + 3], 200.0 / 255.0);
        assert_eq!(s.weight[4 * w + 4], 0.0);
        assert!(!s.background_keyed);
        let none = config(&[("alpha_threshold", Value::Int(201))]);
        assert_eq!(
            subject(&rgb, &alpha, h, w, &none).unwrap_err().0,
            "image is fully transparent after alpha thresholding"
        );
        let zero = config(&[("alpha_threshold", Value::Int(0))]);
        let s = subject(&rgb, &alpha, h, w, &zero).unwrap();
        assert_eq!(s.weight[0], 1.0 / 255.0);
        assert_eq!(s.crop, (0, h, 0, w));
    }

    #[test]
    fn subject_keys_an_opaque_backdrop() {
        let (rgb, h, w) = keyed_scene();
        let alpha = vec![255u8; h * w];
        let s = subject(
            &rgb,
            &alpha,
            h,
            w,
            &config(&[("key_bg", Value::Bool(true))]),
        )
        .unwrap();
        assert!(s.background_keyed);
        assert!(!s.mask[0]);
        assert_eq!(s.weight[0], 0.0);
        assert_eq!(s.crop.0, 2);
    }
}
