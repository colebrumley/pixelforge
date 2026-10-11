//! Post-processing on the index image (Section 8.3), ported from `pixelforge/postprocess.py`.
//!
//! Order: (ramps →) saturation → orphans → jaggies → outline. Every pass works on a row-major
//! index image (`-1` = transparent, any negative value counts as transparent) of explicit
//! `(height, width)` plus the LAB palette as interleaved triples, and is deterministic.
//!
//! Palette ramps (`palette.regularize_ramps`) are not part of [`run`]: the caller applies
//! them to the palette first (the Python `postprocess.run` does).
//!
//! Every stage is bitwise identical to the numpy version. Region labels are numbered in
//! raster order of each region's first pixel, unlike `ndimage.label` per index value; the
//! orphan pass only uses region membership, sizes and one pixel per region, so the numbering
//! does not affect any decision.

use crate::color;

/// The 8-neighbourhood in the order the Python `_OFFSETS8` lists it.
const OFFSETS8: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// `ORPHAN_MAX_PASSES`.
pub const ORPHAN_MAX_PASSES: usize = 5;
/// `ORPHAN_MAX_DELTA`.
pub const ORPHAN_MAX_DELTA: f64 = 25.0;

/// Multiply a and b of every LAB entry by `beta` (in place).
pub fn apply_saturation(palette_lab: &mut [f64], beta: f64) {
    for c in palette_lab.chunks_exact_mut(3) {
        c[1] *= beta;
        c[2] *= beta;
    }
}

#[inline]
fn neighbour(y: usize, x: usize, dy: isize, dx: isize, h: usize, w: usize) -> Option<usize> {
    let ny = y as isize + dy;
    let nx = x as isize + dx;
    if ny < 0 || nx < 0 || ny >= h as isize || nx >= w as isize {
        None
    } else {
        Some(ny as usize * w + nx as usize)
    }
}

// ----------------------------------------------------------------------------- orphans

/// Label the 8-connected regions of equal (non-negative) index. 0 = transparent, regions
/// are `1..=n`, numbered in raster order of their first pixel. Returns `(labels, n)`.
pub fn label_regions(idx: &[i64], h: usize, w: usize) -> (Vec<usize>, usize) {
    assert_eq!(idx.len(), h * w, "label_regions: idx length is not h*w");
    let mut labels = vec![0usize; h * w];
    let mut n = 0usize;
    let mut stack = Vec::new();
    for start in 0..h * w {
        if idx[start] < 0 || labels[start] != 0 {
            continue;
        }
        n += 1;
        let value = idx[start];
        labels[start] = n;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let (y, x) = (p / w, p % w);
            for &(dy, dx) in &OFFSETS8 {
                if let Some(q) = neighbour(y, x, dy, dx, h, w) {
                    if labels[q] == 0 && idx[q] == value {
                        labels[q] = n;
                        stack.push(q);
                    }
                }
            }
        }
    }
    (labels, n)
}

/// What one pass decided for a small region.
#[derive(Clone, Copy, PartialEq)]
enum Decision {
    None,
    Merge(i64),
    Freeze,
}

/// Merge every region smaller than `min_region` into its most frequent neighbour index.
///
/// A region is merged only when the ΔE between its colour and the chosen replacement is
/// below `max_delta`; otherwise its pixels are marked in `frozen` and the region is never
/// examined again in this `remove_orphans` call. Returns whether any pixel changed.
#[allow(clippy::too_many_arguments)]
fn orphan_pass(
    idx: &mut [i64],
    h: usize,
    w: usize,
    distances: &[f64],
    n_colors: usize,
    min_region: i64,
    max_delta: f64,
    frozen: &mut [bool],
) -> bool {
    let (labels, n_regions) = label_regions(idx, h, w);
    let mut sizes = vec![0i64; n_regions + 1];
    for &l in &labels {
        sizes[l] += 1;
    }
    let mut small: Vec<bool> = sizes.iter().map(|&s| s < min_region).collect();
    small[0] = false;
    // Small pixels in raster order (np.nonzero).
    let small_pixels: Vec<usize> = (0..h * w).filter(|&p| small[labels[p]]).collect();
    if small_pixels.is_empty() {
        return false;
    }
    let mut candidate = vec![true; n_regions + 1];
    for (p, &f) in frozen.iter().enumerate() {
        if f {
            candidate[labels[p]] = false;
        }
    }

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for &p in &small_pixels {
        let r = labels[p];
        if !candidate[r] {
            continue;
        }
        let (y, x) = (p / w, p % w);
        for &(dy, dx) in &OFFSETS8 {
            if let Some(q) = neighbour(y, x, dy, dx, h, w) {
                let nl = labels[q];
                // DEVIATION: Section 8.3 — pixels of other small regions do not vote.
                // Otherwise two touching orphans (a 2-px sparkle) adopt each other's color,
                // swap back on the next pass, and the result depends on the parity of
                // max_passes. A cluster of orphans with no large neighbor is left as is until
                // a neighbor of it has been merged.
                if nl > 0 && !small[nl] {
                    pairs.push((r, q));
                }
            }
        }
    }
    if pairs.is_empty() {
        return false;
    }
    // Each neighboring pixel counts once per region, however many region pixels touch it.
    pairs.sort_unstable();
    pairs.dedup();

    // Region's own index, looked up through any one of its pixels.
    let mut first_pixel = vec![usize::MAX; n_regions + 1];
    for &p in &small_pixels {
        let r = labels[p];
        if first_pixel[r] == usize::MAX {
            first_pixel[r] = p;
        }
    }

    let mut decision = vec![Decision::None; n_regions + 1];
    let mut counts = vec![0i64; n_colors];
    let mut start = 0;
    while start < pairs.len() {
        let r = pairs[start].0;
        let mut end = start;
        counts.iter_mut().for_each(|c| *c = 0);
        while end < pairs.len() && pairs[end].0 == r {
            counts[idx[pairs[end].1] as usize] += 1;
            end += 1;
        }
        start = end;
        let own = idx[first_pixel[r]] as usize;
        let row = &distances[own * n_colors..(own + 1) * n_colors];
        // Most frequent neighbor; ties → nearest in LAB to the orphan's color; then lowest
        // index (np.argmin over the tied entries, inf elsewhere: the first minimum).
        let max_count = counts.iter().copied().max().unwrap_or(0);
        let mut target = 0usize;
        let mut best = f64::INFINITY;
        for (c, &count) in counts.iter().enumerate() {
            let d = if count == max_count {
                row[c]
            } else {
                f64::INFINITY
            };
            if d < best {
                best = d;
                target = c;
            }
        }
        // Contrast exemption: high-contrast singles (eyes, highlights) are features, not
        // noise.
        decision[r] = if row[target] < max_delta {
            Decision::Merge(target as i64)
        } else {
            Decision::Freeze
        };
    }

    let mut changed = false;
    for &p in &small_pixels {
        match decision[labels[p]] {
            Decision::Merge(target) => {
                idx[p] = target;
                changed = true;
            }
            Decision::Freeze => frozen[p] = true,
            Decision::None => {}
        }
    }
    changed
}

/// Merge 8-connected regions with fewer than `min_region` pixels into their surroundings.
///
/// Transparent pixels are never reassigned and never count as neighbors. A region is merged
/// only if its color is within ΔE `max_delta` (CIE76) of the replacement, so high-contrast
/// singles such as eye pixels and highlights are kept. Passes repeat until nothing changes,
/// a pass reproduces an earlier index image, or `max_passes` is reached.
///
/// Panics if an index is `>=` the palette size.
pub fn remove_orphans(
    idx: &[i64],
    h: usize,
    w: usize,
    palette_lab: &[f64],
    min_region: i64,
    max_passes: usize,
    max_delta: f64,
) -> Vec<i64> {
    assert_eq!(idx.len(), h * w, "remove_orphans: idx length is not h*w");
    let mut idx = idx.to_vec();
    if min_region <= 1 {
        return idx;
    }
    let n_colors = palette_lab.len() / 3;
    assert!(
        idx.iter().all(|&v| v < n_colors as i64),
        "remove_orphans: index out of the palette's range"
    );
    let distances = color::palette_distance_matrix(palette_lab);
    let mut frozen = vec![false; h * w];
    // The Python version keeps SHA-256 digests; comparing the images is the same test.
    let mut seen = vec![idx.clone()];
    for _ in 0..max_passes {
        if !orphan_pass(
            &mut idx,
            h,
            w,
            &distances,
            n_colors,
            min_region,
            max_delta,
            &mut frozen,
        ) {
            break;
        }
        if seen.contains(&idx) {
            break;
        }
        seen.push(idx.clone());
    }
    idx
}

// ----------------------------------------------------------------------------- jaggies

/// Remove "doubles" from 1-px diagonal staircases.
///
/// A double is the corner pixel p of an L-shaped triple on a diagonal line of index A over a
/// background B: p has exactly two A neighbors, one horizontal (n_h) and one vertical (n_v),
/// its six other neighbors are all the same opaque index B, and the line continues with a
/// diagonal 1-px step beyond both n_h and n_v. Removing p (setting it to B) leaves n_h and
/// n_v connected diagonally. Anything that does not match exactly is left alone, so a clean
/// 45° line is unchanged. Every test reads the input image, never a pixel already replaced.
pub fn fix_jaggies(idx: &[i64], h: usize, w: usize) -> Vec<i64> {
    assert_eq!(idx.len(), h * w, "fix_jaggies: idx length is not h*w");
    let mut out = idx.to_vec();
    if h < 4 || w < 4 {
        return out;
    }
    // -2 = outside the canvas (the Python version pads by 2 with -2).
    let at = |y: usize, x: usize, dy: isize, dx: isize| -> i64 {
        match neighbour(y, x, dy, dx, h, w) {
            Some(q) => idx[q],
            None => -2,
        }
    };
    type Corner = (isize, isize, Vec<(isize, isize)>);
    let mut corners: Vec<Corner> = Vec::with_capacity(4);
    for sy in [-1isize, 1] {
        for sx in [-1isize, 1] {
            let others: Vec<(isize, isize)> = OFFSETS8
                .iter()
                .copied()
                .filter(|&o| o != (0, sx) && o != (sy, 0))
                .collect();
            corners.push((sy, sx, others));
        }
    }
    for y in 0..h {
        for x in 0..w {
            let center = idx[y * w + x];
            if center < 0 {
                continue;
            }
            for (sy, sx, others) in &corners {
                let (sy, sx) = (*sy, *sx);
                let background = at(y, x, others[0].0, others[0].1);
                let matched = at(y, x, 0, sx) == center
                    && at(y, x, sy, 0) == center
                    && background >= 0
                    && background != center
                    && others[1..]
                        .iter()
                        .all(|&(dy, dx)| at(y, x, dy, dx) == background)
                    // The staircase continues diagonally beyond both arms of the L.
                    && at(y, x, -sy, 2 * sx) == center
                    && at(y, x, 2 * sy, -sx) == center;
                if matched {
                    out[y * w + x] = background;
                    break;
                }
            }
        }
    }
    out
}

// ----------------------------------------------------------------------------- outline

/// The outline colour in LAB: `"auto"` = the darkest entry (lowest L, first on ties) blended
/// toward LAB black by `darken`; otherwise an explicit `#rrggbb`.
pub fn outline_color_lab(
    palette_lab: &[f64],
    outline: &str,
    darken: f64,
) -> Result<[f64; 3], String> {
    if outline == "auto" {
        // np.argmin over L: the first minimum.
        let mut best = 0usize;
        let mut best_l = f64::INFINITY;
        for (k, c) in palette_lab.chunks_exact(3).enumerate() {
            if c[0] < best_l {
                best_l = c[0];
                best = k;
            }
        }
        let d = palette_lab
            .get(best * 3..best * 3 + 3)
            .ok_or_else(|| "outline: the palette is empty".to_string())?;
        let f = 1.0 - darken; // blend toward LAB black (0, 0, 0)
        return Ok([d[0] * f, d[1] * f, d[2] * f]);
    }
    let rgb = color::hex_to_rgb8(outline)?;
    let lab = color::rgb8_to_lab(&rgb);
    Ok([lab[0], lab[1], lab[2]])
}

/// Nearest palette entry to one LAB colour (CIE76); ties → lowest index.
///
/// The comparison is on ΔE itself, not its square, as `np.argmin(delta_e(...))`: two
/// squared distances that differ can round to the same ΔE, and then the lower index wins.
fn nearest_entry(lab: &[f64; 3], palette_lab: &[f64]) -> usize {
    let target: Vec<f64> = palette_lab.chunks_exact(3).flat_map(|_| *lab).collect();
    first_min(&color::delta_e(palette_lab, &target))
}

/// `np.argmin` for values without NaN: the index of the first minimum (0 when empty).
fn first_min(values: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v < values[best] {
            best = i;
        }
    }
    best
}

/// `add_outline`'s statistics (`stats["outline_index"]`, `stats["outline_clipped"]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutlineStats {
    /// The palette index of the outline, or `None` when no outline pixel was drawn.
    pub outline_index: Option<usize>,
    /// The silhouette touches an edge and padding was not allowed.
    pub outline_clipped: bool,
}

/// An index image with its dimensions and palette, as the outline and [`run`] return it.
#[derive(Clone, Debug, PartialEq)]
pub struct Processed {
    pub indices: Vec<i64>,
    pub height: usize,
    pub width: usize,
    pub palette_lab: Vec<f64>,
    pub stats: OutlineStats,
}

/// Set every transparent pixel with an opaque 4-neighbor to the outline colour.
///
/// The outline is drawn outside the silhouette. With `allow_pad` the canvas is padded by 1 px
/// first if any opaque pixel touches its edge; without it the size is kept and the ring is
/// clipped on those edges (`outline_clipped`).
///
/// The outline colour ("auto": the darkest entry blended toward black; or an explicit
/// #rrggbb) reuses an entry with the same RGB8 value, else is appended as a new entry. With
/// `fixed_palette` (a named hardware palette) the palette is never changed: the colour snaps
/// to the nearest existing entry in LAB (CIE76, ties → lowest index). A non-fixed palette
/// that is already full (256 entries) has its nearest entry overwritten instead.
#[allow(clippy::too_many_arguments)]
pub fn add_outline(
    idx: &[i64],
    h: usize,
    w: usize,
    palette_lab: &[f64],
    outline: &str,
    darken: f64,
    allow_pad: bool,
    fixed_palette: bool,
) -> Result<Processed, String> {
    assert_eq!(idx.len(), h * w, "add_outline: idx length is not h*w");
    let mut palette_lab = palette_lab.to_vec();
    let touches_edge = h > 0
        && w > 0
        && ((0..w).any(|x| idx[x] >= 0 || idx[(h - 1) * w + x] >= 0)
            || (0..h).any(|y| idx[y * w] >= 0 || idx[y * w + w - 1] >= 0));
    let mut stats = OutlineStats {
        outline_index: None,
        outline_clipped: touches_edge && !allow_pad,
    };
    let (mut out, h, w) = if allow_pad && touches_edge {
        let (ph, pw) = (h + 2, w + 2);
        let mut p = vec![-1i64; ph * pw];
        for y in 0..h {
            p[(y + 1) * pw + 1..(y + 1) * pw + 1 + w].copy_from_slice(&idx[y * w..(y + 1) * w]);
        }
        (p, ph, pw)
    } else {
        (idx.to_vec(), h, w)
    };

    let opaque = |p: usize| out[p] >= 0;
    let mut ring = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            if opaque(p) {
                continue;
            }
            let near = (y > 0 && opaque(p - w))
                || (y + 1 < h && opaque(p + w))
                || (x > 0 && opaque(p - 1))
                || (x + 1 < w && opaque(p + 1));
            if near {
                ring.push(p);
            }
        }
    }
    if ring.is_empty() {
        return Ok(Processed {
            indices: out,
            height: h,
            width: w,
            palette_lab,
            stats,
        });
    }

    let line_lab = outline_color_lab(&palette_lab, outline, darken)?;
    let line_index = if fixed_palette {
        // DEVIATION: Section 8.3 — hardware palettes are fixed; the outline snaps to the
        // nearest existing entry instead of adding (or overwriting) one.
        nearest_entry(&line_lab, &palette_lab)
    } else {
        let line_rgb8 = color::lab_to_rgb8(&line_lab);
        let palette_rgb8 = color::lab_to_rgb8(&palette_lab);
        let k = palette_lab.len() / 3;
        if let Some(same) = palette_rgb8
            .chunks_exact(3)
            .position(|c| c == line_rgb8.as_slice())
        {
            same
        } else if k < 256 {
            palette_lab.extend_from_slice(&line_lab);
            k
        } else {
            // A full palette: the nearest entry becomes the outline color.
            let i = nearest_entry(&line_lab, &palette_lab);
            palette_lab[i * 3..i * 3 + 3].copy_from_slice(&line_lab);
            i
        }
    };
    for p in ring {
        out[p] = line_index as i64;
    }
    stats.outline_index = Some(line_index);
    Ok(Processed {
        indices: out,
        height: h,
        width: w,
        palette_lab,
        stats,
    })
}

// --------------------------------------------------------------------------------- run

/// The Config fields [`run`] reads.
#[derive(Clone, Debug, PartialEq)]
pub struct PostConfig {
    pub saturation_beta: f64,
    pub remove_orphans: bool,
    pub orphan_min_region: i64,
    pub orphan_max_delta: f64,
    pub fix_jaggies: bool,
    pub outline: String,
    pub outline_darken: f64,
    pub tileset: bool,
}

impl From<&crate::config::Config> for PostConfig {
    fn from(c: &crate::config::Config) -> PostConfig {
        PostConfig {
            saturation_beta: c.saturation_beta,
            remove_orphans: c.remove_orphans,
            orphan_min_region: c.orphan_min_region,
            orphan_max_delta: c.orphan_max_delta,
            fix_jaggies: c.fix_jaggies,
            outline: c.outline.clone(),
            outline_darken: c.outline_darken,
            tileset: c.tileset,
        }
    }
}

/// Apply all passes except palette ramps, which the caller applies to `palette_lab` first
/// (when the palette is not fixed and `palette_ramps` is set).
///
/// `saturated`: β was already applied (gerstner). `fixed_palette`: a named hardware palette
/// is in use, whose colours are left untouched (the outline snaps to one of them).
/// `outline_margin`: the caller already reserved a transparent margin for the outline, so
/// the canvas is never padded here.
#[allow(clippy::too_many_arguments)]
pub fn run(
    indices: &[i64],
    h: usize,
    w: usize,
    palette_lab: &[f64],
    config: &PostConfig,
    saturated: bool,
    fixed_palette: bool,
    outline_margin: i64,
) -> Result<Processed, String> {
    let mut palette_lab = palette_lab.to_vec();
    // DEVIATION: Section 8.3 — ramps and saturation are skipped for a named palette
    // (palette_name): hardware colors are fixed by definition and stay exactly as loaded.
    if !fixed_palette && !saturated {
        apply_saturation(&mut palette_lab, config.saturation_beta);
    }
    let mut idx = indices.to_vec();
    if config.remove_orphans {
        idx = remove_orphans(
            &idx,
            h,
            w,
            &palette_lab,
            config.orphan_min_region,
            ORPHAN_MAX_PASSES,
            config.orphan_max_delta,
        );
    }
    if config.fix_jaggies {
        idx = fix_jaggies(&idx, h, w);
    }
    let mut result = Processed {
        indices: idx,
        height: h,
        width: w,
        palette_lab,
        stats: OutlineStats::default(),
    };
    if config.outline != "none" {
        // DEVIATION: Section 8.3 — the canvas is never grown by the pipeline: the outline
        // margin is reserved inside the requested size (outline_margin), and with
        // tileset=True nothing is reserved or padded because the canvas must stay a multiple
        // of tile_size; an outline on edges the silhouette touches is then clipped
        // (stats.outline_clipped).
        result = add_outline(
            &result.indices,
            result.height,
            result.width,
            &result.palette_lab,
            &config.outline,
            config.outline_darken,
            !config.tileset && outline_margin == 0,
            fixed_palette,
        )?;
        if config.remove_orphans {
            // The outline can create 1-px nubs.
            result.indices = remove_orphans(
                &result.indices,
                result.height,
                result.width,
                &result.palette_lab,
                config.orphan_min_region,
                1,
                config.orphan_max_delta,
            );
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> (Vec<i64>, usize, usize) {
        let h = rows.len();
        let w = rows[0].len();
        let data = rows
            .iter()
            .flat_map(|r| {
                r.bytes().map(|b| match b {
                    b'.' => -1,
                    d => (d - b'0') as i64,
                })
            })
            .collect();
        (data, h, w)
    }

    #[test]
    fn labelling_is_eight_connected_per_value() {
        let (idx, h, w) = grid(&["00.1", ".0.1", "1..0", "1100"]);
        let (labels, n) = label_regions(&idx, h, w);
        // {0 at (0,0),(0,1),(1,1)}, {1 at (0,3),(1,3)}, {1 at (2,0),(3,0),(3,1)},
        // {0 at (2,3),(3,2),(3,3)}: (1,3) and (2,3) differ in value.
        assert_eq!(n, 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[5]);
        assert_eq!(labels[3], labels[7]);
        assert_ne!(labels[7], labels[11]);
        assert_eq!(labels[11], labels[14]);
        assert_eq!(labels[8], labels[13]);
        assert_eq!(labels[2], 0);
        // Diagonal-only contact connects.
        let (idx, h, w) = grid(&["0.", ".0"]);
        assert_eq!(label_regions(&idx, h, w).1, 1);
    }

    #[test]
    fn two_pixel_sparkle_does_not_oscillate() {
        // Two touching single pixels of different colours inside a large region of colour 0.
        let (mut idx, h, w) = grid(&["00000", "00000", "00120", "00000", "00000"]);
        idx[12] = 1;
        idx[13] = 2;
        let palette = [50.0, 0.0, 0.0, 52.0, 0.0, 0.0, 54.0, 0.0, 0.0];
        let results: Vec<Vec<i64>> = (1..6)
            .map(|n| remove_orphans(&idx, h, w, &palette, 2, n, ORPHAN_MAX_DELTA))
            .collect();
        for r in &results {
            assert_eq!(r, &vec![0; 25]);
        }
    }

    #[test]
    fn contrast_exemption_keeps_eye_pixel() {
        let (idx, h, w) = grid(&["000", "010", "000"]);
        let palette = [50.0, 0.0, 0.0, 95.0, 0.0, 0.0];
        assert_eq!(remove_orphans(&idx, h, w, &palette, 2, 5, 25.0), idx);
        let merged = remove_orphans(&idx, h, w, &palette, 2, 5, f64::INFINITY);
        assert_eq!(merged, vec![0; 9]);
    }

    #[test]
    fn jaggy_double_is_removed() {
        // A 1-px diagonal of 1 over 0 with an L-shaped double at (2, 2).
        let (clean, h, w) = grid(&[
            "1000000", "0100000", "0010000", "0001000", "0000100", "0000010", "0000001",
        ]);
        let mut doubled = clean.clone();
        doubled[2 * w + 3] = 1; // (2, 3): now (2,2),(2,3),(3,3) form an L; corner (2,3)
        assert_eq!(fix_jaggies(&clean, h, w), clean);
        assert_eq!(fix_jaggies(&doubled, h, w), clean);
    }

    #[test]
    fn outline_ring_around_square() {
        // A 5×5 opaque square in the middle of a 9×9 canvas: the ring is the 4-neighbour
        // border, corners excluded.
        let (h, w) = (9, 9);
        let mut idx = vec![-1i64; h * w];
        for y in 2..7 {
            for x in 2..7 {
                idx[y * w + x] = 0;
            }
        }
        let palette = [60.0, 10.0, 10.0];
        let out = add_outline(&idx, h, w, &palette, "auto", 0.5, true, false).unwrap();
        assert_eq!((out.height, out.width), (9, 9));
        assert_eq!(out.palette_lab.len(), 6);
        assert_eq!(&out.palette_lab[3..], &[30.0, 5.0, 5.0]);
        assert_eq!(out.stats.outline_index, Some(1));
        let ring = out.indices.iter().filter(|&&v| v == 1).count();
        assert_eq!(ring, 20);
        assert_eq!(out.indices[w + 1], -1); // corner
        assert_eq!(out.indices[w + 2], 1);
        // Touching the edge: padded, or clipped without padding.
        let (idx, h, w) = grid(&["0"]);
        let padded = add_outline(&idx, h, w, &palette, "auto", 0.5, true, false).unwrap();
        assert_eq!((padded.height, padded.width), (3, 3));
        assert!(!padded.stats.outline_clipped);
        let clipped = add_outline(&idx, h, w, &palette, "auto", 0.5, false, false).unwrap();
        assert!(clipped.stats.outline_clipped);
        assert_eq!(clipped.stats.outline_index, None);
    }
}
