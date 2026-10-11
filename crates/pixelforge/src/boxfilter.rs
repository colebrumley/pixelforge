//! The box (area-mean) downscaler in CIELAB, `pixelforge/downscale/box.py`.
//!
//! Exact stage (docs/rust-migration-plan.md, Section 4.2): bitwise identical to the Python
//! version. `np.bincount(..., weights=...)` adds the weights of each bin sequentially in
//! pixel order, starting from 0.0, which is what the per-cell accumulators below do; the
//! colour sums are offsets from the first opaque pixel's colour (so a solid image averages
//! back to itself exactly), as in Python.

use crate::downscale::{Downscaled, StatValue};
use crate::mask::PixelforgeError;

/// Output cell of every input coordinate along one axis: `(i * n_out) // n_in` in exact
/// integer arithmetic (`box.cell_index`).
pub fn cell_index(n_in: usize, n_out: usize) -> Vec<usize> {
    (0..n_in)
        .map(|i| ((i as u128 * n_out as u128) / n_in as u128) as usize)
        .collect()
}

/// The cell of every pixel of an `(hi, wi)` image on an `(ho, wo)` grid, row-major
/// (`cell_index(hi, ho)[:, None] * wo + cell_index(wi, wo)[None, :]`).
pub fn pixel_cells(hi: usize, wi: usize, ho: usize, wo: usize) -> Vec<usize> {
    let (cy, cx) = (cell_index(hi, ho), cell_index(wi, wo));
    let mut cell = Vec::with_capacity(hi * wi);
    for &y in &cy {
        cell.extend(cx.iter().map(|&x| y * wo + x));
    }
    cell
}

/// `box.run`: mean LAB per output cell, each opaque pixel weighted by `weight` (None = 1);
/// a cell is opaque when at least half of its pixels are (binary mask count).
///
/// `lab` is `(hi, wi, 3)`, `mask` and `weight` `(hi, wi)`, row-major. Errors with Python's
/// message when no pixel is opaque.
pub fn run(
    lab: &[f64],
    mask: &[bool],
    hi: usize,
    wi: usize,
    out_width: usize,
    out_height: usize,
    weight: Option<&[f64]>,
) -> Result<Downscaled, PixelforgeError> {
    assert_eq!(lab.len(), hi * wi * 3, "box::run: lab is not (hi, wi, 3)");
    assert_eq!(mask.len(), hi * wi, "box::run: mask is not (hi, wi)");
    if let Some(w) = weight {
        assert_eq!(w.len(), hi * wi, "box::run: weight is not (hi, wi)");
    }
    let Some(first) = mask.iter().position(|&m| m) else {
        return Err(PixelforgeError("no opaque pixels".into()));
    };
    let (ho, wo) = (out_height, out_width);
    let n_cells = ho * wo;
    let cell = pixel_cells(hi, wi, ho, wo);
    let mut total = vec![0u64; n_cells];
    let mut opaque = vec![0u64; n_cells];
    let mut mass_w = vec![0.0f64; n_cells];
    let mut sums = vec![0.0f64; n_cells * 3];
    let reference = [lab[3 * first], lab[3 * first + 1], lab[3 * first + 2]];
    for (i, &k) in cell.iter().enumerate() {
        total[k] += 1;
        if !mask[i] {
            continue;
        }
        opaque[k] += 1;
        // The three channel sums are independent bincounts; each adds in pixel order.
        match weight {
            None => {
                for c in 0..3 {
                    sums[3 * k + c] += lab[3 * i + c] - reference[c];
                }
            }
            Some(w) => {
                let wi_ = w[i];
                mass_w[k] += wi_;
                for c in 0..3 {
                    sums[3 * k + c] += wi_ * (lab[3 * i + c] - reference[c]);
                }
            }
        }
    }
    let mut small_lab = vec![0.0f64; n_cells * 3];
    let mut small_mask = vec![false; n_cells];
    for k in 0..n_cells {
        let filled = opaque[k] > 0;
        if filled {
            let mass = match weight {
                None => opaque[k] as f64,
                Some(_) => mass_w[k],
            };
            for c in 0..3 {
                small_lab[3 * k + c] = reference[c] + sums[3 * k + c] / mass;
            }
        }
        small_mask[k] = opaque[k] * 2 >= total[k] && filled;
    }
    Ok(Downscaled {
        h: ho,
        w: wo,
        small_lab,
        small_mask,
        palette_lab: None,
        palette_lab_unsaturated: None,
        indices: None,
        mean_lab: None,
        stats: vec![("iterations", StatValue::Int(1))],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_index_is_floor_of_scaled_index() {
        assert_eq!(cell_index(5, 3), vec![0, 0, 1, 1, 2]);
        assert_eq!(cell_index(4, 4), vec![0, 1, 2, 3]);
        assert_eq!(cell_index(3, 5), vec![0, 1, 3]);
    }

    #[test]
    fn solid_image_averages_back_exactly() {
        let (hi, wi) = (7, 9);
        let lab: Vec<f64> = (0..hi * wi).flat_map(|_| [53.1, -7.3, 21.9]).collect();
        let mask = vec![true; hi * wi];
        let out = run(&lab, &mask, hi, wi, 4, 3, None).unwrap();
        assert!(out.small_mask.iter().all(|&m| m));
        for k in 0..12 {
            assert_eq!(&out.small_lab[3 * k..3 * k + 3], &[53.1, -7.3, 21.9]);
        }
        assert_eq!(out.stats, vec![("iterations", StatValue::Int(1))]);
    }

    #[test]
    fn transparent_cells_and_half_rule() {
        // 4×4 → 2×2: the top-left cell has 2 of 4 opaque (kept), top-right 1 of 4 (dropped,
        // but filled), the bottom row none.
        let mut mask = vec![false; 16];
        mask[0] = true;
        mask[1] = true;
        mask[3] = true;
        let lab: Vec<f64> = (0..16).flat_map(|i| [i as f64, 0.0, 0.0]).collect();
        let out = run(&lab, &mask, 4, 4, 2, 2, None).unwrap();
        assert_eq!(out.small_mask, vec![true, false, false, false]);
        assert_eq!(out.small_lab[0], 0.5);
        assert_eq!(out.small_lab[3], 3.0);
        assert_eq!(out.small_lab[6], 0.0);
    }

    #[test]
    fn all_transparent_is_an_error() {
        let err = run(&[0.0; 12], &[false; 4], 2, 2, 1, 1, None).unwrap_err();
        assert_eq!(err.0, "no opaque pixels");
    }
}
