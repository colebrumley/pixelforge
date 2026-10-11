//! What every downscaler returns, and the grid helper kopf and gerstner share.

/// One entry of a downscaler's `stats` (becomes a JSON value in `_meta.json`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatValue {
    Int(i64),
    Bool(bool),
    Float(f64),
}

/// The result of a downscaler (`pixelforge.downscale.Downscaled`). Arrays are row-major
/// over the (h, w) output grid; colours are interleaved LAB triples.
#[derive(Clone, Debug, PartialEq)]
pub struct Downscaled {
    pub h: usize,
    pub w: usize,
    /// (h, w, 3) float64 CIELAB.
    pub small_lab: Vec<f64>,
    /// (h, w), true = opaque.
    pub small_mask: Vec<bool>,
    /// gerstner only: the converged palette (K, 3), β applied.
    pub palette_lab: Option<Vec<f64>>,
    /// gerstner only: the palette before β.
    pub palette_lab_unsaturated: Option<Vec<f64>>,
    /// gerstner only: (h, w) palette index per pixel.
    pub indices: Option<Vec<i64>>,
    /// gerstner only: (h, w, 3) smoothed superpixel mean colours.
    pub mean_lab: Option<Vec<f64>>,
    /// In insertion order, as the Python dict.
    pub stats: Vec<(&'static str, StatValue)>,
}

/// Mean of each value's 4-connected neighbours; edge cells average the neighbours they
/// have, and a cell with none (a 1×1 grid) keeps its value.
///
/// Bitwise equal to `pixelforge.downscale.neighbor_mean4`: numpy adds the shifted copies
/// in the order above, below, left, right, starting from 0.0, and divides by the count.
pub fn neighbor_mean4(grid: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(grid.len(), h * w, "neighbor_mean4: grid is not h × w");
    let mut out = Vec::with_capacity(grid.len());
    for y in 0..h {
        for x in 0..w {
            let mut total = 0.0;
            let mut count = 0u32;
            if y > 0 {
                total += grid[(y - 1) * w + x];
                count += 1;
            }
            if y + 1 < h {
                total += grid[(y + 1) * w + x];
                count += 1;
            }
            if x > 0 {
                total += grid[y * w + x - 1];
                count += 1;
            }
            if x + 1 < w {
                total += grid[y * w + x + 1];
                count += 1;
            }
            out.push(if count > 0 {
                total / count as f64
            } else {
                grid[y * w + x]
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_and_edges() {
        // 2×3 grid: [[1, 2, 3], [4, 5, 6]]
        let g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = neighbor_mean4(&g, 2, 3);
        assert_eq!(m[0], (4.0 + 2.0) / 2.0); // (0,0): below, right
        assert_eq!(m[1], (5.0 + 1.0 + 3.0) / 3.0); // (0,1): below, left, right
        assert_eq!(m[4], (2.0 + 4.0 + 6.0) / 3.0); // (1,1): above, left, right
    }

    #[test]
    fn single_cell_keeps_its_value() {
        assert_eq!(neighbor_mean4(&[2.5], 1, 1), vec![2.5]);
    }

    #[test]
    fn addition_order_is_above_below_left_right() {
        // 3×3 with a centre whose neighbours only sum correctly in one order.
        let mut g = [0.0; 9];
        g[1] = 1e16; // above
        g[7] = 1.0; // below
        g[3] = -1e16; // left
        g[5] = 1.0; // right
        let m = neighbor_mean4(&g, 3, 3);
        // ((0 + 1e16) + 1) rounds to 1e16; + (-1e16) = 0; + 1 = 1 → 1 / 4
        assert_eq!(m[4], 0.25);
    }
}
