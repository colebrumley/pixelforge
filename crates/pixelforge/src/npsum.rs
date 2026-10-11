//! numpy's float64 reductions, bit for bit.
//!
//! The Python code sums with `np.sum` / `.mean()` / `np.cumsum`, and several decisions in the
//! pipeline compare those sums (median cut's box errors, ramp bins' mean L, tile means).
//! numpy does not add left to right: a contiguous reduction uses `pairwise_sum`
//! (numpy/_core/src/umath/loops_utils.h.src), which is sequential below 8 elements, eight
//! interleaved accumulators up to 128 elements, and a recursive split above that. Reductions
//! over a leading axis (`arr.sum(axis=0)` on a C-contiguous array) add the rows in order, so
//! they are plain sequential sums per column. Verified against numpy 2.4.6 in
//! `tests/test_npsum.py` while the Python package exists.

/// numpy's pairwise block size.
const PW_BLOCKSIZE: usize = 128;

/// `np.sum(a)` for a 1-D float64 array (also `a.sum()` over all elements of a contiguous
/// array, and `a.sum(axis=-1)` row by row).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        let unrolled = n - (n % 8);
        while i < unrolled {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.mean(a)`: the pairwise sum divided by the length.
pub fn pairwise_mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `arr.sum(axis=0)` for a C-contiguous (rows, cols) array: rows added in order per column.
pub fn sum_axis0(a: &[f64], cols: usize) -> Vec<f64> {
    assert!(
        cols > 0 && a.len().is_multiple_of(cols),
        "sum_axis0: length {} is not a multiple of {cols}",
        a.len()
    );
    let mut out = vec![0.0; cols];
    for row in a.chunks_exact(cols) {
        for (o, &x) in out.iter_mut().zip(row) {
            *o += x;
        }
    }
    out
}

/// `np.cumsum(a)`: left-to-right running sums.
pub fn cumsum(a: &[f64]) -> Vec<f64> {
    let mut total = 0.0;
    a.iter()
        .map(|&x| {
            total += x;
            total
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_is_sequential() {
        // 1e16 + 1 rounds back to 1e16, so the left-to-right sum is 1.0.
        assert_eq!(pairwise_sum(&[1e16, 1.0, -1e16, 1.0]), 1.0);
    }

    #[test]
    fn block_uses_eight_accumulators() {
        // 16 elements: r[j] = a[j] + a[j + 8]; a pure left-to-right sum would differ.
        let mut a = vec![0.0; 16];
        a[0] = 1e16;
        a[1] = 1.0;
        a[8] = -1e16;
        a[9] = 1.0;
        let sequential: f64 = a.iter().fold(0.0, |s, &x| s + x);
        assert_eq!(sequential, 2.0 - 1.0 + 1.0 - 1.0 + 1.0 - 1.0); // 1.0, the rounding bites
        assert_eq!(pairwise_sum(&a), 2.0); // (1e16 - 1e16) + (1 + 1)
    }

    #[test]
    fn split_above_block_size() {
        let a: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();
        let n2 = 500 - (500 % 8);
        assert_eq!(
            pairwise_sum(&a),
            pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
        );
    }

    #[test]
    fn axis0_and_cumsum() {
        assert_eq!(sum_axis0(&[1.0, 2.0, 3.0, 4.0], 2), vec![4.0, 6.0]);
        assert_eq!(cumsum(&[1.0, 2.0, 3.0]), vec![1.0, 3.0, 6.0]);
    }
}
