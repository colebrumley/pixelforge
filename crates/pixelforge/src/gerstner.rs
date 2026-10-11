//! gerstner hot loops: the modified-SLIC assignment pass, the superpixel bilateral filter
//! and the per-iteration superpixel statistics.
//!
//! The first two are per-pixel independent (no reductions across pixels), so they run over
//! fixed row bands on worker threads; the thread count changes the speed only, never a bit of
//! the output. Each pixel replays the operations of the numpy reference
//! (`pixelforge/downscale/_reference/gerstner.py`) in the same order. `superpixel_stats` is
//! a reduction over pixels and runs single-threaded on purpose (see its documentation).
//!
//! The entry points take row-major slices and plain dimensions; the Python binding
//! (crates/pixelforge-py) validates dtypes, shapes, contiguity and value ranges before
//! calling them.

use crate::parallel::thread_count;

/// Rows per band. Fixed, so the partition never depends on the thread count.
const BAND_ROWS: usize = 8;

/// Run `fill(first_row, band)` over consecutive bands of `BAND_ROWS` rows of `out` (rows of
/// `row_len` elements). The bands are fixed by the shape alone and band b goes to thread
/// b mod n. Every output element is written by exactly one call that reads only the inputs
/// (no cross-band reductions), so the result is independent of the thread count.
fn for_each_band<T, F>(out: &mut [T], row_len: usize, fill: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if out.is_empty() || row_len == 0 {
        return;
    }
    let bands: Vec<(usize, &mut [T])> = out
        .chunks_mut(BAND_ROWS * row_len)
        .enumerate()
        .map(|(b, chunk)| (b * BAND_ROWS, chunk))
        .collect();
    let n = thread_count().min(bands.len()).max(1);
    if n == 1 {
        for (row, chunk) in bands {
            fill(row, chunk);
        }
        return;
    }
    let mut per_thread: Vec<Vec<(usize, &mut [T])>> = (0..n).map(|_| Vec::new()).collect();
    for (b, band) in bands.into_iter().enumerate() {
        per_thread[b % n].push(band);
    }
    let fill = &fill;
    std::thread::scope(|scope| {
        for work in per_thread {
            scope.spawn(move || {
                for (row, chunk) in work {
                    fill(row, chunk);
                }
            });
        }
    });
}

/// Candidate superpixels per home cell, in the reference's visiting order (dy outer, dx
/// inner, offsets clipped to the grid), flattened: cell c owns items[start[c]..start[c + 1]].
struct CandidateLists {
    start: Vec<usize>,
    items: Vec<u32>,
}

/// Build the per-cell candidate lists, dropping only candidates that cannot change any
/// pixel's result:
/// - a repeat of a superpixel already visited (offsets clipped to the border cell): it
///   yields the same distance again, which never passes the strict `<`;
/// - in windowed mode, a superpixel whose center fails the window test for every pixel of
///   the cell. Rounded subtraction is monotone, so fl(px − cx) for any pixel of the cell
///   lies between the values at the cell's smallest and largest pixel coordinate; if
///   fl(px_max − cx) < −rx or fl(px_min − cx) > rx, every pixel fails |px − cx| ≤ rx
///   (likewise for y; an infinite or NaN center fails too). This is exact, not a heuristic.
#[allow(clippy::too_many_arguments)]
fn candidate_lists(
    home_y: &[i64],
    home_x: &[i64],
    px: &[f64],
    py: &[f64],
    centers: &[[f64; 5]],
    ho: usize,
    wo: usize,
    radius: i64,
    rx: f64,
    ry: f64,
    windowed: bool,
) -> CandidateLists {
    let range = |home: &[i64], coord: &[f64], n: usize| {
        let mut lo = vec![f64::INFINITY; n];
        let mut hi = vec![f64::NEG_INFINITY; n];
        for (&h, &v) in home.iter().zip(coord) {
            let h = h as usize;
            lo[h] = lo[h].min(v);
            hi[h] = hi[h].max(v);
        }
        (lo, hi)
    };
    let (x_lo, x_hi) = range(home_x, px, wo);
    let (y_lo, y_hi) = range(home_y, py, ho);
    let (ho_i, wo_i) = (ho as i64, wo as i64);
    let mut start = Vec::with_capacity(ho * wo + 1);
    let mut items = Vec::new();
    start.push(0);
    for hy in 0..ho_i {
        for hx in 0..wo_i {
            let (cy, cx) = (hy as usize, hx as usize);
            // A cell without pixels gets an empty list; it is never looked up.
            if x_lo[cx] <= x_hi[cx] && y_lo[cy] <= y_hi[cy] {
                for dy in -radius..=radius {
                    let gy = (hy + dy).clamp(0, ho_i - 1);
                    // clamp is monotone, so a clipped row equals the previous offset's row,
                    // and every cell of it was already visited (the dx sweep is the same).
                    if dy > -radius && gy == (hy + dy - 1).clamp(0, ho_i - 1) {
                        continue;
                    }
                    for dx in -radius..=radius {
                        let gx = (hx + dx).clamp(0, wo_i - 1);
                        if dx > -radius && gx == (hx + dx - 1).clamp(0, wo_i - 1) {
                            continue;
                        }
                        let s = (gy * wo_i + gx) as u32;
                        if windowed {
                            let c = &centers[s as usize];
                            let x_ok = x_hi[cx] - c[0] >= -rx && x_lo[cx] - c[0] <= rx;
                            let y_ok = y_hi[cy] - c[1] >= -ry && y_lo[cy] - c[1] <= ry;
                            if !(x_ok && y_ok) {
                                continue;
                            }
                        }
                        items.push(s);
                    }
                }
            }
            start.push(items.len());
        }
    }
    CandidateLists { start, items }
}

/// One modified-SLIC assignment pass over the (hi, wi) pixel grid; the body of
/// `gerstner._assign` fused into one loop per pixel.
///
/// For each pixel the candidates are the superpixels of grid cells
/// (clip(home_y + dy), clip(home_x + dx)), dy in −r..=r outer, dx inner, compared with a
/// strict `<` so the first one visited wins ties. Centers that may not be chosen carry
/// center_x = +inf. With `windowed`, a candidate also needs |px − cx| ≤ rx and
/// |py − cy| ≤ ry. The distance is coef·‖Δxy‖ (+ ‖Δlab‖ when `color` = (channels
/// (3, hi, wi), sp_color (ho, wo, 3)) is given), evaluated in the reference's operation
/// order. Returns the (hi, wi) assignment, −1 where nothing qualified.
///
/// `home_y` (hi) and `home_x` (wi) must lie in `0..ho` and `0..wo`; `center_x`/`center_y`
/// are (ho, wo), `px` (wi), `py` (hi).
#[allow(clippy::too_many_arguments)]
pub fn slic_assign(
    home_y: &[i64],
    home_x: &[i64],
    px: &[f64],
    py: &[f64],
    center_x: &[f64],
    center_y: &[f64],
    ho: usize,
    wo: usize,
    radius: i64,
    rx: f64,
    ry: f64,
    coef: f64,
    windowed: bool,
    color: Option<(&[f64], &[f64])>,
) -> Vec<i64> {
    let hi = home_y.len();
    let wi = home_x.len();
    let (cx, cy) = (center_x, center_y);
    assert!(
        px.len() == wi && py.len() == hi,
        "px/py do not match home_x/home_y"
    );
    assert!(
        cx.len() == ho * wo && cy.len() == ho * wo,
        "centers must be (ho, wo)"
    );

    // Per-superpixel record [cx, cy, L, a, b] for locality in the candidate loop.
    let n_sp = ho * wo;
    let mut centers = vec![[0.0_f64; 5]; n_sp];
    for (s, rec) in centers.iter_mut().enumerate() {
        rec[0] = cx[s];
        rec[1] = cy[s];
        if let Some((_, sp)) = color {
            rec[2] = sp[3 * s];
            rec[3] = sp[3 * s + 1];
            rec[4] = sp[3 * s + 2];
        }
    }
    let plane = hi * wi;
    let lists = candidate_lists(
        home_y, home_x, px, py, &centers, ho, wo, radius, rx, ry, windowed,
    );

    let mut best = vec![-1_i64; plane];
    for_each_band(&mut best, wi, |row0, band| {
        for (i, out_row) in band.chunks_mut(wi).enumerate() {
            let y = row0 + i;
            let hy = home_y[y] as usize;
            let pyv = py[y];
            for (x, out) in out_row.iter_mut().enumerate() {
                let cell = hy * wo + home_x[x] as usize;
                let pxv = px[x];
                let k = y * wi + x;
                let mut best_d = f64::INFINITY;
                let mut best_s = -1_i64;
                for &s in &lists.items[lists.start[cell]..lists.start[cell + 1]] {
                    let c = &centers[s as usize];
                    let ddx = pxv - c[0];
                    let ddy = pyv - c[1];
                    // The reference ANDs the window test into `better`; skipping the
                    // candidate is the same decision (NaN fails both ways).
                    if windowed && !(ddx.abs() <= rx && ddy.abs() <= ry) {
                        continue;
                    }
                    let mut d = ddx * ddx;
                    d += ddy * ddy;
                    d = d.sqrt();
                    d *= coef;
                    if let Some((ch, _)) = color {
                        let d0 = ch[k] - c[2];
                        let d1 = ch[plane + k] - c[3];
                        let d2 = ch[2 * plane + k] - c[4];
                        let mut acc = d0 * d0;
                        acc += d1 * d1;
                        acc += d2 * d2;
                        d += acc.sqrt();
                    }
                    if d < best_d {
                        best_d = d;
                        best_s = s as i64;
                    }
                }
                *out = best_s;
            }
        }
    });
    best
}

/// Bilateral filter of an (h, w, 3) LAB image over the pixels with nonzero `active` (h, w).
///
/// radius = ceil(2·σ_s); weight(p, p+d) = exp(−‖I(p+d) − I(p)‖²·inv_color −
/// (dx² + dy²)·inv_spatial) · active(p+d), with inv_x = 1 / (2σ_x²); out = Σ w·I / Σ w,
/// accumulated per pixel over offsets in (dy, dx) lexicographic order like the reference.
/// Pixels whose weights sum to 0 keep their input. `exp` is `libm::exp` so the bits do not
/// depend on the platform's libm (they differ from numpy's exp at rounding level).
/// Both sigmas must be finite and > 0. Returns the (h, w, 3) result.
pub fn bilateral_filter(
    img: &[f64],
    act: &[f64],
    h: usize,
    w: usize,
    sigma_color: f64,
    sigma_spatial: f64,
) -> Vec<f64> {
    assert!(
        img.len() == h * w * 3 && act.len() == h * w,
        "image must be (h, w, 3), active (h, w)"
    );
    let radius = (2.0 * sigma_spatial).ceil() as i64;
    let inv_color = 1.0 / (2.0 * sigma_color * sigma_color);
    let inv_spatial = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    let (h_i, w_i) = (h as i64, w as i64);
    let offsets: Vec<(i64, i64)> = (-radius..=radius)
        .flat_map(|dy| (-radius..=radius).map(move |dx| (dy, dx)))
        .collect();
    let n = offsets.len();
    let c = n / 2; // (0, 0)
    let half = c; // offsets after (0, 0); offset o < c mirrors to n − 1 − o > c

    // The unmasked weight is symmetric, bitwise: the colour differences of (p, q) and (q, p)
    // are exact negatives (IEEE subtraction), so their squares, their sum in the same order
    // and the spatial term are identical. Pass 1 evaluates exp() once per pair, for the
    // offsets after (0, 0), wherever either end is active; pass 2 reads offset d at p from
    // the table of p and offset −d at p from the table of p − d. Same values as one exp()
    // per offset, so the accumulation below is unchanged.
    let weight_of = |p: usize, q: usize, dy: i64, dx: i64| -> f64 {
        let e0 = img[3 * q] - img[3 * p];
        let e1 = img[3 * q + 1] - img[3 * p + 1];
        let e2 = img[3 * q + 2] - img[3 * p + 2];
        let d2 = e0 * e0 + e1 * e1 + e2 * e2;
        let spatial = (dx * dx + dy * dy) as f64;
        libm::exp(-d2 * inv_color - spatial * inv_spatial)
    };
    let mut table = vec![0.0_f64; h * w * half];
    let mut out = vec![0.0_f64; h * w * 3];
    if half > 0 {
        for_each_band(&mut table, w * half, |row0, band| {
            for (i, row) in band.chunks_mut(w * half).enumerate() {
                let y = (row0 + i) as i64;
                for x in 0..w_i {
                    let p = (y * w_i + x) as usize;
                    let slots = &mut row[x as usize * half..(x as usize + 1) * half];
                    for (slot, &(dy, dx)) in slots.iter_mut().zip(&offsets[c + 1..]) {
                        let (ny, nx) = (y + dy, x + dx);
                        if ny < 0 || ny >= h_i || nx < 0 || nx >= w_i {
                            continue;
                        }
                        let q = (ny * w_i + nx) as usize;
                        if act[p] != 0.0 || act[q] != 0.0 {
                            *slot = weight_of(p, q, dy, dx);
                        }
                    }
                }
            }
        });
    }
    let table = &table;
    for_each_band(&mut out, w * 3, |row0, band| {
        for (i, out_row) in band.chunks_mut(w * 3).enumerate() {
            let y = (row0 + i) as i64;
            for x in 0..w_i {
                let p = (y * w_i + x) as usize;
                let mut num = [0.0_f64; 3];
                let mut den = 0.0_f64;
                for (o, &(dy, dx)) in offsets.iter().enumerate() {
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || ny >= h_i || nx < 0 || nx >= w_i {
                        continue; // zero padding: weight · active = 0, adds +0
                    }
                    let q = (ny * w_i + nx) as usize;
                    let a = act[q];
                    if a == 0.0 {
                        continue; // masked out: weight 0 adds ±0 to num, +0 to den
                    }
                    let unmasked = match o.cmp(&c) {
                        std::cmp::Ordering::Greater => table[p * half + (o - c - 1)],
                        std::cmp::Ordering::Less => table[q * half + (n - 1 - o - c - 1)],
                        std::cmp::Ordering::Equal => weight_of(p, p, 0, 0),
                    };
                    let weight = unmasked * a;
                    num[0] += weight * img[3 * q];
                    num[1] += weight * img[3 * q + 1];
                    num[2] += weight * img[3 * q + 2];
                    den += weight;
                }
                let o = &mut out_row[3 * x as usize..3 * x as usize + 3];
                if den > 0.0 {
                    o[0] = num[0] / den;
                    o[1] = num[1] / den;
                    o[2] = num[2] / den;
                } else {
                    o.copy_from_slice(&img[3 * p..3 * p + 3]);
                }
            }
        }
    });
    out
}

/// Per-superpixel sums returned by [`superpixel_stats`].
pub struct SuperpixelStats {
    /// Opaque pixels assigned to each superpixel, (n_sp,).
    pub count: Vec<i64>,
    /// Σ w, (n_sp,).
    pub mass: Vec<f64>,
    /// Σ w · colour, (n_sp, 3).
    pub sums: Vec<f64>,
    /// Σ w · (x + 0.5), (n_sp,).
    pub sum_x: Vec<f64>,
    /// Σ w · (y + 0.5), (n_sp,).
    pub sum_y: Vec<f64>,
}

/// Count, mass, colour sums and centre sums per superpixel; gerstner's seven per-iteration
/// `np.bincount` calls fused into one pass.
///
/// Pixels are visited in row-major order and only those with a nonzero `mask` and an
/// assignment `s >= 0` contribute: `count[s] += 1`, `mass[s] += w`, `sum_c[s] += w · col_c`,
/// `sum_x[s] += w · (x + 0.5)`, `sum_y[s] += w · (y + 0.5)`, the product rounded before the
/// add (without `weight`, w is not applied at all and mass counts pixels). `np.bincount`
/// adds each bin's values sequentially in input order, and its input is the masked pixel
/// list in row-major order, so these sums are bitwise equal to the reference.
///
/// SINGLE-THREADED ON PURPOSE: a banded parallel version would sum each bin as per-band
/// partial sums merged afterwards, which changes the rounding of the sums and therefore the
/// superpixel means, centres and the output pixels. Keep the one sequential pass.
///
/// `assigned` and `mask` (and `weight`) are (hi, wi), `ch` is (3, hi, wi); every assignment
/// must be < `n_sp`.
pub fn superpixel_stats(
    assigned: &[i64],
    mask: &[bool],
    weight: Option<&[f64]>,
    ch: &[f64],
    hi: usize,
    wi: usize,
    n_sp: usize,
) -> SuperpixelStats {
    let plane = hi * wi;
    assert!(assigned.len() == plane && mask.len() == plane && ch.len() == 3 * plane);
    assert!(weight.is_none_or(|w| w.len() == plane));
    let mut count = vec![0_i64; n_sp];
    let mut mass = vec![0.0_f64; n_sp];
    let mut sums = vec![0.0_f64; 3 * n_sp];
    let mut sum_x = vec![0.0_f64; n_sp];
    let mut sum_y = vec![0.0_f64; n_sp];

    for y in 0..hi {
        let pyv = y as f64 + 0.5;
        for x in 0..wi {
            let k = y * wi + x;
            let s = assigned[k];
            if !mask[k] || s < 0 {
                continue;
            }
            let s = s as usize;
            let pxv = x as f64 + 0.5;
            count[s] += 1;
            match weight {
                Some(wt) => {
                    let w = wt[k];
                    mass[s] += w;
                    sums[3 * s] += w * ch[k];
                    sums[3 * s + 1] += w * ch[plane + k];
                    sums[3 * s + 2] += w * ch[2 * plane + k];
                    sum_x[s] += w * pxv;
                    sum_y[s] += w * pyv;
                }
                None => {
                    mass[s] += 1.0;
                    sums[3 * s] += ch[k];
                    sums[3 * s + 1] += ch[plane + k];
                    sums[3 * s + 2] += ch[2 * plane + k];
                    sum_x[s] += pxv;
                    sum_y[s] += pyv;
                }
            }
        }
    }
    SuperpixelStats {
        count,
        mass,
        sums,
        sum_x,
        sum_y,
    }
}
