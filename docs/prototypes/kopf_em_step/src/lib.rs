//! Prototype of kopf's E-step + M-step for one iteration, fused per pixel.
//! Deterministic: fixed pixel-row bands per thread, private accumulators, fixed merge order.
use std::slice;

const RK: f64 = 2.0;
const NS: usize = 4; // slots per axis (ceil(2*RK))

struct Axis { pos: Vec<f64>, base: Vec<i64>, n_out: usize }
impl Axis {
    fn new(n_in: usize, n_out: usize) -> Axis {
        let s = n_out as f64 / n_in as f64;
        let pos: Vec<f64> = (0..n_in).map(|i| (i as f64 + 0.5) * s).collect();
        let base = pos.iter().map(|p| (p - 0.5 - RK).floor() as i64 + 1).collect();
        Axis { pos, base, n_out }
    }
    #[inline] fn valid(&self, i: usize, k: i64) -> bool {
        k >= 0 && (k as usize) < self.n_out && (self.pos[i] - (k as f64 + 0.5)).abs() < RK
    }
}

#[allow(clippy::too_many_arguments)]
fn band(ax: &Axis, ay: &Axis, y0: usize, y1: usize, wi: usize, wo: usize,
        ch: &[f64], mask: &[u8], mu_x: &[f64], mu_y: &[f64], inv_a: &[f64], inv_b: &[f64], inv_c: &[f64],
        nu: &[f64], inv_color: &[f64], ghost: &[u8], acc: &mut [f64]) {
    let n = wi * ay.pos.len();
    let mut logw = [0f64; NS * NS];
    let mut dxs = [0f64; NS * NS];
    let mut dys = [0f64; NS * NS];
    for y in y0..y1 {
        let py = ay.pos[y];
        for x in 0..wi {
            let i = y * wi + x;
            if mask[i] == 0 { continue; }
            let px = ax.pos[x];
            let (c0, c1, c2) = (ch[i], ch[n + i], ch[2 * n + i]);
            let mut peak = f64::NEG_INFINITY;
            for jy in 0..NS { for jx in 0..NS {
                let s = jy * NS + jx;
                let (ky, kx) = (ay.base[y] + jy as i64, ax.base[x] + jx as i64);
                if !(ay.valid(y, ky) && ax.valid(x, kx)) { logw[s] = f64::NEG_INFINITY; continue; }
                let k = ky as usize * wo + kx as usize;
                if ghost[k] != 0 { logw[s] = f64::NEG_INFINITY; continue; }
                let dx = px - mu_x[k]; let dy = py - mu_y[k];
                dxs[s] = dx; dys[s] = dy;
                let quad = inv_a[k] * dx * dx + 2.0 * inv_b[k] * dx * dy + inv_c[k] * dy * dy;
                let d0 = c0 - nu[3 * k]; let d1 = c1 - nu[3 * k + 1]; let d2 = c2 - nu[3 * k + 2];
                let dist = (d0 * d0 + d1 * d1 + d2 * d2) * inv_color[k];
                let lw = -0.5 * quad - dist;
                logw[s] = lw; if lw > peak { peak = lw; }
            }}
            if !peak.is_finite() { continue; }
            let mut sum = 0.0;
            for s in 0..NS * NS { let g = (logw[s] - peak).exp(); logw[s] = g; sum += g; }
            if sum == 0.0 { continue; }
            let inv = 1.0 / sum;
            for jy in 0..NS { for jx in 0..NS {
                let s = jy * NS + jx; let g = logw[s] * inv;
                if g == 0.0 { continue; }
                let k = (ay.base[y] + jy as i64) as usize * wo + (ax.base[x] + jx as i64) as usize;
                let a = &mut acc[9 * k..9 * k + 9];
                let (dx, dy) = (dxs[s], dys[s]); let gx = g * dx; let gy = g * dy;
                a[0] += g; a[1] += gx; a[2] += gy; a[3] += gx * dx; a[4] += gx * dy; a[5] += gy * dy;
                a[6] += g * c0; a[7] += g * c1; a[8] += g * c2;
            }}
        }
    }
}

/// sums: (ho*wo*9) f64 out. Returns nothing.
#[no_mangle]
pub unsafe extern "C" fn kopf_em_step(hi: usize, wi: usize, ho: usize, wo: usize, threads: usize,
        ch: *const f64, mask: *const u8, mu_x: *const f64, mu_y: *const f64,
        inv_a: *const f64, inv_b: *const f64, inv_c: *const f64, nu: *const f64,
        inv_color: *const f64, ghost: *const u8, sums: *mut f64) {
    let n = hi * wi; let m = ho * wo;
    let ch = slice::from_raw_parts(ch, 3 * n); let mask = slice::from_raw_parts(mask, n);
    let mu_x = slice::from_raw_parts(mu_x, m); let mu_y = slice::from_raw_parts(mu_y, m);
    let inv_a = slice::from_raw_parts(inv_a, m); let inv_b = slice::from_raw_parts(inv_b, m);
    let inv_c = slice::from_raw_parts(inv_c, m); let nu = slice::from_raw_parts(nu, 3 * m);
    let inv_color = slice::from_raw_parts(inv_color, m); let ghost = slice::from_raw_parts(ghost, m);
    let sums = slice::from_raw_parts_mut(sums, 9 * m);
    let ax = Axis::new(wi, wo); let ay = Axis::new(hi, ho);
    let t = threads.max(1);
    let mut accs: Vec<Vec<f64>> = (0..t).map(|_| vec![0.0; 9 * m]).collect();
    std::thread::scope(|sc| {
        for (ti, acc) in accs.iter_mut().enumerate() {
            let (ax, ay) = (&ax, &ay);
            let y0 = hi * ti / t; let y1 = hi * (ti + 1) / t;
            sc.spawn(move || band(ax, ay, y0, y1, wi, wo, ch, mask, mu_x, mu_y, inv_a, inv_b, inv_c, nu, inv_color, ghost, acc));
        }
    });
    for v in sums.iter_mut() { *v = 0.0; }
    for acc in &accs { for (s, a) in sums.iter_mut().zip(acc.iter()) { *s += a; } }
}
