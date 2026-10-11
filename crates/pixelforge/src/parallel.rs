//! Thread-count-independent parallelism over a fixed partition.
//!
//! Work is split into bands whose boundaries never depend on the thread count. Each band
//! produces its own result (a private accumulator); the caller merges the results in band
//! order once every band has finished. Threads only decide *when* a band runs, never what it
//! computes or in which order results are combined, so 1 thread and 16 threads produce the
//! same bits (DESIGN.md, "Determinism contract").

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Pixel rows per band. Part of the numerics (it fixes the summation order of the per-kernel
/// sums), so changing it changes outputs: treat it like a constant of the algorithm.
pub const BAND_ROWS: usize = 32;

/// Number of bands covering `rows` rows (the last band may be shorter).
pub fn n_bands(rows: usize) -> usize {
    rows.div_ceil(BAND_ROWS)
}

/// Row range `[y0, y1)` of band `band` over `rows` rows.
pub fn band_rows(band: usize, rows: usize) -> (usize, usize) {
    let y0 = band * BAND_ROWS;
    (y0, (y0 + BAND_ROWS).min(rows))
}

/// Worker threads: `PIXELFORGE_THREADS` if set to an integer >= 1, else the available
/// parallelism. Affects speed only, never results.
pub fn thread_count() -> usize {
    if let Ok(value) = std::env::var("PIXELFORGE_THREADS") {
        if let Ok(n) = value.trim().parse::<usize>() {
            if n >= 1 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Run `f(band, input)` for every band (one `input` per band) on up to `threads` threads and
/// return the results in band order. Threads pull band indices from a shared counter, so the
/// execution order varies but each band's result does not.
pub fn run_bands<I, T, F>(inputs: Vec<I>, threads: usize, f: F) -> Vec<T>
where
    I: Send,
    T: Send,
    F: Fn(usize, I) -> T + Sync,
{
    let n = inputs.len();
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        return inputs
            .into_iter()
            .enumerate()
            .map(|(band, input)| f(band, input))
            .collect();
    }
    let slots: Vec<Mutex<Option<I>>> = inputs.into_iter().map(|i| Mutex::new(Some(i))).collect();
    let results: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let band = next.fetch_add(1, Ordering::Relaxed);
                if band >= n {
                    break;
                }
                let input = slots[band].lock().unwrap().take().unwrap();
                let out = f(band, input);
                *results[band].lock().unwrap() = Some(out);
            });
        }
    });
    results
        .into_iter()
        .map(|r| r.into_inner().unwrap().unwrap())
        .collect()
}
