//! Helpers shared by the behavioural integration tests (the ports of tests/test_*.py):
//! fixture images, Config construction, pipeline runs on arrays and temporary directories.
//! Each test binary uses a subset, hence `dead_code` is allowed.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use pixelforge::config::{Config, Value};
use pixelforge::io;
use pixelforge::pipeline::{self, Input, Output};

/// The repository root (where `tests/fixtures` and `tests/golden.json` live).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// `tests/fixtures/<name>.png`.
pub fn fixture_path(name: &str) -> PathBuf {
    repo_root()
        .join("tests/fixtures")
        .join(format!("{name}.png"))
}

/// An 8-bit image in C order, `(h, w, c)` with `c` 3 or 4 (an `np.ndarray` of uint8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<u8>,
}

impl Image {
    /// A solid image.
    pub fn filled(h: usize, w: usize, px: &[u8]) -> Image {
        let mut data = Vec::with_capacity(h * w * px.len());
        for _ in 0..h * w {
            data.extend_from_slice(px);
        }
        Image {
            h,
            w,
            c: px.len(),
            data,
        }
    }

    pub fn px(&self, y: usize, x: usize) -> &[u8] {
        let i = (y * self.w + x) * self.c;
        &self.data[i..i + self.c]
    }

    pub fn px_mut(&mut self, y: usize, x: usize) -> &mut [u8] {
        let i = (y * self.w + x) * self.c;
        &mut self.data[i..i + self.c]
    }

    /// `image[::step, ::step]`.
    pub fn subsample(&self, step: usize) -> Image {
        let (h, w) = (self.h.div_ceil(step), self.w.div_ceil(step));
        let mut data = Vec::with_capacity(h * w * self.c);
        for y in (0..self.h).step_by(step) {
            for x in (0..self.w).step_by(step) {
                data.extend_from_slice(self.px(y, x));
            }
        }
        Image {
            h,
            w,
            c: self.c,
            data,
        }
    }

    /// The image with alpha 255 added (RGB) or kept (RGBA), `h * w * 4` bytes.
    pub fn rgba(&self) -> Vec<u8> {
        match self.c {
            4 => self.data.clone(),
            3 => self
                .data
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            c => panic!("unsupported channel count {c}"),
        }
    }

    /// `io.from_image(array)`.
    pub fn loaded(&self) -> io::Loaded {
        io::from_image_array(
            &self.shape(),
            io::ArrayData::U8(&self.data),
            io::MAX_INPUT_PIXELS,
        )
        .unwrap()
    }

    pub fn shape(&self) -> [usize; 3] {
        [self.h, self.w, self.c]
    }
}

/// A fixture as `make_fixtures.<name>()` returns it: RGB, except `circle_alpha` (RGBA).
/// Read from the checked-in PNG (test_fixtures.py checks the PNGs equal the generator).
pub fn fixture(name: &str) -> Image {
    let loaded = io::load(&fixture_path(name), io::MAX_INPUT_PIXELS).unwrap();
    let c = if name == "circle_alpha" { 4 } else { 3 };
    let mut data = Vec::with_capacity(loaded.h * loaded.w * c);
    for (p, &a) in loaded.rgb.chunks_exact(3).zip(&loaded.alpha) {
        for v in p {
            data.push((v * 255.0).round_ties_even() as u8);
        }
        if c == 4 {
            data.push(a);
        }
    }
    Image {
        h: loaded.h,
        w: loaded.w,
        c,
        data,
    }
}

/// `_small(name, step)` of test_branches.py / test_pipeline.py: a subsampled fixture.
pub fn small(name: &str, step: usize) -> Image {
    let img = fixture(name).subsample(step);
    assert!(img.h.max(img.w) <= 64);
    img
}

// ---------------------------------------------------------------------------------------
// Config

pub fn int(i: i64) -> Value {
    Value::Int(i)
}

pub fn float(x: f64) -> Value {
    Value::Float(x)
}

pub fn s(text: &str) -> Value {
    Value::str(text)
}

pub fn b(v: bool) -> Value {
    Value::Bool(v)
}

/// Keyword arguments from `(name, value)` pairs.
pub fn kw(pairs: &[(&str, Value)]) -> Vec<(String, Value)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// `Config(**pairs)` (panics on a ConfigError).
pub fn config(pairs: &[(&str, Value)]) -> Config {
    Config::new(&kw(pairs)).unwrap_or_else(|e| panic!("Config({pairs:?}): {e}"))
}

/// `Config(**pairs)`'s error message (panics when it is valid).
pub fn config_err(pairs: &[(&str, Value)]) -> String {
    match Config::new(&kw(pairs)) {
        Ok(_) => panic!("Config({pairs:?}) was accepted"),
        Err(e) => e.0,
    }
}

/// `config.replace(**pairs)`.
pub fn replace(config: &Config, pairs: &[(&str, Value)]) -> Config {
    config.replace(&kw(pairs)).unwrap()
}

/// `FAST` of the Python tests: no denoise, no keying, 16 × 16 output.
pub fn fast() -> Vec<(&'static str, Value)> {
    vec![
        ("denoise", s("none")),
        ("key_bg", b(false)),
        ("out_width", int(16)),
        ("out_height", int(16)),
    ]
}

/// `RAW`: no post-processing.
pub fn raw() -> Vec<(&'static str, Value)> {
    vec![
        ("outline", s("none")),
        ("remove_orphans", b(false)),
        ("fix_jaggies", b(false)),
    ]
}

/// Concatenate keyword lists (a later entry replaces an earlier one, like `{**a, **b}`).
pub fn merge(parts: &[&[(&'static str, Value)]]) -> Vec<(&'static str, Value)> {
    let mut out: Vec<(&'static str, Value)> = Vec::new();
    for part in parts {
        for (k, v) in part.iter() {
            out.retain(|(name, _)| name != k);
            out.push((k, v.clone()));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Pipeline

/// `pixelforge.run(array, config)`.
pub fn run_image(image: &Image, config: &Config) -> Result<Output, pipeline::Error> {
    pipeline::run(
        Input::Array {
            shape: &image.shape(),
            data: io::ArrayData::U8(&image.data),
        },
        config,
        None,
    )
}

/// `pixelforge.run(path, config)`.
pub fn run_path(path: &Path, config: &Config) -> Result<Output, pipeline::Error> {
    pipeline::run(Input::Path(path), config, None)
}

// ---------------------------------------------------------------------------------------
// Temporary directories

/// A fresh, empty directory for one test (removed when dropped).
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(name: &str) -> TempDir {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("pixelforge-test-{}-{n}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(std::fs::canonicalize(&dir).unwrap())
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
