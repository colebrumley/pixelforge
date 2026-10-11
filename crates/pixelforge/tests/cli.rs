//! The `pixelforge` program end to end: the cases of tests/test_cli.py,
//! the CLI cases of tests/test_pipeline.py and the subprocess cases of
//! tests/test_determinism.py, run on the built binary with `assert_cmd`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use assert_cmd::Command;
use pixelforge::config::{Config, Value};
use predicates::prelude::*;

const FAST: &[&str] = &["--method", "box", "--out-width", "16", "--out-height", "16"];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(format!("{name}.png"))
}

/// A fresh, empty directory for one test (removed when dropped).
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("pixelforge-cli-{}-{n}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Canonical, so paths printed by the program compare equal to ours.
        TempDir(std::fs::canonicalize(&dir).unwrap())
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pixelforge() -> Command {
    let mut cmd = Command::cargo_bin("pixelforge").unwrap();
    cmd.env_remove("PIXELFORGE_DEBUG");
    cmd
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    /// stdout and stderr together (click's `result.output`).
    fn output(&self) -> String {
        format!("{}{}", self.stdout, self.stderr)
    }

    fn json_lines(&self) -> Vec<serde_json::Value> {
        self.stdout
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn last_json(&self) -> serde_json::Value {
        self.json_lines().pop().expect("a JSON line on stdout")
    }
}

fn run_with(mut cmd: Command, args: &[&dyn AsRef<std::ffi::OsStr>]) -> Run {
    for a in args {
        cmd.arg(a.as_ref());
    }
    let out = cmd.output().unwrap();
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8(out.stdout).unwrap(),
        stderr: String::from_utf8(out.stderr).unwrap(),
    }
}

/// Run the program with `args` (strings or paths).
macro_rules! invoke {
    ($($a:expr),* $(,)?) => {{
        #[allow(unused_mut)]
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        $( push_args(&mut args, $a); )*
        let refs: Vec<&dyn AsRef<std::ffi::OsStr>> =
            args.iter().map(|a| a as &dyn AsRef<std::ffi::OsStr>).collect();
        run_with(pixelforge(), &refs)
    }};
}

trait PushArgs {
    fn push_to(self, out: &mut Vec<std::ffi::OsString>);
}

impl PushArgs for &str {
    fn push_to(self, out: &mut Vec<std::ffi::OsString>) {
        out.push(self.into());
    }
}

impl PushArgs for &String {
    fn push_to(self, out: &mut Vec<std::ffi::OsString>) {
        out.push(self.into());
    }
}

impl PushArgs for &PathBuf {
    fn push_to(self, out: &mut Vec<std::ffi::OsString>) {
        out.push(self.into());
    }
}

impl PushArgs for &Path {
    fn push_to(self, out: &mut Vec<std::ffi::OsString>) {
        out.push(self.into());
    }
}

impl PushArgs for &[&str] {
    fn push_to(self, out: &mut Vec<std::ffi::OsString>) {
        out.extend(self.iter().map(|s| s.into()));
    }
}

fn push_args(out: &mut Vec<std::ffi::OsString>, a: impl PushArgs) {
    a.push_to(out);
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// RGBA8 pixels and size of a PNG.
fn rgba(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap().to_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

fn size(path: &Path) -> (u32, u32) {
    let (_, w, h) = rgba(path);
    (w, h)
}

/// `(height, width, y0, x0)` of the opaque pixels of an output PNG.
fn opaque_box(path: &Path) -> (i64, i64, i64, i64) {
    let (px, w, h) = rgba(path);
    let (mut y0, mut y1, mut x0, mut x1) = (i64::MAX, -1, i64::MAX, -1);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            if px[((y * w as i64 + x) * 4 + 3) as usize] > 0 {
                y0 = y0.min(y);
                y1 = y1.max(y);
                x0 = x0.min(x);
                x1 = x1.max(x);
            }
        }
    }
    (y1 - y0 + 1, x1 - x0 + 1, y0, x0)
}

fn save_rgba(path: &Path, w: u32, h: u32, data: Vec<u8>) {
    image::RgbaImage::from_raw(w, h, data)
        .unwrap()
        .save(path)
        .unwrap();
}

/// `np.linspace(lo, hi, n).astype(np.uint8)`.
fn linspace_u8(lo: f64, hi: f64, n: usize) -> Vec<u8> {
    let step = (hi - lo) / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i == n - 1 {
                hi as u8
            } else {
                (lo + step * i as f64) as u8
            }
        })
        .collect()
}

enum Shape {
    Disk(f64),
    Rect(f64, f64),
}

/// An RGBA frame with an opaque disk or rectangle centred on it (test_cli.py's `_frame`).
fn frame(path: &Path, (w, h): (usize, usize), shape: Shape) {
    let (cy, cx) = ((h as f64 - 1.0) / 2.0, (w as f64 - 1.0) / 2.0);
    let greens = linspace_u8(40.0, 200.0, w);
    let mut data = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for (x, &green) in greens.iter().enumerate() {
            let (dy, dx) = (y as f64 - cy, x as f64 - cx);
            let opaque = match shape {
                Shape::Disk(r) => dy * dy + dx * dx <= r * r,
                Shape::Rect(rw, rh) => dx.abs() < rw / 2.0 && dy.abs() < rh / 2.0,
            };
            data.extend_from_slice(&[200, green, 40, if opaque { 255 } else { 0 }]);
        }
    }
    save_rgba(path, w as u32, h as u32, data);
}

fn batch(tmp: &TempDir, extra: &[&str]) -> Vec<serde_json::Value> {
    let r = invoke!(
        "batch",
        &tmp.join("in"),
        "-o",
        &tmp.join("out"),
        &["--method", "box", "--palette-size", "4"][..],
        extra
    );
    assert_eq!(r.code, 0, "{}", r.output());
    r.json_lines()
}

// ---------------------------------------------------------------------------------------
// tests/test_cli.py

#[test]
fn convert_into_input_dir_refused() {
    let tmp = TempDir::new("refused");
    let src = tmp.join("two_color.png");
    std::fs::copy(fixture("two_color"), &src).unwrap();
    let before = std::fs::read(&src).unwrap();
    let r = invoke!("convert", &src, "-o", tmp.path(), FAST);
    assert_eq!(r.code, 2, "{}", r.output());
    assert!(r.stderr.contains("would overwrite input"));
    assert!(r.stderr.contains(&src.display().to_string()));
    assert!(r.stderr.contains("(use another --outdir, or --force)"));
    assert_eq!(std::fs::read(&src).unwrap(), before);
    assert!(!tmp.join("two_color_meta.json").exists());
}

#[test]
fn convert_force_overwrites() {
    let tmp = TempDir::new("force");
    let src = tmp.join("two_color.png");
    std::fs::copy(fixture("two_color"), &src).unwrap();
    let before = std::fs::read(&src).unwrap();
    let r = invoke!("convert", &src, "-o", tmp.path(), "--force", FAST);
    assert_eq!(r.code, 0, "{}", r.output());
    assert_ne!(std::fs::read(&src).unwrap(), before);
    assert!(tmp.join("two_color_meta.json").is_file());
}

#[test]
fn collision_check_resolves_relative_and_dotted_spellings() {
    let tmp = TempDir::new("dotted");
    std::fs::create_dir(tmp.join("sub")).unwrap();
    let src = tmp.join("two_color.png");
    std::fs::copy(fixture("two_color"), &src).unwrap();
    let mut cmd = pixelforge();
    cmd.current_dir(tmp.path());
    let r = run_with(
        cmd,
        &[
            &"convert",
            &"./two_color.png",
            &"-o",
            &"sub/..",
            &"--method",
            &"box",
        ],
    );
    assert_eq!(r.code, 2, "{}", r.output());
    assert_eq!(
        r.stderr.trim(),
        "config error: sub/../two_color.png would overwrite input two_color.png \
         (use another --outdir, or --force)"
    );
}

#[test]
fn batch_into_input_dir_refused() {
    let tmp = TempDir::new("batch-refused");
    std::fs::copy(fixture("two_color"), tmp.join("a.png")).unwrap();
    let r = invoke!("batch", tmp.path(), "-o", tmp.path(), FAST);
    assert_eq!(r.code, 2, "{}", r.output());
    assert!(!tmp.join("shared_palette.hex").exists());
}

#[test]
fn batch_stem_collision_refused() {
    let tmp = TempDir::new("stems");
    let (src, out) = (tmp.join("in"), tmp.join("out"));
    std::fs::create_dir(&src).unwrap();
    std::fs::copy(fixture("two_color"), src.join("a.png")).unwrap();
    image::open(fixture("two_color"))
        .unwrap()
        .to_rgb8()
        .save(src.join("a.bmp"))
        .unwrap();
    let r = invoke!("batch", &src, "-o", &out, FAST);
    assert_eq!(r.code, 2, "{}", r.output());
    assert!(r.stderr.contains(&src.join("a.bmp").display().to_string()));
    assert!(r.stderr.contains(&src.join("a.png").display().to_string()));
    assert!(r.stderr.contains("inputs share an output name"));
    assert!(!out.exists());
    let r = invoke!("batch", &src, "-o", &out, "--force", FAST);
    assert_eq!(r.code, 0, "{}", r.output());
    assert!(out.join("a.png").is_file());
}

#[test]
fn batch_skips_own_outputs() {
    let tmp = TempDir::new("skips");
    let (src, out) = (tmp.join("in"), tmp.join("out"));
    std::fs::create_dir(&src).unwrap();
    std::fs::copy(fixture("two_color"), src.join("a.png")).unwrap();
    std::fs::copy(fixture("circle_alpha"), src.join("a_preview.png")).unwrap();
    let r = invoke!("batch", &src, "-o", &out, FAST);
    assert_eq!(r.code, 0, "{}", r.output());
    assert!(r.stderr.contains(&format!(
        "pixelforge: skipping {}: looks like a pixelforge output",
        src.join("a_preview.png").display()
    )));
    assert!(!out.join("a_preview_preview.png").exists());
    assert!(out.join("a.png").is_file());
}

#[test]
fn batch_without_images_is_an_error() {
    let tmp = TempDir::new("empty");
    let r = invoke!("batch", tmp.path(), "-o", &tmp.join("out"));
    assert_eq!(r.code, 1, "{}", r.output());
    assert_eq!(
        r.stderr.trim(),
        format!("error: no images found in {}", tmp.path().display())
    );
}

/// tests/test_branches.py: a directory with no image (only a text file) is an error, and
/// the output directory is not created.
#[test]
fn batch_on_empty_directory() {
    let tmp = TempDir::new("empty-frames");
    let empty = tmp.join("frames");
    std::fs::create_dir(&empty).unwrap();
    std::fs::write(empty.join("notes.txt"), "not an image").unwrap();
    let r = invoke!("batch", &empty, "-o", &tmp.join("out"));
    assert_eq!(r.code, 1, "{}", r.output());
    assert!(r
        .output()
        .contains(&format!("no images found in {}", empty.display())));
    assert!(!tmp.join("out").exists());
}

#[test]
fn batch_frames_share_one_scale_and_canvas() {
    let tmp = TempDir::new("share-scale");
    std::fs::create_dir(tmp.join("in")).unwrap();
    frame(&tmp.join("in/big.png"), (64, 64), Shape::Disk(24.0));
    frame(&tmp.join("in/small.png"), (64, 64), Shape::Disk(12.0));
    let lines = batch(&tmp, &[]);
    assert_eq!(lines[0]["canvas"], "64x64");
    assert!(lines[0]["scale"].as_f64().unwrap() > 0.0);
    assert!(lines[0].get("shared_palette").is_some());
    let (big, small) = (tmp.join("out/big.png"), tmp.join("out/small.png"));
    assert_eq!(size(&big), (64, 64));
    assert_eq!(size(&small), (64, 64));
    let (bh, bw, _, _) = opaque_box(&big);
    let (sh, sw, _, _) = opaque_box(&small);
    assert!((bh - 2 * sh).abs() <= 2 && (bw - 2 * sw).abs() <= 2);
    for line in &lines[1..] {
        let meta = read_json(Path::new(line["outputs"]["meta"].as_str().unwrap()));
        assert_eq!(meta["config"]["scale"], lines[0]["scale"]);
        assert_eq!(meta["config"]["canvas"], lines[0]["canvas"]);
    }
}

#[test]
fn batch_wide_frame_sets_the_shared_scale() {
    let tmp = TempDir::new("wide");
    std::fs::create_dir(tmp.join("in")).unwrap();
    frame(
        &tmp.join("in/a_wide.png"),
        (128, 64),
        Shape::Rect(112.0, 56.0),
    );
    frame(&tmp.join("in/b.png"), (64, 64), Shape::Rect(56.0, 56.0));
    frame(&tmp.join("in/c.png"), (64, 64), Shape::Rect(56.0, 28.0));
    let lines = batch(&tmp, &[]);
    let scale = lines[0]["scale"].as_f64().unwrap();
    assert!((scale - 114.0 / 62.0).abs() < 1e-9, "{scale}"); // 112 + 1 px crop ring per side
    for n in ["a_wide", "b", "c"] {
        assert_eq!(size(&tmp.join(&format!("out/{n}.png"))), (64, 34)); // round(58 / scale) + 2
    }
    // Subject sizes without the 1-px outline ring on each side.
    let inner = |n: &str| {
        let (h, w, _, _) = opaque_box(&tmp.join(&format!("out/{n}.png")));
        (h - 2, w - 2)
    };
    let (wide, square, flat) = (inner("a_wide"), inner("b"), inner("c"));
    assert!((wide.1 - 2 * square.1).abs() <= 2 && (square.0 - wide.0).abs() <= 1);
    assert!((square.0 - 2 * flat.0).abs() <= 2 && square.1 == flat.1);
}

#[test]
fn batch_explicit_scale_is_used_as_given() {
    let tmp = TempDir::new("explicit-scale");
    std::fs::create_dir(tmp.join("in")).unwrap();
    frame(&tmp.join("in/a.png"), (64, 64), Shape::Disk(24.0));
    let lines = batch(&tmp, &["--scale", "2", "--no-crop-to-alpha"]);
    let keys: Vec<&String> = lines[0].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["shared_palette"]);
    assert_eq!(size(&tmp.join("out/a.png")), (32, 32));
}

#[test]
fn canvas_pads_and_centers() {
    let tmp = TempDir::new("canvas");
    std::fs::create_dir(tmp.join("in")).unwrap();
    frame(&tmp.join("in/a.png"), (64, 64), Shape::Rect(40.0, 20.0));
    batch(&tmp, &["--canvas", "48x48"]);
    let out = tmp.join("out/a.png");
    assert_eq!(size(&out), (48, 48));
    let (h, w, y0, x0) = opaque_box(&out);
    assert!(w >= 46 && h < 30);
    assert!((y0 - (48 - (y0 + h))).abs() <= 1 && (x0 - (48 - (x0 + w))).abs() <= 1);
}

#[test]
fn remove_bg_without_rembg_is_config_error() {
    let tmp = TempDir::new("rembg");
    std::fs::create_dir(tmp.join("empty-path")).unwrap();
    let mut cmd = pixelforge();
    cmd.env("PATH", tmp.join("empty-path"));
    let src = fixture("two_color");
    let out = tmp.join("out");
    let mut args: Vec<&dyn AsRef<std::ffi::OsStr>> =
        vec![&"convert", &src, &"-o", &out, &"--remove-bg"];
    args.extend(FAST.iter().map(|s| s as &dyn AsRef<std::ffi::OsStr>));
    let r = run_with(cmd, &args);
    assert_eq!(r.code, 2, "{}", r.output());
    assert!(r.stderr.contains("config error") && r.stderr.contains("rembg"));
}

#[test]
fn runtime_error_exit_code_1() {
    let tmp = TempDir::new("runtime");
    let bad = tmp.join("bad.png");
    std::fs::write(&bad, b"not an image").unwrap();
    let r = invoke!("convert", &bad, "-o", &tmp.join("out"), FAST);
    assert_eq!(r.code, 1, "{}", r.output());
    assert!(r.stderr.starts_with("error: ") && r.stderr.contains("not a supported image"));
}

#[test]
fn debug_prints_traceback() {
    let tmp = TempDir::new("debug");
    let bad = tmp.join("bad.png");
    std::fs::write(&bad, b"not an image").unwrap();
    let out = tmp.join("out");
    let plain = invoke!("convert", &bad, "-o", &out, FAST);
    assert!(plain.code == 1 && !plain.stderr.contains("Traceback"));
    let debug = invoke!("--debug", "convert", &bad, "-o", &out, FAST);
    assert_eq!(debug.code, 1, "{}", debug.output());
    assert!(debug.stderr.contains("Traceback") && debug.stderr.contains("not a supported image"));
    let config = invoke!(
        "--debug",
        "convert",
        &bad,
        "-o",
        &out,
        "--palette-size",
        "1"
    );
    assert!(config.code == 2 && config.stderr.contains("Traceback"));
    let env = pixelforge()
        .env("PIXELFORGE_DEBUG", "1")
        .args(["convert"])
        .arg(&bad)
        .arg("-o")
        .arg(&out)
        .args(FAST)
        .output()
        .unwrap();
    assert_eq!(env.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&env.stderr).contains("Traceback"));
    let off = pixelforge()
        .env("PIXELFORGE_DEBUG", "0")
        .arg("convert")
        .arg(&bad)
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&off.stderr).contains("Traceback"));
}

#[test]
fn batch_metadata_does_not_depend_on_outdir() {
    let tmp = TempDir::new("outdir-free");
    std::fs::create_dir(tmp.join("in")).unwrap();
    frame(&tmp.join("in/a.png"), (64, 64), Shape::Disk(12.0));
    let mut runs = Vec::new();
    for name in ["one", "two"] {
        let out = tmp.join(name).join("deeper");
        let r = invoke!(
            "batch",
            &tmp.join("in"),
            "-o",
            &out,
            FAST,
            &["--palette-size", "4"][..]
        );
        assert_eq!(r.code, 0, "{}", r.output());
        let text: BTreeMap<String, String> = pixelforge::io::read_png_text(&out.join("a.png"))
            .unwrap()
            .into_iter()
            .collect();
        let out_text = out.display().to_string();
        assert!(text.values().all(|v| !v.contains(&out_text)));
        let meta = read_json(&out.join("a_meta.json"));
        let shared: Vec<String> = std::fs::read_to_string(out.join("shared_palette.hex"))
            .unwrap()
            .split_whitespace()
            .map(String::from)
            .collect();
        let meta_palette: Vec<String> = meta["palette"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(meta_palette, shared);
        assert_eq!(text["pixelforge:palette"], meta_palette.join("\n"));
        assert_eq!(
            text["pixelforge:palette_sha256"],
            meta["palette_sha256"].as_str().unwrap()
        );
        runs.push((
            text,
            std::fs::read(out.join("a.png")).unwrap(),
            meta["config_hash"].clone(),
        ));
    }
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn verbose_progress_on_stderr_json_on_stdout() {
    let tmp = TempDir::new("verbose");
    let r = invoke!(
        "-vv",
        "convert",
        &fixture("circle_alpha"),
        "-o",
        tmp.path(),
        &[
            "--method",
            "kopf",
            "--out-width",
            "16",
            "--out-height",
            "16",
            "--kopf-max-iters",
            "3"
        ][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    assert!(r.stderr.contains("pixelforge: kopf iter 1/3"));
    assert!(r.stderr.contains("pixelforge: downscale kopf"));
    assert!(r.stderr.lines().all(|l| l.starts_with("pixelforge: ")));
    assert!(r.last_json().get("outputs").is_some());
    assert_eq!(r.stdout.lines().count(), 1);
    // -v: the stage lines without the iterations.
    let r = invoke!(
        "-v",
        "convert",
        &fixture("circle_alpha"),
        "-o",
        &tmp.join("v"),
        &[
            "--method",
            "kopf",
            "--out-width",
            "16",
            "--out-height",
            "16",
            "--kopf-max-iters",
            "3"
        ][..]
    );
    assert!(r.stderr.contains("pixelforge: downscale kopf"));
    assert!(!r.stderr.contains("kopf iter"));
}

#[test]
fn quiet_clean_run_prints_nothing_on_stderr() {
    let tmp = TempDir::new("quiet");
    let r = invoke!(
        "-q",
        "convert",
        &fixture("noisy_gradient"),
        "-o",
        tmp.path(),
        FAST
    );
    assert_eq!(r.code, 0, "{}", r.output());
    assert_eq!(r.stderr, "");
    r.last_json();
}

#[test]
fn hint_when_no_background_to_key() {
    // An opaque image without a flat border: key_bg (sprite default) finds nothing to remove.
    let tmp = TempDir::new("hint");
    let r = invoke!(
        "convert",
        &fixture("noisy_gradient"),
        "-o",
        tmp.path(),
        FAST
    );
    assert_eq!(r.code, 0, "{}", r.output());
    assert_eq!(r.last_json()["stats"]["background_keyed"], false);
    assert_eq!(r.stderr.matches("--remove-bg").count(), 1);
    let r = invoke!("convert", &fixture("circle_alpha"), "-o", tmp.path(), FAST);
    assert_eq!(r.stderr, "");
}

#[test]
fn every_config_field_has_a_flag_and_help() {
    let r = invoke!("convert", "--help");
    assert_eq!(r.code, 0);
    for spec in pixelforge::config::FIELDS {
        let flag = format!("--{}", spec.name.replace('_', "-").to_lowercase());
        assert!(
            r.stdout.contains(&format!("{flag} ")) || r.stdout.contains(&format!("{flag}\n")),
            "{flag}"
        );
        assert!(!spec.help.trim().is_empty() && !spec.help.contains('\n'));
        assert!(r.stdout.contains(spec.help), "{}", spec.name);
    }
}

#[test]
fn convert_help_has_text_and_choices() {
    let r = invoke!("convert", "--help");
    assert_eq!(r.code, 0, "{}", r.output());
    let text = r.stdout.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(text.contains("Number of palette colors K; ignored with --palette-name."));
    assert!(text.contains("--dither <none|bayer4|bayer8|auto>"));
    assert!(text.contains("only applies with --tileset"));
    assert!(text.contains("[default: 16 (sprite), 32 (background)]"));
    assert!(text.contains("--g-t-final") && !text.contains("--g-T-final"));
    // Per-preset defaults, on/off for booleans, the default shown only when one exists.
    assert!(r.stdout.contains(
        "Key out a flat opaque background color.  [default: on (sprite), off (background)]"
    ));
    assert!(r
        .stdout
        .contains("[default: gerstner (sprite), kopf (background)]"));
    assert!(r.stdout.contains(
        "kopf convergence tolerance (RMS color change in the unit cube).  [default: 0.001]"
    ));
    assert!(r
        .stdout
        .contains("Seed for the optional MCDA jitter (unset = fully analytic).\n"));
    assert!(r.stdout.contains("Output directory. [default: out]"));
    assert!(r
        .stdout
        .contains("Refuse inputs with more pixels than this [default: 24000000]."));
    // Flags follow the field order.
    let at = |s: &str| r.stdout.find(s).unwrap_or_else(|| panic!("{s}"));
    assert!(at("--preset ") < at("--out-width ") && at("--out-width ") < at("--scale-preview "));
}

#[test]
fn g_t_final_flag_and_legacy_alias() {
    for flag in ["--g-t-final", "--g-T-final"] {
        let tmp = TempDir::new("gt");
        let r = invoke!(
            "convert",
            &fixture("two_color"),
            "-o",
            tmp.path(),
            FAST,
            flag,
            "2.5"
        );
        assert_eq!(r.code, 0, "{}", r.output());
        let meta = read_json(&tmp.join("two_color_meta.json"));
        assert_eq!(meta["config"]["g_T_final"], 2.5);
    }
    // The current spelling wins over the old one, in either order.
    let tmp = TempDir::new("gt-both");
    let r = invoke!(
        "convert",
        &fixture("two_color"),
        "-o",
        tmp.path(),
        FAST,
        &["--g-t-final", "3", "--g-T-final", "2"][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    assert_eq!(
        read_json(&tmp.join("two_color_meta.json"))["config"]["g_T_final"],
        3.0
    );
}

#[test]
fn group_help_lists_one_line_summaries() {
    let r = invoke!("--help");
    assert_eq!(r.code, 0);
    let commands = r.stdout.split("Commands:").nth(1).unwrap();
    let commands = commands.split("\n\n").next().unwrap();
    let lines: Vec<&str> = commands.trim().lines().collect();
    assert_eq!(lines.len(), 4, "{commands}");
    for line in lines {
        assert!(!line.trim_end().ends_with("..."), "{line}");
    }
    assert!(r
        .stdout
        .contains("Usage: pixelforge [OPTIONS] COMMAND [ARGS]..."));
    assert!(r
        .stdout
        .contains("Deterministic conversion of images into 16-bit-style pixel art."));
    let r = invoke!("batch", "--help");
    assert!(r
        .stdout
        .contains("Every image in INPUT_DIR is converted with ONE shared palette"));
}

/// The `--help` of the group and of each command equals its record in tests/cli_help/
/// (rewrite the records with `cargo xtask help-record` and review the diff).
#[test]
fn help_texts_match_the_records() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/cli_help");
    for command in ["group", "batch", "compare", "convert", "palettes"] {
        let r = if command == "group" {
            invoke!("--help")
        } else {
            invoke!(command, "--help")
        };
        assert_eq!(r.code, 0, "{command}: {}", r.stderr);
        let path = dir.join(format!("rust-{command}.txt"));
        let recorded = std::fs::read_to_string(&path).unwrap();
        assert!(
            r.stdout == recorded,
            "`pixelforge {command} --help` differs from {}; if the change is intended, run \
             `cargo xtask help-record` and commit the records.\n--- live ---\n{}",
            path.display(),
            r.stdout
        );
    }
}

#[test]
fn version_and_usage_errors_like_click() {
    let r = invoke!("--version");
    assert_eq!(
        (r.code, r.stdout.as_str()),
        (0, "pixelforge, version 0.1.0\n")
    );
    // Eager: before the command is parsed.
    let r = invoke!("-v", "--version", "convert");
    assert_eq!(
        (r.code, r.stdout.as_str()),
        (0, "pixelforge, version 0.1.0\n")
    );
    let r = invoke!("convert", "x.png", "--version");
    assert_eq!(r.code, 2);
    let r = invoke!();
    assert_eq!(r.code, 2);
    assert!(r.stderr.contains("Commands:") && r.stdout.is_empty());
    let r = invoke!("-v");
    assert_eq!(r.code, 2);
    assert!(r.stderr.contains("Error: Missing command."));
    for args in [
        &["nosuch"][..],
        &["help"],
        &["convert"],
        &["convert", "x.png", "-h"],
        &["convert", "x.png", "--bogus"],
        &["convert", "x.png", "--out-width", "abc"],
        &["convert", "x.png", "--dither", "bogus"],
        &["convert", "x.png", "--max-input-pixels", "0"],
        &["palettes", "extra"],
    ] {
        let r = run_with(
            pixelforge(),
            &args
                .iter()
                .map(|a| a as &dyn AsRef<std::ffi::OsStr>)
                .collect::<Vec<_>>(),
        );
        assert_eq!(r.code, 2, "{args:?}: {}", r.output());
    }
    let r = invoke!("convert", "x.png", "--out-width", "abc");
    assert!(r
        .stderr
        .contains("Invalid value for '--out-width': 'abc' is not a valid integer."));
    let r = invoke!("convert", "x.png", "--max-input-pixels", "0");
    assert!(r
        .stderr
        .contains("Invalid value for '--max-input-pixels': 0 is not in the range x>=1."));
}

#[test]
fn path_arguments_are_checked_like_click_path() {
    let tmp = TempDir::new("paths");
    let src = fixture("two_color");
    let r = invoke!("convert", tmp.path());
    assert_eq!(r.code, 2);
    assert!(r.stderr.contains("Invalid value for 'INPUT': File '"));
    assert!(r.stderr.contains("' is a directory."));
    let r = invoke!("convert", &src, "-o", &src);
    assert_eq!(r.code, 2);
    assert!(r
        .stderr
        .contains("Invalid value for '-o' / '--outdir': Directory '"));
    let r = invoke!("batch", &src);
    assert_eq!(r.code, 2);
    assert!(r
        .stderr
        .contains("Invalid value for 'INPUT_DIR': Directory '"));
    let r = invoke!("convert", &src, "--config", tmp.path());
    assert_eq!(r.code, 2);
    assert!(r.stderr.contains("Invalid value for '--config': File '"));
}

#[test]
fn bool_pairs_last_one_wins() {
    for (args, expected) in [
        (&["--key-bg", "--no-key-bg"][..], false),
        (&["--no-key-bg", "--key-bg"], true),
        (&["--no-key-bg"], false),
        (&["--key-bg", "--key-bg"], true),
        (&[], true), // the sprite preset
    ] {
        let tmp = TempDir::new("pairs");
        let r = invoke!(
            "convert",
            &fixture("two_color"),
            "-o",
            tmp.path(),
            FAST,
            args
        );
        assert_eq!(r.code, 0, "{}", r.output());
        let meta = read_json(&tmp.join("two_color_meta.json"));
        assert_eq!(meta["config"]["key_bg"], expected, "{args:?}");
    }
    // A repeated option keeps its last value.
    let tmp = TempDir::new("repeat");
    let r = invoke!(
        "convert",
        &fixture("two_color"),
        "-o",
        tmp.path(),
        FAST,
        &["--palette-size", "8", "--palette-size", "4"][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let meta = read_json(&tmp.join("two_color_meta.json"));
    assert_eq!(meta["config"]["palette_size"], 4);
    // Option values may start with a dash (click takes the next argument as the value).
    let r = invoke!("convert", &fixture("two_color"), "--seed", "-5");
    assert_eq!(r.code, 2, "{}", r.output());
    assert_eq!(r.stderr.trim(), "config error: seed must be >= 0, got -5");
}

fn disc_frames(dir: &Path) {
    std::fs::create_dir(dir).unwrap();
    for (name, flash) in [("f0", false), ("f1", true)] {
        let mut data = Vec::with_capacity(96 * 96 * 4);
        for y in 0..96i64 {
            for x in 0..96i64 {
                let inside = (y - 48) * (y - 48) + (x - 48) * (x - 48) < 40 * 40;
                let shade = (90.0 + 0.8 * (x - 8) as f64).clamp(0.0, 255.0);
                let mut px = [
                    shade as u8,
                    (shade * 0.7) as u8,
                    (shade * 0.5) as u8,
                    if inside { 255 } else { 0 },
                ];
                if flash && (20..32).contains(&y) && (42..54).contains(&x) {
                    px[..3].copy_from_slice(&[255, 255, 255]);
                }
                data.extend_from_slice(&px);
            }
        }
        save_rgba(&dir.join(format!("{name}.png")), 96, 96, data);
    }
}

#[test]
fn batch_enhance_uses_one_pair_of_levels() {
    // A flash in one frame must not change the brightness of the rest of that frame.
    let tmp = TempDir::new("levels");
    disc_frames(&tmp.join("frames"));
    let out = tmp.join("out");
    let r = invoke!(
        "batch",
        &tmp.join("frames"),
        "-o",
        &out,
        &[
            "--method",
            "box",
            "--out-height",
            "24",
            "--enhance",
            "1",
            "--outline",
            "none",
            "--denoise",
            "none"
        ][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let header = &r.json_lines()[0];
    let (black, white) = (
        header["enhance_black"].as_f64().unwrap(),
        header["enhance_white"].as_f64().unwrap(),
    );
    assert!(0.0 <= black && black < white && white <= 1.0);
    for name in ["f0", "f1"] {
        let meta = read_json(&out.join(format!("{name}_meta.json")));
        assert_eq!(meta["config"]["enhance_white"], header["enhance_white"]);
    }
    let (a, w, h) = rgba(&out.join("f0.png"));
    let (b, w2, h2) = rgba(&out.join("f1.png"));
    assert_eq!((w, h), (w2, h2));
    let (mut differing, mut total) = (0, 0);
    for y in 12..h as usize {
        for x in 0..w as usize {
            let i = (y * w as usize + x) * 4;
            total += 1;
            if (0..4).any(|c| a[i + c] != b[i + c]) {
                differing += 1;
            }
        }
    }
    assert!(
        (differing as f64 / total as f64) < 0.05,
        "{differing}/{total}"
    );
}

// ---------------------------------------------------------------------------------------
// tests/test_pipeline.py (CLI cases)

#[test]
fn cli_convert_sprite() {
    let tmp = TempDir::new("sprite");
    let r = invoke!(
        "convert",
        &fixture("circle_alpha"),
        "-o",
        tmp.path(),
        "--preset",
        "sprite"
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let payload = r.last_json();
    let names = [
        "circle_alpha.png",
        "circle_alpha_preview.png",
        "circle_alpha_palette.json",
        "circle_alpha_palette.hex",
        "circle_alpha_meta.json",
    ];
    for name in names {
        assert!(tmp.join(name).is_file(), "{name}");
    }
    let mut outputs: Vec<String> = payload["outputs"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    outputs.sort();
    let mut expected: Vec<String> = names
        .iter()
        .map(|n| tmp.join(n).display().to_string())
        .collect();
    expected.sort();
    assert_eq!(outputs, expected);

    let config = Config::new(&[("preset".to_string(), Value::str("sprite"))]).unwrap();
    let text: BTreeMap<String, String> =
        pixelforge::io::read_png_text(&tmp.join("circle_alpha.png"))
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(text["pixelforge:config"], config.canonical_json());
    assert_eq!(
        text["pixelforge:input_sha256"],
        pixelforge::io::sha256_file(&fixture("circle_alpha")).unwrap()
    );

    let (px, w, h) = rgba(&tmp.join("circle_alpha.png"));
    let mut colors = std::collections::BTreeSet::new();
    let mut transparent = false;
    for p in px.chunks_exact(4) {
        if p[3] > 0 {
            colors.insert([p[0], p[1], p[2]]);
        } else {
            transparent = true;
        }
    }
    assert!(2 <= colors.len() && colors.len() as i64 <= config.palette_size + 1);
    assert!(transparent);
    assert_eq!(w.max(h), 64); // 64 px sprite, outline drawn inside
    assert_eq!(size(&tmp.join("circle_alpha_preview.png")), (w * 8, h * 8));

    let meta = read_json(&tmp.join("circle_alpha_meta.json"));
    let hash = config.hash().unwrap();
    assert_eq!(meta["config_hash"], hash.as_str());
    assert_eq!(payload["stats"]["config_hash"], hash.as_str());
    let palette = read_json(&tmp.join("circle_alpha_palette.json"));
    let entries = palette.as_array().unwrap();
    assert_eq!(
        entries.len() as u64,
        payload["stats"]["final_palette_size"].as_u64().unwrap()
    );
    let keys: Vec<&String> = entries[0].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["hex", "lab", "used"]);
    let used = entries.iter().filter(|e| e["used"] == true).count() as u64;
    assert_eq!(used, meta["stats"]["colors_used"].as_u64().unwrap());
    assert_eq!(
        meta["stats"]["colors_unused"].as_u64().unwrap(),
        entries.len() as u64 - used
    );
}

#[test]
fn cli_exit_codes() {
    let tmp = TempDir::new("exit-codes");
    let bad = invoke!(
        "convert",
        &fixture("two_color"),
        "-o",
        tmp.path(),
        "--palette-size",
        "1"
    );
    assert_eq!(bad.code, 2);
    assert_eq!(
        bad.stderr.trim(),
        "config error: palette_size must be in [2, 256], got 1"
    );
    let missing = invoke!("convert", &tmp.join("nope.png"), "-o", tmp.path());
    assert_eq!(missing.code, 1);
    assert!(missing
        .stderr
        .starts_with("error: [Errno 2] No such file or directory: '"));
    let listing = invoke!("palettes");
    assert_eq!(listing.code, 0);
    assert!(listing.stdout.contains("pico8\t16 colors"));
    assert!(listing.stdout.lines().all(|l| l.ends_with(" colors")));
}

#[test]
fn cli_config_file_and_flag_override() {
    let tmp = TempDir::new("config-file");
    let config_path = tmp.join("cfg.json");
    std::fs::write(
        &config_path,
        r#"{"method": "box", "palette_size": 4, "out_height": 16}"#,
    )
    .unwrap();
    let r = invoke!(
        "convert",
        &fixture("gradient6"),
        "-o",
        tmp.path(),
        "--config",
        &config_path,
        &["--palette-size", "6", "--outline", "none"][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let text: BTreeMap<String, String> = pixelforge::io::read_png_text(&tmp.join("gradient6.png"))
        .unwrap()
        .into_iter()
        .collect();
    let used: serde_json::Value = serde_json::from_str(&text["pixelforge:config"]).unwrap();
    assert_eq!(used["method"], "box");
    assert_eq!(used["palette_size"], 6);
    assert_eq!(used["out_height"], 16);
    assert_eq!(size(&tmp.join("gradient6.png")), (16, 16));
}

#[test]
fn preset_does_not_override_json_values_warning() {
    let tmp = TempDir::new("preset-warning");
    let config_path = tmp.join("cfg.json");
    std::fs::write(
        &config_path,
        r#"{"method": "box", "dither": "none", "out_height": 16}"#,
    )
    .unwrap();
    let r = invoke!(
        "convert",
        &fixture("gradient6"),
        "-o",
        tmp.path(),
        "--config",
        &config_path,
        &[
            "--preset",
            "background",
            "--dither",
            "bayer4",
            "--no-tileset"
        ][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    assert_eq!(
        r.stderr.trim(),
        format!(
            "pixelforge: warning: --preset background does not override method from {}; \
             the JSON values win",
            config_path.display()
        )
    );
}

#[test]
fn cli_batch_shares_one_palette() {
    let tmp = TempDir::new("batch-palette");
    let frames = tmp.join("frames");
    std::fs::create_dir(&frames).unwrap();
    for name in ["gradient6", "noisy_gradient"] {
        std::fs::copy(fixture(name), frames.join(format!("{name}.png"))).unwrap();
    }
    let out = tmp.join("out");
    let r = invoke!(
        "batch",
        &frames,
        "-o",
        &out,
        &[
            "--method",
            "box",
            "--out-height",
            "16",
            "--palette-size",
            "8",
            "--outline",
            "none"
        ][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let words = |p: PathBuf| -> Vec<String> {
        std::fs::read_to_string(p)
            .unwrap()
            .split_whitespace()
            .map(String::from)
            .collect()
    };
    let shared = words(out.join("shared_palette.hex"));
    assert_eq!(shared.len(), 8);
    for name in ["gradient6", "noisy_gradient"] {
        assert_eq!(words(out.join(format!("{name}_palette.hex"))), shared);
    }
    let lines = r.json_lines();
    assert_eq!(lines.len(), 3);
    assert_eq!(
        lines[0]["shared_palette"],
        out.join("shared_palette.hex").display().to_string()
    );
    assert_eq!(
        lines[1]["input"],
        frames.join("gradient6.png").display().to_string()
    );
}

fn gameboy_convert(src: &Path, outdir: &Path, palette: &str, extra: &[&str]) -> serde_json::Value {
    let r = invoke!(
        "convert",
        src,
        "-o",
        outdir,
        &[
            "--preset",
            "sprite",
            "--method",
            "box",
            "--out-height",
            "16"
        ][..],
        "--palette-name",
        palette,
        extra
    );
    assert_eq!(r.code, 0, "{}", r.output());
    r.last_json()
}

#[test]
fn named_palette_outline_stays_in_palette() {
    let tmp = TempDir::new("gameboy");
    let payload = gameboy_convert(&fixture("circle_alpha"), tmp.path(), "gameboy", &[]);
    let gameboy: Vec<String> = pixelforge::palette::parse_hex_lines("gameboy")
        .unwrap()
        .iter()
        .map(|h| format!("#{h}"))
        .collect();
    let entries = read_json(&tmp.join("circle_alpha_palette.json"));
    let hexes: Vec<String> = entries
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["hex"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(hexes, gameboy);
    let (px, _, _) = rgba(&tmp.join("circle_alpha.png"));
    for p in px.chunks_exact(4).filter(|p| p[3] > 0) {
        let hex = format!("#{:02x}{:02x}{:02x}", p[0], p[1], p[2]);
        assert!(gameboy.contains(&hex), "{hex}");
    }
    assert_eq!(payload["stats"]["final_palette_size"], 4);
    let outline = payload["stats"]["outline_index"].as_i64().unwrap();
    assert!((0..4).contains(&outline));
}

#[test]
fn palette_hex_round_trip_is_stable() {
    let tmp = TempDir::new("hex-round-trip");
    gameboy_convert(&fixture("circle_alpha"), &tmp.join("a"), "gameboy", &[]);
    let first = tmp.join("a/circle_alpha_palette.hex");
    gameboy_convert(
        &fixture("circle_alpha"),
        &tmp.join("b"),
        &first.display().to_string(),
        &[],
    );
    assert_eq!(
        std::fs::read_to_string(tmp.join("b/circle_alpha_palette.hex")).unwrap(),
        std::fs::read_to_string(&first).unwrap()
    );
}

#[test]
fn transparent_first_hex_round_trips_named_palette() {
    let tmp = TempDir::new("transparent-first");
    let extra = ["--transparent-index", "first"];
    gameboy_convert(&fixture("circle_alpha"), &tmp.join("a"), "gameboy", &extra);
    let hex_path = tmp.join("a/circle_alpha_palette.hex");
    let text = std::fs::read_to_string(&hex_path).unwrap();
    assert_eq!(text.lines().next(), Some("; transparent"));
    assert_eq!(
        pixelforge::palette::load_palette(&hex_path.display().to_string())
            .unwrap()
            .len(),
        4
    );
    let again = gameboy_convert(
        &fixture("circle_alpha"),
        &tmp.join("b"),
        &hex_path.display().to_string(),
        &extra,
    );
    assert_eq!(again["stats"]["final_palette_size"], 4);
    assert_eq!(
        std::fs::read_to_string(tmp.join("b/circle_alpha_palette.hex")).unwrap(),
        text
    );
}

#[test]
fn cli_rejects_non_finite_json_and_oversize_config() {
    let tmp = TempDir::new("json-checks");
    let image = fixture("two_color");
    let config_path = tmp.join("cfg.json");
    for literal in ["NaN", "Infinity", "-Infinity"] {
        std::fs::write(&config_path, format!("{{\"saturation_beta\": {literal}}}")).unwrap();
        let r = invoke!(
            "convert",
            &image,
            "-o",
            tmp.path(),
            "--config",
            &config_path
        );
        assert_eq!(r.code, 2, "{literal}");
        assert_eq!(
            r.stderr.trim(),
            format!("config error: config JSON must not contain {literal}")
        );
    }
    let big = tmp.join("big.json");
    let mut text = String::from("{\"palette_size\": 4");
    text.push_str(&" ".repeat(1024 * 1024));
    text.push('}');
    std::fs::write(&big, text).unwrap();
    let r = invoke!("convert", &image, "-o", tmp.path(), "--config", &big);
    assert_eq!(r.code, 2);
    assert!(r.stderr.contains("larger than"));
    let r = invoke!("convert", &image, "-o", tmp.path(), "--config", tmp.path());
    assert_ne!(r.code, 0);
    // Malformed JSON: Python's message and position.
    std::fs::write(&config_path, "{\"method\": \"box\",\n \"seed\" 1}").unwrap();
    let r = invoke!(
        "convert",
        &image,
        "-o",
        tmp.path(),
        "--config",
        &config_path
    );
    assert_eq!(r.code, 2);
    assert_eq!(
        r.stderr.trim(),
        "config error: Expecting ':' delimiter: line 2 column 9 (char 26)"
    );
}

#[test]
fn cli_max_input_pixels() {
    let tmp = TempDir::new("max-pixels");
    let r = invoke!(
        "convert",
        &fixture("two_color"),
        "-o",
        tmp.path(),
        "--max-input-pixels",
        "100"
    );
    assert_eq!(r.code, 1);
    assert!(r.output().contains("input budget of 100 pixels"));
}

/// The `PIXELFORGE_THREADS=3` part of test_pipeline.py's
/// `test_meta_environment_and_rounded_palette_lab` (the rest is in tests/pipeline.rs, where
/// the environment is not changed in process): `_meta.json` records the thread count, and
/// the environment is not hashed.
#[test]
fn meta_environment_threads_from_env() {
    let tmp = TempDir::new("threads-env");
    let mut cmd = pixelforge();
    cmd.env("PIXELFORGE_THREADS", "3");
    let r = run_with(
        cmd,
        &[
            &"convert",
            &fixture("circle_alpha"),
            &"-o",
            &tmp.path(),
            &"--method",
            &"box",
            &"--out-height",
            &"16",
            &"--palette-size",
            &"4",
        ],
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let meta = read_json(&tmp.join("circle_alpha_meta.json"));
    let env = meta["environment"].as_object().unwrap();
    let mut keys: Vec<&str> = env.keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(keys, ["native", "platform", "threads"]);
    assert_eq!(env["threads"], 3);
    let config = Config::new(&[
        ("method".to_string(), Value::str("box")),
        ("out_height".to_string(), Value::Int(16)),
        ("palette_size".to_string(), Value::Int(4)),
    ])
    .unwrap();
    assert_eq!(meta["config_hash"], config.hash().unwrap().as_str());
}

// ---------------------------------------------------------------------------------------
// tests/test_determinism.py (fresh processes)

/// Every output file of one `convert` in a fresh process, `_meta.json` without the timings
/// and the environment (wall clock and host only).
fn convert_in_process(
    dir: &Path,
    method: &str,
    preset: &str,
    threads: &str,
) -> BTreeMap<String, Vec<u8>> {
    let extra: &[&str] = if preset == "sprite" {
        &["--out-width", "16"]
    } else {
        &[
            "--out-width",
            "16",
            "--out-height",
            "16",
            "--tile-size",
            "8",
        ]
    };
    let out = pixelforge()
        .env("PIXELFORGE_THREADS", threads)
        .arg("convert")
        .arg(fixture("two_color"))
        .arg("-o")
        .arg(dir)
        .args(["--method", method, "--preset", preset])
        .args(extra)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let mut data = std::fs::read(&p).unwrap();
        if name.ends_with("_meta.json") {
            let mut meta: serde_json::Value = serde_json::from_slice(&data).unwrap();
            meta["stats"].as_object_mut().unwrap().remove("timings");
            meta.as_object_mut().unwrap().remove("environment");
            data = serde_json::to_vec(&meta).unwrap();
        }
        files.insert(name, data);
    }
    files
}

#[test]
fn outputs_identical_across_processes_and_thread_counts() {
    for preset in ["background", "sprite"] {
        for method in ["box", "kopf", "gerstner"] {
            let tmp = TempDir::new("processes");
            let runs: Vec<BTreeMap<String, Vec<u8>>> = ["1", "2", "3", "8"]
                .iter()
                .map(|t| convert_in_process(&tmp.join(t), method, preset, t))
                .collect();
            let mut expected: Vec<String> = [
                "two_color.png",
                "two_color_preview.png",
                "two_color_palette.json",
                "two_color_palette.hex",
                "two_color_meta.json",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            if preset == "background" {
                expected.extend(
                    [
                        "two_color_tileset.png",
                        "two_color_tilemap.json",
                        "two_color.tmj",
                        "two_color_tilemap.csv",
                    ]
                    .iter()
                    .map(|s| s.to_string()),
                );
            }
            expected.sort();
            for run in &runs {
                let names: Vec<String> = run.keys().cloned().collect();
                assert_eq!(names, expected, "{preset} {method}");
            }
            for run in &runs[1..] {
                for name in &expected {
                    assert!(
                        run[name] == runs[0][name],
                        "{preset} {method}: {name} differs"
                    );
                }
            }
        }
    }
}

#[test]
fn stdout_is_only_json_and_stderr_only_prefixed_lines() {
    let tmp = TempDir::new("streams");
    let r = invoke!(
        "-v",
        "convert",
        &fixture("two_color"),
        "-o",
        tmp.path(),
        FAST
    );
    assert_eq!(r.code, 0, "{}", r.output());
    assert_eq!(r.stdout.lines().count(), 1);
    let payload = r.last_json();
    let keys: Vec<&String> = payload.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["outputs", "stats"]);
    // Compact, sorted keys.
    assert!(r.stdout.starts_with("{\"outputs\":{\"image\":"));
    assert!(!r.stderr.is_empty() && r.stderr.lines().all(|l| l.starts_with("pixelforge: ")));
}

// ---------------------------------------------------------------------------------------
// compare (tests/test_cli.py, tests/test_pipeline.py)

/// The files of a directory, sorted.
fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// The PNG's colour type (the sheets are plain RGBA, without text chunks).
fn png_color_type(path: &Path) -> png::ColorType {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    decoder.read_info().unwrap().info().color_type
}

#[test]
fn cli_compare() {
    let tmp = TempDir::new("compare");
    let r = invoke!(
        "compare",
        &fixture("two_color"),
        "-o",
        tmp.path(),
        &[
            "--out-height",
            "8",
            "--palette-size",
            "2",
            "--method",
            "kopf"
        ][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let payload = r.last_json();
    let sheet = tmp.join("two_color_compare.png");
    assert_eq!(payload["compare"], sheet.display().to_string());
    let methods = payload["methods"].as_object().unwrap();
    let keys: Vec<&String> = methods.keys().collect();
    assert_eq!(keys, ["box", "gerstner", "kopf"]);
    let mut widths = 0;
    let mut height = 0;
    for (method, entry) in methods {
        // --method is ignored: every method runs with its own name in the config.
        assert_eq!(entry["stats"]["config_hash"].as_str().unwrap().len(), 64);
        let image = tmp.join(&format!("two_color_{method}.png"));
        assert_eq!(entry["outputs"]["image"], image.display().to_string());
        for (_, path) in entry["outputs"].as_object().unwrap() {
            assert!(Path::new(path.as_str().unwrap()).is_file(), "{path}");
        }
        let meta = read_json(&tmp.join(&format!("two_color_{method}_meta.json")));
        assert_eq!(meta["config"]["method"], method.as_str());
        let (w, h) = size(&tmp.join(&format!("two_color_{method}_preview.png")));
        widths += w;
        height = height.max(h);
    }
    // The previews in a row, scale_preview (8) pixels apart, as plain RGBA.
    assert_eq!(size(&sheet), (widths + 2 * 8, height));
    assert_eq!(png_color_type(&sheet), png::ColorType::Rgba);
    assert!(pixelforge::io::read_png_text(&sheet).unwrap().is_empty());
    let (px, w, _) = rgba(&sheet);
    let (box_w, _) = size(&tmp.join("two_color_box_preview.png"));
    let (box_px, _, _) = rgba(&tmp.join("two_color_box_preview.png"));
    assert_eq!(&px[..box_w as usize * 4], &box_px[..box_w as usize * 4]);
    // The gap after the first preview is transparent.
    assert!(px[box_w as usize * 4..(box_w as usize + 8) * 4]
        .chunks(4)
        .all(|p| p[3] == 0));
    assert!(w > box_w);
}

#[test]
fn compare_prepass_writes_only_a_proof_sheet() {
    let tmp = TempDir::new("prepass");
    let r = invoke!(
        "-v",
        "compare",
        &fixture("noisy_gradient"),
        "-o",
        tmp.path(),
        "--prepass",
        &[
            "--out-height",
            "8",
            "--preset",
            "background",
            "--no-tileset"
        ][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    assert_eq!(listing(tmp.path()), ["noisy_gradient_compare.png"]);
    let payload = r.last_json();
    let cells = payload["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 3 * 4);
    let methods: Vec<&str> = cells
        .iter()
        .step_by(4)
        .map(|c| c["method"].as_str().unwrap())
        .collect();
    assert_eq!(methods, ["box", "kopf", "gerstner"]);
    let columns: Vec<&str> = cells[..4]
        .iter()
        .map(|c| c["column"].as_str().unwrap())
        .collect();
    assert_eq!(
        columns,
        [
            "enhance=0 ink=0",
            "enhance=0.5 ink=0",
            "enhance=1 ink=0",
            "enhance=1 ink=0.6"
        ]
    );
    let last = cells.last().unwrap();
    assert_eq!(last["flags"], "--method gerstner --enhance 1 --ink 0.6");
    assert_eq!(last["column"], "enhance=1 ink=0.6");
    assert_eq!(last["config"]["method"], "gerstner");
    assert_eq!(last["config"]["ink"], 0.6);
    assert_eq!(last["config"]["preset"], "background");
    let hashes: std::collections::BTreeSet<&str> = cells
        .iter()
        .map(|c| c["stats"]["config_hash"].as_str().unwrap())
        .collect();
    assert_eq!(hashes.len(), cells.len());
    // One progress line per cell.
    let progress: Vec<&str> = r
        .stderr
        .lines()
        .filter(|l| l.starts_with("pixelforge: cell "))
        .collect();
    assert_eq!(progress.len(), cells.len());
    assert_eq!(progress[0], "pixelforge: cell box / enhance=0 ink=0");
    let sheet = tmp.join("noisy_gradient_compare.png");
    assert_eq!(png_color_type(&sheet), png::ColorType::Rgba);
    assert!(pixelforge::io::read_png_text(&sheet).unwrap().is_empty());
    let (w, h) = size(&sheet);
    assert!(w > 4 * 8 * 8 && h > 3 * 8 * 8, "{w}x{h}");
}

#[test]
fn compare_cell_flags_reproduce_the_cell() {
    let tmp = TempDir::new("cellflags");
    let base = [
        "--out-height",
        "8",
        "--preset",
        "background",
        "--no-tileset",
        "--enhance",
        "1",
    ];
    let r = invoke!(
        "compare",
        &fixture("noisy_gradient"),
        "-o",
        &tmp.join("sheet"),
        &["--sweep", "palette-size=4,8", "--method", "box"][..],
        &base[..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let cells = r.last_json()["cells"].as_array().unwrap().clone();
    let summary: Vec<(&str, &str, &str)> = cells
        .iter()
        .map(|c| {
            (
                c["method"].as_str().unwrap(),
                c["column"].as_str().unwrap(),
                c["flags"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            ("box", "palette_size=4", "--method box --palette-size 4"),
            ("box", "palette_size=8", "--method box --palette-size 8")
        ]
    );
    assert_eq!(listing(&tmp.join("sheet")), ["noisy_gradient_compare.png"]);
    let flags: Vec<&str> = cells[1]["flags"].as_str().unwrap().split(' ').collect();
    let again = invoke!(
        "convert",
        &fixture("noisy_gradient"),
        "-o",
        &tmp.join("one"),
        &base[..],
        &flags[..]
    );
    assert_eq!(again.code, 0, "{}", again.output());
    assert_eq!(
        again.last_json()["stats"]["config_hash"],
        cells[1]["stats"]["config_hash"]
    );
}

#[test]
fn compare_cell_config_reproduces_an_unset_value() {
    let tmp = TempDir::new("cellconfig");
    let base = tmp.join("base.json");
    std::fs::write(
        &base,
        r#"{"preset": "background", "tileset": false, "out_width": 16, "method": "box"}"#,
    )
    .unwrap();
    let r = invoke!(
        "compare",
        &fixture("noisy_gradient"),
        "-o",
        &tmp.join("sheet"),
        "--config",
        &base,
        &["--method", "box", "--sweep", "out-width=none,8"][..]
    );
    assert_eq!(r.code, 0, "{}", r.output());
    let cell = r.last_json()["cells"][0].clone();
    assert_eq!(cell["column"], "out_width=none");
    assert_eq!(cell["flags"], "--method box");
    assert!(cell["config"]["out_width"].is_null());
    let replay = tmp.join("cell.json");
    std::fs::write(&replay, cell["config"].to_string()).unwrap();
    let again = invoke!(
        "convert",
        &fixture("noisy_gradient"),
        "-o",
        &tmp.join("one"),
        "--config",
        &replay
    );
    assert_eq!(again.code, 0, "{}", again.output());
    assert_eq!(
        again.last_json()["stats"]["config_hash"],
        cell["stats"]["config_hash"]
    );
}

#[test]
fn compare_sweep_errors() {
    let cases: &[(&[&str], &str)] = &[
        (&["--sweep", "nope=1"], "FIELD=VALUE"),
        (&["--sweep", "palette-size"], "FIELD=VALUE"),
        (&["--sweep", "method=box,kopf"], "rows of the sheet"),
        (&["--sweep", "palette-size=4,,8"], "non-empty"),
        (&["--sweep", "palette-size=four"], "is not int"),
        (&["--sweep", "key-bg=maybe"], "on/off"),
        (
            &["--sweep", "palette-size=4,999"],
            "palette_size must be in",
        ),
        (
            &["--sweep", "palette-size=4", "--prepass"],
            "config error: --prepass and --sweep cannot be combined",
        ),
        (
            &["--sweep", "palette-size=4", "--sweep", "dither=none"],
            "config error: --sweep can be given once: a sheet has one row per method and one \
             column per value",
        ),
        (
            &["--sweep", "palette-name=pico8,missing_palette.hex"],
            "missing_palette.hex",
        ),
    ];
    for (args, message) in cases {
        let tmp = TempDir::new("sweeperr");
        let out = tmp.join("out");
        let r = invoke!("compare", &fixture("two_color"), "-o", &out, *args);
        assert_eq!(r.code, 2, "{args:?}: {}", r.output());
        assert!(
            r.stderr.starts_with("config error: "),
            "{args:?}: {}",
            r.stderr
        );
        assert!(r.stderr.contains(message), "{args:?}: {}", r.stderr);
        assert!(!out.exists(), "{args:?}");
    }
}

#[cfg(unix)]
#[test]
fn compare_refuses_to_overwrite_its_input() {
    // An output that resolves to the input (here through a symlink) is refused, in both
    // modes, before anything runs.
    let tmp = TempDir::new("comparecollide");
    let src = tmp.join("two_color.png");
    std::fs::copy(fixture("two_color"), &src).unwrap();
    let before = std::fs::read(&src).unwrap();
    let out = tmp.join("out");
    std::fs::create_dir(&out).unwrap();
    for (link, extra) in [
        ("two_color_kopf_preview.png", &[][..]),
        (
            "two_color_compare.png",
            &["--prepass", "--method", "box"][..],
        ),
    ] {
        let alias = out.join(link);
        std::os::unix::fs::symlink(&src, &alias).unwrap();
        let r = invoke!("compare", &src, "-o", &out, FAST, extra);
        assert_eq!(r.code, 2, "{}", r.output());
        assert_eq!(
            r.stderr.trim(),
            format!(
                "config error: {} would overwrite input {} (use another --outdir, or --force)",
                alias.display(),
                src.display()
            )
        );
        assert_eq!(listing(&out), [link]);
        std::fs::remove_file(&alias).unwrap();
    }
    assert_eq!(std::fs::read(&src).unwrap(), before);
}

#[test]
fn compare_help_lists_its_options() {
    let r = invoke!("compare", "--help");
    assert_eq!(r.code, 0);
    assert!(r.stdout.contains("--sweep <FIELD=V1,V2,...>"));
    assert!(r.stdout.contains("--prepass") && r.stdout.contains("INPUT_compare.png"));
}

#[test]
fn stdout_contains_predicate_smoke() {
    // assert_cmd's own assertions on the palettes listing.
    pixelforge()
        .arg("palettes")
        .assert()
        .success()
        .stdout(predicate::str::contains("gameboy\t4 colors"));
}
