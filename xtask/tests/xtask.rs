//! The xtask commands against the committed files: fixtures bitwise, golden up to date, the
//! bench JSON schema (also the Python-written files), snapshot save/compare, the gallery.

use std::path::{Path, PathBuf};
use std::process::Command;

use pixelforge::pyjson::{dumps, Json, Layout};
use xtask::bench::{self, CaseResult};
use xtask::json::{get, loads};
use xtask::snapshot::{self, Snapshot};
use xtask::{fixtures, golden, root};

/// A fresh temporary directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let path = std::env::temp_dir().join(format!("xtask-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn xtask(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn every_fixture_is_drawn_bitwise() {
    for name in fixtures::FIXTURES {
        let committed = fixtures::decode(&fixtures::default_dir().join(format!("{name}.png")))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let drawn = fixtures::draw(name).unwrap();
        assert_eq!(
            fixtures::differences(&committed, &drawn, 10),
            Vec::<String>::new(),
            "tests/fixtures/{name}.png differs from xtask fixtures"
        );
    }
}

#[test]
fn written_fixtures_decode_to_the_drawn_pixels() {
    let dir = TempDir::new("fixtures");
    let written = fixtures::write(&dir.0).unwrap();
    assert_eq!(written.len(), fixtures::FIXTURES.len());
    let out = xtask(&["fixtures", "--check", dir.0.to_str().unwrap()]);
    assert!(out.status.success(), "{}", text(&out.stderr));
}

#[test]
fn fixtures_check_reports_a_changed_pixel_and_a_missing_file() {
    let dir = TempDir::new("fixtures-bad");
    fixtures::write(&dir.0).unwrap();
    let mut image = fixtures::draw("two_color").unwrap();
    image.data[0] = 7;
    std::fs::write(
        dir.0.join("two_color.png"),
        fixtures::encode(&image).unwrap(),
    )
    .unwrap();
    std::fs::remove_file(dir.0.join("gradient6.png")).unwrap();
    let out = xtask(&["fixtures", "--check", dir.0.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(err.contains("two_color.png: pixel (y=0, x=0)"), "{err}");
    assert!(err.contains("missing test fixture gradient6"), "{err}");
}

#[test]
fn golden_file_is_up_to_date() {
    let path = golden::golden_path();
    let committed = std::fs::read_to_string(&path).unwrap();
    let parsed = loads(&committed).unwrap();
    let mut keys: Vec<String> = xtask::json::entries(&parsed)
        .iter()
        .map(|(k, _)| k.clone())
        .collect();
    keys.sort();
    let mut expected = golden::golden_keys();
    expected.sort();
    assert_eq!(keys, expected);
    assert_eq!(golden::render(&parsed), committed);
    // The pixels (the --check of `cargo xtask golden`).
    assert!(
        golden::main(&path, true).is_ok(),
        "tests/golden.json is out of date"
    );
}

#[test]
fn golden_check_fails_on_a_stale_file() {
    let dir = TempDir::new("golden");
    let stale = dir.0.join("golden.json");
    std::fs::write(&stale, "{}\n").unwrap();
    let err = golden::main(&stale, true).unwrap_err();
    assert_eq!(err.code, 1);
    assert_eq!(err.message, format!("{} is out of date", stale.display()));
}

#[test]
fn python_bench_files_round_trip_and_compare() {
    for name in ["0edd462.json", "9f53516.json"] {
        let path = root().join("bench").join(name);
        let content = std::fs::read_to_string(&path).unwrap();
        let report = bench::read_report(&path).unwrap();
        assert_eq!(dumps(&report, Layout::INDENT2) + "\n", content, "{name}");
        let cases = bench::cases_of(&report);
        assert!(!cases.is_empty());
        let rebuilt: Vec<Json> = cases.iter().map(CaseResult::to_json).collect();
        assert_eq!(Some(&Json::List(rebuilt)), get(&report, "cases"), "{name}");
    }
    // Comparing the old Python bench with itself: every case matched, speedup 1.00x.
    let old = bench::read_report(&root().join("bench").join("0edd462.json")).unwrap();
    let table = bench::compare_table(&old, &bench::cases_of(&old));
    let lines: Vec<&str> = table.lines().collect();
    assert!(lines[0].starts_with("input") && lines[0].contains("speedup"));
    assert!(lines[2..].iter().all(|l| l.contains("1.00x")), "{table}");
}

#[test]
fn bench_writes_the_schema_and_compares() {
    let dir = TempDir::new("bench");
    let out_path = dir.0.join("bench.json");
    let case_args = [
        "--inputs",
        "two_color",
        "--methods",
        "box",
        "--presets",
        "sprite",
        "--no-warmup",
    ];
    let mut args = vec!["bench"];
    args.extend(case_args);
    args.extend(["--out", out_path.to_str().unwrap()]);
    let out = xtask(&args);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("two_color"));

    let report = bench::read_report(&out_path).unwrap();
    let keys: Vec<&str> = xtask::json::entries(&report)
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(
        keys,
        [
            "timestamp",
            "git_sha",
            "git_dirty",
            "package_tree",
            "package_dirty",
            "machine",
            "platform",
            "python",
            "numpy",
            "scipy",
            "rustc",
            "pixelforge",
            "env",
            "repeat",
            "warmup",
            "cases"
        ]
    );
    assert_eq!(get(&report, "python"), Some(&Json::Null));
    assert_eq!(get(&report, "warmup"), Some(&Json::Bool(false)));
    let (case,) = match bench::cases_of(&report).as_slice() {
        [c] => (c.clone(),),
        other => panic!("{other:?}"),
    };
    assert_eq!(
        (
            case.label.as_str(),
            case.preset.as_str(),
            case.method.as_str()
        ),
        ("two_color", "sprite", "box")
    );
    assert_eq!(case.input, "tests/fixtures/two_color.png");
    assert!(case.total_s > 0.0 || case.timings.iter().any(|(k, _)| k == "downscale"));
    assert_eq!((case.iterations, case.output_size), (1, [64, 64]));
    // The file is exactly what json.dumps(indent=2) writes for the parsed report.
    assert_eq!(
        dumps(&report, Layout::INDENT2) + "\n",
        std::fs::read_to_string(&out_path).unwrap()
    );

    let mut args = vec!["bench"];
    args.extend(case_args);
    args.extend(["--no-write", "--compare", out_path.to_str().unwrap()]);
    let out = xtask(&args);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("speedup"));

    // Against the Python baseline (which has no two_color case): "-" columns, exit 0.
    let old = root().join("bench").join("0edd462.json");
    let mut args = vec!["bench"];
    args.extend(case_args);
    args.extend(["--no-write", "--compare", old.to_str().unwrap()]);
    let out = xtask(&args);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("compare: "));
}

#[test]
fn bench_errors_exit_like_the_script() {
    let out = xtask(&["bench", "--presets", "nope", "--no-write"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        text(&out.stderr).trim(),
        "unknown preset 'nope'; choose from sprite, background"
    );
    let out = xtask(&["bench", "--inputs", "no_such_input", "--no-write"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("input not found: no_such_input"));
    let out = xtask(&["bench", "--bogus"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn same_stem_inputs_get_distinct_keys() {
    let dir = TempDir::new("stems");
    let fixture = fixtures::default_dir().join("two_color.png");
    let (a, b) = (dir.0.join("a/two_color.png"), dir.0.join("b/two_color.png"));
    for p in [&a, &b] {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::copy(&fixture, p).unwrap();
    }
    let unique = fixtures::default_dir().join("line_diag.png");
    let names: Vec<String> = [&a, &b, &unique]
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    let cases = bench::build_cases(&names, &["sprite".into()], &["box".into()], &[]).unwrap();
    let keys = bench::case_keys(&cases);
    assert!(keys[0].starts_with("two_color-") && keys[1].starts_with("two_color-"));
    assert_ne!(keys[0], keys[1]);
    assert_eq!(keys[2], "line_diag/sprite/box");
}

fn save_two(dir: &Path) {
    let cases = bench::build_cases(
        &["two_color".into(), "line_diag".into()],
        &["sprite".into()],
        &["box".into()],
        &[],
    )
    .unwrap();
    assert_eq!(snapshot::save(dir, &cases).unwrap(), 2);
}

#[test]
fn snapshot_save_then_compare_is_identical() {
    let dir = TempDir::new("snap");
    let snap = dir.0.join("snap");
    let out = xtask(&[
        "snapshot",
        "save",
        snap.to_str().unwrap(),
        "--inputs",
        "two_color",
        "line_diag",
        "--methods",
        "box",
        "--presets",
        "sprite",
    ]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("saved 2 case(s)"));
    let manifest = loads(&std::fs::read_to_string(snap.join("manifest.json")).unwrap()).unwrap();
    let cases = xtask::json::entries(get(&manifest, "cases").unwrap());
    assert_eq!(
        cases.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        ["two_color/sprite/box", "line_diag/sprite/box"]
    );
    assert_eq!(
        get(&cases[0].1, "input"),
        Some(&Json::str("tests/fixtures/two_color.png"))
    );
    // From another working directory: the relative inputs resolve against the root.
    let out = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["snapshot", "compare", snap.to_str().unwrap()])
        .current_dir(&dir.0)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let stdout = text(&out.stdout);
    assert!(stdout.contains("all identical"), "{stdout}");
    assert!(stdout.contains("ΔE mean / max"), "{stdout}");
}

#[test]
fn snapshot_negative_tolerance_is_a_usage_error() {
    let dir = TempDir::new("snap-negative");
    let snap = dir.0.to_str().unwrap();
    let out = xtask(&["snapshot", "compare", snap, "--tolerance-pixels=-1"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(text(&out.stderr).contains("invalid value '-1' for '--tolerance-pixels <N>'"));
    // Without `=`, `-1` reads as an unknown flag: also a usage error.
    let out = xtask(&["snapshot", "compare", snap, "--tolerance-pixels", "-1"]);
    assert_eq!(out.status.code(), Some(2));
}

fn data_path(snap: &Path, key: &str) -> PathBuf {
    snap.join(snapshot::case_filename(key))
}

fn rewrite(path: &Path, f: impl FnOnce(&mut Snapshot)) {
    let mut data =
        Snapshot::from_json(&loads(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
    f(&mut data);
    std::fs::write(path, dumps(&data.to_json(), Layout::COMPACT)).unwrap();
}

#[test]
fn snapshot_one_pixel_changed() {
    let dir = TempDir::new("snap-pixel");
    save_two(&dir.0);
    rewrite(&data_path(&dir.0, "two_color/sprite/box"), |s| {
        let at = s.indices.iter().position(|&i| i >= 0).unwrap();
        let current = s.palette[s.indices[at] as usize];
        let other = s.palette.iter().position(|c| *c != current).unwrap();
        s.indices[at] = other as i64;
    });
    let (code, reports) = snapshot::compare(&dir.0, 0).unwrap();
    assert_eq!(code, 1);
    let r = &reports[0];
    assert!(!r.identical && r.diff_pixels == 1 && r.palette_identical && r.config_hash_match);
    assert!(r.max_delta_e.unwrap() > 0.0);
    assert!(reports[1].identical);
    assert_eq!(snapshot::compare(&dir.0, 1).unwrap().0, 0);
}

#[test]
fn snapshot_reordered_palette_is_pixels_identical_only() {
    let dir = TempDir::new("snap-order");
    save_two(&dir.0);
    rewrite(&data_path(&dir.0, "two_color/sprite/box"), |s| {
        assert!(s.palette.len() >= 2);
        let k = s.palette.len() as i64;
        s.palette.reverse();
        let lab: Vec<[f64; 3]> = s
            .palette_lab
            .chunks_exact(3)
            .rev()
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        s.palette_lab = lab.concat();
        for i in &mut s.indices {
            if *i >= 0 {
                *i = k - 1 - *i;
            }
        }
    });
    let (code, reports) = snapshot::compare(&dir.0, 0).unwrap();
    assert_eq!(code, 1);
    let r = &reports[0];
    assert!(r.pixels_identical && r.diff_pixels == 0);
    assert!(!r.identical && !r.palette_identical);
}

#[test]
fn snapshot_compare_without_manifest_fails() {
    let dir = TempDir::new("snap-none");
    let err = snapshot::compare(&dir.0, 0).unwrap_err();
    assert_eq!(err.code, 1);
    assert!(err.message.starts_with("no snapshot manifest at "));
}

fn check_gallery(index: &Path, stems: &[&str]) {
    let html = std::fs::read_to_string(index).unwrap();
    assert!(html.starts_with("<!doctype html>\n"));
    assert_eq!(
        html.matches("<tr><td class=\"name\">").count(),
        stems.len() * 2
    );
    assert_eq!(html.matches("_preview.png\"><img").count(), stems.len() * 6);
    assert_eq!(html.matches("iterations</small>").count(), stems.len() * 6);
    let img = index.parent().unwrap().join("img");
    for name in stems {
        assert!(img.join(format!("{name}_input.png")).is_file());
        assert!(img
            .join(format!("{name}_background_kopf_preview.png"))
            .is_file());
    }
}

#[test]
fn gallery_quick_writes_one_row_per_input_and_preset() {
    let dir = TempDir::new("gallery");
    let stems = ["two_color", "seamless_src"];
    let inputs: Vec<PathBuf> = stems
        .iter()
        .map(|n| fixtures::default_dir().join(format!("{n}.png")))
        .collect();
    let index = xtask::gallery::run(&dir.0.join("out"), &inputs, true).unwrap();
    check_gallery(&index, &stems);
}

#[test]
fn gallery_same_stem_inputs_get_distinct_image_sets() {
    let dir = TempDir::new("gallery-stems");
    let fixture = fixtures::default_dir().join("two_color.png");
    let sample = dir.0.join("samples/two_color.png");
    std::fs::create_dir_all(sample.parent().unwrap()).unwrap();
    // A different image under the same stem, so an overwrite would be visible.
    std::fs::copy(fixtures::default_dir().join("line_diag.png"), &sample).unwrap();
    let inputs = vec![fixture.clone(), sample.clone()];
    let index = xtask::gallery::run(&dir.0.join("out"), &inputs, true).unwrap();
    let labels: Vec<String> = bench::input_labels(inputs.iter().map(PathBuf::as_path))
        .into_iter()
        .map(|(_, l)| l)
        .collect();
    assert_eq!(labels.len(), 2);
    assert_ne!(labels[0], labels[1]);
    assert!(labels.iter().all(|l| l.starts_with("two_color-")));
    let html = std::fs::read_to_string(&index).unwrap();
    // Both rows keep the stem as their caption.
    assert_eq!(
        html.matches("<tr><td class=\"name\">two_color<small>")
            .count(),
        4
    );
    let img = index.parent().unwrap().join("img");
    for (label, source) in labels.iter().zip([&fixture, &sample]) {
        let copy = img.join(format!("{label}_input.png"));
        assert_eq!(
            std::fs::read(&copy).unwrap(),
            std::fs::read(source).unwrap()
        );
        assert!(html.contains(&format!("img/{label}_input.png")));
        for preset in ["sprite", "background"] {
            for method in ["box", "kopf", "gerstner"] {
                let preview = format!("{label}_{preset}_{method}_preview.png");
                assert!(img.join(&preview).is_file(), "{preview}");
                assert!(html.contains(&format!("img/{preview}")), "{preview}");
            }
        }
    }
    assert!(!img.join("two_color_input.png").exists());
}

/// Every fixture (42 pipeline runs; about 100 s in a debug build).
#[test]
#[ignore = "slow: run with --ignored"]
fn gallery_quick_on_every_fixture() {
    let dir = TempDir::new("gallery-all");
    let out = xtask(&[
        "gallery",
        "--quick",
        "--out",
        dir.0.join("out").to_str().unwrap(),
        "--samples",
        dir.0.join("no-samples").to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    check_gallery(&dir.0.join("out/index.html"), &fixtures::FIXTURES);
}
