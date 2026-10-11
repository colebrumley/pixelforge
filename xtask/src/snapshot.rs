//! `xtask snapshot save|compare`: save the pipeline's outputs per case, and compare a later
//! run against them (`scripts/snapshot_outputs.py`).
//!
//! `save` runs the same case matrix as `xtask bench` and writes, per case, a JSON file with
//! `shape`, `indices`, `palette` (RGB8) and `palette_lab`, plus `DIR/manifest.json` with the
//! git sha and each case's input path and `Config.hash()`. Input paths inside the repository
//! are stored relative to the repository root (and resolved against it by `compare`,
//! whatever the working directory); other inputs are stored as absolute paths. The Python
//! script's `.npz` snapshots are not readable here: re-save them.
//!
//! "identical" means the same indices and palette, i.e. the same indexed PNG; "pixels
//! identical" means the same rendered colors and transparency, which a reordered palette
//! with remapped indices also satisfies. Exit code 0 if every case is identical, or with
//! `--tolerance-pixels N > 0` if every case has at most N differing pixels; 1 otherwise.

use std::path::{Path, PathBuf};

use clap::{Arg, ArgMatches};
use pixelforge::color::{delta_e, rgb8_to_lab};
use pixelforge::npsum::pairwise_mean;
use pixelforge::pipeline::{self, Input, Output};
use pixelforge::pyjson::{dumps, Json, Layout};

use crate::bench::{self, Case};
use crate::json::{as_bool, as_f64, as_i64, as_list, as_str, entries, get, loads};
use crate::{display_path, fail, root, Fail};

pub const MANIFEST: &str = "manifest.json";
pub const IDENTICAL_HELP: &str = "\"identical\" = same indices and palette, i.e. the same PNG; \
                                  \"pixels identical\" = same rendered colors.";

pub fn default_dir() -> PathBuf {
    root().join("bench").join("snapshots")
}

pub fn case_filename(key: &str) -> String {
    key.replace('/', "__") + ".json"
}

/// The inverse of the manifest's input path: relative paths are relative to the root.
pub fn resolve_manifest_path(stored: &str) -> PathBuf {
    let path = PathBuf::from(stored);
    if path.is_absolute() {
        path
    } else {
        root().join(path)
    }
}

fn run_case(path: &Path, preset: &str, method: &str) -> Result<(String, Output), Fail> {
    let config = bench::case_config(preset, method)?;
    let out = pipeline::run(Input::Path(path), &config, None)
        .map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
    let hash = config.hash().map_err(|e| fail(1, e.to_string()))?;
    Ok((hash, out))
}

/// A stored (or new) result: `(h, w)` indices (`< 0` transparent) and the RGB8 palette.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub h: usize,
    pub w: usize,
    pub indices: Vec<i64>,
    pub palette: Vec<[u8; 3]>,
    pub palette_lab: Vec<f64>,
}

impl Snapshot {
    pub fn of(out: &Output) -> Snapshot {
        Snapshot {
            h: out.indices.h,
            w: out.indices.w,
            indices: out.indices.indices.clone(),
            palette: out.palette.clone(),
            palette_lab: out.palette_lab.clone(),
        }
    }

    pub fn to_json(&self) -> Json {
        let rows = (0..self.h)
            .map(|y| {
                Json::List(
                    self.indices[y * self.w..(y + 1) * self.w]
                        .iter()
                        .map(|&i| Json::Int(i))
                        .collect(),
                )
            })
            .collect();
        Json::Object(vec![
            (
                "shape".into(),
                Json::List(vec![Json::Int(self.h as i64), Json::Int(self.w as i64)]),
            ),
            ("indices".into(), Json::List(rows)),
            (
                "palette".into(),
                Json::List(
                    self.palette
                        .iter()
                        .map(|c| Json::List(c.iter().map(|&v| Json::Int(i64::from(v))).collect()))
                        .collect(),
                ),
            ),
            (
                "palette_lab".into(),
                Json::List(
                    self.palette_lab
                        .chunks_exact(3)
                        .map(|c| Json::List(c.iter().map(|&v| Json::Float(v)).collect()))
                        .collect(),
                ),
            ),
        ])
    }

    pub fn from_json(value: &Json) -> Result<Snapshot, String> {
        let bad = |what: &str| format!("snapshot data: bad or missing {what}");
        let shape = get(value, "shape")
            .and_then(as_list)
            .ok_or_else(|| bad("shape"))?;
        let dim = |n: usize| {
            shape
                .get(n)
                .and_then(as_i64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| bad("shape"))
        };
        let (h, w) = (dim(0)?, dim(1)?);
        let mut indices = Vec::with_capacity(h * w);
        for row in get(value, "indices")
            .and_then(as_list)
            .ok_or_else(|| bad("indices"))?
        {
            let row = as_list(row).ok_or_else(|| bad("indices"))?;
            if row.len() != w {
                return Err(bad("indices"));
            }
            for v in row {
                indices.push(as_i64(v).ok_or_else(|| bad("indices"))?);
            }
        }
        if indices.len() != h * w {
            return Err(bad("indices"));
        }
        let mut palette = Vec::new();
        for c in get(value, "palette")
            .and_then(as_list)
            .ok_or_else(|| bad("palette"))?
        {
            let c = as_list(c)
                .filter(|c| c.len() == 3)
                .ok_or_else(|| bad("palette"))?;
            let mut rgb = [0u8; 3];
            for (dst, v) in rgb.iter_mut().zip(c) {
                *dst = as_i64(v)
                    .and_then(|v| u8::try_from(v).ok())
                    .ok_or_else(|| bad("palette"))?;
            }
            palette.push(rgb);
        }
        let mut palette_lab = Vec::new();
        for c in get(value, "palette_lab")
            .and_then(as_list)
            .ok_or_else(|| bad("palette_lab"))?
        {
            for v in as_list(c).ok_or_else(|| bad("palette_lab"))? {
                palette_lab.push(as_f64(v).ok_or_else(|| bad("palette_lab"))?);
            }
        }
        Ok(Snapshot {
            h,
            w,
            indices,
            palette,
            palette_lab,
        })
    }
}

/// `save`: run every case and store its result; returns the number of cases.
pub fn save(directory: &Path, cases: &[Case]) -> Result<usize, Fail> {
    std::fs::create_dir_all(directory)
        .map_err(|e| fail(1, format!("{}: {e}", directory.display())))?;
    let mut manifest_cases = Vec::new();
    for (case, key) in cases.iter().zip(bench::case_keys(cases)) {
        eprintln!("saving {key} ...");
        let (hash, out) = run_case(&case.path, &case.preset, &case.method)?;
        let file = case_filename(&key);
        let path = directory.join(&file);
        std::fs::write(
            &path,
            dumps(&Snapshot::of(&out).to_json(), Layout::COMPACT) + "\n",
        )
        .map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
        // A repeated key (the same case listed twice) keeps its first position.
        let entry = Json::Object(vec![
            ("input".into(), Json::Str(display_path(&case.path))),
            ("preset".into(), Json::Str(case.preset.clone())),
            ("method".into(), Json::Str(case.method.clone())),
            ("config_hash".into(), Json::Str(hash)),
            ("file".into(), Json::Str(file)),
        ]);
        match manifest_cases
            .iter_mut()
            .find(|(k, _): &&mut (String, Json)| *k == key)
        {
            Some(existing) => existing.1 = entry,
            None => manifest_cases.push((key, entry)),
        }
    }
    let n = manifest_cases.len();
    let mut manifest = bench::git_info();
    manifest.push(("cases".into(), Json::Object(manifest_cases)));
    let path = directory.join(MANIFEST);
    std::fs::write(
        &path,
        dumps(&Json::Object(manifest), Layout::INDENT2) + "\n",
    )
    .map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
    Ok(n)
}

/// The differences between a stored snapshot and a new result.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub key: String,
    pub identical: bool,
    pub pixels_identical: bool,
    pub shape_mismatch: bool,
    pub diff_pixels: usize,
    pub alpha_diff_pixels: Option<usize>,
    pub mean_delta_e: Option<f64>,
    pub max_delta_e: Option<f64>,
    pub palette_identical: bool,
    pub config_hash_match: bool,
}

fn color(palette: &[[u8; 3]], i: i64) -> [u8; 3] {
    palette.get(i.max(0) as usize).copied().unwrap_or([0, 0, 0])
}

/// `compare_arrays`: count differences in rendered colors, not raw indices, so a reordered
/// palette with remapped indices shows 0 differing pixels (`pixels_identical`) while
/// `identical` stays false (the indexed PNG differs). ΔE is CIE76 over the pixels that are
/// opaque in both and differ in color.
pub fn compare_arrays(old: &Snapshot, new: &Snapshot) -> Report {
    let palette_identical = old.palette == new.palette;
    let mut report = Report {
        key: String::new(),
        identical: false,
        pixels_identical: false,
        shape_mismatch: true,
        diff_pixels: old.indices.len().max(new.indices.len()),
        alpha_diff_pixels: None,
        mean_delta_e: None,
        max_delta_e: None,
        palette_identical,
        config_hash_match: true,
    };
    if (old.h, old.w) != (new.h, new.w) {
        return report;
    }
    let mut old_rgb = Vec::new();
    let mut new_rgb = Vec::new();
    let (mut n_diff, mut alpha_diff) = (0, 0);
    for (&a, &b) in old.indices.iter().zip(&new.indices) {
        let (ca, cb) = (color(&old.palette, a), color(&new.palette, b));
        let color_diff = ca != cb && a >= 0 && b >= 0;
        let a_diff = (a >= 0) != (b >= 0);
        if color_diff {
            old_rgb.extend_from_slice(&ca);
            new_rgb.extend_from_slice(&cb);
        }
        alpha_diff += usize::from(a_diff);
        n_diff += usize::from(color_diff || a_diff);
    }
    let (mean_de, max_de) = if old_rgb.is_empty() {
        (0.0, 0.0)
    } else {
        let de = delta_e(&rgb8_to_lab(&old_rgb), &rgb8_to_lab(&new_rgb));
        let max = de.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (pairwise_mean(&de), max)
    };
    report.shape_mismatch = false;
    report.diff_pixels = n_diff;
    report.alpha_diff_pixels = Some(alpha_diff);
    report.mean_delta_e = Some(mean_de);
    report.max_delta_e = Some(max_de);
    report.pixels_identical = n_diff == 0;
    report.identical = n_diff == 0 && palette_identical && old.indices == new.indices;
    report
}

/// `compare`: re-run every case of the manifest; the exit code (0 or 1) and the reports.
pub fn compare(directory: &Path, tolerance_pixels: i64) -> Result<(i32, Vec<Report>), Fail> {
    let manifest_path = directory.join(MANIFEST);
    if !manifest_path.is_file() {
        return Err(fail(
            1,
            format!(
                "no snapshot manifest at {}; run `save` first",
                manifest_path.display()
            ),
        ));
    }
    let read = |path: &Path| -> Result<Json, Fail> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
        loads(&text).map_err(|e| fail(1, format!("{}: {e}", path.display())))
    };
    let manifest = read(&manifest_path)?;
    let mut reports = Vec::new();
    let empty = Json::Object(vec![]);
    for (key, entry) in entries(get(&manifest, "cases").unwrap_or(&empty)) {
        eprintln!("comparing {key} ...");
        let field = |k: &str| {
            get(entry, k).and_then(as_str).ok_or_else(|| {
                fail(
                    1,
                    format!("{}: case {key} has no {k}", manifest_path.display()),
                )
            })
        };
        let (hash, out) = run_case(
            &resolve_manifest_path(field("input")?),
            field("preset")?,
            field("method")?,
        )?;
        let data_path = directory.join(field("file")?);
        if data_path.extension().is_some_and(|e| e == "npz") {
            return Err(fail(
                1,
                format!(
                    "{}: .npz snapshots (scripts/snapshot_outputs.py) are not readable by \
                     xtask snapshot; re-save the snapshot directory",
                    data_path.display()
                ),
            ));
        }
        let old = Snapshot::from_json(&read(&data_path)?)
            .map_err(|e| fail(1, format!("{}: {e}", data_path.display())))?;
        let mut report = compare_arrays(&old, &Snapshot::of(&out));
        report.key = key.clone();
        report.config_hash_match = field("config_hash")? == hash;
        reports.push(report);
    }

    let rows: Vec<Vec<String>> = reports
        .iter()
        .map(|r| {
            let de = match (r.mean_delta_e, r.max_delta_e) {
                (Some(mean), Some(max)) => format!("{mean:.2} / {max:.2}"),
                _ => "-".into(),
            };
            let yes_no = |b: bool, no: &str| if b { "yes".to_string() } else { no.to_string() };
            vec![
                r.key.clone(),
                yes_no(r.identical, "NO"),
                yes_no(r.pixels_identical, "no"),
                format!(
                    "{}{}",
                    r.diff_pixels,
                    if r.shape_mismatch { " (shape)" } else { "" }
                ),
                de,
                yes_no(r.palette_identical, "no"),
                yes_no(r.config_hash_match, "NO (not like-for-like)"),
            ]
        })
        .collect();
    let short = |s: &str| s.chars().take(7).collect::<String>();
    let current = bench::git_info();
    let current_sha = current
        .iter()
        .find(|(k, _)| k == "git_sha")
        .and_then(|(_, v)| as_str(v))
        .unwrap_or("?");
    println!(
        "snapshot {} (git {}{}) vs {}",
        directory.display(),
        short(get(&manifest, "git_sha").and_then(as_str).unwrap_or("?")),
        if get(&manifest, "git_dirty").and_then(as_bool) == Some(true) {
            ", dirty"
        } else {
            ""
        },
        short(current_sha)
    );
    println!(
        "{}",
        bench::format_table(
            &rows,
            &[
                "case",
                "identical",
                "pixels same",
                "diff px",
                "ΔE mean / max",
                "palette same",
                "config hash same",
            ],
        )
    );
    if reports.iter().any(|r| !r.config_hash_match) {
        println!(
            "note: config hash changed for some cases; those comparisons are not \
             like-for-like (the configuration itself differs)."
        );
    }
    // With N = 0 any difference fails (a changed palette included); with N > 0 a case fails
    // only if more than N pixels differ.
    let failing: Vec<&str> = reports
        .iter()
        .filter(|r| {
            !r.identical
                && (tolerance_pixels == 0
                    || r.shape_mismatch
                    || r.diff_pixels as i64 > tolerance_pixels)
        })
        .map(|r| r.key.as_str())
        .collect();
    let reordered: Vec<&str> = reports
        .iter()
        .filter(|r| r.pixels_identical && !r.identical)
        .map(|r| r.key.as_str())
        .collect();
    if !reordered.is_empty() {
        println!(
            "{} case(s) render the same pixels but are not identical (indices or palette \
             differ, so the PNG differs): {}",
            reordered.len(),
            reordered.join(", ")
        );
    }
    if failing.is_empty() && reports.iter().all(|r| r.identical) {
        println!("all identical");
    } else if failing.is_empty() {
        println!("all within tolerance ({tolerance_pixels} differing pixels per case)");
    } else {
        println!(
            "{} of {} case(s) differ beyond tolerance ({tolerance_pixels} px): {}",
            failing.len(),
            reports.len(),
            failing.join(", ")
        );
    }
    Ok((i32::from(!failing.is_empty()), reports))
}

pub fn command() -> clap::Command {
    let dir = || {
        Arg::new("dir")
            .value_parser(clap::value_parser!(PathBuf))
            .help("snapshot directory (default: bench/snapshots/)")
    };
    clap::Command::new("snapshot")
        .about("Behavior baseline: save the pipeline's outputs per case, and compare a later run against them.")
        .after_help(IDENTICAL_HELP)
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            clap::Command::new("save")
                .about("run the case matrix and store the outputs")
                .args_override_self(true)
                .arg(dir())
                .args(bench::case_arguments()),
        )
        .subcommand(
            clap::Command::new("compare")
                .about("re-run the stored cases and compare")
                .long_about(IDENTICAL_HELP)
                .arg(dir())
                .arg(
                    Arg::new("tolerance-pixels")
                        .long("tolerance-pixels")
                        .value_name("N")
                        .value_parser(clap::value_parser!(i64).range(0..))
                        .default_value("0")
                        .help("up to N differing pixels per case still exit 0 (default: 0)"),
                ),
        )
}

/// The exit code of `xtask snapshot …`.
pub fn main(matches: &ArgMatches) -> Result<i32, Fail> {
    match matches.subcommand() {
        Some(("save", m)) => {
            let directory = m
                .get_one::<PathBuf>("dir")
                .cloned()
                .unwrap_or_else(default_dir);
            let n = save(&directory, &bench::cases_from(m)?)?;
            println!("saved {n} case(s) to {}", directory.display());
            Ok(0)
        }
        Some(("compare", m)) => {
            let directory = m
                .get_one::<PathBuf>("dir")
                .cloned()
                .unwrap_or_else(default_dir);
            let tolerance = *m.get_one::<i64>("tolerance-pixels").expect("default");
            Ok(compare(&directory, tolerance)?.0)
        }
        _ => unreachable!("subcommand_required"),
    }
}
