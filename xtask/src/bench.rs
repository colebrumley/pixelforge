//! `xtask bench`: run the pipeline over a case matrix and record per-stage timings
//! (`scripts/bench.py`).
//!
//! Each case is (input, preset, method), run in-process with `pipeline::run` and the
//! preset's defaults. Prints a table and writes `bench/<short git sha>.json` with the
//! Python script's schema: `python`, `numpy` and `scipy` are `null`, and `rustc` and
//! `pixelforge` (the crate version) are added, so `--compare` works across the port.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::{Arg, ArgAction, ArgMatches};
use pixelforge::config::{Config, Value, METHODS, PRESET_NAMES};
use pixelforge::io::ArrayData;
use pixelforge::pipeline::{self, Input};
use pixelforge::pyjson::{dumps, py_round, Json, Layout};
use pixelforge::pyrepr::str_repr;
use sha2::{Digest, Sha256};

use crate::json::{as_bool, as_f64, as_i64, as_list, as_str, get};
use crate::{display_path, fail, git, resolve, root, stem, Fail};

pub const DEFAULT_INPUTS: [&str; 4] = ["line_diag", "circle_alpha", "noisy_gradient", "gradient6"];
pub const THREAD_ENV: [&str; 3] = [
    "PIXELFORGE_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
];
pub const STAGES: [&str; 6] = [
    "load",
    "preprocess",
    "downscale",
    "palette_quantize",
    "postprocess",
    "tiles",
];

/// One case: the input file, the preset and the method.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub path: PathBuf,
    pub preset: String,
    pub method: String,
}

/// A fixture name (`tests/fixtures/<name>.png`) or a path to an image.
pub fn resolve_input(name: &str) -> Result<PathBuf, Fail> {
    let fixture = crate::fixtures::default_dir().join(format!("{name}.png"));
    if fixture.is_file() {
        return Ok(fixture);
    }
    let path = expand_user(name);
    if !path.is_file() {
        return Err(fail(
            1,
            format!("input not found: {name} (neither a fixture name nor a file)"),
        ));
    }
    Ok(path)
}

/// `Path.expanduser()` for a leading `~` or `~/`.
fn expand_user(name: &str) -> PathBuf {
    if name == "~" || name.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(name.trim_start_matches('~').trim_start_matches('/'));
        }
    }
    PathBuf::from(name)
}

/// Label per input, keyed by resolved path: the file stem, or, when two different files
/// share a stem, `STEM-HHHHHHHH` with the first 8 hex digits of the SHA-256 of the file's
/// display path (so labels of inputs with a unique stem never change).
pub fn input_labels<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Vec<(PathBuf, String)> {
    let mut by_stem: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for path in paths {
        let resolved = resolve(path);
        let s = stem(path);
        match by_stem.iter_mut().find(|(k, _)| *k == s) {
            Some((_, group)) => {
                if !group.contains(&resolved) {
                    group.push(resolved);
                }
            }
            None => by_stem.push((s, vec![resolved])),
        }
    }
    let mut labels = Vec::new();
    for (s, group) in &by_stem {
        for path in group {
            let label = if group.len() == 1 {
                s.clone()
            } else {
                let digest = Sha256::digest(display_path(path).as_bytes());
                let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
                format!("{s}-{}", &hex[..8])
            };
            labels.push((path.clone(), label));
        }
    }
    labels
}

fn check_choices(presets: &[String], methods: &[String]) -> Result<(), Fail> {
    for preset in presets {
        if !PRESET_NAMES.contains(&preset.as_str()) {
            return Err(fail(
                1,
                format!(
                    "unknown preset {}; choose from {}",
                    str_repr(preset),
                    PRESET_NAMES.join(", ")
                ),
            ));
        }
    }
    for method in methods {
        if !METHODS.contains(&method.as_str()) {
            return Err(fail(
                1,
                format!(
                    "unknown method {}; choose from {}",
                    str_repr(method),
                    METHODS.join(", ")
                ),
            ));
        }
    }
    Ok(())
}

/// `[(input, preset, method)]` in a fixed order: `--inputs` × presets, then `--case`; every
/// pair with every method.
pub fn build_cases(
    inputs: &[String],
    presets: &[String],
    methods: &[String],
    extra_cases: &[String],
) -> Result<Vec<Case>, Fail> {
    check_choices(presets, methods)?;
    let mut pairs = Vec::new();
    for name in inputs {
        for preset in presets {
            pairs.push((resolve_input(name)?, preset.clone()));
        }
    }
    for spec in extra_cases {
        let (name, preset) = match spec.rfind(':') {
            Some(i) if PRESET_NAMES.contains(&&spec[i + 1..]) => (&spec[..i], &spec[i + 1..]),
            _ => {
                return Err(fail(
                    1,
                    format!(
                        "--case expects INPUT:PRESET with a preset in {}, got {}",
                        PRESET_NAMES.join(", "),
                        str_repr(spec)
                    ),
                ))
            }
        };
        pairs.push((resolve_input(name)?, preset.to_string()));
    }
    let mut cases = Vec::new();
    for (path, preset) in pairs {
        for method in methods {
            cases.push(Case {
                path: path.clone(),
                preset: preset.clone(),
                method: method.clone(),
            });
        }
    }
    Ok(cases)
}

pub fn case_key(label: &str, preset: &str, method: &str) -> String {
    format!("{label}/{preset}/{method}")
}

/// One key per case, built over the whole list so that inputs sharing a file stem get
/// distinct labels (see [`input_labels`]).
pub fn case_keys(cases: &[Case]) -> Vec<String> {
    let labels = input_labels(cases.iter().map(|c| c.path.as_path()));
    cases
        .iter()
        .map(|c| {
            let resolved = resolve(&c.path);
            let label = labels
                .iter()
                .find(|(p, _)| *p == resolved)
                .map(|(_, l)| l.as_str())
                .unwrap_or_default();
            case_key(label, &c.preset, &c.method)
        })
        .collect()
}

/// The preset's defaults with `method`.
pub fn case_config(preset: &str, method: &str) -> Result<Config, Fail> {
    Config::new(&[
        ("preset".into(), Value::str(preset)),
        ("method".into(), Value::str(method)),
    ])
    .map_err(|e| fail(2, format!("config error: {}", e.0)))
}

// ---------------------------------------------------------------------------------------
// Arguments

fn default_list(items: &[&str]) -> String {
    let quoted: Vec<String> = items.iter().map(|s| str_repr(s)).collect();
    format!("[{}]", quoted.join(", "))
}

/// `--inputs`, `--methods`, `--presets`, `--case` (bench and `snapshot save`).
pub fn case_arguments() -> Vec<Arg> {
    vec![
        Arg::new("inputs")
            .long("inputs")
            .num_args(1..)
            .value_name("NAME_OR_PATH")
            .help(format!(
                "fixture names or image paths (default: {})",
                default_list(&DEFAULT_INPUTS)
            )),
        Arg::new("methods")
            .long("methods")
            .num_args(1..)
            .help(format!(
                "downscale methods (default: {})",
                default_list(METHODS)
            )),
        Arg::new("presets")
            .long("presets")
            .num_args(1..)
            .help(format!(
                "presets applied to every --inputs entry (default: {})",
                default_list(PRESET_NAMES)
            )),
        Arg::new("case")
            .long("case")
            .action(ArgAction::Append)
            .value_name("INPUT:PRESET")
            .help("add one input with one preset (all methods); repeatable"),
    ]
}

fn strings(matches: &ArgMatches, id: &str, default: &[&str]) -> Vec<String> {
    match matches.get_many::<String>(id) {
        Some(values) => values.cloned().collect(),
        None => default.iter().map(|s| s.to_string()).collect(),
    }
}

/// The methods `--methods` names (default: all).
pub fn methods_from(matches: &ArgMatches) -> Vec<String> {
    strings(matches, "methods", METHODS)
}

pub fn cases_from(matches: &ArgMatches) -> Result<Vec<Case>, Fail> {
    build_cases(
        &strings(matches, "inputs", &DEFAULT_INPUTS),
        &strings(matches, "presets", PRESET_NAMES),
        &methods_from(matches),
        &strings(matches, "case", &[]),
    )
}

// ---------------------------------------------------------------------------------------
// Environment

/// The measured code: the library crate.
pub const PACKAGE_DIR: &str = "crates/pixelforge";

/// `git_sha`, `git_dirty`, `package_tree`, `package_dirty`: HEAD, whether tracked files are
/// modified, and the same for the library crate alone (its tree hash identifies the
/// measured code even when only the tooling changed).
pub fn git_info() -> Vec<(String, Json)> {
    let sha = git(&["rev-parse", "HEAD"]);
    let status = git(&["status", "--porcelain", "--untracked-files=no"]);
    let package_status = git(&["status", "--porcelain", "--", PACKAGE_DIR]);
    let tree = git(&["rev-parse", &format!("HEAD:{PACKAGE_DIR}")]);
    vec![
        (
            "git_sha".into(),
            Json::Str(if sha.is_empty() {
                "unknown".into()
            } else {
                sha
            }),
        ),
        ("git_dirty".into(), Json::Bool(!status.is_empty())),
        (
            "package_tree".into(),
            Json::Str(if tree.is_empty() {
                "unknown".into()
            } else {
                tree
            }),
        ),
        (
            "package_dirty".into(),
            Json::Bool(!package_status.is_empty()),
        ),
    ]
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// `platform.machine()`: arm64 on Apple silicon, the architecture name elsewhere.
pub fn machine() -> String {
    let platform = pipeline::platform();
    platform
        .split_once(' ')
        .map_or(platform.clone(), |(_, m)| m.to_string())
}

/// An approximation of `platform.platform()`: `macOS-<version>-<machine>-<processor>-64bit`
/// on macOS (as Python writes it), `<uname -s>-<uname -r>-<machine>` elsewhere.
pub fn platform_string() -> String {
    let m = machine();
    if cfg!(target_os = "macos") {
        let version = command_output("sw_vers", &["-productVersion"]).unwrap_or_default();
        let processor = command_output("uname", &["-p"]).unwrap_or_default();
        return format!("macOS-{version}-{m}-{processor}-64bit");
    }
    match (
        command_output("uname", &["-s"]),
        command_output("uname", &["-r"]),
    ) {
        (Some(s), Some(r)) => format!("{s}-{r}-{m}"),
        _ => pipeline::platform(),
    }
}

pub fn environment() -> Vec<(String, Json)> {
    let env = THREAD_ENV
        .iter()
        .filter_map(|name| {
            std::env::var(name)
                .ok()
                .map(|v| (name.to_string(), Json::Str(v)))
        })
        .collect();
    vec![
        ("machine".into(), Json::Str(machine())),
        ("platform".into(), Json::Str(platform_string())),
        ("python".into(), Json::Null),
        ("numpy".into(), Json::Null),
        ("scipy".into(), Json::Null),
        ("rustc".into(), Json::str(env!("XTASK_RUSTC_VERSION"))),
        ("pixelforge".into(), Json::str(pipeline::VERSION)),
        ("env".into(), Json::Object(env)),
    ]
}

/// `datetime.now(UTC).isoformat(timespec="seconds")`, e.g. `2026-10-11T03:25:27+00:00`.
pub fn utc_timestamp(now: SystemTime) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

// ---------------------------------------------------------------------------------------
// Running

/// One untimed run per preset and method on a tiny in-memory image, so one-time costs
/// (page faults, thread start-up) are not charged to the first measured case.
pub fn warm_up(methods: &[String]) -> Result<(), Fail> {
    let mut rgb = Vec::with_capacity(48 * 48 * 3);
    for y in 0..48u32 {
        for x in 0..48u32 {
            rgb.extend_from_slice(&[(x * 5) as u8, (y * 5) as u8, ((x + y) * 2) as u8]);
        }
    }
    for preset in PRESET_NAMES {
        for method in methods {
            let config = Config::new(&[
                ("preset".into(), Value::str(*preset)),
                ("method".into(), Value::str(method.as_str())),
                ("out_width".into(), Value::Int(16)),
                ("out_height".into(), Value::Int(16)),
            ])
            .map_err(|e| fail(2, format!("config error: {}", e.0)))?;
            let input = Input::Array {
                shape: &[48, 48, 3],
                data: ArrayData::U8(&rgb),
            };
            pipeline::run(input, &config, None)
                .map_err(|e| fail(1, format!("warm-up {preset}/{method}: {e}")))?;
        }
    }
    Ok(())
}

/// A measured case, as the JSON has it.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseResult {
    /// Repository-relative for an input in the checkout, the bare file name otherwise
    /// (older files may hold an absolute path).
    pub input: String,
    /// The input is outside the repository (`"external": true`, written only when set).
    pub external: bool,
    pub label: String,
    pub preset: String,
    pub method: String,
    pub config_hash: String,
    pub total_s: f64,
    pub timings: Vec<(String, Json)>,
    pub iterations: i64,
    pub method_stats: Vec<(String, Json)>,
    pub output_size: [i64; 2],
    pub palette_size: i64,
    pub repeat: i64,
}

impl CaseResult {
    pub fn timing(&self, stage: &str) -> f64 {
        self.timings
            .iter()
            .find(|(k, _)| k == stage)
            .and_then(|(_, v)| as_f64(v))
            .unwrap_or(0.0)
    }

    pub fn to_json(&self) -> Json {
        let mut items = vec![("input".into(), Json::Str(self.input.clone()))];
        if self.external {
            items.push(("external".into(), Json::Bool(true)));
        }
        items.extend([
            ("label".into(), Json::Str(self.label.clone())),
            ("preset".into(), Json::Str(self.preset.clone())),
            ("method".into(), Json::Str(self.method.clone())),
            ("config_hash".into(), Json::Str(self.config_hash.clone())),
            ("total_s".into(), Json::Float(self.total_s)),
            ("timings".into(), Json::Object(self.timings.clone())),
            ("iterations".into(), Json::Int(self.iterations)),
            (
                "method_stats".into(),
                Json::Object(self.method_stats.clone()),
            ),
            (
                "output_size".into(),
                Json::List(self.output_size.iter().map(|&v| Json::Int(v)).collect()),
            ),
            ("palette_size".into(), Json::Int(self.palette_size)),
            ("repeat".into(), Json::Int(self.repeat)),
        ]);
        Json::Object(items)
    }
}

/// The `input` and `external` fields for an input file: the repository-relative path for a
/// file in the checkout, the bare file name (and `true`) for one outside it, so the
/// committed reports carry no local paths.
pub fn recorded_input(path: &Path) -> (String, bool) {
    let path = resolve(path);
    match path.strip_prefix(root()) {
        Ok(rel) => (rel.display().to_string(), false),
        Err(_) => (
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            true,
        ),
    }
}

/// Run one case `repeat` times (at least once) and keep the fastest run.
pub fn run_case(case: &Case, repeat: i64, label: &str) -> Result<CaseResult, Fail> {
    let mut best: Option<(f64, pipeline::Output)> = None;
    let mut config = None;
    for _ in 0..repeat.max(1) {
        let c = case_config(&case.preset, &case.method)?;
        let start = Instant::now();
        let out = pipeline::run(Input::Path(&case.path), &c, None)
            .map_err(|e| fail(1, format!("{}: {e}", case.path.display())))?;
        let total = start.elapsed().as_secs_f64();
        if best.as_ref().is_none_or(|(t, _)| total < *t) {
            best = Some((total, out));
        }
        config = Some(c);
    }
    let (total, out) = best.expect("at least one run");
    let config = config.expect("at least one run");
    let stats = &out.stats;
    let timings = get(stats, "timings")
        .map(|t| crate::json::entries(t).to_vec())
        .unwrap_or_default();
    let method_stats = get(stats, "method")
        .map(|m| {
            crate::json::entries(m)
                .iter()
                .filter(|(_, v)| {
                    matches!(
                        v,
                        Json::Bool(_) | Json::Int(_) | Json::Float(_) | Json::Str(_)
                    )
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let (input, external) = recorded_input(&case.path);
    Ok(CaseResult {
        input,
        external,
        label: label.to_string(),
        preset: case.preset.clone(),
        method: case.method.clone(),
        config_hash: config.hash().map_err(|e| fail(1, e.to_string()))?,
        total_s: py_round(total, 4),
        timings,
        iterations: get(stats, "iterations").and_then(as_i64).unwrap_or(0),
        method_stats,
        output_size: [out.indices.w as i64, out.indices.h as i64],
        palette_size: out.palette.len() as i64,
        repeat: repeat.max(1),
    })
}

// ---------------------------------------------------------------------------------------
// Tables

/// The scripts' table: the first three columns left-aligned, the rest right-aligned, two
/// spaces between columns, a dashed rule under the header. Widths count characters.
pub fn format_table(rows: &[Vec<String>], header: &[&str]) -> String {
    let header: Vec<String> = header.iter().map(|s| s.to_string()).collect();
    let all: Vec<&Vec<String>> = std::iter::once(&header).chain(rows.iter()).collect();
    let widths: Vec<usize> = (0..header.len())
        .map(|i| all.iter().map(|r| r[i].chars().count()).max().unwrap_or(0))
        .collect();
    let mut lines: Vec<String> = all
        .iter()
        .map(|row| {
            row.iter()
                .zip(&widths)
                .enumerate()
                .map(|(i, (c, &w))| {
                    if i < 3 {
                        format!("{c:<w$}")
                    } else {
                        format!("{c:>w$}")
                    }
                })
                .collect::<Vec<_>>()
                .join("  ")
        })
        .collect();
    let rule = widths
        .iter()
        .map(|&w| "-".repeat(w))
        .collect::<Vec<_>>()
        .join("  ");
    lines.insert(1, rule);
    lines.join("\n")
}

pub fn results_table(cases: &[CaseResult]) -> String {
    let rows: Vec<Vec<String>> = cases
        .iter()
        .map(|c| {
            vec![
                c.label.clone(),
                c.preset.clone(),
                c.method.clone(),
                format!("{:.3}", c.total_s),
                format!("{:.3}", c.timing("downscale")),
                c.iterations.to_string(),
                format!("{}x{}", c.output_size[0], c.output_size[1]),
            ]
        })
        .collect();
    format_table(
        &rows,
        &[
            "input",
            "preset",
            "method",
            "total s",
            "downscale s",
            "iters",
            "size",
        ],
    )
}

fn ratio(old: f64, new: f64) -> String {
    if new > 0.0 {
        format!("{:.2}x", old / new)
    } else {
        "-".into()
    }
}

/// Old vs new total and downscale seconds per case; `old` is a bench JSON (either tool's).
pub fn compare_table(old: &Json, new: &[CaseResult]) -> String {
    let field = |c: &Json, k: &str| get(c, k).and_then(as_str).unwrap_or_default().to_string();
    let old_cases: Vec<(String, &Json)> = get(old, "cases")
        .and_then(as_list)
        .unwrap_or_default()
        .iter()
        .map(|c| {
            (
                case_key(&field(c, "label"), &field(c, "preset"), &field(c, "method")),
                c,
            )
        })
        .collect();
    let mut rows = Vec::new();
    for c in new {
        let key = case_key(&c.label, &c.preset, &c.method);
        // A repeated key: the last one wins (a dict comprehension).
        let o = old_cases
            .iter()
            .rev()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v);
        let nd = c.timing("downscale");
        let Some(o) = o else {
            rows.push(vec![
                c.label.clone(),
                c.preset.clone(),
                c.method.clone(),
                "-".into(),
                format!("{:.3}", c.total_s),
                "-".into(),
                "-".into(),
                format!("{nd:.3}"),
                "-".into(),
                String::new(),
            ]);
            continue;
        };
        let ot = get(o, "total_s").and_then(as_f64).unwrap_or(0.0);
        let nt = c.total_s;
        let od = get(o, "timings")
            .and_then(|t| get(t, "downscale"))
            .and_then(as_f64)
            .unwrap_or(0.0);
        let same_config = get(o, "config_hash").and_then(as_str) == Some(c.config_hash.as_str());
        rows.push(vec![
            c.label.clone(),
            c.preset.clone(),
            c.method.clone(),
            format!("{ot:.3}"),
            format!("{nt:.3}"),
            ratio(ot, nt),
            format!("{od:.3}"),
            format!("{nd:.3}"),
            ratio(od, nd),
            if same_config {
                String::new()
            } else {
                "config differs".into()
            },
        ]);
    }
    format_table(
        &rows,
        &[
            "input",
            "preset",
            "method",
            "old total",
            "new total",
            "speedup",
            "old down",
            "new down",
            "speedup",
            "note",
        ],
    )
}

// ---------------------------------------------------------------------------------------
// The command

pub fn command() -> clap::Command {
    clap::Command::new("bench")
        .about("Performance baseline: run the pipeline over a case matrix and record per-stage timings.")
        .long_about(
            "Performance baseline: run the pipeline over a case matrix and record per-stage \
             timings.\n\nInputs are fixture names (tests/fixtures/<name>.png) or paths. Every \
             input in --inputs runs with every preset in --presets; each --case INPUT:PRESET \
             adds one input with one preset. All inputs run every method in --methods. \
             --repeat N keeps the minimum total time of N runs (and the per-stage timings of \
             that run). Before measuring, every preset/method runs once untimed on a tiny \
             image (--no-warmup to skip).",
        )
        .args(case_arguments())
        .arg(
            Arg::new("repeat")
                .long("repeat")
                .value_parser(clap::value_parser!(i64))
                .allow_negative_numbers(true)
                .default_value("1")
                .help("runs per case; the minimum total is reported (default: 1)"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .value_parser(clap::value_parser!(PathBuf))
                .help("JSON output path (default: bench/<short git sha>.json)"),
        )
        .arg(
            Arg::new("no-write")
                .long("no-write")
                .action(ArgAction::SetTrue)
                .help("do not write the JSON file"),
        )
        .arg(
            Arg::new("no-warmup")
                .long("no-warmup")
                .action(ArgAction::SetTrue)
                .help("skip the untimed warm-up run per preset and method"),
        )
        .arg(
            Arg::new("compare")
                .long("compare")
                .value_name("OLD.json")
                .value_parser(clap::value_parser!(PathBuf))
                .help("also print old vs new total and downscale seconds per case"),
        )
}

/// Read a bench JSON (written by `scripts/bench.py` or `xtask bench`).
pub fn read_report(path: &Path) -> Result<Json, Fail> {
    let text =
        std::fs::read_to_string(path).map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
    crate::json::loads(&text).map_err(|e| fail(1, format!("{}: {e}", path.display())))
}

/// The report: the header keys, then `cases`.
pub fn report_json(header: &[(String, Json)], cases: &[CaseResult]) -> Json {
    let mut items = header.to_vec();
    items.push((
        "cases".into(),
        Json::List(cases.iter().map(CaseResult::to_json).collect()),
    ));
    Json::Object(items)
}

/// Parse a bench JSON's `cases` back into [`CaseResult`]s (missing keys get defaults).
pub fn cases_of(report: &Json) -> Vec<CaseResult> {
    let s = |c: &Json, k: &str| get(c, k).and_then(as_str).unwrap_or_default().to_string();
    let i = |c: &Json, k: &str| get(c, k).and_then(as_i64).unwrap_or(0);
    get(report, "cases")
        .and_then(as_list)
        .unwrap_or_default()
        .iter()
        .map(|c| {
            let size = get(c, "output_size").and_then(as_list).unwrap_or_default();
            let dim = |n: usize| size.get(n).and_then(as_i64).unwrap_or(0);
            CaseResult {
                input: s(c, "input"),
                external: get(c, "external").and_then(as_bool).unwrap_or(false),
                label: s(c, "label"),
                preset: s(c, "preset"),
                method: s(c, "method"),
                config_hash: s(c, "config_hash"),
                total_s: get(c, "total_s").and_then(as_f64).unwrap_or(0.0),
                timings: get(c, "timings")
                    .map(|t| crate::json::entries(t).to_vec())
                    .unwrap_or_default(),
                iterations: i(c, "iterations"),
                method_stats: get(c, "method_stats")
                    .map(|t| crate::json::entries(t).to_vec())
                    .unwrap_or_default(),
                output_size: [dim(0), dim(1)],
                palette_size: i(c, "palette_size"),
                repeat: i(c, "repeat"),
            }
        })
        .collect()
}

pub fn main(matches: &ArgMatches) -> Result<(), Fail> {
    let cases = cases_from(matches)?;
    let methods = methods_from(matches);
    let repeat = *matches.get_one::<i64>("repeat").expect("default");
    let no_warmup = matches.get_flag("no-warmup");
    let compare = matches.get_one::<PathBuf>("compare");
    let old = compare.map(|p| read_report(p)).transpose()?;

    let mut header = vec![(
        "timestamp".to_string(),
        Json::Str(utc_timestamp(SystemTime::now())),
    )];
    header.extend(git_info());
    header.extend(environment());
    header.push(("repeat".into(), Json::Int(repeat.max(1))));
    header.push(("warmup".into(), Json::Bool(!no_warmup)));

    if !no_warmup {
        warm_up(&methods)?;
    }
    let mut results = Vec::new();
    for (case, key) in cases.iter().zip(case_keys(&cases)) {
        eprintln!("running {key} ...");
        let label = key.split('/').next().unwrap_or_default();
        results.push(run_case(case, repeat, label)?);
    }

    println!("{}", results_table(&results));
    let report = report_json(&header, &results);
    let sha = get(&report, "git_sha").and_then(as_str).unwrap_or("?");
    let short = |s: &str| s.chars().take(7).collect::<String>();
    if let (Some(old), Some(path)) = (&old, compare) {
        let dirty = get(&report, "git_dirty").and_then(crate::json::as_bool) == Some(true);
        println!();
        println!(
            "compare: {} ({}) -> {}{}",
            path.display(),
            short(get(old, "git_sha").and_then(as_str).unwrap_or("?")),
            short(sha),
            if dirty { " (dirty)" } else { "" }
        );
        println!("{}", compare_table(old, &results));
    }
    if !matches.get_flag("no-write") {
        let out = matches
            .get_one::<PathBuf>("out")
            .cloned()
            .unwrap_or_else(|| root().join("bench").join(format!("{}.json", short(sha))));
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| fail(1, format!("{}: {e}", parent.display())))?;
        }
        std::fs::write(&out, dumps(&report, Layout::INDENT2) + "\n")
            .map_err(|e| fail(1, format!("{}: {e}", out.display())))?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}
