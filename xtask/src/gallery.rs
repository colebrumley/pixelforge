//! `xtask gallery`: box / kopf / gerstner × sprite / background on every fixture
//! (`scripts/gallery.py`).
//!
//! Runs all three methods with both presets on every image in tests/fixtures plus every PNG
//! in an optional samples/ directory and writes `gallery/index.html`: one row per image and
//! preset with the input and the three results (the `*_preview.png` files, i.e. upscaled by
//! `scale_preview`), with timings and iteration counts under each cell.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Arg, ArgAction, ArgMatches};
use pixelforge::config::{Config, Value, METHODS, PRESET_NAMES};
use pixelforge::pipeline::{self, Input};

use crate::bench;
use crate::fixtures::{self, FIXTURES};
use crate::json::{as_i64, get};
use crate::{fail, resolve, root, stem, Fail};

const STYLE: &str = "
:root { color-scheme: light dark; --bg: #f4f1ea; --fg: #23201c; --muted: #6d665c;
        --card: #fffdf8; --line: #d9d2c4; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #17161a; --fg: #ebe7df; --muted: #9a94a3; --card: #211f26; --line: #37333f; } }
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
h1 { font-size: 20px; margin: 0 0 4px; }
p.lead { color: var(--muted); margin: 0 0 20px; max-width: 80ch; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; }
th, td { border: 1px solid var(--line); padding: 10px; vertical-align: top; text-align: left;
         background: var(--card); }
th { font-weight: 600; white-space: nowrap; }
td.name { white-space: nowrap; }
img { display: block; width: 256px; height: auto; image-rendering: pixelated;
      background: repeating-conic-gradient(#8883 0% 25%, transparent 0% 50%) 0 0 / 16px 16px; }
small { display: block; margin-top: 6px; color: var(--muted); }
";

/// `html.escape(s)` (quote=True).
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            c => out.push(c),
        }
    }
    out
}

/// The fixtures (drawn into tests/fixtures if any is missing) and `samples_dir/*.png`
/// sorted.
pub fn collect_inputs(samples_dir: &Path) -> Result<Vec<PathBuf>, Fail> {
    let fixtures_dir = fixtures::default_dir();
    if FIXTURES
        .iter()
        .any(|name| !fixtures_dir.join(format!("{name}.png")).is_file())
    {
        fixtures::write(&fixtures_dir)?;
    }
    let mut inputs: Vec<PathBuf> = FIXTURES
        .iter()
        .map(|name| fixtures_dir.join(format!("{name}.png")))
        .collect();
    if samples_dir.is_dir() {
        let mut samples: Vec<PathBuf> = std::fs::read_dir(samples_dir)
            .map_err(|e| fail(1, format!("{}: {e}", samples_dir.display())))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        samples.sort();
        inputs.extend(samples);
    }
    Ok(inputs)
}

pub fn command() -> clap::Command {
    clap::Command::new("gallery")
        .about("Comparison gallery: box / kopf / gerstner × sprite / background on every fixture.")
        .arg(
            Arg::new("out")
                .long("out")
                .value_parser(clap::value_parser!(PathBuf))
                .help("output directory (default: gallery/)"),
        )
        .arg(
            Arg::new("samples")
                .long("samples")
                .value_parser(clap::value_parser!(PathBuf))
                .help("directory of extra PNG inputs (default: samples/)"),
        )
        .arg(
            Arg::new("quick")
                .long("quick")
                .action(ArgAction::SetTrue)
                .help("run the background preset at 64 px instead of 256 px"),
        )
}

pub fn main(matches: &ArgMatches) -> Result<PathBuf, Fail> {
    let out_dir = matches
        .get_one::<PathBuf>("out")
        .cloned()
        .unwrap_or_else(|| root().join("gallery"));
    let samples = matches
        .get_one::<PathBuf>("samples")
        .cloned()
        .unwrap_or_else(|| root().join("samples"));
    run(
        &out_dir,
        &collect_inputs(&samples)?,
        matches.get_flag("quick"),
    )
}

type Row = (String, String, String, Vec<(String, String)>);

/// Write the gallery of `inputs` into `out_dir`; returns the path of `index.html`.
pub fn run(out_dir: &Path, inputs: &[PathBuf], quick: bool) -> Result<PathBuf, Fail> {
    let image_dir = out_dir.join("img");
    std::fs::create_dir_all(&image_dir)
        .map_err(|e| fail(1, format!("{}: {e}", image_dir.display())))?;

    // Files are named by `bench::input_labels` (the stem, with a hash suffix when two
    // inputs share it) so same-stem inputs do not overwrite each other; captions keep the
    // stem.
    let labels = bench::input_labels(inputs.iter().map(PathBuf::as_path));
    let mut rows: Vec<Row> = Vec::new();
    for path in inputs {
        let path = path.as_path();
        let name = stem(path);
        let resolved = resolve(path);
        let label = labels
            .iter()
            .find(|(p, _)| *p == resolved)
            .map_or_else(|| name.clone(), |(_, l)| l.clone());
        let input_name = format!("{label}_input.png");
        let input_copy = image_dir.join(&input_name);
        std::fs::copy(path, &input_copy)
            .map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
        let is_fixture = path
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|n| n == "fixtures");
        for preset in PRESET_NAMES {
            let mut cells = Vec::new();
            for method in METHODS {
                let mut kwargs = vec![
                    ("preset".to_string(), Value::str(*preset)),
                    ("method".to_string(), Value::str(*method)),
                ];
                if quick && *preset == "background" {
                    kwargs.push(("out_height".into(), Value::Int(64)));
                    kwargs.push(("out_width".into(), Value::Int(64)));
                }
                if is_fixture {
                    // The fixtures are test patterns, not sprites: keep their backgrounds.
                    kwargs.push(("key_bg".into(), Value::Bool(false)));
                }
                let config =
                    Config::new(&kwargs).map_err(|e| fail(2, format!("config error: {}", e.0)))?;
                let start = Instant::now();
                let result = pipeline::run(Input::Path(path), &config, None)
                    .map_err(|e| fail(1, format!("{}: {e}", path.display())))?;
                let seconds = start.elapsed().as_secs_f64();
                let outputs = result
                    .save(&image_dir.join(format!("{label}_{preset}_{method}")))
                    .map_err(|e| fail(1, e.to_string()))?;
                let preview = outputs
                    .iter()
                    .find(|(k, _)| k == "preview")
                    .and_then(|(_, p)| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let stat = |k: &str| get(&result.stats, k).and_then(as_i64).unwrap_or(0);
                let caption = format!(
                    "{}×{} · {} colors · {seconds:.2} s · {} iterations",
                    result.indices.w,
                    result.indices.h,
                    stat("colors_used"),
                    stat("iterations")
                );
                println!("{label:>16} {preset:>10} {method:>8}  {caption}");
                cells.push((preview, caption));
            }
            rows.push((name.clone(), preset.to_string(), input_name.clone(), cells));
        }
    }

    let index = out_dir.join("index.html");
    std::fs::write(&index, render(&rows))
        .map_err(|e| fail(1, format!("{}: {e}", index.display())))?;
    println!("wrote {}", index.display());
    Ok(index)
}

fn render(rows: &[Row]) -> String {
    let mut parts: Vec<String> = vec![
        "<!doctype html>".into(),
        "<html lang=\"en\"><head><meta charset=\"utf-8\">".into(),
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">".into(),
        "<title>pixelforge gallery</title>".into(),
        format!("<style>{STYLE}</style></head><body>"),
        "<h1>pixelforge gallery</h1>".into(),
        "<p class=\"lead\">Every fixture converted with the three downscalers under both \
         presets. Results are the <code>*_preview.png</code> files (nearest-neighbor upscale \
         by <code>scale_preview</code>), displayed at a fixed width; open an image for full \
         size.</p>"
            .into(),
        "<div class=\"scroll\"><table><thead><tr><th>image · preset</th><th>input</th>".into(),
    ];
    parts.extend(METHODS.iter().map(|m| format!("<th>{}</th>", escape(m))));
    parts.push("</tr></thead><tbody>".into());
    for (name, preset, input_name, cells) in rows {
        parts.push(format!(
            "<tr><td class=\"name\">{}<small>{preset}</small></td>",
            escape(name)
        ));
        parts.push(format!(
            "<td><a href=\"img/{0}\"><img src=\"img/{0}\" alt=\"{1} input\"></a></td>",
            escape(input_name),
            escape(name)
        ));
        for ((preview, caption), method) in cells.iter().zip(METHODS) {
            parts.push(format!(
                "<td><a href=\"img/{0}\"><img src=\"img/{0}\" loading=\"lazy\" \
                 alt=\"{1} {preset} {method}\"></a><small>{2}</small></td>",
                escape(preview),
                escape(name),
                escape(caption)
            ));
        }
        parts.push("</tr>".into());
    }
    parts.push("</tbody></table></div></body></html>".into());
    parts.join("\n") + "\n"
}
