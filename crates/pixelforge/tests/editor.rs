//! The editor core end to end (docs/editor.md, Section 3.14): opening pixelforge outputs and
//! plain PNGs, the save round trip, the tileset path, and the replay golden: the recorded
//! logs in tests/edits/ replayed on their conversions must give the PNGs whose SHA-256
//! tests/edits/replay.json records.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::*;
use pixelforge::config::Config;
use pixelforge::editor::adjust::{self, Adjust};
use pixelforge::editor::{palette, save, tools};
use pixelforge::editor::{Document, EditLog, History, Op, PixelChange, Selection, Session};
use pixelforge::io;
use pixelforge::pyjson::Json;
use sha2::{Digest, Sha256};

/// The golden.rs config: 32 × 32 output without denoising or background keying.
fn golden_config(method: &str, preset: &str) -> Config {
    config(&[
        ("preset", s(preset)),
        ("method", s(method)),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("denoise", s("none")),
        ("key_bg", b(false)),
    ])
}

/// `fixture/method/preset` converted.
fn conversion(key: &str) -> Document {
    let parts: Vec<&str> = key.split('/').collect();
    let [fixture, method, preset] = parts[..] else {
        panic!("bad key {key}")
    };
    let out = run_path(&fixture_path(fixture), &golden_config(method, preset)).unwrap();
    Document::from_output(&out).unwrap()
}

/// A scripted session touching every op kind and most tools; works on any palette.
fn scripted_edits(doc: &mut Document) -> History {
    let mut h = History::new();
    let (w, hh) = (doc.width() as i64, doc.height() as i64);
    let k = doc.palette.len() as i64;
    let push = |h: &mut History, doc: &mut Document, op: Op| {
        h.push(doc, op).unwrap();
    };
    let mut stroke = tools::Stroke::new("pencil");
    stroke.extend(tools::pencil(doc, None, &[(1, 1), (w / 2, hh / 3)], 2, 0));
    stroke.extend(tools::pencil(
        doc,
        None,
        &[(w / 2, hh / 3), (w - 2, 2)],
        2,
        0,
    ));
    push(&mut h, doc, stroke.into_op());
    let fill = tools::flood_fill(doc, None, (0, hh - 1), k - 1);
    push(&mut h, doc, Op::pixels("fill", fill));
    let rect = tools::rectangle(doc, None, (3, 3), (w - 4, hh / 2), false, 1 % k);
    push(&mut h, doc, Op::pixels("rectangle", rect));
    let op = palette::set_color(doc, 0, [0x4a, 0x3b, 0x2c]).unwrap();
    push(&mut h, doc, op);
    let op = palette::add_color(doc, [0x12, 0x34, 0x56]).unwrap();
    push(&mut h, doc, op);
    let new = doc.palette.len() as i64 - 1;
    let line = tools::line(doc, None, (0, hh - 2), (w - 1, hh / 2), 1, new);
    push(&mut h, doc, Op::pixels("line", line));
    let sel = Selection::from_corners((2, 2), (w - 3, hh - 3), doc.width(), doc.height());
    let params = Adjust {
        brightness: 6.0,
        contrast: 1.5,
        saturation: 1.2,
        only_selection_colors: false,
    };
    let changes = adjust::adjust(doc, sel.as_ref(), &params);
    push(&mut h, doc, Op::pixels("adjust", changes));
    let changes = adjust::ramp_shift(doc, sel.as_ref(), true);
    push(&mut h, doc, Op::pixels("ramp_shift", changes));
    let op = palette::remove_entry(doc, 1).unwrap();
    push(&mut h, doc, op);
    let erase = tools::eraser(doc, None, &[(w - 1, 0), (w - 6, 5)], 3);
    push(&mut h, doc, Op::pixels("eraser", erase));
    // An undone op is not in the log.
    let extra = tools::fill_global(doc, None, (w / 2, hh / 2), -1);
    push(&mut h, doc, Op::pixels("fill", extra));
    assert!(h.undo(doc));
    h
}

/// `stats[key]`.
fn stat<'a>(stats: &'a Json, key: &str) -> Option<&'a Json> {
    match stats {
        Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn sha256_file(path: &std::path::Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn edits_dir() -> PathBuf {
    repo_root().join("tests/edits")
}

fn replay_golden() -> BTreeMap<String, serde_json::Value> {
    let text = std::fs::read_to_string(edits_dir().join("replay.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// The replayed PNG's hash for `key` with the log `log_name`.
fn replayed_png_sha(key: &str, log: &EditLog) -> String {
    let mut doc = conversion(key);
    log.replay(&mut doc).unwrap();
    let tmp = TempDir::new("replay");
    let prefix = tmp.join("out");
    let saved = save::save(&doc, log, &prefix).unwrap();
    assert_eq!(saved.outputs[0].0, "image");
    sha256_file(&saved.outputs[0].1)
}

#[test]
fn replay_golden_matches() {
    let golden = replay_golden();
    assert_eq!(golden.len(), 2);
    for (key, entry) in &golden {
        let log_path = edits_dir().join(entry["log"].as_str().unwrap());
        let log = EditLog::read(&log_path).unwrap();
        // The committed file is exactly what the log writes.
        assert_eq!(std::fs::read_to_string(&log_path).unwrap(), log.to_text());
        assert_eq!(
            replayed_png_sha(key, &log),
            entry["png_sha256"].as_str().unwrap(),
            "{key}: the replayed PNG changed. If intended, rerun \
             `PIXELFORGE_RECORD_EDITS=1 cargo test -p pixelforge --test editor -- \
             --ignored record_replay_golden` and commit tests/edits/."
        );
    }
}

/// Rewrites tests/edits/ from [`scripted_edits`] (only with `PIXELFORGE_RECORD_EDITS=1`).
#[test]
#[ignore] // records; see replay_golden_matches
fn record_replay_golden() {
    if std::env::var_os("PIXELFORGE_RECORD_EDITS").is_none() {
        return;
    }
    let mut golden = serde_json::Map::new();
    for key in ["two_color/box/sprite", "circle_alpha/box/background"] {
        let mut doc = conversion(key);
        let history = scripted_edits(&mut doc);
        let log = history.edit_log(&doc);
        let name = format!("{}.json", key.replace('/', "_"));
        log.write(&edits_dir().join(&name)).unwrap();
        let mut entry = serde_json::Map::new();
        entry.insert("log".into(), name.into());
        entry.insert("png_sha256".into(), replayed_png_sha(key, &log).into());
        golden.insert(key.to_string(), entry.into());
    }
    let text = serde_json::to_string_pretty(&golden).unwrap() + "\n";
    std::fs::write(edits_dir().join("replay.json"), text).unwrap();
}

#[test]
fn replay_reproduces_the_session() {
    for key in ["two_color/box/sprite", "circle_alpha/box/background"] {
        let mut edited = conversion(key);
        let history = scripted_edits(&mut edited);
        let log = history.edit_log(&edited);
        assert!(log.ops.len() >= 8, "{key}: {} ops", log.ops.len());
        let mut replayed = conversion(key);
        EditLog::parse(&log.to_text())
            .unwrap()
            .replay(&mut replayed)
            .unwrap();
        assert_eq!(replayed.indices, edited.indices, "{key}");
        assert_eq!(replayed.palette, edited.palette, "{key}");
        // Undoing everything restores the conversion.
        let mut h = history;
        while h.undo(&mut edited) {}
        let fresh = conversion(key);
        assert_eq!(edited.indices, fresh.indices);
        assert_eq!(edited.palette, fresh.palette);
        assert_eq!(edited.palette_lab, fresh.palette_lab);
    }
}

#[test]
fn save_round_trip() {
    for key in ["two_color/box/sprite", "circle_alpha/box/background"] {
        let mut doc = conversion(key);
        let history = scripted_edits(&mut doc);
        let log = history.edit_log(&doc);
        let tmp = TempDir::new("roundtrip");
        let prefix = tmp.join("hero");
        let saved = save::save(&doc, &log, &prefix).unwrap();
        let names: Vec<&str> = saved.outputs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.first(), Some(&"image"));
        assert_eq!(names.last(), Some(&"edits"));
        assert_eq!(names.contains(&"tileset"), doc.origin.config.tileset);

        let opened = Document::open(&tmp.join("hero.png")).unwrap();
        assert_eq!(opened.indices, doc.indices, "{key}");
        assert_eq!(opened.palette, doc.palette);
        assert_eq!(
            opened.origin.config.canonical_json(),
            doc.origin.config.canonical_json()
        );
        assert_eq!(opened.origin.config.hash(), doc.origin.config.hash());
        assert_eq!(opened.origin.input_sha256, doc.origin.input_sha256);
        // The earlier edits come along, so a further save still replays on the conversion.
        assert!(!opened.origin.prior_log_missing);
        assert_eq!(opened.origin.prior_ops, log.ops);
        assert_eq!(opened.baseline, doc.baseline);
        assert_eq!(opened.baseline_palette, doc.baseline_palette);
        assert_eq!(EditLog::of(&opened, &[]).sha256(), log.sha256());

        let text: BTreeMap<String, String> = io::read_png_text(&tmp.join("hero.png"))
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(text[io::META_EDITS_KEY], log.sha256());
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("hero_meta.json")).unwrap())
                .unwrap();
        let edited = &meta["stats"]["edited"];
        assert_eq!(edited["ops"], log.ops.len());
        assert_eq!(edited["edits_sha256"], log.sha256().as_str());
        assert_eq!(edited["palette_changed"], true);
        assert_eq!(edited["pixelforge"], env!("CARGO_PKG_VERSION"));
        assert!(edited["pixels_changed"].as_i64().unwrap() > 0);
        assert_eq!(
            edited.get("tiles_rerender_px_changed").is_some(),
            doc.origin.config.tileset
        );
        assert_eq!(meta["stats"]["final_palette_size"], doc.palette.len());
        // The config hash does not change: the config did not.
        assert_eq!(
            meta["config_hash"],
            doc.origin.config.hash().unwrap().as_str()
        );
        assert!(stat(&opened.origin.stats, "edited").is_some());
    }
}

#[test]
fn prior_log_with_out_of_range_old_index_is_treated_as_missing() {
    // A log whose hash matches the PNG but whose old values are outside the palette: the
    // reverted baseline would hold an index the diff and the save cannot look up.
    let doc = conversion("two_color/box/sprite");
    let here = doc.get(0, 0);
    let log = EditLog {
        pixelforge: "test".into(),
        width: doc.width(),
        height: doc.height(),
        palette_len: doc.palette.len(),
        ops: vec![Op::Pixels {
            tool: "pencil".into(),
            changes: vec![PixelChange {
                x: 0,
                y: 0,
                old: 999,
                new: here,
            }],
        }],
    };
    let tmp = TempDir::new("badlog");
    save::save(&doc, &log, &tmp.join("hero")).unwrap();
    let opened = Document::open(&tmp.join("hero.png")).unwrap();
    assert!(opened.origin.prior_log_missing);
    assert!(opened.origin.prior_ops.is_empty());
    assert_eq!(opened.pixels_changed(), 0);
}

#[test]
fn open_reads_conversions_in_both_transparent_layouts() {
    for layout in ["last", "first"] {
        let cfg = config(&[
            ("method", s("box")),
            ("out_width", int(24)),
            ("out_height", int(24)),
            ("denoise", s("none")),
            ("transparent_index", s(layout)),
        ]);
        let out = run_path(&fixture_path("circle_alpha"), &cfg).unwrap();
        assert!(out.indices.indices.contains(&-1));
        let tmp = TempDir::new("open");
        out.save(&tmp.join("c")).unwrap();
        let doc = Document::open(&tmp.join("c.png")).unwrap();
        assert_eq!(doc.indices, out.indices, "{layout}");
        assert_eq!(doc.palette, out.palette, "{layout}");
        assert_eq!(doc.origin.input_sha256, out.input_sha256);
        assert_eq!(
            stat(&doc.origin.stats, "colors_used"),
            stat(&out.stats, "colors_used")
        );
        assert!(doc.origin.prior_ops.is_empty() && !doc.origin.prior_log_missing);
    }
}

#[test]
fn open_plain_png_with_default_config() {
    let tmp = TempDir::new("plain");
    let path = tmp.join("plain.png");
    let (h, w) = (4, 5);
    let mut rgba = Vec::new();
    for p in 0..h * w {
        let c = (p % 3) as u8 * 100;
        let a = if p == 7 { 0 } else { 255 };
        rgba.extend_from_slice(&[c, 50, 200 - c, a]);
    }
    io::save_rgba_png(&path, &rgba, h, w).unwrap();
    let doc = Document::open(&path).unwrap();
    assert_eq!(doc.palette.len(), 3);
    assert_eq!(doc.get(2, 1), -1);
    assert_eq!(doc.rgba(), {
        let mut expected = rgba.clone();
        expected[28..32].copy_from_slice(&[0, 0, 0, 0]);
        expected
    });
    assert_eq!(
        doc.origin.config.canonical_json(),
        Config::new(&[]).unwrap().canonical_json()
    );
    assert_eq!(doc.origin.input_sha256, io::sha256_file(&path).unwrap());
    // Saved, it becomes a pixelforge output with the same pixels.
    let log = EditLog::of(&doc, &[]);
    save::save(&doc, &log, &tmp.join("out/plain")).unwrap();
    let again = Document::open(&tmp.join("out/plain.png")).unwrap();
    assert_eq!(again.indices, doc.indices);
    assert_eq!(again.palette, doc.palette);

    // More than 256 colors is not an indexed image.
    let many: Vec<u8> = (0..300u32)
        .flat_map(|i| [(i % 256) as u8, (i / 256) as u8, 0, 255])
        .collect();
    let path = tmp.join("many.png");
    io::save_rgba_png(&path, &many, 1, 300).unwrap();
    let e = Document::open(&path).unwrap_err();
    assert!(e
        .0
        .ends_with("has more than 256 colors; the editor opens indexed images only"));
}

#[test]
fn save_refuses_to_overwrite_the_input() {
    let tmp = TempDir::new("refuse");
    let input = tmp.join("hero.png");
    std::fs::copy(fixture_path("two_color"), &input).unwrap();
    let mut doc = conversion("two_color/box/sprite");
    doc.origin.input_path = Some(input.clone());
    let log = EditLog::of(&doc, &[]);
    let e = save::save(&doc, &log, &tmp.join("hero")).unwrap_err();
    assert!(e
        .to_string()
        .ends_with(&format!("would overwrite input {}", input.display())));
    assert_eq!(
        std::fs::read(&input).unwrap(),
        std::fs::read(fixture_path("two_color")).unwrap()
    );
    save::save(&doc, &log, &tmp.join("out/hero")).unwrap();
}

#[test]
fn a_missing_prior_log_is_reported() {
    let mut doc = conversion("two_color/box/sprite");
    let history = scripted_edits(&mut doc);
    let log = history.edit_log(&doc);
    let tmp = TempDir::new("prior");
    save::save(&doc, &log, &tmp.join("hero")).unwrap();
    std::fs::remove_file(tmp.join("hero_edits.json")).unwrap();
    let opened = Document::open(&tmp.join("hero.png")).unwrap();
    assert!(opened.origin.prior_log_missing);
    assert!(opened.origin.prior_ops.is_empty());
    assert_eq!(opened.baseline, opened.indices);
}

#[test]
fn tileset_rerender_is_recorded_and_reopens_as_edited() {
    // A pixel edited to a near-duplicate color: the tile merges back into its unedited
    // twin when the tileset is re-extracted, so the saved image differs from the edit.
    let cfg = config(&[
        ("preset", s("background")),
        ("method", s("box")),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("denoise", s("none")),
        ("dither", s("none")),
        ("tile_dedupe_tolerance", float(5.0)),
    ]);
    let out = run_path(&fixture_path("two_color"), &cfg).unwrap();
    let mut doc = Document::from_output(&out).unwrap();
    let (x, y) = (2, 18);
    let index = doc.get(x, y) as usize;
    let c = doc.palette[index];
    let near = [c[0], c[1] ^ 2, c[2]];
    let mut h = History::new();
    // Recolored to its own RGB, the entry's LAB is rgb8_to_lab of it (a conversion's LAB
    // can lie outside the RGB gamut, far from its rendering).
    let op = palette::set_color(&doc, index, c).unwrap();
    h.push(&mut doc, op).unwrap();
    let op = palette::add_color(&doc, near).unwrap();
    h.push(&mut doc, op).unwrap();
    let k = doc.palette.len() as i64 - 1;
    let dot = tools::pencil(&doc, None, &[(x as i64, y as i64)], 1, k);
    h.push(&mut doc, Op::pixels("pencil", dot)).unwrap();
    let log = h.edit_log(&doc);
    let tmp = TempDir::new("rerender");
    let saved = save::save(&doc, &log, &tmp.join("bg")).unwrap();
    let edited = stat(&saved.stats, "edited").unwrap();
    assert_eq!(
        stat(edited, "tiles_rerender_px_changed"),
        Some(&Json::Int(1))
    );
    assert_eq!(stat(edited, "pixels_changed"), Some(&Json::Int(1)));

    let opened = Document::open(&tmp.join("bg.png")).unwrap();
    assert!(!opened.origin.prior_log_missing);
    assert_eq!(opened.indices, doc.indices);
    assert_eq!(opened.get(x, y), k);
}

/// `pixelforge edit PLAIN.png --source INPUT -o OUTDIR` without the window: a PNG that is not
/// a pixelforge output opens with its colors as the palette, the source as an unaligned
/// reference, and saves under OUTDIR; saving refuses to overwrite the source.
#[test]
fn session_opens_a_plain_png_with_a_source() {
    let tmp = TempDir::new("session-plain");
    let plain = fixture_path("two_color");
    let session = Session::open(&plain, Some(&plain), Some(tmp.path())).unwrap();
    let doc = &session.doc;
    assert_eq!((doc.width(), doc.height()), (128, 128));
    assert_eq!(doc.palette.len(), 2);
    let source = doc.origin.source.as_ref().unwrap();
    assert_eq!((source.w, source.h, source.factor), (128, 128, None));
    assert_eq!(session.prefix, tmp.join("two_color"));
    assert_eq!(session.image_path(), tmp.join("two_color.png"));
    assert!(!session.is_dirty());

    let mut session = session;
    let op = Op::pixels("pencil", tools::pencil(&session.doc, None, &[(0, 0)], 1, 1));
    session.apply(op).unwrap();
    session.save().unwrap();
    assert!(!session.is_dirty());
    let reopened = Document::open(&tmp.join("two_color.png")).unwrap();
    assert_eq!(reopened.indices, session.doc.indices);
    // Saving next to the source would overwrite it.
    let err = session
        .save_as(plain.with_extension(""))
        .unwrap_err()
        .to_string();
    assert!(err.contains("would overwrite input"), "{err}");
    assert_eq!(session.prefix, tmp.join("two_color"));
}

#[test]
fn edit_source_recovers_the_alignment_of_convert_edit() {
    // convert --edit: the source rendered through the conversion's own preprocessing.
    let dir = TempDir::new("editor-source-align");
    let input = fixture_path("circle_alpha");
    let config = config(&[
        ("preset", s("sprite")),
        ("method", s("box")),
        ("out_width", int(24)),
        ("denoise", s("none")),
    ]);
    let (out, loaded, pre) =
        pixelforge::pipeline::run_keeping(pixelforge::pipeline::Input::Path(&input), &config, None)
            .unwrap();
    let factor = pixelforge::pipeline::source_reference_factor(&pre, out.indices.w);
    let expected = pixelforge::pipeline::source_reference(&loaded, &pre, factor);
    assert!(expected.factor.is_some());
    out.save(&dir.join("circle")).unwrap();
    // edit --source: the same reference, recovered from the PNG's config.
    let png = dir.join("circle.png");
    let session = Session::open(&png, Some(&input), None).unwrap();
    assert_eq!(session.doc.origin.source.as_ref(), Some(&expected));
    // An image of another shape does not give the output's size: shown fitted, unaligned.
    let other = dir.join("wide.png");
    io::save_rgba_png(&other, &[200, 40, 40, 255].repeat(60 * 10), 10, 60).unwrap();
    let session = Session::open(&png, Some(&other), None).unwrap();
    let src = session.doc.origin.source.unwrap();
    assert_eq!(src.factor, None);
    assert_eq!(src, pixelforge::editor::SourceRef::load(&other).unwrap());
    // A plain PNG has no conversion config to recover the geometry from.
    let plain = dir.join("plain.png");
    io::save_rgba_png(&plain, &[9, 9, 9, 255].repeat(4), 2, 2).unwrap();
    let session = Session::open(&plain, Some(&input), None).unwrap();
    assert_eq!(session.doc.origin.source.unwrap().factor, None);
}
