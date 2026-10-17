//! The editor window driven headlessly with egui_kittest (docs/editor.md, Section 3.14): the
//! application opens a conversion, renders frames, paints through synthetic pointer input,
//! undoes and saves with shortcuts, and the document shows every change.

#![cfg(feature = "gui")]

mod common;

use common::*;
use eframe::egui::{self, Key, Modifiers, PointerButton, Pos2};
use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use pixelforge::config::Config;
use pixelforge::editor::{Document, Selection, Session};
use pixelforge::gui::tools::Tool;
use pixelforge::gui::{Action, EditorApp};

fn small_config() -> Config {
    config(&[
        ("preset", s("sprite")),
        ("method", s("box")),
        ("out_width", int(16)),
        ("out_height", int(16)),
        ("denoise", s("none")),
    ])
}

/// A conversion of circle_alpha saved under `dir`, opened as `pixelforge edit` would.
fn opened(dir: &TempDir) -> Session {
    let out = run_path(&fixture_path("circle_alpha"), &small_config()).unwrap();
    let prefix = dir.join("circle");
    out.save(&prefix).unwrap();
    Session::open(&dir.join("circle.png"), None, None).unwrap()
}

fn harness(session: Session) -> Harness<'static, EditorApp> {
    let app = EditorApp::new(session, &egui::Context::default());
    let mut harness = Harness::builder()
        .with_size(egui::vec2(900.0, 600.0))
        .build_ui_state(|ui, app: &mut EditorApp| app.show(ui), app);
    harness.run();
    harness
}

fn press(h: &Harness<'_, EditorApp>, pos: Pos2, button: PointerButton, pressed: bool) {
    h.event(egui::Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: Modifiers::NONE,
    });
}

/// The first opaque and the first transparent pixel.
fn opaque_and_clear(doc: &Document) -> ((i64, i64), (i64, i64)) {
    let w = doc.width();
    let find = |want_opaque: bool| {
        let p = doc
            .indices
            .indices
            .iter()
            .position(|&i| (i >= 0) == want_opaque)
            .expect("the sprite has both");
        ((p % w) as i64, (p / w) as i64)
    };
    (find(true), find(false))
}

#[test]
fn paints_a_pixel_undoes_and_saves() {
    let dir = TempDir::new("gui-smoke");
    let mut h = harness(opened(&dir));
    {
        let app = h.state();
        // Zoom to fit: a 16 × 16 image on a canvas of several hundred points.
        assert!(app.canvas.view.zoom >= 8, "zoom {}", app.canvas.view.zoom);
        assert!(!app.session.is_dirty());
        assert_eq!(app.tools.tool, Tool::Pencil);
    }
    let (_, clear) = opaque_and_clear(&h.state().session.doc);
    let primary = h.state().tools.primary;
    assert!(primary >= 0);
    let pos = h.state().canvas.pixel_center(clear.0, clear.1);

    // One click with the pencil paints the primary color.
    h.hover_at(pos);
    h.run();
    assert_eq!(
        h.state().canvas.pixel_center(clear.0, clear.1),
        pos,
        "the view moved"
    );
    press(&h, pos, PointerButton::Primary, true);
    h.run();
    press(&h, pos, PointerButton::Primary, false);
    h.run();
    {
        let app = h.state();
        let doc = &app.session.doc;
        assert_eq!(doc.get(clear.0 as usize, clear.1 as usize), primary);
        assert!(app.session.is_dirty());
        assert_eq!(app.session.history.applied().len(), 1);
        // The canvas shows the change.
        let [r, g, b] = doc.palette[primary as usize];
        let shown = app.canvas.pixels()[clear.1 as usize * doc.width() + clear.0 as usize];
        assert_eq!(shown, egui::Color32::from_rgb(r, g, b));
    }

    // Cmd/Ctrl+Z undoes it.
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run();
    {
        let app = h.state();
        assert_eq!(app.session.doc.get(clear.0 as usize, clear.1 as usize), -1);
        assert!(!app.session.is_dirty());
        let shown =
            app.canvas.pixels()[clear.1 as usize * app.session.doc.width() + clear.0 as usize];
        assert_eq!(shown, egui::Color32::TRANSPARENT);
    }
    // Redo, then Cmd/Ctrl+S writes the output set and clears the unsaved marker.
    h.key_press_modifiers(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::Z);
    h.run();
    assert!(h.state().session.is_dirty());
    h.key_press_modifiers(Modifiers::COMMAND, Key::S);
    h.run();
    let app = h.state();
    assert!(!app.session.is_dirty(), "{:?}", app.message);
    assert!(dir.join("circle_edits.json").is_file());
    let reopened = Document::open(&dir.join("circle.png")).unwrap();
    assert_eq!(reopened.indices, app.session.doc.indices);
    assert_eq!(reopened.origin.prior_ops.len(), 1);
}

#[test]
fn a_drag_is_one_step_and_alt_click_picks() {
    let dir = TempDir::new("gui-drag");
    let mut h = harness(opened(&dir));
    let (opaque, _) = opaque_and_clear(&h.state().session.doc);
    let picked = h
        .state()
        .session
        .doc
        .get(opaque.0 as usize, opaque.1 as usize);

    // Alt+right click with the pencil picks the secondary color.
    h.state_mut().tools.secondary = -1;
    let at = h.state().canvas.pixel_center(opaque.0, opaque.1);
    h.hover_at(at);
    h.run();
    h.event_modifiers(
        egui::Event::PointerButton {
            pos: at,
            button: PointerButton::Secondary,
            pressed: true,
            modifiers: Modifiers::ALT,
        },
        Modifiers::ALT,
    );
    h.event_modifiers(
        egui::Event::PointerButton {
            pos: at,
            button: PointerButton::Secondary,
            pressed: false,
            modifiers: Modifiers::ALT,
        },
        Modifiers::ALT,
    );
    h.run();
    assert_eq!(h.state().tools.secondary, picked);
    assert!(!h.state().session.is_dirty());

    // The eraser dragged across the middle of the circle: three pixels, one undo step.
    h.key_press(Key::E);
    h.run();
    assert_eq!(h.state().tools.tool, Tool::Eraser);
    h.state_mut()
        .perform(Action::BrushSize(1), &egui::Context::default());
    for x in 7..10 {
        assert!(h.state().session.doc.get(x, 8) >= 0);
    }
    let row: Vec<Pos2> = (7..10)
        .map(|x| h.state().canvas.pixel_center(x, 8))
        .collect();
    h.hover_at(row[0]);
    h.run();
    press(&h, row[0], PointerButton::Primary, true);
    h.run();
    for &p in &row[1..] {
        h.hover_at(p);
        h.run();
    }
    // Before the release the stroke is a preview only.
    assert_eq!(h.state().session.history.applied().len(), 0);
    press(&h, row[2], PointerButton::Primary, false);
    h.run();
    let app = h.state();
    assert_eq!(app.session.history.applied().len(), 1);
    for x in 7..10 {
        assert_eq!(app.session.doc.get(x, 8), -1);
    }
}

#[test]
fn fill_and_palette_edits_go_through_the_session() {
    let dir = TempDir::new("gui-fill");
    let mut h = harness(opened(&dir));
    let (_, clear) = opaque_and_clear(&h.state().session.doc);
    let transparent = h.state().session.transparent_count();
    h.key_press(Key::G);
    h.run();
    let at = h.state().canvas.pixel_center(clear.0, clear.1);
    h.hover_at(at);
    h.run();
    press(&h, at, PointerButton::Primary, true);
    press(&h, at, PointerButton::Primary, false);
    h.run();
    let app = h.state();
    assert!(app.session.transparent_count() < transparent);
    assert_eq!(app.session.history.applied().len(), 1);
    assert_eq!(app.session.history.applied()[0].kind(), "pixels");
    let counts: usize = app.session.counts().iter().sum();
    assert_eq!(counts + app.session.transparent_count(), 16 * 16);
}

#[test]
fn removing_a_used_entry_asks_and_remaps() {
    let dir = TempDir::new("gui-remove");
    let mut h = harness(opened(&dir));
    let k = h.state().session.doc.palette.len();
    let used = h.state().session.counts()[0];
    assert!(used > 0 && k > 1);
    h.state_mut().tools.primary = 0;
    h.run();
    h.get_by_label("Remove…").click();
    h.run();
    assert!(h.state().dialog.is_some());
    assert_eq!(h.state().session.doc.palette.len(), k);
    h.get_by_label("Remove").click();
    h.run();
    let app = h.state();
    assert!(app.dialog.is_none());
    assert_eq!(app.session.doc.palette.len(), k - 1);
    assert_eq!(app.session.counts().len(), k - 1);
    let total: usize = app.session.counts().iter().sum();
    assert_eq!(total + app.session.transparent_count(), 16 * 16);
    // The primary color followed its pixels to the entry they moved to.
    assert!(app.tools.primary >= 0 && (app.tools.primary as usize) < k - 1);
    assert_eq!(app.session.history.applied()[0].kind(), "palette_remove");
}

/// Drag with the primary button from pixel `a` to pixel `b` through `(b.x, a.y)`, with
/// `modifiers` held.
fn drag(h: &mut Harness<'_, EditorApp>, a: (i64, i64), b: (i64, i64), modifiers: Modifiers) {
    let center =
        |h: &Harness<'_, EditorApp>, p: (i64, i64)| h.state().canvas.pixel_center(p.0, p.1);
    let start = center(h, a);
    h.hover_at(start);
    h.run();
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: PointerButton::Primary,
        pressed,
        modifiers,
    };
    h.event_modifiers(button(start, true), modifiers);
    h.run();
    for p in [(b.0, a.1), b] {
        let pos = center(h, p);
        h.event_modifiers(egui::Event::PointerMoved(pos), modifiers);
        h.run();
    }
    let end = center(h, b);
    h.event_modifiers(button(end, false), modifiers);
    h.run();
}

/// A `w × h` PNG that is not a pixelforge output (R3): a grey ramp of four entries.
fn plain_ramp(dir: &TempDir, w: usize, h: usize) -> Session {
    let greys = [30u8, 90, 150, 210];
    let mut rgba = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let g = greys[(x + y) % 4];
            rgba.extend_from_slice(&[g, g, g, 255]);
        }
    }
    let path = dir.join("ramp.png");
    pixelforge::io::save_rgba_png(&path, &rgba, h, w).unwrap();
    Session::open(&path, None, None).unwrap()
}

fn indices(app: &EditorApp) -> Vec<i64> {
    app.session.doc.indices.indices.clone()
}

#[test]
fn tools_stay_inside_the_selection() {
    let dir = TempDir::new("gui-select");
    let mut h = harness(opened(&dir));
    h.key_press(Key::M);
    h.run();
    assert_eq!(h.state().tools.tool, Tool::Select);
    drag(&mut h, (4, 4), (7, 7), Modifiers::NONE);
    let sel = h.state().session.selection.expect("a selection");
    assert_eq!((sel.x, sel.y, sel.w, sel.h), (4, 4, 4, 4));
    assert!(h.state().session.history.applied().is_empty());

    // The pencil dragged across the selection's edge paints only inside it.
    h.key_press(Key::B);
    h.run();
    h.state_mut().tools.primary = -1;
    let before = indices(h.state());
    drag(&mut h, (1, 5), (12, 5), Modifiers::NONE);
    let app = h.state();
    assert_eq!(app.session.history.applied().len(), 1);
    let w = app.session.doc.width();
    for (p, (&a, &b)) in before.iter().zip(&indices(app)).enumerate() {
        let (x, y) = (p % w, p / w);
        if a != b {
            assert!(
                sel.contains(x, y),
                "({x}, {y}) changed outside the selection"
            );
        }
    }
    assert!((4..8).all(|x| app.session.doc.get(x, 5) == -1));

    // Esc clears the selection; Select > All selects everything.
    h.key_press(Key::Escape);
    h.run();
    assert_eq!(h.state().session.selection, None);
    h.key_press_modifiers(Modifiers::COMMAND, Key::A);
    h.run();
    assert_eq!(
        h.state().session.selection.map(|s| (s.w, s.h)),
        Some((16, 16))
    );
}

#[test]
fn moving_the_selection_moves_its_contents() {
    let dir = TempDir::new("gui-move");
    let mut h = harness(opened(&dir));
    h.key_press(Key::M);
    h.run();
    drag(&mut h, (6, 6), (9, 9), Modifiers::NONE);
    let before = h.state().session.doc.clone();
    // Drag from inside the selection two pixels right and one down.
    drag(&mut h, (7, 7), (9, 8), Modifiers::NONE);
    let app = h.state();
    let sel = app.session.selection.unwrap();
    assert_eq!((sel.x, sel.y, sel.w, sel.h), (8, 7, 4, 4));
    let applied = app.session.history.applied();
    assert_eq!(applied.len(), 1);
    assert!(matches!(&applied[0], pixelforge::editor::Op::Pixels { tool, .. } if tool == "move"));
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(
                app.session.doc.get(8 + x, 7 + y),
                before.get(6 + x, 6 + y),
                "moved pixel ({x}, {y})"
            );
        }
    }
}

#[test]
fn line_and_rectangle_preview_then_commit() {
    let dir = TempDir::new("gui-shapes");
    let mut h = harness(opened(&dir));
    h.state_mut().tools.primary = -1;
    h.key_press(Key::L);
    h.run();
    assert_eq!(h.state().tools.tool, Tool::Line);
    let start = h.state().canvas.pixel_center(2, 3);
    h.hover_at(start);
    h.run();
    press(&h, start, PointerButton::Primary, true);
    h.run();
    h.hover_at(h.state().canvas.pixel_center(12, 3));
    h.run();
    // While dragging: shown on the canvas, not applied.
    {
        let app = h.state();
        assert!(app.session.history.applied().is_empty());
        let w = app.session.doc.width();
        assert!((2..=12).all(|x| app.canvas.pixels()[3 * w + x] == egui::Color32::TRANSPARENT));
    }
    press(
        &h,
        h.state().canvas.pixel_center(12, 3),
        PointerButton::Primary,
        false,
    );
    h.run();
    {
        let app = h.state();
        assert_eq!(app.session.history.applied().len(), 1);
        assert!((2..=12).all(|x| app.session.doc.get(x, 3) == -1));
        assert_eq!(app.session.history.applied()[0].kind(), "pixels");
    }

    // Shift + rectangle fills it.
    h.key_press(Key::R);
    h.run();
    drag(&mut h, (5, 6), (10, 10), Modifiers::SHIFT);
    let app = h.state();
    assert_eq!(app.session.history.applied().len(), 2);
    for y in 6..=10 {
        for x in 5..=10 {
            assert_eq!(app.session.doc.get(x, y), -1);
        }
    }
}

#[test]
fn adjust_previews_and_applies_as_one_step() {
    let dir = TempDir::new("gui-adjust");
    let mut h = harness(plain_ramp(&dir, 6, 4));
    let before = indices(h.state());
    let shown = h.state().canvas.pixels().to_vec();
    h.state_mut().adjust.params.brightness = 40.0;
    h.run();
    {
        let app = h.state();
        assert!(!app.adjust.preview().is_empty());
        assert_ne!(app.canvas.pixels(), &shown[..], "no preview drawn");
        assert_eq!(indices(app), before, "the preview was applied");
        assert!(app.session.history.applied().is_empty());
    }
    h.get_by_label("Cancel").click();
    h.run();
    {
        let app = h.state();
        assert!(!app.adjust.active());
        assert!(app.adjust.preview().is_empty());
        assert_eq!(app.canvas.pixels(), &shown[..]);
        assert!(app.session.history.applied().is_empty());
    }
    h.state_mut().adjust.params.brightness = 40.0;
    h.state_mut().adjust.params.contrast = 1.5;
    h.run();
    let expected = h.state().adjust.preview().to_vec();
    h.get_by_label("Apply").click();
    h.run();
    {
        let app = h.state();
        assert_eq!(app.session.history.applied().len(), 1);
        match &app.session.history.applied()[0] {
            pixelforge::editor::Op::Pixels { tool, changes } => {
                assert_eq!(tool, "adjust");
                assert_eq!(changes, &expected);
            }
            other => panic!("{other:?}"),
        }
        assert!(!app.adjust.active());
    }
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run();
    assert_eq!(indices(h.state()), before);
    assert_eq!(h.state().canvas.pixels(), &shown[..]);
}

#[test]
fn large_adjust_previews_come_from_the_worker() {
    let dir = TempDir::new("gui-adjust-large");
    let (w, h_px) = (512, 400);
    assert!(w * h_px > pixelforge::gui::app::ADJUST_SYNC_PIXELS);
    let mut h = harness(plain_ramp(&dir, w, h_px));
    h.state_mut().adjust.params.brightness = -30.0;
    h.run();
    let start = std::time::Instant::now();
    while h.state().adjust.preview().is_empty() {
        assert!(
            start.elapsed().as_secs() < 30,
            "the adjust worker timed out"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
        h.run();
    }
    assert!(!h.state().adjust.busy());
    let expected =
        pixelforge::editor::adjust::adjust(&h.state().session.doc, None, &h.state().adjust.params);
    assert_eq!(h.state().adjust.preview(), &expected[..]);
    assert!(h.state().session.history.applied().is_empty());
}

#[test]
fn ramp_shift_keys() {
    let dir = TempDir::new("gui-ramp");
    let mut h = harness(plain_ramp(&dir, 6, 4));
    let lightness = |app: &EditorApp, x: usize, y: usize| {
        let i = app.session.doc.get(x, y) as usize;
        app.session.doc.palette[i][0]
    };
    let row = |app: &EditorApp| -> Vec<u8> { (0..6).map(|x| lightness(app, x, 0)).collect() };
    // One step along the grey ramp, staying at its ends.
    let greys = [30u8, 90, 150, 210];
    let step = |v: u8, up: bool| {
        let i = greys.iter().position(|&g| g == v).unwrap();
        greys[if up {
            (i + 1).min(3)
        } else {
            i.saturating_sub(1)
        }]
    };
    let before = row(h.state());
    h.key_press(Key::Period);
    h.run();
    let lighter: Vec<u8> = before.iter().map(|&v| step(v, true)).collect();
    assert_eq!(row(h.state()), lighter);
    assert_eq!(h.state().session.history.applied().len(), 1);
    h.key_press(Key::Comma);
    h.run();
    h.key_press(Key::Comma);
    h.run();
    let darker: Vec<u8> = lighter
        .iter()
        .map(|&v| step(step(v, false), false))
        .collect();
    assert_eq!(row(h.state()), darker);
    assert_eq!(h.state().session.history.applied().len(), 3);
}

#[test]
fn copy_and_paste_place_a_movable_selection() {
    let dir = TempDir::new("gui-paste");
    let mut h = harness(opened(&dir));
    h.key_press(Key::M);
    h.run();
    drag(&mut h, (5, 5), (8, 8), Modifiers::NONE);
    let source = h.state().session.doc.clone();
    h.key_press_modifiers(Modifiers::COMMAND, Key::C);
    h.run();
    assert!(h.state().clipboard.is_some());
    // Select elsewhere and paste there.
    drag(&mut h, (0, 0), (1, 1), Modifiers::NONE);
    h.key_press_modifiers(Modifiers::COMMAND, Key::V);
    h.run();
    {
        let app = h.state();
        let sel = app.session.selection.unwrap();
        assert_eq!((sel.x, sel.y, sel.w, sel.h), (0, 0, 4, 4));
        assert_eq!(app.session.history.applied().len(), 1);
        assert_eq!(app.tools.tool, Tool::Select);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(app.session.doc.get(x, y), source.get(5 + x, 5 + y));
            }
        }
    }
    // The window backend's clipboard events do the same.
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run();
    let at = |x, y, w| Some(Selection { x, y, w, h: w });
    h.state_mut().session.selection = at(5, 5, 4);
    h.event(egui::Event::Copy);
    h.run();
    h.state_mut().session.selection = at(0, 0, 1);
    h.event(egui::Event::Paste(format!(
        "{} 4x4",
        pixelforge::gui::app::CLIP_MARKER
    )));
    h.run();
    let app = h.state();
    assert_eq!(app.session.history.applied().len(), 1);
    assert_eq!(app.session.selection, at(0, 0, 4));
    // Text from elsewhere is not pasted.
    h.event(egui::Event::Paste("hello".into()));
    h.run();
    assert_eq!(h.state().session.history.applied().len(), 1);
}

#[test]
fn the_diff_overlay_only_draws() {
    let dir = TempDir::new("gui-diff");
    let mut h = harness(opened(&dir));
    let (_, clear) = opaque_and_clear(&h.state().session.doc);
    let pos = h.state().canvas.pixel_center(clear.0, clear.1);
    h.hover_at(pos);
    h.run();
    press(&h, pos, PointerButton::Primary, true);
    press(&h, pos, PointerButton::Primary, false);
    h.run();
    let doc = h.state().session.doc.indices.clone();
    let shown = h.state().canvas.pixels().to_vec();
    h.key_press(Key::D);
    h.run();
    {
        let app = h.state();
        assert!(app.canvas.diff_shown());
        let tint = app.canvas.diff_pixels().unwrap();
        let w = app.session.doc.width();
        let tinted: Vec<usize> = (0..tint.len())
            .filter(|&p| tint[p] != egui::Color32::TRANSPARENT)
            .collect();
        assert_eq!(tinted, vec![clear.1 as usize * w + clear.0 as usize]);
        assert_eq!(app.session.doc.indices, doc);
        assert_eq!(app.canvas.pixels(), &shown[..]);
        assert_eq!(app.session.history.applied().len(), 1);
    }
    // Undo clears the tint; D hides the overlay.
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run();
    assert!(h
        .state()
        .canvas
        .diff_pixels()
        .unwrap()
        .iter()
        .all(|&c| c == egui::Color32::TRANSPARENT));
    h.key_press(Key::D);
    h.run();
    assert!(!h.state().canvas.diff_shown());
}

#[test]
fn replace_and_merge_through_dialogs() {
    let dir = TempDir::new("gui-replace");
    let mut h = harness(plain_ramp(&dir, 6, 4));
    let counts = h.state().session.counts().to_vec();
    h.state_mut().tools.primary = 0;
    h.state_mut().tools.secondary = 3;
    h.get_by_label("Replace…").click();
    h.run();
    assert!(h.state().dialog.is_some());
    h.get_by_label("Replace").click();
    h.run();
    {
        let app = h.state();
        assert!(app.dialog.is_none());
        assert_eq!(app.session.counts()[0], 0);
        assert_eq!(app.session.counts()[3], counts[0] + counts[3]);
    }
    h.state_mut().tools.primary = 1;
    h.state_mut().tools.secondary = 2;
    h.get_by_label("Merge…").click();
    h.run();
    h.get_by_label("Merge").click();
    h.run();
    let app = h.state();
    assert_eq!(app.session.doc.palette.len(), 3);
    assert_eq!(app.session.history.applied().len(), 2);
    assert_eq!(app.session.history.applied()[1].kind(), "palette_remove");
    // The primary color followed its pixels into the merged entry.
    assert_eq!(app.tools.primary, 1);
}

#[test]
fn a_plain_png_opens_edits_and_saves() {
    let dir = TempDir::new("gui-plain");
    let mut h = harness(plain_ramp(&dir, 6, 4));
    assert_eq!(h.state().session.doc.palette.len(), 4);
    h.key_press(Key::Num4);
    h.run();
    assert_eq!(h.state().tools.primary, 3);
    let pos = h.state().canvas.pixel_center(0, 0);
    h.hover_at(pos);
    h.run();
    press(&h, pos, PointerButton::Primary, true);
    press(&h, pos, PointerButton::Primary, false);
    h.run();
    assert_eq!(h.state().session.doc.get(0, 0), 3);
    h.key_press_modifiers(Modifiers::COMMAND, Key::S);
    h.run();
    assert!(!h.state().session.is_dirty(), "{:?}", h.state().message);
    let reopened = Document::open(&dir.join("ramp.png")).unwrap();
    assert_eq!(reopened.indices, h.state().session.doc.indices);
    assert!(dir.join("ramp_meta.json").is_file());
}

#[test]
fn the_tiles_panel_shows_the_reconstruction() {
    let dir = TempDir::new("gui-tiles");
    let config = config(&[
        ("preset", s("background")),
        ("method", s("box")),
        ("out_width", int(32)),
        ("out_height", int(32)),
        ("tile_size", int(8)),
    ]);
    let out = run_path(&fixture_path("circle_alpha"), &config).unwrap();
    out.save(&dir.join("bg")).unwrap();
    let session = Session::open(&dir.join("bg.png"), None, None).unwrap();
    assert!(session.doc.origin.config.tileset);
    let mut h = harness(session);
    let start = std::time::Instant::now();
    while h.state().tiles_info().is_none() || h.state().tiles_busy() {
        assert!(start.elapsed().as_secs() < 30, "the tiles worker timed out");
        std::thread::sleep(std::time::Duration::from_millis(10));
        h.run();
    }
    let summary = match h.state().tiles_info() {
        Some(Ok(Some(t))) => t.clone(),
        other => panic!("{other:?}"),
    };
    assert!(summary.tiles >= 1);
    assert_eq!(summary.rendered.w, 32);
    h.get_by_label("Show what the tilemap draws").click();
    h.run();
    assert!(h.state().show_reconstructed);
    assert!(h.state().canvas.reconstructed_shown());
    h.get_by_label("Show what the tilemap draws").click();
    h.run();
    assert!(!h.state().canvas.reconstructed_shown());
}

#[test]
fn the_shortcuts_sheet_lists_the_table() {
    let dir = TempDir::new("gui-help");
    let mut h = harness(opened(&dir));
    let rows = pixelforge::gui::app::shortcut_sheet(&h.ctx);
    let mut actions: Vec<_> = pixelforge::gui::app::SHORTCUTS
        .iter()
        .map(|(_, a)| *a)
        .collect();
    actions.dedup();
    let unique = {
        let mut seen = Vec::new();
        for a in &actions {
            if !seen.contains(a) {
                seen.push(*a);
            }
        }
        seen.len()
    };
    assert_eq!(rows.len(), unique);
    assert!(rows.iter().any(|(_, what)| what == "Ramp shift lighter"));
    h.state_mut()
        .perform(Action::Shortcuts, &egui::Context::default());
    h.run();
    assert!(h.state().show_shortcuts);
    h.get_by_label("Ramp shift lighter");
}

#[test]
fn tab_toggles_the_aligned_source_overlay() {
    let dir = TempDir::new("gui-source");
    let mut session = opened(&dir);
    let (w, h_px) = (session.doc.width(), session.doc.height());
    session.doc.origin.source = Some(pixelforge::editor::SourceRef {
        rgba: vec![200; w * 2 * h_px * 2 * 4],
        h: h_px * 2,
        w: w * 2,
        factor: Some(2),
    });
    let mut h = harness(session);
    assert!(!h.state().canvas.show_source);
    h.key_press(Key::Tab);
    h.run();
    assert!(h.state().canvas.show_source);
    // The panel's checkbox is the same switch.
    h.get_by_label("Overlay (Tab)").click();
    h.run();
    assert!(!h.state().canvas.show_source);
    assert!(h.state().session.history.applied().is_empty());
}

#[test]
fn dither_previews_then_applies_as_one_step() {
    let dir = TempDir::new("gui-dither");
    let mut h = harness(plain_ramp(&dir, 8, 8));
    h.state_mut().session.selection = Some(Selection {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    });
    h.state_mut().tools.primary = 0;
    h.state_mut().tools.secondary = 3;
    let before = indices(h.state());
    let shown = h.state().canvas.pixels().to_vec();
    h.state_mut()
        .perform(Action::Dither, &egui::Context::default());
    h.run();
    {
        // The dialog's result is drawn, not applied.
        let app = h.state();
        assert!(matches!(
            app.dialog,
            Some(pixelforge::gui::dialogs::Dialog::Dither {
                a: 0,
                b: 3,
                density: 32,
                ..
            })
        ));
        assert_eq!(indices(app), before);
        assert_ne!(app.canvas.pixels(), &shown[..]);
        assert!(app.session.history.applied().is_empty());
    }
    h.get_by_label("Dither").click();
    h.run();
    let app = h.state();
    assert!(app.dialog.is_none());
    assert_eq!(app.session.history.applied().len(), 1);
    assert_eq!(
        app.session.history.applied()[0],
        pixelforge::editor::Op::pixels(
            "dither",
            pixelforge::editor::adjust::dither(
                &plain_doc(&before, &app.session.doc),
                app.session.selection.as_ref(),
                0,
                3,
                32
            )
        )
    );
    let w = app.session.doc.width();
    for (p, (&a, &b)) in before.iter().zip(&indices(app)).enumerate() {
        if a != b {
            assert!(p % w < 4 && p / w < 4, "pixel {p} changed outside");
        }
        if p % w < 4 && p / w < 4 {
            assert!(b == 0 || b == 3);
        }
    }

    // Closing the dialog drops the preview.
    let shown = h.state().canvas.pixels().to_vec();
    h.state_mut()
        .perform(Action::Dither, &egui::Context::default());
    h.run();
    if let Some(pixelforge::gui::dialogs::Dialog::Dither { density, .. }) =
        h.state_mut().dialog.as_mut()
    {
        *density = 60;
    }
    h.run();
    assert_ne!(h.state().canvas.pixels(), &shown[..]);
    h.key_press(Key::Escape);
    h.run();
    assert!(h.state().dialog.is_none());
    assert_eq!(h.state().canvas.pixels(), &shown[..]);
    assert_eq!(h.state().session.history.applied().len(), 1);
}

/// `doc` with the indices `indices` (the document before an edit).
fn plain_doc(indices: &[i64], doc: &Document) -> Document {
    let mut d = doc.clone();
    d.indices.indices = indices.to_vec();
    d
}

#[test]
fn an_image_paste_arrives_as_the_key_release() {
    // With an image on the system clipboard the backend reports only the release of
    // Cmd/Ctrl+V; it pastes once. (The system clipboard stays off in tests: the editor's
    // own copy is pasted.)
    let dir = TempDir::new("gui-paste-release");
    let mut h = harness(opened(&dir));
    h.state_mut().session.selection = Some(Selection {
        x: 5,
        y: 5,
        w: 3,
        h: 3,
    });
    h.state_mut()
        .perform(Action::Copy, &egui::Context::default());
    h.state_mut().session.selection = Some(Selection {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    });
    let release = |pressed| egui::Event::Key {
        key: Key::V,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers: Modifiers::COMMAND,
    };
    h.event(release(false));
    h.run();
    assert_eq!(h.state().session.history.applied().len(), 1);
    assert_eq!(
        h.state().session.selection.map(|s| (s.w, s.h)),
        Some((3, 3))
    );
    // Text on the clipboard: the press reports it, and the release does not paste again.
    h.event(egui::Event::Paste("some text".into()));
    h.run();
    h.event(release(false));
    h.run();
    assert_eq!(h.state().session.history.applied().len(), 1);
}

#[test]
fn per_tile_editing_paints_every_instance() {
    use pixelforge::config::Value;
    use pixelforge::editor::Origin;
    use pixelforge::pipeline::IndexImage;
    // Two identical 8 × 8 tiles side by side, and a different third one.
    let (w, h) = (24, 8);
    let indices = (0..w * h)
        .map(|p| {
            let (x, y) = (p % w, p / w);
            if x >= 16 {
                2
            } else {
                ((x % 8 + y) % 2) as i64
            }
        })
        .collect();
    let config = pixelforge::config::Config::new(&[
        ("tileset".into(), Value::Bool(true)),
        ("tile_size".into(), Value::Int(8)),
    ])
    .unwrap();
    let origin = Origin::new(
        config,
        String::new(),
        pixelforge::pyjson::Json::Object(vec![]),
    );
    let doc = Document::new(
        IndexImage { indices, h, w },
        vec![[0, 0, 0], [255, 255, 255], [200, 0, 0]],
        origin,
    )
    .unwrap();
    let dir = TempDir::new("gui-tile-mode");
    let mut h = harness(Session::new(doc, dir.join("tiles")));
    h.key_press(Key::T);
    h.run();
    assert!(h.state().tile_mode);
    h.state_mut().tools.primary = 2;
    drag(&mut h, (1, 1), (1, 1), Modifiers::NONE);
    {
        let app = h.state();
        assert_eq!(app.session.history.applied().len(), 1);
        let doc = &app.session.doc;
        assert_eq!((doc.get(1, 1), doc.get(9, 1)), (2, 2));
        assert_eq!(doc.get(17, 1), 2);
        assert_eq!(app.session.doc.pixels_changed(), 2);
    }
    // Off again: one pixel.
    h.key_press(Key::T);
    h.run();
    drag(&mut h, (2, 1), (2, 1), Modifiers::NONE);
    assert_eq!(h.state().session.doc.pixels_changed(), 3);
}
