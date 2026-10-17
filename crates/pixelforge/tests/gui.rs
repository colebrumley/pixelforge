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
use pixelforge::editor::{Document, Session};
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
