//! The pixel editor's window (docs/editor.md, Sections 3.7 and 3.11 to 3.13), behind the
//! `gui` feature: an eframe (egui, OpenGL) application around one [`Session`]. The editing
//! itself is the headless `editor` module; this layer maps input to it and draws.
//!
//! - `app`: [`EditorApp`], the window: menus, toolbar, panels, status bar, dialogs, the
//!   adjust preview and the clipboard; every command is an [`Action`], with the shortcut
//!   table [`app::SHORTCUTS`] (also the Help menu's shortcuts sheet).
//! - `canvas`: the canvas widget (texture with partial uploads, checkerboard, grids, the
//!   selection's marquee, the diff, source and tilemap layers, pan and zoom input, pointer
//!   events in image pixels).
//! - `view`: the integer screen ↔ pixel math (power-of-two zoom, whole-point pan).
//! - `tools`: the active tool, brush and colors; pointer events to edits.
//! - `palette_panel`: the palette list and its actions.
//! - `panels`: the adjust, tiles and source reference panels.
//! - `dialogs`: entry color, remove confirmation, replace color, merge entries, dither, unsaved
//!   changes.
//! - `worker`: the generation-counted worker thread (tile re-extraction, adjust previews of
//!   large selections).
//!
//! Screen math is integer (the zoom is a power of two, positions are floored to whole
//! points), so the determinism guard applies here unchanged; nothing in this module computes
//! an edit's pixels.

pub mod app;
pub mod canvas;
pub mod dialogs;
pub mod palette_panel;
pub mod panels;
pub mod tools;
pub mod view;
pub mod worker;

pub use app::{Action, EditorApp};

use eframe::egui;

use crate::editor::Session;

/// Open the editor window on `session` and return when it is closed. Errors when no window
/// can be opened (no display, no OpenGL).
pub fn run(session: Session) -> Result<(), String> {
    let name = session
        .image_path()
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(format!("{name} — pixelforge"))
            .with_inner_size([1100.0, 760.0])
            .with_min_inner_size([480.0, 320.0]),
        ..Default::default()
    };
    eframe::run_native(
        "pixelforge",
        options,
        Box::new(|cc| Ok(Box::new(EditorApp::new(session, &cc.egui_ctx)))),
    )
    .map_err(|e| e.to_string())
}
