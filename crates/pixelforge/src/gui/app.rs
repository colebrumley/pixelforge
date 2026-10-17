//! The editor window (docs/editor.md, Section 3.12): menus, the toolbar, the canvas, the
//! palette panel and the status bar around one [`Session`].
//!
//! Every command, whether from a menu, a toolbar button or a shortcut, is an [`Action`] run
//! by [`EditorApp::perform`]; the shortcut table is [`SHORTCUTS`]. Canvas pointer events go
//! to the [`ToolBox`], whose [`Effect`]s become ops on the session. A change reports the part
//! of the image it touched, and the canvas re-uploads only that.
//!
//! Adding a command: an `Action` variant, its arm in `perform`, and its menu entry and
//! shortcut. Adding a panel: a `Panel` in [`EditorApp::show`] before the central canvas.

use eframe::egui;
use std::path::PathBuf;

use egui::{Color32, Key, KeyboardShortcut, Modifiers, Panel, Ui, ViewportCommand};

use super::canvas::{index_color, Canvas, CanvasInput};
use super::dialogs::{self, ColorDialog, Dialog, Outcome};
use super::palette_panel::{self, PaletteAction};
use super::tools::{Effect, Tool, ToolBox, BRUSH_SIZES};
use super::worker::Worker;
use crate::color;
use crate::config::Config;
use crate::editor::{palette, Dirty, Op, Session};
use crate::pipeline::IndexImage;

/// A command of the editor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Save,
    SaveAs,
    Close,
    Undo,
    Redo,
    Tool(Tool),
    BrushSize(usize),
    BrushSmaller,
    BrushLarger,
    SwapColors,
    /// Palette entry `n` becomes the primary color (the 1 to 9 keys).
    PickEntry(usize),
    ZoomIn,
    ZoomOut,
    ZoomFit,
    ToggleGrid,
    ToggleTileGrid,
    /// Esc: drop the stroke in progress.
    Cancel,
}

const fn key(modifiers: Modifiers, key: Key) -> KeyboardShortcut {
    KeyboardShortcut::new(modifiers, key)
}

const CMD: Modifiers = Modifiers::COMMAND;
const CMD_SHIFT: Modifiers = Modifiers::COMMAND.plus(Modifiers::SHIFT);
const NONE: Modifiers = Modifiers::NONE;

/// The keyboard shortcuts, most specific first (egui ignores extra Shift and Alt when
/// matching, so Cmd+Shift+Z must be tried before Cmd+Z).
pub const SHORTCUTS: &[(KeyboardShortcut, Action)] = &[
    (key(CMD_SHIFT, Key::Z), Action::Redo),
    (key(CMD_SHIFT, Key::S), Action::SaveAs),
    (key(CMD, Key::Z), Action::Undo),
    (key(CMD, Key::Y), Action::Redo),
    (key(CMD, Key::S), Action::Save),
    (key(CMD, Key::W), Action::Close),
    (key(NONE, Key::B), Action::Tool(Tool::Pencil)),
    (key(NONE, Key::E), Action::Tool(Tool::Eraser)),
    (key(NONE, Key::I), Action::Tool(Tool::Eyedropper)),
    (key(NONE, Key::G), Action::Tool(Tool::Fill)),
    (key(NONE, Key::X), Action::SwapColors),
    (key(NONE, Key::OpenBracket), Action::BrushSmaller),
    (key(NONE, Key::CloseBracket), Action::BrushLarger),
    (key(NONE, Key::Plus), Action::ZoomIn),
    (key(NONE, Key::Equals), Action::ZoomIn),
    (key(NONE, Key::Minus), Action::ZoomOut),
    (key(NONE, Key::Num0), Action::ZoomFit),
    (key(NONE, Key::Num1), Action::PickEntry(0)),
    (key(NONE, Key::Num2), Action::PickEntry(1)),
    (key(NONE, Key::Num3), Action::PickEntry(2)),
    (key(NONE, Key::Num4), Action::PickEntry(3)),
    (key(NONE, Key::Num5), Action::PickEntry(4)),
    (key(NONE, Key::Num6), Action::PickEntry(5)),
    (key(NONE, Key::Num7), Action::PickEntry(6)),
    (key(NONE, Key::Num8), Action::PickEntry(7)),
    (key(NONE, Key::Num9), Action::PickEntry(8)),
    (key(NONE, Key::Escape), Action::Cancel),
];

/// The first shortcut of `action`, for the menus.
fn shortcut_of(action: Action) -> Option<&'static KeyboardShortcut> {
    SHORTCUTS.iter().find(|(_, a)| *a == action).map(|(s, _)| s)
}

/// The tile re-extraction job of the worker: a snapshot of the image and LAB palette.
type TilesJob = (IndexImage, Vec<f64>, Config);
/// Its result: `(tile count, pixels de-duplication changes)`, `None` without a tileset.
type TilesResult = Result<Option<(usize, usize)>, String>;

/// A status-bar message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub text: String,
    pub error: bool,
}

/// The editor application: one session and the state of its window.
pub struct EditorApp {
    pub session: Session,
    pub tools: ToolBox,
    pub canvas: Canvas,
    pub dialog: Option<Dialog>,
    pub message: Option<Message>,
    /// The image pixel under the pointer.
    hover: Option<(i64, i64)>,
    /// Re-extracts the tiles after edits (only with `config.tileset`).
    tiles: Option<Worker<TilesJob, TilesResult>>,
    tiles_revision: Option<u64>,
    tiles_info: Option<TilesResult>,
    /// The user chose to close despite unsaved changes.
    allow_close: bool,
    title: String,
    /// Run the Save As dialog on the next frame (outside of the menu that asked for it).
    save_as_pending: bool,
}

impl EditorApp {
    /// The editor on `session`. `ctx` is used by the worker thread to request a repaint
    /// when a result is ready.
    pub fn new(session: Session, ctx: &egui::Context) -> EditorApp {
        let tools = ToolBox::new(session.doc.palette.len());
        let canvas = Canvas::new(&session.doc);
        let tiles = session.doc.origin.config.tileset.then(|| {
            let ctx = ctx.clone();
            Worker::spawn(
                "pixelforge-tiles",
                |(idx, lab, config): TilesJob| {
                    Session::tiles_summary(&idx, &lab, &config).map_err(|e| e.to_string())
                },
                move || ctx.request_repaint(),
            )
        });
        EditorApp {
            session,
            tools,
            canvas,
            dialog: None,
            message: None,
            hover: None,
            tiles,
            tiles_revision: None,
            tiles_info: None,
            allow_close: false,
            title: String::new(),
            save_as_pending: false,
        }
    }

    fn info(&mut self, text: impl Into<String>) {
        self.message = Some(Message {
            text: text.into(),
            error: false,
        });
    }

    fn error(&mut self, text: impl Into<String>) {
        self.message = Some(Message {
            text: text.into(),
            error: true,
        });
    }

    /// The canvas and the tools after the document changed.
    fn changed(&mut self, dirty: Dirty) {
        self.canvas.refresh(&self.session.doc, dirty);
        self.tools.fit_palette(self.session.doc.palette.len());
    }

    /// Apply `op` as one undo step, reporting a refused op in the status bar.
    pub fn apply(&mut self, op: Op) -> bool {
        match self.session.apply(op) {
            Ok(Some(dirty)) => {
                self.changed(dirty);
                true
            }
            Ok(None) => false,
            Err(e) => {
                self.error(e.0);
                false
            }
        }
    }

    fn effect(&mut self, effect: Effect) {
        match effect {
            Effect::None => {}
            Effect::Preview(changes) => self.canvas.preview(&self.session.doc, &changes),
            Effect::Cancel(changes) => self.canvas.restore(&self.session.doc, &changes),
            Effect::Commit(op) => {
                let touched: Vec<_> = match &op {
                    Op::Pixels { changes, .. } => changes.clone(),
                    _ => Vec::new(),
                };
                if !self.apply(op) {
                    // Refused or empty: drop any preview of it.
                    self.canvas.restore(&self.session.doc, &touched);
                }
            }
            Effect::Pick(button, index) => self.tools.set_color(button, index),
        }
    }

    /// Run a command.
    pub fn perform(&mut self, action: Action, ctx: &egui::Context) {
        match action {
            Action::Save => {
                self.save();
            }
            Action::SaveAs => self.save_as_pending = true,
            Action::Close => ctx.send_viewport_cmd(ViewportCommand::Close),
            Action::Undo | Action::Redo => {
                let cancelled = self.tools.cancel();
                self.effect(cancelled);
                let dirty = if action == Action::Undo {
                    self.session.undo()
                } else {
                    self.session.redo()
                };
                if let Some(dirty) = dirty {
                    self.changed(dirty);
                }
            }
            Action::Tool(tool) => self.tools.tool = tool,
            Action::BrushSize(n) => self.tools.brush = n,
            Action::BrushSmaller => self.tools.resize_brush(-1),
            Action::BrushLarger => self.tools.resize_brush(1),
            Action::SwapColors => self.tools.swap_colors(),
            Action::PickEntry(n) => {
                if n < self.session.doc.palette.len() {
                    self.tools.primary = n as i64;
                }
            }
            Action::ZoomIn => self.canvas.zoom_steps(1),
            Action::ZoomOut => self.canvas.zoom_steps(-1),
            Action::ZoomFit => self.canvas.zoom_to_fit(),
            Action::ToggleGrid => self.canvas.show_grid = !self.canvas.show_grid,
            Action::ToggleTileGrid => self.canvas.show_tile_grid = !self.canvas.show_tile_grid,
            Action::Cancel => {
                let cancelled = self.tools.cancel();
                self.effect(cancelled);
            }
        }
    }

    /// Save under the session's prefix; `true` on success.
    pub fn save(&mut self) -> bool {
        match self.session.save() {
            Ok(saved) => {
                let dir = self
                    .session
                    .prefix
                    .parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                self.info(format!("saved {} files to {dir}", saved.outputs.len()));
                true
            }
            Err(e) => {
                self.error(format!("save failed: {e}"));
                false
            }
        }
    }

    /// Ask for a new PNG name and save there (later saves go there too).
    fn save_as(&mut self) {
        let current = self.session.image_path();
        let mut dialog = rfd::FileDialog::new()
            .set_title("Save as")
            .add_filter("PNG image", &["png"]);
        if let Some(dir) = current.parent() {
            dialog = dialog.set_directory(dir);
        }
        if let Some(name) = current.file_name() {
            dialog = dialog.set_file_name(name.to_string_lossy());
        }
        let Some(path) = dialog.save_file() else {
            return;
        };
        let prefix = match path.extension() {
            Some(ext) if ext.eq_ignore_ascii_case("png") => path.with_extension(""),
            _ => path,
        };
        match self.session.save_as(prefix) {
            Ok(saved) => self.info(format!(
                "saved {} files as {}",
                saved.outputs.len(),
                self.session.image_path().display()
            )),
            Err(e) => self.error(format!("save failed: {e}")),
        }
    }

    fn palette_action(&mut self, action: PaletteAction) {
        let doc = &self.session.doc;
        match action {
            PaletteAction::Pick(button, index) => self.tools.set_color(button, index),
            PaletteAction::Edit(i) => {
                let choices = palette::fixed_choices(doc);
                self.dialog = Some(Dialog::Color(ColorDialog::new(
                    Some(i),
                    doc.palette[i],
                    choices,
                )));
            }
            PaletteAction::Add => {
                let choices = palette::fixed_choices(doc);
                // A named palette starts at its first unused color.
                let start = choices
                    .as_ref()
                    .and_then(|c| c.iter().find(|(_, used)| !used).or(c.first()))
                    .map_or([128, 128, 128], |(rgb, _)| *rgb);
                self.dialog = Some(Dialog::Color(ColorDialog::new(None, start, choices)));
            }
            PaletteAction::Remove(i) => {
                let moved = self.session.counts()[i];
                match palette::nearest_other(doc, i) {
                    Some(_) if moved == 0 => self.remove_entry(i),
                    Some(target) => {
                        self.dialog = Some(Dialog::Remove {
                            index: i,
                            moved,
                            target,
                            target_rgb: doc.palette[target],
                        })
                    }
                    None => self.error("the last palette entry cannot be removed"),
                }
            }
        }
    }

    fn remove_entry(&mut self, index: usize) {
        let doc = &self.session.doc;
        let op = match palette::remove_entry(doc, index) {
            Ok(op) => op,
            Err(e) => return self.error(e.0),
        };
        // The colors keep pointing at the same entries after the shift; one on the removed
        // entry follows its pixels to the nearest entry.
        let target = palette::nearest_other(doc, index).map_or(-1, |t| t as i64);
        let index = index as i64;
        let follow = |c: i64| {
            let c = if c == index { target } else { c };
            if c > index {
                c - 1
            } else {
                c
            }
        };
        let colors = (follow(self.tools.primary), follow(self.tools.secondary));
        if self.apply(op) {
            (self.tools.primary, self.tools.secondary) = colors;
        }
    }

    fn dialog_outcome(&mut self, outcome: Outcome, ctx: &egui::Context) {
        self.dialog = None;
        match outcome {
            Outcome::Cancel => {}
            Outcome::Color(Some(i), rgb) => {
                if self.session.doc.palette[i] != rgb {
                    match palette::set_color(&self.session.doc, i, rgb) {
                        Ok(op) => {
                            self.apply(op);
                        }
                        Err(e) => self.error(e.0),
                    }
                }
            }
            Outcome::Color(None, rgb) => match palette::add_color(&self.session.doc, rgb) {
                Ok(op) => {
                    if self.apply(op) {
                        self.tools.primary = self.session.doc.palette.len() as i64 - 1;
                    }
                }
                Err(e) => self.error(e.0),
            },
            Outcome::Remove(i) => self.remove_entry(i),
            Outcome::SaveAndClose => {
                if self.save() {
                    self.allow_close = true;
                    ctx.send_viewport_cmd(ViewportCommand::Close);
                }
            }
            Outcome::DiscardAndClose => {
                self.allow_close = true;
                ctx.send_viewport_cmd(ViewportCommand::Close);
            }
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if self.dialog.is_some() || ctx.text_edit_focused() {
            return;
        }
        let pressed: Vec<Action> = ctx.input_mut(|i| {
            SHORTCUTS
                .iter()
                .filter(|(s, _)| i.consume_shortcut(s))
                .map(|(_, a)| *a)
                .collect()
        });
        for action in pressed {
            self.perform(action, ctx);
        }
    }

    /// The window title: the PNG's name, `*` with unsaved changes.
    fn update_title(&mut self, ctx: &egui::Context) {
        let name = self
            .session
            .image_path()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let dirty = if self.session.is_dirty() { " *" } else { "" };
        let title = format!("{name}{dirty} — pixelforge");
        if title != self.title {
            ctx.send_viewport_cmd(ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    /// Hand a changed image to the tiles worker and pick up its result.
    fn update_tiles(&mut self) {
        let Some(worker) = self.tiles.as_mut() else {
            return;
        };
        if let Some(result) = worker.poll() {
            self.tiles_info = Some(result);
        }
        let revision = self.session.revision();
        if self.tiles_revision != Some(revision) && !worker.busy() {
            let doc = &self.session.doc;
            worker.submit((
                doc.indices.clone(),
                doc.palette_lab.clone(),
                doc.origin.config.clone(),
            ));
            self.tiles_revision = Some(revision);
        }
    }

    fn menu_item(&mut self, ui: &mut Ui, label: &str, action: Action, enabled: bool) {
        let mut button = egui::Button::new(label);
        if let Some(s) = shortcut_of(action) {
            button = button.shortcut_text(ui.ctx().format_shortcut(s));
        }
        if ui.add_enabled(enabled, button).clicked() {
            self.perform(action, &ui.ctx().clone());
        }
    }

    fn menu_bar(&mut self, ui: &mut Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                self.menu_item(ui, "Save", Action::Save, true);
                self.menu_item(ui, "Save As…", Action::SaveAs, true);
                ui.separator();
                self.menu_item(ui, "Close", Action::Close, true);
            });
            ui.menu_button("Edit", |ui| {
                let (undo, redo) = (
                    self.session.history.can_undo(),
                    self.session.history.can_redo(),
                );
                self.menu_item(ui, "Undo", Action::Undo, undo);
                self.menu_item(ui, "Redo", Action::Redo, redo);
                ui.separator();
                self.menu_item(ui, "Swap colors", Action::SwapColors, true);
            });
            ui.menu_button("View", |ui| {
                self.menu_item(ui, "Zoom in", Action::ZoomIn, true);
                self.menu_item(ui, "Zoom out", Action::ZoomOut, true);
                self.menu_item(ui, "Zoom to fit", Action::ZoomFit, true);
                ui.separator();
                let grid = if self.canvas.show_grid {
                    "Hide pixel grid"
                } else {
                    "Show pixel grid"
                };
                self.menu_item(ui, grid, Action::ToggleGrid, true);
                let tiles = if self.canvas.show_tile_grid {
                    "Hide tile grid"
                } else {
                    "Show tile grid"
                };
                let tileset = self.session.doc.origin.config.tileset;
                self.menu_item(ui, tiles, Action::ToggleTileGrid, tileset);
            });
        });
    }

    fn toolbar(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        ui.add_space(4.0);
        for tool in Tool::ALL {
            let text = format!("{} ({})", tool.label(), tool.key().name());
            let resp = ui
                .add(egui::Button::selectable(
                    self.tools.tool == tool,
                    tool.label(),
                ))
                .on_hover_text(format!("{text}\n{}", tool.hint()));
            if resp.clicked() {
                self.perform(Action::Tool(tool), &ctx);
            }
        }
        ui.separator();
        ui.weak("Brush");
        for n in BRUSH_SIZES {
            let resp = ui
                .add(egui::Button::selectable(
                    self.tools.brush == n,
                    n.to_string(),
                ))
                .on_hover_text("Brush size ([ and ] step by one)");
            if resp.clicked() {
                self.perform(Action::BrushSize(n), &ctx);
            }
        }
        if !BRUSH_SIZES.contains(&self.tools.brush) {
            ui.label(format!("{} px", self.tools.brush));
        }
        ui.separator();
        // The two colors; click swaps them.
        let doc = &self.session.doc;
        let (primary, secondary) = (self.tools.primary, self.tools.secondary);
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(40.0, 40.0), egui::Sense::click());
        let painter = ui.painter();
        let back =
            egui::Rect::from_min_size(rect.min + egui::vec2(14.0, 14.0), egui::vec2(24.0, 24.0));
        let front = egui::Rect::from_min_size(rect.min, egui::vec2(24.0, 24.0));
        for (r, index) in [(back, secondary), (front, primary)] {
            let c = index_color(doc, index);
            if c == Color32::TRANSPARENT {
                painter.rect_filled(r, 2.0, Color32::from_gray(200));
                painter.line_segment(
                    [r.left_bottom(), r.right_top()],
                    egui::Stroke::new(1.5, Color32::RED),
                );
            } else {
                painter.rect_filled(r, 2.0, c);
            }
            painter.rect_stroke(
                r,
                2.0,
                egui::Stroke::new(1.0, Color32::BLACK),
                egui::StrokeKind::Inside,
            );
        }
        if resp
            .on_hover_text("Primary and secondary color (X swaps)")
            .clicked()
        {
            self.perform(Action::SwapColors, &ctx);
        }
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        let doc = &self.session.doc;
        ui.horizontal(|ui| {
            match self.hover {
                Some((x, y)) => {
                    let index = doc.get(x as usize, y as usize);
                    let hex = if index < 0 {
                        "transparent".to_string()
                    } else {
                        color::rgb8_to_hex(doc.palette[index as usize])
                    };
                    ui.monospace(format!("{x},{y}  idx {index}  {hex}"));
                }
                None => {
                    ui.monospace("-");
                }
            }
            ui.separator();
            ui.monospace(format!("{}x", self.canvas.view.zoom));
            ui.separator();
            ui.monospace(format!("{}x{}", doc.width(), doc.height()));
            ui.separator();
            ui.monospace(format!("{} colors", doc.palette.len()));
            if let Some(info) = &self.tiles_info {
                ui.separator();
                match info {
                    Ok(Some((n, changed))) => {
                        ui.monospace(format!("{n} tiles")).on_hover_text(format!(
                            "Saving re-extracts the tileset; de-duplication changes {changed} \
                                 pixel(s) of the image."
                        ));
                    }
                    Ok(None) => {}
                    Err(e) => {
                        ui.colored_label(ui.visuals().error_fg_color, format!("tiles: {e}"));
                    }
                }
            }
            if self.session.is_dirty() {
                ui.separator();
                ui.strong("unsaved");
            }
            if let Some(m) = &self.message {
                ui.separator();
                if m.error {
                    ui.colored_label(ui.visuals().error_fg_color, &m.text);
                } else {
                    ui.label(&m.text);
                }
            }
        });
    }

    /// One frame of the whole window (also what the GUI tests drive).
    pub fn show(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        if ctx.input(|i| i.viewport().close_requested())
            && self.session.is_dirty()
            && !self.allow_close
        {
            ctx.send_viewport_cmd(ViewportCommand::CancelClose);
            self.dialog = Some(Dialog::Unsaved);
        }
        self.shortcuts(&ctx);
        if std::mem::take(&mut self.save_as_pending) {
            self.save_as();
        }
        self.update_tiles();

        Panel::top("pixelforge-menu").show(ui, |ui| self.menu_bar(ui));
        Panel::bottom("pixelforge-status").show(ui, |ui| self.status_bar(ui));
        Panel::left("pixelforge-tools")
            .resizable(false)
            .exact_size(92.0)
            .show(ui, |ui| self.toolbar(ui));
        let mut palette_action = None;
        Panel::right("pixelforge-palette")
            .default_size(220.0)
            .show(ui, |ui| {
                palette_action = palette_panel::show(ui, &self.session, &self.tools);
            });
        let mut input = CanvasInput::default();
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let brush = if self.tools.tool.uses_brush() {
                self.tools.brush
            } else {
                1
            };
            let tile_size = self
                .session
                .doc
                .origin
                .config
                .tileset
                .then_some(self.session.doc.origin.config.tile_size as usize);
            input = self.canvas.show(ui, brush, tile_size);
        });
        self.hover = input.hover;
        if self.dialog.is_none() {
            for event in input.events {
                let effect =
                    self.tools
                        .pointer(event, &self.session.doc, self.session.selection.as_ref());
                self.effect(effect);
            }
        }
        if let Some(action) = palette_action {
            self.palette_action(action);
        }
        if let Some(dialog) = self.dialog.as_mut() {
            if let Some(outcome) = dialogs::show(&ctx, dialog) {
                self.dialog_outcome(outcome, &ctx);
            }
        }
        self.update_title(&ctx);
    }

    /// The default save target's name, for messages.
    pub fn image_path(&self) -> PathBuf {
        self.session.image_path()
    }
}

impl eframe::App for EditorApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.show(ui);
    }
}
