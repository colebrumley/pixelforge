//! The editor window (docs/editor.md, Section 3.12): menus, the toolbar, the canvas, the
//! palette, adjust, tiles and source panels and the status bar around one [`Session`].
//!
//! Every command, whether from a menu, a toolbar button or a shortcut, is an [`Action`] run
//! by [`EditorApp::perform`]; the shortcut table is [`SHORTCUTS`]. Canvas pointer events go
//! to the [`ToolBox`], whose [`Effect`]s become ops on the session. A change reports the part
//! of the image it touched, and the canvas re-uploads only that.
//!
//! The adjust panel's preview (R15) is the change list of the current slider values drawn on
//! the canvas, not applied; it is recomputed when the sliders, the selection or the document
//! change, on the worker thread for selections above [`ADJUST_SYNC_PIXELS`] (the canvas keeps
//! the latest finished preview meanwhile). Apply recomputes it and applies it as one op.
//!
//! Adding a command: an `Action` variant, its arms in `perform` and `Action::describe`, and
//! its menu entry and shortcut. Adding a panel: a `Panel` in [`EditorApp::show`] before the central canvas.

use eframe::egui;
use std::path::PathBuf;
use std::sync::Arc;

use egui::{Color32, Key, KeyboardShortcut, Modifiers, Panel, Ui, ViewportCommand};

use super::canvas::{index_color, Canvas, CanvasInput, Overlay};
use super::dialogs::{self, ColorDialog, Dialog, Outcome};
use super::palette_panel::{self, PaletteAction};
use super::panels::{self, AdjustRequest};
use super::tools::{Effect, Tool, ToolBox, BRUSH_SIZES};
use super::worker::Worker;
use crate::color;
use crate::config::Config;
use crate::editor::adjust::{self, Adjust, AdjustInput, Cleanup};
use crate::editor::tools::{self as edit_tools, Clip};
use crate::editor::{
    palette, Dirty, Document, EditError, Frames, Op, PixelChange, Selection, Session, TilesSummary,
};
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
    /// Esc: drop the stroke in progress, or else clear the selection.
    Cancel,
    SelectAll,
    SelectNone,
    /// Copy the selection (or the image) to the editor's clipboard and, as an RGBA image, to
    /// the system clipboard (R30).
    Copy,
    /// Paste at the selection's top-left corner (or the image's), as a new selection: the
    /// system clipboard's image snapped to the palette, or the editor's own copy with its
    /// exact indices when the image is that copy.
    Paste,
    /// Ramp shift (R16): every selected pixel one step darker or lighter.
    RampDarker,
    RampLighter,
    /// Cleanup on the selection (R17).
    RemoveOrphans,
    FixJaggies,
    /// Open the replace color dialog (R14).
    ReplaceColor,
    /// Open the merge entries dialog (R9).
    MergeEntries,
    /// Open the ordered dither dialog (R18).
    Dither,
    /// Apply or drop the adjust preview (R15).
    AdjustApply,
    AdjustCancel,
    /// The diff overlay (R23).
    ToggleDiff,
    /// The aligned source over the canvas (R22).
    ToggleSourceOverlay,
    /// The source reference panel (R22).
    ToggleSourcePanel,
    /// Draw the image the tilemap draws instead of the document (R24).
    ToggleReconstructed,
    /// Per-tile editing (Section 3.9): a tool's edit also reaches every identical instance
    /// of the tiles it touched.
    ToggleTileMode,
    /// The keyboard shortcuts window.
    Shortcuts,
    /// The next or previous frame of a batch (`batch --edit`, R4).
    NextFrame,
    PrevFrame,
}

impl Action {
    /// What the action does, for the shortcuts sheet.
    pub fn describe(self) -> String {
        let text = match self {
            Action::Save => "Save",
            Action::SaveAs => "Save as",
            Action::Close => "Close the window",
            Action::Undo => "Undo",
            Action::Redo => "Redo",
            Action::Tool(tool) => return format!("{} tool", tool.label()),
            Action::BrushSize(n) => return format!("Brush size {n}"),
            Action::BrushSmaller => "Smaller brush",
            Action::BrushLarger => "Larger brush",
            Action::SwapColors => "Swap primary and secondary color",
            Action::PickEntry(n) => return format!("Palette entry {n} as primary color"),
            Action::ZoomIn => "Zoom in",
            Action::ZoomOut => "Zoom out",
            Action::ZoomFit => "Zoom to fit",
            Action::ToggleGrid => "Pixel grid",
            Action::ToggleTileGrid => "Tile grid",
            Action::Cancel => "Cancel the drag, or clear the selection",
            Action::SelectAll => "Select all",
            Action::SelectNone => "Select none",
            Action::Copy => "Copy the selection",
            Action::Paste => "Paste as a new selection",
            Action::RampDarker => "Ramp shift darker",
            Action::RampLighter => "Ramp shift lighter",
            Action::RemoveOrphans => "Remove orphan pixels",
            Action::FixJaggies => "Fix jaggies",
            Action::ReplaceColor => "Replace color",
            Action::MergeEntries => "Merge palette entries",
            Action::Dither => "Ordered dither",
            Action::AdjustApply => "Apply the adjustment",
            Action::AdjustCancel => "Cancel the adjustment",
            Action::ToggleDiff => "Diff overlay",
            Action::ToggleSourceOverlay => "Source overlay",
            Action::ToggleSourcePanel => "Source panel",
            Action::ToggleReconstructed => "Show what the tilemap draws",
            Action::ToggleTileMode => "Per-tile editing (every instance of a tile)",
            Action::Shortcuts => "Keyboard shortcuts",
            Action::NextFrame => "Next frame (batch)",
            Action::PrevFrame => "Previous frame (batch)",
        };
        text.to_string()
    }
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
    (key(CMD, Key::A), Action::SelectAll),
    (key(CMD, Key::D), Action::SelectNone),
    (key(CMD, Key::C), Action::Copy),
    (key(CMD, Key::V), Action::Paste),
    (key(NONE, Key::B), Action::Tool(Tool::Pencil)),
    (key(NONE, Key::E), Action::Tool(Tool::Eraser)),
    (key(NONE, Key::I), Action::Tool(Tool::Eyedropper)),
    (key(NONE, Key::G), Action::Tool(Tool::Fill)),
    (key(NONE, Key::L), Action::Tool(Tool::Line)),
    (key(NONE, Key::R), Action::Tool(Tool::Rectangle)),
    (key(NONE, Key::M), Action::Tool(Tool::Select)),
    (key(NONE, Key::X), Action::SwapColors),
    (key(NONE, Key::Comma), Action::RampDarker),
    (key(NONE, Key::Period), Action::RampLighter),
    (key(NONE, Key::D), Action::ToggleDiff),
    (key(NONE, Key::Tab), Action::ToggleSourceOverlay),
    (key(NONE, Key::T), Action::ToggleTileMode),
    (key(NONE, Key::ArrowRight), Action::NextFrame),
    (key(NONE, Key::ArrowLeft), Action::PrevFrame),
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

/// The shortcuts sheet's rows: every action of [`SHORTCUTS`] once, in table order, with all
/// of its keys.
pub fn shortcut_sheet(ctx: &egui::Context) -> Vec<(String, String)> {
    let mut rows: Vec<(Action, Vec<String>)> = Vec::new();
    for (shortcut, action) in SHORTCUTS {
        let keys = ctx.format_shortcut(shortcut);
        match rows.iter_mut().find(|(a, _)| a == action) {
            Some((_, all)) => all.push(keys),
            None => rows.push((*action, vec![keys])),
        }
    }
    rows.into_iter()
        .map(|(action, keys)| (keys.join(", "), action.describe()))
        .collect()
}

/// The first shortcut of `action`, for the menus.
fn shortcut_of(action: Action) -> Option<&'static KeyboardShortcut> {
    SHORTCUTS.iter().find(|(_, a)| *a == action).map(|(s, _)| s)
}

/// Inputs the canvas handles itself, listed in the shortcuts sheet after [`SHORTCUTS`].
pub const POINTER_HELP: &[(&str, &str)] = &[
    ("Alt+click", "Pick the color under the pointer (any tool)"),
    ("Shift+fill", "Fill every pixel of that color"),
    ("Shift+rectangle", "Filled rectangle"),
    ("Right button", "Paint with the secondary color"),
    ("Space+drag, middle drag, scroll", "Pan"),
    ("Cmd/Ctrl+scroll, pinch", "Zoom"),
];

/// The start of the text copying puts on the system clipboard when the editor has no access
/// to it as an image (see [`EditorApp::enable_os_clipboard`]), so that the window backend
/// reports Cmd/Ctrl+V; pasting other text does nothing.
pub const CLIP_MARKER: &str = "pixelforge selection";

/// How far (per channel) the system clipboard's image may be from the editor's own copy and
/// still count as that copy: the system may convert colors on the way (color management).
pub const CLIP_MATCH_TOLERANCE: u8 = 8;

/// The tile re-extraction job of the worker: a snapshot of the image and LAB palette.
type TilesJob = (IndexImage, Vec<f64>, Config);
/// Its result, `None` without a tileset.
type TilesResult = Result<Option<TilesSummary>, String>;

/// The adjust preview job: the document revision and selection it is for, the snapshot and
/// the slider values.
type AdjustJob = (u64, Option<Selection>, Arc<AdjustInput>, Adjust);
/// Its result: the job's revision, selection and values with the change list.
type AdjustResult = (u64, Option<Selection>, Adjust, Vec<PixelChange>);

/// What the dither preview was computed for: the revision, the selection, entries A and B
/// and the density.
type DitherKey = (u64, Option<Selection>, usize, usize, usize);

/// Selections up to this many pixels preview within the frame; larger ones on the worker
/// thread (N5).
pub const ADJUST_SYNC_PIXELS: usize = 200_000;

/// The adjust panel's state (R15): the sliders and the preview drawn on the canvas.
pub struct AdjustState {
    pub params: Adjust,
    /// The changes the canvas shows as the preview (not applied).
    preview: Vec<PixelChange>,
    /// The revision, selection and values the preview was last requested for.
    requested: Option<(u64, Option<Selection>, Adjust)>,
    /// The snapshot of the last revision and selection.
    snapshot: Option<(u64, Option<Selection>, Arc<AdjustInput>)>,
    worker: Worker<AdjustJob, AdjustResult>,
}

impl AdjustState {
    fn new(ctx: &egui::Context) -> AdjustState {
        let ctx = ctx.clone();
        AdjustState {
            params: Adjust::default(),
            preview: Vec::new(),
            requested: None,
            snapshot: None,
            worker: Worker::spawn(
                "pixelforge-adjust",
                |(revision, sel, input, params): AdjustJob| {
                    let changes = adjust::adjust_input(&input, &params);
                    (revision, sel, params, changes)
                },
                move || ctx.request_repaint(),
            ),
        }
    }

    /// The identity of the current "only colors in selection" choice.
    fn identity(&self) -> Adjust {
        Adjust {
            only_selection_colors: self.params.only_selection_colors,
            ..Adjust::default()
        }
    }

    /// Whether the sliders are away from the identity.
    pub fn active(&self) -> bool {
        self.params != self.identity()
    }

    /// The changes the canvas shows as the preview.
    pub fn preview(&self) -> &[PixelChange] {
        &self.preview
    }

    /// Whether a preview is being computed on the worker.
    pub fn busy(&self) -> bool {
        self.worker.busy()
    }
}

/// A status-bar message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub text: String,
    pub error: bool,
}

/// The editor application: one session (the current frame's, with a batch) and the state of
/// its window.
pub struct EditorApp {
    pub session: Session,
    /// The other frames of a batch (`batch --edit`); `None` for one image.
    pub frames: Option<Frames>,
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
    /// Draw the image the tilemap draws instead of the document (R24).
    pub show_reconstructed: bool,
    /// Per-tile editing: the tools' edits are copied to every identical tile instance
    /// (`editor::tools::mirror_tiles`; only with `config.tileset`).
    pub tile_mode: bool,
    pub adjust: AdjustState,
    /// The dither dialog's preview: the revision, selection and values it shows, and its
    /// changes (drawn on the canvas, not applied).
    dither_preview: Option<(DitherKey, Vec<PixelChange>)>,
    /// The editor's clipboard (R30).
    pub clipboard: Option<Clip>,
    /// The system clipboard (`None` unless [`enable_os_clipboard`](Self::enable_os_clipboard)
    /// was called, as the window does; tests leave the user's clipboard alone).
    os_clipboard: Option<arboard::Clipboard>,
    /// The RGBA image the last copy put on the system clipboard.
    clip_image: Option<Vec<u8>>,
    /// A paste was triggered by the press of Cmd/Ctrl+V, so its release does not paste again
    /// (the backend reports only the release when the clipboard holds no text).
    paste_pressed: bool,
    /// The source reference panel is shown (when there is a source).
    pub show_source_panel: bool,
    /// The keyboard shortcuts window is open.
    pub show_shortcuts: bool,
    /// Text for the system clipboard, set when copying (see [`CLIP_MARKER`]).
    copied_text: Option<String>,
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
            frames: None,
            tools,
            canvas,
            dialog: None,
            message: None,
            hover: None,
            tiles,
            tiles_revision: None,
            tiles_info: None,
            show_reconstructed: false,
            tile_mode: false,
            adjust: AdjustState::new(ctx),
            dither_preview: None,
            clipboard: None,
            os_clipboard: None,
            clip_image: None,
            paste_pressed: false,
            show_source_panel: true,
            show_shortcuts: false,
            copied_text: None,
            allow_close: false,
            title: String::new(),
            save_as_pending: false,
        }
    }

    /// The editor on the frames of a batch, the first one shown (`batch --edit`); `None`
    /// without sessions. Palette edits apply to every frame while they share the palette.
    pub fn with_frames(sessions: Vec<Session>, ctx: &egui::Context) -> Option<EditorApp> {
        let (frames, first) = Frames::new(sessions)?;
        let mut app = EditorApp::new(first, ctx);
        if frames.len() > 1 {
            app.frames = Some(frames);
        }
        Some(app)
    }

    /// Whether any frame has unsaved changes.
    pub fn is_dirty(&self) -> bool {
        match &self.frames {
            Some(frames) => !frames.dirty(&self.session).is_empty(),
            None => self.session.is_dirty(),
        }
    }

    /// Show frame `to` of a batch: the tools, the canvas (with its view and overlays), the
    /// tiles and the adjust preview move to that frame's document; the selection follows
    /// when it fits. `false` without such a frame.
    pub fn switch_frame(&mut self, to: usize) -> bool {
        let Some(frames) = self.frames.as_ref() else {
            return false;
        };
        if to >= frames.len() || to == frames.current() {
            return false;
        }
        let cancelled = self.tools.cancel();
        self.effect(cancelled);
        self.finish_adjust(false);
        let sel = self.session.selection;
        let frames = self.frames.as_mut().expect("checked above");
        frames.switch(&mut self.session, to);
        let (n, len) = (frames.current() + 1, frames.len());
        let doc = &self.session.doc;
        if let Some(s) = sel.filter(|s| s.x + s.w <= doc.width() && s.y + s.h <= doc.height()) {
            self.session.selection = Some(s);
        }
        self.canvas = self.canvas.switched(doc);
        self.tools.fit_palette(doc.palette.len());
        self.tiles_info = None;
        self.tiles_revision = None;
        if let Some(worker) = self.tiles.as_mut() {
            // Submitted now, so a result for the previous frame still running is dropped.
            worker.submit((
                doc.indices.clone(),
                doc.palette_lab.clone(),
                doc.origin.config.clone(),
            ));
            self.tiles_revision = Some(self.session.revision());
        }
        self.adjust.requested = None;
        self.adjust.snapshot = None;
        self.dither_preview = None;
        let name = self
            .session
            .image_path()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.info(format!("frame {n}/{len}: {name}"));
        true
    }

    /// Apply the palette op `build` makes from a document: to every frame of a batch that
    /// shares its palette, else to this document. `true` when it was applied.
    fn apply_palette(&mut self, build: impl Fn(&Document) -> Result<Op, EditError>) -> bool {
        let result = match self.frames.as_mut() {
            Some(frames) => frames.apply_palette(&mut self.session, build),
            None => build(&self.session.doc).and_then(|op| self.session.apply(op)),
        };
        match result {
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
            Effect::Repreview { old, new } => {
                self.canvas.restore(&self.session.doc, &old);
                self.canvas.preview(&self.session.doc, &new);
            }
            Effect::Commit(op) => {
                let op = match op {
                    Op::Pixels { tool, changes } if self.tile_mode => {
                        let sel = self.session.selection;
                        let changes =
                            edit_tools::mirror_tiles(&self.session.doc, sel.as_ref(), &changes);
                        Op::Pixels { tool, changes }
                    }
                    op => op,
                };
                self.commit(op);
            }
            Effect::Move(op, moved) => {
                // The selection follows its contents (unless the move was refused).
                let empty = op.is_empty();
                if self.commit(op) || empty {
                    self.session.selection = moved;
                }
            }
            Effect::Select(sel) => self.session.selection = sel,
            Effect::Pick(button, index) => self.tools.set_color(button, index),
        }
    }

    /// Apply an op whose pixels may be previewed on the canvas; a refused or empty op drops
    /// its preview. `true` when it was applied.
    fn commit(&mut self, op: Op) -> bool {
        let touched: Vec<_> = match &op {
            Op::Pixels { changes, .. } => changes.clone(),
            _ => Vec::new(),
        };
        let applied = self.apply(op);
        if !applied {
            self.canvas.restore(&self.session.doc, &touched);
        }
        applied
    }

    /// Apply the change list of a command on the selection as one op named `tool`.
    fn apply_changes(&mut self, tool: &str, changes: Vec<PixelChange>, what: &str) {
        if changes.is_empty() {
            self.info(format!("{what}: nothing to change"));
        } else if self.apply(Op::pixels(tool, changes)) {
            let n = match self.session.history.next_undo() {
                Some(Op::Pixels { changes, .. }) => changes.len(),
                _ => 0,
            };
            self.info(format!(
                "{what}: {n} pixel{} changed",
                if n == 1 { "" } else { "s" }
            ));
        }
    }

    /// Use the system clipboard for images (R30). Without it, copy and paste stay inside
    /// the editor. `false` when the clipboard cannot be opened (no display server).
    pub fn enable_os_clipboard(&mut self) -> bool {
        self.os_clipboard = arboard::Clipboard::new().ok();
        self.os_clipboard.is_some()
    }

    /// Copy the selection (or the image) to the editor's clipboard, and as an RGBA image to
    /// the system clipboard when the editor has it.
    pub fn copy(&mut self) {
        let sel = self.session.selection;
        let clip = edit_tools::copy(&self.session.doc, sel.as_ref());
        let (w, h) = (clip.w, clip.h);
        let rgba = edit_tools::clip_rgba(&self.session.doc, &clip);
        let image = arboard::ImageData {
            width: w,
            height: h,
            bytes: std::borrow::Cow::Borrowed(&rgba),
        };
        let os = self
            .os_clipboard
            .as_mut()
            .map(|c| c.set_image(image).is_ok());
        match os {
            Some(true) => {
                self.info(format!("copied {w}x{h} pixels"));
                self.clip_image = Some(rgba);
            }
            failed => {
                if failed.is_some() {
                    self.info(format!(
                        "copied {w}x{h} pixels (inside the editor: the system clipboard \
                         refused the image)"
                    ));
                } else {
                    self.info(format!("copied {w}x{h} pixels"));
                }
                self.clip_image = None;
                self.copied_text = Some(format!("{CLIP_MARKER} {w}x{h}"));
            }
        }
        self.clipboard = Some(clip);
    }

    /// What a paste places: the system clipboard's image (the editor's own copy when it is
    /// that image, else snapped to the palette), or the editor's copy. `None` with nothing
    /// to paste. The flag says whether the image was snapped.
    fn clip_to_paste(&mut self) -> Option<(Clip, bool)> {
        let image = self.os_clipboard.as_mut().and_then(|c| c.get_image().ok());
        let Some(image) = image else {
            return self.clipboard.clone().map(|c| (c, false));
        };
        let own = match (&self.clipboard, &self.clip_image) {
            (Some(clip), Some(rgba)) => {
                clip.w == image.width
                    && clip.h == image.height
                    && rgba.len() == image.bytes.len()
                    && rgba
                        .iter()
                        .zip(image.bytes.iter())
                        .all(|(a, b)| a.abs_diff(*b) <= CLIP_MATCH_TOLERANCE)
            }
            _ => false,
        };
        if own {
            return self.clipboard.clone().map(|c| (c, false));
        }
        edit_tools::snap_rgba(&self.session.doc, &image.bytes, image.width, image.height)
            .map(|c| (c, true))
    }

    /// Paste at the selection's top-left corner (or the image's), clipped to the canvas; the
    /// pasted rectangle becomes the selection, so it can be moved with the selection tool.
    pub fn paste(&mut self) {
        let Some((clip, snapped)) = self.clip_to_paste() else {
            return self.info("nothing to paste");
        };
        let at = self
            .session
            .selection
            .map_or((0, 0), |s| (s.x as i64, s.y as i64));
        let (changes, placed) = edit_tools::paste(&self.session.doc, &clip, at);
        let op = Op::pixels("paste", changes);
        if op.is_empty() || self.apply(op) {
            self.session.selection = placed;
            self.tools.tool = Tool::Select;
            if snapped {
                self.info(format!(
                    "pasted {}x{} pixels from the system clipboard, snapped to the palette",
                    clip.w, clip.h
                ));
            }
        }
    }

    /// Recompute the adjust preview when the sliders, the selection or the document
    /// changed, and pick up the worker's preview.
    fn update_adjust(&mut self) {
        let revision = self.session.revision();
        let sel = self.session.selection;
        let params = self.adjust.params;
        let key = (revision, sel, params);
        if let Some((rev, s, p, changes)) = self.adjust.worker.poll() {
            if (rev, s, p) == key && self.adjust.active() {
                self.show_adjust_preview(changes);
            }
        }
        if self.adjust.requested == Some(key) {
            return;
        }
        self.adjust.requested = Some(key);
        if !self.adjust.active() {
            self.show_adjust_preview(Vec::new());
            return;
        }
        let input = match &self.adjust.snapshot {
            Some((r, s, input)) if (*r, *s) == (revision, sel) => input.clone(),
            _ => {
                let input = Arc::new(AdjustInput::new(&self.session.doc, sel.as_ref()));
                self.adjust.snapshot = Some((revision, sel, input.clone()));
                input
            }
        };
        if input.len() <= ADJUST_SYNC_PIXELS {
            let changes = adjust::adjust_input(&input, &params);
            self.show_adjust_preview(changes);
        } else {
            // The canvas keeps the latest finished preview meanwhile.
            self.adjust.worker.submit((revision, sel, input, params));
        }
    }

    fn show_adjust_preview(&mut self, changes: Vec<PixelChange>) {
        let old = std::mem::replace(&mut self.adjust.preview, changes);
        self.canvas.restore(&self.session.doc, &old);
        self.canvas.preview(&self.session.doc, &self.adjust.preview);
    }

    /// Apply (`apply`) or drop the adjust preview, and reset the sliders.
    fn finish_adjust(&mut self, apply: bool) {
        let params = self.adjust.params;
        let active = self.adjust.active();
        self.adjust.params = self.adjust.identity();
        self.show_adjust_preview(Vec::new());
        self.adjust.requested = None;
        if apply && active {
            // Computed afresh: the preview may be from the worker or a frame old.
            let sel = self.session.selection;
            let changes = adjust::adjust(&self.session.doc, sel.as_ref(), &params);
            self.apply_changes("adjust", changes, "adjust");
        }
    }

    /// Show the dither dialog's result on the canvas while it is open, recomputed when its
    /// values, the selection or the document change; drop it when the dialog closes.
    fn update_dither(&mut self) {
        let want = match &self.dialog {
            Some(Dialog::Dither { a, b, density, .. }) => Some((
                self.session.revision(),
                self.session.selection,
                *a,
                *b,
                *density,
            )),
            _ => None,
        };
        if want == self.dither_preview.as_ref().map(|(k, _)| *k) {
            return;
        }
        if let Some((_, old)) = self.dither_preview.take() {
            self.canvas.restore(&self.session.doc, &old);
        }
        if let Some(key) = want {
            let (_, sel, a, b, density) = key;
            let changes = adjust::dither(&self.session.doc, sel.as_ref(), a, b, density);
            self.canvas.preview(&self.session.doc, &changes);
            self.dither_preview = Some((key, changes));
        }
    }

    /// Hand the reconstruction to the canvas while it is shown.
    fn sync_reconstructed(&mut self) {
        let rendered = match (&self.tiles_info, self.show_reconstructed) {
            (Some(Ok(Some(t))), true) => Some(&t.rendered),
            _ => None,
        };
        self.canvas.set_reconstructed(&self.session.doc, rendered);
    }

    /// Run a command.
    pub fn perform(&mut self, action: Action, ctx: &egui::Context) {
        match action {
            Action::Save => {
                self.save();
            }
            Action::SaveAs if self.frames.is_some() => {
                self.error("a batch saves every frame under its own name (no Save As)");
            }
            Action::SaveAs => self.save_as_pending = true,
            Action::Close => ctx.send_viewport_cmd(ViewportCommand::Close),
            Action::Undo | Action::Redo => {
                let cancelled = self.tools.cancel();
                self.effect(cancelled);
                let undo = action == Action::Undo;
                let stepped = match (self.frames.as_mut(), undo) {
                    (Some(frames), true) => frames.undo(&mut self.session),
                    (Some(frames), false) => frames.redo(&mut self.session),
                    (None, true) => Ok(self.session.undo()),
                    (None, false) => Ok(self.session.redo()),
                };
                match stepped {
                    Ok(Some(dirty)) => self.changed(dirty),
                    Ok(None) => {}
                    Err(e) => self.error(e.0),
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
                if self.tools.dragging() {
                    let cancelled = self.tools.cancel();
                    self.effect(cancelled);
                } else {
                    self.session.selection = None;
                }
            }
            Action::SelectAll => {
                let doc = &self.session.doc;
                self.session.selection = Some(Selection::all(doc.width(), doc.height()));
            }
            Action::SelectNone => self.session.selection = None,
            Action::Copy => self.copy(),
            Action::Paste => {
                let cancelled = self.tools.cancel();
                self.effect(cancelled);
                self.paste();
            }
            Action::RampDarker | Action::RampLighter => {
                let lighter = action == Action::RampLighter;
                let sel = self.session.selection;
                let changes = adjust::ramp_shift(&self.session.doc, sel.as_ref(), lighter);
                let what = if lighter {
                    "ramp shift lighter"
                } else {
                    "ramp shift darker"
                };
                self.apply_changes("ramp_shift", changes, what);
            }
            Action::RemoveOrphans | Action::FixJaggies => {
                let (pass, tool, what) = if action == Action::RemoveOrphans {
                    (Cleanup::Orphans, "orphans", "orphan removal")
                } else {
                    (Cleanup::Jaggies, "jaggies", "jaggy fix")
                };
                let sel = self.session.selection;
                let changes = adjust::cleanup(&self.session.doc, sel.as_ref(), pass);
                self.apply_changes(tool, changes, what);
            }
            Action::ReplaceColor => {
                self.dialog = Some(Dialog::Replace {
                    from: self.tools.primary,
                    to: self.tools.secondary,
                    in_selection: self.session.selection.is_some(),
                    palette: self.session.doc.palette.clone(),
                });
            }
            Action::MergeEntries => {
                let k = self.session.doc.palette.len();
                if k < 2 {
                    return self.error("merging needs two palette entries");
                }
                let entry = |c: i64, fallback: usize| {
                    usize::try_from(c)
                        .ok()
                        .filter(|&i| i < k)
                        .unwrap_or(fallback)
                };
                let from = entry(self.tools.primary, 0);
                let into = entry(self.tools.secondary, usize::from(from == 0));
                self.dialog = Some(Dialog::Merge {
                    from,
                    into,
                    palette: self.session.doc.palette.clone(),
                });
            }
            Action::Dither => {
                let k = self.session.doc.palette.len();
                if k == 0 {
                    return self.error("dithering needs a palette entry");
                }
                self.finish_adjust(false);
                let entry = |c: i64, fallback: usize| {
                    usize::try_from(c)
                        .ok()
                        .filter(|&i| i < k)
                        .unwrap_or(fallback)
                };
                let a = entry(self.tools.primary, 0);
                let b = entry(self.tools.secondary, usize::from(a == 0 && k > 1));
                self.dialog = Some(Dialog::Dither {
                    a,
                    b,
                    density: adjust::DITHER_LEVELS / 2,
                    in_selection: self.session.selection.is_some(),
                    palette: self.session.doc.palette.clone(),
                });
            }
            Action::AdjustApply => self.finish_adjust(true),
            Action::AdjustCancel => self.finish_adjust(false),
            Action::ToggleDiff => {
                let on = !self.canvas.diff_shown();
                self.canvas.set_diff(&self.session.doc, on);
            }
            Action::ToggleSourceOverlay => {
                let aligned = self
                    .session
                    .doc
                    .origin
                    .source
                    .as_ref()
                    .is_some_and(|s| s.factor.is_some());
                if aligned {
                    self.canvas.show_source = !self.canvas.show_source;
                } else {
                    self.info("no aligned source to overlay (its geometry is not known)");
                }
            }
            Action::ToggleSourcePanel => self.show_source_panel = !self.show_source_panel,
            Action::ToggleReconstructed => {
                self.show_reconstructed = !self.show_reconstructed;
                self.sync_reconstructed();
            }
            Action::ToggleTileMode => {
                if self.session.doc.origin.config.tileset {
                    self.tile_mode = !self.tile_mode;
                    self.info(if self.tile_mode {
                        "per-tile editing: edits reach every instance of a tile"
                    } else {
                        "per-tile editing off"
                    });
                } else {
                    self.info("per-tile editing needs a tileset (config.tileset)");
                }
            }
            Action::Shortcuts => self.show_shortcuts = !self.show_shortcuts,
            Action::NextFrame | Action::PrevFrame => match self.frames.as_ref() {
                Some(frames) => {
                    let (n, len) = (frames.current(), frames.len());
                    let to = if action == Action::NextFrame {
                        (n + 1) % len
                    } else {
                        (n + len - 1) % len
                    };
                    self.switch_frame(to);
                }
                None => self.info("one image: frames are for batch --edit"),
            },
        }
    }

    /// Save under the session's prefix (with a batch: every frame with unsaved changes,
    /// so the frames keep one palette); `true` on success.
    pub fn save(&mut self) -> bool {
        let dir = self
            .session
            .prefix
            .parent()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let result = match self.frames.as_mut() {
            Some(frames) => frames.save(&mut self.session).map(|saved| {
                let files: usize = saved.iter().map(|s| s.outputs.len()).sum();
                let n = saved.len();
                format!(
                    "saved {n} frame{} ({files} files) to {dir}",
                    if n == 1 { "" } else { "s" }
                )
            }),
            None => self
                .session
                .save()
                .map(|saved| format!("saved {} files to {dir}", saved.outputs.len())),
        };
        match result {
            Ok(message) => {
                self.info(message);
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

    fn palette_action(&mut self, action: PaletteAction, ctx: &egui::Context) {
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
            PaletteAction::Merge => self.perform(Action::MergeEntries, ctx),
            PaletteAction::Replace => self.perform(Action::ReplaceColor, ctx),
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
        match palette::nearest_other(&self.session.doc, index) {
            Some(target) => {
                self.remove_with(|d| palette::remove_entry(d, index), index, target);
            }
            None => self.error("the last palette entry cannot be removed"),
        }
    }

    fn merge_entries(&mut self, from: usize, into: usize) {
        self.remove_with(|d| palette::merge_entries(d, from, into), from, into);
    }

    /// Apply the op `build` makes, which removes entry `index` and moves its pixels to
    /// `target`.
    fn remove_with(
        &mut self,
        build: impl Fn(&Document) -> Result<Op, EditError>,
        index: usize,
        target: usize,
    ) {
        // The colors keep pointing at the same entries after the shift; one on the removed
        // entry follows its pixels.
        let (index, target) = (index as i64, target as i64);
        let follow = |c: i64| {
            let c = if c == index { target } else { c };
            if c > index {
                c - 1
            } else {
                c
            }
        };
        let colors = (follow(self.tools.primary), follow(self.tools.secondary));
        if self.apply_palette(build) {
            (self.tools.primary, self.tools.secondary) = colors;
        }
    }

    fn dialog_outcome(&mut self, outcome: Outcome, ctx: &egui::Context) {
        self.dialog = None;
        match outcome {
            Outcome::Cancel => {}
            Outcome::Color(Some(i), rgb) => {
                if self.session.doc.palette[i] != rgb {
                    self.apply_palette(|d| palette::set_color(d, i, rgb));
                }
            }
            Outcome::Color(None, rgb) => {
                if self.apply_palette(|d| palette::add_color(d, rgb)) {
                    self.tools.primary = self.session.doc.palette.len() as i64 - 1;
                }
            }
            Outcome::Remove(i) => self.remove_entry(i),
            Outcome::Replace(from, to) => {
                let sel = self.session.selection;
                let changes = edit_tools::replace_color(&self.session.doc, sel.as_ref(), from, to);
                self.apply_changes("replace_color", changes, "replace color");
            }
            Outcome::Merge(from, into) => self.merge_entries(from, into),
            Outcome::Dither(a, b, density) => {
                // The preview goes first: it is drawn over the pixels the op changes.
                self.update_dither();
                let sel = self.session.selection;
                let changes = adjust::dither(&self.session.doc, sel.as_ref(), a, b, density);
                self.apply_changes("dither", changes, "dither");
            }
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
            let mut pressed: Vec<Action> = SHORTCUTS
                .iter()
                .filter(|(s, _)| i.consume_shortcut(s))
                .map(|(_, a)| *a)
                .collect();
            let mut paste_pressed = pressed.contains(&Action::Paste);
            // The window backend turns Cmd/Ctrl+C and V into clipboard events instead of
            // key presses: V as a Paste event when the system clipboard holds text (other
            // text than CLIP_MARKER's is not pasted), and otherwise (an image) only as the
            // key's release, which pastes unless the press already did.
            i.events.retain(|e| match e {
                egui::Event::Copy => {
                    pressed.push(Action::Copy);
                    false
                }
                egui::Event::Paste(text) => {
                    if text.starts_with(CLIP_MARKER) {
                        pressed.push(Action::Paste);
                    }
                    paste_pressed = true;
                    false
                }
                egui::Event::Key {
                    key: Key::V,
                    pressed: false,
                    modifiers,
                    ..
                } if modifiers.command => {
                    if !std::mem::take(&mut paste_pressed) && !self.paste_pressed {
                        pressed.push(Action::Paste);
                    }
                    self.paste_pressed = false;
                    true
                }
                _ => true,
            });
            if paste_pressed {
                self.paste_pressed = true;
            }
            pressed
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
        let dirty = if self.is_dirty() { " *" } else { "" };
        let frame = match &self.frames {
            Some(f) => format!(" [{}/{}]", f.current() + 1, f.len()),
            None => String::new(),
        };
        let title = format!("{name}{frame}{dirty} — pixelforge");
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
        let mut arrived = false;
        if let Some(result) = worker.poll() {
            self.tiles_info = Some(result);
            arrived = true;
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
        if arrived && self.show_reconstructed {
            self.sync_reconstructed();
        }
    }

    /// Whether the tiles worker has not delivered the newest result yet.
    pub fn tiles_busy(&self) -> bool {
        self.tiles.as_ref().is_some_and(|w| w.busy())
    }

    /// The latest tiles result (R24).
    pub fn tiles_info(&self) -> Option<&TilesResult> {
        self.tiles_info.as_ref()
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

    fn toggle_item(&mut self, ui: &mut Ui, label: &str, on: bool, action: Action, enabled: bool) {
        let mut button = egui::Button::selectable(on, label);
        if let Some(s) = shortcut_of(action) {
            button = button.shortcut_text(ui.ctx().format_shortcut(s));
        }
        if ui.add_enabled(enabled, button).clicked() {
            self.perform(action, &ui.ctx().clone());
        }
    }

    fn menu_bar(&mut self, ui: &mut Ui) {
        let has_clip = self.clipboard.is_some() || self.os_clipboard.is_some();
        let k = self.session.doc.palette.len();
        let source = self.session.doc.origin.source.as_ref();
        let has_source = source.is_some();
        let aligned = source.is_some_and(|s| s.factor.is_some());
        let tileset = self.session.doc.origin.config.tileset;
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                self.menu_item(ui, "Save", Action::Save, true);
                let single = self.frames.is_none();
                self.menu_item(ui, "Save As…", Action::SaveAs, single);
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
                self.menu_item(ui, "Copy", Action::Copy, true);
                self.menu_item(ui, "Paste", Action::Paste, has_clip);
                ui.separator();
                self.menu_item(ui, "Swap colors", Action::SwapColors, true);
                self.menu_item(ui, "Replace color…", Action::ReplaceColor, true);
                self.menu_item(ui, "Merge palette entries…", Action::MergeEntries, k > 1);
            });
            ui.menu_button("View", |ui| {
                self.menu_item(ui, "Zoom in", Action::ZoomIn, true);
                self.menu_item(ui, "Zoom out", Action::ZoomOut, true);
                self.menu_item(ui, "Zoom to fit", Action::ZoomFit, true);
                ui.separator();
                let grid = self.canvas.show_grid;
                self.toggle_item(ui, "Pixel grid", grid, Action::ToggleGrid, true);
                let tiles = self.canvas.show_tile_grid;
                self.toggle_item(ui, "Tile grid", tiles, Action::ToggleTileGrid, tileset);
                let diff = self.canvas.diff_shown();
                self.toggle_item(ui, "Diff overlay", diff, Action::ToggleDiff, true);
                ui.separator();
                let panel = self.show_source_panel;
                self.toggle_item(
                    ui,
                    "Source panel",
                    panel,
                    Action::ToggleSourcePanel,
                    has_source,
                );
                let overlay = self.canvas.show_source;
                self.toggle_item(
                    ui,
                    "Source overlay",
                    overlay,
                    Action::ToggleSourceOverlay,
                    aligned,
                );
                let mode = self.tile_mode;
                self.toggle_item(
                    ui,
                    "Per-tile editing",
                    mode,
                    Action::ToggleTileMode,
                    tileset,
                );
                let rec = self.show_reconstructed;
                self.toggle_item(
                    ui,
                    "What the tilemap draws",
                    rec,
                    Action::ToggleReconstructed,
                    tileset,
                );
            });
            ui.menu_button("Select", |ui| {
                self.menu_item(ui, "All", Action::SelectAll, true);
                let any = self.session.selection.is_some();
                self.menu_item(ui, "None", Action::SelectNone, any);
            });
            ui.menu_button("Image", |ui| {
                self.menu_item(ui, "Ramp shift darker", Action::RampDarker, k > 0);
                self.menu_item(ui, "Ramp shift lighter", Action::RampLighter, k > 0);
                ui.separator();
                self.menu_item(ui, "Remove orphan pixels", Action::RemoveOrphans, k > 0);
                self.menu_item(ui, "Fix jaggies", Action::FixJaggies, true);
                ui.separator();
                self.menu_item(ui, "Dither…", Action::Dither, k > 0);
            });
            if self.frames.is_some() {
                ui.menu_button("Frame", |ui| {
                    self.menu_item(ui, "Next frame", Action::NextFrame, true);
                    self.menu_item(ui, "Previous frame", Action::PrevFrame, true);
                });
            }
            ui.menu_button("Help", |ui| {
                self.menu_item(ui, "Keyboard shortcuts", Action::Shortcuts, true);
            });
        });
    }

    /// The shortcuts sheet, generated from [`SHORTCUTS`] and [`POINTER_HELP`].
    fn shortcuts_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_shortcuts;
        egui::Window::new("Keyboard shortcuts")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_height(480.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("pixelforge-shortcuts")
                        .striped(true)
                        .num_columns(2)
                        .show(ui, |ui| {
                            for (keys, action) in shortcut_sheet(ctx) {
                                ui.monospace(keys);
                                ui.label(action);
                                ui.end_row();
                            }
                            for (keys, what) in POINTER_HELP {
                                ui.monospace(*keys);
                                ui.label(*what);
                                ui.end_row();
                            }
                        });
                });
            });
        self.show_shortcuts = open;
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
            if let Some(sel) = self.session.selection {
                ui.separator();
                ui.monospace(format!("sel {},{} {}x{}", sel.x, sel.y, sel.w, sel.h));
            }
            if let Some(info) = &self.tiles_info {
                ui.separator();
                match info {
                    Ok(Some(t)) => {
                        ui.monospace(format!("{} tiles", t.tiles))
                            .on_hover_text(format!(
                                "Saving re-extracts the tileset; de-duplication changes {} \
                                 pixel(s) of the image.",
                                t.changed
                            ));
                    }
                    Ok(None) => {}
                    Err(e) => {
                        ui.colored_label(ui.visuals().error_fg_color, format!("tiles: {e}"));
                    }
                }
            }
            if self.tile_mode {
                ui.separator();
                ui.strong("per tile").on_hover_text(
                    "Edits reach every identical instance of the tiles they touch (T)",
                );
            }
            if let Some(frames) = &self.frames {
                ui.separator();
                ui.monospace(format!("frame {}/{}", frames.current() + 1, frames.len()))
                    .on_hover_text("Left and right arrows: previous and next frame");
            }
            if self.session.is_dirty() {
                ui.separator();
                ui.strong("unsaved");
            }
            let others = self.frames.as_ref().map_or(0, |f| {
                f.dirty(&self.session)
                    .iter()
                    .filter(|&&i| i != f.current())
                    .count()
            });
            if others > 0 {
                ui.separator();
                ui.strong(format!(
                    "{others} other frame{} unsaved",
                    if others == 1 { "" } else { "s" }
                ));
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
        if ctx.input(|i| i.viewport().close_requested()) && self.is_dirty() && !self.allow_close {
            ctx.send_viewport_cmd(ViewportCommand::CancelClose);
            self.dialog = Some(Dialog::Unsaved);
        }
        self.shortcuts(&ctx);
        if let Some(text) = self.copied_text.take() {
            ctx.copy_text(text);
        }
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
        let mut adjust_request = None;
        let mut reconstructed_toggled = false;
        Panel::right("pixelforge-palette")
            .default_size(240.0)
            .show(ui, |ui| {
                Panel::bottom("pixelforge-side-tools")
                    .resizable(false)
                    .show(ui, |ui| {
                        ui.add_space(4.0);
                        let active = self.adjust.active();
                        let busy = self.adjust.busy();
                        adjust_request = panels::adjust(
                            ui,
                            &mut self.adjust.params,
                            self.session.selection.is_some(),
                            active,
                            busy,
                        );
                        if self.tiles.is_some() {
                            ui.separator();
                            let busy = self.tiles_busy();
                            reconstructed_toggled = panels::tiles(
                                ui,
                                self.tiles_info.as_ref(),
                                &mut self.show_reconstructed,
                                busy,
                            );
                        }
                        ui.add_space(4.0);
                    });
                palette_action = palette_panel::show(ui, &self.session, &self.tools);
            });
        if let (true, Some(src)) = (
            self.show_source_panel,
            self.session.doc.origin.source.as_ref(),
        ) {
            let texture = self.canvas.source_texture(&ctx, src);
            Panel::bottom("pixelforge-source")
                .resizable(true)
                .default_size(180.0)
                .min_size(80.0)
                .show(ui, |ui| {
                    panels::source(
                        ui,
                        src,
                        texture,
                        &mut self.canvas.show_source,
                        &mut self.canvas.source_opacity,
                    );
                });
        }
        if reconstructed_toggled {
            self.sync_reconstructed();
        }
        match adjust_request {
            Some(AdjustRequest::Apply) => self.perform(Action::AdjustApply, &ctx),
            Some(AdjustRequest::Cancel) => self.perform(Action::AdjustCancel, &ctx),
            None => {}
        }
        // The overlay outline follows a selection drag; otherwise it is the selection.
        let doc = &self.session.doc;
        let selection = self
            .tools
            .marquee(doc.width(), doc.height())
            .unwrap_or(self.session.selection);
        let mut input = CanvasInput::default();
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let brush = if self.tools.tool.uses_brush() {
                self.tools.brush
            } else {
                1
            };
            let config = &self.session.doc.origin.config;
            let overlay = Overlay {
                brush,
                tile_size: config.tileset.then_some(config.tile_size as usize),
                selection,
                source: self.session.doc.origin.source.as_ref(),
            };
            input = self.canvas.show(ui, &overlay);
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
            self.palette_action(action, &ctx);
        }
        self.update_adjust();
        if let Some(dialog) = self.dialog.as_mut() {
            if let Some(outcome) = dialogs::show(&ctx, dialog) {
                self.dialog_outcome(outcome, &ctx);
            }
        }
        self.update_dither();
        if self.show_shortcuts {
            self.shortcuts_window(&ctx);
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
