//! The GUI's tools: which tool is active, the brush size, the primary and secondary colors,
//! and the translation of canvas pointer events into edits (docs/editor.md, Sections 3.4 and
//! 3.12).
//!
//! The canvas reports [`Pointer`] events in image pixels; [`ToolBox::pointer`] turns each into
//! an [`Effect`] by calling the headless tools in `editor::tools`. A drag accumulates into
//! one `editor::tools::Stroke`, shown as a preview and committed as one op on release (one
//! undo step, R11). Alt+press picks the color under the pointer with any tool.
//!
//! Line and rectangle follow the pointer: each drag event replaces the shape's preview, and
//! the release commits it. The selection tool draws a marquee, or, pressed inside the
//! selection, moves its contents (one op on release, and the selection moves with them).
//!
//! Adding a tool: a [`Tool`] variant, its entries in the `Tool` methods (name, label,
//! shortcut, hint) and in [`Tool::ALL`], its arm in [`ToolBox::press`] (and in
//! [`ToolBox::drag`] and [`ToolBox::release`] when it drags), and an `Action::Tool`
//! shortcut in `app::SHORTCUTS`.

use eframe::egui;
use egui::Key;

use crate::editor::tools::{self, Stroke, MAX_BRUSH};
use crate::editor::{Document, Op, PixelChange, Selection};

/// A tool of the toolbar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tool {
    Pencil,
    Eraser,
    Eyedropper,
    Fill,
    Line,
    Rectangle,
    Select,
}

impl Tool {
    /// Every tool, in toolbar order.
    pub const ALL: [Tool; 7] = [
        Tool::Pencil,
        Tool::Eraser,
        Tool::Eyedropper,
        Tool::Fill,
        Tool::Line,
        Tool::Rectangle,
        Tool::Select,
    ];

    /// The name ops are annotated with in the edit log.
    pub fn name(self) -> &'static str {
        match self {
            Tool::Pencil => "pencil",
            Tool::Eraser => "eraser",
            Tool::Eyedropper => "eyedropper",
            Tool::Fill => "fill",
            Tool::Line => "line",
            Tool::Rectangle => "rectangle",
            Tool::Select => "move",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Tool::Pencil => "Pencil",
            Tool::Eraser => "Eraser",
            Tool::Eyedropper => "Eyedropper",
            Tool::Fill => "Fill",
            Tool::Line => "Line",
            Tool::Rectangle => "Rectangle",
            Tool::Select => "Select",
        }
    }

    /// The key that selects it (Aseprite's).
    pub fn key(self) -> Key {
        match self {
            Tool::Pencil => Key::B,
            Tool::Eraser => Key::E,
            Tool::Eyedropper => Key::I,
            Tool::Fill => Key::G,
            Tool::Line => Key::L,
            Tool::Rectangle => Key::R,
            Tool::Select => Key::M,
        }
    }

    /// One line of help for the toolbar's tooltip.
    pub fn hint(self) -> &'static str {
        match self {
            Tool::Pencil => "Left: primary color, right: secondary. Alt+click picks a color.",
            Tool::Eraser => "Paints transparency.",
            Tool::Eyedropper => "Left: pick the primary color, right: the secondary.",
            Tool::Fill => "Fill the 4-connected area; Shift fills every pixel of that color.",
            Tool::Line => "Drag a line with the brush.",
            Tool::Rectangle => "Drag a rectangle's outline; Shift fills it.",
            Tool::Select => {
                "Drag a rectangle to select (Esc clears); drag inside the selection to move \
                 its contents."
            }
        }
    }

    /// Whether the brush size applies.
    pub fn uses_brush(self) -> bool {
        matches!(self, Tool::Pencil | Tool::Eraser | Tool::Line)
    }
}

/// Which mouse button: the primary paints the primary color, the secondary the secondary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Primary,
    Secondary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Press,
    Drag,
    Release,
}

/// A pointer event on the canvas, in image pixels (possibly outside the image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pointer {
    pub phase: Phase,
    pub button: Button,
    pub at: (i64, i64),
    pub shift: bool,
    pub alt: bool,
}

/// What a pointer event does.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    None,
    /// Show these changes on the canvas without applying them (a stroke in progress).
    Preview(Vec<PixelChange>),
    /// Apply this op as one undo step.
    Commit(Op),
    /// Make this index (`-1` transparent) the button's color.
    Pick(Button, i64),
    /// Drop the preview of these changes (a stroke was cancelled).
    Cancel(Vec<PixelChange>),
    /// Replace the preview of `old` with the preview of `new` (a shape or a move following
    /// the pointer).
    Repreview {
        old: Vec<PixelChange>,
        new: Vec<PixelChange>,
    },
    /// The selection becomes this (`None` clears it).
    Select(Option<Selection>),
    /// Apply this op (moving the selection's contents), then the selection becomes this.
    Move(Op, Option<Selection>),
}

/// What a drag does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DragKind {
    /// Pencil and eraser: every event adds to the stroke.
    Paint,
    /// Line and rectangle: the shape from the press to the pointer.
    Shape,
    /// The selection tool outside the selection: a new selection.
    Marquee,
    /// The selection tool inside this selection: move its contents.
    Move(Selection),
}

/// A drag in progress.
#[derive(Clone, Debug)]
struct Drag {
    kind: DragKind,
    button: Button,
    index: i64,
    start: (i64, i64),
    last: (i64, i64),
    stroke: Stroke,
}

/// The tool state of the editor.
#[derive(Clone, Debug)]
pub struct ToolBox {
    pub tool: Tool,
    /// The square brush's side, `1..=MAX_BRUSH`.
    pub brush: usize,
    /// Palette indices; `-1` is the transparent slot.
    pub primary: i64,
    pub secondary: i64,
    drag: Option<Drag>,
}

/// The brush sizes the toolbar offers.
pub const BRUSH_SIZES: [usize; 4] = [1, 2, 4, 8];

impl ToolBox {
    /// The pencil, size 1, entry 0 as primary and entry 1 (or transparency) as secondary.
    pub fn new(palette_len: usize) -> ToolBox {
        let first = if palette_len > 0 { 0 } else { -1 };
        ToolBox {
            tool: Tool::Pencil,
            brush: 1,
            primary: first,
            secondary: if palette_len > 1 { 1 } else { -1 },
            drag: None,
        }
    }

    /// The color of `button`.
    pub fn color(&self, button: Button) -> i64 {
        match button {
            Button::Primary => self.primary,
            Button::Secondary => self.secondary,
        }
    }

    pub fn set_color(&mut self, button: Button, index: i64) {
        match button {
            Button::Primary => self.primary = index,
            Button::Secondary => self.secondary = index,
        }
    }

    pub fn swap_colors(&mut self) {
        std::mem::swap(&mut self.primary, &mut self.secondary);
    }

    /// Grow (`delta > 0`) or shrink the brush, within `1..=MAX_BRUSH`.
    pub fn resize_brush(&mut self, delta: i64) {
        self.brush = (self.brush as i64 + delta).clamp(1, MAX_BRUSH as i64) as usize;
    }

    /// Keep the colors valid after the palette changed to `palette_len` entries: an index
    /// past the end becomes the last entry (or transparency).
    pub fn fit_palette(&mut self, palette_len: usize) {
        let last = palette_len as i64 - 1;
        for c in [&mut self.primary, &mut self.secondary] {
            if *c > last {
                *c = last;
            }
        }
    }

    /// Whether a stroke is in progress.
    pub fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Drop the stroke in progress (Esc, a lost pointer).
    pub fn cancel(&mut self) -> Effect {
        match self.drag.take() {
            Some(d) => Effect::Cancel(d.stroke.changes().to_vec()),
            None => Effect::None,
        }
    }

    /// The selection outline to draw while the selection tool drags: the marquee, or the
    /// selection where its contents would land; `None` without such a drag.
    pub fn marquee(&self, width: usize, height: usize) -> Option<Option<Selection>> {
        let d = self.drag.as_ref()?;
        match d.kind {
            DragKind::Marquee => Some(Selection::from_corners(d.start, d.last, width, height)),
            DragKind::Move(sel) => {
                Some(sel.translated(d.last.0 - d.start.0, d.last.1 - d.start.1, width, height))
            }
            _ => None,
        }
    }

    /// Handle one pointer event against the document as it is (a stroke's changes stay
    /// previews until it ends, so the document does not change during a drag).
    pub fn pointer(&mut self, p: Pointer, doc: &Document, sel: Option<&Selection>) -> Effect {
        match p.phase {
            Phase::Press if self.drag.is_none() => self.press(p, doc, sel),
            Phase::Press => Effect::None,
            Phase::Drag => self.drag(p, doc, sel),
            Phase::Release => match self.drag.take() {
                Some(d) if d.button == p.button => self.release(d, doc),
                other => {
                    self.drag = other;
                    Effect::None
                }
            },
        }
    }

    fn release(&mut self, d: Drag, doc: &Document) -> Effect {
        let (w, h) = (doc.width(), doc.height());
        match d.kind {
            DragKind::Paint | DragKind::Shape => Effect::Commit(d.stroke.into_op()),
            // A click without a drag clears the selection.
            DragKind::Marquee if d.start == d.last => Effect::Select(None),
            DragKind::Marquee => Effect::Select(Selection::from_corners(d.start, d.last, w, h)),
            DragKind::Move(sel) => {
                let (dx, dy) = (d.last.0 - d.start.0, d.last.1 - d.start.1);
                let moved = sel.translated(dx, dy, w, h);
                Effect::Move(d.stroke.into_op(), moved)
            }
        }
    }

    /// The shape of a line or rectangle drag from `start` to `at`.
    fn shape(
        &self,
        doc: &Document,
        sel: Option<&Selection>,
        start: (i64, i64),
        at: (i64, i64),
        index: i64,
        shift: bool,
    ) -> Vec<PixelChange> {
        match self.tool {
            Tool::Rectangle => tools::rectangle(doc, sel, start, at, shift, index),
            _ => tools::line(doc, sel, start, at, self.brush, index),
        }
    }

    fn press(&mut self, p: Pointer, doc: &Document, sel: Option<&Selection>) -> Effect {
        if p.alt || self.tool == Tool::Eyedropper {
            return match tools::eyedropper(doc, p.at) {
                Some(index) => Effect::Pick(p.button, index),
                None => Effect::None,
            };
        }
        let index = self.color(p.button);
        match self.tool {
            Tool::Pencil | Tool::Eraser => {
                let changes = self.paint(doc, sel, &[p.at], index);
                self.start_drag(DragKind::Paint, p, index, changes.clone());
                Effect::Preview(changes)
            }
            Tool::Line | Tool::Rectangle => {
                let changes = self.shape(doc, sel, p.at, p.at, index, p.shift);
                self.start_drag(DragKind::Shape, p, index, changes.clone());
                Effect::Preview(changes)
            }
            Tool::Select => {
                let kind = match sel {
                    Some(s) if p.button == Button::Primary && s.contains_i(p.at.0, p.at.1) => {
                        DragKind::Move(*s)
                    }
                    _ => DragKind::Marquee,
                };
                self.start_drag(kind, p, index, Vec::new());
                Effect::None
            }
            Tool::Fill => {
                let changes = if p.shift {
                    tools::fill_global(doc, sel, p.at, index)
                } else {
                    tools::flood_fill(doc, sel, p.at, index)
                };
                Effect::Commit(Op::pixels(self.tool.name(), changes))
            }
            Tool::Eyedropper => unreachable!("handled above"),
        }
    }

    fn start_drag(&mut self, kind: DragKind, p: Pointer, index: i64, changes: Vec<PixelChange>) {
        let name = match kind {
            DragKind::Move(_) => "move",
            _ => self.tool.name(),
        };
        let mut stroke = Stroke::new(name);
        stroke.extend(changes);
        self.drag = Some(Drag {
            kind,
            button: p.button,
            index,
            start: p.at,
            last: p.at,
            stroke,
        });
    }

    fn drag(&mut self, p: Pointer, doc: &Document, sel: Option<&Selection>) -> Effect {
        let Some(d) = self.drag.as_ref() else {
            return Effect::None;
        };
        if d.button != p.button || d.last == p.at {
            return Effect::None;
        }
        let (kind, start, last, index) = (d.kind, d.start, d.last, d.index);
        let replaced = match kind {
            DragKind::Paint => {
                let changes = self.paint(doc, sel, &[last, p.at], index);
                let d = self.drag.as_mut().expect("checked above");
                d.last = p.at;
                d.stroke.extend(changes.clone());
                return Effect::Preview(changes);
            }
            DragKind::Marquee => {
                self.drag.as_mut().expect("checked above").last = p.at;
                return Effect::None;
            }
            DragKind::Shape => self.shape(doc, sel, start, p.at, index, p.shift),
            DragKind::Move(moving) => {
                tools::move_selection(doc, &moving, p.at.0 - start.0, p.at.1 - start.1).0
            }
        };
        let d = self.drag.as_mut().expect("checked above");
        d.last = p.at;
        let fresh = Stroke::new(d.stroke.tool());
        let old = std::mem::replace(&mut d.stroke, fresh);
        d.stroke.extend(replaced.clone());
        Effect::Repreview {
            old: old.changes().to_vec(),
            new: replaced,
        }
    }

    /// The brush along `points` with `index` (the eraser always paints transparency).
    fn paint(
        &self,
        doc: &Document,
        sel: Option<&Selection>,
        points: &[(i64, i64)],
        index: i64,
    ) -> Vec<PixelChange> {
        match self.tool {
            Tool::Eraser => tools::eraser(doc, sel, points, self.brush),
            _ => tools::pencil(doc, sel, points, self.brush, index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::{doc, rows};

    const PAL: [[u8; 3]; 3] = [[0, 0, 0], [255, 255, 255], [255, 0, 0]];

    fn at(phase: Phase, button: Button, x: i64, y: i64) -> Pointer {
        Pointer {
            phase,
            button,
            at: (x, y),
            shift: false,
            alt: false,
        }
    }

    #[test]
    fn a_stroke_previews_then_commits_once() {
        let mut d = doc(&["0000", "0000"], &PAL);
        let mut t = ToolBox::new(3);
        t.secondary = 2;
        let e = t.pointer(at(Phase::Press, Button::Secondary, 0, 0), &d, None);
        assert!(matches!(e, Effect::Preview(ref c) if c.len() == 1));
        assert!(t.dragging());
        // Same pixel again: nothing new. Another button: ignored.
        assert_eq!(
            t.pointer(at(Phase::Drag, Button::Secondary, 0, 0), &d, None),
            Effect::None
        );
        assert_eq!(
            t.pointer(at(Phase::Release, Button::Primary, 0, 0), &d, None),
            Effect::None
        );
        let e = t.pointer(at(Phase::Drag, Button::Secondary, 3, 1), &d, None);
        assert!(matches!(e, Effect::Preview(_)));
        let Effect::Commit(mut op) =
            t.pointer(at(Phase::Release, Button::Secondary, 3, 1), &d, None)
        else {
            panic!("no commit")
        };
        assert!(!t.dragging());
        d.apply(&mut op).unwrap();
        assert_eq!(rows(&d), ["2200", "0022"]);
    }

    #[test]
    fn eraser_fill_pick_and_cancel() {
        let d = doc(&["01", "11"], &PAL);
        let mut t = ToolBox::new(3);
        t.tool = Tool::Eraser;
        t.brush = 2;
        let Effect::Preview(c) = t.pointer(at(Phase::Press, Button::Primary, 0, 0), &d, None)
        else {
            panic!()
        };
        assert!(c.iter().all(|c| c.new == -1));
        assert_eq!(t.cancel(), Effect::Cancel(c));
        assert_eq!(t.cancel(), Effect::None);

        t.tool = Tool::Fill;
        let Effect::Commit(op) = t.pointer(at(Phase::Press, Button::Primary, 1, 1), &d, None)
        else {
            panic!()
        };
        assert!(matches!(op, Op::Pixels { ref changes, .. } if changes.len() == 3));

        let mut alt = at(Phase::Press, Button::Secondary, 1, 0);
        alt.alt = true;
        assert_eq!(t.pointer(alt, &d, None), Effect::Pick(Button::Secondary, 1));
        t.tool = Tool::Eyedropper;
        assert_eq!(
            t.pointer(at(Phase::Press, Button::Primary, 5, 5), &d, None),
            Effect::None
        );
    }

    fn shifted(mut p: Pointer) -> Pointer {
        p.shift = true;
        p
    }

    #[test]
    fn shapes_follow_the_pointer_and_commit_once() {
        let blank = ["0000", "0000", "0000"];
        let mut d = doc(&blank, &PAL);
        let mut t = ToolBox::new(3);
        t.primary = 1;
        t.tool = Tool::Line;
        let Effect::Preview(first) = t.pointer(at(Phase::Press, Button::Primary, 0, 0), &d, None)
        else {
            panic!()
        };
        assert_eq!(first.len(), 1);
        let Effect::Repreview { old, new } =
            t.pointer(at(Phase::Drag, Button::Primary, 3, 0), &d, None)
        else {
            panic!()
        };
        assert_eq!((old, new.len()), (first, 4));
        // Back to a shorter line: the preview shrinks, it does not accumulate.
        let Effect::Repreview { old, new } =
            t.pointer(at(Phase::Drag, Button::Primary, 1, 1), &d, None)
        else {
            panic!()
        };
        assert_eq!((old.len(), new.len()), (4, 2));
        let Effect::Commit(mut op) = t.pointer(at(Phase::Release, Button::Primary, 1, 1), &d, None)
        else {
            panic!()
        };
        d.apply(&mut op).unwrap();
        assert_eq!(rows(&d), ["1000", "0100", "0000"]);

        // Shift fills the rectangle; the selection clips it.
        let mut d = doc(&blank, &PAL);
        t.tool = Tool::Rectangle;
        let sel = Selection {
            x: 0,
            y: 0,
            w: 3,
            h: 3,
        };
        t.pointer(at(Phase::Press, Button::Primary, 0, 0), &d, Some(&sel));
        t.pointer(
            shifted(at(Phase::Drag, Button::Primary, 3, 1)),
            &d,
            Some(&sel),
        );
        let Effect::Commit(mut op) = t.pointer(
            shifted(at(Phase::Release, Button::Primary, 3, 1)),
            &d,
            Some(&sel),
        ) else {
            panic!()
        };
        d.apply(&mut op).unwrap();
        assert_eq!(rows(&d), ["1110", "1110", "0000"]);
    }

    #[test]
    fn the_selection_tool_selects_and_moves() {
        let d = doc(&["0120", "0000"], &PAL);
        let mut t = ToolBox::new(3);
        t.tool = Tool::Select;
        assert_eq!(
            t.pointer(at(Phase::Press, Button::Primary, 1, 0), &d, None),
            Effect::None
        );
        t.pointer(at(Phase::Drag, Button::Primary, 2, 0), &d, None);
        let sel = Selection {
            x: 1,
            y: 0,
            w: 2,
            h: 1,
        };
        assert_eq!(t.marquee(4, 2), Some(Some(sel)));
        assert_eq!(
            t.pointer(at(Phase::Release, Button::Primary, 2, 0), &d, None),
            Effect::Select(Some(sel))
        );
        assert_eq!(t.marquee(4, 2), None);
        // Pressed inside: the contents move with the pointer.
        t.pointer(at(Phase::Press, Button::Primary, 2, 0), &d, Some(&sel));
        let Effect::Repreview { new, .. } =
            t.pointer(at(Phase::Drag, Button::Primary, 3, 1), &d, Some(&sel))
        else {
            panic!()
        };
        assert!(!new.is_empty());
        let moved = Selection { x: 2, y: 1, ..sel };
        assert_eq!(t.marquee(4, 2), Some(Some(moved)));
        let Effect::Move(mut op, after) =
            t.pointer(at(Phase::Release, Button::Primary, 3, 1), &d, Some(&sel))
        else {
            panic!()
        };
        assert_eq!(after, Some(moved));
        assert!(matches!(op, Op::Pixels { ref tool, .. } if tool == "move"));
        let mut d2 = d.clone();
        d2.apply(&mut op).unwrap();
        assert_eq!(rows(&d2), ["0..0", "0012"]);
        // A click without a drag clears the selection.
        t.pointer(at(Phase::Press, Button::Primary, 0, 1), &d, Some(&sel));
        assert_eq!(
            t.pointer(at(Phase::Release, Button::Primary, 0, 1), &d, Some(&sel)),
            Effect::Select(None)
        );
    }

    #[test]
    fn colors_and_brush_stay_valid() {
        let mut t = ToolBox::new(1);
        assert_eq!((t.primary, t.secondary), (0, -1));
        t.swap_colors();
        assert_eq!((t.primary, t.secondary), (-1, 0));
        t.resize_brush(-5);
        assert_eq!(t.brush, 1);
        t.resize_brush(100);
        assert_eq!(t.brush, MAX_BRUSH);
        t.primary = 7;
        t.fit_palette(3);
        assert_eq!(t.primary, 2);
    }
}
