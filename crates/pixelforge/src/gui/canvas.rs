//! The canvas widget (docs/design/editor.md, Section 3.7): the image as one RGBA texture with
//! nearest filtering at a power-of-two zoom, a checkerboard under transparent pixels, the
//! pixel grid from 8x on, the tile grid with a tileset, the brush outline under the
//! pointer, the selection's marquee, and three optional layers: the diff overlay (a tint
//! where the document differs from the conversion, R23), the aligned source image at an
//! adjustable opacity (R22) and, with a tileset, the image the tilemap draws instead of the
//! document (R24).
//!
//! The canvas keeps a copy of the image's colors; edits update the copy for the pixels they
//! touched and the texture is updated by a partial upload of that rectangle once per frame.
//! A stroke in progress is drawn into the copy as a preview and restored from the document
//! when it is cancelled.
//!
//! Input: Space+drag and the middle button pan, scrolling pans, Cmd/Ctrl+scroll and pinch
//! zoom about the pointer. The primary and secondary buttons become tool [`Pointer`] events
//! in image pixels.

use eframe::egui;
use egui::{
    pos2, Color32, ColorImage, CursorIcon, PointerButton, Pos2, Rect, Sense, Stroke, TextureHandle,
    TextureOptions, Ui,
};

use super::tools::{Button, Phase, Pointer};
use super::view::{View, GRID_ZOOM};
use crate::editor::{Dirty, Document, PixelChange, Selection, SourceRef};
use crate::pipeline::IndexImage;

/// Points per checkerboard cell.
const CHECKER_CELL: f32 = 8.0;
/// Points of the image a pan always leaves on the canvas.
const KEEP_VISIBLE: i64 = 32;
/// The accumulated pinch / Cmd+scroll factor that makes one zoom step.
const ZOOM_STEP_FACTOR: f32 = 1.25;

const BACKGROUND: Color32 = Color32::from_gray(48);
const CHECKER_LIGHT: Color32 = Color32::from_gray(204);
const CHECKER_DARK: Color32 = Color32::from_gray(153);
const GRID: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 60);
const TILE_GRID: Color32 = Color32::from_rgba_premultiplied(0, 90, 160, 160);
/// The diff overlay's tint (premultiplied magenta at about 45 %).
const DIFF_TINT: Color32 = Color32::from_rgba_premultiplied(115, 0, 115, 115);

/// The color a palette index shows on the canvas.
pub fn index_color(doc: &Document, index: i64) -> Color32 {
    if index < 0 {
        Color32::TRANSPARENT
    } else {
        let [r, g, b] = doc.palette[index as usize];
        Color32::from_rgb(r, g, b)
    }
}

/// What the canvas draws on top of the image this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Overlay<'a> {
    /// The brush outline's side under the pointer (0: none).
    pub brush: usize,
    /// Draw the tile grid with this tile size.
    pub tile_size: Option<usize>,
    /// The selection's marquee.
    pub selection: Option<Selection>,
    /// The aligned source image (drawn only with [`Canvas::show_source`] and when it has a
    /// `factor`).
    pub source: Option<&'a SourceRef>,
}

/// What the canvas saw this frame.
#[derive(Clone, Debug, Default)]
pub struct CanvasInput {
    /// The image pixel under the pointer, when it is inside the image.
    pub hover: Option<(i64, i64)>,
    /// Tool events, in order.
    pub events: Vec<Pointer>,
}

pub struct Canvas {
    pub view: View,
    pub show_grid: bool,
    pub show_tile_grid: bool,
    /// Zoom to fit on the next frame (the canvas size is known only when it is laid out).
    fit_pending: bool,
    texture: Option<TextureHandle>,
    checker: Option<TextureHandle>,
    pixels: Vec<Color32>,
    w: usize,
    h: usize,
    /// The part of `pixels` the texture does not have yet.
    upload: Option<Dirty>,
    /// The canvas's top-left corner and size at the last frame (size in whole points).
    origin: Pos2,
    size: (i64, i64),
    scroll_rest: (f32, f32),
    zoom_rest: f32,
    /// The button of the tool drag in progress.
    pressed: Option<Button>,
    panning: bool,
    /// The diff overlay's tint per pixel, while it is shown.
    diff: Option<Vec<Color32>>,
    diff_texture: Option<TextureHandle>,
    diff_upload: Option<Dirty>,
    /// Draw the aligned source over the image, at `source_opacity` (0 to 1).
    pub show_source: bool,
    pub source_opacity: f32,
    source_texture: Option<TextureHandle>,
    /// The image the tilemap draws, drawn instead of the document while set.
    reconstructed: Option<Vec<Color32>>,
    reconstructed_texture: Option<TextureHandle>,
    reconstructed_upload: bool,
}

impl Canvas {
    pub fn new(doc: &Document) -> Canvas {
        let mut canvas = Canvas {
            view: View::default(),
            show_grid: true,
            show_tile_grid: true,
            fit_pending: true,
            texture: None,
            checker: None,
            pixels: Vec::new(),
            w: doc.width(),
            h: doc.height(),
            upload: None,
            origin: Pos2::ZERO,
            size: (0, 0),
            scroll_rest: (0.0, 0.0),
            zoom_rest: 1.0,
            pressed: None,
            panning: false,
            diff: None,
            diff_texture: None,
            diff_upload: None,
            show_source: false,
            source_opacity: 0.5,
            source_texture: None,
            reconstructed: None,
            reconstructed_texture: None,
            reconstructed_upload: false,
        };
        canvas.pixels = doc
            .indices
            .indices
            .iter()
            .map(|&i| index_color(doc, i))
            .collect();
        canvas
    }

    /// A canvas for another document (the next frame of a batch) with this one's settings:
    /// the grids, the diff and source overlays, and the zoom and pan when the size is the
    /// same (else it fits the new image).
    pub fn switched(&self, doc: &Document) -> Canvas {
        let mut canvas = Canvas::new(doc);
        canvas.show_grid = self.show_grid;
        canvas.show_tile_grid = self.show_tile_grid;
        canvas.show_source = self.show_source;
        canvas.source_opacity = self.source_opacity;
        if (canvas.w, canvas.h) == (self.w, self.h) {
            canvas.view = self.view;
            canvas.fit_pending = self.fit_pending;
            canvas.origin = self.origin;
            canvas.size = self.size;
        }
        canvas.set_diff(doc, self.diff_shown());
        canvas
    }

    fn mark(&mut self, dirty: Dirty) {
        self.upload = Some(match self.upload {
            Some(d) => d.union(dirty),
            None => dirty,
        });
    }

    /// Take the colors of `dirty` from the document (after an edit, undo or redo).
    pub fn refresh(&mut self, doc: &Document, dirty: Dirty) {
        match dirty {
            Dirty::All => {
                for (p, &i) in self.pixels.iter_mut().zip(&doc.indices.indices) {
                    *p = index_color(doc, i);
                }
            }
            Dirty::Rect(r) => {
                for (x, y) in r.positions() {
                    self.pixels[y * self.w + x] = index_color(doc, doc.get(x, y));
                }
            }
        }
        self.mark(dirty);
        self.refresh_diff(doc, dirty);
    }

    fn refresh_diff(&mut self, doc: &Document, dirty: Dirty) {
        let Some(diff) = self.diff.as_mut() else {
            return;
        };
        let tint = |d: bool| if d { DIFF_TINT } else { Color32::TRANSPARENT };
        match dirty {
            Dirty::All => {
                for (p, d) in diff.iter_mut().zip(doc.diff_mask()) {
                    *p = tint(d);
                }
            }
            Dirty::Rect(r) => {
                for (x, y) in r.positions() {
                    diff[y * self.w + x] = tint(doc.differs_at(x, y));
                }
            }
        }
        self.diff_upload = Some(match self.diff_upload {
            Some(d) => d.union(dirty),
            None => dirty,
        });
    }

    /// Show (`on`) or hide the diff overlay (R23). It only draws: the document and the
    /// image's colors are untouched.
    pub fn set_diff(&mut self, doc: &Document, on: bool) {
        if on == self.diff.is_some() {
            return;
        }
        if on {
            self.diff = Some(vec![Color32::TRANSPARENT; self.w * self.h]);
            self.refresh_diff(doc, Dirty::All);
        } else {
            self.diff = None;
            self.diff_texture = None;
            self.diff_upload = None;
        }
    }

    /// Whether the diff overlay is shown.
    pub fn diff_shown(&self) -> bool {
        self.diff.is_some()
    }

    /// The diff overlay's tint per pixel (for tests).
    pub fn diff_pixels(&self) -> Option<&[Color32]> {
        self.diff.as_deref()
    }

    /// Draw `image` (the tilemap's reconstruction, R24) instead of the document, or the
    /// document again with `None`. The image must have the document's size.
    pub fn set_reconstructed(&mut self, doc: &Document, image: Option<&IndexImage>) {
        self.reconstructed = image
            .filter(|i| i.w == self.w && i.h == self.h)
            .map(|i| i.indices.iter().map(|&k| index_color(doc, k)).collect());
        self.reconstructed_upload = self.reconstructed.is_some();
        if self.reconstructed.is_none() {
            self.reconstructed_texture = None;
        }
    }

    /// Whether the tilemap's reconstruction is drawn instead of the document.
    pub fn reconstructed_shown(&self) -> bool {
        self.reconstructed.is_some()
    }

    /// Draw changes that are not applied (a stroke in progress).
    pub fn preview(&mut self, doc: &Document, changes: &[PixelChange]) {
        for c in changes {
            self.pixels[c.y * self.w + c.x] = index_color(doc, c.new);
        }
        if let Some(d) = Dirty::of_positions(changes.iter().map(|c| (c.x, c.y))) {
            self.mark(d);
        }
    }

    /// Undo a [`preview`](Self::preview): the pixels of `changes` as the document has them.
    pub fn restore(&mut self, doc: &Document, changes: &[PixelChange]) {
        if let Some(d) = Dirty::of_positions(changes.iter().map(|c| (c.x, c.y))) {
            self.refresh(doc, d);
        }
    }

    /// The colors the canvas shows (for tests).
    pub fn pixels(&self) -> &[Color32] {
        &self.pixels
    }

    /// The screen position of pixel `(x, y)`'s center at the last frame's layout.
    pub fn pixel_center(&self, x: i64, y: i64) -> Pos2 {
        let (sx, sy) = self.view.screen_of((x, y));
        let half = self.view.zoom as f32 / 2.0;
        self.origin + egui::vec2(sx as f32 + half, sy as f32 + half)
    }

    pub fn zoom_to_fit(&mut self) {
        self.fit_pending = true;
    }

    /// Zoom by powers of two about the canvas center.
    pub fn zoom_steps(&mut self, steps: i32) {
        let center = (self.size.0 / 2, self.size.1 / 2);
        self.view.zoom_by(steps, center);
    }

    /// Upload what changed since the last frame (all of it the first time).
    fn upload(&mut self, ui: &Ui) {
        let (w, h) = (self.w, self.h);
        let Some(texture) = self.texture.as_mut() else {
            let image = ColorImage::new([w, h], self.pixels.clone());
            self.texture = Some(ui.ctx().load_texture(
                "pixelforge-canvas",
                image,
                TextureOptions::NEAREST,
            ));
            self.upload = None;
            return;
        };
        match self.upload.take() {
            None => {}
            Some(Dirty::All) => {
                let image = ColorImage::new([w, h], self.pixels.clone());
                texture.set(image, TextureOptions::NEAREST);
            }
            Some(Dirty::Rect(r)) => {
                let mut part = Vec::with_capacity(r.w * r.h);
                for y in r.y..r.y + r.h {
                    part.extend_from_slice(&self.pixels[y * w + r.x..y * w + r.x + r.w]);
                }
                let image = ColorImage::new([r.w, r.h], part);
                texture.set_partial([r.x, r.y], image, TextureOptions::NEAREST);
            }
        }
    }

    /// Upload the diff tint and the reconstruction when they changed.
    fn upload_layers(&mut self, ui: &Ui, source: Option<&SourceRef>) {
        let (w, h) = (self.w, self.h);
        if let Some(diff) = &self.diff {
            match (self.diff_texture.as_mut(), self.diff_upload.take()) {
                (None, _) | (Some(_), Some(Dirty::All)) => {
                    let image = ColorImage::new([w, h], diff.clone());
                    self.diff_texture = Some(ui.ctx().load_texture(
                        "pixelforge-diff",
                        image,
                        TextureOptions::NEAREST,
                    ));
                }
                (Some(texture), Some(Dirty::Rect(r))) => {
                    let mut part = Vec::with_capacity(r.w * r.h);
                    for y in r.y..r.y + r.h {
                        part.extend_from_slice(&diff[y * w + r.x..y * w + r.x + r.w]);
                    }
                    let image = ColorImage::new([r.w, r.h], part);
                    texture.set_partial([r.x, r.y], image, TextureOptions::NEAREST);
                }
                (Some(_), None) => {}
            }
        }
        if let Some(pixels) = &self.reconstructed {
            if std::mem::take(&mut self.reconstructed_upload)
                || self.reconstructed_texture.is_none()
            {
                let image = ColorImage::new([w, h], pixels.clone());
                self.reconstructed_texture = Some(ui.ctx().load_texture(
                    "pixelforge-tiles",
                    image,
                    TextureOptions::NEAREST,
                ));
            }
        }
        if let Some(src) = source {
            self.source_texture(ui.ctx(), src);
        }
    }

    /// The source image's texture (loaded on first use), for the overlay and the reference
    /// panel.
    pub fn source_texture(&mut self, ctx: &egui::Context, src: &SourceRef) -> egui::TextureId {
        self.source_texture
            .get_or_insert_with(|| {
                // The backend may allow less than the reference's size: keep every n-th
                // pixel so the texture fits. It is drawn over the same rectangle either way.
                let max_side = ctx.input(|i| i.max_texture_side).max(1);
                let n = src.w.max(src.h).div_ceil(max_side).max(1);
                let image = if n == 1 {
                    ColorImage::from_rgba_unmultiplied([src.w, src.h], &src.rgba)
                } else {
                    let (w, h) = (src.w.div_ceil(n), src.h.div_ceil(n));
                    let mut rgba = Vec::with_capacity(w * h * 4);
                    for y in 0..h {
                        for x in 0..w {
                            let i = ((y * n) * src.w + x * n) * 4;
                            rgba.extend_from_slice(&src.rgba[i..i + 4]);
                        }
                    }
                    ColorImage::from_rgba_unmultiplied([w, h], &rgba)
                };
                ctx.load_texture("pixelforge-source", image, TextureOptions::LINEAR)
            })
            .id()
    }

    fn checker(&mut self, ui: &Ui) -> egui::TextureId {
        self.checker
            .get_or_insert_with(|| {
                let (a, b) = (CHECKER_LIGHT, CHECKER_DARK);
                let image = ColorImage::new([2, 2], vec![a, b, b, a]);
                ui.ctx()
                    .load_texture("pixelforge-checker", image, TextureOptions::NEAREST_REPEAT)
            })
            .id()
    }

    /// Lay out the canvas in all of `ui`'s remaining space, handle view input and draw the
    /// image with `overlay` on top.
    pub fn show(&mut self, ui: &mut Ui, overlay: &Overlay<'_>) -> CanvasInput {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let size = (rect.width().floor() as i64, rect.height().floor() as i64);
        self.origin = rect.min;
        self.size = size;
        if self.fit_pending && size.0 > 0 && size.1 > 0 {
            self.view = View::fit(self.w, self.h, size.0, size.1);
            self.fit_pending = false;
        }
        let to_canvas = |p: Pos2| {
            (
                (p.x - rect.min.x).floor() as i64,
                (p.y - rect.min.y).floor() as i64,
            )
        };
        let hovered = response.hovered();
        let input = ui.input(|i| i.clone());
        let pointer_pos = input.pointer.interact_pos().or(input.pointer.hover_pos());
        let mut out = CanvasInput::default();

        // Panning and zooming.
        let space = input.key_down(egui::Key::Space);
        let primary_down = input.pointer.button_down(PointerButton::Primary);
        let middle_down = input.pointer.button_down(PointerButton::Middle);
        let pan_drag = (space && primary_down) || middle_down;
        if pan_drag && (self.panning || (hovered && self.pressed.is_none())) {
            self.panning = true;
            let d = input.pointer.delta();
            self.pan_points(d.x, d.y);
        } else if !pan_drag {
            self.panning = false;
        }
        if hovered {
            let scroll = input.smooth_scroll_delta;
            if scroll != egui::Vec2::ZERO {
                self.pan_points(scroll.x, scroll.y);
            }
            let zoom = input.zoom_delta();
            if zoom != 1.0 {
                self.zoom_rest *= zoom;
                let anchor = pointer_pos.map_or((size.0 / 2, size.1 / 2), to_canvas);
                if self.zoom_rest >= ZOOM_STEP_FACTOR {
                    self.view.zoom_by(1, anchor);
                    self.zoom_rest = 1.0;
                } else if self.zoom_rest * ZOOM_STEP_FACTOR <= 1.0 {
                    self.view.zoom_by(-1, anchor);
                    self.zoom_rest = 1.0;
                }
            }
        }
        self.view
            .keep_visible(self.w, self.h, size.0, size.1, KEEP_VISIBLE);

        // Tool events.
        let pixel = pointer_pos.map(|p| self.view.pixel_at(to_canvas(p)));
        let inside =
            |(x, y): (i64, i64)| x >= 0 && y >= 0 && x < self.w as i64 && y < self.h as i64;
        out.hover = pixel.filter(|&p| hovered && inside(p));
        let modifiers = input.modifiers;
        let event = |phase, button, at| Pointer {
            phase,
            button,
            at,
            shift: modifiers.shift,
            alt: modifiers.alt,
        };
        let egui_button = |b: Button| match b {
            Button::Primary => PointerButton::Primary,
            Button::Secondary => PointerButton::Secondary,
        };
        if let Some(at) = pixel {
            if self.pressed.is_none() && hovered && !self.panning && !space {
                for b in [Button::Primary, Button::Secondary] {
                    if input.pointer.button_pressed(egui_button(b)) {
                        self.pressed = Some(b);
                        out.events.push(event(Phase::Press, b, at));
                        // A focused button would otherwise take Space and Enter.
                        ui.memory_mut(|m| {
                            if let Some(id) = m.focused() {
                                m.surrender_focus(id);
                            }
                        });
                        break;
                    }
                }
            }
            if let Some(b) = self.pressed {
                let released = input.pointer.button_released(egui_button(b))
                    || !input.pointer.button_down(egui_button(b));
                out.events.push(event(Phase::Drag, b, at));
                if released {
                    out.events.push(event(Phase::Release, b, at));
                    self.pressed = None;
                }
            }
        } else if let Some(b) = self.pressed.take() {
            // The pointer left the window: end the drag where it was.
            out.events.push(event(Phase::Release, b, (-1, -1)));
        }
        if hovered || self.panning {
            let icon = if self.panning {
                CursorIcon::Grabbing
            } else if space {
                CursorIcon::Grab
            } else {
                CursorIcon::Crosshair
            };
            ui.ctx().set_cursor_icon(icon);
        }

        self.upload(ui);
        self.upload_layers(ui, overlay.source);
        self.paint(ui, rect, out.hover, overlay);
        out
    }

    /// Pan by a float delta, keeping the fraction for the next frame (the view is whole
    /// points).
    fn pan_points(&mut self, dx: f32, dy: f32) {
        let x = self.scroll_rest.0 + dx;
        let y = self.scroll_rest.1 + dy;
        let (wx, wy) = (x.trunc(), y.trunc());
        self.scroll_rest = (x - wx, y - wy);
        self.view.pan_by(wx as i64, wy as i64);
    }

    fn paint(&mut self, ui: &Ui, rect: Rect, hover: Option<(i64, i64)>, overlay: &Overlay<'_>) {
        let (brush, tile_size) = (overlay.brush, overlay.tile_size);
        let checker = self.checker(ui);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BACKGROUND);
        let View { zoom, pan } = self.view;
        let at = |x: i64, y: i64| {
            pos2(
                rect.min.x + (pan.0 + x * zoom) as f32,
                rect.min.y + (pan.1 + y * zoom) as f32,
            )
        };
        let image_rect = Rect::from_min_max(at(0, 0), at(self.w as i64, self.h as i64));
        let cells = image_rect.size() / (2.0 * CHECKER_CELL);
        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(cells.x, cells.y));
        painter.image(checker, image_rect, uv, Color32::WHITE);
        let full = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        let image = match &self.reconstructed_texture {
            Some(t) if self.reconstructed.is_some() => Some(t),
            _ => self.texture.as_ref(),
        };
        if let Some(texture) = image {
            painter.image(texture.id(), image_rect, full, Color32::WHITE);
        }
        let aligned = overlay.source.is_some_and(|s| s.factor.is_some());
        if let (true, true, Some(texture)) = (self.show_source, aligned, &self.source_texture) {
            let tint = Color32::WHITE.gamma_multiply(self.source_opacity.clamp(0.0, 1.0));
            painter.image(texture.id(), image_rect, full, tint);
        }
        if let (Some(_), Some(texture)) = (&self.diff, &self.diff_texture) {
            painter.image(texture.id(), image_rect, full, Color32::WHITE);
        }

        let (x0, y0, x1, y1) = self.view.visible(self.w, self.h, self.size.0, self.size.1);
        let lines = |step: i64, color: Color32| {
            let stroke = Stroke::new(1.0, color);
            let first = |v: i64| v.div_euclid(step) * step;
            let (ys, ye) = (at(0, y0).y, at(0, y1).y);
            let (xs, xe) = (at(x0, 0).x, at(x1, 0).x);
            let mut x = first(x0);
            while x <= x1 {
                painter.vline(at(x, 0).x, ys..=ye, stroke);
                x += step;
            }
            let mut y = first(y0);
            while y <= y1 {
                painter.hline(xs..=xe, at(0, y).y, stroke);
                y += step;
            }
        };
        if self.show_grid && zoom >= GRID_ZOOM && x1 > x0 && y1 > y0 {
            lines(1, GRID);
        }
        if let Some(t) = tile_size.filter(|&t| t > 0 && self.show_tile_grid) {
            if t as i64 * zoom >= 4 && x1 > x0 && y1 > y0 {
                lines(t as i64, TILE_GRID);
            }
        }

        if let Some(sel) = overlay.selection {
            let (x0, y0) = (sel.x as i64, sel.y as i64);
            let r = Rect::from_min_max(at(x0, y0), at(x0 + sel.w as i64, y0 + sel.h as i64));
            painter.rect_stroke(
                r,
                0.0,
                Stroke::new(1.0, Color32::BLACK),
                egui::StrokeKind::Outside,
            );
            let dash = 4.0;
            let corners = [
                r.left_top(),
                r.right_top(),
                r.right_bottom(),
                r.left_bottom(),
            ];
            for i in 0..4 {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                painter.add(egui::Shape::dashed_line(
                    &[a, b],
                    Stroke::new(1.0, Color32::WHITE),
                    dash,
                    dash,
                ));
            }
        }

        if let (Some((x, y)), true) = (hover, brush > 0) {
            let half = (brush as i64 - 1) / 2;
            let (bx, by) = (x - half, y - half);
            let r = Rect::from_min_max(at(bx, by), at(bx + brush as i64, by + brush as i64));
            painter.rect_stroke(
                r,
                0.0,
                Stroke::new(1.0, Color32::WHITE),
                egui::StrokeKind::Outside,
            );
            painter.rect_stroke(
                r,
                0.0,
                Stroke::new(1.0, Color32::BLACK),
                egui::StrokeKind::Inside,
            );
        }
    }
}
