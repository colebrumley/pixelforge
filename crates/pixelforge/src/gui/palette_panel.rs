//! The palette panel (R6, R7, R10): every entry with its index, hex and pixel count, and the
//! transparent slot. Left click makes an entry the primary color, right click the secondary,
//! double click edits it. The panel only reports what the user asked for
//! ([`PaletteAction`]); the app turns that into ops and dialogs.

use eframe::egui;
use egui::{vec2, Align2, Color32, FontId, Rect, Sense, Stroke, StrokeKind, Ui};

use super::tools::{Button, ToolBox};
use crate::color;
use crate::editor::palette::{over_budget, MAX_COLORS};
use crate::editor::Session;

/// A request from the panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaletteAction {
    /// Make this index (`-1` transparent) the button's color.
    Pick(Button, i64),
    /// Edit this entry's color.
    Edit(usize),
    /// Add an entry.
    Add,
    /// Remove this entry (asks first when pixels use it).
    Remove(usize),
    /// Merge two entries (R9; the dialog asks which).
    Merge,
    /// Replace one color with another (R14; the dialog asks which).
    Replace,
}

const ROW: f32 = 20.0;
const SWATCH: f32 = 14.0;

/// One row: swatch, index, hex, count and the primary / secondary markers. Returns the
/// clicks on it.
fn row(
    ui: &mut Ui,
    swatch: Option<Color32>,
    label: &str,
    count: usize,
    tools: &ToolBox,
    index: i64,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), ROW), Sense::click());
    let visuals = ui.visuals();
    let painter = ui.painter_at(rect);
    if response.hovered() {
        painter.rect_filled(rect, 2.0, visuals.widgets.hovered.weak_bg_fill);
    }
    let text = visuals.text_color();
    let sw = Rect::from_min_size(
        rect.left_center() + vec2(4.0, -SWATCH / 2.0),
        vec2(SWATCH, SWATCH),
    );
    match swatch {
        Some(c) => {
            painter.rect_filled(sw, 2.0, c);
        }
        None => {
            // The transparent slot: a crossed-out box.
            painter.line_segment([sw.left_bottom(), sw.right_top()], Stroke::new(1.0, text));
        }
    }
    painter.rect_stroke(sw, 2.0, Stroke::new(1.0, text), StrokeKind::Inside);
    let font = FontId::monospace(12.0);
    painter.text(
        sw.right_center() + vec2(6.0, 0.0),
        Align2::LEFT_CENTER,
        label,
        font.clone(),
        text,
    );
    let mut right = rect.right() - 4.0;
    for (marker, color) in [("S", tools.secondary), ("P", tools.primary)] {
        if color == index {
            painter.text(
                egui::pos2(right, rect.center().y),
                Align2::RIGHT_CENTER,
                marker,
                font.clone(),
                visuals.strong_text_color(),
            );
        }
        right -= 12.0;
    }
    painter.text(
        egui::pos2(right, rect.center().y),
        Align2::RIGHT_CENTER,
        count.to_string(),
        font,
        visuals.weak_text_color(),
    );
    response
}

/// Show the panel; returns the action the user took, if any.
pub fn show(ui: &mut Ui, session: &Session, tools: &ToolBox) -> Option<PaletteAction> {
    let doc = &session.doc;
    let k = doc.palette.len();
    let mut action = None;
    ui.horizontal(|ui| {
        ui.strong("Palette");
        let budget = doc.origin.config.palette_size;
        let text = format!("{k} / {budget}");
        if over_budget(doc) {
            ui.colored_label(ui.visuals().warn_fg_color, text)
                .on_hover_text("More entries than the config's palette_size.");
        } else {
            ui.weak(text);
        }
    });
    if let Some(name) = &doc.origin.config.palette_name {
        ui.weak(format!("named palette: {name}"));
    }
    ui.separator();
    let pick = |r: &egui::Response, index: i64, action: &mut Option<PaletteAction>| {
        if r.clicked() {
            *action = Some(PaletteAction::Pick(Button::Primary, index));
        } else if r.secondary_clicked() {
            *action = Some(PaletteAction::Pick(Button::Secondary, index));
        }
    };
    let bottom = 3.0 * ui.spacing().interact_size.y + 4.0 * ui.spacing().item_spacing.y;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .max_height((ui.available_height() - bottom).max(ROW))
        .show(ui, |ui| {
            for (i, &rgb) in doc.palette.iter().enumerate() {
                let [r, g, b] = rgb;
                let label = format!("{i:>3}  {}", color::rgb8_to_hex(rgb));
                let count = session.counts()[i];
                let resp = row(
                    ui,
                    Some(Color32::from_rgb(r, g, b)),
                    &label,
                    count,
                    tools,
                    i as i64,
                )
                .on_hover_text("Left: primary, right: secondary, double click: edit.");
                if resp.double_clicked() {
                    action = Some(PaletteAction::Edit(i));
                } else {
                    pick(&resp, i as i64, &mut action);
                }
            }
            let resp = row(
                ui,
                None,
                "  T  transparent",
                session.transparent_count(),
                tools,
                -1,
            );
            pick(&resp, -1, &mut action);
        });
    ui.separator();
    let entry = usize::try_from(tools.primary).ok().filter(|&i| i < k);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(k < MAX_COLORS, egui::Button::new("Add…"))
            .on_hover_text("Add an entry.")
            .clicked()
        {
            action = Some(PaletteAction::Add);
        }
        if ui
            .add_enabled(entry.is_some(), egui::Button::new("Edit…"))
            .on_hover_text("Change the primary entry's color (every pixel of it changes).")
            .clicked()
        {
            action = entry.map(PaletteAction::Edit);
        }
        if ui
            .add_enabled(entry.is_some() && k > 1, egui::Button::new("Remove…"))
            .on_hover_text("Remove the primary entry; its pixels move to the nearest entry.")
            .clicked()
        {
            action = entry.map(PaletteAction::Remove);
        }
    });
    ui.horizontal(|ui| {
        if ui
            .add_enabled(k > 1, egui::Button::new("Merge…"))
            .on_hover_text("Merge one entry into another (the first is removed).")
            .clicked()
        {
            action = Some(PaletteAction::Merge);
        }
        if ui
            .button("Replace…")
            .on_hover_text("Replace one color with another, in the selection or the image.")
            .clicked()
        {
            action = Some(PaletteAction::Replace);
        }
    });
    action
}
