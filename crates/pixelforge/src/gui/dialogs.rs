//! The editor's modal dialogs: a palette entry's color (edit or add, R7, R8), the
//! confirmation of removing a used entry (R7), and unsaved changes on close (R29). A dialog
//! only reports the user's answer ([`Outcome`]); the app acts on it.

use eframe::egui;
use egui::color_picker::{color_picker_color32, Alpha};
use egui::{vec2, Color32, Id, Modal, Sense, Stroke, StrokeKind};

use crate::color;

/// The color dialog's state.
#[derive(Clone, Debug)]
pub struct ColorDialog {
    /// The entry being edited; `None` adds a new entry.
    pub entry: Option<usize>,
    pub color: [u8; 3],
    /// The hex field as typed.
    pub hex: String,
    /// A named palette's colors, each with whether an entry already has it: the only
    /// choices (R8). `None` for a free palette.
    pub choices: Option<Vec<([u8; 3], bool)>>,
}

impl ColorDialog {
    pub fn new(
        entry: Option<usize>,
        color: [u8; 3],
        choices: Option<Vec<([u8; 3], bool)>>,
    ) -> Self {
        ColorDialog {
            entry,
            color,
            hex: color::rgb8_to_hex(color),
            choices,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Dialog {
    Color(ColorDialog),
    /// Remove `index`, whose `moved` pixels go to entry `target`.
    Remove {
        index: usize,
        moved: usize,
        target: usize,
        target_rgb: [u8; 3],
    },
    /// The window is closing with unsaved changes.
    Unsaved,
}

/// The user's answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Cancel,
    /// The color dialog's color, for its entry (or a new one).
    Color(Option<usize>, [u8; 3]),
    Remove(usize),
    SaveAndClose,
    DiscardAndClose,
}

fn swatch(ui: &mut egui::Ui, rgb: [u8; 3], selected: bool, used: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::click());
    let [r, g, b] = rgb;
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, Color32::from_rgb(r, g, b));
    let stroke = if selected {
        Stroke::new(2.0, ui.visuals().strong_text_color())
    } else {
        Stroke::new(1.0, ui.visuals().weak_text_color())
    };
    painter.rect_stroke(rect, 2.0, stroke, StrokeKind::Inside);
    if !used {
        // Unused colors of a named palette carry a dot.
        painter.circle_filled(rect.right_top() + vec2(-4.0, 4.0), 2.0, Color32::WHITE);
    }
    resp.on_hover_text(format!(
        "{}{}",
        color::rgb8_to_hex(rgb),
        if used { "" } else { " (unused)" }
    ))
}

fn color_body(ui: &mut egui::Ui, d: &mut ColorDialog) -> Option<Outcome> {
    ui.heading(match d.entry {
        Some(i) => format!("Palette entry {i}"),
        None => "New palette entry".to_string(),
    });
    match &d.choices {
        Some(choices) => {
            ui.label("A named palette: pick one of its colors (dot: not in use).");
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(3.0, 3.0);
                for &(rgb, used) in choices {
                    if swatch(ui, rgb, rgb == d.color, used).clicked() {
                        d.color = rgb;
                        d.hex = color::rgb8_to_hex(rgb);
                    }
                }
            });
        }
        None => {
            let [r, g, b] = d.color;
            let mut c = Color32::from_rgb(r, g, b);
            if color_picker_color32(ui, &mut c, Alpha::Opaque) {
                d.color = [c.r(), c.g(), c.b()];
                d.hex = color::rgb8_to_hex(d.color);
            }
            ui.horizontal(|ui| {
                ui.label("Hex");
                if ui.text_edit_singleline(&mut d.hex).changed() {
                    if let Ok(rgb) = color::hex_to_rgb8(d.hex.trim()) {
                        d.color = rgb;
                    }
                }
            });
        }
    }
    ui.separator();
    let mut out = None;
    ui.horizontal(|ui| {
        if ui.button("Apply").clicked() {
            out = Some(Outcome::Color(d.entry, d.color));
        }
        if ui.button("Cancel").clicked() {
            out = Some(Outcome::Cancel);
        }
    });
    out
}

/// Show `dialog`; returns the answer once the user gives one.
pub fn show(ctx: &egui::Context, dialog: &mut Dialog) -> Option<Outcome> {
    let response = Modal::new(Id::new("pixelforge-dialog")).show(ctx, |ui| {
        ui.set_max_width(360.0);
        match dialog {
            Dialog::Color(d) => color_body(ui, d),
            Dialog::Remove {
                index,
                moved,
                target,
                target_rgb,
            } => {
                ui.heading(format!("Remove palette entry {index}?"));
                ui.label(format!(
                    "{moved} pixel{} use it and move to the nearest remaining entry, \
                     {target} ({}). Entries above {index} shift down by one.",
                    if *moved == 1 { "" } else { "s" },
                    color::rgb8_to_hex(*target_rgb)
                ));
                ui.separator();
                let mut out = None;
                ui.horizontal(|ui| {
                    if ui.button("Remove").clicked() {
                        out = Some(Outcome::Remove(*index));
                    }
                    if ui.button("Cancel").clicked() {
                        out = Some(Outcome::Cancel);
                    }
                });
                out
            }
            Dialog::Unsaved => {
                ui.heading("Unsaved changes");
                ui.label("Save the edits before closing?");
                ui.separator();
                let mut out = None;
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        out = Some(Outcome::SaveAndClose);
                    }
                    if ui.button("Don't save").clicked() {
                        out = Some(Outcome::DiscardAndClose);
                    }
                    if ui.button("Cancel").clicked() {
                        out = Some(Outcome::Cancel);
                    }
                });
                out
            }
        }
    });
    if response.should_close() && response.inner.is_none() {
        return Some(Outcome::Cancel);
    }
    response.inner
}
