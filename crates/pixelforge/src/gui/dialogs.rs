//! The editor's modal dialogs: a palette entry's color (edit or add, R7, R8), the
//! confirmation of removing a used entry (R7), replace color (R14), merge entries (R9), and
//! unsaved changes on close (R29). A dialog only reports the user's answer ([`Outcome`]);
//! the app acts on it.

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
    /// Replace every pixel of `from` with `to` (`-1` transparent), in the selection when
    /// `in_selection`, else in the whole image. `palette` lists the choices.
    Replace {
        from: i64,
        to: i64,
        in_selection: bool,
        palette: Vec<[u8; 3]>,
    },
    /// Merge entry `from` into `into`: its pixels move to `into` and it is removed.
    Merge {
        from: usize,
        into: usize,
        palette: Vec<[u8; 3]>,
    },
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
    Replace(i64, i64),
    Merge(usize, usize),
}

/// An entry's name in a list: its index and hex, or "transparent" for `-1`.
fn entry_name(palette: &[[u8; 3]], index: i64) -> String {
    match usize::try_from(index).ok().and_then(|i| palette.get(i)) {
        Some(&rgb) => format!("{index}  {}", color::rgb8_to_hex(rgb)),
        None => "transparent".to_string(),
    }
}

/// A drop-down of the palette's entries (and the transparent slot with `transparent`).
fn entry_picker(
    ui: &mut egui::Ui,
    id: &str,
    palette: &[[u8; 3]],
    value: &mut i64,
    transparent: bool,
) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(entry_name(palette, *value))
        .show_ui(ui, |ui| {
            let first = if transparent { -1 } else { 0 };
            for i in first..palette.len() as i64 {
                ui.selectable_value(value, i, entry_name(palette, i));
            }
        });
}

/// Two buttons, OK (named `ok`) and Cancel; `ok_outcome` when OK is clicked.
fn buttons(ui: &mut egui::Ui, ok: &str, enabled: bool, ok_outcome: Outcome) -> Option<Outcome> {
    let mut out = None;
    ui.horizontal(|ui| {
        if ui.add_enabled(enabled, egui::Button::new(ok)).clicked() {
            out = Some(ok_outcome);
        }
        if ui.button("Cancel").clicked() {
            out = Some(Outcome::Cancel);
        }
    });
    out
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
                    // A new entry takes one of the unused colors (R8).
                    let offered = !(used && d.entry.is_none());
                    let resp =
                        ui.add_enabled_ui(offered, |ui| swatch(ui, rgb, rgb == d.color, used));
                    if resp.inner.clicked() {
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
    // Adding a color a named palette's entries already use would only duplicate it.
    let duplicate = d.entry.is_none()
        && d.choices
            .as_ref()
            .is_some_and(|c| c.iter().any(|&(rgb, used)| used && rgb == d.color));
    if duplicate {
        ui.weak("Every color of the palette is in use.");
    }
    buttons(ui, "Apply", !duplicate, Outcome::Color(d.entry, d.color))
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
            Dialog::Replace {
                from,
                to,
                in_selection,
                palette,
            } => {
                ui.heading("Replace color");
                ui.label(if *in_selection {
                    "Every pixel of one entry inside the selection becomes another."
                } else {
                    "Every pixel of one entry in the image becomes another."
                });
                egui::Grid::new("pixelforge-replace").show(ui, |ui| {
                    ui.label("Pixels of");
                    entry_picker(ui, "pixelforge-replace-from", palette, from, true);
                    ui.end_row();
                    ui.label("become");
                    entry_picker(ui, "pixelforge-replace-to", palette, to, true);
                    ui.end_row();
                });
                ui.separator();
                buttons(ui, "Replace", from != to, Outcome::Replace(*from, *to))
            }
            Dialog::Merge {
                from,
                into,
                palette,
            } => {
                ui.heading("Merge palette entries");
                ui.label(
                    "Every pixel of the first entry moves to the second, and the first entry \
                     is removed (entries above it shift down by one).",
                );
                let (mut a, mut b) = (*from as i64, *into as i64);
                egui::Grid::new("pixelforge-merge").show(ui, |ui| {
                    ui.label("Entry");
                    entry_picker(ui, "pixelforge-merge-from", palette, &mut a, false);
                    ui.end_row();
                    ui.label("into");
                    entry_picker(ui, "pixelforge-merge-into", palette, &mut b, false);
                    ui.end_row();
                });
                (*from, *into) = (a as usize, b as usize);
                ui.separator();
                buttons(ui, "Merge", from != into, Outcome::Merge(*from, *into))
            }
        }
    });
    if response.should_close() && response.inner.is_none() {
        return Some(Outcome::Cancel);
    }
    response.inner
}
