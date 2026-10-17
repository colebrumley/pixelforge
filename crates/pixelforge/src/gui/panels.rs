//! The side panels below the palette and the canvas (docs/editor.md, Section 3.12): adjust
//! (R15), tiles (R24) and the source reference (R22), and the shortcuts sheet. Like the
//! palette panel they only show state and report what the user asked for; the app acts on
//! it.

use eframe::egui;
use egui::{vec2, Ui};

use crate::editor::adjust::Adjust;
use crate::editor::{SourceRef, TilesSummary};

/// A request from the adjust panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjustRequest {
    Apply,
    Cancel,
}

/// The adjust sliders, "only colors in selection", and Apply / Cancel. `active`: the
/// sliders are away from the identity; `busy`: a preview is being computed.
pub fn adjust(
    ui: &mut Ui,
    params: &mut Adjust,
    selection: bool,
    active: bool,
    busy: bool,
) -> Option<AdjustRequest> {
    let mut out = None;
    ui.horizontal(|ui| {
        ui.strong("Adjust");
        ui.weak(if selection {
            "selection"
        } else {
            "whole image"
        });
        if busy {
            // No spinner: the worker requests a frame when the preview is ready.
            ui.weak("previewing…");
        }
    });
    egui::Grid::new("pixelforge-adjust")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("Brightness");
            ui.add(egui::Slider::new(&mut params.brightness, -50.0..=50.0).step_by(1.0));
            ui.end_row();
            ui.label("Contrast");
            ui.add(egui::Slider::new(&mut params.contrast, 0.5..=2.0).step_by(0.05));
            ui.end_row();
            ui.label("Saturation");
            ui.add(egui::Slider::new(&mut params.saturation, 0.0..=2.0).step_by(0.05));
            ui.end_row();
        });
    ui.checkbox(
        &mut params.only_selection_colors,
        "Only colors in selection",
    )
    .on_hover_text(
        "Snap to the entries already used inside the selection, so pixels move within \
             their ramp instead of pulling in other colors.",
    );
    ui.horizontal(|ui| {
        if ui
            .add_enabled(active, egui::Button::new("Apply"))
            .on_hover_text("Apply the preview as one undo step.")
            .clicked()
        {
            out = Some(AdjustRequest::Apply);
        }
        if ui
            .add_enabled(active, egui::Button::new("Cancel"))
            .on_hover_text("Drop the preview and reset the sliders.")
            .clicked()
        {
            out = Some(AdjustRequest::Cancel);
        }
    });
    out
}

/// The tiles panel: tile count after de-duplication, the pixels it changes, and the toggle
/// that shows the image the tilemap draws. Returns whether the toggle changed.
pub fn tiles(
    ui: &mut Ui,
    info: Option<&Result<Option<TilesSummary>, String>>,
    show_reconstructed: &mut bool,
    busy: bool,
) -> bool {
    ui.horizontal(|ui| {
        ui.strong("Tiles");
        if busy {
            ui.weak("updating…");
        }
    });
    match info {
        None => {
            ui.weak("extracting…");
        }
        Some(Ok(Some(t))) => {
            ui.label(format!("{} tiles after de-duplication", t.tiles));
            let text = format!(
                "de-duplication changes {} pixel{}",
                t.changed,
                if t.changed == 1 { "" } else { "s" }
            );
            if t.changed > 0 {
                ui.colored_label(ui.visuals().warn_fg_color, text)
                    .on_hover_text(
                        "Saving writes the image the tilemap draws: near-duplicate tiles are \
                         merged under tile_dedupe_tolerance.",
                    );
            } else {
                ui.weak(text);
            }
        }
        Some(Ok(None)) => {
            ui.weak("no tileset");
        }
        Some(Err(e)) => {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
    }
    ui.checkbox(show_reconstructed, "Show what the tilemap draws")
        .on_hover_text("Draw the re-extracted tiles instead of the document.")
        .changed()
}

/// The source reference panel: the input image fitted to the panel, with the overlay
/// toggle and its opacity when the source is aligned with the output.
pub fn source(
    ui: &mut Ui,
    src: &SourceRef,
    texture: egui::TextureId,
    overlay: &mut bool,
    opacity: &mut f32,
) {
    let aligned = src.factor.is_some();
    ui.horizontal(|ui| {
        ui.strong("Source");
        if aligned {
            ui.weak("aligned with the output");
            ui.separator();
            ui.checkbox(overlay, "Overlay (Tab)");
            ui.add_enabled(
                *overlay,
                egui::Slider::new(opacity, 0.0..=1.0).text("opacity"),
            );
        } else {
            ui.weak("unaligned (opened with --source)");
        }
    });
    let avail = ui.available_size();
    let (w, h) = (src.w.max(1) as f32, src.h.max(1) as f32);
    let scale = (avail.x / w).min(avail.y / h).max(0.0);
    let size = vec2(w * scale, h * scale);
    if size.x >= 1.0 && size.y >= 1.0 {
        ui.centered_and_justified(|ui| {
            ui.add(egui::Image::new((texture, size)).fit_to_exact_size(size));
        });
    }
}
