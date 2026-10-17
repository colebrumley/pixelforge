//! The pixel editor's core (docs/editor.md, Section 3): the document, its operations and
//! history, the tools, adjust and cleanup, the edit log and saving. No GUI dependency; the
//! GUI (`crate::gui`, behind the `gui` feature) owns one [`Session`] and calls into this.
//!
//! Every edit is a pure function of the document and the operation (N3): tools reduce to a
//! change list ([`PixelChange`]) applied as one [`Op`], and colors use `color.rs` only
//! (CIELAB, CIE76 nearest index) plus arithmetic. The [`EditLog`] stores the applied ops as
//! pixel and palette changes, so replaying it does not depend on the tool algorithms:
//! `pixelforge convert INPUT ... --edits NAME_edits.json` reproduces an edited PNG byte for
//! byte.
//!
//! - `document`: [`Document`] and [`Origin`], built from an in-memory `pipeline::Output` or
//!   from a PNG on disk (a pixelforge output, or any PNG with at most 256 colors).
//! - `op`: [`Op`] and [`PixelChange`]; applying, checking and reverting ops.
//! - `history`: [`History`], undo and redo with a savepoint.
//! - `selection`: the rectangular [`Selection`].
//! - `tools`: pencil, eraser, fill, line, rectangle, replace color, move, copy and paste,
//!   eyedropper.
//! - `adjust`: brightness/contrast/saturation, ramp shift, ordered dither, cleanup.
//! - `palette`: palette entry edits and the named-palette rules.
//! - `log`: [`EditLog`], its JSON form, hash and replay.
//! - `save`: the output files of `convert` written from a document.
//! - `session`: [`Session`], a document with its history and save prefix, as the GUI uses
//!   it (every change reports the [`Dirty`] part of the image).
//! - `frames`: [`Frames`], the frames of a batch (`batch --edit`): one session each, palette
//!   edits shared while the frames share the palette.

use std::fmt;

pub mod adjust;
pub mod document;
pub mod frames;
pub mod history;
pub mod log;
pub mod op;
pub mod palette;
pub mod save;
pub mod selection;
pub mod session;
pub mod tools;

pub use document::{Document, Origin, SourceRef};
pub use frames::Frames;
pub use history::History;
pub use log::EditLog;
pub use op::{Op, PixelChange};
pub use selection::Selection;
pub use session::{Dirty, Session, TilesSummary};

/// An edit that cannot be applied, a log that does not fit the document, or a file the
/// editor cannot open. The CLI reports it as a config error (exit 2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditError(pub String);

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EditError {}

pub(crate) fn err<T>(message: impl Into<String>) -> Result<T, EditError> {
    Err(EditError(message.into()))
}
