//! An editing session (docs/editor.md, Sections 3.1 and 3.10): one [`Document`], its
//! [`History`], the selection and the prefix saves go to. The GUI owns one session and turns
//! pointer and keyboard input into calls here; everything a session does is headless and
//! tested without a window (N6).
//!
//! Every change reports the part of the image it touched ([`Dirty`]), so the canvas updates
//! only those pixels (N5), and the per-entry pixel counts of the palette panel are kept up to
//! date incrementally for pixel ops.

use std::path::{Path, PathBuf};

use super::save::{self, Saved};
use super::{Document, EditError, History, Op, Selection, SourceRef};
use crate::pipeline::{self, Error};

/// The part of the image a change touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dirty {
    /// Only pixels inside this rectangle changed.
    Rect(Selection),
    /// Any pixel may have changed color (a palette edit).
    All,
}

impl Dirty {
    /// The smallest change covering both.
    pub fn union(self, other: Dirty) -> Dirty {
        match (self, other) {
            (Dirty::Rect(a), Dirty::Rect(b)) => {
                let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
                let (x1, y1) = ((a.x + a.w).max(b.x + b.w), (a.y + a.h).max(b.y + b.h));
                Dirty::Rect(Selection {
                    x: x0,
                    y: y0,
                    w: x1 - x0,
                    h: y1 - y0,
                })
            }
            _ => Dirty::All,
        }
    }

    /// The bounding rectangle of `positions`; `None` without any.
    pub fn of_positions(positions: impl IntoIterator<Item = (usize, usize)>) -> Option<Dirty> {
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        for (x, y) in positions {
            bounds = Some(match bounds {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
        }
        bounds.map(|(x0, y0, x1, y1)| {
            Dirty::Rect(Selection {
                x: x0,
                y: y0,
                w: x1 - x0 + 1,
                h: y1 - y0 + 1,
            })
        })
    }

    /// What applying or reverting `op` touches (`None` for an op without changes).
    pub fn of_op(op: &Op) -> Option<Dirty> {
        match op {
            Op::Pixels { changes, .. } => Dirty::of_positions(changes.iter().map(|c| (c.x, c.y))),
            _ => Some(Dirty::All),
        }
    }
}

/// `<dir>/<stem>`: the save prefix of an output PNG, in `outdir` when given.
pub fn prefix_of(png: &Path, outdir: Option<&Path>) -> PathBuf {
    let stem = png
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    match outdir {
        Some(dir) => dir.join(stem),
        None => png.with_file_name(stem),
    }
}

/// One document being edited.
#[derive(Clone, Debug)]
pub struct Session {
    pub doc: Document,
    pub history: History,
    /// Where [`save`](Self::save) writes: `<prefix>.png`, `<prefix>_meta.json`, ...
    pub prefix: PathBuf,
    /// The rectangle tools are limited to (R12); `None` is the whole image.
    pub selection: Option<Selection>,
    counts: Vec<usize>,
    transparent: usize,
    revision: u64,
}

impl Session {
    /// A session on `doc`, saving under `prefix`; the document as it is counts as saved.
    pub fn new(doc: Document, prefix: PathBuf) -> Session {
        let mut session = Session {
            doc,
            history: History::new(),
            prefix,
            selection: None,
            counts: Vec::new(),
            transparent: 0,
            revision: 0,
        };
        session.recount();
        session
    }

    /// `pixelforge edit PNG [--source INPUT] [-o OUTDIR]` (R1, R3): the PNG opened with
    /// [`Document::open`], the source image as an unaligned reference (saving also refuses to
    /// overwrite it), saves to `<OUTDIR or the PNG's directory>/<stem>`.
    pub fn open(
        png: &Path,
        source: Option<&Path>,
        outdir: Option<&Path>,
    ) -> Result<Session, EditError> {
        let mut doc = Document::open(png)?;
        if let Some(source) = source {
            doc.origin.source = Some(SourceRef::load(source)?);
            doc.origin.input_path = Some(source.to_path_buf());
        }
        Ok(Session::new(doc, prefix_of(png, outdir)))
    }

    fn recount(&mut self) {
        self.counts = self.doc.counts();
        self.transparent = self.doc.indices.indices.iter().filter(|&&i| i < 0).count();
    }

    fn count(&mut self, index: i64, add: bool) {
        let slot = if index < 0 {
            &mut self.transparent
        } else {
            &mut self.counts[index as usize]
        };
        if add {
            *slot += 1;
        } else {
            *slot -= 1;
        }
    }

    /// Keep the counts after `op` was applied (`forward`) or reverted.
    fn recount_after(&mut self, op: &Op, forward: bool) {
        match op {
            Op::Pixels { changes, .. } => {
                for c in changes {
                    let (from, to) = if forward {
                        (c.old, c.new)
                    } else {
                        (c.new, c.old)
                    };
                    self.count(from, false);
                    self.count(to, true);
                }
            }
            _ => self.recount(),
        }
        self.revision += 1;
    }

    /// Pixels per palette entry.
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Transparent pixels.
    pub fn transparent_count(&self) -> usize {
        self.transparent
    }

    /// A number that changes whenever the document does.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Apply `op` as one undo step. `Ok(None)`: the op changes nothing and is not recorded.
    pub fn apply(&mut self, op: Op) -> Result<Option<Dirty>, EditError> {
        if op.is_empty() {
            return Ok(None);
        }
        let dirty = Dirty::of_op(&op);
        self.history.push(&mut self.doc, op)?;
        let applied = self.history.next_undo().cloned().expect("just recorded");
        self.recount_after(&applied, true);
        Ok(dirty)
    }

    /// Undo the last step; `None` when there is none.
    pub fn undo(&mut self) -> Option<Dirty> {
        let op = self.history.next_undo()?.clone();
        self.history.undo(&mut self.doc);
        self.recount_after(&op, false);
        Dirty::of_op(&op)
    }

    /// Redo the next undone step; `None` when there is none.
    pub fn redo(&mut self) -> Option<Dirty> {
        let op = self.history.next_redo()?.clone();
        self.history.redo(&mut self.doc);
        self.recount_after(&op, true);
        Dirty::of_op(&op)
    }

    /// Whether there are changes since the last save (or since opening).
    pub fn is_dirty(&self) -> bool {
        self.history.is_dirty()
    }

    /// The PNG a save writes (`<prefix>.png`).
    pub fn image_path(&self) -> PathBuf {
        save::output_paths(&self.doc, &self.prefix)[0].1.clone()
    }

    /// Write every output file and the edit log under the prefix (R26, R28) and move the
    /// savepoint here.
    pub fn save(&mut self) -> Result<Saved, Error> {
        let log = self.history.edit_log(&self.doc);
        let saved = save::save(&self.doc, &log, &self.prefix)?;
        self.history.mark_saved();
        Ok(saved)
    }

    /// [`save`](Self::save) under a new prefix, which later saves keep.
    pub fn save_as(&mut self, prefix: PathBuf) -> Result<Saved, Error> {
        let old = std::mem::replace(&mut self.prefix, prefix);
        let result = self.save();
        if result.is_err() {
            self.prefix = old;
        }
        result
    }

    /// The tiles a save would write, re-extracted from a snapshot of the image (see
    /// [`save::rerender_indices`]); `None` without `config.tileset`.
    pub fn tiles_summary(
        idx: &pipeline::IndexImage,
        palette_lab: &[f64],
        config: &crate::config::Config,
    ) -> Result<Option<TilesSummary>, Error> {
        Ok(
            save::rerender_indices(idx, palette_lab, config)?.map(|(tiles, rendered)| {
                let changed = rendered
                    .indices
                    .iter()
                    .zip(&idx.indices)
                    .filter(|(a, b)| a != b)
                    .count();
                TilesSummary {
                    tiles: tiles.n_tiles,
                    changed,
                    rendered,
                }
            }),
        )
    }
}

/// What the tiles panel shows (R24, docs/editor.md Section 3.9).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TilesSummary {
    /// Tiles after de-duplication.
    pub tiles: usize,
    /// Pixels of the image the de-duplication changes.
    pub changed: usize,
    /// The image the tilemap draws (what a save writes).
    pub rendered: pipeline::IndexImage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::{doc, rows};
    use crate::editor::{palette, tools};

    #[test]
    fn changes_report_their_extent_and_keep_counts() {
        let pal = [[0, 0, 0], [255, 255, 255]];
        let mut s = Session::new(doc(&["000", "0.0"], &pal), PathBuf::from("x"));
        assert_eq!((s.counts(), s.transparent_count()), (&[5, 0][..], 1));
        let op = Op::pixels(
            "pencil",
            tools::pencil(&s.doc, None, &[(1, 0), (2, 1)], 1, 1),
        );
        let dirty = s.apply(op).unwrap();
        assert_eq!(
            dirty,
            Some(Dirty::Rect(Selection {
                x: 1,
                y: 0,
                w: 2,
                h: 2
            }))
        );
        assert_eq!(rows(&s.doc), ["010", "0.1"]);
        assert_eq!(s.counts(), &[3, 2]);
        assert!(s.is_dirty());
        assert_eq!(s.apply(Op::pixels("pencil", [])).unwrap(), None);
        let rev = s.revision();
        assert!(s.undo().is_some());
        assert!(s.revision() > rev);
        assert_eq!((s.counts(), s.transparent_count()), (&[5, 0][..], 1));
        assert!(!s.is_dirty());
        assert!(s.undo().is_none());
        assert!(s.redo().is_some());
        let op = palette::add_color(&s.doc, [9, 9, 9]).unwrap();
        assert_eq!(s.apply(op).unwrap(), Some(Dirty::All));
        assert_eq!(s.counts(), &[3, 2, 0]);
        assert!(s.redo().is_none());
    }

    #[test]
    fn dirty_union_and_prefix() {
        let a = Dirty::of_positions([(1, 1)]).unwrap();
        let b = Dirty::of_positions([(3, 0), (4, 2)]).unwrap();
        assert_eq!(
            a.union(b),
            Dirty::Rect(Selection {
                x: 1,
                y: 0,
                w: 4,
                h: 3
            })
        );
        assert_eq!(a.union(Dirty::All), Dirty::All);
        assert_eq!(
            prefix_of(Path::new("d/hero.png"), None),
            PathBuf::from("d/hero")
        );
        assert_eq!(
            prefix_of(Path::new("d/hero.png"), Some(Path::new("o"))),
            PathBuf::from("o/hero")
        );
    }
}
