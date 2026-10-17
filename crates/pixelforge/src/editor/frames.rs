//! The frames of a batch edited in one window (`batch --edit`, docs/editor.md R4). Each frame
//! is its own [`Session`]: its own document, history, selection and save prefix. The caller
//! holds the current frame's session; [`Frames`] keeps the others and swaps them in.
//!
//! `batch` converts every frame with one palette. While the frames still share it, a palette
//! edit applies to every frame as one undo step in each history (a removal remaps each
//! frame's own pixels), and undoing or redoing it does the same in every frame. That is only
//! possible while it is the next step to undo (or redo) in every frame; otherwise the step
//! is refused and the error names the frames with later steps. Saving writes every frame
//! with unsaved changes, so the files on disk keep one palette.

use super::save::Saved;
use super::{Dirty, Document, EditError, Op, Session};
use crate::pipeline::Error;

/// The frames other than the current one, and which one is current.
#[derive(Debug)]
pub struct Frames {
    /// Every frame's session; the current frame's slot is empty (its session is the
    /// caller's).
    slots: Vec<Option<Session>>,
    current: usize,
    /// The frames had one palette when opened: palette edits apply to all of them.
    shared_palette: bool,
}

/// What identifies a palette op across frames (a removal's remap differs per frame).
fn palette_key(op: &Op) -> Option<(&'static str, usize, [u8; 3], [u8; 3])> {
    match op {
        Op::Pixels { .. } => None,
        Op::PaletteSet {
            index, old, new, ..
        } => Some(("set", *index, *old, *new)),
        Op::PaletteAdd { rgb } => Some(("add", 0, *rgb, *rgb)),
        Op::PaletteRemove { index, rgb, .. } => Some(("remove", *index, *rgb, *rgb)),
    }
}

/// `1, 3 and 4` for frames 0, 2 and 3 (one-based, as the window shows them).
fn frame_list(frames: &[usize]) -> String {
    let names: Vec<String> = frames.iter().map(|i| (i + 1).to_string()).collect();
    match names.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

impl Frames {
    /// The frames of `sessions`, the first one current: returns the set and the first
    /// frame's session. `None` without sessions.
    pub fn new(sessions: Vec<Session>) -> Option<(Frames, Session)> {
        let first = sessions.first()?;
        let shared_palette = sessions.iter().all(|s| {
            s.doc.palette == first.doc.palette
                && s.doc.palette_lab == first.doc.palette_lab
                && s.doc.fixed == first.doc.fixed
        });
        let mut slots: Vec<Option<Session>> = sessions.into_iter().map(Some).collect();
        let session = slots[0].take().expect("not empty");
        Some((
            Frames {
                slots,
                current: 0,
                shared_palette,
            },
            session,
        ))
    }

    /// How many frames there are.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The current frame (zero-based).
    pub fn current(&self) -> usize {
        self.current
    }

    /// Whether palette edits apply to every frame.
    pub fn shared_palette(&self) -> bool {
        self.shared_palette
    }

    /// The frames other than the current one, with their numbers.
    pub fn others(&self) -> impl Iterator<Item = (usize, &Session)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
    }

    /// The frames with unsaved changes (`session` is the current one).
    pub fn dirty(&self, session: &Session) -> Vec<usize> {
        (0..self.len())
            .filter(|&i| match &self.slots[i] {
                Some(s) => s.is_dirty(),
                None => session.is_dirty(),
            })
            .collect()
    }

    /// Make frame `to` current: `session` (the current frame) goes back to its slot and
    /// becomes frame `to`'s. `false` (nothing changes) for the current frame or one that
    /// does not exist.
    pub fn switch(&mut self, session: &mut Session, to: usize) -> bool {
        if to == self.current || to >= self.len() {
            return false;
        }
        let next = self.slots[to]
            .take()
            .expect("only the current slot is empty");
        self.slots[self.current] = Some(std::mem::replace(session, next));
        self.current = to;
        true
    }

    /// The other frames' sessions, mutably.
    fn others_mut(&mut self) -> impl Iterator<Item = (usize, &mut Session)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.as_mut().map(|s| (i, s)))
    }

    /// Apply the palette op `build` makes for each frame's document: to every frame with a
    /// shared palette, else to `session` only. Each op is built and checked against its own
    /// frame; when one is refused, the frames it was applied to are reverted and the error
    /// is returned. `Ok(None)` when the op changes nothing.
    pub fn apply_palette(
        &mut self,
        session: &mut Session,
        build: impl Fn(&Document) -> Result<Op, EditError>,
    ) -> Result<Option<Dirty>, EditError> {
        let dirty = session.apply(build(&session.doc)?)?;
        if !self.shared_palette || dirty.is_none() {
            return Ok(dirty);
        }
        let mut done: Vec<usize> = Vec::new();
        let mut failed = None;
        for (i, other) in self.others_mut() {
            match build(&other.doc).and_then(|op| other.apply(op)) {
                Ok(Some(_)) => done.push(i),
                Ok(None) => {}
                Err(e) => {
                    failed = Some(EditError(format!("frame {}: {}", i + 1, e.0)));
                    break;
                }
            }
        }
        if let Some(e) = failed {
            session.undo();
            for i in done {
                self.slots[i].as_mut().expect("another frame").undo();
            }
            return Err(e);
        }
        Ok(dirty)
    }

    /// Undo the current frame's last step; a shared palette step in every frame.
    pub fn undo(&mut self, session: &mut Session) -> Result<Option<Dirty>, EditError> {
        self.step(session, true)
    }

    /// Redo the current frame's next step; a shared palette step in every frame.
    pub fn redo(&mut self, session: &mut Session) -> Result<Option<Dirty>, EditError> {
        self.step(session, false)
    }

    fn step(&mut self, session: &mut Session, undo: bool) -> Result<Option<Dirty>, EditError> {
        let next = |s: &Session| {
            if undo {
                s.history.next_undo().and_then(palette_key)
            } else {
                s.history.next_redo().and_then(palette_key)
            }
        };
        let run = |s: &mut Session| if undo { s.undo() } else { s.redo() };
        let Some(key) = next(session).filter(|_| self.shared_palette) else {
            return Ok(run(session));
        };
        let blocked: Vec<usize> = self
            .others()
            .filter(|(_, s)| next(s) != Some(key))
            .map(|(i, _)| i)
            .collect();
        if !blocked.is_empty() {
            let (verb, what) = if undo {
                ("undo", "later edits")
            } else {
                ("redo", "other edits")
            };
            let many = blocked.len() > 1;
            return Err(EditError(format!(
                "cannot {verb} the palette edit: frame{} {} {} {what} ({verb} or undo them \
                 there first; the frames share one palette)",
                if many { "s" } else { "" },
                frame_list(&blocked),
                if many { "have" } else { "has" },
            )));
        }
        let dirty = run(session);
        for (_, other) in self.others_mut() {
            run(other);
        }
        Ok(dirty)
    }

    /// Save every frame with unsaved changes (`session` is the current one), so that the
    /// files of all frames keep one palette. Returns what each save wrote, in frame order.
    pub fn save(&mut self, session: &mut Session) -> Result<Vec<Saved>, Error> {
        let mut saved = Vec::new();
        for i in 0..self.len() {
            let s = match self.slots[i].as_mut() {
                Some(s) => s,
                None => &mut *session,
            };
            if s.is_dirty() {
                saved.push(s.save()?);
            }
        }
        Ok(saved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::{doc, rows};
    use crate::editor::{palette, tools};
    use std::path::PathBuf;

    const PAL: [[u8; 3]; 3] = [[0, 0, 0], [255, 255, 255], [255, 0, 0]];

    fn frames() -> (Frames, Session) {
        let sessions = ["00", "11", "22"]
            .iter()
            .enumerate()
            .map(|(i, row)| Session::new(doc(&[row], &PAL), PathBuf::from(format!("f{i}"))))
            .collect();
        Frames::new(sessions).unwrap()
    }

    fn paint(s: &mut Session, x: i64, index: i64) {
        let op = Op::pixels("pencil", tools::pencil(&s.doc, None, &[(x, 0)], 1, index));
        s.apply(op).unwrap();
    }

    #[test]
    fn frames_switch_and_keep_their_own_history() {
        let (mut f, mut s) = frames();
        assert_eq!((f.len(), f.current(), f.shared_palette()), (3, 0, true));
        paint(&mut s, 0, 2);
        assert!(!f.switch(&mut s, 0) && !f.switch(&mut s, 3));
        assert!(f.switch(&mut s, 2));
        assert_eq!((f.current(), rows(&s.doc)), (2, vec!["22".to_string()]));
        assert!(!s.is_dirty());
        assert_eq!(f.dirty(&s), [0]);
        // A pixel step is undone in its own frame only.
        paint(&mut s, 1, 0);
        assert!(f.undo(&mut s).unwrap().is_some());
        assert_eq!(rows(&s.doc), ["22"]);
        assert!(f.switch(&mut s, 0));
        assert_eq!(rows(&s.doc), ["20"]);
        assert_eq!(f.others().map(|(i, _)| i).collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn palette_edits_reach_every_frame() {
        let (mut f, mut s) = frames();
        // Recolor: every frame, one step each.
        let dirty = f
            .apply_palette(&mut s, |d| palette::set_color(d, 1, [9, 9, 9]))
            .unwrap();
        assert_eq!(dirty, Some(Dirty::All));
        assert_eq!(s.doc.palette[1], [9, 9, 9]);
        assert!(f.others().all(|(_, o)| o.doc.palette[1] == [9, 9, 9]));
        assert_eq!(f.dirty(&s), [0, 1, 2]);
        // Remove entry 2: each frame remaps its own pixels.
        f.apply_palette(&mut s, |d| palette::remove_entry(d, 2))
            .unwrap();
        assert!(f.switch(&mut s, 2));
        assert_eq!(s.doc.palette.len(), 2);
        assert_ne!(rows(&s.doc), ["22"]);
        // Undo from another frame undoes it everywhere.
        f.undo(&mut s).unwrap();
        assert_eq!(rows(&s.doc), ["22"]);
        assert!(f.others().all(|(_, o)| o.doc.palette.len() == 3));
        // A later pixel step in frame 1 blocks undoing the recolor from frame 3.
        assert!(f.switch(&mut s, 1));
        paint(&mut s, 0, 0);
        assert!(f.switch(&mut s, 2));
        let e = f.undo(&mut s).unwrap_err();
        assert!(e.0.contains("frame 2 has later edits"), "{}", e.0);
        assert_eq!(s.doc.palette[1], [9, 9, 9]);
        // Once it is undone there, the recolor goes away in all frames.
        assert!(f.switch(&mut s, 1));
        f.undo(&mut s).unwrap();
        f.undo(&mut s).unwrap();
        assert_eq!(s.doc.palette[1], [255, 255, 255]);
        assert!(f.others().all(|(_, o)| o.doc.palette[1] == [255, 255, 255]));
        assert!(f.dirty(&s).is_empty());
        // And redo brings it back everywhere.
        f.redo(&mut s).unwrap();
        assert!(f.others().all(|(_, o)| o.doc.palette[1] == [9, 9, 9]));
    }

    #[test]
    fn a_refused_palette_edit_changes_no_frame() {
        let (mut f, mut s) = frames();
        // Frame 3 gets a fourth entry the others lack, so it no longer shares the palette's
        // length: adding to the full set is refused there and reverted everywhere.
        let mut slots = std::mem::take(&mut f.slots);
        if let Some(Some(third)) = slots.get_mut(2) {
            for i in 0..palette::MAX_COLORS - 3 {
                let op = palette::add_color(&third.doc, [i as u8, 1, 2]).unwrap();
                third.apply(op).unwrap();
            }
        }
        f.slots = slots;
        let e = f
            .apply_palette(&mut s, |d| palette::add_color(d, [7, 7, 7]))
            .unwrap_err();
        assert!(e.0.starts_with("frame 3: "), "{}", e.0);
        assert_eq!(s.doc.palette.len(), 3);
        assert!(!s.is_dirty());
        assert_eq!(f.others().next().unwrap().1.doc.palette.len(), 3);
    }

    #[test]
    fn separate_palettes_stay_separate() {
        let a = Session::new(doc(&["0"], &PAL), PathBuf::from("a"));
        let b = Session::new(doc(&["0"], &PAL[..2]), PathBuf::from("b"));
        let (mut f, mut s) = Frames::new(vec![a, b]).unwrap();
        assert!(!f.shared_palette());
        f.apply_palette(&mut s, |d| palette::set_color(d, 0, [1, 2, 3]))
            .unwrap();
        assert_eq!(f.others().next().unwrap().1.doc.palette[0], [0, 0, 0]);
        assert!(Frames::new(Vec::new()).is_none());
        assert_eq!(frame_list(&[0, 2, 3]), "1, 3 and 4");
    }
}
