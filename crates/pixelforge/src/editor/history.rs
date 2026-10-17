//! Unlimited undo and redo with a savepoint (R25). The history keeps the applied ops in
//! order with an undo pointer: ops before it are applied, ops after it can be redone until a
//! new op replaces them. Memory is proportional to the pixels touched.

use super::{Document, EditError, EditLog, Op};

#[derive(Clone, Debug, Default)]
pub struct History {
    ops: Vec<Op>,
    /// How many of `ops` are applied.
    pointer: usize,
    /// The pointer when the document was last saved (or opened); `None` once that state
    /// cannot be reached again.
    savepoint: Option<usize>,
}

impl History {
    /// An empty history; the document as it is counts as saved.
    pub fn new() -> History {
        History {
            ops: Vec::new(),
            pointer: 0,
            savepoint: Some(0),
        }
    }

    /// Apply `op` to `doc` and record it, dropping the ops that could be redone. An op
    /// without changes is not recorded (`Ok(false)`).
    pub fn push(&mut self, doc: &mut Document, mut op: Op) -> Result<bool, EditError> {
        if op.is_empty() {
            return Ok(false);
        }
        doc.apply(&mut op)?;
        self.record(op);
        Ok(true)
    }

    /// Record an op the caller already applied with [`Document::apply`].
    pub fn record(&mut self, op: Op) {
        self.ops.truncate(self.pointer);
        if self.savepoint.is_some_and(|s| s > self.pointer) {
            self.savepoint = None;
        }
        self.ops.push(op);
        self.pointer += 1;
    }

    /// Undo the last applied op; `false` when there is none.
    pub fn undo(&mut self, doc: &mut Document) -> bool {
        if self.pointer == 0 {
            return false;
        }
        self.pointer -= 1;
        doc.revert(&self.ops[self.pointer]);
        true
    }

    /// Redo the next undone op; `false` when there is none.
    pub fn redo(&mut self, doc: &mut Document) -> bool {
        if self.pointer == self.ops.len() {
            return false;
        }
        doc.redo(&self.ops[self.pointer]);
        self.pointer += 1;
        true
    }

    pub fn can_undo(&self) -> bool {
        self.pointer > 0
    }

    pub fn can_redo(&self) -> bool {
        self.pointer < self.ops.len()
    }

    /// The applied ops, oldest first (what the edit log records).
    pub fn applied(&self) -> &[Op] {
        &self.ops[..self.pointer]
    }

    /// Whether the document differs from its last saved state.
    pub fn is_dirty(&self) -> bool {
        self.savepoint != Some(self.pointer)
    }

    /// Mark the current state as saved.
    pub fn mark_saved(&mut self) {
        self.savepoint = Some(self.pointer);
    }

    /// The edit log of `doc` with this history's applied ops.
    pub fn edit_log(&self, doc: &Document) -> EditLog {
        EditLog::of(doc, self.applied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::{doc, rows};
    use crate::editor::{tools, PixelChange};

    #[test]
    fn undo_redo_and_savepoint() {
        let pal = [[0, 0, 0], [255, 255, 255]];
        let mut d = doc(&["000"], &pal);
        let mut h = History::new();
        assert!(!h.is_dirty());
        let op = Op::pixels("pencil", tools::pencil(&d, None, &[(0, 0)], 1, 1));
        assert!(h.push(&mut d, op).unwrap());
        let op = Op::pixels("pencil", tools::pencil(&d, None, &[(1, 0)], 1, 1));
        h.push(&mut d, op).unwrap();
        assert!(!h.push(&mut d, Op::pixels("pencil", [])).unwrap());
        assert_eq!(rows(&d), ["110"]);
        assert!(h.is_dirty());
        h.mark_saved();
        assert!(h.undo(&mut d));
        assert_eq!(rows(&d), ["100"]);
        assert!(h.is_dirty());
        assert!(h.redo(&mut d));
        assert!(!h.is_dirty());
        assert!(!h.redo(&mut d));
        h.undo(&mut d);
        h.undo(&mut d);
        assert!(!h.undo(&mut d));
        assert_eq!(rows(&d), ["000"]);
        // A new op drops the redo branch, and with it the savepoint.
        let op = Op::pixels(
            "pencil",
            [PixelChange {
                x: 2,
                y: 0,
                old: 0,
                new: 1,
            }],
        );
        h.push(&mut d, op).unwrap();
        assert!(!h.can_redo());
        assert_eq!(h.applied().len(), 1);
        assert!(h.is_dirty());
        h.undo(&mut d);
        assert!(h.is_dirty());
        assert_eq!(h.edit_log(&d).ops.len(), 0);
    }
}
