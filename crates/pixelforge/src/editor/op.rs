//! The one primitive edit type: [`Op`]. Every tool computes a change list and applies it as
//! one `Pixels` op; palette edits are the other three variants.
//!
//! Applying an op checks it completely before changing anything, so a failed op leaves the
//! document as it was. [`Document::apply`] records the undo data (`old` values, the LAB of
//! a replaced or removed entry); [`Document::replay`] additionally requires the recorded
//! `old` values to match the document, which is how a log proves it belongs to a conversion.

use std::collections::BTreeMap;

use super::palette::MAX_COLORS;
use super::{err, Document, EditError};
use crate::color;

/// One pixel set from `old` to `new` (palette indices, `-1` transparent).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PixelChange {
    pub x: usize,
    pub y: usize,
    pub old: i64,
    pub new: i64,
}

/// An edit. The LAB fields are undo bookkeeping, filled in when the op is applied (an
/// entry's LAB is not always `rgb8_to_lab` of its color: a conversion's palette keeps the
/// pipeline's values). They are not part of the log and not compared by `==`.
#[derive(Clone, Debug)]
pub enum Op {
    /// Pixels set to new indices; `tool` names what made them (annotation only).
    Pixels {
        tool: String,
        changes: Vec<PixelChange>,
    },
    /// Entry `index` recolored (every pixel of that index changes color).
    PaletteSet {
        index: usize,
        old: [u8; 3],
        new: [u8; 3],
        old_lab: [f64; 3],
    },
    /// A new entry appended at index K.
    PaletteAdd { rgb: [u8; 3] },
    /// Entry `index` removed: the `remap` pixels (all of that index; `new` in the numbering
    /// before the removal) move to other entries, then every index above `index` shifts
    /// down by one.
    PaletteRemove {
        index: usize,
        rgb: [u8; 3],
        lab: [f64; 3],
        remap: Vec<PixelChange>,
    },
}

impl PartialEq for Op {
    fn eq(&self, other: &Op) -> bool {
        match (self, other) {
            (
                Op::Pixels { tool, changes },
                Op::Pixels {
                    tool: t2,
                    changes: c2,
                },
            ) => tool == t2 && changes == c2,
            (
                Op::PaletteSet {
                    index, old, new, ..
                },
                Op::PaletteSet {
                    index: i2,
                    old: o2,
                    new: n2,
                    ..
                },
            ) => index == i2 && old == o2 && new == n2,
            (Op::PaletteAdd { rgb }, Op::PaletteAdd { rgb: r2 }) => rgb == r2,
            (
                Op::PaletteRemove {
                    index, rgb, remap, ..
                },
                Op::PaletteRemove {
                    index: i2,
                    rgb: r2,
                    remap: m2,
                    ..
                },
            ) => index == i2 && rgb == r2 && remap == m2,
            _ => false,
        }
    }
}

/// `changes` as one op's list: one change per pixel (its first `old`, its last `new`),
/// without the ones that change nothing, in row-major order.
pub fn normalize(changes: impl IntoIterator<Item = PixelChange>) -> Vec<PixelChange> {
    let mut by_pos: BTreeMap<(usize, usize), (i64, i64)> = BTreeMap::new();
    for c in changes {
        by_pos
            .entry((c.y, c.x))
            .and_modify(|e| e.1 = c.new)
            .or_insert((c.old, c.new));
    }
    by_pos
        .into_iter()
        .filter(|(_, (old, new))| old != new)
        .map(|((y, x), (old, new))| PixelChange { x, y, old, new })
        .collect()
}

impl Op {
    /// A `Pixels` op from a change list ([`normalize`]d).
    pub fn pixels(tool: &str, changes: impl IntoIterator<Item = PixelChange>) -> Op {
        Op::Pixels {
            tool: tool.to_string(),
            changes: normalize(changes),
        }
    }

    /// A `Pixels` op without changes (applying it would do nothing).
    pub fn is_empty(&self) -> bool {
        matches!(self, Op::Pixels { changes, .. } if changes.is_empty())
    }

    /// The name the log gives this op (`"pixels"`, `"palette_set"`, ...).
    pub fn kind(&self) -> &'static str {
        match self {
            Op::Pixels { .. } => "pixels",
            Op::PaletteSet { .. } => "palette_set",
            Op::PaletteAdd { .. } => "palette_add",
            Op::PaletteRemove { .. } => "palette_remove",
        }
    }
}

fn lab_of(doc: &Document, index: usize) -> [f64; 3] {
    let l = &doc.palette_lab[index * 3..index * 3 + 3];
    [l[0], l[1], l[2]]
}

fn rgb8_lab(rgb: [u8; 3]) -> [f64; 3] {
    let l = color::rgb8_to_lab(&rgb);
    [l[0], l[1], l[2]]
}

impl Document {
    /// Apply `op`, recording its undo data (`old` values and LAB) from the document. Errors
    /// (nothing changed): a pixel outside the image, an index outside the palette, a color
    /// a named palette does not have, a full palette, removing a used entry without a
    /// complete remap.
    pub fn apply(&mut self, op: &mut Op) -> Result<(), EditError> {
        self.check(op, false)?;
        match op {
            Op::Pixels { changes, .. } => {
                for c in changes.iter_mut() {
                    c.old = self.indices.indices[c.y * self.indices.w + c.x];
                }
            }
            Op::PaletteSet {
                index,
                old,
                old_lab,
                ..
            } => {
                *old = self.palette[*index];
                *old_lab = lab_of(self, *index);
            }
            Op::PaletteRemove {
                index, rgb, lab, ..
            } => {
                *rgb = self.palette[*index];
                *lab = lab_of(self, *index);
            }
            Op::PaletteAdd { .. } => {}
        }
        self.redo(op);
        Ok(())
    }

    /// Apply a logged op: as [`apply`](Self::apply), but every recorded `old` value (pixel
    /// index, entry color) must match the document. The op's LAB fields are not read.
    pub fn replay(&mut self, op: &Op) -> Result<(), EditError> {
        self.check(op, true)?;
        let mut op = op.clone();
        match &mut op {
            Op::PaletteSet { index, old_lab, .. } => *old_lab = lab_of(self, *index),
            Op::PaletteRemove { index, lab, .. } => *lab = lab_of(self, *index),
            _ => {}
        }
        self.redo(&op);
        Ok(())
    }

    fn check(&self, op: &Op, check_old: bool) -> Result<(), EditError> {
        let (w, h) = (self.indices.w, self.indices.h);
        let k = self.palette.len();
        let pixel = |c: &PixelChange| -> Result<i64, EditError> {
            if c.x >= w || c.y >= h {
                return err(format!(
                    "pixel ({}, {}) is outside the {w}x{h} image",
                    c.x, c.y
                ));
            }
            Ok(self.indices.indices[c.y * w + c.x])
        };
        let in_range = |i: i64, transparent_ok: bool| {
            (transparent_ok && i == -1) || (i >= 0 && (i as usize) < k)
        };
        match op {
            Op::Pixels { changes, .. } => {
                for c in changes {
                    let current = pixel(c)?;
                    if !in_range(c.new, true) {
                        return err(format!(
                            "index {} at ({}, {}) is outside the palette of {k} entries",
                            c.new, c.x, c.y
                        ));
                    }
                    if check_old && current != c.old {
                        return err(format!(
                            "pixel ({}, {}) is {current}, the log expects {}",
                            c.x, c.y, c.old
                        ));
                    }
                }
            }
            Op::PaletteSet {
                index, old, new, ..
            } => {
                if *index >= k {
                    return err(format!(
                        "palette entry {index} does not exist ({k} entries)"
                    ));
                }
                if check_old && self.palette[*index] != *old {
                    return err(format!(
                        "palette entry {index} is {}, the log expects {}",
                        color::rgb8_to_hex(self.palette[*index]),
                        color::rgb8_to_hex(*old)
                    ));
                }
                self.check_color(*new)?;
            }
            Op::PaletteAdd { rgb } => {
                if k >= MAX_COLORS {
                    return err(format!("the palette is full ({MAX_COLORS} colors)"));
                }
                self.check_color(*rgb)?;
            }
            Op::PaletteRemove {
                index, rgb, remap, ..
            } => {
                if *index >= k {
                    return err(format!(
                        "palette entry {index} does not exist ({k} entries)"
                    ));
                }
                if k < 2 {
                    return err("cannot remove the only palette entry");
                }
                if check_old && self.palette[*index] != *rgb {
                    return err(format!(
                        "palette entry {index} is {}, the log expects {}",
                        color::rgb8_to_hex(self.palette[*index]),
                        color::rgb8_to_hex(*rgb)
                    ));
                }
                let mut moved = vec![false; w * h];
                for c in remap {
                    let current = pixel(c)?;
                    if current != *index as i64 || c.old != *index as i64 {
                        return err(format!(
                            "the remap of entry {index} names pixel ({}, {}), which is {current}",
                            c.x, c.y
                        ));
                    }
                    if !in_range(c.new, false) || c.new == *index as i64 {
                        return err(format!(
                            "entry {index} cannot be remapped to {} ({k} entries)",
                            c.new
                        ));
                    }
                    moved[c.y * w + c.x] = true;
                }
                let left = self
                    .indices
                    .indices
                    .iter()
                    .zip(&moved)
                    .filter(|(&i, &m)| i == *index as i64 && !m)
                    .count();
                if left > 0 {
                    return err(format!(
                        "palette entry {index} is still used by {left} pixels"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Set the op's new state (no checks; `op` was checked when first applied).
    pub(crate) fn redo(&mut self, op: &Op) {
        let w = self.indices.w;
        match op {
            Op::Pixels { changes, .. } => {
                for c in changes {
                    self.indices.indices[c.y * w + c.x] = c.new;
                }
            }
            Op::PaletteSet { index, new, .. } => {
                self.palette[*index] = *new;
                let lab = rgb8_lab(*new);
                self.palette_lab[index * 3..index * 3 + 3].copy_from_slice(&lab);
            }
            Op::PaletteAdd { rgb } => {
                self.palette.push(*rgb);
                self.palette_lab.extend_from_slice(&rgb8_lab(*rgb));
            }
            Op::PaletteRemove { index, remap, .. } => {
                for c in remap {
                    self.indices.indices[c.y * w + c.x] = c.new;
                }
                let index = *index as i64;
                for i in self.indices.indices.iter_mut() {
                    if *i > index {
                        *i -= 1;
                    }
                }
                let index = index as usize;
                self.palette.remove(index);
                self.palette_lab.drain(index * 3..index * 3 + 3);
            }
        }
    }

    /// Undo `op`, the last op applied.
    pub fn revert(&mut self, op: &Op) {
        let w = self.indices.w;
        match op {
            Op::Pixels { changes, .. } => {
                for c in changes {
                    self.indices.indices[c.y * w + c.x] = c.old;
                }
            }
            Op::PaletteSet {
                index,
                old,
                old_lab,
                ..
            } => {
                self.palette[*index] = *old;
                self.palette_lab[index * 3..index * 3 + 3].copy_from_slice(old_lab);
            }
            Op::PaletteAdd { .. } => {
                self.palette.pop();
                let k = self.palette.len();
                self.palette_lab.truncate(k * 3);
            }
            Op::PaletteRemove {
                index,
                rgb,
                lab,
                remap,
            } => {
                let i = *index as i64;
                for v in self.indices.indices.iter_mut() {
                    if *v >= i {
                        *v += 1;
                    }
                }
                for c in remap {
                    self.indices.indices[c.y * w + c.x] = c.old;
                }
                self.palette.insert(*index, *rgb);
                let at = index * 3;
                self.palette_lab.splice(at..at, lab.iter().copied());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::doc;

    fn px(x: usize, y: usize, old: i64, new: i64) -> PixelChange {
        PixelChange { x, y, old, new }
    }

    #[test]
    fn normalize_keeps_first_old_and_last_new() {
        let n = normalize([
            px(1, 0, 0, 1),
            px(0, 0, 0, 2),
            px(1, 0, 1, 0),
            px(0, 0, 2, 1),
        ]);
        assert_eq!(n, vec![px(0, 0, 0, 1)]);
    }

    #[test]
    fn apply_then_revert_restores_every_op_kind() {
        let mut d = doc(&["01", "1."], &[[0, 0, 0], [255, 255, 255]]);
        let before = d.clone();
        let mut ops = vec![
            Op::pixels("pencil", [px(0, 0, 0, 1), px(1, 1, -1, 0)]),
            Op::PaletteSet {
                index: 1,
                old: [0; 3],
                new: [10, 20, 30],
                old_lab: [0.0; 3],
            },
            Op::PaletteAdd { rgb: [1, 2, 3] },
        ];
        let mut done = Vec::new();
        for mut op in ops.drain(..) {
            d.apply(&mut op).unwrap();
            done.push(op);
        }
        assert_eq!(d.indices.indices, vec![1, 1, 1, 0]);
        assert_eq!(d.palette, vec![[0, 0, 0], [10, 20, 30], [1, 2, 3]]);
        let mut remove = Op::PaletteRemove {
            index: 0,
            rgb: [0; 3],
            lab: [0.0; 3],
            remap: vec![px(1, 1, 0, 2)],
        };
        d.apply(&mut remove).unwrap();
        assert_eq!(d.indices.indices, vec![0, 0, 0, 1]);
        assert_eq!(d.palette, vec![[10, 20, 30], [1, 2, 3]]);
        assert_eq!(d.palette_lab.len(), 6);
        done.push(remove);
        for op in done.iter().rev() {
            d.revert(op);
        }
        assert_eq!(d.indices, before.indices);
        assert_eq!(d.palette, before.palette);
        assert_eq!(d.palette_lab, before.palette_lab);
    }

    #[test]
    fn invalid_ops_change_nothing() {
        let mut d = doc(&["01"], &[[0, 0, 0], [255, 255, 255]]);
        let before = d.clone();
        let bad = [
            Op::pixels("pencil", [px(0, 0, 0, 1), px(2, 0, 0, 1)]),
            Op::pixels("pencil", [px(0, 0, 0, 1), px(1, 0, 1, 2)]),
            Op::PaletteSet {
                index: 2,
                old: [0; 3],
                new: [0; 3],
                old_lab: [0.0; 3],
            },
            // Entry 0 is used and the remap misses its pixel.
            Op::PaletteRemove {
                index: 0,
                rgb: [0; 3],
                lab: [0.0; 3],
                remap: vec![],
            },
        ];
        for mut op in bad {
            assert!(d.apply(&mut op).is_err(), "{op:?}");
            assert_eq!(d.indices, before.indices);
            assert_eq!(d.palette, before.palette);
        }
    }

    #[test]
    fn replay_checks_old_values() {
        let mut d = doc(&["01"], &[[0, 0, 0], [255, 255, 255]]);
        let wrong = Op::pixels("pencil", [px(0, 0, 1, 0)]);
        assert!(d.replay(&wrong).is_err());
        let right = Op::pixels("pencil", [px(0, 0, 0, 1)]);
        d.replay(&right).unwrap();
        assert_eq!(d.indices.indices, vec![1, 1]);
        let set = Op::PaletteSet {
            index: 0,
            old: [1, 1, 1],
            new: [2, 2, 2],
            old_lab: [0.0; 3],
        };
        assert!(d.replay(&set).is_err());
    }
}
