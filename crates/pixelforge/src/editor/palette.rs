//! Palette entry edits and their rules (docs/editor.md, Section 3.6): recolor, add, remove
//! with a remap to the nearest remaining entry, merge. With a named palette (`fixed`) an
//! entry can only take one of that palette's colors. `palette_size` from the config is
//! informational ([`over_budget`]).

use super::{err, Document, EditError, Op, PixelChange};
use crate::color;

/// The most palette entries a document can have (the PNG's 256 slots minus transparency).
pub const MAX_COLORS: usize = 255;

impl Document {
    /// A named palette accepts only its own colors.
    pub fn check_color(&self, rgb: [u8; 3]) -> Result<(), EditError> {
        match &self.fixed {
            Some(fixed) if !fixed.contains(&rgb) => err(format!(
                "{} is not a color of the palette {}",
                color::rgb8_to_hex(rgb),
                self.origin.config.palette_name.as_deref().unwrap_or("")
            )),
            _ => Ok(()),
        }
    }
}

/// Recolor entry `index` (every pixel of that index changes color).
pub fn set_color(doc: &Document, index: usize, rgb: [u8; 3]) -> Result<Op, EditError> {
    if index >= doc.palette.len() {
        return err(format!("palette entry {index} does not exist"));
    }
    doc.check_color(rgb)?;
    Ok(Op::PaletteSet {
        index,
        old: doc.palette[index],
        new: rgb,
        old_lab: [0.0; 3],
    })
}

/// Append an entry (at index K).
pub fn add_color(doc: &Document, rgb: [u8; 3]) -> Result<Op, EditError> {
    if doc.palette.len() >= MAX_COLORS {
        return err(format!("the palette is full ({MAX_COLORS} colors)"));
    }
    doc.check_color(rgb)?;
    Ok(Op::PaletteAdd { rgb })
}

fn remove_to(doc: &Document, index: usize, target: usize) -> Op {
    let w = doc.width();
    let remap = doc
        .indices
        .indices
        .iter()
        .enumerate()
        .filter(|(_, &i)| i == index as i64)
        .map(|(p, _)| PixelChange {
            x: p % w,
            y: p / w,
            old: index as i64,
            new: target as i64,
        })
        .collect();
    Op::PaletteRemove {
        index,
        rgb: doc.palette[index],
        lab: [0.0; 3],
        remap,
    }
}

/// The remaining entry nearest to entry `index` (CIE76; ties to the lowest index).
pub fn nearest_other(doc: &Document, index: usize) -> Option<usize> {
    let k = doc.palette.len();
    if index >= k || k < 2 {
        return None;
    }
    let mut others = doc.palette_lab.clone();
    others.drain(index * 3..index * 3 + 3);
    let lab = &doc.palette_lab[index * 3..index * 3 + 3];
    let j = color::nearest_index(lab, &others)[0] as usize;
    Some(if j >= index { j + 1 } else { j })
}

/// Remove entry `index`; its pixels (if any) move to the nearest remaining entry and the
/// entries above shift down. The op's `remap` says how many pixels move.
pub fn remove_entry(doc: &Document, index: usize) -> Result<Op, EditError> {
    let Some(target) = nearest_other(doc, index) else {
        return err(format!(
            "cannot remove palette entry {index} of {}",
            doc.palette.len()
        ));
    };
    Ok(remove_to(doc, index, target))
}

/// Merge entry `from` into `into` (R9): every pixel of `from` becomes `into`, `from` is
/// removed.
pub fn merge_entries(doc: &Document, from: usize, into: usize) -> Result<Op, EditError> {
    let k = doc.palette.len();
    if from >= k || into >= k || from == into {
        return err(format!(
            "cannot merge entry {from} into {into} ({k} entries)"
        ));
    }
    Ok(remove_to(doc, from, into))
}

/// The named palette's colors, each with whether an entry already has it; `None` for a
/// free palette.
pub fn fixed_choices(doc: &Document) -> Option<Vec<([u8; 3], bool)>> {
    doc.fixed.as_ref().map(|fixed| {
        fixed
            .iter()
            .map(|c| (*c, doc.palette.contains(c)))
            .collect()
    })
}

/// More entries than the config's `palette_size` (a warning, not an error).
pub fn over_budget(doc: &Document) -> bool {
    doc.palette.len() as i64 > doc.origin.config.palette_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Value};
    use crate::editor::document::tests::{doc, rows};
    use crate::editor::Origin;
    use crate::pipeline::IndexImage;
    use crate::pyjson::Json;

    #[test]
    fn remove_remaps_to_nearest_and_reindexes() {
        // 0 black, 1 white, 2 near-black, 3 red.
        let pal = [[0, 0, 0], [255, 255, 255], [20, 20, 20], [255, 0, 0]];
        let mut d = doc(&["0123", "3210"], &pal);
        let mut op = remove_entry(&d, 2).unwrap();
        let Op::PaletteRemove { remap, .. } = &op else {
            panic!()
        };
        assert_eq!(remap.len(), 2);
        assert!(remap.iter().all(|c| c.new == 0));
        d.apply(&mut op).unwrap();
        assert_eq!(rows(&d), ["0102", "2010"]);
        assert_eq!(d.palette, vec![[0, 0, 0], [255, 255, 255], [255, 0, 0]]);
        d.revert(&op);
        assert_eq!(rows(&d), ["0123", "3210"]);
        // Removing entry 0: its nearest is 2 (near-black), which becomes 1 after the shift.
        let mut op = remove_entry(&d, 0).unwrap();
        d.apply(&mut op).unwrap();
        assert_eq!(rows(&d), ["1012", "2101"]);
        assert_eq!(d.palette[1], [20, 20, 20]);
    }

    #[test]
    fn merge_moves_pixels_to_the_chosen_entry() {
        let pal = [[0, 0, 0], [255, 255, 255], [20, 20, 20]];
        let mut d = doc(&["012."], &pal);
        let mut op = merge_entries(&d, 0, 1).unwrap();
        d.apply(&mut op).unwrap();
        assert_eq!(rows(&d), ["001."]);
        assert_eq!(d.palette, vec![[255, 255, 255], [20, 20, 20]]);
        assert!(merge_entries(&d, 1, 1).is_err());
    }

    #[test]
    fn named_palette_rejects_foreign_colors() {
        let config = Config::new(&[(
            "palette_name".to_string(),
            Value::str("hex:000000,ffffff,ff0000"),
        )])
        .unwrap();
        let origin = Origin::new(config, String::new(), Json::Null);
        let img = IndexImage {
            indices: vec![0, 1],
            h: 1,
            w: 2,
        };
        let mut d = Document::new(img, vec![[0, 0, 0], [255, 255, 255]], origin).unwrap();
        assert!(set_color(&d, 0, [1, 2, 3]).is_err());
        assert!(add_color(&d, [1, 2, 3]).is_err());
        // Also when the op is built by hand (or read from a log).
        let mut foreign = Op::PaletteAdd { rgb: [1, 2, 3] };
        assert!(d.apply(&mut foreign).is_err());
        let mut op = add_color(&d, [255, 0, 0]).unwrap();
        d.apply(&mut op).unwrap();
        let mut op = set_color(&d, 0, [255, 0, 0]).unwrap();
        d.apply(&mut op).unwrap();
        assert_eq!(
            fixed_choices(&d).unwrap(),
            vec![
                ([0, 0, 0], false),
                ([255, 255, 255], true),
                ([255, 0, 0], true)
            ]
        );
    }

    #[test]
    fn palette_limits() {
        let mut d = doc(&["0"], &[[0, 0, 0]]);
        assert!(remove_entry(&d, 0).is_err());
        assert!(!over_budget(&d));
        for i in 1..MAX_COLORS {
            let mut op = add_color(&d, [i as u8, 0, 0]).unwrap();
            d.apply(&mut op).unwrap();
        }
        assert_eq!(d.palette.len(), MAX_COLORS);
        assert!(add_color(&d, [0, 1, 0]).is_err());
        assert!(over_budget(&d));
    }
}
