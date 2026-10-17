//! Adjust, ramp shift and cleanup (docs/editor.md, Section 3.5). Each works on the palette
//! color a pixel currently has, so two pixels of the same index always get the same result,
//! and snaps back to the palette with `color::nearest_index` (CIE76, ties to the lowest
//! index). Only `+`, `-`, `*` and `/` are involved besides `color.rs`. Transparent pixels
//! never change.

use super::op::normalize;
use super::selection::region;
use super::{Document, PixelChange, Selection};
use crate::color;
use crate::postprocess::{self, ORPHAN_MAX_PASSES};

/// Ramp shift's neighbor radius: the largest distance over `a` and `b` only (ΔE ignoring
/// L) at which another entry counts as the same hue ramp.
pub const RAMP_CHROMA: f64 = 20.0;

/// The adjust sliders. `brightness` in −50..50 (added to L), `contrast` in 0.5..2 (scales
/// L about the selection's mean L), `saturation` in 0..2 (scales a and b). Values outside
/// are clamped. `only_selection_colors` snaps to the entries used inside the selection
/// instead of the whole palette.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adjust {
    pub brightness: f64,
    pub contrast: f64,
    pub saturation: f64,
    pub only_selection_colors: bool,
}

impl Default for Adjust {
    /// The identity: no change.
    fn default() -> Adjust {
        Adjust {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            only_selection_colors: false,
        }
    }
}

fn lab(doc: &Document, i: i64) -> [f64; 3] {
    let i = i as usize;
    let l = &doc.palette_lab[i * 3..i * 3 + 3];
    [l[0], l[1], l[2]]
}

/// The opaque pixels inside the region, row-major.
fn opaque(doc: &Document, sel: Option<&Selection>) -> Vec<(usize, usize, i64)> {
    region(sel, doc.width(), doc.height())
        .positions()
        .map(|(x, y)| (x, y, doc.get(x, y)))
        .filter(|&(_, _, i)| i >= 0)
        .collect()
}

/// The changes that map every pixel through `map` (a new index per entry).
fn remap(pixels: &[(usize, usize, i64)], map: &[i64]) -> Vec<PixelChange> {
    normalize(pixels.iter().map(|&(x, y, i)| PixelChange {
        x,
        y,
        old: i,
        new: map[i as usize],
    }))
}

/// The entries used by `pixels`, ascending.
fn used(pixels: &[(usize, usize, i64)], k: usize) -> Vec<usize> {
    let mut seen = vec![false; k];
    for &(_, _, i) in pixels {
        seen[i as usize] = true;
    }
    (0..k).filter(|&i| seen[i]).collect()
}

/// Brightness, contrast and saturation in CIELAB on the selection (or the image), snapped
/// to the palette (R15). The contrast pivot is the mean L of the selected pixels after the
/// brightness shift; L is clamped to 0..100 last.
pub fn adjust(doc: &Document, sel: Option<&Selection>, params: &Adjust) -> Vec<PixelChange> {
    let pixels = opaque(doc, sel);
    if pixels.is_empty() {
        return Vec::new();
    }
    let b = params.brightness.clamp(-50.0, 50.0);
    let c = params.contrast.clamp(0.5, 2.0);
    let s = params.saturation.clamp(0.0, 2.0);
    let mut sum = 0.0;
    for &(_, _, i) in &pixels {
        sum += lab(doc, i)[0];
    }
    let mean = sum / pixels.len() as f64 + b;
    let k = doc.palette.len();
    let entries = used(&pixels, k);
    let candidates: Vec<usize> = if params.only_selection_colors {
        entries.clone()
    } else {
        (0..k).collect()
    };
    let candidate_lab: Vec<f64> = candidates
        .iter()
        .flat_map(|&j| lab(doc, j as i64))
        .collect();
    let mut map: Vec<i64> = (0..k as i64).collect();
    for &i in &entries {
        let [l, a, bb] = lab(doc, i as i64);
        let l = (mean + c * (l + b - mean)).clamp(0.0, 100.0);
        let target = [l, a * s, bb * s];
        let j = color::nearest_index(&target, &candidate_lab)[0] as usize;
        map[i] = candidates[j] as i64;
    }
    remap(&pixels, &map)
}

/// Ramp shift (R16): every selected pixel moves to the entry of the same hue ramp (within
/// [`RAMP_CHROMA`] over a and b) with the next higher L (`lighter`) or the next lower L.
/// Ties go to the lowest index; pixels without such a neighbor stay.
pub fn ramp_shift(doc: &Document, sel: Option<&Selection>, lighter: bool) -> Vec<PixelChange> {
    let pixels = opaque(doc, sel);
    let k = doc.palette.len();
    let mut map: Vec<i64> = (0..k as i64).collect();
    for i in used(&pixels, k) {
        let [l, a, b] = lab(doc, i as i64);
        let mut best: Option<(usize, f64)> = None;
        for j in 0..k {
            let [lj, aj, bj] = lab(doc, j as i64);
            let (da, db) = (aj - a, bj - b);
            if j == i || da * da + db * db > RAMP_CHROMA * RAMP_CHROMA {
                continue;
            }
            let step = if lighter { lj - l } else { l - lj };
            if step > 0.0 && best.is_none_or(|(_, s)| step < s) {
                best = Some((j, step));
            }
        }
        if let Some((j, _)) = best {
            map[i] = j as i64;
        }
    }
    remap(&pixels, &map)
}

/// Which postprocess pass [`cleanup`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cleanup {
    /// `postprocess::remove_orphans` with the config's `orphan_min_region` and
    /// `orphan_max_delta`.
    Orphans,
    /// `postprocess::fix_jaggies`.
    Jaggies,
}

/// Run a postprocess pass on the selection (R17): on its bounding box plus a one-pixel
/// margin, keeping only the changes inside the selection.
pub fn cleanup(doc: &Document, sel: Option<&Selection>, pass: Cleanup) -> Vec<PixelChange> {
    let (w, h) = (doc.width(), doc.height());
    let inner = region(sel, w, h);
    let outer = inner.grown(1, w, h);
    let cut: Vec<i64> = outer.positions().map(|(x, y)| doc.get(x, y)).collect();
    let config = &doc.origin.config;
    let out = match pass {
        Cleanup::Orphans => postprocess::remove_orphans(
            &cut,
            outer.h,
            outer.w,
            &doc.palette_lab,
            config.orphan_min_region,
            ORPHAN_MAX_PASSES,
            config.orphan_max_delta,
        ),
        Cleanup::Jaggies => postprocess::fix_jaggies(&cut, outer.h, outer.w),
    };
    normalize(
        outer
            .positions()
            .zip(cut.iter().zip(&out))
            .filter(|((x, y), _)| inner.contains(*x, *y))
            .map(|((x, y), (&old, &new))| PixelChange { x, y, old, new }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::{doc, rows};
    use crate::editor::Op;

    /// A grey ramp, a red and a transparent pixel.
    fn sample() -> Document {
        let pal = [
            [20, 20, 20],
            [80, 80, 80],
            [140, 140, 140],
            [200, 200, 200],
            [200, 30, 30],
        ];
        doc(&["0123", "4.12", "3210"], &pal)
    }

    fn applied(d: &Document, changes: Vec<PixelChange>) -> Document {
        let mut d = d.clone();
        d.apply(&mut Op::pixels("adjust", changes)).unwrap();
        d
    }

    #[test]
    fn identity_and_snap_are_idempotent() {
        let d = sample();
        assert!(adjust(&d, None, &Adjust::default()).is_empty());
        let params = Adjust {
            brightness: 12.5,
            contrast: 1.7,
            saturation: 0.4,
            only_selection_colors: false,
        };
        let changes = adjust(&d, None, &params);
        assert!(!changes.is_empty());
        let once = applied(&d, changes.clone());
        // Snapping a snapped result again changes nothing.
        assert!(adjust(&once, None, &Adjust::default()).is_empty());
        // Deterministic: the same input gives the same list.
        assert_eq!(adjust(&d, None, &params), changes);
        // Same index in, same index out; transparency untouched.
        assert_eq!(once.get(1, 1), -1);
        assert_eq!(once.get(0, 0), once.get(3, 2));
    }

    #[test]
    fn brightness_moves_along_the_greys() {
        let d = sample();
        let up = Adjust {
            brightness: 50.0,
            ..Adjust::default()
        };
        let sel = Selection {
            x: 0,
            y: 0,
            w: 4,
            h: 1,
        };
        let out = applied(&d, adjust(&d, Some(&sel), &up));
        assert_eq!(rows(&out)[1..], ["4.12", "3210"]);
        assert!(rows(&out)[0]
            .bytes()
            .zip("0123".bytes())
            .all(|(a, b)| a >= b));
        assert_ne!(rows(&out)[0], "0123");
        // Limited to the colors in the selection: the red never appears.
        let sel = Selection {
            x: 0,
            y: 2,
            w: 4,
            h: 1,
        };
        let only = Adjust {
            saturation: 2.0,
            brightness: -20.0,
            only_selection_colors: true,
            ..Adjust::default()
        };
        let out = applied(&d, adjust(&d, Some(&sel), &only));
        assert!(!rows(&out)[2].contains('4'));
    }

    #[test]
    fn ramp_shift_steps_within_the_ramp() {
        let d = sample();
        let lighter = applied(&d, ramp_shift(&d, None, true));
        assert_eq!(rows(&lighter), ["1233", "4.23", "3321"]);
        let darker = applied(&d, ramp_shift(&d, None, false));
        assert_eq!(rows(&darker), ["0012", "4.01", "2100"]);
    }

    #[test]
    fn cleanup_keeps_changes_inside_the_selection() {
        // A lone pixel of 1 in a field of 0 (close colors, so it is merged).
        let pal = [[100, 100, 100], [104, 100, 100]];
        let d = doc(&["0000", "0100", "0000", "0010"], &pal);
        let all = applied(&d, cleanup(&d, None, Cleanup::Orphans));
        assert_eq!(rows(&all), ["0000", "0000", "0000", "0000"]);
        let sel = Selection {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
        };
        let part = applied(&d, cleanup(&d, Some(&sel), Cleanup::Orphans));
        assert_eq!(rows(&part), ["0000", "0000", "0000", "0010"]);
    }
}
