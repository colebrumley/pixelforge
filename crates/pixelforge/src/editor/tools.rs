//! The drawing tools (docs/editor.md, Section 3.4) as pure functions from the document, the
//! selection and pointer positions to a change list. Positions are pixel coordinates and may
//! lie outside the image (a drag that leaves the canvas); whatever falls outside the image
//! or the selection is dropped. Each change carries the document's current index as `old`.
//!
//! A drag accumulates its change lists in a [`Stroke`] and commits them as one op on
//! release, so a drag is one undo step.

use super::op::normalize;
use super::selection::region;
use super::{Document, Op, PixelChange, Selection};

/// The largest brush (a square of `MAX_BRUSH × MAX_BRUSH` pixels).
pub const MAX_BRUSH: usize = 8;

/// The pixels of a Bresenham line from `a` to `b`, both included.
pub fn bresenham(a: (i64, i64), b: (i64, i64)) -> Vec<(i64, i64)> {
    let (mut x, mut y) = a;
    let dx = (b.0 - a.0).abs();
    let dy = -(b.1 - a.1).abs();
    let sx = if a.0 < b.0 { 1 } else { -1 };
    let sy = if a.1 < b.1 { 1 } else { -1 };
    let mut e = dx + dy;
    let mut out = Vec::with_capacity((dx - dy + 1) as usize);
    loop {
        out.push((x, y));
        if (x, y) == b {
            return out;
        }
        let e2 = 2 * e;
        if e2 >= dy {
            e += dy;
            x += sx;
        }
        if e2 <= dx {
            e += dx;
            y += sy;
        }
    }
}

/// A change setting `(x, y)` to `index`, if the position is in the image and the region.
fn change(doc: &Document, region: &Selection, x: i64, y: i64, index: i64) -> Option<PixelChange> {
    if !region.contains_i(x, y) {
        return None;
    }
    let (x, y) = (x as usize, y as usize);
    Some(PixelChange {
        x,
        y,
        old: doc.get(x, y),
        new: index,
    })
}

/// Square brush stamps of `size` (clamped to 1..=[`MAX_BRUSH`]) at every point of the
/// polyline through `points` (Bresenham between successive points, so a fast drag leaves no
/// gaps). The stamp covers `x - (size - 1) / 2` onward, likewise for `y`.
fn stamps(
    doc: &Document,
    sel: Option<&Selection>,
    points: &[(i64, i64)],
    size: usize,
    index: i64,
) -> Vec<PixelChange> {
    let region = region(sel, doc.width(), doc.height());
    let size = size.clamp(1, MAX_BRUSH) as i64;
    let half = (size - 1) / 2;
    let mut path: Vec<(i64, i64)> = Vec::new();
    match points {
        [] => {}
        [p] => path.push(*p),
        _ => {
            for (i, pair) in points.windows(2).enumerate() {
                let line = bresenham(pair[0], pair[1]);
                // The shared point is drawn once.
                path.extend(line.into_iter().skip(usize::from(i > 0)));
            }
        }
    }
    let mut out = Vec::new();
    for (px, py) in path {
        for y in py - half..py - half + size {
            for x in px - half..px - half + size {
                out.extend(change(doc, &region, x, y, index));
            }
        }
    }
    normalize(out)
}

/// Pencil: paint `index` with a square brush along `points`.
pub fn pencil(
    doc: &Document,
    sel: Option<&Selection>,
    points: &[(i64, i64)],
    size: usize,
    index: i64,
) -> Vec<PixelChange> {
    stamps(doc, sel, points, size, index)
}

/// Eraser: the pencil with transparency (`-1`).
pub fn eraser(
    doc: &Document,
    sel: Option<&Selection>,
    points: &[(i64, i64)],
    size: usize,
) -> Vec<PixelChange> {
    stamps(doc, sel, points, size, -1)
}

/// Line from `a` to `b` with the brush.
pub fn line(
    doc: &Document,
    sel: Option<&Selection>,
    a: (i64, i64),
    b: (i64, i64),
    size: usize,
    index: i64,
) -> Vec<PixelChange> {
    stamps(doc, sel, &[a, b], size, index)
}

/// Rectangle with corners `a` and `b` (included): its four edges, or every pixel inside
/// with `filled`.
pub fn rectangle(
    doc: &Document,
    sel: Option<&Selection>,
    a: (i64, i64),
    b: (i64, i64),
    filled: bool,
    index: i64,
) -> Vec<PixelChange> {
    let region = region(sel, doc.width(), doc.height());
    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
    let mut out = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            if filled || x == x0 || x == x1 || y == y0 || y == y1 {
                out.extend(change(doc, &region, x, y, index));
            }
        }
    }
    normalize(out)
}

/// Flood fill: the 4-connected area of the index at `at` (inside the selection) becomes
/// `index`. Nothing when `at` is outside the image or the selection.
pub fn flood_fill(
    doc: &Document,
    sel: Option<&Selection>,
    at: (i64, i64),
    index: i64,
) -> Vec<PixelChange> {
    let region = region(sel, doc.width(), doc.height());
    if !region.contains_i(at.0, at.1) {
        return Vec::new();
    }
    let (w, h) = (doc.width(), doc.height());
    let target = doc.get(at.0 as usize, at.1 as usize);
    if target == index {
        return Vec::new();
    }
    let mut seen = vec![false; w * h];
    let mut stack = vec![(at.0 as usize, at.1 as usize)];
    seen[at.1 as usize * w + at.0 as usize] = true;
    let mut out = Vec::new();
    while let Some((x, y)) = stack.pop() {
        out.push(PixelChange {
            x,
            y,
            old: target,
            new: index,
        });
        let neighbours = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in neighbours {
            if nx < w
                && ny < h
                && region.contains(nx, ny)
                && !seen[ny * w + nx]
                && doc.get(nx, ny) == target
            {
                seen[ny * w + nx] = true;
                stack.push((nx, ny));
            }
        }
    }
    normalize(out)
}

/// Global fill (Shift + fill): every pixel of the index at `at`, inside the selection,
/// becomes `index`.
pub fn fill_global(
    doc: &Document,
    sel: Option<&Selection>,
    at: (i64, i64),
    index: i64,
) -> Vec<PixelChange> {
    match doc.get_at(at.0, at.1) {
        Some(from) if region(sel, doc.width(), doc.height()).contains_i(at.0, at.1) => {
            replace_color(doc, sel, from, index)
        }
        _ => Vec::new(),
    }
}

/// Replace color: every pixel of index `from` inside the selection (or the image) becomes
/// `to`.
pub fn replace_color(
    doc: &Document,
    sel: Option<&Selection>,
    from: i64,
    to: i64,
) -> Vec<PixelChange> {
    let region = region(sel, doc.width(), doc.height());
    if from == to {
        return Vec::new();
    }
    region
        .positions()
        .filter(|&(x, y)| doc.get(x, y) == from)
        .map(|(x, y)| PixelChange {
            x,
            y,
            old: from,
            new: to,
        })
        .collect()
}

/// Move the selection's contents by `(dx, dy)`: transparency where they were, the moved
/// indices where they land (clipped to the image). Returns the changes and the moved
/// selection (`None` when it left the image).
pub fn move_selection(
    doc: &Document,
    sel: &Selection,
    dx: i64,
    dy: i64,
) -> (Vec<PixelChange>, Option<Selection>) {
    let (w, h) = (doc.width(), doc.height());
    let moved = sel.translated(dx, dy, w, h);
    if dx == 0 && dy == 0 {
        return (Vec::new(), Some(*sel));
    }
    let mut out: Vec<PixelChange> = sel
        .positions()
        .map(|(x, y)| PixelChange {
            x,
            y,
            old: doc.get(x, y),
            new: -1,
        })
        .collect();
    for (x, y) in sel.positions() {
        let (tx, ty) = (x as i64 + dx, y as i64 + dy);
        if let Some(old) = doc.get_at(tx, ty) {
            out.push(PixelChange {
                x: tx as usize,
                y: ty as usize,
                old,
                new: doc.get(x, y),
            });
        }
    }
    (normalize(out), moved)
}

/// A copied rectangle of indices (`-1` transparent), row-major (R30).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clip {
    pub w: usize,
    pub h: usize,
    pub indices: Vec<i64>,
}

/// Copy the selection's indices (the whole image without one).
pub fn copy(doc: &Document, sel: Option<&Selection>) -> Clip {
    let region = region(sel, doc.width(), doc.height());
    Clip {
        w: region.w,
        h: region.h,
        indices: region.positions().map(|(x, y)| doc.get(x, y)).collect(),
    }
}

/// Paste `clip` with its top-left corner at `at`: every pixel of the rectangle, transparent
/// ones included (as moving a selection does), clipped to the image. Returns the changes and
/// the pasted rectangle as the new selection (`None` when nothing of it is inside).
pub fn paste(doc: &Document, clip: &Clip, at: (i64, i64)) -> (Vec<PixelChange>, Option<Selection>) {
    if clip.w == 0 || clip.h == 0 {
        return (Vec::new(), None);
    }
    let mut out = Vec::new();
    for cy in 0..clip.h {
        for cx in 0..clip.w {
            let (x, y) = (at.0 + cx as i64, at.1 + cy as i64);
            if let Some(old) = doc.get_at(x, y) {
                out.push(PixelChange {
                    x: x as usize,
                    y: y as usize,
                    old,
                    new: clip.indices[cy * clip.w + cx],
                });
            }
        }
    }
    let placed = Selection::from_corners(
        at,
        (at.0 + clip.w as i64 - 1, at.1 + clip.h as i64 - 1),
        doc.width(),
        doc.height(),
    );
    (normalize(out), placed)
}

/// Eyedropper: the index at `at` (`-1` transparent), or `None` outside the image.
pub fn eyedropper(doc: &Document, at: (i64, i64)) -> Option<i64> {
    doc.get_at(at.0, at.1)
}

/// A drag in progress: the change lists of its pointer events, committed as one op.
#[derive(Clone, Debug, Default)]
pub struct Stroke {
    tool: String,
    changes: Vec<PixelChange>,
}

impl Stroke {
    pub fn new(tool: &str) -> Stroke {
        Stroke {
            tool: tool.to_string(),
            changes: Vec::new(),
        }
    }

    /// Add one event's changes (computed on the document as it was before the stroke).
    pub fn extend(&mut self, changes: Vec<PixelChange>) {
        self.changes.extend(changes);
        self.changes = normalize(std::mem::take(&mut self.changes));
    }

    /// The tool name the op is annotated with.
    pub fn tool(&self) -> &str {
        &self.tool
    }

    /// The stroke so far, for the preview.
    pub fn changes(&self) -> &[PixelChange] {
        &self.changes
    }

    /// The stroke as one `Pixels` op.
    pub fn into_op(self) -> Op {
        Op::pixels(&self.tool, self.changes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::{doc, rows};

    const PAL: [[u8; 3]; 3] = [[0, 0, 0], [255, 255, 255], [255, 0, 0]];

    fn applied(rows_in: &[&str], changes: Vec<PixelChange>) -> Vec<String> {
        let mut d = doc(rows_in, &PAL);
        d.apply(&mut Op::pixels("t", changes)).unwrap();
        rows(&d)
    }

    #[test]
    fn bresenham_lines() {
        assert_eq!(
            bresenham((0, 0), (3, 1)),
            vec![(0, 0), (1, 0), (2, 1), (3, 1)]
        );
        assert_eq!(bresenham((2, 2), (2, 2)), vec![(2, 2)]);
        assert_eq!(
            bresenham((0, 3), (0, 0)),
            vec![(0, 3), (0, 2), (0, 1), (0, 0)]
        );
    }

    #[test]
    fn pencil_sizes_and_gaps() {
        let blank = ["00000", "00000", "00000", "00000"];
        let d = doc(&blank, &PAL);
        assert_eq!(
            applied(&blank, pencil(&d, None, &[(0, 0), (4, 2)], 1, 1)),
            ["10000", "01100", "00011", "00000"]
        );
        assert_eq!(
            applied(&blank, pencil(&d, None, &[(1, 1)], 2, 2)),
            ["00000", "02200", "02200", "00000"]
        );
        assert_eq!(
            applied(&blank, pencil(&d, None, &[(0, 0)], 3, 1)),
            ["11000", "11000", "00000", "00000"]
        );
        let sel = Selection {
            x: 1,
            y: 0,
            w: 2,
            h: 4,
        };
        assert_eq!(
            applied(&blank, eraser(&d, Some(&sel), &[(0, 1), (4, 1)], 1)),
            ["00000", "0..00", "00000", "00000"]
        );
    }

    #[test]
    fn flood_fill_is_four_connected_and_clipped() {
        let img = ["0010", "0101", "1100", "0000"];
        let d = doc(&img, &PAL);
        assert_eq!(
            applied(&img, flood_fill(&d, None, (0, 0), 2)),
            ["2210", "2101", "1100", "0000"]
        );
        assert_eq!(
            applied(&img, flood_fill(&d, None, (3, 3), 2)),
            ["0010", "0121", "1122", "2222"]
        );
        let sel = Selection {
            x: 2,
            y: 2,
            w: 2,
            h: 2,
        };
        assert_eq!(
            applied(&img, flood_fill(&d, Some(&sel), (3, 3), 2)),
            ["0010", "0101", "1122", "0022"]
        );
        assert!(flood_fill(&d, Some(&sel), (0, 0), 2).is_empty());
        assert_eq!(
            applied(&img, fill_global(&d, None, (2, 0), 2)),
            ["0020", "0202", "2200", "0000"]
        );
    }

    #[test]
    fn line_and_rectangles() {
        let blank = ["0000", "0000", "0000", "0000"];
        let d = doc(&blank, &PAL);
        assert_eq!(
            applied(&blank, line(&d, None, (0, 3), (3, 0), 1, 1)),
            ["0001", "0010", "0100", "1000"]
        );
        assert_eq!(
            applied(&blank, rectangle(&d, None, (3, 3), (0, 1), false, 2)),
            ["0000", "2222", "2002", "2222"]
        );
        assert_eq!(
            applied(&blank, rectangle(&d, None, (1, 1), (5, 2), true, 1)),
            ["0000", "0111", "0111", "0000"]
        );
    }

    #[test]
    fn replace_and_move() {
        let img = ["0120", "0120"];
        let d = doc(&img, &PAL);
        let sel = Selection {
            x: 0,
            y: 0,
            w: 2,
            h: 1,
        };
        assert_eq!(
            applied(&img, replace_color(&d, Some(&sel), 0, 2)),
            ["2120", "0120"]
        );
        let sel = Selection {
            x: 1,
            y: 0,
            w: 2,
            h: 1,
        };
        let (changes, moved) = move_selection(&d, &sel, 1, 1);
        assert_eq!(applied(&img, changes), ["0..0", "0112"]);
        assert_eq!(
            moved,
            Some(Selection {
                x: 2,
                y: 1,
                w: 2,
                h: 1
            })
        );
        let (changes, moved) = move_selection(&d, &sel, 2, 0);
        assert_eq!(applied(&img, changes), ["0..1", "0120"]);
        assert_eq!(
            moved,
            Some(Selection {
                x: 3,
                y: 0,
                w: 1,
                h: 1
            })
        );
        assert_eq!(eyedropper(&d, (2, 1)), Some(2));
        assert_eq!(eyedropper(&d, (-1, 0)), None);
    }

    #[test]
    fn copy_and_paste() {
        let img = ["012.", "1200", "0000"];
        let d = doc(&img, &PAL);
        let sel = Selection {
            x: 1,
            y: 0,
            w: 3,
            h: 2,
        };
        let clip = copy(&d, Some(&sel));
        assert_eq!((clip.w, clip.h), (3, 2));
        assert_eq!(clip.indices, vec![1, 2, -1, 2, 0, 0]);
        assert_eq!(copy(&d, None).indices.len(), 12);
        // Pasted at (0, 1): transparency included, clipped at the bottom edge.
        let (changes, placed) = paste(&d, &clip, (0, 1));
        assert_eq!(applied(&img, changes), ["012.", "12.0", "2000"]);
        assert_eq!(
            placed,
            Some(Selection {
                x: 0,
                y: 1,
                w: 3,
                h: 2
            })
        );
        let (changes, placed) = paste(&d, &clip, (3, 2));
        assert_eq!(applied(&img, changes), ["012.", "1200", "0001"]);
        assert_eq!(placed.map(|s| (s.w, s.h)), Some((1, 1)));
        let (changes, placed) = paste(&d, &clip, (9, 9));
        assert!(changes.is_empty() && placed.is_none());
    }

    #[test]
    fn a_stroke_is_one_op() {
        let blank = ["000", "000"];
        let d = doc(&blank, &PAL);
        let mut stroke = Stroke::new("pencil");
        stroke.extend(pencil(&d, None, &[(0, 0), (1, 0)], 1, 1));
        stroke.extend(pencil(&d, None, &[(1, 0), (2, 1)], 1, 2));
        let op = stroke.into_op();
        let Op::Pixels { changes, .. } = &op else {
            panic!()
        };
        assert_eq!(changes.len(), 3);
        assert_eq!(applied(&blank, changes.clone()), ["120", "002"]);
    }
}
