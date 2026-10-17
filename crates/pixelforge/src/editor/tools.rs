//! The drawing tools (docs/design/editor.md, Section 3.4) as pure functions from the document, the
//! selection and pointer positions to a change list. Positions are pixel coordinates and may
//! lie outside the image (a drag that leaves the canvas); whatever falls outside the image
//! or the selection is dropped. Each change carries the document's current index as `old`.
//!
//! A drag accumulates its change lists in a [`Stroke`] and commits them as one op on
//! release, so a drag is one undo step.

use super::op::normalize;
use super::selection::region;
use super::{Document, Op, PixelChange, Selection};
use crate::color;
use crate::tiles::{self, TileConfig};

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

/// Pixels of an image pasted from the OS clipboard with alpha below this are transparent.
pub const PASTE_ALPHA: u8 = 128;

/// A clip as RGBA8 (transparent pixels `0, 0, 0, 0`): what copying puts on the OS clipboard.
pub fn clip_rgba(doc: &Document, clip: &Clip) -> Vec<u8> {
    let mut out = Vec::with_capacity(clip.indices.len() * 4);
    for &i in &clip.indices {
        match usize::try_from(i).ok().and_then(|i| doc.palette.get(i)) {
            Some(&[r, g, b]) => out.extend_from_slice(&[r, g, b, 255]),
            None => out.extend_from_slice(&[0, 0, 0, 0]),
        }
    }
    out
}

/// An RGBA8 image of `w × h` (from the OS clipboard) as a clip of the document's palette:
/// alpha below [`PASTE_ALPHA`] is transparent, every other pixel takes the nearest entry
/// (`color::nearest_index` on the LAB palette, CIE76, ties to the lowest index; the RGB is
/// converted with `color::rgb8_to_lab`). Only the top-left part that fits the document is
/// kept (a paste is clipped to the canvas anyway). Without palette entries every pixel is
/// transparent. `None` when `rgba` is shorter than `w × h × 4`.
pub fn snap_rgba(doc: &Document, rgba: &[u8], w: usize, h: usize) -> Option<Clip> {
    if rgba.len() < w.checked_mul(h)?.checked_mul(4)? {
        return None;
    }
    let (cw, ch) = (w.min(doc.width()), h.min(doc.height()));
    let mut colors: std::collections::BTreeMap<[u8; 3], i64> = Default::default();
    let mut indices = Vec::with_capacity(cw * ch);
    for y in 0..ch {
        for x in 0..cw {
            let p = &rgba[(y * w + x) * 4..(y * w + x) * 4 + 4];
            if p[3] < PASTE_ALPHA || doc.palette.is_empty() {
                indices.push(-1);
                continue;
            }
            let rgb = [p[0], p[1], p[2]];
            let i = *colors.entry(rgb).or_insert_with(|| {
                color::nearest_index(&color::rgb8_to_lab(&rgb), &doc.palette_lab)[0]
            });
            indices.push(i);
        }
    }
    Some(Clip {
        w: cw,
        h: ch,
        indices,
    })
}

/// Per-tile editing (docs/design/editor.md, Section 3.9): `changes` plus the same change at the
/// corresponding pixel of every other instance of each edited tile, as one change list.
/// Instances are the cells `tiles::extract` maps to one tileset tile with tolerance 0
/// (exact copies, flipped copies included: a change on a mirrored instance lands mirrored),
/// computed on the document before the change. Copies only change inside the selection
/// (R12), and a pixel the caller changes itself keeps the caller's value. Without
/// `config.tileset`, or when the tile size does not divide the image, `changes` as they are.
pub fn mirror_tiles(
    doc: &Document,
    sel: Option<&Selection>,
    changes: &[PixelChange],
) -> Vec<PixelChange> {
    let config = &doc.origin.config;
    if !config.tileset || changes.is_empty() {
        return changes.to_vec();
    }
    let exact = TileConfig {
        tile_dedupe_tolerance: 0.0,
        ..TileConfig::from(config)
    };
    let (w, h) = (doc.width(), doc.height());
    let Ok(extracted) = tiles::extract(&doc.indices.indices, h, w, &doc.palette_lab, &exact) else {
        return changes.to_vec();
    };
    let t = extracted.tile_size;
    let map = &extracted.tilemap.tiles;
    let columns = w / t;
    let mut instances: std::collections::BTreeMap<i64, Vec<usize>> = Default::default();
    for (cell, entry) in map.iter().enumerate() {
        instances.entry(entry[0]).or_default().push(cell);
    }
    let region = region(sel, w, h);
    // Within a tile, (x, y) of a cell drawn with flips (fh, fv) is the tile's pixel
    // (t-1-x if fh else x, t-1-y if fv else y); flips are involutions, so the same map
    // goes back.
    let flip = |v: usize, f: i64| if f != 0 { t - 1 - v } else { v };
    let mut out = Vec::new();
    for c in changes {
        let cell = (c.y / t) * columns + c.x / t;
        let [id, fh, fv] = map[cell];
        let (tx, ty) = (flip(c.x % t, fh), flip(c.y % t, fv));
        for &other in &instances[&id] {
            if other == cell {
                continue;
            }
            let [_, oh, ov] = map[other];
            let x = (other % columns) * t + flip(tx, oh);
            let y = (other / columns) * t + flip(ty, ov);
            if region.contains(x, y) {
                out.push(PixelChange {
                    x,
                    y,
                    old: doc.get(x, y),
                    new: c.new,
                });
            }
        }
    }
    // Last, so the caller's own changes win.
    out.extend_from_slice(changes);
    normalize(out)
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
    fn os_clipboard_images_snap_to_the_palette() {
        let d = doc(&["0120", "0000"], &PAL);
        // Copying gives the palette colors, transparency as zeros.
        let clip = copy(&d, None);
        let rgba = clip_rgba(&d, &clip);
        assert_eq!(&rgba[4..12], &[255, 255, 255, 255, 255, 0, 0, 255]);
        // An exact image snaps back to the same indices.
        let mut img = doc(&["01.2"], &PAL);
        img.palette = PAL.to_vec();
        let exact = copy(&img, None);
        let back = snap_rgba(&d, &clip_rgba(&img, &exact), 4, 1).unwrap();
        assert_eq!(back, exact);
        // Foreign colors take the nearest entry; low alpha is transparent.
        let foreign = [
            10, 5, 0, 255, // near black
            240, 250, 250, 200, // near white
            200, 40, 30, 128, // near red, alpha at the threshold
            255, 255, 255, 127, // below the threshold
            250, 10, 0, 255, // only the part that fits is kept
            9, 9, 9, 255, //
        ];
        let clip = snap_rgba(&d, &foreign, 3, 2).unwrap();
        assert_eq!((clip.w, clip.h), (3, 2));
        assert_eq!(clip.indices, [0, 1, 2, -1, 2, 0]);
        let wide: Vec<u8> = [9u8, 9, 9, 255].repeat(6 * 3);
        let clip = snap_rgba(&d, &wide, 6, 3).unwrap();
        assert_eq!((clip.w, clip.h, clip.indices.len()), (4, 2, 8));
        let (changes, placed) = paste(&d, &clip, (1, 1));
        assert_eq!(applied(&["0120", "0000"], changes), ["0120", "0000"]);
        assert_eq!(placed.map(|s| (s.x, s.y, s.w, s.h)), Some((1, 1, 3, 1)));
        // A short buffer is refused.
        assert_eq!(snap_rgba(&d, &foreign, 4, 2), None);
    }

    #[test]
    fn per_tile_edits_reach_every_instance() {
        use crate::config::{Config, Value};
        // Three 2 × 2 cells: a tile, an exact copy, and a mirrored (flip_h) copy.
        let mut d = doc(&["010110", "222222"], &PAL);
        let paint = |d: &Document, sel: Option<&Selection>| {
            let direct = pencil(d, None, &[(0, 1)], 1, 0);
            applied(&["010110", "222222"], mirror_tiles(d, sel, &direct))
        };
        // Without a tileset: only the painted pixel.
        assert_eq!(paint(&d, None), ["010110", "022222"]);
        d.origin.config = Config::new(&[
            ("tileset".into(), Value::Bool(true)),
            ("tile_size".into(), Value::Int(2)),
        ])
        .unwrap();
        assert_eq!(paint(&d, None), ["010110", "020220"]);
        // Copies stay inside the selection.
        let sel = Selection {
            x: 0,
            y: 0,
            w: 4,
            h: 2,
        };
        assert_eq!(paint(&d, Some(&sel)), ["010110", "020222"]);
        // A cell that is not a copy is untouched; the caller's change wins on overlap.
        let d2 = Document {
            origin: d.origin.clone(),
            ..doc(&["010100", "222222"], &PAL)
        };
        let direct = [
            PixelChange {
                x: 0,
                y: 0,
                old: 0,
                new: 2,
            },
            PixelChange {
                x: 2,
                y: 0,
                old: 0,
                new: 1,
            },
        ];
        let all = mirror_tiles(&d2, None, &direct);
        assert_eq!(applied(&["010100", "222222"], all), ["211100", "222222"]);
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
