//! The rectangular selection, in pixel coordinates. Tools never change a pixel outside it
//! (R12); without a selection they work on the whole image.

/// A non-empty rectangle `x .. x + w` by `y .. y + h` inside the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Selection {
    /// The whole `width × height` image.
    pub fn all(width: usize, height: usize) -> Selection {
        Selection {
            x: 0,
            y: 0,
            w: width,
            h: height,
        }
    }

    /// The rectangle between two pixel positions (both included, in any order), clipped to
    /// the image; `None` when nothing of it is inside.
    pub fn from_corners(
        a: (i64, i64),
        b: (i64, i64),
        width: usize,
        height: usize,
    ) -> Option<Selection> {
        let (x0, x1) = (a.0.min(b.0).max(0), a.0.max(b.0).min(width as i64 - 1));
        let (y0, y1) = (a.1.min(b.1).max(0), a.1.max(b.1).min(height as i64 - 1));
        if x0 > x1 || y0 > y1 {
            return None;
        }
        Some(Selection {
            x: x0 as usize,
            y: y0 as usize,
            w: (x1 - x0 + 1) as usize,
            h: (y1 - y0 + 1) as usize,
        })
    }

    /// Whether `(x, y)` is inside.
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }

    /// Whether a pointer position (possibly negative) is inside.
    pub fn contains_i(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && self.contains(x as usize, y as usize)
    }

    /// The rectangle moved by `(dx, dy)` and clipped to the image.
    pub fn translated(&self, dx: i64, dy: i64, width: usize, height: usize) -> Option<Selection> {
        let x = self.x as i64 + dx;
        let y = self.y as i64 + dy;
        Selection::from_corners(
            (x, y),
            (x + self.w as i64 - 1, y + self.h as i64 - 1),
            width,
            height,
        )
    }

    /// The rectangle grown by `margin` on every side, clipped to the image.
    pub fn grown(&self, margin: usize, width: usize, height: usize) -> Selection {
        let x0 = self.x.saturating_sub(margin);
        let y0 = self.y.saturating_sub(margin);
        let x1 = (self.x + self.w + margin).min(width);
        let y1 = (self.y + self.h + margin).min(height);
        Selection {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    /// Every position inside, row-major.
    pub fn positions(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (self.y..self.y + self.h).flat_map(move |y| (self.x..self.x + self.w).map(move |x| (x, y)))
    }
}

/// The region a tool works in: `selection`, or the whole image.
pub fn region(selection: Option<&Selection>, width: usize, height: usize) -> Selection {
    selection
        .copied()
        .unwrap_or_else(|| Selection::all(width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_clip_and_order() {
        let s = Selection::from_corners((5, -2), (1, 3), 4, 10).unwrap();
        assert_eq!(
            s,
            Selection {
                x: 1,
                y: 0,
                w: 3,
                h: 4
            }
        );
        assert!(Selection::from_corners((5, 0), (7, 1), 4, 4).is_none());
        assert!(s.contains(3, 3) && !s.contains(0, 0) && !s.contains_i(-1, 0));
        assert_eq!(
            s.translated(-2, 1, 4, 10),
            Some(Selection {
                x: 0,
                y: 1,
                w: 2,
                h: 4
            })
        );
        assert_eq!(
            s.grown(1, 4, 10),
            Selection {
                x: 0,
                y: 0,
                w: 4,
                h: 5
            }
        );
        assert_eq!(s.positions().count(), 12);
    }
}
