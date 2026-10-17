//! The canvas view: zoom and pan, and the mapping between screen positions and image pixels
//! (docs/editor.md, Section 3.7). Everything is integer: the zoom is a power of two from
//! [`MIN_ZOOM`] to [`MAX_ZOOM`] screen points per image pixel, the pan is whole points, and
//! a pointer position maps to a pixel by floor division. Nothing here enters the edit
//! numerics; it only decides which pixel a pointer is over.

/// The smallest zoom (one point per pixel).
pub const MIN_ZOOM: i64 = 1;
/// The largest zoom.
pub const MAX_ZOOM: i64 = 64;
/// From this zoom on, the pixel grid is drawn (R19).
pub const GRID_ZOOM: i64 = 8;

/// Where the image is on the canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    /// Screen points per image pixel: a power of two in `MIN_ZOOM..=MAX_ZOOM`.
    pub zoom: i64,
    /// The image's top-left corner relative to the canvas's top-left corner, in points.
    pub pan: (i64, i64),
}

impl Default for View {
    fn default() -> View {
        View {
            zoom: MIN_ZOOM,
            pan: (0, 0),
        }
    }
}

impl View {
    /// The largest zoom at which a `w × h` image fits a `view_w × view_h` canvas (at least
    /// [`MIN_ZOOM`]), centered.
    pub fn fit(w: usize, h: usize, view_w: i64, view_h: i64) -> View {
        let (w, h) = (w.max(1) as i64, h.max(1) as i64);
        let mut zoom = MIN_ZOOM;
        while zoom < MAX_ZOOM && w * zoom * 2 <= view_w && h * zoom * 2 <= view_h {
            zoom *= 2;
        }
        View {
            zoom,
            pan: (
                (view_w - w * zoom).div_euclid(2),
                (view_h - h * zoom).div_euclid(2),
            ),
        }
    }

    /// The pixel under a canvas position (points from the canvas's top-left corner, already
    /// floored to whole points). May be outside the image.
    pub fn pixel_at(&self, pos: (i64, i64)) -> (i64, i64) {
        (
            (pos.0 - self.pan.0).div_euclid(self.zoom),
            (pos.1 - self.pan.1).div_euclid(self.zoom),
        )
    }

    /// The canvas position of a pixel's top-left corner.
    pub fn screen_of(&self, pixel: (i64, i64)) -> (i64, i64) {
        (
            self.pan.0 + pixel.0 * self.zoom,
            self.pan.1 + pixel.1 * self.zoom,
        )
    }

    /// Zoom in (`steps > 0`) or out by powers of two, keeping the pixel under `anchor` (a
    /// canvas position) where it is.
    pub fn zoom_by(&mut self, steps: i32, anchor: (i64, i64)) {
        let mut zoom = self.zoom;
        for _ in 0..steps.unsigned_abs() {
            zoom = if steps > 0 { zoom * 2 } else { zoom / 2 };
        }
        let zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        if zoom == self.zoom {
            return;
        }
        // The image offset under the anchor, in points at the old zoom, scaled exactly (the
        // ratio is a power of two; zooming out drops the sub-pixel part).
        let scale = |d: i64| {
            if zoom > self.zoom {
                d * (zoom / self.zoom)
            } else {
                d.div_euclid(self.zoom / zoom)
            }
        };
        let (dx, dy) = (anchor.0 - self.pan.0, anchor.1 - self.pan.1);
        self.pan = (anchor.0 - scale(dx), anchor.1 - scale(dy));
        self.zoom = zoom;
    }

    /// Move the image by `(dx, dy)` points.
    pub fn pan_by(&mut self, dx: i64, dy: i64) {
        self.pan = (self.pan.0 + dx, self.pan.1 + dy);
    }

    /// Keep at least `margin` points of a `w × h` image inside a `view_w × view_h` canvas,
    /// so a pan cannot lose the image.
    pub fn keep_visible(&mut self, w: usize, h: usize, view_w: i64, view_h: i64, margin: i64) {
        let (iw, ih) = (w as i64 * self.zoom, h as i64 * self.zoom);
        let clamp = |p: i64, size: i64, view: i64| {
            let m = margin.min(size).min(view);
            p.clamp(m - size, view - m)
        };
        self.pan = (clamp(self.pan.0, iw, view_w), clamp(self.pan.1, ih, view_h));
    }

    /// The pixels of a `w × h` image visible on a `view_w × view_h` canvas: `(x0, y0, x1,
    /// y1)`, ends excluded (empty when the image is off screen).
    pub fn visible(&self, w: usize, h: usize, view_w: i64, view_h: i64) -> (i64, i64, i64, i64) {
        let (x0, y0) = self.pixel_at((0, 0));
        let (x1, y1) = self.pixel_at((view_w - 1, view_h - 1));
        (
            x0.clamp(0, w as i64),
            y0.clamp(0, h as i64),
            (x1 + 1).clamp(0, w as i64),
            (y1 + 1).clamp(0, h as i64),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_picks_the_largest_power_of_two() {
        assert_eq!(
            View::fit(16, 8, 100, 100),
            View {
                zoom: 4,
                pan: (18, 34)
            }
        );
        assert_eq!(View::fit(1, 1, 1000, 1000).zoom, MAX_ZOOM);
        assert_eq!(View::fit(4096, 4096, 800, 600).zoom, MIN_ZOOM);
    }

    #[test]
    fn pixel_mapping_is_floor_division() {
        let v = View {
            zoom: 8,
            pan: (10, -4),
        };
        assert_eq!(v.pixel_at((10, -4)), (0, 0));
        assert_eq!(v.pixel_at((17, 3)), (0, 0));
        assert_eq!(v.pixel_at((18, 4)), (1, 1));
        assert_eq!(v.pixel_at((9, -5)), (-1, -1));
        assert_eq!(v.screen_of((1, 1)), (18, 4));
    }

    #[test]
    fn zoom_keeps_the_anchor_pixel() {
        let mut v = View {
            zoom: 4,
            pan: (3, 5),
        };
        let anchor = (50, 41);
        let before = v.pixel_at(anchor);
        v.zoom_by(1, anchor);
        assert_eq!((v.zoom, v.pixel_at(anchor)), (8, before));
        v.zoom_by(-2, anchor);
        assert_eq!((v.zoom, v.pixel_at(anchor)), (2, before));
        v.zoom_by(10, anchor);
        assert_eq!(v.zoom, MAX_ZOOM);
        v.zoom_by(-10, anchor);
        assert_eq!(v.zoom, MIN_ZOOM);
    }

    #[test]
    fn visible_range_and_pan_limits() {
        let mut v = View {
            zoom: 2,
            pan: (-5, 0),
        };
        assert_eq!(v.visible(100, 10, 20, 100), (2, 0, 13, 10));
        v.pan_by(-1000, 0);
        v.keep_visible(100, 10, 20, 100, 8);
        assert_eq!(v.pan.0, 8 - 200);
    }
}
