//! Port of tests/test_postprocess.py: orphan removal, jaggy fix, outline and the full pass.
//! Index images are row-major `Vec<i64>` with explicit `(h, w)`; palettes are interleaved
//! LAB triples.

mod common;

use common::*;
use pixelforge::color;
use pixelforge::postprocess::{self, PostConfig, ORPHAN_MAX_DELTA, ORPHAN_MAX_PASSES};

/// dark, light, red.
fn palette3() -> Vec<f64> {
    color::rgb8_to_lab(&[20, 20, 20, 240, 240, 240, 200, 40, 40])
}

/// dark, light, red, near-dark (ΔE 4.8 from dark), near-light (ΔE 3.9 from light).
fn near() -> Vec<f64> {
    color::rgb8_to_lab(&[
        20, 20, 20, 240, 240, 240, 200, 40, 40, 32, 26, 26, 228, 232, 236,
    ])
}

/// Entry `i` of an interleaved palette.
fn entry(palette: &[f64], i: usize) -> &[f64] {
    &palette[i * 3..i * 3 + 3]
}

/// `np.allclose` (rtol 1e-5, atol 1e-8).
fn allclose(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-8 + 1e-5 * y.abs())
}

/// An `(h, w)` index image.
#[derive(Clone, Debug, PartialEq)]
struct Grid {
    h: usize,
    w: usize,
    v: Vec<i64>,
}

impl Grid {
    fn full(h: usize, w: usize, value: i64) -> Grid {
        Grid {
            h,
            w,
            v: vec![value; h * w],
        }
    }

    fn get(&self, y: usize, x: usize) -> i64 {
        self.v[y * self.w + x]
    }

    fn set(&mut self, y: usize, x: usize, value: i64) {
        self.v[y * self.w + x] = value;
    }

    /// `idx[y0:y1, x0:x1] = value`.
    fn fill(&mut self, ys: std::ops::Range<usize>, xs: std::ops::Range<usize>, value: i64) {
        for y in ys {
            for x in xs.clone() {
                self.set(y, x, value);
            }
        }
    }

    /// `idx[:, ::-1]`.
    fn flip_lr(&self) -> Grid {
        let mut out = self.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                out.set(y, x, self.get(y, self.w - 1 - x));
            }
        }
        out
    }

    fn with(&self, v: Vec<i64>) -> Grid {
        assert_eq!(v.len(), self.h * self.w);
        Grid {
            h: self.h,
            w: self.w,
            v,
        }
    }

    fn count(&self, value: i64) -> usize {
        self.v.iter().filter(|&&x| x == value).count()
    }

    fn remove_orphans(&self, palette: &[f64], min_region: i64) -> Grid {
        self.remove_orphans_with(palette, min_region, ORPHAN_MAX_PASSES, ORPHAN_MAX_DELTA)
    }

    fn remove_orphans_with(
        &self,
        palette: &[f64],
        min_region: i64,
        max_passes: usize,
        max_delta: f64,
    ) -> Grid {
        self.with(postprocess::remove_orphans(
            &self.v, self.h, self.w, palette, min_region, max_passes, max_delta,
        ))
    }

    fn fix_jaggies(&self) -> Grid {
        self.with(postprocess::fix_jaggies(&self.v, self.h, self.w))
    }

    /// `add_outline(idx, palette, outline, darken, fixed_palette=fixed)` (allow_pad=True).
    fn add_outline(
        &self,
        palette: &[f64],
        outline: &str,
        darken: f64,
        fixed: bool,
    ) -> (Grid, Vec<f64>, postprocess::OutlineStats) {
        let p = postprocess::add_outline(
            &self.v, self.h, self.w, palette, outline, darken, true, fixed,
        )
        .unwrap();
        let grid = Grid {
            h: p.height,
            w: p.width,
            v: p.indices,
        };
        (grid, p.palette_lab, p.stats)
    }
}

#[test]
fn test_orphan_removal_removes_single_stray_pixel_only() {
    let mut idx = Grid::full(12, 12, 0);
    idx.fill(0..12, 6..12, 1);
    idx.set(3, 2, 3); // the stray pixel, within ΔE 5 of its surroundings
    let mut expected = idx.clone();
    expected.set(3, 2, 0);
    let out = idx.remove_orphans(&near(), 2);
    assert_eq!(out, expected);
    assert_eq!(idx.get(3, 2), 3); // input not modified
}

#[test]
fn test_orphans_ignore_transparency_and_keep_pairs() {
    let mut idx = Grid::full(8, 8, -1);
    idx.fill(2..6, 2..6, 0);
    idx.set(2, 2, 3); // low-contrast orphan at the sprite corner → merged into 0
    idx.set(4, 3, 2); // a 2-px region survives min_region=2
    idx.set(4, 4, 2);
    idx.set(0, 0, 1); // isolated pixel with no opaque neighbors stays
    let out = idx.remove_orphans(&near(), 2);
    assert!(out.get(2, 2) == 0 && out.get(4, 3) == 2 && out.get(4, 4) == 2 && out.get(0, 0) == 1);
    let mask = |g: &Grid| g.v.iter().map(|&v| v < 0).collect::<Vec<_>>();
    assert_eq!(mask(&out), mask(&idx));
}

#[test]
fn test_orphans_keep_high_contrast_eyes_with_defaults() {
    let palette = color::rgb8_to_lab(&[230, 200, 170, 40, 30, 30]); // skin, eye
    let mut idx = Grid::full(12, 12, -1);
    idx.fill(1..11, 1..11, 0);
    idx.set(4, 4, 1); // two 1-px eyes, ΔE ≈ 72 from the face
    idx.set(4, 7, 1);
    let cfg = config(&[]);
    let out = idx.remove_orphans_with(
        &palette,
        cfg.orphan_min_region,
        ORPHAN_MAX_PASSES,
        cfg.orphan_max_delta,
    );
    assert_eq!(out, idx);
    let run_cfg = config(&[("palette_ramps", b(false)), ("outline", s("none"))]);
    let run_out = postprocess::run(
        &idx.v,
        12,
        12,
        &palette,
        &PostConfig::from(&run_cfg),
        false,
        false,
        0,
    )
    .unwrap();
    assert_eq!(run_out.indices[4 * 12 + 4], 1);
    assert_eq!(run_out.indices[4 * 12 + 7], 1);
    // Without the exemption they would be erased.
    let merged = idx.remove_orphans_with(&palette, 2, ORPHAN_MAX_PASSES, f64::INFINITY);
    assert_eq!(merged.count(1), 0);
}

#[test]
fn test_orphan_sparkle_does_not_depend_on_pass_parity() {
    for pair in [(1, 4), (1, 2)] {
        // ΔE 3.9 (would swap every pass) and ΔE 89
        let mut idx = Grid::full(6, 6, -1);
        idx.set(2, 2, pair.0);
        idx.set(2, 3, pair.1);
        let results: Vec<Grid> = (1..6)
            .map(|n| idx.remove_orphans_with(&near(), 2, n, ORPHAN_MAX_DELTA))
            .collect();
        for out in &results {
            assert_eq!(out, &results[0], "{pair:?}");
        }
        assert_eq!(results[0], idx, "{pair:?}");
    }
}

#[test]
fn test_orphan_gem_of_distant_colors_survives() {
    let palette = color::rgb8_to_lab(&[
        128, 128, 128, 255, 220, 0, 0, 160, 60, 40, 60, 220, 230, 40, 140,
    ]);
    for background in [-1, 0] {
        // on transparency and inside a grey field
        let mut idx = Grid::full(8, 8, background);
        idx.set(3, 3, 1);
        idx.set(3, 4, 2);
        idx.set(4, 3, 3);
        idx.set(4, 4, 4);
        assert_eq!(
            idx.remove_orphans(&palette, 2),
            idx,
            "background {background}"
        );
    }
}

#[test]
fn test_orphan_merges_into_cluster_once_neighbor_merges() {
    let palette = color::rgb8_to_lab(&[20, 20, 20, 32, 26, 26, 40, 30, 30]);
    let mut idx = Grid::full(7, 7, 0);
    idx.set(1, 3, -1);
    idx.set(2, 2, -1);
    idx.set(2, 4, -1);
    idx.fill(3..5, 2..5, -1);
    idx.set(2, 3, 1); // touches the field: merged in pass 1
    idx.set(3, 3, 2); // touches only the other single: merged in pass 2
    let out = idx.remove_orphans(&palette, 2);
    assert!(out.get(2, 3) == 0 && out.get(3, 3) == 0);
    let one = idx.remove_orphans_with(&palette, 2, 1, ORPHAN_MAX_DELTA);
    assert!(one.get(2, 3) == 0 && one.get(3, 3) == 2);
}

#[test]
fn test_orphan_max_delta_validation() {
    assert_eq!(
        config(&[("orphan_max_delta", int(0))]).orphan_max_delta,
        0.0
    );
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let msg = config_err(&[("orphan_max_delta", float(bad))]);
        assert!(msg.contains("orphan_max_delta"), "{msg}");
    }
}

fn diagonal(n: usize) -> Grid {
    let mut idx = Grid::full(n, n, 0);
    for i in 0..n {
        idx.set(i, i, 1);
    }
    idx
}

#[test]
fn test_jaggy_fix_leaves_clean_diagonal_alone() {
    let line = diagonal(12);
    assert_eq!(line.fix_jaggies(), line);
    assert_eq!(line.flip_lr().fix_jaggies(), line.flip_lr());
}

#[test]
fn test_jaggy_fix_removes_injected_double() {
    let clean = diagonal(12);
    for (dy, dx) in [(0, 1), (1, 0)] {
        // 2×1 and 1×2 doubles
        let mut doubled = clean.clone();
        doubled.set(5 + dy, 5 + dx, 1);
        assert_eq!(doubled.fix_jaggies(), clean);
        assert_eq!(doubled.flip_lr().fix_jaggies(), clean.flip_lr());
    }
}

#[test]
fn test_jaggy_fix_leaves_shapes_alone() {
    let mut idx = Grid::full(10, 10, 0);
    idx.fill(2..8, 2..8, 1); // filled square
    idx.fill(2..8, 2..3, 2); // with an L-shaped 1-px border
    idx.fill(2..3, 2..8, 2);
    assert_eq!(idx.fix_jaggies(), idx);
}

#[test]
fn test_outline_adds_exactly_the_4_neighbor_ring() {
    let sprite = Grid::full(5, 5, 0);
    let palette = palette3();
    let (out, palette_lab, _) = sprite.add_outline(&palette[..6], "auto", 0.55, false);
    assert!(out.h == 7 && out.w == 7 && palette_lab.len() / 3 == 3);
    let mut expected = Grid::full(7, 7, 2);
    expected.fill(1..6, 1..6, 0);
    for (y, x) in [(0, 0), (0, 6), (6, 0), (6, 6)] {
        expected.set(y, x, -1); // diagonal neighbors are not outlined
    }
    assert_eq!(out, expected);
    assert_eq!(out.count(2), 20);
    // auto color = darkest palette color blended toward black
    let want: Vec<f64> = entry(&palette, 0).iter().map(|v| v * 0.45).collect();
    assert!(allclose(entry(&palette_lab, 2), &want));
}

#[test]
fn test_outline_explicit_color_and_no_padding_needed() {
    let mut sprite = Grid::full(6, 6, -1);
    sprite.fill(2..4, 2..4, 1);
    let (out, palette_lab, _) = sprite.add_outline(&palette3(), "#c82828", 0.55, false);
    assert!(out.h == 6 && out.w == 6);
    assert!(palette_lab.len() / 3 == 3 && out.count(2) == 8); // reuses the existing red
}

#[test]
fn test_run_applies_passes_in_order() {
    let mut idx = Grid::full(8, 8, -1);
    idx.fill(2..6, 2..6, 1);
    idx.set(3, 3, 2); // low-contrast orphan inside the sprite
    let n = near();
    let palette: Vec<f64> = [0, 1, 4]
        .iter()
        .flat_map(|&i| entry(&n, i).to_vec())
        .collect();
    let cfg = config(&[("palette_ramps", b(false))]);
    let out = postprocess::run(
        &idx.v,
        8,
        8,
        &palette,
        &PostConfig::from(&cfg),
        false,
        false,
        0,
    )
    .unwrap();
    let grid = Grid {
        h: out.height,
        w: out.width,
        v: out.indices,
    };
    assert!(grid.get(3, 3) == 1 && out.palette_lab.len() / 3 == 4);
    assert!(grid.count(3) == 16 && grid.h == 8 && grid.w == 8);
    let want: Vec<f64> = entry(&palette3(), 1)[1..].iter().map(|v| v * 1.1).collect();
    assert!(allclose(&entry(&out.palette_lab, 1)[1..], &want));
}

#[test]
fn test_outline_fixed_palette_snaps_to_nearest_entry() {
    let mut sprite = Grid::full(6, 6, -1);
    sprite.fill(2..4, 2..4, 1);
    let palette = palette3();
    let (out, palette_lab, stats) = sprite.add_outline(&palette, "auto", 0.55, true);
    assert_eq!(palette_lab, palette); // nothing appended
    assert!(stats.outline_index == Some(0) && out.count(0) == 8); // darkest entry
                                                                  // An explicit color snaps too: dark orange → the red entry.
    let (out, palette_lab, _) = sprite.add_outline(&palette, "#b04010", 0.55, true);
    assert!(palette_lab == palette && out.count(2) == 8);
}

#[test]
fn test_outline_full_palette_overwritten_only_when_not_fixed() {
    let greys: Vec<u8> = (0..=255u8).flat_map(|i| [i, i, i]).collect();
    let full = color::rgb8_to_lab(&greys);
    let mut sprite = Grid::full(6, 6, -1);
    sprite.fill(2..4, 2..4, 200);
    let (_, palette_lab, _) = sprite.add_outline(&full, "#c82828", 0.55, true);
    assert_eq!(palette_lab, full);
    // Non-fixed full palette: the nearest entry is replaced by the outline color (documented).
    let (out, palette_lab, _) = sprite.add_outline(&full, "#c82828", 0.55, false);
    assert!(palette_lab.len() / 3 == 256 && palette_lab != full);
    let red = color::rgb8_to_lab(&color::hex_to_rgb8("#c82828").unwrap());
    assert!(allclose(entry(&palette_lab, out.get(1, 2) as usize), &red));
}
