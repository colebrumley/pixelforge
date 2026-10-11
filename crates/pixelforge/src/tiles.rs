//! Tileset / tilemap extraction (Section 9), ported from `pixelforge/tiles.py`.
//!
//! Index images are row-major `i64` slices of explicit `(height, width)`, `-1` (any negative
//! value) = transparent. A tile stack is `n` tiles of `t × t`, concatenated. The extraction is
//! bitwise identical to the numpy version: tile means are sequential sums over the tile's
//! pixels in raster order, the first-row screen and the mean ΔE are numpy's pairwise sums.

use std::collections::HashMap;

use crate::color;
use crate::npsum::pairwise_sum;

/// Tiled GID flip flags (https://doc.mapeditor.org/en/stable/reference/global-tile-ids/).
pub const GID_FLIP_H: i64 = 0x8000_0000;
pub const GID_FLIP_V: i64 = 0x4000_0000;
/// ΔE charged when a transparent pixel is compared with an opaque one.
pub const TRANSPARENT_DISTANCE: f64 = 100.0;
/// DEVIATION: Section 9 — near-duplicate tiles must also agree pixel by pixel: every pixel's
/// ΔE (LAB units) must be below this cap, whatever `tile_dedupe_tolerance` is. A mean-ΔE test
/// alone lets a few very different pixels (a star, a highlight) merge into a plain tile and
/// vanish. Transparent-vs-opaque pixels cost TRANSPARENT_DISTANCE, so they never merge.
pub const TILE_DEDUPE_MAX_PIXEL_DELTA: f64 = 10.0;

/// (flip_h, flip_v) in the order they are tried.
pub const FLIPS: [(i64, i64); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// The `out_tilemap.json` payload. Entries are `[tile_id, flip_h, flip_v]`, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tilemap {
    pub tile_size: i64,
    pub width_tiles: i64,
    pub height_tiles: i64,
    pub tileset_columns: i64,
    pub tiles: Vec<[i64; 3]>,
}

/// A tileset (`n` tiles of `t × t`, -1 = transparent) and the tilemap that draws the image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tiles {
    pub tiles: Vec<i64>,
    pub n_tiles: usize,
    pub tile_size: usize,
    pub tilemap: Tilemap,
}

/// The Config fields [`extract`] reads.
#[derive(Clone, Debug, PartialEq)]
pub struct TileConfig {
    pub tile_size: i64,
    pub tile_dedupe_tolerance: f64,
    pub tileset_columns: i64,
}

impl From<&crate::config::Config> for TileConfig {
    fn from(c: &crate::config::Config) -> TileConfig {
        TileConfig {
            tile_size: c.tile_size,
            tile_dedupe_tolerance: c.tile_dedupe_tolerance,
            tileset_columns: c.tileset_columns,
        }
    }
}

/// Tiles per tileset row: `requested`, or ceil(sqrt(n_tiles)) when it is 0.
pub fn tileset_columns(n_tiles: usize, requested: i64) -> i64 {
    // DEVIATION: Section 9 — the spec fixes 16 columns, which leaves a 15-tile sheet 94 %
    // empty. `tileset_columns=16` restores it; the default 0 makes a near-square sheet.
    if requested != 0 {
        return requested;
    }
    let m = n_tiles.max(1) - 1;
    (m.isqrt() + 1).max(1) as i64
}

/// A `t × t` tile flipped horizontally (columns reversed) and/or vertically (rows reversed).
pub fn flip(tile: &[i64], t: usize, flip_h: bool, flip_v: bool) -> Vec<i64> {
    let mut out = Vec::with_capacity(t * t);
    for y in 0..t {
        let sy = if flip_v { t - 1 - y } else { y };
        for x in 0..t {
            let sx = if flip_h { t - 1 - x } else { x };
            out.push(tile[sy * t + sx]);
        }
    }
    out
}

/// Cut `idx` into tiles, merging exact duplicates, near-duplicates and flipped copies.
///
/// A map entry `[tile_id, flip_h, flip_v]` means: draw tileset tile `tile_id` with those
/// flips. An exact match (any flip) is used when one exists, preferring the lowest tile id,
/// then the flip order in [`FLIPS`]. Otherwise a tile is a near-duplicate of a tileset entry
/// (any flip) only if the mean ΔE over the tile is below `tile_dedupe_tolerance` AND every
/// pixel is within [`TILE_DEDUPE_MAX_PIXEL_DELTA`]; the nearest qualifying entry wins (lowest
/// mean ΔE, then lowest tile id, then flip order). Tolerance 0 merges exact copies only, so
/// `reconstruct(extract(idx)) == idx`.
///
/// Errors when the tile size does not divide the image or an index is outside the palette.
pub fn extract(
    idx: &[i64],
    h: usize,
    w: usize,
    palette_lab: &[f64],
    config: &TileConfig,
) -> Result<Tiles, String> {
    assert_eq!(idx.len(), h * w, "extract: idx length is not h*w");
    let ts = config.tile_size;
    if ts <= 0 || h as i64 % ts != 0 || w as i64 % ts != 0 {
        return Err(format!(
            "tile_size ({ts}) must divide the output size ({w}x{h})"
        ));
    }
    let t = ts as usize;
    let tt = t * t;
    let tolerance = config.tile_dedupe_tolerance;
    let n_colors = palette_lab.len() / 3;
    let clear = n_colors; // code for a transparent pixel
    if idx.iter().any(|&v| v >= n_colors as i64) {
        return Err("extract: index out of the palette's range".to_string());
    }
    let nc = n_colors + 1;
    let mut distances = vec![TRANSPARENT_DISTANCE; nc * nc];
    let block = color::palette_distance_matrix(palette_lab);
    for a in 0..n_colors {
        distances[a * nc..a * nc + n_colors]
            .copy_from_slice(&block[a * n_colors..(a + 1) * n_colors]);
    }
    distances[clear * nc + clear] = 0.0;
    // LAB per code, transparent = 0 so that sums over a tile only count opaque pixels.
    let mut lab = palette_lab.to_vec();
    lab.extend_from_slice(&[0.0, 0.0, 0.0]);

    let (ht, wt) = (h / t, w / t);
    let n_cells = ht * wt;
    let mut cells = Vec::with_capacity(n_cells * tt);
    for ty in 0..ht {
        for tx in 0..wt {
            for y in 0..t {
                for x in 0..t {
                    let v = idx[(ty * t + y) * w + tx * t + x];
                    cells.push(if v < 0 { clear as i64 } else { v });
                }
            }
        }
    }
    // Pruning keys. Flips preserve both. Equal opaque counts are necessary because any
    // transparent/opaque mismatch costs TRANSPARENT_DISTANCE > the per-pixel cap. By
    // convexity of CIE76, ΔE(mean colors) <= mean over opaque pixels of ΔE = mean ΔE · t² /
    // n_opaque, and <= max per-pixel ΔE, so the bound below never drops a qualifying
    // candidate.
    let mut n_opaque = Vec::with_capacity(n_cells);
    let mut means = Vec::with_capacity(n_cells * 3);
    for cell in cells.chunks_exact(tt) {
        let n = cell.iter().filter(|&&c| c != clear as i64).count();
        n_opaque.push(n);
        // lab[cells].sum(axis=(1, 2)): sequential over the pixels in raster order.
        let mut sum = [0.0f64; 3];
        for &c in cell {
            let c = c as usize;
            for (s, &v) in sum.iter_mut().zip(&lab[c * 3..c * 3 + 3]) {
                *s += v;
            }
        }
        let div = n.max(1) as f64;
        means.extend(sum.iter().map(|s| s / div));
    }

    let dist = |a: i64, b: i64| distances[a as usize * nc + b as usize];
    let mut unique: Vec<i64> = Vec::new();
    let mut unique_means: Vec<f64> = Vec::new();
    let mut unique_opaque: Vec<usize> = Vec::new();
    let mut by_digest: HashMap<Vec<i64>, usize> = HashMap::new();
    let mut entries = Vec::with_capacity(n_cells);
    let limit = tolerance * t as f64 * t as f64;
    let mut row = vec![0.0f64; t];
    let mut d = vec![0.0f64; tt];
    for i in 0..n_cells {
        let tile = &cells[i * tt..(i + 1) * tt];
        // Flips are involutions: flip(unique) == tile  ⇔  unique == flip(tile).
        let variants: Vec<Vec<i64>> = FLIPS
            .iter()
            .map(|&(fh, fv)| flip(tile, t, fh != 0, fv != 0))
            .collect();
        let n_unique = unique_opaque.len();
        // min over (tile id, flip order) of the exact matches.
        let mut matched: Option<(usize, usize)> = variants
            .iter()
            .enumerate()
            .filter_map(|(order, v)| by_digest.get(v).map(|&id| (id, order)))
            .min();
        if matched.is_none() && n_unique > 0 && tolerance > 0.0 && n_opaque[i] > 0 {
            let raw = tolerance * t as f64 * t as f64 / n_opaque[i] as f64;
            let bound = if TILE_DEDUPE_MAX_PIXEL_DELTA < raw {
                TILE_DEDUPE_MAX_PIXEL_DELTA
            } else {
                raw
            } + 1e-9;
            let mean_i = &means[i * 3..i * 3 + 3];
            let candidates: Vec<usize> = (0..n_unique)
                .filter(|&k| {
                    unique_opaque[k] == n_opaque[i]
                        && color::delta_e(&unique_means[k * 3..k * 3 + 3], mean_i)[0] <= bound
                })
                // Cheap screen on the first row: one pixel at or over the cap, or a partial
                // ΔE sum that already reaches tolerance · t², rules a (flip, candidate) out.
                .filter(|&k| {
                    let first = &unique[k * tt..k * tt + t];
                    variants.iter().any(|v| {
                        for x in 0..t {
                            row[x] = dist(v[x], first[x]);
                        }
                        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        max < TILE_DEDUPE_MAX_PIXEL_DELTA && pairwise_sum(&row) < limit
                    })
                })
                .collect();
            // score = mean ΔE where ok, inf elsewhere; the minimum, first by tile id then
            // flip order (np.lexsort((orders, ks))[0]).
            let mut best: Option<(f64, usize, usize)> = None;
            for &k in &candidates {
                let u = &unique[k * tt..(k + 1) * tt];
                for (order, v) in variants.iter().enumerate() {
                    for p in 0..tt {
                        d[p] = dist(v[p], u[p]);
                    }
                    let mean_d = pairwise_sum(&d) / tt as f64;
                    let max = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let ok = mean_d < tolerance && max < TILE_DEDUPE_MAX_PIXEL_DELTA;
                    if ok && best.is_none_or(|(s, _, _)| mean_d < s) {
                        best = Some((mean_d, k, order));
                    }
                }
            }
            if let Some((_, k, order)) = best {
                matched = Some((k, order));
            }
        }
        let (tile_id, order) = match matched {
            Some(m) => m,
            None => {
                by_digest.insert(tile.to_vec(), n_unique);
                unique.extend_from_slice(tile);
                unique_means.extend_from_slice(&means[i * 3..i * 3 + 3]);
                unique_opaque.push(n_opaque[i]);
                (n_unique, 0)
            }
        };
        entries.push([tile_id as i64, FLIPS[order].0, FLIPS[order].1]);
    }

    let n_unique = unique_opaque.len();
    let tilemap = Tilemap {
        tile_size: ts,
        width_tiles: wt as i64,
        height_tiles: ht as i64,
        tileset_columns: tileset_columns(n_unique, config.tileset_columns),
        tiles: entries,
    };
    for v in unique.iter_mut() {
        if *v == clear as i64 {
            *v = -1;
        }
    }
    Ok(Tiles {
        tiles: unique,
        n_tiles: n_unique,
        tile_size: t,
        tilemap,
    })
}

/// The tileset as an index image: row-major grid, `columns` tiles wide; only the last row is
/// padded (with transparent pixels). Returns `(sheet, height, width)`.
pub fn tileset_indices(tiles: &[i64], t: usize, columns: usize) -> (Vec<i64>, usize, usize) {
    assert!(
        t > 0 && columns > 0,
        "tileset_indices: zero tile size or columns"
    );
    let n = tiles.len() / (t * t);
    let rows = n.div_ceil(columns);
    let (sh, sw) = (rows * t, columns * t);
    let mut sheet = vec![-1i64; sh * sw];
    for (i, tile) in tiles.chunks_exact(t * t).enumerate() {
        let (y, x) = ((i / columns) * t, (i % columns) * t);
        for r in 0..t {
            sheet[(y + r) * sw + x..(y + r) * sw + x + t]
                .copy_from_slice(&tile[r * t..(r + 1) * t]);
        }
    }
    (sheet, sh, sw)
}

/// Rebuild the index image described by a tileset and tilemap. Returns `(image, h, w)`.
pub fn reconstruct(tiles: &[i64], tilemap: &Tilemap) -> Result<(Vec<i64>, usize, usize), String> {
    let t = usize::try_from(tilemap.tile_size)
        .ok()
        .filter(|&t| t > 0)
        .ok_or("reconstruct: tile_size must be positive")?;
    let wt = usize::try_from(tilemap.width_tiles).map_err(|_| "reconstruct: negative width")?;
    let ht = usize::try_from(tilemap.height_tiles).map_err(|_| "reconstruct: negative height")?;
    if tilemap.tiles.len() > wt * ht {
        return Err("reconstruct: more tilemap entries than tiles in the map".into());
    }
    let n = tiles.len() / (t * t);
    let (h, w) = (ht * t, wt * t);
    let mut out = vec![-1i64; h * w];
    for (i, &[tile_id, fh, fv]) in tilemap.tiles.iter().enumerate() {
        let id = usize::try_from(tile_id)
            .ok()
            .filter(|&id| id < n)
            .ok_or_else(|| format!("reconstruct: tile id {tile_id} out of range"))?;
        let tile = flip(&tiles[id * t * t..(id + 1) * t * t], t, fh != 0, fv != 0);
        let (y, x) = ((i / wt) * t, (i % wt) * t);
        for r in 0..t {
            out[(y + r) * w + x..(y + r) * w + x + t].copy_from_slice(&tile[r * t..(r + 1) * t]);
        }
    }
    Ok((out, h, w))
}

/// Tiled global tile ids, row-major (firstgid 1), with the flip flags when `flips`.
pub fn gids(tilemap: &Tilemap, flips: bool) -> Vec<i64> {
    tilemap
        .tiles
        .iter()
        .map(|&[tile_id, fh, fv]| {
            let mut gid = tile_id + 1;
            if flips {
                gid |=
                    (if fh != 0 { GID_FLIP_H } else { 0 }) | (if fv != 0 { GID_FLIP_V } else { 0 });
            }
            gid
        })
        .collect()
}

/// The embedded tileset of a Tiled map; fields in the order of the Python dict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiledTileset {
    pub firstgid: i64,
    pub name: String,
    pub image: String,
    pub imagewidth: i64,
    pub imageheight: i64,
    pub columns: i64,
    pub tilecount: i64,
    pub tilewidth: i64,
    pub tileheight: i64,
    pub margin: i64,
    pub spacing: i64,
}

/// The tile layer of a Tiled map; fields in the order of the Python dict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiledLayer {
    pub id: i64,
    pub name: &'static str,
    pub kind: &'static str, // "type"
    pub x: i64,
    pub y: i64,
    pub width: i64,
    pub height: i64,
    pub opacity: i64,
    pub visible: bool,
    pub data: Vec<i64>,
}

/// A Tiled 1.10 JSON map (.tmj); fields in the order of the Python dict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiledMap {
    pub kind: &'static str, // "type"
    pub version: &'static str,
    pub orientation: &'static str,
    pub renderorder: &'static str,
    pub infinite: bool,
    pub width: i64,
    pub height: i64,
    pub tilewidth: i64,
    pub tileheight: i64,
    pub nextlayerid: i64,
    pub nextobjectid: i64,
    pub layers: Vec<TiledLayer>,
    pub tilesets: Vec<TiledTileset>,
}

/// A Tiled JSON map: one tile layer and one embedded tileset whose image is `image` (a path
/// relative to the .tmj) of `image_size` = (width, height) pixels.
pub fn tiled_map(
    tilemap: &Tilemap,
    n_tiles: i64,
    image: &str,
    image_size: (i64, i64),
    name: &str,
) -> TiledMap {
    let t = tilemap.tile_size;
    let (wt, ht) = (tilemap.width_tiles, tilemap.height_tiles);
    let tileset = TiledTileset {
        firstgid: 1,
        name: name.to_string(),
        image: image.to_string(),
        imagewidth: image_size.0,
        imageheight: image_size.1,
        columns: tilemap.tileset_columns,
        tilecount: n_tiles,
        tilewidth: t,
        tileheight: t,
        margin: 0,
        spacing: 0,
    };
    let layer = TiledLayer {
        id: 1,
        name: "background",
        kind: "tilelayer",
        x: 0,
        y: 0,
        width: wt,
        height: ht,
        opacity: 1,
        visible: true,
        data: gids(tilemap, true),
    };
    TiledMap {
        kind: "map",
        version: "1.10",
        orientation: "orthogonal",
        renderorder: "right-down",
        infinite: false,
        width: wt,
        height: ht,
        tilewidth: t,
        tileheight: t,
        nextlayerid: 2,
        nextobjectid: 1,
        layers: vec![layer],
        tilesets: vec![tileset],
    }
}

/// One line per tile row of comma-separated GIDs (firstgid 1, no flip flags).
pub fn tilemap_csv(tilemap: &Tilemap) -> String {
    let values = gids(tilemap, false);
    let wt = tilemap.width_tiles.max(1) as usize;
    let rows: Vec<String> = values
        .chunks(wt)
        .map(|r| {
            r.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    rows.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(tol: f64) -> TileConfig {
        TileConfig {
            tile_size: 2,
            tile_dedupe_tolerance: tol,
            tileset_columns: 0,
        }
    }

    #[test]
    fn checker_tiles_with_flips() {
        // Four 2×2 tiles side by side: A, A flipped horizontally, A flipped vertically, B.
        // A = [0 1; 2 2]; hflip = [1 0; 2 2]; vflip = [2 2; 0 1]; B = all 3.
        let (h, w) = (2, 8);
        let idx: Vec<i64> = vec![0, 1, 1, 0, 2, 2, 3, 3, 2, 2, 2, 2, 0, 1, 3, 3];
        let palette = [
            10.0, 0.0, 0.0, 30.0, 0.0, 0.0, 50.0, 0.0, 0.0, 90.0, 0.0, 0.0,
        ];
        let out = extract(&idx, h, w, &palette, &config(0.0)).unwrap();
        assert_eq!(out.n_tiles, 2);
        assert_eq!(
            out.tilemap.tiles,
            vec![[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
        );
        assert_eq!(out.tilemap.tileset_columns, 2);
        let (back, bh, bw) = reconstruct(&out.tiles, &out.tilemap).unwrap();
        assert_eq!((bh, bw), (h, w));
        assert_eq!(back, idx);
        assert_eq!(
            gids(&out.tilemap, true),
            vec![1, 0x8000_0001, 0x4000_0001, 2]
        );
        assert_eq!(tilemap_csv(&out.tilemap), "1,1,1,2\n");
        let (sheet, sh, sw) = tileset_indices(&out.tiles, 2, 3);
        assert_eq!((sh, sw), (2, 6));
        assert_eq!(sheet[4..6], [-1, -1]);
    }

    #[test]
    fn near_duplicates_merge_within_tolerance() {
        // Tile 2 differs from tile 1 by a ΔE of 4 in one pixel of four: mean 1.
        let (h, w) = (2, 4);
        let idx: Vec<i64> = vec![0, 0, 0, 1, 0, 0, 0, 0];
        let palette = [50.0, 0.0, 0.0, 54.0, 0.0, 0.0];
        let strict = extract(&idx, h, w, &palette, &config(0.0)).unwrap();
        assert_eq!(strict.n_tiles, 2);
        let loose = extract(&idx, h, w, &palette, &config(2.0)).unwrap();
        assert_eq!(loose.n_tiles, 1);
        assert_eq!(loose.tilemap.tiles, vec![[0, 0, 0], [0, 0, 0]]);
        assert!(extract(
            &idx,
            h,
            w,
            &palette,
            &TileConfig {
                tile_size: 3,
                ..config(0.0)
            }
        )
        .unwrap_err()
        .contains("must divide the output size (4x2)"));
    }

    #[test]
    fn columns() {
        assert_eq!(tileset_columns(0, 0), 1);
        assert_eq!(tileset_columns(1, 0), 1);
        assert_eq!(tileset_columns(15, 0), 4);
        assert_eq!(tileset_columns(16, 0), 4);
        assert_eq!(tileset_columns(17, 0), 5);
        assert_eq!(tileset_columns(17, 16), 16);
    }
}
