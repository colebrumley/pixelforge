"""Tileset / tilemap extraction (Section 9).

The extraction, reconstruction and Tiled export run in the native library
(crates/pixelforge/src/tiles.rs), bit for bit as the numpy version did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _native

# Tiled GID flip flags (https://doc.mapeditor.org/en/stable/reference/global-tile-ids/).
GID_FLIP_H = 0x80000000
GID_FLIP_V = 0x40000000
# ΔE charged when a transparent pixel is compared with an opaque one.
TRANSPARENT_DISTANCE = 100.0
# DEVIATION: Section 9 — near-duplicate tiles must also agree pixel by pixel: every pixel's
# ΔE (LAB units) must be below this cap, whatever `tile_dedupe_tolerance` is. A mean-ΔE test
# alone lets a few very different pixels (a star, a highlight) merge into a plain tile and
# vanish. Transparent-vs-opaque pixels cost TRANSPARENT_DISTANCE, so they never merge.
TILE_DEDUPE_MAX_PIXEL_DELTA = 10.0


def _stack(tiles) -> np.ndarray:
    return np.ascontiguousarray(tiles, dtype=np.int64)


@dataclass
class Tiles:
    tiles: np.ndarray     # (n, t, t) index arrays, -1 = transparent
    tilemap: dict         # the out_tilemap.json payload

    @property
    def columns(self) -> int:
        return int(self.tilemap["tileset_columns"])

    def tileset_indices(self) -> np.ndarray:
        """Tileset as an index image: row-major grid, `columns` tiles wide; only the last row
        is padded (with transparent pixels)."""
        return _native.tileset_indices(_stack(self.tiles), self.columns)


def tileset_columns(n_tiles: int, requested: int) -> int:
    """Tiles per tileset row: `requested`, or ceil(sqrt(n_tiles)) when it is 0."""
    return _native.tileset_columns(int(n_tiles), int(requested))


def flip(tile: np.ndarray, flip_h: int, flip_v: int) -> np.ndarray:
    if flip_h:
        tile = tile[:, ::-1]
    if flip_v:
        tile = tile[::-1, :]
    return tile


def extract(idx: np.ndarray, palette_lab: np.ndarray, config) -> Tiles:
    """Cut idx into tiles, merging exact duplicates, near-duplicates and flipped copies.

    A map entry [tile_id, flip_h, flip_v] means: draw tileset tile `tile_id` with those flips.
    An exact match (any flip) is used when one exists, preferring the lowest tile id, then the
    flip order (0, 0), (1, 0), (0, 1), (1, 1). Otherwise a tile is a near-duplicate of a
    tileset entry (any flip) only if the mean ΔE over the tile is below
    `tile_dedupe_tolerance` AND every pixel is within TILE_DEDUPE_MAX_PIXEL_DELTA; the nearest
    qualifying entry wins (lowest mean ΔE, then lowest tile id, then flip order). Tolerance 0
    merges exact copies only, so reconstruct(extract(idx)) == idx.
    """
    stack, tilemap = _native.tiles_extract(
        np.ascontiguousarray(idx, dtype=np.int64),
        np.ascontiguousarray(palette_lab, dtype=np.float64).reshape(-1, 3), config.to_dict())
    return Tiles(tiles=stack, tilemap=tilemap)


def reconstruct(tiles: np.ndarray, tilemap: dict) -> np.ndarray:
    """Rebuild the index image described by a tileset + tilemap."""
    return _native.tiles_reconstruct(_stack(tiles), tilemap)


def gids(tilemap: dict, flips: bool = True) -> list[int]:
    """Tiled global tile ids, row-major (firstgid 1), with the flip flags when `flips`."""
    return _native.tilemap_gids(tilemap, bool(flips))


def tiled_map(tilemap: dict, n_tiles: int, image: str, image_size: tuple[int, int],
              name: str) -> dict:
    """A Tiled JSON map (.tmj): one tile layer and one embedded tileset whose image is
    `image` (a path relative to the .tmj) of `image_size` = (width, height) pixels."""
    return _native.tiled_map(tilemap, int(n_tiles), image,
                             (int(image_size[0]), int(image_size[1])), name)


def tilemap_csv(tilemap: dict) -> str:
    """One line per tile row of comma-separated GIDs (firstgid 1, no flip flags)."""
    return _native.tilemap_csv(tilemap)
