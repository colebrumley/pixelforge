"""Tileset / tilemap extraction (Section 9)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import color

TILESET_COLUMNS = 16
# ΔE charged when a transparent pixel is compared with an opaque one.
TRANSPARENT_DISTANCE = 100.0

# (flip_h, flip_v) in the order they are tried.
_FLIPS = [(0, 0), (1, 0), (0, 1), (1, 1)]


@dataclass
class Tiles:
    tiles: np.ndarray     # (n, t, t) index arrays, -1 = transparent
    tilemap: dict         # the out_tilemap.json payload

    def tileset_indices(self) -> np.ndarray:
        """Tileset as an index image: row-major grid, TILESET_COLUMNS tiles wide."""
        n, t = len(self.tiles), self.tiles.shape[1]
        rows = -(-n // TILESET_COLUMNS)
        sheet = np.full((rows * t, TILESET_COLUMNS * t), -1, dtype=np.int64)
        for i, tile in enumerate(self.tiles):
            y, x = (i // TILESET_COLUMNS) * t, (i % TILESET_COLUMNS) * t
            sheet[y:y + t, x:x + t] = tile
        return sheet


def flip(tile: np.ndarray, flip_h: int, flip_v: int) -> np.ndarray:
    if flip_h:
        tile = tile[:, ::-1]
    if flip_v:
        tile = tile[::-1, :]
    return tile


def _digest(tile: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tile, dtype=np.int16).tobytes()).hexdigest()


def extract(idx: np.ndarray, palette_lab: np.ndarray, config) -> Tiles:
    """Cut idx into tiles, merging exact duplicates, near-duplicates and flipped copies.

    A map entry [tile_id, flip_h, flip_v] means: draw tileset tile `tile_id` with those flips.
    Each tile merges into the earliest matching tileset entry.
    """
    t = config.tile_size
    h, w = idx.shape
    if h % t or w % t:
        raise ValueError(f"tile_size ({t}) must divide the output size ({w}x{h})")
    n_colors = len(palette_lab)
    distances = np.full((n_colors + 1, n_colors + 1), TRANSPARENT_DISTANCE)
    distances[:n_colors, :n_colors] = color.palette_distance_matrix(palette_lab)
    distances[n_colors, n_colors] = 0.0

    unique: list[np.ndarray] = []
    by_digest: dict[str, int] = {}
    entries = []
    for ty in range(h // t):
        for tx in range(w // t):
            tile = idx[ty * t:(ty + 1) * t, tx * t:(tx + 1) * t]
            variants = [flip(tile, fh, fv) for fh, fv in _FLIPS]
            # Flips are involutions: flip(unique) == tile  ⇔  unique == flip(tile).
            match = None
            exact = [(by_digest[d], order) for order, d in enumerate(map(_digest, variants))
                     if d in by_digest]
            if exact:
                match = min(exact)
            elif unique and config.tile_dedupe_tolerance > 0:
                stack = np.where(np.stack(unique) < 0, n_colors, np.stack(unique))
                close = np.stack([
                    distances[stack, np.where(v < 0, n_colors, v)[None]].mean(axis=(1, 2))
                    < config.tile_dedupe_tolerance for v in variants])      # (4, n_unique)
                candidates = np.nonzero(close.any(axis=0))[0]
                if len(candidates):
                    tile_id = int(candidates[0])
                    match = (tile_id, int(np.argmax(close[:, tile_id])))
            if match is None:
                by_digest[_digest(tile)] = len(unique)
                unique.append(np.array(tile, dtype=np.int64))
                match = (len(unique) - 1, 0)
            tile_id, order = match
            entries.append([int(tile_id), _FLIPS[order][0], _FLIPS[order][1]])

    tilemap = {"tile_size": t, "width_tiles": w // t, "height_tiles": h // t,
               "tileset_columns": TILESET_COLUMNS, "tiles": entries}
    return Tiles(tiles=np.stack(unique), tilemap=tilemap)


def reconstruct(tiles: np.ndarray, tilemap: dict) -> np.ndarray:
    """Rebuild the index image described by a tileset + tilemap."""
    t = tilemap["tile_size"]
    wt, ht = tilemap["width_tiles"], tilemap["height_tiles"]
    out = np.full((ht * t, wt * t), -1, dtype=np.int64)
    for i, (tile_id, flip_h, flip_v) in enumerate(tilemap["tiles"]):
        y, x = (i // wt) * t, (i % wt) * t
        out[y:y + t, x:x + t] = flip(tiles[tile_id], flip_h, flip_v)
    return out
