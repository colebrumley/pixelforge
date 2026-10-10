"""Tileset / tilemap extraction (Section 9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color

TILESET_COLUMNS = 16
# ΔE charged when a transparent pixel is compared with an opaque one.
TRANSPARENT_DISTANCE = 100.0
# DEVIATION: Section 9 — near-duplicate tiles must also agree pixel by pixel: every pixel's
# ΔE (LAB units) must be below this cap, whatever `tile_dedupe_tolerance` is. A mean-ΔE test
# alone lets a few very different pixels (a star, a highlight) merge into a plain tile and
# vanish. Transparent-vs-opaque pixels cost TRANSPARENT_DISTANCE, so they never merge.
TILE_DEDUPE_MAX_PIXEL_DELTA = 10.0

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


def _digest(tile: np.ndarray) -> bytes:
    """Exact-match key: the tile's raw codes."""
    return np.ascontiguousarray(tile, dtype=np.int32).tobytes()


def extract(idx: np.ndarray, palette_lab: np.ndarray, config) -> Tiles:
    """Cut idx into tiles, merging exact duplicates, near-duplicates and flipped copies.

    A map entry [tile_id, flip_h, flip_v] means: draw tileset tile `tile_id` with those flips.
    An exact match (any flip) is used when one exists, preferring the lowest tile id, then the
    flip order in _FLIPS. Otherwise a tile is a near-duplicate of a tileset entry (any flip)
    only if the mean ΔE over the tile is below `tile_dedupe_tolerance` AND every pixel is
    within TILE_DEDUPE_MAX_PIXEL_DELTA; the nearest qualifying entry wins (lowest mean ΔE,
    then lowest tile id, then flip order). Tolerance 0 merges exact copies only, so
    reconstruct(extract(idx)) == idx.
    """
    t = config.tile_size
    h, w = idx.shape
    if h % t or w % t:
        raise ValueError(f"tile_size ({t}) must divide the output size ({w}x{h})")
    tolerance = float(config.tile_dedupe_tolerance)
    n_colors = len(palette_lab)
    clear = n_colors                                   # code for a transparent pixel
    distances = np.full((n_colors + 1, n_colors + 1), TRANSPARENT_DISTANCE)
    distances[:n_colors, :n_colors] = color.palette_distance_matrix(palette_lab)
    distances[clear, clear] = 0.0
    # LAB per code, transparent = 0 so that sums over a tile only count opaque pixels.
    lab = np.zeros((n_colors + 1, 3))
    lab[:n_colors] = palette_lab

    ht, wt = h // t, w // t
    codes = np.where(idx < 0, clear, idx).astype(np.int64)
    cells = np.ascontiguousarray(codes.reshape(ht, t, wt, t).transpose(0, 2, 1, 3))
    cells = cells.reshape(ht * wt, t, t)
    # Pruning keys. Flips preserve both. Equal opaque counts are necessary because any
    # transparent/opaque mismatch costs TRANSPARENT_DISTANCE > the per-pixel cap. By convexity
    # of CIE76, ΔE(mean colors) <= mean over opaque pixels of ΔE = mean ΔE · t² / n_opaque,
    # and <= max per-pixel ΔE, so the bound below never drops a qualifying candidate.
    n_opaque = np.count_nonzero(cells != clear, axis=(1, 2))
    means = lab[cells].sum(axis=(1, 2)) / np.maximum(n_opaque, 1)[:, None]

    # Preallocated tileset (at most one entry per cell); rows [:n_unique] are in use.
    unique = np.empty_like(cells)
    unique_means = np.empty((len(cells), 3))
    unique_opaque = np.empty(len(cells), dtype=np.int64)
    n_unique = 0
    by_digest: dict[bytes, int] = {}
    entries = []
    for i, tile in enumerate(cells):
        # Flips are involutions: flip(unique) == tile  ⇔  unique == flip(tile).
        variants = np.stack([flip(tile, fh, fv) for fh, fv in _FLIPS])
        match = None
        exact = [(by_digest[d], order) for order, d in enumerate(map(_digest, variants))
                 if d in by_digest]
        if exact:
            match = min(exact)
        elif n_unique and tolerance > 0 and n_opaque[i]:
            bound = min(tolerance * t * t / n_opaque[i], TILE_DEDUPE_MAX_PIXEL_DELTA) + 1e-9
            candidates = np.nonzero(
                (unique_opaque[:n_unique] == n_opaque[i])
                & (color.delta_e(unique_means[:n_unique], means[i]) <= bound))[0]
            if len(candidates):
                # Cheap screen on the first row: one pixel at or over the cap, or a partial
                # ΔE sum that already reaches tolerance · t², rules a (flip, candidate) out.
                row = distances[variants[:, None, 0], unique[candidates, 0][None]]  # (4, k, t)
                alive = ((row.max(axis=2) < TILE_DEDUPE_MAX_PIXEL_DELTA)
                         & (row.sum(axis=2) < tolerance * t * t)).any(axis=0)
                candidates = candidates[alive]
            if len(candidates):
                d = distances[variants[:, None], unique[candidates][None]]   # (4, k, t, t)
                mean_d = d.mean(axis=(2, 3))
                ok = (mean_d < tolerance) & (d.max(axis=(2, 3)) < TILE_DEDUPE_MAX_PIXEL_DELTA)
                if ok.any():
                    score = np.where(ok, mean_d, np.inf)
                    orders, ks = np.nonzero(score == score.min())
                    j = np.lexsort((orders, ks))[0]      # lowest tile id, then flip order
                    match = (int(candidates[ks[j]]), int(orders[j]))
        if match is None:
            by_digest[_digest(tile)] = n_unique
            unique[n_unique] = tile
            unique_means[n_unique] = means[i]
            unique_opaque[n_unique] = n_opaque[i]
            n_unique += 1
            match = (n_unique - 1, 0)
        tile_id, order = match
        entries.append([int(tile_id), _FLIPS[order][0], _FLIPS[order][1]])

    tilemap = {"tile_size": t, "width_tiles": wt, "height_tiles": ht,
               "tileset_columns": TILESET_COLUMNS, "tiles": entries}
    out = unique[:n_unique].copy()
    out[out == clear] = -1
    return Tiles(tiles=out, tilemap=tilemap)


def reconstruct(tiles: np.ndarray, tilemap: dict) -> np.ndarray:
    """Rebuild the index image described by a tileset + tilemap."""
    t = tilemap["tile_size"]
    wt, ht = tilemap["width_tiles"], tilemap["height_tiles"]
    out = np.full((ht * t, wt * t), -1, dtype=np.int64)
    for i, (tile_id, flip_h, flip_v) in enumerate(tilemap["tiles"]):
        y, x = (i // wt) * t, (i % wt) * t
        out[y:y + t, x:x + t] = flip(tiles[tile_id], flip_h, flip_v)
    return out
