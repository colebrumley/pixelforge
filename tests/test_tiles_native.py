"""The native tiles module (crates/pixelforge/src/tiles.rs) reproduces the numpy
implementation it replaced, bit for bit. The numpy code is kept below as the reference."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pixelforge import Config, color, io, pipeline, tiles
from pixelforge.errors import PixelforgeError

# ------------------------------------------------------------------ reference (numpy)

GID_FLIP_H = 0x80000000
GID_FLIP_V = 0x40000000
TRANSPARENT_DISTANCE = 100.0
TILE_DEDUPE_MAX_PIXEL_DELTA = 10.0
_FLIPS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def ref_tileset_columns(n_tiles, requested):
    if requested:
        return int(requested)
    return max(1, math.isqrt(max(n_tiles, 1) - 1) + 1)


def ref_tileset_indices(stack, columns):
    n, t = len(stack), stack.shape[1]
    rows = -(-n // columns)
    sheet = np.full((rows * t, columns * t), -1, dtype=np.int64)
    for i, tile in enumerate(stack):
        y, x = (i // columns) * t, (i % columns) * t
        sheet[y:y + t, x:x + t] = tile
    return sheet


def ref_flip(tile, flip_h, flip_v):
    if flip_h:
        tile = tile[:, ::-1]
    if flip_v:
        tile = tile[::-1, :]
    return tile


def _digest(tile):
    return np.ascontiguousarray(tile, dtype=np.int32).tobytes()


def ref_extract(idx, palette_lab, config):
    t = config.tile_size
    h, w = idx.shape
    if h % t or w % t:
        raise ValueError(f"tile_size ({t}) must divide the output size ({w}x{h})")
    tolerance = float(config.tile_dedupe_tolerance)
    n_colors = len(palette_lab)
    clear = n_colors
    distances = np.full((n_colors + 1, n_colors + 1), TRANSPARENT_DISTANCE)
    distances[:n_colors, :n_colors] = color.palette_distance_matrix(palette_lab)
    distances[clear, clear] = 0.0
    lab = np.zeros((n_colors + 1, 3))
    lab[:n_colors] = palette_lab
    ht, wt = h // t, w // t
    codes = np.where(idx < 0, clear, idx).astype(np.int64)
    cells = np.ascontiguousarray(codes.reshape(ht, t, wt, t).transpose(0, 2, 1, 3))
    cells = cells.reshape(ht * wt, t, t)
    n_opaque = np.count_nonzero(cells != clear, axis=(1, 2))
    means = lab[cells].sum(axis=(1, 2)) / np.maximum(n_opaque, 1)[:, None]
    unique = np.empty_like(cells)
    unique_means = np.empty((len(cells), 3))
    unique_opaque = np.empty(len(cells), dtype=np.int64)
    n_unique = 0
    by_digest = {}
    entries = []
    for i, tile in enumerate(cells):
        variants = np.stack([ref_flip(tile, fh, fv) for fh, fv in _FLIPS])
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
                row = distances[variants[:, None, 0], unique[candidates, 0][None]]
                alive = ((row.max(axis=2) < TILE_DEDUPE_MAX_PIXEL_DELTA)
                         & (row.sum(axis=2) < tolerance * t * t)).any(axis=0)
                candidates = candidates[alive]
            if len(candidates):
                d = distances[variants[:, None], unique[candidates][None]]
                mean_d = d.mean(axis=(2, 3))
                ok = (mean_d < tolerance) & (d.max(axis=(2, 3)) < TILE_DEDUPE_MAX_PIXEL_DELTA)
                if ok.any():
                    score = np.where(ok, mean_d, np.inf)
                    orders, ks = np.nonzero(score == score.min())
                    j = np.lexsort((orders, ks))[0]
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
               "tileset_columns": ref_tileset_columns(n_unique, config.tileset_columns),
               "tiles": entries}
    out = unique[:n_unique].copy()
    out[out == clear] = -1
    return out, tilemap


def ref_reconstruct(stack, tilemap):
    t = tilemap["tile_size"]
    wt, ht = tilemap["width_tiles"], tilemap["height_tiles"]
    out = np.full((ht * t, wt * t), -1, dtype=np.int64)
    for i, (tile_id, flip_h, flip_v) in enumerate(tilemap["tiles"]):
        y, x = (i // wt) * t, (i % wt) * t
        out[y:y + t, x:x + t] = ref_flip(stack[tile_id], flip_h, flip_v)
    return out


def ref_gids(tilemap, flips=True):
    out = []
    for tile_id, flip_h, flip_v in tilemap["tiles"]:
        gid = tile_id + 1
        if flips:
            gid |= (GID_FLIP_H if flip_h else 0) | (GID_FLIP_V if flip_v else 0)
        out.append(gid)
    return out


def ref_tiled_map(tilemap, n_tiles, image, image_size, name):
    t = tilemap["tile_size"]
    wt, ht = tilemap["width_tiles"], tilemap["height_tiles"]
    tileset = {"firstgid": 1, "name": name, "image": image,
               "imagewidth": int(image_size[0]), "imageheight": int(image_size[1]),
               "columns": tilemap["tileset_columns"], "tilecount": int(n_tiles),
               "tilewidth": t, "tileheight": t, "margin": 0, "spacing": 0}
    layer = {"id": 1, "name": "background", "type": "tilelayer", "x": 0, "y": 0,
             "width": wt, "height": ht, "opacity": 1, "visible": True,
             "data": ref_gids(tilemap)}
    return {"type": "map", "version": "1.10", "orientation": "orthogonal",
            "renderorder": "right-down", "infinite": False, "width": wt, "height": ht,
            "tilewidth": t, "tileheight": t, "nextlayerid": 2, "nextobjectid": 1,
            "layers": [layer], "tilesets": [tileset]}


def ref_tilemap_csv(tilemap):
    wt = tilemap["width_tiles"]
    values = ref_gids(tilemap, flips=False)
    rows = [",".join(map(str, values[i:i + wt])) for i in range(0, len(values), wt)]
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------------- helpers

def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check(idx, palette_lab, config):
    want_stack, want_map = ref_extract(idx, palette_lab, config)
    got = tiles.extract(idx, palette_lab, config)
    assert same(got.tiles, want_stack)
    assert json.dumps(got.tilemap) == json.dumps(want_map)
    assert same(tiles.reconstruct(got.tiles, got.tilemap),
                ref_reconstruct(want_stack, want_map))
    assert same(got.tileset_indices(), ref_tileset_indices(want_stack,
                                                           want_map["tileset_columns"]))
    assert tiles.gids(got.tilemap) == ref_gids(want_map)
    assert tiles.gids(got.tilemap, flips=False) == ref_gids(want_map, flips=False)
    sheet = ref_tileset_indices(want_stack, want_map["tileset_columns"])
    args = (len(want_stack), "x_tileset.png", (sheet.shape[1], sheet.shape[0]), "x")
    assert (json.dumps(tiles.tiled_map(got.tilemap, *args), indent=1)
            == json.dumps(ref_tiled_map(want_map, *args), indent=1))
    assert tiles.tilemap_csv(got.tilemap) == ref_tilemap_csv(want_map)
    return want_map


def tiled_image(rng, ht, wt, t, k, n_base=4, noise=0.3, transparent=0.1):
    """Tiles drawn from a few bases, flipped, with near-duplicate pixel changes."""
    bases = rng.integers(0, k, (n_base, t, t))
    bases[rng.random(bases.shape) < transparent] = -1
    out = np.empty((ht * t, wt * t), dtype=np.int64)
    for y in range(ht):
        for x in range(wt):
            tile = ref_flip(bases[rng.integers(n_base)], rng.integers(2), rng.integers(2)).copy()
            if rng.random() < noise:
                yy, xx = rng.integers(t, size=2)
                if tile[yy, xx] >= 0:
                    tile[yy, xx] = rng.integers(k)
            out[y * t:(y + 1) * t, x * t:(x + 1) * t] = tile
    return out


# --------------------------------------------------------------------------- tests

@pytest.mark.parametrize("tol", [0.0, 2.0, 50.0])
@pytest.mark.parametrize("columns", [0, 16])
def test_checker_tiles_fixture(fixture_path, tol, columns):
    loaded = io.load(fixture_path("checker_tiles"))
    lab = color.rgb8_to_lab(loaded.rgb)
    palette = np.unique(lab.reshape(-1, 3), axis=0)
    idx = color.nearest_index(lab.reshape(-1, 3), palette).reshape(lab.shape[:2])
    config = Config(tile_size=16, tile_dedupe_tolerance=tol, tileset_columns=columns)
    check(idx, palette, config)


@pytest.mark.parametrize("tol", [0.0, 2.0, 50.0])
@pytest.mark.parametrize("columns", [0, 16])
@pytest.mark.parametrize("k", [2, 3, 6])
def test_random_tiles(tol, columns, k):
    rng = np.random.default_rng(int(tol * 10) + columns + 100 * k)
    # Close colours so that near-duplicates fall inside the per-pixel cap.
    palette = np.column_stack([50 + rng.uniform(-6, 6, k), rng.uniform(-4, 4, k),
                               rng.uniform(-4, 4, k)])
    far = np.column_stack([rng.uniform(0, 100, k), rng.uniform(-60, 60, k),
                           rng.uniform(-60, 60, k)])
    for t, ht, wt in ((1, 3, 5), (2, 4, 4), (4, 5, 3), (8, 4, 4), (16, 2, 3)):
        for pal in (palette, far):
            idx = tiled_image(rng, ht, wt, t, k)
            config = Config(tile_size=t, tile_dedupe_tolerance=tol, tileset_columns=columns)
            check(idx, pal, config)
            check(np.full_like(idx, -1), pal, config)


def test_merges_happen():
    # The random cases above include near-duplicate merges and flipped matches.
    rng = np.random.default_rng(1)
    palette = np.array([[50.0, 0, 0], [52.0, 0, 0], [49.0, 1.0, 0]])
    idx = tiled_image(rng, 6, 6, 4, 3, noise=0.8, transparent=0.0)
    strict = check(idx, palette, Config(tile_size=4, tile_dedupe_tolerance=0.0))
    loose = check(idx, palette, Config(tile_size=4, tile_dedupe_tolerance=2.0))
    assert max(e[0] for e in loose["tiles"]) < max(e[0] for e in strict["tiles"])
    assert any(fh or fv for _, fh, fv in strict["tiles"])


def test_bad_tile_size():
    idx = np.zeros((10, 12), dtype=np.int64)
    with pytest.raises(ValueError, match=r"tile_size \(8\) must divide the output size \(12x10\)"):
        tiles.extract(idx, np.zeros((1, 3)), Config(tile_size=8))


@pytest.mark.parametrize("n", [0, 1, 2, 4, 5, 15, 16, 17, 100])
@pytest.mark.parametrize("requested", [0, 1, 16])
def test_tileset_columns(n, requested):
    assert tiles.tileset_columns(n, requested) == ref_tileset_columns(n, requested)


@pytest.mark.parametrize("preset", ["sprite", "background"])
@pytest.mark.parametrize("method", ["box", "kopf"])
@pytest.mark.parametrize("name", ["checker_tiles", "circle_alpha", "gradient6", "line_diag",
                                  "noisy_gradient", "seamless_src", "two_color"])
def test_pipeline_inputs(fixture_path, monkeypatch, name, preset, method):
    captured = []
    real = tiles.extract

    def spy(indices, palette_lab, config):
        captured.append((np.array(indices), np.array(palette_lab)))
        return real(indices, palette_lab, config)

    monkeypatch.setattr(pipeline.tiles, "extract", spy)
    config = Config(preset=preset, method=method, out_width=32, out_height=32, tileset=True,
                    tile_size=8)
    try:
        pipeline.run(fixture_path(name), config)
    except PixelforgeError as exc:      # e.g. a 1-px line keyed out entirely at 32 px
        pytest.skip(str(exc))
    assert len(captured) == 1
    indices, palette_lab = captured[0]
    for tol in (0.0, 2.0, 50.0):
        for columns in (0, 16):
            check(indices, palette_lab, config.replace(tile_dedupe_tolerance=tol,
                                                       tileset_columns=columns))
