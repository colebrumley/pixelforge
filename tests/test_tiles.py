import numpy as np

from pixelforge import Config, color, run, tiles


def test_checker_tiles_dedupe_with_flips(fixture_path):
    config = Config(preset="background", out_width=256, out_height=256)
    assert config.tileset and config.tile_size == 16
    result = run(fixture_path("checker_tiles"), config)
    n_tiles = result.stats["tiles"]
    entries = result.tilemap["tiles"]
    flips_used = any(fh or fv for _, fh, fv in entries)
    assert n_tiles == 2 or (n_tiles == 1 and flips_used)
    assert isinstance(result.stats["tiles_rerender_px_changed"], int)
    assert 0 <= result.stats["tiles_rerender_px_changed"] <= 256 * 256
    assert result.tilemap["width_tiles"] == 16 and result.tilemap["height_tiles"] == 16
    assert len(entries) == 256 and result.tilemap["tileset_columns"] == 16

    # The tilemap reconstructs the index image exactly.
    t = result.tilemap["tile_size"]
    sheet = result.tileset_indices
    tile_stack = np.stack([sheet[(i // 16) * t:(i // 16 + 1) * t, (i % 16) * t:(i % 16 + 1) * t]
                           for i in range(n_tiles)])
    assert np.array_equal(tiles.reconstruct(tile_stack, result.tilemap), result.indices)
    assert result.tileset.shape == (16, 256, 4)
    # Every output color is one of the two checker colors (the denoiser may shift them by a
    # level or two, which median cut keeps as separate palette entries).
    source = color.rgb8_to_lab(np.array([[60, 90, 160], [230, 200, 90]]))
    used = result.palette_lab[np.unique(result.indices)]
    assert color.delta_e(used[:, None, :], source[None, :, :]).min(axis=1).max() < 12


def test_extract_exact_near_and_flipped():
    palette_lab = color.rgb8_to_lab(np.array([[0, 0, 0], [255, 255, 255], [250, 250, 250]]))
    a = np.zeros((4, 4), dtype=np.int64)
    a[:, 0] = 1                                   # asymmetric tile
    near = a.copy()
    near[3, 3] = 0
    near[:, 0] = 2                                # almost the same white
    other = np.ones((4, 4), dtype=np.int64)
    idx = np.block([[a, a[:, ::-1]], [near, other]])
    out = tiles.extract(idx, palette_lab, Config(tile_size=4))
    assert len(out.tiles) == 2
    assert out.tilemap["tiles"] == [[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]]
    strict = tiles.extract(idx, palette_lab, Config(tile_size=4, tile_dedupe_tolerance=0.0))
    assert len(strict.tiles) == 3
    assert np.array_equal(tiles.reconstruct(strict.tiles, strict.tilemap), idx)


def test_seamless_edges_match(fixture_path):
    config = Config(preset="background", method="kopf", out_width=32, out_height=32,
                    tileset=False, seamless=True)
    result = run(fixture_path("seamless_src"), config)
    assert result.indices.shape == (32, 32)
    lab = result.palette_lab[result.indices]
    left_right = color.delta_e(lab[:, 0], lab[:, -1]).mean()
    top_bottom = color.delta_e(lab[0, :], lab[-1, :]).mean()
    interior = color.delta_e(lab[:, 1:-2], lab[:, 2:-1]).mean()
    assert interior > 0
    assert left_right < 3 * interior
    assert top_bottom < 3 * interior


def _lab_gray(*lightness):
    return np.array([[v, 0.0, 0.0] for v in lightness])


def test_sparse_stars_stay_distinct():
    # The review's "6 stars": a near-black sky with 6 single white pixels, each in its own
    # tile at a different spot. Mean ΔE per tile is ~0.4, far under the default tolerance,
    # but the star pixel's ΔE (~95) exceeds the per-pixel cap, so nothing is lost.
    palette_lab = color.rgb8_to_lab(np.array([[8, 8, 12], [255, 255, 255]]))
    idx = np.zeros((64, 64), dtype=np.int64)
    stars = [(1, 2), (5, 22), (20, 7), (27, 41), (40, 50), (54, 13)]
    for y, x in stars:
        idx[y, x] = 1
    config = Config(tile_size=16)
    assert config.tile_dedupe_tolerance > 0
    out = tiles.extract(idx, palette_lab, config)
    assert len(out.tiles) == 7                    # 6 star tiles + 1 plain sky tile
    assert np.array_equal(tiles.reconstruct(out.tiles, out.tilemap), idx)


def test_single_pixel_small_delta_still_merges():
    palette_lab = _lab_gray(20.0, 24.0)           # ΔE 4, under the per-pixel cap
    a = np.zeros((16, 16), dtype=np.int64)
    b = a.copy()
    b[7, 9] = 1
    out = tiles.extract(np.hstack([a, b]), palette_lab, Config(tile_size=16))
    assert len(out.tiles) == 1
    assert out.tilemap["tiles"] == [[0, 0, 0], [0, 0, 0]]


def test_near_duplicate_picks_nearest_not_earliest():
    palette_lab = _lab_gray(50.0, 51.5, 52.5)
    a = np.zeros((4, 4), dtype=np.int64)          # L 50
    b = np.full((4, 4), 2)                        # L 52.5: ΔE 2.5 from A, stays distinct
    c = np.ones((4, 4), dtype=np.int64)           # L 51.5: ΔE 1.5 from A, 1.0 from B
    out = tiles.extract(np.hstack([a, b, c]), palette_lab, Config(tile_size=4))
    assert len(out.tiles) == 2
    assert out.tilemap["tiles"] == [[0, 0, 0], [1, 0, 0], [1, 0, 0]]


def test_near_duplicate_tie_prefers_lowest_id():
    palette_lab = _lab_gray(50.0, 51.0, 52.5)
    a = np.zeros((4, 4), dtype=np.int64)
    b = np.full((4, 4), 2)
    c = np.ones((4, 4), dtype=np.int64)           # ΔE 1.0 from A, 1.5 from B
    out = tiles.extract(np.hstack([a, b, c]), palette_lab, Config(tile_size=4))
    assert out.tilemap["tiles"][2] == [0, 0, 0]
    tie = _lab_gray(50.0, 51.25, 52.5)            # equidistant from A and B
    out = tiles.extract(np.hstack([a, b, c]), tie, Config(tile_size=4))
    assert out.tilemap["tiles"][2] == [0, 0, 0]


def test_flipped_copies_map_to_one_tile():
    palette_lab = color.rgb8_to_lab(np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]]))
    a = np.zeros((4, 4), dtype=np.int64)
    a[0, :2] = 1
    a[1, 0] = 2
    a[3, 3] = -1                                  # transparency survives flips too
    idx = np.hstack([a, a[:, ::-1], a[::-1, :], a[::-1, ::-1], a])
    for tol in (0.0, 2.0):
        out = tiles.extract(idx, palette_lab, Config(tile_size=4, tile_dedupe_tolerance=tol))
        assert len(out.tiles) == 1
        assert out.tilemap["tiles"] == [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0]]
        assert np.array_equal(tiles.reconstruct(out.tiles, out.tilemap), idx)


def test_transparency_mismatch_never_merges():
    palette_lab = color.rgb8_to_lab(np.array([[0, 0, 0]]))
    a = np.zeros((16, 16), dtype=np.int64)
    b = a.copy()
    b[0, 0] = -1                                  # mean ΔE 0.39, but one clear/opaque pixel
    out = tiles.extract(np.hstack([a, b]), palette_lab, Config(tile_size=16))
    assert len(out.tiles) == 2


def test_zero_tolerance_round_trips():
    rng = np.random.default_rng(7)
    palette_lab = color.rgb8_to_lab(rng.integers(0, 256, (6, 3)))
    for _ in range(5):
        base = rng.integers(-1, 6, (8, 8))
        # A few repeated / flipped / one-pixel-off tiles so that exact merges do happen.
        near = base.copy()
        near[2, 5] = (near[2, 5] + 1) % 6
        idx = np.block([[base, base[:, ::-1], near],
                        [rng.integers(-1, 6, (8, 8)), base[::-1, :], base]])
        out = tiles.extract(idx, palette_lab, Config(tile_size=8, tile_dedupe_tolerance=0.0))
        assert len(out.tiles) == 3
        assert np.array_equal(tiles.reconstruct(out.tiles, out.tilemap), idx)
