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
