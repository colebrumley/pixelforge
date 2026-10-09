import numpy as np

from pixelforge import Config, color, postprocess

PALETTE = color.rgb8_to_lab(np.array([[20, 20, 20], [240, 240, 240], [200, 40, 40]]))


def test_orphan_removal_removes_single_stray_pixel_only():
    idx = np.zeros((12, 12), dtype=np.int64)
    idx[:, 6:] = 1
    idx[3, 2] = 2                      # the stray pixel
    expected = idx.copy()
    expected[3, 2] = 0
    out = postprocess.remove_orphans(idx, PALETTE, min_region=2)
    assert np.array_equal(out, expected)
    assert idx[3, 2] == 2              # input not modified


def test_orphans_ignore_transparency_and_keep_pairs():
    idx = np.full((8, 8), -1, dtype=np.int64)
    idx[2:6, 2:6] = 0
    idx[2, 2] = 1                      # orphan at the sprite corner → merged into 0
    idx[4, 3] = idx[4, 4] = 2          # a 2-px region survives min_region=2
    idx[0, 0] = 1                      # isolated pixel with no opaque neighbors stays
    out = postprocess.remove_orphans(idx, PALETTE, min_region=2)
    assert out[2, 2] == 0 and out[4, 3] == 2 and out[4, 4] == 2 and out[0, 0] == 1
    assert np.array_equal(out < 0, idx < 0)


def _diagonal(n=12):
    idx = np.zeros((n, n), dtype=np.int64)
    idx[np.arange(n), np.arange(n)] = 1
    return idx


def test_jaggy_fix_leaves_clean_diagonal_alone():
    line = _diagonal()
    assert np.array_equal(postprocess.fix_jaggies(line), line)
    assert np.array_equal(postprocess.fix_jaggies(line[:, ::-1]), line[:, ::-1])


def test_jaggy_fix_removes_injected_double():
    clean = _diagonal()
    for dy, dx in ((0, 1), (1, 0)):        # 2×1 and 1×2 doubles
        doubled = clean.copy()
        doubled[5 + dy, 5 + dx] = 1
        assert np.array_equal(postprocess.fix_jaggies(doubled), clean)
        assert np.array_equal(postprocess.fix_jaggies(doubled[:, ::-1]), clean[:, ::-1])


def test_jaggy_fix_leaves_shapes_alone():
    idx = np.zeros((10, 10), dtype=np.int64)
    idx[2:8, 2:8] = 1                      # filled square
    idx[2:8, 2] = idx[2, 2:8] = 2          # with an L-shaped 1-px border
    assert np.array_equal(postprocess.fix_jaggies(idx), idx)


def test_outline_adds_exactly_the_4_neighbor_ring():
    sprite = np.zeros((5, 5), dtype=np.int64)
    out, palette_lab = postprocess.add_outline(sprite, PALETTE[:2], "auto", 0.55)
    assert out.shape == (7, 7) and len(palette_lab) == 3
    expected = np.full((7, 7), 2, dtype=np.int64)
    expected[1:6, 1:6] = 0
    for corner in ((0, 0), (0, 6), (6, 0), (6, 6)):
        expected[corner] = -1              # diagonal neighbors are not outlined
    assert np.array_equal(out, expected)
    assert (out == 2).sum() == 20
    # auto color = darkest palette color blended toward black
    assert np.allclose(palette_lab[2], PALETTE[0] * 0.45)


def test_outline_explicit_color_and_no_padding_needed():
    sprite = np.full((6, 6), -1, dtype=np.int64)
    sprite[2:4, 2:4] = 1
    out, palette_lab = postprocess.add_outline(sprite, PALETTE, "#c82828", 0.55)
    assert out.shape == (6, 6)
    assert len(palette_lab) == 3 and (out == 2).sum() == 8   # reuses the existing red


def test_run_applies_passes_in_order():
    idx = np.full((8, 8), -1, dtype=np.int64)
    idx[2:6, 2:6] = 1
    idx[3, 3] = 0                          # orphan inside the sprite
    out, palette_lab = postprocess.run(idx, PALETTE[:2], Config(palette_ramps=False))
    assert out[3, 3] == 1 and len(palette_lab) == 3
    assert (out == 2).sum() == 16 and out.shape == (8, 8)
    assert np.allclose(palette_lab[1, 1:], PALETTE[1, 1:] * 1.1)
