import numpy as np
import pytest

from pixelforge import Config, color, postprocess

PALETTE = color.rgb8_to_lab(np.array([[20, 20, 20], [240, 240, 240], [200, 40, 40]]))


# dark, light, red, near-dark (ΔE 4.8 from dark), near-light (ΔE 3.9 from light)
NEAR = color.rgb8_to_lab(np.array([[20, 20, 20], [240, 240, 240], [200, 40, 40],
                                   [32, 26, 26], [228, 232, 236]]))


def test_orphan_removal_removes_single_stray_pixel_only():
    idx = np.zeros((12, 12), dtype=np.int64)
    idx[:, 6:] = 1
    idx[3, 2] = 3                      # the stray pixel, within ΔE 5 of its surroundings
    expected = idx.copy()
    expected[3, 2] = 0
    out = postprocess.remove_orphans(idx, NEAR, min_region=2)
    assert np.array_equal(out, expected)
    assert idx[3, 2] == 3              # input not modified


def test_orphans_ignore_transparency_and_keep_pairs():
    idx = np.full((8, 8), -1, dtype=np.int64)
    idx[2:6, 2:6] = 0
    idx[2, 2] = 3                      # low-contrast orphan at the sprite corner → merged into 0
    idx[4, 3] = idx[4, 4] = 2          # a 2-px region survives min_region=2
    idx[0, 0] = 1                      # isolated pixel with no opaque neighbors stays
    out = postprocess.remove_orphans(idx, NEAR, min_region=2)
    assert out[2, 2] == 0 and out[4, 3] == 2 and out[4, 4] == 2 and out[0, 0] == 1
    assert np.array_equal(out < 0, idx < 0)


def test_orphans_keep_high_contrast_eyes_with_defaults():
    palette = color.rgb8_to_lab(np.array([[230, 200, 170], [40, 30, 30]]))   # skin, eye
    idx = np.full((12, 12), -1, dtype=np.int64)
    idx[1:11, 1:11] = 0
    idx[4, 4] = idx[4, 7] = 1          # two 1-px eyes, ΔE ≈ 72 from the face
    cfg = Config()
    out = postprocess.remove_orphans(idx, palette, cfg.orphan_min_region,
                                     max_delta=cfg.orphan_max_delta)
    assert np.array_equal(out, idx)
    run_out, _ = postprocess.run(idx, palette, Config(palette_ramps=False, outline="none"))
    assert run_out[4, 4] == 1 and run_out[4, 7] == 1
    # Without the exemption they would be erased.
    merged = postprocess.remove_orphans(idx, palette, 2, max_delta=np.inf)
    assert (merged == 1).sum() == 0


def test_orphan_sparkle_does_not_depend_on_pass_parity():
    for pair in ((1, 4), (1, 2)):      # ΔE 3.9 (would swap every pass) and ΔE 89
        idx = np.full((6, 6), -1, dtype=np.int64)
        idx[2, 2], idx[2, 3] = pair
        results = [postprocess.remove_orphans(idx, NEAR, 2, max_passes=n) for n in range(1, 6)]
        for out in results:
            assert np.array_equal(out, results[0])
        assert np.array_equal(results[0], idx)


def test_orphan_gem_of_distant_colors_survives():
    palette = color.rgb8_to_lab(np.array([[128, 128, 128], [255, 220, 0], [0, 160, 60],
                                          [40, 60, 220], [230, 40, 140]]))
    for background in (-1, 0):         # on transparency and inside a grey field
        idx = np.full((8, 8), background, dtype=np.int64)
        idx[3:5, 3:5] = [[1, 2], [3, 4]]
        assert np.array_equal(postprocess.remove_orphans(idx, palette, 2), idx)


def test_orphan_merges_into_cluster_once_neighbor_merges():
    palette = color.rgb8_to_lab(np.array([[20, 20, 20], [32, 26, 26], [40, 30, 30]]))
    idx = np.zeros((7, 7), dtype=np.int64)
    idx[1, 3] = idx[2, 2] = idx[2, 4] = -1
    idx[3, 2:5] = idx[4, 2:5] = -1
    idx[2, 3] = 1                      # touches the field: merged in pass 1
    idx[3, 3] = 2                      # touches only the other single: merged in pass 2
    out = postprocess.remove_orphans(idx, palette, 2)
    assert out[2, 3] == 0 and out[3, 3] == 0
    one = postprocess.remove_orphans(idx, palette, 2, max_passes=1)
    assert one[2, 3] == 0 and one[3, 3] == 2


def test_orphan_max_delta_validation():
    assert Config(orphan_max_delta=0).orphan_max_delta == 0.0
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="orphan_max_delta"):
            Config(orphan_max_delta=bad)


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
    idx[3, 3] = 2                          # low-contrast orphan inside the sprite
    palette = NEAR[[0, 1, 4]]
    out, palette_lab = postprocess.run(idx, palette, Config(palette_ramps=False))
    assert out[3, 3] == 1 and len(palette_lab) == 4
    assert (out == 3).sum() == 16 and out.shape == (8, 8)
    assert np.allclose(palette_lab[1, 1:], PALETTE[1, 1:] * 1.1)


def test_outline_fixed_palette_snaps_to_nearest_entry():
    sprite = np.full((6, 6), -1, dtype=np.int64)
    sprite[2:4, 2:4] = 1
    stats = {}
    out, palette_lab = postprocess.add_outline(sprite, PALETTE, "auto", 0.55,
                                               fixed_palette=True, stats=stats)
    assert np.array_equal(palette_lab, PALETTE)                    # nothing appended
    assert stats["outline_index"] == 0 and (out == 0).sum() == 8   # darkest entry
    # An explicit color snaps too: dark orange → the red entry.
    out, palette_lab = postprocess.add_outline(sprite, PALETTE, "#b04010", 0.55,
                                               fixed_palette=True)
    assert np.array_equal(palette_lab, PALETTE) and (out == 2).sum() == 8


def test_outline_full_palette_overwritten_only_when_not_fixed():
    full = color.rgb8_to_lab(np.array([[i, i, i] for i in range(256)], dtype=np.uint8))
    sprite = np.full((6, 6), -1, dtype=np.int64)
    sprite[2:4, 2:4] = 200
    out, palette_lab = postprocess.add_outline(sprite, full, "#c82828", 0.55,
                                               fixed_palette=True)
    assert np.array_equal(palette_lab, full)
    # Non-fixed full palette: the nearest entry is replaced by the outline color (documented).
    out, palette_lab = postprocess.add_outline(sprite, full, "#c82828", 0.55)
    assert len(palette_lab) == 256 and not np.array_equal(palette_lab, full)
    assert np.allclose(palette_lab[out[1, 2]], color.rgb8_to_lab(color.hex_to_rgb8("#c82828")))
