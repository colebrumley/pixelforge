import numpy as np

from pixelforge import Config, color
from pixelforge.downscale import gerstner


def test_two_color_image_gives_two_colors(preprocessed):
    pre, config = preprocessed("two_color", method="gerstner", palette_size=2, out_width=8,
                               out_height=8, saturation_beta=1.0)
    out = gerstner.run(pre.lab, pre.mask, 8, 8, config)
    assert out.palette_lab.shape == (2, 3)
    red, blue = color.rgb8_to_lab(np.array([[255, 0, 0], [0, 0, 255]]))
    left, right = out.indices[0, 0], out.indices[0, 7]
    assert left != right
    assert color.delta_e(out.palette_lab[left], red) < 3
    assert color.delta_e(out.palette_lab[right], blue) < 3
    assert (out.indices[:, :4] == left).all() and (out.indices[:, 4:] == right).all()
    assert out.small_mask.all()


def test_palette_grows_to_K(preprocessed):
    pre, config = preprocessed("gradient6", method="gerstner", palette_size=6, out_width=16,
                               out_height=16)
    out = gerstner.run(pre.lab, pre.mask, 16, 16, config)
    assert out.stats["palette_size"] == 6 and len(out.palette_lab) == 6
    assert not out.stats["hit_iteration_cap"]
    assert len(np.unique(out.indices)) == 6


def test_iteration_cap(preprocessed):
    pre, config = preprocessed("gradient6", method="gerstner", palette_size=6, out_width=16,
                               out_height=16, g_max_iters=3)
    out = gerstner.run(pre.lab, pre.mask, 16, 16, config)
    assert out.stats["iterations"] == 3 and out.stats["hit_iteration_cap"]
    assert 1 <= len(out.palette_lab) <= 6
    assert out.indices.shape == (16, 16)
    assert out.indices.min() >= 0 and out.indices.max() < len(out.palette_lab)
    assert np.isfinite(out.small_lab).all() and out.small_mask.all()


def test_transparent_superpixels_are_masked_out(preprocessed):
    pre, config = preprocessed("circle_alpha", method="gerstner", palette_size=4, out_width=32,
                               out_height=32, crop_to_alpha=False)
    out = gerstner.run(pre.lab, pre.mask, 32, 32, config)
    support = pre.mask.reshape(32, 8, 32, 8).any(axis=(1, 3))
    assert not (out.small_mask & ~support).any()
    assert out.small_mask[12:20, 12:20].all()


def _run_lab(lab, palette_size, **overrides):
    h, w = lab.shape[:2]
    config = Config(method="gerstner", palette_size=palette_size, out_width=w // 4,
                    out_height=h // 4, **overrides)
    return gerstner.run(lab, np.ones((h, w), dtype=bool), w // 4, h // 4, config), config


def test_close_colors_split():
    # ΔE 4 is clearly visible; after the bilateral filter the halves sit ≈ 2.8 apart, which
    # the 3·2δ growth rule alone never reached (1 color, annealing ran down to T ≈ 1e-3).
    lab = np.zeros((32, 32, 3))
    lab[..., 0] = 50.0
    lab[:, 16:, 0] = 54.0
    out, config = _run_lab(lab, 2)
    assert len(out.palette_lab) == 2
    assert (out.indices[:, :4] == out.indices[0, 0]).all()
    assert (out.indices[:, 4:] == out.indices[0, 7]).all()
    assert out.indices[0, 0] != out.indices[0, 7]
    assert out.stats["final_temperature"] >= 1e-2 * config.g_T_final


def test_single_color_stops_early():
    out, config = _run_lab(np.tile([60.0, 10.0, -20.0], (32, 32, 1)), 8)
    assert len(out.palette_lab) == 1
    assert not out.stats["hit_iteration_cap"]
    assert out.stats["iterations"] <= 5
    assert out.stats["final_temperature"] >= 1e-2 * config.g_T_final   # not the backstop


def test_three_colors_with_large_K():
    lab = np.zeros((32, 32, 3))
    lab[:, :11] = [30.0, 20.0, 10.0]
    lab[:, 11:22] = [60.0, -30.0, 40.0]
    lab[:, 22:] = [85.0, 5.0, -40.0]
    out, _ = _run_lab(lab, 16, saturation_beta=1.0)
    assert len(out.palette_lab) == 3
    for c in lab[0, [0, 16, 31]]:
        assert color.delta_e(out.palette_lab, c).min() < 3


def test_checker_tiles_iteration_count(preprocessed):
    # T0 ∝ spread (not variance): 14 iterations here, down from 25. Bound = 14 + 20%.
    pre, config = preprocessed("checker_tiles", method="gerstner", palette_size=8, out_width=16,
                               out_height=16)
    out = gerstner.run(pre.lab, pre.mask, 16, 16, config)
    assert not out.stats["hit_iteration_cap"]
    assert out.stats["iterations"] <= 16


def test_stalled_pair_ends_annealing(monkeypatch):
    # With the spread cap disabled the L 50 / L 54 pair keeps creeping apart without ever
    # becoming ready; annealing must stop once it stalls, not at the 1e-3·T_final backstop.
    monkeypatch.setattr(gerstner, "SPLIT_SPREAD", 1e9)
    lab = np.zeros((32, 32, 3))
    lab[..., 0] = 50.0
    lab[:, 16:, 0] = 54.0
    out, config = _run_lab(lab, 2)
    assert not out.stats["hit_iteration_cap"]
    assert out.stats["final_temperature"] >= 1e-2 * config.g_T_final


def test_empty_mask_raises_pixelforge_error():
    import pytest

    from pixelforge import PixelforgeError

    lab = np.zeros((16, 16, 3))
    with pytest.raises(PixelforgeError, match="no opaque pixels"):
        gerstner.run(lab, np.zeros((16, 16), dtype=bool), 8, 8, Config(method="gerstner"))
