import numpy as np

from pixelforge import color
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
