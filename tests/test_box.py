import numpy as np

from pixelforge import Config, color
from pixelforge.downscale import box


def test_solid_color_is_returned_exactly():
    rgb8 = np.array([200, 120, 40], dtype=np.uint8)
    lab = np.broadcast_to(color.rgb8_to_lab(rgb8), (96, 128, 3)).copy()
    mask = np.ones((96, 128), dtype=bool)
    out = box.run(lab, mask, 16, 12, Config(method="box"))
    assert out.small_lab.shape == (12, 16, 3) and out.small_mask.all()
    assert np.array_equal(out.small_lab, np.broadcast_to(lab[0, 0], (12, 16, 3)))
    assert (color.lab_to_rgb8(out.small_lab) == rgb8).all()


def test_non_integer_ratio_and_mask():
    lab = np.zeros((50, 70, 3))
    lab[..., 0] = np.arange(70)[None, :]
    mask = np.ones((50, 70), dtype=bool)
    mask[:, :35] = False
    out = box.run(lab, mask, 10, 8, Config(method="box"))
    assert not out.small_mask[:, :5].any() and out.small_mask[:, 5:].all()
    assert np.allclose(out.small_lab[0, 5:, 0], np.arange(35, 70).reshape(5, 7).mean(axis=1))


def test_empty_mask_raises_pixelforge_error():
    import pytest

    from pixelforge import PixelforgeError

    lab = np.zeros((16, 16, 3))
    with pytest.raises(PixelforgeError, match="no opaque pixels"):
        box.run(lab, np.zeros((16, 16), dtype=bool), 8, 8, Config(method="box"))


def _fringed_square():
    """Red square with a 4-px white fringe at alpha 128 (right) and 127 (left)."""
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[8:56, 8:56] = (220, 0, 20, 255)
    rgba[8:56, 56:60] = (255, 255, 255, 128)
    rgba[8:56, 4:8] = (255, 255, 255, 127)
    return rgba


def test_semi_transparent_fringe_pulls_in_proportion_to_alpha():
    from pixelforge import preprocess

    rgba = _fringed_square()
    config = Config(method="box", out_width=16, out_height=16, palette_size=4,
                    alpha_threshold=128, outline="none", denoise="none")
    pre = preprocess.run(rgba[..., :3] / 255.0, rgba[..., 3], config)
    assert np.allclose(np.unique(pre.weight), [0.0, 128 / 255, 1.0])
    assert np.array_equal(pre.weight > 0, pre.mask)
    plain = box.run(pre.lab, pre.mask, 16, 16, config)
    out = box.run(pre.lab, pre.mask, 16, 16, config, weight=pre.weight)
    assert np.array_equal(out.small_mask, plain.small_mask)     # membership stays binary
    # The output column that mixes red with the alpha-128 fringe is the alpha-weighted mean.
    red_l, white_l = color.rgb8_to_lab(np.array([[220, 0, 20], [255, 255, 255]]))[:, 0]
    mixed = np.nonzero(np.abs(plain.small_lab[8, :, 0] - (red_l + white_l) / 2) < 25)[0]
    assert len(mixed) == 1
    col = int(mixed[0])
    hi, wi = pre.mask.shape
    xs = np.nonzero(box.cell_index(wi, 16) == col)[0]
    ys = np.nonzero(box.cell_index(hi, 16) == 8)[0]
    w = pre.weight[np.ix_(ys, xs)]
    expected = (w * pre.lab[np.ix_(ys, xs)][..., 0]).sum() / w.sum()
    assert np.isclose(out.small_lab[8, col, 0], expected)
    assert out.small_lab[8, col, 0] < plain.small_lab[8, col, 0] - 5.0
    assert not pre.mask[:, 0].any()     # the alpha-127 fringe is transparent (and cropped)


def test_all_ones_weight_matches_unweighted(preprocessed):
    for name in ("line_diag", "circle_alpha", "noisy_gradient"):
        pre, config = preprocessed(name, method="box", out_width=16, out_height=16)
        plain = box.run(pre.lab, pre.mask, 16, 16, config)
        ones = box.run(pre.lab, pre.mask, 16, 16, config, weight=np.ones(pre.mask.shape))
        assert np.array_equal(ones.small_lab, plain.small_lab)
        assert np.array_equal(ones.small_mask, plain.small_mask)
