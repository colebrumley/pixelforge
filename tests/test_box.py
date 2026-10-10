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
