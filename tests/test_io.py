import os

import numpy as np
import pytest
from PIL import Image

from pixelforge import Config, io, preprocess
from pixelforge.errors import PixelforgeError


def _png(path, array, mode=None):
    Image.fromarray(array, mode=mode).save(path)
    return path


def test_pixel_budget_is_checked_before_decoding(tmp_path):
    path = _png(tmp_path / "big.png", np.zeros((20, 30, 3), dtype=np.uint8))
    with pytest.raises(PixelforgeError, match="600 pixels exceeds the input budget of 599"):
        io.load(path, max_pixels=599)
    assert io.load(path, max_pixels=600).rgb.shape == (20, 30, 3)


def test_preprocessing_errors_still_importable_from_preprocess():
    assert preprocess.PixelforgeError is PixelforgeError


def test_non_image_bytes_give_a_clean_error(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n")
    with pytest.raises(PixelforgeError, match="not a supported image"):
        io.load(path)


def test_16_bit_grayscale_is_rescaled_not_clipped(tmp_path):
    values = np.array([[7710, 64250], [64250, 7710]], dtype=np.uint16)
    path = tmp_path / "gray16.png"
    Image.fromarray(values).save(path)
    with Image.open(path) as im:
        assert im.mode in ("I;16", "I")
    loaded = io.load(path)
    rgb8 = np.round(loaded.rgb * 255).astype(int)
    assert rgb8[0, 0].tolist() == [30, 30, 30]
    assert rgb8[0, 1].tolist() == [250, 250, 250]


def test_exif_orientation_is_applied(tmp_path):
    array = np.zeros((4, 8, 3), dtype=np.uint8)
    array[:, :4] = 255                     # left half white, stored 8 wide × 4 high
    im = Image.fromarray(array)
    exif = im.getexif()
    exif[0x0112] = 6                       # rotate 90° clockwise to display
    path = tmp_path / "rotated.png"
    im.save(path, exif=exif)
    loaded = io.load(path)
    assert loaded.rgb.shape == (8, 4, 3)
    assert loaded.rgb[:4].min() == 1.0 and loaded.rgb[4:].max() == 0.0   # white on top


def test_multi_frame_uses_first_frame_with_a_warning(tmp_path):
    frames = [Image.new("RGB", (4, 4), c) for c in ((255, 0, 0), (0, 0, 255))]
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    with pytest.warns(UserWarning, match="2 frames"):
        loaded = io.load(path)
    assert loaded.rgb[0, 0, 0] > 0.9 and loaded.rgb[0, 0, 2] < 0.1


def test_directory_is_refused(tmp_path):
    with pytest.raises(PixelforgeError, match="not a regular file"):
        io.load(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no mkfifo on this platform")
def test_fifo_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    with pytest.raises(PixelforgeError, match="not a regular file"):
        io.load(fifo)
    with pytest.raises(PixelforgeError, match="not a regular file"):
        io.sha256_file(fifo)


def test_small_input_repeat_respects_the_budget():
    rgb = np.ones((1, 4000, 3)) * 0.5
    alpha = np.full((1, 4000), 255, dtype=np.uint8)
    config = Config(preset="background", out_width=512, out_height=512, tileset=False,
                    denoise="none")
    with pytest.raises(PixelforgeError, match="too small"):
        preprocess.run(rgb, alpha, config)
