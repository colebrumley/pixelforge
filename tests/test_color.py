import numpy as np

from pixelforge import color


def test_round_trip_lattice():
    axis = np.linspace(0.0, 1.0, 10)
    rgb = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    assert len(rgb) == 1000
    back = color.lab_to_rgb(color.rgb_to_lab(rgb))
    assert np.abs(back - rgb).max() <= 1.0 / 255.0


def test_white_and_black():
    lab = color.rgb_to_lab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert abs(lab[0, 0] - 100.0) <= 0.01
    assert abs(lab[1, 0]) <= 0.01


def test_rgb8_rounding_and_hex():
    rgb8 = np.array([[255, 0, 0], [12, 34, 56], [0, 0, 0]], dtype=np.uint8)
    assert np.array_equal(color.lab_to_rgb8(color.rgb8_to_lab(rgb8)), rgb8)
    assert color.rgb8_to_hex(rgb8[1]) == "#0c2238"
    assert np.array_equal(color.hex_to_rgb8("#0c2238"), rgb8[1])


def test_nearest_index():
    palette = color.rgb8_to_lab(np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]]))
    pixels = color.rgb8_to_lab(np.array([[250, 10, 10], [20, 20, 20], [240, 240, 240]]))
    assert color.nearest_index(pixels, palette).tolist() == [2, 0, 1]
