import json
from pathlib import Path

import numpy as np
import pytest

from pixelforge import _native, color


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


# ------------------------------------------------------------------ native conversions

ORACLE = Path(__file__).parent / "oracle"


def _oracle(name):
    return json.loads((ORACLE / name).read_text())


def test_native_matches_skimage_oracle():
    lattice = _oracle("lab_lattice.json")
    lab = color.rgb_to_lab(np.array(lattice["rgb"]))
    assert np.abs(lab - np.array(lattice["lab"])).max() <= 1e-9
    rt = _oracle("lab_roundtrip.json")
    lab_in = np.array(rt["lab"])
    assert np.abs(color.lab_to_rgb(lab_in) - np.array(rt["lab_to_rgb"])).max() <= 1e-9
    assert np.array_equal(color.lab_to_rgb8(lab_in), np.array(rt["lab_to_rgb8"], np.uint8))
    r8 = _oracle("rgb8_lab.json")
    got = color.rgb8_to_lab(np.array(r8["rgb8"], np.uint8))
    assert np.abs(got - np.array(r8["lab"])).max() <= 1e-9


@pytest.mark.parametrize("shape", [(3,), (0, 3), (5, 3), (4, 6, 3), (2, 1, 2, 3)])
def test_native_keeps_leading_shape(shape):
    rng = np.random.default_rng(7)
    rgb8 = rng.integers(0, 256, shape, dtype=np.uint8)
    rgb = rgb8 / 255.0
    lab = color.rgb_to_lab(rgb)
    assert lab.shape == shape and lab.dtype == np.float64
    assert np.array_equal(color.rgb8_to_lab(rgb8), lab)
    back = color.lab_to_rgb(lab)
    assert back.shape == shape and back.dtype == np.float64
    out8 = color.lab_to_rgb8(lab)
    assert out8.shape == shape and out8.dtype == np.uint8
    assert np.array_equal(out8, rgb8)
    # Flattening first gives the same values: the conversion is per pixel.
    assert np.array_equal(_native.rgb_to_lab(rgb.reshape(-1, 3)), lab.reshape(-1, 3))


def test_rgb8_to_lab_accepts_other_dtypes():
    rgb8 = np.array([[12, 34, 56], [255, 0, 128]])
    assert rgb8.dtype != np.uint8
    assert np.array_equal(color.rgb8_to_lab(rgb8), color.rgb8_to_lab(rgb8.astype(np.uint8)))
    assert np.array_equal(color.rgb8_to_lab(rgb8.tolist()), color.rgb8_to_lab(rgb8))


def test_python_wrappers_accept_non_contiguous_and_lists():
    rgb = np.random.default_rng(3).random((6, 3))
    assert np.array_equal(color.rgb_to_lab(rgb[::2]), color.rgb_to_lab(rgb[::2].copy()))
    assert np.array_equal(color.rgb_to_lab(rgb.tolist()), color.rgb_to_lab(rgb))
    assert np.array_equal(color.lab_to_rgb8(color.rgb_to_lab(rgb).astype(np.float32)),
                          color.lab_to_rgb8(color.rgb_to_lab(rgb).astype(np.float32)
                                            .astype(np.float64)))
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        color.rgb_to_lab(np.zeros((2, 4)))


@pytest.mark.parametrize("func, name, bad, message", [
    ("rgb_to_lab", "rgb", np.zeros((2, 3), np.float32),
     "rgb_to_lab: rgb must be a float64 numpy.ndarray, got float32"),
    ("lab_to_rgb", "lab", [[50.0, 0.0, 0.0]],
     "lab_to_rgb: lab must be a float64 numpy.ndarray, got <class 'list'>"),
    ("rgb8_to_lab", "rgb8", np.zeros((2, 3)),
     "rgb8_to_lab: rgb8 must be a uint8 numpy.ndarray, got float64"),
    ("lab_to_rgb8", "lab", np.zeros((2, 4)),
     "lab_to_rgb8: lab must have shape (..., 3), got (2, 4)"),
    ("rgb_to_lab", "rgb", np.zeros(()),
     "rgb_to_lab: rgb must have shape (..., 3), got ()"),
    ("lab_to_rgb", "lab", np.zeros((4, 3))[::2],
     "lab_to_rgb: lab must be C-contiguous (use numpy.ascontiguousarray)"),
])
def test_native_conversion_errors(func, name, bad, message):
    with pytest.raises(ValueError) as err:
        getattr(_native, func)(bad)
    assert str(err.value) == message


def test_native_nearest_index_errors():
    with pytest.raises(ValueError, match=r"^nearest_index: pixels must have shape \(N, 3\), "
                                         r"got \(2, 2, 3\)$"):
        _native.nearest_index(np.zeros((2, 2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match=r"^nearest_index: palette must have shape \(N, 3\), "
                                         r"got \(3,\)$"):
        _native.nearest_index(np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="^nearest_index: palette must be a float64"):
        _native.nearest_index(np.zeros((2, 3)), np.zeros((1, 3), np.int64))
    with pytest.raises(ValueError, match="^nearest_index: palette must not be empty$"):
        _native.nearest_index(np.zeros((2, 3)), np.zeros((0, 3)))
    out = _native.nearest_index(np.zeros((0, 3)), np.zeros((0, 3)))
    assert out.shape == (0,) and out.dtype == np.int64


def test_nearest_index_ties_go_to_lowest_index():
    palette = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pixels = np.array([[5.0, 0.0, 0.0],      # exact duplicate entries 2 and 3
                       [2.5, 0.0, 0.0],      # halfway between entries 0 and 2
                       [7.5, 0.0, 0.0],      # halfway between entries 2 and 1
                       [-1.0, 0.0, 0.0]])
    out = color.nearest_index(pixels, palette)
    assert out.dtype == np.int64 and out.tolist() == [2, 0, 1, 0]
    # Leading shape is flattened like before.
    assert color.nearest_index(pixels.reshape(2, 2, 3), palette).tolist() == [2, 0, 1, 0]


def test_nearest_index_matches_brute_force_argmin():
    rng = np.random.default_rng(11)
    palette = rng.uniform(-50, 50, (37, 3))
    pixels = rng.uniform(-60, 60, (500, 3))
    d = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
    assert np.array_equal(color.nearest_index(pixels, palette), d.argmin(axis=1))
