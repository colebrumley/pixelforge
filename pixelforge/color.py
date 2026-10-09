"""sRGB <-> CIELAB conversion and palette distance helpers."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree
from skimage import color as skcolor


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] (..., 3) -> CIELAB (D65). L in [0, 100]."""
    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    return skcolor.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(rgb.shape)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (..., 3) -> sRGB in [0, 1], clipped to the gamut."""
    lab = np.ascontiguousarray(lab, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # out-of-gamut colors are clipped below
        rgb = skcolor.lab2rgb(lab.reshape(-1, 1, 3)).reshape(lab.shape)
    return np.clip(rgb, 0.0, 1.0)


def rgb8_to_lab(rgb8: np.ndarray) -> np.ndarray:
    return rgb_to_lab(np.asarray(rgb8, dtype=np.float64) / 255.0)


def lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    rgb = lab_to_rgb(lab) * 255.0
    return np.round(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 ΔE (Euclidean distance in LAB) along the last axis."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt(np.sum(d * d, axis=-1))


def nearest_index(pixels_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color (CIE76) for every row of pixels_lab."""
    pixels_lab = np.ascontiguousarray(pixels_lab, dtype=np.float64).reshape(-1, 3)
    if len(pixels_lab) == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(np.ascontiguousarray(palette_lab, dtype=np.float64))
    return tree.query(pixels_lab)[1].astype(np.int64)


def palette_distance_matrix(palette_lab: np.ndarray) -> np.ndarray:
    p = np.asarray(palette_lab, dtype=np.float64)
    return delta_e(p[:, None, :], p[None, :, :])


def hex_to_rgb8(text: str) -> np.ndarray:
    text = text.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"not a hex color: {text!r}")
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def rgb8_to_hex(rgb8) -> str:
    r, g, b = (int(v) for v in rgb8)
    return f"#{r:02x}{g:02x}{b:02x}"
