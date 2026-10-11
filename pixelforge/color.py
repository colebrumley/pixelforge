"""sRGB <-> CIELAB conversion and palette distance helpers.

The conversions and the nearest-colour search run natively (`pixelforge._native`, from
crates/pixelforge/src/color.rs): scikit-image 0.26's formulas with libm's `pow`/`cbrt`, so
results agree with scikit-image's `rgb2lab`/`lab2rgb` to about 1e-13 and are bit-identical
on every platform.
"""

from __future__ import annotations

import numpy as np

from . import _native


def _triples(arr: np.ndarray, dtype) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=dtype)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"expected an array of shape (..., 3), got {arr.shape}")
    return arr


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] (..., 3) -> CIELAB (D65). L in [0, 100]."""
    return _native.rgb_to_lab(_triples(rgb, np.float64))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (..., 3) -> sRGB in [0, 1], clipped to the gamut."""
    return _native.lab_to_rgb(_triples(lab, np.float64))


def rgb8_to_lab(rgb8: np.ndarray) -> np.ndarray:
    rgb8 = np.asarray(rgb8)
    if rgb8.dtype == np.uint8:
        return _native.rgb8_to_lab(_triples(rgb8, np.uint8))
    # Other integer or float inputs: the same `/ 255.0` without a lossy cast to uint8.
    return rgb_to_lab(np.asarray(rgb8, dtype=np.float64) / 255.0)


def lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    """CIELAB (..., 3) -> 8-bit sRGB, rounded half to even like `np.round`."""
    return _native.lab_to_rgb8(_triples(lab, np.float64))


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 ΔE (Euclidean distance in LAB) along the last axis."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt(np.sum(d * d, axis=-1))


def nearest_index(pixels_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color (CIE76) for every row of pixels_lab.

    Brute force over the squared distance; ties go to the lowest palette index.
    """
    pixels_lab = np.ascontiguousarray(pixels_lab, dtype=np.float64).reshape(-1, 3)
    palette_lab = np.ascontiguousarray(palette_lab, dtype=np.float64).reshape(-1, 3)
    return _native.nearest_index(pixels_lab, palette_lab)


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
