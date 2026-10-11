"""Baseline: box (area-mean) downsample in CIELAB.

The filter runs natively (crates/pixelforge/src/boxfilter.rs, bitwise the numpy version it
replaced; tests/test_box_native.py keeps that version as the reference).
"""

from __future__ import annotations

import numpy as np

from .. import _native
from . import Downscaled


def cell_index(n_in: int, n_out: int) -> np.ndarray:
    """Output cell of every input coordinate along one axis (exact integer arithmetic)."""
    return (np.arange(n_in, dtype=np.int64) * n_out) // n_in


def run(lab: np.ndarray, mask: np.ndarray, out_width: int, out_height: int, config,
        weight: np.ndarray | None = None) -> Downscaled:
    """Mean LAB per cell, each opaque pixel weighted by `weight`; a cell is opaque when at
    least half of its pixels are (binary mask count). Raises PixelforgeError when no pixel
    is opaque."""
    out = _native.box_run(
        np.ascontiguousarray(lab, dtype=np.float64), np.ascontiguousarray(mask, dtype=bool),
        int(out_width), int(out_height), config.to_dict(),
        None if weight is None else np.ascontiguousarray(weight, dtype=np.float64))
    return Downscaled(**out)
