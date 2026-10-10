"""Baseline: box (area-mean) downsample in CIELAB."""

from __future__ import annotations

import numpy as np

from ..errors import PixelforgeError
from . import Downscaled


def cell_index(n_in: int, n_out: int) -> np.ndarray:
    """Output cell of every input coordinate along one axis (exact integer arithmetic)."""
    return (np.arange(n_in, dtype=np.int64) * n_out) // n_in


def run(lab: np.ndarray, mask: np.ndarray, out_width: int, out_height: int, config,
        weight: np.ndarray | None = None) -> Downscaled:
    """Mean LAB per cell, each opaque pixel weighted by `weight`; a cell is opaque when at
    least half of its pixels are (binary mask count)."""
    if not mask.any():
        raise PixelforgeError("no opaque pixels")
    hi, wi = mask.shape
    cell = (cell_index(hi, out_height)[:, None] * out_width
            + cell_index(wi, out_width)[None, :]).ravel()
    n_cells = out_width * out_height
    flat_mask = mask.ravel()
    total = np.bincount(cell, minlength=n_cells)
    opaque = np.bincount(cell[flat_mask], minlength=n_cells)

    colors = lab.reshape(-1, 3)[flat_mask]
    if weight is None:
        w, mass = None, opaque
    else:
        w = weight.ravel()[flat_mask]
        mass = np.bincount(cell[flat_mask], weights=w, minlength=n_cells)
    # Sum offsets from a reference color so a solid image averages back to itself exactly.
    reference = colors[0]
    small = np.zeros((n_cells, 3), dtype=np.float64)
    filled = opaque > 0
    for c in range(3):
        offset = colors[:, c] - reference[c]
        sums = np.bincount(cell[flat_mask], weights=offset if w is None else w * offset,
                           minlength=n_cells)
        small[filled, c] = reference[c] + sums[filled] / mass[filled]

    small_mask = (opaque * 2 >= total) & filled
    return Downscaled(small_lab=small.reshape(out_height, out_width, 3),
                      small_mask=small_mask.reshape(out_height, out_width),
                      stats={"iterations": 1})
