"""Downscalers. Each module exposes run(lab, mask, out_width, out_height, config, weight=None).

`mask` decides which input pixels are opaque; `weight` (H, W) in [0, 1], the input alpha,
scales how much each opaque pixel counts in the color averages (None = 1 everywhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Downscaled:
    small_lab: np.ndarray                  # (ho, wo, 3) float64 CIELAB
    small_mask: np.ndarray                 # (ho, wo) bool, True = opaque
    palette_lab: np.ndarray | None = None  # gerstner only: converged palette (K, 3), β applied
    palette_lab_unsaturated: np.ndarray | None = None  # gerstner only: palette before β
    indices: np.ndarray | None = None      # gerstner only: (ho, wo) palette index per pixel
    mean_lab: np.ndarray | None = None     # gerstner only: smoothed superpixel mean colors
    stats: dict = field(default_factory=dict)


def neighbor_mean4(grid: np.ndarray) -> np.ndarray:
    """Mean of each value's 4-connected neighbors (edges average the neighbors they have)."""
    total = np.zeros_like(grid)
    count = np.zeros_like(grid)
    total[1:] += grid[:-1]
    count[1:] += 1
    total[:-1] += grid[1:]
    count[:-1] += 1
    total[:, 1:] += grid[:, :-1]
    count[:, 1:] += 1
    total[:, :-1] += grid[:, 1:]
    count[:, :-1] += 1
    return np.where(count > 0, total / np.maximum(count, 1), grid)


def get(method: str):
    from . import box, gerstner, kopf

    return {"box": box, "kopf": kopf, "gerstner": gerstner}[method]
