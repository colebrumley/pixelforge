"""Map an image to a palette, with optional ordered dithering (Section 8.2)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import color

# Typical L step between ramp colors; the threshold matrix perturbs L by up to ±half of this.
DITHER_L_AMPLITUDE = 12.0


def bayer(n: int) -> np.ndarray:
    """n×n Bayer threshold matrix (n a power of two), normalized to B/n² − 0.5."""
    m = np.array([[0, 2], [3, 1]], dtype=np.int64)
    while len(m) < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m / float(n * n) - 0.5


def local_std(lab: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Local LAB standard deviation over a 3×3 window (root of the summed channel variances).

    With `mask`, only opaque cells enter the window (masked mean and variance); transparent
    cells get 0.
    """
    lab = np.asarray(lab, dtype=np.float64)
    if mask is None or mask.all():
        variance = np.zeros(lab.shape[:2])
        for c in range(3):
            mean = ndimage.uniform_filter(lab[..., c], size=3, mode="nearest")
            mean_sq = ndimage.uniform_filter(lab[..., c] * lab[..., c], size=3, mode="nearest")
            variance += np.maximum(mean_sq - mean * mean, 0.0)
        return np.sqrt(variance)
    weight = mask.astype(np.float64)
    count = ndimage.uniform_filter(weight, size=3, mode="nearest")
    safe = np.where(mask, np.maximum(count, 1e-12), 1.0)   # an opaque cell counts itself
    variance = np.zeros(lab.shape[:2])
    for c in range(3):
        x = np.where(mask, lab[..., c], 0.0)
        mean = ndimage.uniform_filter(x, size=3, mode="nearest") / safe
        mean_sq = ndimage.uniform_filter(x * x, size=3, mode="nearest") / safe
        variance += np.maximum(mean_sq - mean * mean, 0.0)
    return np.where(mask, np.sqrt(variance), 0.0)


def dither_offsets(small_lab: np.ndarray, config,
                   mask: np.ndarray | None = None) -> np.ndarray | None:
    """Per-pixel LAB offset (H, W, 3) from the ordered threshold matrix, or None.

    `mask` (True = opaque) keeps transparent cells out of the "auto" local-std window.
    """
    if config.dither == "none" or config.dither_strength == 0:
        return None
    h, w = small_lab.shape[:2]
    n = 8 if config.dither == "bayer8" else 4
    matrix = bayer(n)
    threshold = matrix[np.arange(h)[:, None] % n, np.arange(w)[None, :] % n]
    amount = config.dither_strength * DITHER_L_AMPLITUDE * threshold
    if config.dither == "auto":
        # Only smooth gradients are dithered, not edges or detailed regions.
        std = local_std(small_lab, mask)
        amount = np.where(std < config.dither_variance_threshold, amount, 0.0)
    return np.stack([amount, 0.5 * amount, 0.5 * amount], axis=-1)


def run(small_lab: np.ndarray, small_mask: np.ndarray, palette_lab: np.ndarray,
        config) -> np.ndarray:
    """Index image (H, W): nearest palette color per opaque pixel, -1 where transparent."""
    lab = np.asarray(small_lab, dtype=np.float64)
    offsets = dither_offsets(lab, config, small_mask)
    if offsets is not None:
        lab = lab + offsets
    indices = np.full(small_mask.shape, -1, dtype=np.int64)
    indices[small_mask] = color.nearest_index(lab[small_mask], palette_lab)
    return indices
