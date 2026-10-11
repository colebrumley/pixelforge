"""Map an image to a palette, with optional ordered dithering (Section 8.2).

Runs natively (`pixelforge._native`, from crates/pixelforge/src/quantize.rs), bitwise
identical to the former numpy/scipy version: the 3×3 window is scipy's
`uniform_filter(size=3, mode="nearest")` reproduced operation for operation.
"""

from __future__ import annotations

import numpy as np

from . import _native

# Typical L step between ramp colors; the threshold matrix perturbs L by up to ±half of this.
DITHER_L_AMPLITUDE = 12.0


def _lab(lab: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(lab, dtype=np.float64)


def _mask(mask: np.ndarray | None) -> np.ndarray | None:
    return None if mask is None else np.ascontiguousarray(mask, dtype=bool)


def bayer(n: int) -> np.ndarray:
    """n×n Bayer threshold matrix (n a power of two), normalized to B/n² − 0.5."""
    return _native.bayer(n)


def local_std(lab: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Local LAB standard deviation over a 3×3 window (root of the summed channel variances).

    With `mask`, only opaque cells enter the window (masked mean and variance); transparent
    cells get 0.
    """
    return _native.local_std(_lab(lab), _mask(mask))


def dither_offsets(small_lab: np.ndarray, config,
                   mask: np.ndarray | None = None) -> np.ndarray | None:
    """Per-pixel LAB offset (H, W, 3) from the ordered threshold matrix, or None.

    `mask` (True = opaque) keeps transparent cells out of the "auto" local-std window.
    """
    return _native.dither_offsets(_lab(small_lab), _mask(mask), config.to_dict())


def run(small_lab: np.ndarray, small_mask: np.ndarray, palette_lab: np.ndarray,
        config) -> np.ndarray:
    """Index image (H, W): nearest palette color per opaque pixel, -1 where transparent."""
    return _native.quantize_run(_lab(small_lab), _mask(small_mask),
                                np.ascontiguousarray(palette_lab, dtype=np.float64)
                                .reshape(-1, 3),
                                config.to_dict())
