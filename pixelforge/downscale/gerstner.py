"""Joint superpixel + palette optimization (Gerstner et al. 2013), REQUIREMENTS.md Section 7.

The input is segmented into exactly wo·ho superpixels (one per output pixel) while a K-color
palette is grown by mass-constrained deterministic annealing (MCDA). `PaletteAnnealer` is the
palette half on its own; `palette.mcda` reuses it directly on pixel colors.

The whole stage runs in the native extension (`pixelforge._native.gerstner_run`, from
crates/pixelforge/src/gerstner_run.rs and annealer.rs over the loops in gerstner.rs); this
module converts the arguments and the result. The Python control flow it replaced is kept in
tests/gerstner_reference.py. Progress is logged to the `pixelforge.gerstner` logger.
"""

from __future__ import annotations

import math

import numpy as np

from .. import _native
from ..errors import PixelforgeError
from . import Downscaled, neighbor_mean4
from .box import cell_index

# The constants of the native stage (`_native.gerstner_constants()` holds the library's
# values). The four tunables are passed to it on every call, so patching them here takes
# effect; the others are fixed in the library.
PERTURB_DELTA = 0.5      # sub-cluster offset along the principal axis, LAB units
SPLIT_GROWTH = 3.0       # a pair splits once it is this many times further apart than placed
SPLIT_SPREAD = 1.0       # ... or once it is this many standard deviations of its cluster apart
STALL_EVENTS = 3         # below g_T_final, stop once no pair grew over this many events
EXHAUSTED_T_FRACTION = 1e-3   # backstop: stop annealing below this fraction of g_T_final
_MAX_SEARCH_RADIUS = 6   # grid cells searched around a pixel's home cell (safety bound)
PROGRESS_EVERY = 10      # iterations between INFO progress lines (DEBUG logs every one)


def _tunables() -> dict:
    return {"PERTURB_DELTA": PERTURB_DELTA, "SPLIT_GROWTH": SPLIT_GROWTH,
            "SPLIT_SPREAD": SPLIT_SPREAD, "EXHAUSTED_T_FRACTION": EXHAUSTED_T_FRACTION}


def _points(points) -> np.ndarray:
    return np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)


def principal_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit first principal component of (n, 3) points and the variance along it.

    Falls back to (1, 0, 0) with zero variance for fewer than 2 points or no spread. The sign
    is normalized (largest-magnitude component positive) so it does not depend on the
    eigensolver. Native (Jacobi eigensolver, sequential sums).
    """
    return _native.principal_axis(_points(points))


class PaletteAnnealer:
    """ASSOCIATE / REFINE / CONVERGENCE CHECK / EXPAND of Algorithm 1 (native).

    While sub-clusters exist, cluster 2k and 2k+1 are the pair of palette color k and the
    displayed color is their average. The state (`T`, `clusters`, `prob`, `placed`,
    `history`, `has_sub`, `assign`, `done`, `colors`) reads and writes through to the
    native annealer.
    """

    def __init__(self, init_points: np.ndarray, palette_size: int, config):
        object.__setattr__(self, "_inner", _native.PaletteAnnealer(
            _points(init_points), int(palette_size), config.to_dict(), _tunables()))

    def step(self, points: np.ndarray, weights: np.ndarray) -> None:
        """One iteration on (n, 3) points with prior P(p_s) = weights."""
        self._inner.step(_points(points), np.ascontiguousarray(weights, dtype=np.float64))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


def _laplacian_smooth(grid: np.ndarray, fraction: float) -> np.ndarray:
    """Move every value `fraction` of the way toward the mean of its 4-connected neighbors."""
    return grid + fraction * (neighbor_mean4(grid) - grid)


def bilateral_filter(image: np.ndarray, active: np.ndarray, sigma_color: float,
                     sigma_spatial: float) -> np.ndarray:
    """Bilateral filter of an (H, W, 3) LAB image; pixels with active == False are excluded.

    radius = ceil(2·σ_s), zero padding, weight = exp(−ΔE²/(2σ_c²) − |d|²/(2σ_s²))·active;
    pixels whose weights sum to 0 keep their value. Native (`_native.bilateral_filter`).
    """
    return _native.bilateral_filter(np.ascontiguousarray(image, dtype=np.float64),
                                    np.ascontiguousarray(active), float(sigma_color),
                                    float(sigma_spatial))


class _Grid:
    """Geometry shared by the SLIC assignment passes."""

    def __init__(self, hi: int, wi: int, ho: int, wo: int):
        self.hi, self.wi, self.ho, self.wo = hi, wi, ho, wo
        self.rx, self.ry = wi / wo, hi / ho
        self.cx0 = np.tile((np.arange(wo) + 0.5) * self.rx, ho)
        self.cy0 = np.repeat((np.arange(ho) + 0.5) * self.ry, wo)
        self.px = np.arange(wi) + 0.5            # pixel centers
        self.py = np.arange(hi) + 0.5
        # Initial-grid cell containing each pixel column / row.
        self.home_x = cell_index(wi, wo)
        self.home_y = cell_index(hi, ho)

    def search_radius(self, cx: np.ndarray, cy: np.ndarray) -> int:
        """Grid cells around the home cell that can hold a center inside the 2rx × 2ry window.

        A center that drifted D cells from its initial grid position can only be within the
        window of pixels whose home cell is < 1.5 + D cells away.
        """
        drift = max(float(np.abs(cx - self.cx0).max()) / self.rx,
                    float(np.abs(cy - self.cy0).max()) / self.ry)
        return min(int(math.floor(1.5 + drift + 1e-9)), _MAX_SEARCH_RADIUS)


def _assign(grid: _Grid, cx, cy, radius: int, candidate_ok=None, channels=None, sp_color=None,
            coef: float = 1.0, windowed: bool = True) -> np.ndarray:
    """One modified-SLIC assignment pass over the whole (hi, wi) pixel grid.

    d(p_i, p_s) = ||c_i − c_k(s)|| + coef·||(x_i, y_i) − (x_s, y_s)||, over superpixels whose
    center lies within the 2rx × 2ry window of the pixel. Candidates are the superpixels of the
    grid cells around the pixel's home cell, visited in ascending s with a strict '<', so ties
    resolve to the lowest s. (Offsets falling off the grid are clipped to the border cell,
    which merely revisits a candidate.) With channels=None only the spatial term is used.
    `channels` is the (3, hi, wi) LAB image. Returns the superpixel index per pixel, -1 where
    no candidate qualified. Native (`_native.slic_assign`), bitwise equal to the numpy
    reference in `_reference/gerstner.py`.
    """
    ho, wo = grid.ho, grid.wo
    center_x = cx.reshape(ho, wo)
    if candidate_ok is not None:
        # An infinitely distant center fails the window test and never wins.
        center_x = np.where(candidate_ok.reshape(ho, wo), center_x, np.inf)
    if channels is not None:
        channels = np.ascontiguousarray(channels, dtype=np.float64)
        sp_color = np.ascontiguousarray(sp_color, dtype=np.float64).reshape(ho, wo, 3)
    return _native.slic_assign(grid.home_y, grid.home_x, grid.px, grid.py,
                               np.ascontiguousarray(center_x, dtype=np.float64),
                               np.ascontiguousarray(cy, dtype=np.float64).reshape(ho, wo),
                               int(radius), float(grid.rx), float(grid.ry), float(coef),
                               bool(windowed), channels, sp_color)


def palette_prior(coverage: np.ndarray) -> np.ndarray:
    """P(p_s) over the active superpixels from their mean pixel weight (alpha).

    Uniform (exactly 1/N) when every superpixel is fully opaque, as in the spec; otherwise
    proportional to the coverage, so a superpixel of semi-transparent fringe pixels holds
    less palette mass. Superpixels lying entirely in the transparent region are not active.
    Native; the DEVIATION note is with the code in crates/pixelforge/src/gerstner_run.rs.
    """
    return _native.palette_prior(np.ascontiguousarray(coverage, dtype=np.float64))


def run(lab: np.ndarray, mask: np.ndarray, out_width: int, out_height: int, config,
        weight: np.ndarray | None = None) -> Downscaled:
    if not mask.any():
        raise PixelforgeError("no opaque pixels")
    result = _native.gerstner_run(
        np.ascontiguousarray(lab, dtype=np.float64), np.ascontiguousarray(mask, dtype=bool),
        int(out_width), int(out_height), config.to_dict(),
        None if weight is None else np.ascontiguousarray(weight, dtype=np.float64),
        _tunables())
    return Downscaled(**result)
