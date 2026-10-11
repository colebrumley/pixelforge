"""Content-adaptive downscaling (Kopf, Shamir, Peers 2013), REQUIREMENTS.md Section 6.

A constrained EM over one anisotropic Gaussian kernel per output pixel. The whole method,
the per-pixel loops and the per-kernel control flow (initialization, M-step and C-step
algebra, shape constraints, starved re-seeding, convergence, the final refill) with its
documented deviations from the paper, runs natively: crates/pixelforge/src/kopf_run.rs over
the loops in crates/pixelforge/src/kopf.rs. Progress lines go to the `pixelforge.kopf`
logger (DEBUG every iteration, INFO every PROGRESS_EVERY). The numpy implementation of the
loops is kept as a test reference in `pixelforge/downscale/_reference/kopf.py`, and the
Python control flow in tests/test_kopf_run_native.py.
"""

from __future__ import annotations

import numpy as np

from .. import _native
from . import Downscaled
from .box import cell_index  # noqa: F401  (part of this module's interface)

RK_HALF_WIDTH = 2.0          # R_k half-width in output units
SIGMA_INIT = 1e-4
SIGMA_CAP = 1.0
SIGMA_GROWTH = 1.1
COV_INIT = 1.0 / 3.0
SINGULAR_MIN, SINGULAR_MAX = 0.05, 0.1
PROGRESS_EVERY = 10   # iterations between INFO progress lines (DEBUG logs every one)


def to_unit_cube(lab: np.ndarray) -> np.ndarray:
    out = np.empty(lab.shape, dtype=np.float64)
    out[..., 0] = lab[..., 0] / 100.0
    out[..., 1] = (lab[..., 1] + 128.0) / 255.0
    out[..., 2] = (lab[..., 2] + 128.0) / 255.0
    return out


def from_unit_cube(cube: np.ndarray) -> np.ndarray:
    out = np.empty(cube.shape, dtype=np.float64)
    out[..., 0] = cube[..., 0] * 100.0
    out[..., 1] = cube[..., 1] * 255.0 - 128.0
    out[..., 2] = cube[..., 2] * 255.0 - 128.0
    return out


def _inverse_cov(cov: np.ndarray):
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    det = a * d - b * b
    return d / det, -b / det, a / det


def run(lab: np.ndarray, mask: np.ndarray, out_width: int, out_height: int, config,
        weight: np.ndarray | None = None) -> Downscaled:
    out = _native.kopf_run(
        np.ascontiguousarray(lab, dtype=np.float64), np.ascontiguousarray(mask, dtype=bool),
        int(out_width), int(out_height), config.to_dict(),
        None if weight is None else np.ascontiguousarray(weight, dtype=np.float64))
    return Downscaled(**out)
