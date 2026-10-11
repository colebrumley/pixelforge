"""`pixelforge::downscale::neighbor_mean4` is bitwise `pixelforge.downscale.neighbor_mean4`."""

import numpy as np
import pytest

from pixelforge import _native
from pixelforge.downscale import neighbor_mean4


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (2, 3), (7, 9), (64, 64), (33, 17)])
def test_neighbor_mean4_matches(shape):
    rng = np.random.default_rng(sum(shape))
    grid = rng.standard_normal(shape) * 1e8 + rng.integers(-3, 3, shape) * 1e-3
    assert np.array_equal(_native.neighbor_mean4(grid), neighbor_mean4(grid))
