"""`pixelforge::npsum::pairwise_sum` reproduces `np.sum` bit for bit (crates/pixelforge/src/npsum.rs)."""

import numpy as np
import pytest

from pixelforge import _native

SIZES = list(range(0, 300)) + [511, 512, 513, 1000, 1024, 2047, 4096, 10_000, 65_536, 65_537]


@pytest.mark.parametrize("scale", [1.0, 1e8])
def test_matches_np_sum(scale):
    rng = np.random.default_rng(20261011)
    for n in SIZES:
        a = rng.standard_normal(n) * scale + rng.integers(-3, 3, n) * 1e-3
        assert _native.np_pairwise_sum(a) == np.sum(a), n


def test_order_sensitive_cases():
    assert _native.np_pairwise_sum(np.array([1e16, 1.0, -1e16, 1.0])) == np.sum([1e16, 1.0, -1e16, 1.0])
    a = np.zeros(16)
    a[0], a[1], a[8], a[9] = 1e16, 1.0, -1e16, 1.0
    assert _native.np_pairwise_sum(a) == np.sum(a) == 2.0
    assert _native.np_pairwise_sum(np.zeros(0)) == 0.0


def test_strided_view():
    base = np.arange(600, dtype=np.float64) * 0.37
    view = base[::3]
    assert not view.flags.c_contiguous
    assert _native.np_pairwise_sum(view) == np.sum(view)
