"""The compiled extension `pixelforge._native` builds, imports and passes arrays through."""

import numpy as np
import pytest

import pixelforge
from pixelforge import _native


def test_native_version_matches_package():
    assert _native.native_version() == pixelforge.__version__


@pytest.mark.parametrize("value, expected", [("1", 1), ("7", 7), (" 4 ", 4)])
def test_thread_count_reads_env(monkeypatch, value, expected):
    monkeypatch.setenv("PIXELFORGE_THREADS", value)
    assert _native.thread_count() == expected


@pytest.mark.parametrize("value", ["0", "-2", "junk", ""])
def test_thread_count_falls_back_to_available_parallelism(monkeypatch, value):
    monkeypatch.delenv("PIXELFORGE_THREADS", raising=False)
    default = _native.thread_count()
    assert isinstance(default, int) and default >= 1
    monkeypatch.setenv("PIXELFORGE_THREADS", value)
    assert _native.thread_count() == default


def _sequential_sum(values):
    total = 0.0
    for v in values:
        total += float(v)
    return total


@pytest.mark.parametrize("values", [
    [],
    [1.5],
    [1e16, 1.0, -1e16, 1.0],          # order-sensitive: pairwise or fsum would differ
    [0.1] * 10,
    list(np.random.default_rng(0).standard_normal(1001) * 1e3),
])
def test_sum_f64_is_sequential(values):
    arr = np.asarray(values, dtype=np.float64)
    assert _native.sum_f64(arr) == _sequential_sum(values)


def test_sum_f64_rounds_in_index_order():
    # 1e16 + 1.0 rounds back to 1e16, so the sequential sum is 1.0, not the exact 2.0.
    assert _native.sum_f64(np.array([1e16, 1.0, -1e16, 1.0])) == 1.0


def test_sum_f64_accepts_strided_view():
    base = np.arange(10, dtype=np.float64)
    view = base[::3]
    assert not view.flags.c_contiguous
    assert _native.sum_f64(view) == _sequential_sum(view)


@pytest.mark.parametrize("bad", [
    np.arange(4, dtype=np.float32),
    np.arange(4, dtype=np.int64),
    np.zeros((2, 2), dtype=np.float64),
    [1.0, 2.0],
])
def test_sum_f64_rejects_wrong_type(bad):
    with pytest.raises((TypeError, ValueError)):
        _native.sum_f64(bad)
