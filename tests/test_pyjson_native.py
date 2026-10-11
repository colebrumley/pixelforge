"""`pixelforge::pyjson` writes JSON like `json.dumps` and rounds like Python's `round`."""

import json
import math

import numpy as np
import pytest

from pixelforge import _native

NESTED = {"b": [1, 2.5, None, True, False, "x\"y\\z\nä✓\U0001F600"], "a": {},
          "c": [], "d": {"z": [[]], "y": {"k": -0.0}, "x": 1e-05, "w": 1e16, "v": -3},
          "e": "", "f": [{"i": 0}, [1, [2, [3]]]]}


@pytest.mark.parametrize("indent", [None, 1, 2, 4])
@pytest.mark.parametrize("sort_keys", [False, True])
def test_layouts(indent, sort_keys):
    expected = json.dumps(NESTED, indent=indent, sort_keys=sort_keys)
    assert _native.json_dumps(NESTED, indent=indent, sort_keys=sort_keys) == expected


def test_compact_separators():
    expected = json.dumps(NESTED, sort_keys=True, separators=(",", ":"))
    assert _native.json_dumps(NESTED, sort_keys=True, compact=True) == expected


def test_float_lists_match_repr():
    rng = np.random.default_rng(20261011)
    values = (rng.standard_normal(2000) * 10.0 ** rng.integers(-8, 8, 2000)).tolist()
    values += [0.0, -0.0, 1.0, 1e16, 1e-5, 123456789012345680.0, 5e-324]
    assert _native.json_dumps(values) == json.dumps(values)
    assert _native.json_dumps(values, indent=2) == json.dumps(values, indent=2)


def test_round_matches_python():
    rng = np.random.default_rng(7)
    values = (rng.standard_normal(100_000) * 10.0 ** rng.integers(-9, 6, 100_000)).tolist()
    # Exact binary ties at 6 and 2 decimals: k / 2**j land exactly on ...5.
    ties = [k / 2.0 ** j for j in range(1, 12) for k in range(-40, 41)]
    for x in values + ties + [0.0, -0.0, 0.5, 1.5, 2.5, 0.125, 0.375, 1e-7, 123456.0000005]:
        for n in (0, 2, 4, 6):
            got, want = _native.round_float(x, n), round(x, n)
            assert got == want and math.copysign(1.0, got) == math.copysign(1.0, want), (x, n, got, want)
