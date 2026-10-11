"""`_native.box_run` (crates/pixelforge/src/boxfilter.rs) is bitwise the numpy box filter.

The reference is the numpy `box.run` that `pixelforge/downscale/box.py` held before it
delegated to the native code, copied here verbatim.
"""

import numpy as np
import pytest

from pixelforge import Config, PixelforgeError, _native
from pixelforge.downscale import box
from pixelforge.errors import PixelforgeError as _PixelforgeError

FIXTURES = ["checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color"]


# ------------------------------------------------------------------ reference (numpy)

def ref_cell_index(n_in: int, n_out: int) -> np.ndarray:
    return (np.arange(n_in, dtype=np.int64) * n_out) // n_in


def ref_run(lab, mask, out_width, out_height, config, weight=None):
    if not mask.any():
        raise _PixelforgeError("no opaque pixels")
    hi, wi = mask.shape
    cell = (ref_cell_index(hi, out_height)[:, None] * out_width
            + ref_cell_index(wi, out_width)[None, :]).ravel()
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
    reference = colors[0]
    small = np.zeros((n_cells, 3), dtype=np.float64)
    filled = opaque > 0
    for c in range(3):
        offset = colors[:, c] - reference[c]
        sums = np.bincount(cell[flat_mask], weights=offset if w is None else w * offset,
                           minlength=n_cells)
        small[filled, c] = reference[c] + sums[filled] / mass[filled]

    small_mask = (opaque * 2 >= total) & filled
    return small.reshape(out_height, out_width, 3), small_mask.reshape(out_height, out_width)


def assert_identical(out, want_lab, want_mask):
    assert out.small_lab.shape == want_lab.shape
    assert out.small_lab.tobytes() == want_lab.tobytes()
    assert np.array_equal(out.small_mask, want_mask)
    assert out.small_mask.dtype == bool
    assert out.stats == {"iterations": 1}
    assert out.palette_lab is None and out.indices is None and out.mean_lab is None


# ------------------------------------------------------------------ fixtures

@pytest.mark.parametrize("preset", ["sprite", "background"])
@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_matches_reference(preprocessed, name, preset):
    pre, config = preprocessed(name, preset=preset, method="box")
    wo, ho = pre.target_width, pre.target_height
    for weight in (None, pre.weight):
        out = box.run(pre.lab, pre.mask, wo, ho, config, weight=weight)
        assert_identical(out, *ref_run(pre.lab, pre.mask, wo, ho, config, weight))
    # A size that does not divide the input.
    out = box.run(pre.lab, pre.mask, 13, 11, config, weight=pre.weight)
    assert_identical(out, *ref_run(pre.lab, pre.mask, 13, 11, config, pre.weight))


# ------------------------------------------------------------------ random inputs

@pytest.mark.parametrize("seed", range(6))
def test_random_with_transparent_cells(seed):
    rng = np.random.default_rng(seed)
    hi, wi = rng.integers(5, 90, 2)
    ho, wo = rng.integers(1, min(hi, wi) + 3, 2)   # also upscaling on some axes
    lab = np.stack([rng.uniform(0, 100, (hi, wi)), rng.uniform(-128, 127, (hi, wi)),
                    rng.uniform(-128, 127, (hi, wi))], axis=-1)
    mask = rng.random((hi, wi)) < rng.uniform(0.2, 0.9)
    mask[: hi // 3, : wi // 3] = False            # whole transparent cells
    mask[-1, -1] = True
    weight = np.where(mask, rng.uniform(0.01, 1.0, (hi, wi)), 0.0)
    config = Config(method="box")
    for w in (None, weight):
        out = box.run(lab, mask, int(wo), int(ho), config, weight=w)
        assert_identical(out, *ref_run(lab, mask, int(wo), int(ho), config, w))


def test_solid_color_is_exact():
    lab = np.broadcast_to(np.array([41.3, 12.7, -33.1]), (37, 23, 3)).copy()
    out = box.run(lab, np.ones((37, 23), bool), 7, 5, Config(method="box"))
    assert (out.small_lab == lab[0, 0]).all()


def test_all_transparent_raises():
    with pytest.raises(PixelforgeError, match="^no opaque pixels$"):
        box.run(np.zeros((4, 4, 3)), np.zeros((4, 4), bool), 2, 2, Config(method="box"))


def test_cell_index_unchanged():
    for n_in, n_out in ((10, 3), (64, 16), (5, 7), (1000, 37)):
        assert np.array_equal(box.cell_index(n_in, n_out), ref_cell_index(n_in, n_out))


# ------------------------------------------------------------------ argument checks

def test_argument_errors():
    lab, mask, cfg = np.zeros((4, 4, 3)), np.ones((4, 4), bool), Config().to_dict()
    with pytest.raises(ValueError, match="box_run: lab must be a float64"):
        _native.box_run(lab.astype(np.float32), mask, 2, 2, cfg)
    with pytest.raises(ValueError, match="box_run: mask must be a bool"):
        _native.box_run(lab, mask.astype(np.uint8), 2, 2, cfg)
    with pytest.raises(ValueError, match="same height and width"):
        _native.box_run(lab, np.ones((4, 5), bool), 2, 2, cfg)
    with pytest.raises(ValueError, match="same height and width"):
        _native.box_run(lab, mask, 2, 2, cfg, np.ones((3, 4)))
    with pytest.raises(ValueError, match="at least 1x1"):
        _native.box_run(lab, mask, 0, 2, cfg)
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.box_run(np.zeros((4, 8, 3))[:, ::2], mask, 2, 2, cfg)
