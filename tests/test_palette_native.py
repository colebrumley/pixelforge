"""The native palette builders (pixelforge.palette -> _native) against the former numpy
implementation, copied here as the reference. Everything is compared bit for bit, except the
a and b of colours regularize_ramps moves: they come from libm's hypot/atan2/cos/sin (plan
Decision 6), which differ from numpy's platform versions in the last bit for a few percent of
arguments (measured on macOS arm64: at most 1 ulp, 7.1e-14 on the output; with libm's values
substituted into the numpy reference the outputs are bitwise identical)."""

import numpy as np
import pytest

from pixelforge import Config, _native, color, palette
from pixelforge.config import PRESETS

FIXTURES = ("checker_tiles", "circle_alpha", "gradient6", "line_diag", "noisy_gradient",
            "seamless_src", "two_color")
KS = [1, 2, 7, 8, 9, 16, 32, 256]


# ------------------------------------------------------------------------- reference

def ref_box_error(box):
    centered = box - box.mean(axis=0)
    return float((centered * centered).sum())


def ref_split_mask(values):
    ordered = np.sort(values, kind="stable")
    n = len(ordered)
    below = np.arange(1, n)
    total, total_sq = np.cumsum(ordered), np.cumsum(ordered * ordered)
    error = (total_sq[:-1] - total[:-1] ** 2 / below) + (
        (total_sq[-1] - total_sq[:-1]) - (total[-1] - total[:-1]) ** 2 / (n - below))
    error[ordered[1:] == ordered[:-1]] = np.inf
    return values <= ordered[int(np.argmin(error))]


def ref_median_cut(lab_pixels, K):
    pixels = np.ascontiguousarray(lab_pixels, dtype=np.float64).reshape(-1, 3)
    if len(pixels) == 0:
        raise ValueError("median_cut needs at least one pixel")
    boxes = [pixels]
    errors = [ref_box_error(pixels)]
    while len(boxes) < K:
        chosen = None
        for i in sorted(range(len(boxes)), key=lambda i: (-errors[i], i)):
            ranges = boxes[i].max(axis=0) - boxes[i].min(axis=0)
            if ranges.max() > palette.MEDIAN_CUT_MIN_RANGE:
                chosen = (i, int(np.argmax(ranges)))
                break
        if chosen is None:
            break
        i, axis = chosen
        box = boxes[i]
        lower = ref_split_mask(box[:, axis])
        halves = [box[lower], box[~lower]]
        boxes[i:i + 1] = halves
        errors[i:i + 1] = [ref_box_error(h) for h in halves]
    pal = np.array([b.mean(axis=0) for b in boxes])
    order = np.lexsort((pal[:, 2], pal[:, 1], pal[:, 0]))
    return pal[order]


def ref_regularize_ramps(palette_lab):
    pal = np.array(palette_lab, dtype=np.float64).reshape(-1, 3)
    L, a, b = pal[:, 0], pal[:, 1], pal[:, 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    bins = np.floor(hue / palette.RAMP_BIN_DEGREES).astype(np.int64)
    bins[chroma < palette.RAMP_ACHROMATIC_CHROMA] = -1
    new_hue = hue.copy()
    for bin_id in np.unique(bins):
        members = np.nonzero(bins == bin_id)[0]
        offset = L[members] - L[members].mean()
        amount = palette.RAMP_MAX_SHIFT_DEGREES * np.minimum(
            1.0, np.abs(offset) / palette.RAMP_FULL_SHIFT_L)
        target = np.where(offset < 0, palette.RAMP_DARK_HUE, palette.RAMP_LIGHT_HUE)
        to_target = (target - hue[members] + 180.0) % 360.0 - 180.0
        to_target[to_target == -180.0] = 180.0
        new_hue[members] = hue[members] + np.sign(to_target) * np.minimum(amount,
                                                                          np.abs(to_target))
    rad = np.radians(new_hue)
    out = pal.copy()
    moved = new_hue != hue
    out[moved, 1] = chroma[moved] * np.cos(rad[moved])
    out[moved, 2] = chroma[moved] * np.sin(rad[moved])
    return out


# --------------------------------------------------------------------------- helpers

def random_lab(rng, n):
    lab = np.empty((n, 3))
    lab[:, 0] = rng.uniform(0, 100, n)
    lab[:, 1:] = rng.uniform(-90, 90, (n, 2))
    return lab


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        a.view(np.uint64), b.view(np.uint64))


def pixel_sets():
    rng = np.random.default_rng(20261010)
    for n in (1, 2, 3, 7, 8, 9, 127, 128, 129, 1000):
        yield f"random{n}", random_lab(rng, n)
    yield "one_colour", np.tile([60.0, 10.0, -20.0], (500, 1))
    yield "two_colours", np.repeat([[20.0, 0.0, 0.0], [80.0, 5.0, 5.0]], [300, 7], axis=0)
    yield "near_one_colour", np.tile([50.0, 0.0, 0.0], (64, 1)) + rng.uniform(-1, 1, (64, 3))
    dup = np.round(random_lab(rng, 2000) / 10.0) * 10.0        # many duplicates and ties
    yield "quantized", dup
    yield "coarse", np.round(random_lab(rng, 600))
    ties = np.zeros((40, 3))
    ties[:, 0] = np.repeat([10.0, 20.0, 30.0, 40.0], 10)       # equal ranges on every axis
    ties[:, 1] = np.tile(np.linspace(-15.0, 15.0, 10), 4)
    ties[:, 2] = np.tile(np.linspace(15.0, -15.0, 10), 4)
    yield "ties", ties
    yield "neg_zero", np.array([[0.0, -0.0, 0.0], [-0.0, 0.0, 5.0], [10.0, -0.0, -0.0]] * 3)


# ----------------------------------------------------------------------------- tests

@pytest.mark.parametrize("k", KS)
def test_median_cut_synthetic(k):
    for name, pixels in pixel_sets():
        assert same(palette.median_cut(pixels, k), ref_median_cut(pixels, k)), (name, k)


def test_median_cut_fixtures(preprocessed):
    for name in FIXTURES:
        for preset in PRESETS:
            pre, _ = preprocessed(name, preset=preset)
            pixels = pre.lab[pre.mask]
            for k in KS:
                assert same(palette.median_cut(pixels, k), ref_median_cut(pixels, k)), (
                    name, preset, k)


def test_median_cut_empty():
    with pytest.raises(ValueError, match="^median_cut needs at least one pixel$"):
        palette.median_cut(np.zeros((0, 3)), 4)


def ramp_palettes():
    rng = np.random.default_rng(99)
    for k in KS:
        yield f"random{k}", random_lab(rng, k)
        yield f"median_cut{k}", ref_median_cut(random_lab(rng, 400), k)
    # Hues on bin edges, opposite the targets, achromatic, duplicates, one bin.
    hues = np.radians([0.0, 30.0, 90.0, 180.0, 270.0, 359.9, -0.0, 60.0, 89.0, 271.0])
    edge = np.stack([np.linspace(10, 90, 10), 40 * np.cos(hues), 40 * np.sin(hues)], axis=1)
    yield "edges", edge
    yield "achromatic", np.array([[20.0, 1.0, -1.0], [80.0, 0.0, 7.9], [50.0, -0.0, 0.0]])
    yield "duplicates", np.repeat([[30.0, 20.0, 5.0], [70.0, 20.0, 6.0]], 3, axis=0)
    one_bin = np.stack([np.linspace(5, 95, 16), np.full(16, 30.0),
                        np.linspace(1, 15, 16)], axis=1)
    yield "one_bin", one_bin
    yield "tiny_negative_b", np.array([[50.0, 30.0, -1e-300], [60.0, 30.0, 1e-300]])


def ramps_match(got, want, pal):
    """L bitwise, the same colours moved (the rest bitwise), a and b within 1e-12."""
    moved_got = (got != pal).any(axis=1)
    return (got.shape == want.shape and same(got[:, 0], want[:, 0])
            and np.array_equal(moved_got, (want != pal).any(axis=1))
            and same(got[~moved_got], want[~moved_got])
            and np.abs(got - want).max(initial=0.0) <= 1e-12)


def test_regularize_ramps():
    for name, pal in ramp_palettes():
        assert ramps_match(palette.regularize_ramps(pal), ref_regularize_ramps(pal), pal), name


def test_regularize_ramps_fixture_palettes(preprocessed):
    for name in FIXTURES:
        for preset in PRESETS:
            pre, config = preprocessed(name, preset=preset)
            pal = palette.build(pre.lab[pre.mask], config.replace(palette_source="median_cut"))
            want = ref_regularize_ramps(pal)
            assert ramps_match(palette.regularize_ramps(pal), want, pal), name
    for name in palette.bundled_palettes():
        pal = color.rgb8_to_lab(palette.load_palette(name))
        assert ramps_match(palette.regularize_ramps(pal), ref_regularize_ramps(pal), pal), name


@pytest.mark.parametrize("overrides", [
    dict(method="box"),
    dict(method="kopf", palette_size=9),
    dict(palette_source="median_cut", method="gerstner", palette_size=3),
    dict(palette_name="pico8"),
    dict(palette_name="hex:ff0000,00ff00,0000ff", method="gerstner"),
    dict(palette_name="tests/oracle/palettes/three.gpl"),
])
def test_build(overrides):
    config = Config(**overrides)
    pixels = random_lab(np.random.default_rng(1), 300)
    if config.palette_name is not None:
        lines = config.palette_hex_lines()
        want = color.rgb8_to_lab(np.array([color.hex_to_rgb8(h) for h in lines], np.uint8))
    else:
        want = ref_median_cut(pixels, config.palette_size)
    assert same(palette.build(pixels, config), want)


def test_build_mcda_stays_in_python():
    config = Config(method="gerstner", g_max_iters=2)
    with pytest.raises(ValueError, match="mcda"):
        _native.palette_build(np.zeros((4, 3)), config.to_dict())
    assert palette.resolve_source(config) == "mcda"
    assert len(palette.build(random_lab(np.random.default_rng(2), 50), config)) >= 1


def test_hex_text(tmp_path):
    rng = np.random.default_rng(4)
    for rgb8, first in [(rng.integers(0, 256, (16, 3), dtype=np.uint8), False),
                        (rng.integers(0, 256, (5, 3), dtype=np.uint8), True),
                        (np.zeros((0, 3), dtype=np.uint8), True),
                        (np.array([[0, 255, 16]], dtype=np.int64), False)]:
        lines = [color.rgb8_to_hex(c)[1:] for c in rgb8]
        if first:
            lines.insert(0, palette.TRANSPARENT_HEX_LINE)
        palette.write_hex(tmp_path / "p.hex", rgb8, transparent_first=first)
        assert (tmp_path / "p.hex").read_text() == "\n".join(lines) + "\n"
    with pytest.raises(ValueError, match="8-bit"):
        palette.write_hex(tmp_path / "p.hex", np.array([[0, 256, 0]]))
