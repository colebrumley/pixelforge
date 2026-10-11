"""The Rust compare pieces (crates/pixelforge/src/sheet.rs) agree with pixelforge/cli.py.

`side_by_side`, `parse_sweep`, `_show`, `override_flags` and `preview_scale` are compared
exactly. The proof sheet's labels use an embedded bitmap font in Rust and Pillow's default
font in Python, so its geometry is compared with Python's layout run on the Rust label
sizes, and its pixels everywhere except the label text.
"""

import math
import random
import struct
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from pixelforge import cli
from pixelforge.config import Config
from pixelforge.errors import ConfigError

native = pytest.importorskip("pixelforge._native")


def _rgba(rng, h, w):
    return rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)


@pytest.mark.parametrize("gap", [0, 1, 5])
def test_side_by_side_bitwise(gap):
    rng = np.random.default_rng(gap)
    for n in (1, 2, 3, 5):
        images = [_rgba(rng, int(rng.integers(1, 30)), int(rng.integers(1, 30)))
                  for _ in range(n)]
        want = cli.side_by_side(images, gap)
        got = native.sheet_side_by_side(images, gap)
        assert got.dtype == np.uint8 and got.shape == want.shape
        assert np.array_equal(got, want)


def test_prepass_columns():
    assert native.sheet_prepass_columns() == list(cli.PREPASS_COLUMNS)


def _typed(columns):
    return [[(k, type(v).__name__, repr(v)) for k, v in col.items()] for col in columns]


def _outcome(func, text):
    try:
        return "ok", _typed(func(text))
    except ConfigError as exc:
        return "error", str(exc)


VALUES = {
    "int": ["8,12,16", "1_000", "-0", "+5", "007", "١٢", "9" * 20, "-" + "9" * 30,
            "1" * 4300, "1" * 4301, "0" * 4300 + "1", "1.5", "abc", "none", "None", "1__0",
            "_1", "1_", "0x10", " 3 , 4 ", "²", "-", "+"],
    "float": ["0.5,1,1e-3", "inf", "-Infinity", "nan", "NaN", "1_0.5", ".5", "5.", "1e",
              "abc", "none", "١.٥", "1e400", "-0.0", "0x1p3", "1__0", "4.94e-324"],
    "bool": ["on,off", "TRUE,False", "yes,no,1,0", "maybe", "none", "2", "On , OFF"],
    "str": ["a,b", "none", "None", "Kopf", "x y", "é"],
}


def _sweep_cases():
    cases = []
    for f in fields(Config):
        base = f.type.split("|")[0].strip()
        flag = "--" + f.name.replace("_", "-")
        names = [f.name, flag, f.name.upper(), f" {flag.lower()} "]
        for name in names[:1] if f.name == "method" else names:
            for values in VALUES[base]:
                cases.append(f"{name}={values}")
    cases += ["nope=1", "palette_size", "=1", "palette_size=", "palette_size=1,",
              "palette_size=" + ",".join(["1"] * 12), "palette_size=" + ",".join(["1"] * 13),
              "method=box", "--method=box,kopf", "METHOD=x", "palette-size==1", "",
              "é=1", "palette_size=1=2"]
    return cases


def test_parse_sweep_matches_python():
    cases = _sweep_cases()
    assert len(cases) > 1000
    outcomes = set()
    for text in cases:
        want = _outcome(cli.parse_sweep, text)
        assert _outcome(native.sheet_parse_sweep, text) == want, text
        outcomes.add(want[0])
    assert outcomes == {"ok", "error"}


def test_parse_sweep_method_refused():
    with pytest.raises(ConfigError, match="cannot vary method"):
        native.sheet_parse_sweep("--method=box,kopf")


SHOW_EDGES = [1.0, 0.6, 1e-05, 1e16, 123456.7, 0.30000000000000004, 1234567.0, -0.0, 5e-324,
              0.0, 1e-4, 999999.5, 1234565.0, 1234575.0, 1.5e300, -2.5, 1e15, 1e-6, 0.1,
              2.0 ** 60, math.inf, -math.inf, math.nan, 1.7976931348623157e308]


def _random_floats(n, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        kind = rng.randrange(4)
        if kind == 0:
            x = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
            if math.isnan(x):
                continue
        elif kind == 1:
            x = rng.uniform(-1e6, 1e6)
        elif kind == 2:
            x = round(rng.uniform(-100, 100), rng.randrange(0, 7))
        else:
            x = rng.randrange(-10**7, 10**7) / 10 ** rng.randrange(0, 8)
        out.append(x)
    return out


def test_show_matches_python():
    values = [0, 1, -7, 10**30, True, False, None, "kopf", "a b", "", "é"]
    values += SHOW_EDGES + _random_floats(2000)
    for v in values:
        assert native.sheet_show(v) == cli._show(v), repr(v)


def test_override_flags_matches_python():
    cases = [
        {},
        {"method": "kopf"},
        {"method": "box", "enhance": 0.5, "ink": 0.0},
        {"method": "gerstner", "tileset": True, "seamless": False, "palette": None},
        {"method": "kopf", "g_T_final": 0.30000000000000004, "seed": 7},
        {"palette": "my palette.hex"},
        {"palette": "it's"},
        {"palette": "$HOME/x"},
        {"palette": "café"},
        {"palette": ""},
        {"outline": "#000000", "dither": "bayer4"},
        {"palette": "a,b;c|d&e", "enhance": 1e-05, "saturation_beta": 1e16},
        {"palette": "plain-name_1.2/x:y,z=w+v@%"},
    ]
    for overrides in cases:
        assert native.sheet_override_flags(overrides) == cli.override_flags(overrides), overrides


@pytest.mark.parametrize("sizes, scales", [
    ([(64, 64)], [8]),
    ([(64, 48)] * 12, [8] * 12),
    ([(256, 256)] * 3, [4, 4, 4]),
    ([(1000, 1000), (1000, 1000)], [8, 2]),
    ([(4000, 4000)], [3]),
    ([(10000, 10000)], [1]),
    ([(1, 1)], [100000]),
    ([(0, 0)], [2]),
    ([(37, 91), (12, 5), (300, 2)], [1, 7, 20]),
])
def test_preview_scale_matches_python(monkeypatch, sizes, scales):
    warnings = []
    monkeypatch.setattr(cli, "_warn", warnings.append)
    images = [SimpleNamespace(shape=(h, w, 4)) for h, w in sizes]
    want = cli.preview_scale(images, scales)
    got, warning = native.sheet_preview_scale(sizes, scales)
    assert got == want
    assert ([warning] if warning is not None else []) == warnings


def test_preview_scale_warns():
    _, warning = native.sheet_preview_scale([(4000, 4000)], [3])
    assert warning == ("proof sheet previews scaled 1x instead of 3x to stay within "
                       "32000000 pixels")


def test_label_size():
    assert native.sheet_label_size("abc") == (6 * 3 + 2 * cli.LABEL_PAD, 10 + 2 * cli.LABEL_PAD)
    assert native.sheet_label_size("") == (12, 22)
    assert native.sheet_label_size("café") == (36, 22)


def _python_sheet_with_native_metrics(monkeypatch, cells, rows, cols):
    """cli.proof_sheet measuring labels with the Rust font and drawing no text."""
    from PIL import ImageDraw

    def textbbox(self, xy, text, *args, **kwargs):
        w, h = native.sheet_label_size(text)
        return 0, 0, w - 2 * cli.LABEL_PAD, h - 2 * cli.LABEL_PAD

    monkeypatch.setattr(ImageDraw.ImageDraw, "textbbox", textbbox)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", lambda self, *a, **k: None)
    return cli.proof_sheet(cells, rows, cols)


@pytest.mark.parametrize("seed", range(4))
def test_proof_sheet_geometry(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    rows = ["box", "kopf", "gerstner"][: 1 + seed % 3]
    cols = [f"palette_size={v}" for v in (8, 12, 16, 24, 32)[: 1 + seed]]
    if seed == 3:
        cols = ["enhance=0 ink=0", "x", "café", "~{}|", ""]
    # Distinct solid colours (alpha 255) so each cell can be found on the sheet.
    colours = iter(range(1, 255))
    cells, solid = [], {}
    for r in range(len(rows)):
        row = []
        for c in range(len(cols)):
            h, w = int(rng.integers(1, 70)), int(rng.integers(1, 120))
            v = next(colours)
            solid[(r, c)] = (v, h, w)
            row.append(np.full((h, w, 4), [v, 255 - v, v // 2, 255], dtype=np.uint8))
        cells.append(row)

    got = native.sheet_proof_sheet(cells, rows, cols)
    want = _python_sheet_with_native_metrics(monkeypatch, cells, rows, cols)
    assert got.shape == want.shape and got.dtype == np.uint8

    # Identical except the label text, which is drawn only on the Rust sheet, in LABEL_FG
    # over LABEL_BG, inside the label bars.
    diff = np.any(got != want, axis=2)
    assert np.all(got[diff] == cli.LABEL_FG)
    assert np.all(want[diff] == cli.LABEL_BG)
    pad = cli.LABEL_PAD
    left = max(native.sheet_label_size(s)[0] for s in rows)
    top = max(native.sheet_label_size(s)[1] for s in cols)
    ys, xs = np.nonzero(diff)
    assert np.all((ys < top) | (xs < left))
    if any(s.strip() for s in rows + cols):
        assert diff.any()

    # Each cell sits where Python's layout puts it: the first pixel of its colour.
    widths = [max(native.sheet_label_size(s)[0], *(solid[(r, c)][2] for r in range(len(rows))))
              for c, s in enumerate(cols)]
    heights = [max(native.sheet_label_size(s)[1], *(solid[(r, c)][1] for c in range(len(cols))))
               for r, s in enumerate(rows)]
    assert got.shape == (top + sum(heights) + pad * len(heights),
                         left + sum(widths) + pad * len(widths), 4)
    for (r, c), (v, h, w) in solid.items():
        match = np.all(got == [v, 255 - v, v // 2, 255], axis=2)
        y, x = np.argwhere(match)[0]
        assert (y, x) == (top + pad + sum(heights[:r]) + pad * r,
                          left + pad + sum(widths[:c]) + pad * c)
        assert match.sum() == h * w


def test_proof_sheet_rejects_ragged_cells():
    cell = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        native.sheet_proof_sheet([[cell], [cell, cell]], ["a", "b"], ["x"])
    with pytest.raises(ValueError):
        native.sheet_proof_sheet([], [], [])
