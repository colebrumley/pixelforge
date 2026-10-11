"""The Rust Config field table (crates/pixelforge/src/config.rs) matches pixelforge/config.py.

The Rust table is the single source of truth the Rust Config, JSON and CLI are built from;
while the Python package exists, any drift between the two tables fails here.
"""

import dataclasses

import pytest

from pixelforge import config as C

native = pytest.importorskip("pixelforge._native")

FIELDS = native.config_fields()
PY_FIELDS = dataclasses.fields(C.Config)


def same(a, b):
    """Equal values of the same type (so 1 != 1.0 != True and floats compare exactly)."""
    return type(a) is type(b) and a == b


def test_field_names_in_order():
    assert [f["name"] for f in FIELDS] == [f.name for f in PY_FIELDS]


@pytest.mark.parametrize("spec, field", list(zip(FIELDS, PY_FIELDS)),
                         ids=[f.name for f in PY_FIELDS])
def test_field(spec, field):
    assert set(spec) == {"name", "type", "default", "help", "choices", "upper_bound"}
    assert spec["name"] == field.name
    assert spec["type"] == field.type
    assert same(spec["default"], field.default), (spec["default"], field.default)
    assert spec["help"] == C.FIELD_HELP[field.name]
    choices = C.FIELD_CHOICES.get(field.name)
    assert spec["choices"] == (None if choices is None else list(choices))
    bound = C.UPPER_BOUNDS.get(field.name)
    assert same(spec["upper_bound"], bound), (spec["upper_bound"], bound)


def test_no_extra_help_choices_or_bounds():
    names = {f.name for f in PY_FIELDS}
    assert set(C.FIELD_HELP) == names
    assert set(C.FIELD_CHOICES) <= names
    assert set(C.UPPER_BOUNDS) <= names


def test_choice_tuples():
    by_name = {f["name"]: f for f in FIELDS}
    for name, values in [("method", C.METHODS), ("denoise", C.DENOISERS),
                         ("dither", C.DITHERS), ("palette_source", C.PALETTE_SOURCES),
                         ("transparent_index", C.TRANSPARENT_INDICES), ("fit", C.FITS),
                         ("preset", tuple(C.PRESETS))]:
        assert by_name[name]["choices"] == list(values)


def test_presets():
    table = native.config_presets()
    assert set(table) == {"presets", "longest_edge"}
    presets = table["presets"]
    assert list(presets) == list(C.PRESETS)
    for name, values in C.PRESETS.items():
        assert list(presets[name]) == list(values), name
        for key, value in values.items():
            assert same(presets[name][key], value), (name, key)
    assert list(table["longest_edge"].items()) == list(C.PRESET_LONGEST_EDGE.items())
