"""The Rust Config (crates/pixelforge/src/config) agrees with pixelforge/config.py.

Runs the oracle corpus and error files (tests/oracle/config_corpus.json,
config_errors.json) through `pixelforge._native` and compares with the recorded strings and
with the live Python `Config`, so the test stays valid if the oracles are re-recorded.
"""

import json
import math
from pathlib import Path

import pytest

from pixelforge import palette
from pixelforge.config import Config
from pixelforge.errors import ConfigError

native = pytest.importorskip("pixelforge._native")

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "tests" / "oracle"
CORPUS = json.loads((ORACLE / "config_corpus.json").read_text())["cases"]
ERRORS = json.loads((ORACLE / "config_errors.json").read_text())["cases"]


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    """Corpus palette paths are relative to the repository root."""
    monkeypatch.chdir(ROOT)


def untag(value):
    """The Python object a JSON oracle value stands for ({"$float": ...}, {"$int": ...})."""
    if isinstance(value, dict):
        if "$float" in value:
            return float(value["$float"])
        if "$int" in value:
            return int(value["$int"])
        raise ValueError(value)
    if isinstance(value, list):
        return [untag(v) for v in value]
    return value


def same(a, b):
    """Equal values of the same type (so 1 != 1.0 != True and floats compare exactly)."""
    return type(a) is type(b) and a == b


def python_config(case):
    config = Config(**case["kwargs"])
    for changes in case.get("replace", []):
        config = config.replace(**changes)
    return config


@pytest.mark.parametrize("case", CORPUS, ids=[json.dumps(c["kwargs"]) for c in CORPUS])
def test_corpus(case):
    py = python_config(case)
    if "replace" in case:
        got = native.config_replace_canonical_json(case["kwargs"], case["replace"])
    else:
        got = native.config_canonical_json(case["kwargs"])
        assert native.config_hash(case["kwargs"]) == case["hash"] == py.hash()
        native_dict = native.config_to_dict(case["kwargs"])
        py_dict = py.to_dict()
        assert list(native_dict) == list(py_dict)
        for name, value in py_dict.items():
            assert same(native_dict[name], value), (name, native_dict[name], value)
        assert native.config_validate_palette(case["kwargs"]) is None
    assert got == case["canonical_json"] == py.canonical_json()
    assert native.config_error(case["kwargs"]) is None
    if "palette" in case:
        name = py.palette_name
        assert native.palette_hex_lines(name) == case["palette"] == py.palette_hex_lines()


@pytest.mark.parametrize("case", ERRORS, ids=[c["python"] for c in ERRORS])
def test_errors(case):
    kwargs = {k: untag(v) for k, v in case["kwargs"].items()}
    with pytest.raises(ConfigError) as info:
        Config(**kwargs)
    assert native.config_error(kwargs) == case["message"] == str(info.value)


def test_invalid_kwargs_raise_config_error():
    with pytest.raises(ConfigError, match="palette_size must be in"):
        native.config_canonical_json({"palette_size": 1})
    with pytest.raises(TypeError, match="keywords must be strings"):
        native.config_error({1: 2})


# Coercions the error file does not cover: numeric strings for float fields (float(str)),
# integral floats out of the int64 range, huge ints for float fields, other containers.
EXTRA = [
    {"saturation_beta": "1.5"}, {"saturation_beta": " 1_2.5e-1 "}, {"g_m": "nan"},
    {"g_m": "-Infinity"}, {"g_m": "1e400"}, {"g_m": "1__0"}, {"g_m": "٣٥"},
    {"palette_size": 1e30}, {"palette_size": float("inf")}, {"palette_size": "16"},
    {"g_m": 2**64}, {"g_m": -2**70}, {"seed": 2**63 - 1}, {"seed": -2**63},
    {"seed": (1,)}, {"g_m": {"a": 1}}, {"method": ["box"]}, {"canvas": 64},
    {"preset": True}, {"outline": "#ABCDEF"}, {"outline": "NONE"}, {"canvas": "99999999999x8"},
    {"palette_name": "PALETTE.GPL"}, {"palette_name": "it's.hex"},
    {"palette_name": "tab\there\x7f ✓.png"}, {"tileset_columns": 0},
    {"preset": "background", "canvas": "64x48", "tile_size": 8},
]


@pytest.mark.parametrize("kwargs", EXTRA, ids=[repr(k) for k in EXTRA])
def test_coercion_parity(kwargs):
    try:
        py = Config(**kwargs)
    except ConfigError as exc:
        assert native.config_error(kwargs) == str(exc)
    else:
        assert native.config_error(kwargs) is None
        assert native.config_canonical_json(kwargs) == py.canonical_json()
        if py.palette_name is None:
            assert native.config_hash(kwargs) == py.hash()
        else:   # no such file: hashing needs the palette, so compare the load error
            with pytest.raises(ConfigError) as info:
                py.validate_palette()
            assert native.config_validate_palette(kwargs) == str(info.value)


@pytest.mark.parametrize("name", palette.bundled_palettes())
def test_bundled_palette_lines(name):
    assert native.palette_hex_lines(name) == palette.parse_hex_lines(name)


@pytest.mark.parametrize("name", [
    "hex:ff0000,00ff00", "hex:#FFAA00,0b0c0d", "hex: ff0000 ;c,//x,#00FF00",
    "tests/oracle/palettes/four.hex", "tests/oracle/palettes/three.gpl",
    "tests/oracle/palettes/pälette ✓.hex",
])
def test_palette_lines(name):
    assert native.palette_hex_lines(name) == palette.parse_hex_lines(name)


PALETTE_FILES = {
    "bad.hex": b"zz\n", "bad.gpl": b"GIMP Palette\n1 2 300\n", "nonutf.hex": b"\xff\xfe",
    "big.hex": b"ff0000\n" * 10000, "empty.hex": b"; nothing\n",
    "super.gpl": "GIMP\n² 1 1\n".encode(), "arabic.gpl": "GIMP\n٣ 1 1\n".encode(),
    "bom.hex": b"\xef\xbb\xbfff0000\n", "crlf.hex": b"ff0000\r\n\x0b00ff00\r",
}


@pytest.mark.parametrize("name", [*PALETTE_FILES, "a-directory.hex", "missing.hex",
                                  "hex:", "hex:ff0000,zz",
                                  "hex:" + ",".join(["000000"] * 257)])
def test_validate_palette_parity(tmp_path, monkeypatch, name):
    for file, data in PALETTE_FILES.items():
        (tmp_path / file).write_bytes(data)
    (tmp_path / "a-directory.hex").mkdir()
    monkeypatch.chdir(tmp_path)
    kwargs = {"palette_name": name}
    try:
        Config(**kwargs).validate_palette()
    except ConfigError as exc:
        assert native.config_validate_palette(kwargs) == str(exc)
        with pytest.raises(ValueError) as info:
            native.palette_hex_lines(name)
        assert str(info.value) == str(exc)
    else:
        assert native.config_validate_palette(kwargs) is None
        assert native.palette_hex_lines(name) == palette.parse_hex_lines(name)
        assert native.config_hash(kwargs) == Config(**kwargs).hash()


def test_replace_parity():
    """Config.replace re-applies a changed preset to the fields the caller did not set."""
    for case in CORPUS:
        if "replace" not in case:
            continue
        py = Config(**case["kwargs"])
        steps = []
        for changes in case["replace"]:
            py = py.replace(**changes)
            steps.append(changes)
            assert native.config_replace_canonical_json(case["kwargs"], steps) == \
                py.canonical_json()
        assert native.config_replace_canonical_json(case["kwargs"], case["replace"][0]) == \
            Config(**case["kwargs"]).replace(**case["replace"][0]).canonical_json()


def test_float_repr_in_canonical_json():
    """Floats in the canonical JSON are Python's repr (checked on the oracle's values)."""
    reprs = json.loads((ORACLE / "float_repr.json").read_text())["repr"]
    for text in reprs[::50]:
        x = abs(float(text))
        if not (math.isfinite(x) and x > 0) or x > 4096:
            continue
        kwargs = {"scale": x}
        assert native.config_canonical_json(kwargs) == Config(**kwargs).canonical_json()
