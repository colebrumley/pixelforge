"""Record tests/oracle/: Python reference outputs the Rust port is tested against.

Everything is deterministic (fixed inputs, seeded numpy RNG), so a re-run reproduces the
directory byte for byte; tests/test_oracle.py checks that while Python exists. The files
are plain JSON (no .npy) so Rust tests can load them with serde_json. See
tests/oracle/README.md for what each file holds.

Usage: python scripts/record_oracles.py [OUTDIR]   (default: tests/oracle)
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DIR = ROOT / "tests" / "oracle"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
SEED = 20261010

# Palette files the config corpus refers to, by path relative to the repository root.
PALETTE_REL = "tests/oracle/palettes"
PALETTE_FILES = {
    "four.hex": "; four colours, mixed case and comments\n#FF0000\n00ff00\n\n0000Ff ; blue\n"
                "// a comment line\nffffff\n",
    "three.gpl": "GIMP Palette\nName: three\nColumns: 3\n#\n255   0   0\tRed\n"
                 "  0 128 255\tAzure\n 12  34  56\n",
    "pälette ✓.hex": "112233\n445566\n#778899\n",
}

LIBRARIES = ("numpy", "scipy", "scikit-image", "Pillow")


# ------------------------------------------------------------------------------ JSON writer

def _plain(obj):
    """numpy arrays and scalars → Python lists, floats, ints, bools."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _scalar(value) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite float {value!r} has no JSON form")
    if not (value is None or isinstance(value, (bool, int, float, str))):
        raise TypeError(f"not a JSON scalar: {type(value).__name__}")
    return json.dumps(value)


def _is_scalar(value) -> bool:
    return not isinstance(value, (dict, list))


def _format(obj, level: int) -> str:
    """json.dumps(sort_keys=True, indent=1), except that a list or dict of scalars stays on
    one line.

    Every scalar goes through json.dumps, so floats are written as Python's repr and parse
    back bit-exactly.
    """
    pad, inner = " " * level, " " * (level + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        if all(_is_scalar(v) for v in obj.values()):
            return "{" + ", ".join(f"{json.dumps(k)}: {_scalar(obj[k])}" for k in sorted(obj)) + "}"
        items = [f"{inner}{json.dumps(k)}: {_format(obj[k], level + 1)}" for k in sorted(obj)]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(_is_scalar(v) for v in obj):
            return "[" + ",".join(_scalar(v) for v in obj) + "]"
        items = [inner + _format(v, level + 1) for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return _scalar(obj)


def write_json(path: Path, obj) -> None:
    text = _format(_plain(obj), 0) + "\n"
    assert json.loads(text) == _plain(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("ascii"))


def mask_rows(mask: np.ndarray) -> list[str]:
    return ["".join("1" if v else "0" for v in row) for row in np.asarray(mask, dtype=bool)]


def load_rgba(name: str) -> np.ndarray:
    from pixelforge import io

    loaded = io.load(FIXTURE_DIR / f"{name}.png")
    rgb8 = np.round(loaded.rgb * 255.0).astype(np.uint8)
    return np.dstack([rgb8, loaded.alpha])


# ------------------------------------------------------------------------- (a) config corpus

# One or more non-default values per Config field (preset defaults: sprite).
FIELD_VALUES = {
    "preset": ["background"],
    "out_width": [100],
    "out_height": [72],
    "scale": [3.5],
    "canvas": ["64x48"],
    "remove_bg": [True],
    "key_bg": [False],
    "key_bg_tolerance": [0.15],
    "key_bg_fringe": [0],
    "crop_to_alpha": [False],
    "alpha_threshold": [1],
    "denoise": ["median", "none"],
    "denoise_sigma_color": [0.2],
    "denoise_sigma_spatial": [4.5],
    "prereduce_max_ratio": [0],
    "enhance": [0.75],
    "enhance_black": [0.05],
    "enhance_white": [0.9],
    "enhance_radius": [3.25],
    "ink": [0.5],
    "method": ["box", "kopf"],
    "kopf_max_iters": [20],
    "kopf_tol": [0.0005],
    "palette_size": [8],
    "palette_name": ["gameboy", "genesis", "nes", "pico8", "snes15", "hex:ff0000,00ff00",
                     "hex:#FFAA00,0b0c0d", f"{PALETTE_REL}/four.hex",
                     f"{PALETTE_REL}/three.gpl", f"{PALETTE_REL}/pälette ✓.hex"],
    "palette_source": ["median_cut", "mcda"],
    "palette_ramps": [False],
    "saturation_beta": [1.25],
    "g_m": [30.0],
    "g_alpha": [0.5],
    "g_T_final": [2.0],
    "g_eps_palette": [0.5],
    "g_eps_cluster": [0.1],
    "g_laplacian": [0.25],
    "g_bilateral_sigma_color": [7.5],
    "g_bilateral_sigma_spatial": [2.5],
    "g_max_iters": [500],
    "dither": ["auto", "bayer4", "bayer8"],
    "dither_strength": [0.25],
    "dither_variance_threshold": [6.0],
    "remove_orphans": [False],
    "orphan_min_region": [3],
    "orphan_max_delta": [12.5],
    "fix_jaggies": [False],
    "outline": ["none", "#AaBbCc"],
    "outline_darken": [0.3],
    "tile_size": [8],
    "tileset": [True],
    "fit": ["stretch", "crop"],
    "tile_dedupe_tolerance": [0.0],
    "seamless": [True],
    "tileset_columns": [4],
    "transparent_index": ["first"],
    "seed": [42],
    "scale_preview": [4],
}

# Floats that exercise Python's repr rules (shortest round trip, exponent form below 1e-4
# and from 1e16, "1.0" not "1"), and int <-> float coercion.
FLOAT_CASES = [
    dict(kopf_tol=1e-05), dict(kopf_tol=0.1), dict(kopf_tol=5e-324),
    dict(g_T_final=1e16), dict(g_T_final=123456789.123456789), dict(g_T_final=1e22),
    dict(g_T_final=1.5e300), dict(g_T_final=9999999999999998.0),
    dict(saturation_beta=1.0), dict(saturation_beta=4.999999999999999),
    dict(scale=0.0001), dict(scale=0.00011), dict(scale=4096),
    dict(dither_strength=0.30000000000000004),
    dict(enhance_black=0.5, enhance_white=0.75),
    dict(seed=2**63 - 1), dict(seed=0),
    # Coercion: ints given for float fields, integral floats for int fields.
    dict(saturation_beta=1), dict(g_m=9007199254740993), dict(palette_size=16.0),
    dict(out_width=64.0),
]

COMBINED_CASES = [
    dict(preset="background", method="box", out_width=32, out_height=32, denoise="none",
         key_bg=False),
    dict(preset="sprite", canvas="48x32", outline="#102030", palette_name="pico8",
         dither="bayer8", seed=7),
    dict(preset="background", tile_size=8, canvas="64x48", tileset_columns=3),
]

REPLACE_CASES = [
    (dict(preset="sprite", palette_size=8), [dict(preset="background")]),
    (dict(preset="sprite"), [dict(preset="background")]),
    (dict(preset="background"), [dict(preset="sprite")]),
    ({}, [dict(method="box"), dict(preset="background")]),
    (dict(preset="background", palette_size=12, dither="bayer8", seed=3), [dict(method="box")]),
    (dict(preset="sprite", outline="#ABCDEF"), [dict(preset="background", tileset=False)]),
]


def config_case(kwargs: dict, replace: list | None = None) -> dict:
    from pixelforge import Config

    cfg = Config(**kwargs)
    for changes in replace or []:
        cfg = cfg.replace(**changes)
    case = {"kwargs": kwargs, "canonical_json": cfg.canonical_json(), "hash": cfg.hash(),
            "to_dict": cfg.to_dict()}
    if replace is not None:
        case["replace"] = replace
    if cfg.palette_name is not None:
        case["palette_sha256"] = cfg.palette_sha256()
        case["palette"] = cfg.palette_hex_lines()
    return case


def record_config_corpus() -> dict:
    from pixelforge import Config

    names = Config.field_names()
    assert list(FIELD_VALUES) == names, "FIELD_VALUES must list every Config field in order"
    cases = [config_case({}), config_case({"preset": "sprite"})]
    for name, values in FIELD_VALUES.items():
        for value in values:
            default = getattr(Config(), name)
            assert value != default or type(value) is not type(default), (name, value)
            cases.append(config_case({name: value}))
    cases += [config_case(kwargs) for kwargs in FLOAT_CASES + COMBINED_CASES]
    cases += [config_case(kwargs, replace) for kwargs, replace in REPLACE_CASES]
    return {"cases": cases}


# Values JSON cannot hold: tagged as {"$float": "nan"|"inf"|"-inf"} or {"$int": "<decimal>"}.
def _tag(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": repr(value)}
    if isinstance(value, int) and not isinstance(value, bool) and not -2**63 <= value < 2**64:
        return {"$int": str(value)}
    return value


ERROR_CASES = [
    # Config.__init__
    dict(nonsense=1), dict(zeta=1, alpha=2, method="box"), dict(preset="tiles"),
    dict(preset=None),
    # _coerce
    dict(method=None), dict(palette_size=None), dict(key_bg=None),
    dict(palette_size=True), dict(palette_size=1.5), dict(palette_size="16"),
    dict(key_bg=1), dict(key_bg="yes"), dict(scale=True), dict(saturation_beta="abc"),
    dict(saturation_beta=[1.0]), dict(saturation_beta=10**400), dict(method=1),
    dict(outline=None), dict(seed=2**63), dict(seed=2**64), dict(seed=-2**63 - 1),
    dict(out_width=2**63), dict(kopf_tol=math.nan), dict(g_m=math.inf),
    dict(orphan_max_delta=-math.inf), dict(scale=math.nan),
    # Config._validate, in its order
    dict(palette_name="nonexistent"), dict(palette_name="palette.png"),
    dict(out_width=7), dict(out_height=4), dict(scale=0.0), dict(scale=-1.0),
    dict(canvas="64*48"), dict(canvas="7x64"), dict(canvas="64x4097"), dict(canvas=""),
    dict(canvas="64x48 "), dict(scale=2.0, out_width=64), dict(canvas="64x48", out_height=64),
    dict(palette_size=1), dict(palette_size=257),
    dict(method="nearest"), dict(denoise="gaussian"), dict(dither="floyd"),
    dict(palette_source="kmeans"), dict(transparent_index="middle"), dict(fit="cover"),
    dict(outline="red"), dict(outline="#abc"), dict(outline="#gggggg"),
    dict(g_alpha=0.0), dict(g_alpha=1.0),
    dict(alpha_threshold=-1), dict(alpha_threshold=256),
    dict(dither_strength=-0.1), dict(dither_strength=1.5),
    dict(key_bg_tolerance=-0.01), dict(key_bg_tolerance=1.01),
    dict(orphan_max_delta=-1.0),
    dict(key_bg_fringe=-1), dict(key_bg_fringe=9),
    dict(enhance_black=-0.5), dict(enhance_white=1.5),
    dict(enhance_black=0.5, enhance_white=0.5), dict(enhance_black=0.8, enhance_white=0.2),
    dict(prereduce_max_ratio=-1), dict(outline_darken=1.5), dict(outline_darken=-0.5),
    dict(kopf_max_iters=0), dict(g_max_iters=0), dict(tile_size=0), dict(scale_preview=0),
    dict(orphan_min_region=0),
    dict(g_T_final=0.0), dict(g_m=-1.0), dict(kopf_tol=0.0), dict(saturation_beta=0.0),
    dict(denoise_sigma_color=0.0), dict(denoise_sigma_spatial=0.0),
    dict(g_bilateral_sigma_color=0.0), dict(g_bilateral_sigma_spatial=-0.5),
    dict(enhance_radius=0.0),
    dict(dither_variance_threshold=-1.0), dict(tile_dedupe_tolerance=-0.5),
    dict(enhance=-1.0), dict(ink=-0.1),
    dict(tileset_columns=257), dict(tileset_columns=-1), dict(seed=-1),
    dict(out_width=4097), dict(out_height=4097), dict(scale=4096.5), dict(scale_preview=65),
    dict(denoise_sigma_spatial=16.5), dict(kopf_max_iters=1001), dict(g_max_iters=10001),
    dict(saturation_beta=5.000000000000001), dict(tile_size=513), dict(enhance=4.25),
    dict(enhance_radius=17.0), dict(ink=1.5),
    dict(preset="background", out_width=40, out_height=32),
    dict(preset="background", out_width=32, out_height=40),
    dict(preset="background", canvas="40x32"),
]


def record_config_errors() -> dict:
    from pixelforge import Config, ConfigError

    cases = []
    for kwargs in ERROR_CASES:
        try:
            Config(**kwargs)
        except ConfigError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"Config(**{kwargs!r}) did not raise")
        call = "Config(" + ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) + ")"
        cases.append({"kwargs": {k: _tag(v) for k, v in kwargs.items()}, "python": call,
                      "message": message})
    return {"cases": cases}


# --------------------------------------------------------------------------------- (b) LAB

def _skimage_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    from skimage import color as skcolor

    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    return skcolor.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(rgb.shape)


def _skimage_lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    import warnings

    from skimage import color as skcolor

    lab = np.ascontiguousarray(lab, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # out-of-gamut colors are clipped below
        rgb = skcolor.lab2rgb(lab.reshape(-1, 1, 3)).reshape(lab.shape)
    return np.clip(rgb, 0.0, 1.0)


def record_lab() -> dict[str, dict]:
    # scikit-image itself (the pre-Phase-2 pixelforge.color), not pixelforge.color: these
    # files are the reference the native conversion is tested against.
    axis = np.linspace(0.0, 1.0, 10)
    rgb = np.array([[r, g, b] for r in axis for g in axis for b in axis])
    lattice = {"axis": axis, "rgb": rgb, "lab": _skimage_rgb_to_lab(rgb)}

    lab = np.array([[L, a, b] for L in np.linspace(0.0, 100.0, 8)
                    for a in np.linspace(-128.0, 127.0, 5) for b in np.linspace(-128.0, 127.0, 5)])
    lab_rgb = _skimage_lab_to_rgb(lab)
    roundtrip = {"lab": lab, "lab_to_rgb": lab_rgb,
                 "lab_to_rgb8": np.round(np.clip(lab_rgb * 255.0, 0.0, 255.0)).astype(np.uint8)}

    greys = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    corners = np.array([[r, g, b] for r in (0, 255) for g in (0, 255) for b in (0, 255)],
                       dtype=np.uint8)
    rgb8 = np.concatenate([greys, corners])
    rgb8_lab = {"rgb8": rgb8, "lab": _skimage_rgb_to_lab(rgb8.astype(np.float64) / 255.0)}
    return {"lab_lattice.json": lattice, "lab_roundtrip.json": roundtrip,
            "rgb8_lab.json": rgb8_lab}


# ---------------------------------------------------------------------------------- (c) EDT

def record_edt() -> dict:
    from scipy import ndimage

    rng = np.random.default_rng(SEED)
    masks = []
    for density in (0.1, 0.3, 0.5, 0.7, 0.9):
        masks.append((f"random_48x48_density_{density}", rng.random((48, 48)) < density))
    alpha = load_rgba("circle_alpha")[..., 3]
    masks.append(("circle_alpha_rows20_84_cols20_84", alpha[20:84, 20:84] >= 128))
    single = np.zeros((48, 48), dtype=bool)
    single[17, 30] = True
    masks.append(("single_opaque_48x48_at_17_30", single))
    masks.append(("all_opaque_16x16", np.ones((16, 16), dtype=bool)))
    cases = []
    for name, mask in masks:
        dist, ind = ndimage.distance_transform_edt(~mask, return_distances=True,
                                                   return_indices=True)
        cases.append({"name": name, "mask": mask_rows(mask), "distances": dist,
                      "indices_row": ind[0], "indices_col": ind[1]})
    return {"call": "scipy.ndimage.distance_transform_edt(~mask, return_distances=True, "
                    "return_indices=True)", "cases": cases}


# ---------------------------------------------------------------------------- (d) bilateral

def record_bilateral() -> dict:
    from pixelforge import Config, preprocess

    # The sigma_spatial 0.5 cases (5 × 5 window) use 32 × 32 crops to stay within the size
    # budget; their top-left corner matches the 48 × 48 crops.
    crops = [("noisy_gradient", 100, 100, 48, 0.08, 2.0),
             ("noisy_gradient", 100, 100, 32, 0.08, 0.5),
             ("circle_alpha", 24, 24, 48, 0.08, 2.0),
             ("circle_alpha", 24, 24, 32, 0.08, 0.5),
             ("noisy_gradient", 0, 0, 24, 0.5, 16.0)]
    cases = []
    for name, top, left, size, sigma_color, sigma_spatial in crops:
        rgba = load_rgba(name)[top:top + size, left:left + size]
        mask = rgba[..., 3] >= 128
        rgb = rgba[..., :3].astype(np.float64) / 255.0
        filled = preprocess._fill_transparent(rgb, mask)
        cfg = Config(denoise="bilateral", denoise_sigma_color=sigma_color,
                     denoise_sigma_spatial=sigma_spatial)
        out = preprocess.denoise(filled, cfg)
        filled8 = np.round(filled * 255.0).astype(np.uint8)
        assert np.array_equal(filled8.astype(np.float64) / 255.0, filled)
        cases.append({"fixture": name, "top": top, "left": left, "size": size,
                      "sigma_color": sigma_color, "sigma_spatial": sigma_spatial,
                      "mask": mask_rows(mask), "filled_rgb8": filled8.reshape(size, -1),
                      "output": out.reshape(size, -1)})
    return {"call": "skimage.restoration.denoise_bilateral(filled_rgb8 / 255.0, "
                    "sigma_color=..., sigma_spatial=..., channel_axis=-1, mode='edge') "
                    "clipped to [0, 1]", "cases": cases}


# ------------------------------------------------------------------------ (e) ndimage, stats

def record_ndimage() -> dict:
    from scipy import ndimage

    rng = np.random.default_rng(SEED + 1)
    arrays = {"random_17x23": rng.random((17, 23)),
              "ramp_40": np.linspace(-1.0, 2.0, 40),
              "constant_6x9": np.full((6, 9), 0.3)}
    filters = []
    for name, arr in arrays.items():
        axes = (0, 1) if arr.ndim == 2 else (0,)
        for axis in axes:
            for mode in ("nearest", "wrap"):
                for sigma in (1.5, 3.9):
                    filters.append({"func": "gaussian_filter1d", "input": name,
                                    "params": {"sigma": sigma, "axis": axis, "mode": mode},
                                    "output": ndimage.gaussian_filter1d(arr, sigma, axis=axis,
                                                                        mode=mode)})
                for size in (3, 5, 9, 10):
                    filters.append({"func": "uniform_filter1d", "input": name,
                                    "params": {"size": size, "axis": axis, "mode": mode},
                                    "output": ndimage.uniform_filter1d(arr, size, axis=axis,
                                                                       mode=mode)})
        filters.append({"func": "uniform_filter", "input": name,
                        "params": {"size": 3, "mode": "nearest"},
                        "output": ndimage.uniform_filter(arr, size=3, mode="nearest")})
        filters.append({"func": "median_filter", "input": name,
                        "params": {"size": 3, "mode": "nearest"},
                        "output": ndimage.median_filter(arr, size=3, mode="nearest")})

    samples = {"odd_7": np.array([3.0, -1.0, 2.5, 2.5, 10.0, 0.0, 7.25]),
               "even_8_duplicates": np.array([1.0, 1.0, 1.0, 2.0, 2.0, 5.0, 9.5, 9.5]),
               "single_1": np.array([0.42]),
               "two_2": np.array([0.1, 0.2]),
               "random_101": rng.random(101),
               "random_17x23_flat": arrays["random_17x23"].ravel()}
    percentiles = [{"input": name, "q": [1.0, 99.0, 50.0],
                    "output": np.percentile(values, (1.0, 99.0, 50.0))}
                   for name, values in samples.items()]
    medians = [{"input": name, "axis": None, "output": float(np.median(values))}
               for name, values in samples.items()]
    border = rng.random((10, 3))
    samples["border_10x3"] = border
    medians.append({"input": "border_10x3", "axis": 0, "output": np.median(border, axis=0)})
    return {"arrays": arrays, "filters": filters, "samples": samples,
            "percentile": percentiles, "median": medians}


# ---------------------------------------------------------------------------- (f) float repr

def record_float_repr() -> dict:
    rng = np.random.default_rng(SEED + 2)
    values: list[float] = [
        0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.1 + 0.2, 0.3, 1 / 3, 2 / 3, 0.5, 1.5, 100.0,
        5e-324, -5e-324, 1e-323, 2.2250738585072014e-308, 2.225073858507201e-308,
        1.7976931348623157e308, -1.7976931348623157e308, 1e16, 1e15, 9999999999999998.0,
        1e16 + 2, 123456789012345680.0, 2.0**53, 2.0**53 + 2, 2.0**63, 2.0**64,
        0.0001, 0.00011, 9.999999999999999e-05, 1e-05, 1e22, 1e23, 5e-05, 123456.789,
        4.35, 0.07, 1.1, 2.675, 1e-07, 3.14159, math.pi, math.e, math.sqrt(2),
    ]
    values += [10.0**e for e in range(-15, 23)]
    values += [float(f"1e{e}") for e in range(-320, 309)]
    values += [-float(f"1e{e}") for e in range(-320, 309, 7)]
    values += [float(i) for i in (2, 3, 7, 10, 12345, 99999, 1000000, 2**31, 2**32 - 1)]
    # Short decimal strings (the shortest round trip is the input string).
    for _ in range(2500):
        digits = int(rng.integers(1, 18))
        mantissa = int(rng.integers(1, 10**digits))
        exponent = int(rng.integers(-325, 309))
        sign = "-" if rng.random() < 0.3 else ""
        x = float(f"{sign}{mantissa}e{exponent}")
        if math.isfinite(x):
            values.append(x)
    # Random mantissas at random binary exponents, and random bit patterns.
    for _ in range(2500):
        x = math.ldexp(float(rng.random()) + 0.5, int(rng.integers(-1074, 1024)))
        if math.isfinite(x):
            values.append(-x if rng.random() < 0.3 else x)
    seen: set[str] = set()
    unique = []
    for v in values:
        r = repr(v)
        if r not in seen:
            seen.add(r)
            unique.append(v)
    while len(unique) < 10_000:
        x = float(rng.integers(0, 2**64, dtype=np.uint64).view(np.float64))
        r = repr(x)
        if math.isfinite(x) and r not in seen:
            seen.add(r)
            unique.append(x)
    unique = unique[:10_000]
    reprs = [repr(v) for v in unique]
    assert all(float(r) == v or (v != v) for r, v in zip(reprs, unique))
    assert [json.dumps(v) for v in unique] == reprs
    assert json.dumps(unique) == "[" + ", ".join(reprs) + "]"
    return {"repr": reprs, "json_dumps_equals_repr": True,
            "canonical_json_of_list": hashlib.sha256(
                json.dumps(unique, separators=(",", ":")).encode()).hexdigest()}


# ---------------------------------------------------------------------- (g) pipeline outputs

OUTPUT_RUNS = [
    ("two_color_sprite", "two_color",
     dict(preset="sprite", method="box", out_width=16, out_height=16, denoise="none",
          key_bg=False)),
    ("checker_tiles_background", "checker_tiles",
     dict(preset="background", method="box", out_width=32, out_height=32, denoise="none",
          key_bg=False)),
]
TEXT_SUFFIXES = ("_palette.json", "_palette.hex", "_meta.json", "_tilemap.json", ".tmj",
                 "_tilemap.csv")


def decode_png(path: Path, full: bool = True) -> dict:
    from PIL import Image

    from pixelforge import io

    with Image.open(path) as im:
        im.load()
        info = {"mode": im.mode, "size": list(im.size), "text": io.read_png_text(path),
                "transparency": im.info.get("transparency")}
        data = np.asarray(im)
        palette = im.getpalette()
    if palette is not None:
        info["palette"] = palette
    info["pixels_sha256"] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    if full:
        info["pixels"] = data
    return info


def record_outputs(out_dir: Path) -> None:
    from pixelforge import Config, run

    target = out_dir / "outputs_small"
    with tempfile.TemporaryDirectory() as tmp:
        for prefix, fixture, kwargs in OUTPUT_RUNS:
            result = run(FIXTURE_DIR / f"{fixture}.png", Config(**kwargs))
            paths = result.save(Path(tmp) / prefix)
            for name, path in paths.items():
                path = Path(path)
                if path.suffix == ".png":
                    decoded = decode_png(path, full=name != "preview")
                    write_json(target / f"{path.name}.json", decoded)
                    continue
                assert path.name.endswith(TEXT_SUFFIXES), path
                if name == "meta":
                    meta = json.loads(path.read_text())
                    del meta["environment"]
                    del meta["stats"]["timings"]
                    text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
                    (target / path.name).write_text(text)
                else:
                    shutil.copyfile(path, target / path.name)
            write_json(target / f"{prefix}_config.json",
                       {"fixture": f"tests/fixtures/{fixture}.png", "kwargs": kwargs})


README = """\
# Oracle fixtures

Reference outputs recorded from the Python implementation (numpy, scipy, scikit-image,
Pillow) while it still exists, for the Rust port to be tested against after the Python code
is gone (docs/rust-migration-plan.md, Sections 4.5 and 5, Phase 1).

Generated by `scripts/record_oracles.py`; do not edit by hand. To regenerate (only when a
recorded behaviour changes on purpose; review the diff):

    PYTHONPATH=. uv run --no-sync python scripts/record_oracles.py

The recorder is deterministic (fixed inputs, `numpy.random.default_rng` with fixed seeds).
`tests/test_oracle.py` re-runs it into a temporary directory, requires every file to be
byte-identical to the committed one, and keeps the directory under 2 MB.

## Format

All data files are JSON, readable with serde_json alone. Every float is written as Python's
`repr` (shortest round-trip digits), so parsing it as f64 gives back the recorded value bit
for bit. The layout is `json.dumps(sort_keys=True, indent=1)`, except that a list or object
whose items are all scalars is written on one line. Conventions:

- Arrays are nested lists in row-major order. Images of shape (H, W, 3) are stored as H
  rows of W*3 values (r, g, b interleaved), so pixel (y, x) channel c is `rows[y][3*x + c]`.
- Boolean masks are lists of row strings of `0`/`1` (`1` = opaque / true).
- Values JSON cannot represent (only in `config_errors.json` kwargs) are tagged:
  `{"$float": "nan" | "inf" | "-inf"}`, and `{"$int": "<decimal>"}` for integers outside
  [-2^63, 2^64).

## Files

- `versions.json`: the numpy, scipy, scikit-image and Pillow versions that produced the
  files, and `pixelforge.__version__` (part of every `Config.hash()`).
- `palettes/`: `four.hex`, `three.gpl` and `pälette ✓.hex`, palette files the config corpus
  refers to by repo-root-relative path (`tests/oracle/palettes/...`). The non-ASCII name
  exercises `ensure_ascii` escaping in the canonical JSON.
- `config_corpus.json`: `cases` of `{kwargs, canonical_json, hash, to_dict}` for
  `Config(**kwargs)`: `Config()`, both presets, every field at one or more non-default
  values (every bundled palette, inline `hex:` palettes, the three palette files), float
  repr stress values (1e-05, 5e-324, 1e16, 123456789.123456789, 4.999999999999999,
  0.30000000000000004, ...), int/float coercions, seed 0 and 2^63-1, and a few combined
  configs. A case with `replace` (a list of keyword objects) is
  `Config(**kwargs).replace(**r1).replace(**r2)...`. Cases with a `palette_name` also hold
  `palette_sha256` and `palette` (the resolved `rrggbb` lines). `hash` is the SHA-256 of
  `canonical_json + "\\n" + version`, plus `"\\npalette:" + palette_sha256` with a palette.
  Palette paths resolve from the repository root.
- `config_errors.json`: `cases` of `{kwargs, python, message}`: invalid keyword arguments,
  the call as Python source, and the exact `str(ConfigError)`. Covers unknown fields, bad
  presets, None for non-optional fields, wrong types (bool for int, 1.5 for int, str, list),
  integers beyond 64 bits, non-finite floats, and every bound and choice in
  `Config._validate`, in its order. Unreachable and so not recorded: `_coerce`'s
  "unsupported field type" (no field has another type) and the `seed` entry of
  `UPPER_BOUNDS` (2^63 already fails the 64-bit check in `_coerce`).
- `lab_lattice.json`: `axis` = `np.linspace(0, 1, 10)`; `rgb`, the 1,000 lattice colours (r
  outermost, then g, then b); `lab` = skimage `rgb2lab(rgb)` (D65). The three LAB files
  are recorded from scikit-image itself, not from `pixelforge.color` (native since Phase 2),
  which is tested against them.
- `lab_roundtrip.json`: `lab`, 200 inputs: L in `linspace(0, 100, 8)` × a and b in
  `linspace(-128, 127, 5)` (L outermost; many are out of gamut); `lab_to_rgb` (floats
  clipped to [0, 1]) and `lab_to_rgb8` (`np.round`, half to even).
- `rgb8_lab.json`: `rgb8`, the 256 greys then the 8 cube corners (r, g, b in {0, 255}, r
  outermost); `lab` = `rgb8_to_lab(rgb8)`.
- `edt_indices.json`: `cases` of `{name, mask, distances, indices_row, indices_col}` from
  `scipy.ndimage.distance_transform_edt(~mask, return_distances=True, return_indices=True)`:
  for each pixel, the distance to and the coordinates of the opaque pixel it copies its
  colour from (`_fill_transparent`), which pins scipy's tie-breaking. Masks: five 48×48
  random masks at opaque densities 0.1, 0.3, 0.5, 0.7, 0.9
  (`default_rng(20261010).random((48, 48)) < density`, drawn in that order);
  `circle_alpha.png` alpha >= 128 at rows 20..84, cols 20..84; a single opaque pixel at
  (17, 30) in 48×48; and an all-opaque 16×16.
- `skimage_bilateral.json`: `cases` of `pixelforge.preprocess.denoise` with
  `denoise="bilateral"`, i.e. `skimage.restoration.denoise_bilateral(rgb, sigma_color,
  sigma_spatial, channel_axis=-1, mode="edge")` clipped to [0, 1]. The input is
  `filled_rgb8 / 255.0`: the fixture crop at (`top`, `left`) of `size` × `size`, with
  transparent pixels (`mask` = alpha >= 128 within the crop) filled by
  `preprocess._fill_transparent` on the crop. Crops: `noisy_gradient.png` and
  `circle_alpha.png` at the default sigmas (0.08, 2.0) at 48×48 and at sigma_spatial 0.5 at
  32×32 (same corner; smaller to stay within the size budget), and a 24×24 corner crop of
  `noisy_gradient.png` at sigma_color 0.5, sigma_spatial 16. `output` is size rows of
  size*3 values.
- `ndimage_filters.json`: `arrays`: `random_17x23` (`default_rng(20261011).random((17,
  23))`), `ramp_40` (`linspace(-1, 2, 40)`), `constant_6x9` (all 0.3). `filters`:
  `{func, input, params, output}` for `scipy.ndimage.gaussian_filter1d` (sigma 1.5 and 3.9,
  default truncate 4) and `uniform_filter1d` (sizes 3, 5, 9, 10) along every axis with
  modes `nearest` and `wrap`, and `uniform_filter` / `median_filter` with size 3, mode
  `nearest`. `samples`, `percentile` (`np.percentile(x, (1.0, 99.0, 50.0))`, linear) and
  `median` (`np.median`, axis None, or 0 for `border_10x3`) on odd, even, duplicate, size-1
  and size-2 arrays (the random ones from the same generator, after `random_17x23`).
- `float_repr.json`: `repr`, 10,000 distinct finite floats as Python `repr` strings (the
  string is the ground truth: parse it as f64, format it back, compare): powers of ten
  1e-320..1e308, short decimals, random mantissas at binary exponents -1074..1023, random
  bit patterns, integers as floats, subnormals, the extremes, negatives and -0.0.
  `json_dumps_equals_repr`: `json.dumps(x) == repr(x)` held for every one;
  `canonical_json_of_list`: SHA-256 of `json.dumps(floats, separators=(",", ":"))`.
- `outputs_small/`: two full pipeline runs saved with `Result.save`; `<prefix>_config.json`
  holds the fixture and the Config kwargs. `two_color_sprite`: sprite, box, 16×16, no
  denoise, no keying, on `two_color.png`. `checker_tiles_background`: background (tileset),
  box, 32×32, no denoise, no keying, on `checker_tiles.png`. The text outputs are kept as
  written (`_palette.json`, `_palette.hex`, `.tmj`, `_tilemap.json`, `_tilemap.csv`), except
  `_meta.json`, whose `environment` and `stats.timings` keys are removed. PNG bytes are not
  committed; `<name>.png.json` holds the decoded PNG: `mode`, `size`, `palette` (PLTE as a
  flat list), `transparency`, `text` (`io.read_png_text`), `pixels` (palette indices) and
  `pixels_sha256` (SHA-256 of the decoded index bytes); for `_preview.png` (the 8× nearest
  upscale) the hash only.

## Recorded with

| Library | Version |
| --- | --- |
{{VERSIONS}}
"""


# ------------------------------------------------------------------------------------ main

def versions() -> dict:
    from pixelforge import __version__

    out = {name: importlib.metadata.version(name) for name in LIBRARIES}
    out["pixelforge"] = __version__
    return out


def record(out_dir: Path) -> None:
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        if any(out_dir.iterdir()) and not (out_dir / "versions.json").exists():
            raise SystemExit(f"{out_dir} is not empty and holds no oracle; not replacing it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    palette_dir = out_dir / "palettes"
    palette_dir.mkdir()
    for name, text in PALETTE_FILES.items():
        (palette_dir / name).write_bytes(text.encode("utf-8"))

    # Palette paths in the corpus are relative to the repository root; resolve them against
    # a scratch root holding this run's palette files, so any OUTDIR records the same thing.
    with tempfile.TemporaryDirectory() as scratch:
        shutil.copytree(palette_dir, Path(scratch) / PALETTE_REL)
        with contextlib.chdir(scratch):
            write_json(out_dir / "config_corpus.json", record_config_corpus())
            write_json(out_dir / "config_errors.json", record_config_errors())
    for name, data in record_lab().items():
        write_json(out_dir / name, data)
    write_json(out_dir / "edt_indices.json", record_edt())
    write_json(out_dir / "skimage_bilateral.json", record_bilateral())
    write_json(out_dir / "ndimage_filters.json", record_ndimage())
    write_json(out_dir / "float_repr.json", record_float_repr())
    record_outputs(out_dir)
    write_json(out_dir / "versions.json", versions())
    (out_dir / "README.md").write_text(readme(versions()), encoding="utf-8")


def readme(libs: dict) -> str:
    table = "\n".join(f"| {name} | {version} |" for name, version in sorted(libs.items()))
    return README.replace("{{VERSIONS}}", table)


if __name__ == "__main__":
    record(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR)
