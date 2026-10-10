"""Config dataclass, presets, validation and hashing (REQUIREMENTS.md Section 4)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from dataclasses import asdict, dataclass, fields

from .version import __version__

METHODS = ("box", "kopf", "gerstner")
DENOISERS = ("none", "bilateral", "median")
DITHERS = ("none", "bayer4", "bayer8", "auto")
PALETTE_SOURCES = ("auto", "median_cut", "mcda")

PRESETS = {
    "sprite": dict(remove_bg=None, key_bg=True, crop_to_alpha=True, method="gerstner",
                   palette_size=16, dither="none", outline="auto", tileset=False,
                   denoise="bilateral"),
    "background": dict(remove_bg=False, key_bg=False, crop_to_alpha=False, method="kopf",
                       palette_size=32, dither="auto", outline="none", tileset=True,
                       denoise="bilateral"),
}

# Longest output edge when neither out_width nor out_height is given.
PRESET_LONGEST_EDGE = {"sprite": 64, "background": 256}

_HEX_COLOR = re.compile(r"#[0-9a-f]{6}")

# Upper bounds that keep a single run's memory and CPU finite (inclusive).
UPPER_BOUNDS = {"out_width": 4096, "out_height": 4096, "scale_preview": 64,
                "denoise_sigma_spatial": 16, "kopf_max_iters": 1000, "g_max_iters": 10000,
                "saturation_beta": 5, "seed": 2**63 - 1, "tile_size": 512}


def rembg_available() -> bool:
    return importlib.util.find_spec("rembg") is not None


@dataclass(frozen=True, init=False)
class Config:
    # --- target ---
    preset: str = "sprite"            # "sprite" | "background"
    out_width: int | None = None      # if None, derived from out_height and aspect
    out_height: int | None = None     # if both None: sprite→64 on longest edge, background→256
    # --- preprocess ---
    remove_bg: bool = False           # sprite preset default True if rembg installed, else False
    key_bg: bool = False              # key out a flat opaque background (sprite default True)
    key_bg_tolerance: float = 0.08    # max per-channel sRGB distance from the background color
    crop_to_alpha: bool = True        # crop to alpha bbox before downscale (only if alpha present)
    alpha_threshold: int = 128        # input alpha < this → transparent
    denoise: str = "bilateral"        # "none" | "bilateral" | "median"
    denoise_sigma_color: float = 0.08  # bilateral, in [0,1] sRGB units
    denoise_sigma_spatial: float = 2.0
    # --- downscale ---
    method: str = "gerstner"          # "box" | "kopf" | "gerstner"
    kopf_max_iters: int = 50
    kopf_tol: float = 1e-3
    # --- palette ---
    palette_size: int = 16            # K. Ignored if palette_name set.
    palette_name: str | None = None   # bundled name or path to .hex/.gpl
    palette_source: str = "auto"      # "auto" | "median_cut" | "mcda"
    palette_ramps: bool = True        # post-cluster hue-ramp regularization (Section 8.1)
    saturation_beta: float = 1.1      # Gerstner β; multiply a,b by this after palette convergence
    # --- gerstner ---
    g_m: float = 45.0                 # SLIC compactness
    g_alpha: float = 0.7              # temperature decay
    g_T_final: float = 1.0
    g_eps_palette: float = 1.0        # total palette change (LAB) below which "converged"
    g_eps_cluster: float = 0.25       # sub-cluster separation (LAB) to split
    g_laplacian: float = 0.4          # superpixel center smoothing fraction
    g_bilateral_sigma_color: float = 10.0   # on the wout×hout mean-color image, LAB units
    g_bilateral_sigma_spatial: float = 1.5
    g_max_iters: int = 2000           # safety cap on total iterations
    # --- dither ---
    dither: str = "auto"              # "none" | "bayer4" | "bayer8" | "auto"
    dither_strength: float = 0.5      # 0..1, scales the threshold matrix amplitude
    dither_variance_threshold: float = 4.0  # "auto": dither only where local LAB std < this
    # --- postprocess ---
    remove_orphans: bool = True
    orphan_min_region: int = 2        # connected regions (8-conn) with < this many px get merged
    orphan_max_delta: float = 25.0    # ...only if ΔE (LAB) to the replacement is below this;
                                      # high-contrast singles (eyes, highlights) are kept
    fix_jaggies: bool = True
    outline: str = "auto"             # "none" | "auto" | "#rrggbb"
    outline_darken: float = 0.55      # auto outline = darkest palette color blended toward black
    # --- tiles (background only) ---
    tile_size: int = 16
    tileset: bool = False             # emit tileset.png + tilemap.json
    tile_dedupe_tolerance: float = 2.0  # mean LAB distance between tiles to consider identical
    seamless: bool = False            # wrap-around filtering so output tiles seamlessly
    tileset_columns: int = 0          # tiles per tileset row; 0 = ceil(sqrt(tile count))
    # --- misc ---
    seed: int | None = None           # only used for optional MCDA jitter; None = fully analytic
    scale_preview: int = 8            # nearest-neighbor upscale factor for *_preview.png

    def __init__(self, **kwargs):
        """Build a config as: field defaults ← preset ← explicit keyword arguments."""
        specs = {f.name: f for f in fields(self)}
        unknown = sorted(set(kwargs) - set(specs))
        if unknown:
            raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
        preset = kwargs.get("preset", specs["preset"].default)
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
        values = {name: f.default for name, f in specs.items()}
        values.update(PRESETS[preset])
        values.update(kwargs)
        if values["remove_bg"] is None:
            values["remove_bg"] = rembg_available()
        for name, f in specs.items():
            object.__setattr__(self, name, _coerce(name, f.type, values[name]))
        # Not a dataclass field: invisible to to_dict(), hashing and equality. replace() uses it
        # so that a changed preset is re-applied to every field the caller did not set.
        object.__setattr__(self, "_explicit", frozenset(kwargs))
        self._validate()

    # ------------------------------------------------------------------ helpers

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        payload = self.canonical_json() + "\n" + __version__
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def replace(self, **changes) -> "Config":
        """A copy with `changes` applied; a changed preset re-applies to non-explicit fields."""
        values = {name: getattr(self, name) for name in self._explicit}
        values.update(changes)
        return Config(**values)

    def _validate(self) -> None:
        for name in ("out_width", "out_height"):
            v = getattr(self, name)
            if v is not None and v < 8:
                raise ValueError(f"{name} must be >= 8, got {v}")
        if not 2 <= self.palette_size <= 256:
            raise ValueError(f"palette_size must be in [2, 256], got {self.palette_size}")
        _choice("method", self.method, METHODS)
        _choice("denoise", self.denoise, DENOISERS)
        _choice("dither", self.dither, DITHERS)
        _choice("palette_source", self.palette_source, PALETTE_SOURCES)
        if self.outline not in ("none", "auto") and not _HEX_COLOR.fullmatch(self.outline):
            raise ValueError(f'outline must be "none", "auto" or "#rrggbb", got {self.outline!r}')
        if not 0.0 < self.g_alpha < 1.0:
            raise ValueError(f"g_alpha must be in (0, 1), got {self.g_alpha}")
        if not 0 <= self.alpha_threshold <= 255:
            raise ValueError(f"alpha_threshold must be in [0, 255], got {self.alpha_threshold}")
        if not 0.0 <= self.dither_strength <= 1.0:
            raise ValueError(f"dither_strength must be in [0, 1], got {self.dither_strength}")
        if not 0.0 <= self.key_bg_tolerance <= 1.0:
            raise ValueError(
                f"key_bg_tolerance must be in [0, 1], got {self.key_bg_tolerance}")
        if not (math.isfinite(self.orphan_max_delta) and self.orphan_max_delta >= 0):
            raise ValueError(
                f"orphan_max_delta must be finite and >= 0, got {self.orphan_max_delta}")
        if not 0.0 <= self.outline_darken <= 1.0:
            raise ValueError(f"outline_darken must be in [0, 1], got {self.outline_darken}")
        for name in ("kopf_max_iters", "g_max_iters", "tile_size", "scale_preview",
                     "orphan_min_region"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("g_T_final", "g_m", "kopf_tol", "saturation_beta", "denoise_sigma_color",
                     "denoise_sigma_spatial", "g_bilateral_sigma_color",
                     "g_bilateral_sigma_spatial"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("dither_variance_threshold", "tile_dedupe_tolerance"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0 <= self.tileset_columns <= 256:
            raise ValueError(f"tileset_columns must be 0 (automatic) or in [1, 256], "
                             f"got {self.tileset_columns}")
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        for name, bound in UPPER_BOUNDS.items():
            v = getattr(self, name)
            if v is not None and v > bound:
                raise ValueError(f"{name} must be <= {bound}, got {v}")
        if self.tileset:
            for name in ("out_width", "out_height"):
                v = getattr(self, name)
                if v is not None and v % self.tile_size != 0:
                    raise ValueError(
                        f"tile_size ({self.tile_size}) must divide {name} ({v}) when tileset=True")


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _coerce(name: str, annotation: str, value):
    """Coerce a value to the field's declared type so equal configs hash equally.

    Non-finite floats and integers outside the signed 64-bit range are rejected for every
    field, so no bound check can be bypassed with NaN or Infinity.
    """
    parts = [p.strip() for p in annotation.split("|")]
    if value is None:
        if "None" in parts:
            return None
        raise ValueError(f"{name} must not be None")
    base = parts[0]
    if base not in ("bool", "int", "float", "str"):
        raise ValueError(f"unsupported field type {annotation!r} for {name}")
    try:
        if base == "bool":
            if isinstance(value, bool):
                return value
            raise TypeError
        if base == "int":
            if isinstance(value, bool) or int(value) != value:
                raise TypeError
            result = int(value)
        elif base == "float":
            if isinstance(value, bool):
                raise TypeError
            result = float(value)
        elif base == "str":
            if not isinstance(value, str):
                raise TypeError
            # "#AABBCC" and "#aabbcc" are the same outline and must hash equally.
            return value.lower() if name == "outline" else value
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be of type {annotation}, got {value!r}") from None
    if base == "int" and not -2**63 <= result <= 2**63 - 1:
        raise ValueError(f"{name} must fit in a signed 64-bit integer, got {value!r}")
    if base == "float" and not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result
