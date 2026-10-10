"""Config dataclass, presets, validation and hashing (REQUIREMENTS.md Section 4)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

from .errors import ConfigError, PixelforgeError
from .version import __version__

METHODS = ("box", "kopf", "gerstner")
DENOISERS = ("none", "bilateral", "median")
DITHERS = ("none", "bayer4", "bayer8", "auto")
PALETTE_SOURCES = ("auto", "median_cut", "mcda")
TRANSPARENT_INDICES = ("last", "first")
FITS = ("pad", "stretch", "crop")

# DEVIATION: Section 4 — the spec's sprite preset has remove_bg=None ("True if rembg is
# importable"), so one command gave different hashes and alpha on different machines and
# silently downloaded a model. remove_bg is opt-in (--remove-bg) in every preset.
# Read-only: a caller mutating a preset must not change every later Config.
PRESETS = MappingProxyType({name: MappingProxyType(values) for name, values in {
    "sprite": dict(remove_bg=False, key_bg=True, crop_to_alpha=True, method="gerstner",
                   palette_size=16, dither="none", outline="auto", tileset=False,
                   denoise="bilateral"),
    "background": dict(remove_bg=False, key_bg=False, crop_to_alpha=False, method="kopf",
                       palette_size=32, dither="auto", outline="none", tileset=True,
                       denoise="bilateral"),
}.items()})

# Longest output edge when neither out_width nor out_height is given.
PRESET_LONGEST_EDGE = MappingProxyType({"sprite": 64, "background": 256})

_HEX_COLOR = re.compile(r"#[0-9a-f]{6}")
_CANVAS = re.compile(r"([0-9]+)x([0-9]+)")

# Upper bounds that keep a single run's memory and CPU finite (inclusive).
UPPER_BOUNDS = {"out_width": 4096, "out_height": 4096, "scale": 4096, "scale_preview": 64,
                "denoise_sigma_spatial": 16, "kopf_max_iters": 1000, "g_max_iters": 10000,
                "saturation_beta": 5, "seed": 2**63 - 1, "tile_size": 512,
                "enhance": 4, "enhance_radius": 16, "ink": 1}


# One-line --help text per Config field (the CLI generates a flag for each).
FIELD_HELP = MappingProxyType({
    "preset": "Starting values for the other fields: sprite or background.",
    "out_width": "Output width in pixels; height follows the aspect if not given.",
    "out_height": "Output height in pixels; width follows the aspect if not given.",
    "scale": "Fixed downscale factor: output = round(cropped input / scale).",
    "canvas": "Fit the subject centered inside a WxH canvas (outline drawn inside).",
    "remove_bg": "Remove the background with rembg AI matting (opt-in, needs the extra).",
    "key_bg": "Key out a flat opaque background color.",
    "key_bg_tolerance": "Max per-channel sRGB distance from the background color, 0..1.",
    "key_bg_fringe": "Passes keying out the anti-aliased background edge (0 = off).",
    "crop_to_alpha": "Crop to the alpha bounding box before downscaling.",
    "alpha_threshold": "Input alpha below this is transparent (0..255).",
    "denoise": "Denoise filter applied before downscaling.",
    "denoise_sigma_color": "Bilateral color sigma in [0, 1] sRGB units.",
    "denoise_sigma_spatial": "Denoise spatial sigma in input pixels (after pre-reduction).",
    "prereduce_max_ratio": "Box-reduce inputs larger than this many times the output first "
                           "(0 = never).",
    "enhance": "Exaggerate features before downscaling: levels stretch plus local contrast "
               "(0 = off; for photos, try 1).",
    "enhance_radius": "Feature size the local contrast acts on, in output pixels.",
    "ink": "Darken thin dark features (lines, lashes, creases) before downscaling, 0..1.",
    "method": "Downscaling algorithm.",
    "kopf_max_iters": "Iteration cap for the kopf downscaler.",
    "kopf_tol": "kopf convergence tolerance (RMS color change in the unit cube).",
    "palette_size": "Number of palette colors K; ignored with --palette-name.",
    "palette_name": "Bundled palette name, a .hex/.gpl path or hex:rrggbb,...",
    "palette_source": "Palette builder (auto = mcda for gerstner, median_cut otherwise).",
    "palette_ramps": "Regularize the palette into hue ramps after clustering.",
    "saturation_beta": "Multiply palette a*, b* (saturation) by this after convergence.",
    "g_m": "gerstner SLIC compactness.",
    "g_alpha": "gerstner annealing temperature decay per step, in (0, 1).",
    "g_T_final": "gerstner final annealing temperature.",
    "g_eps_palette": "gerstner total palette change (LAB) below which a step has converged.",
    "g_eps_cluster": "gerstner sub-cluster separation (LAB) needed to split a color.",
    "g_laplacian": "gerstner superpixel center smoothing fraction.",
    "g_bilateral_sigma_color": "gerstner bilateral color sigma on the mean-color image (LAB).",
    "g_bilateral_sigma_spatial": "gerstner bilateral spatial sigma, in output pixels.",
    "g_max_iters": "Safety cap on gerstner's total iterations.",
    "dither": "Ordered dithering; auto dithers only smooth regions.",
    "dither_strength": "Dither threshold amplitude, 0..1.",
    "dither_variance_threshold": "auto dither: only where local LAB std is below this.",
    "remove_orphans": "Merge tiny isolated color regions into their neighbors.",
    "orphan_min_region": "Regions (8-connected) with fewer pixels than this get merged.",
    "orphan_max_delta": "Only merge orphans whose replacement is within this LAB distance.",
    "fix_jaggies": "Clean up staircase artifacts on diagonal edges.",
    "outline": 'Silhouette outline: "none", "auto" (darkened palette color) or "#rrggbb".',
    "outline_darken": "auto outline: blend of the darkest palette color toward black, 0..1.",
    "tile_size": "Tile edge in pixels; only applies with --tileset.",
    "tileset": "Also write a de-duplicated tileset and tilemap.",
    "fit": "How a derived size meets tile_size: pad, stretch or crop.",
    "tile_dedupe_tolerance": "Mean LAB distance below which two tiles count as identical.",
    "seamless": "Wrap-around filtering so the output tiles seamlessly.",
    "tileset_columns": "Tiles per tileset row (0 = ceil(sqrt(tile count))).",
    "transparent_index": "PNG palette slot used for transparency.",
    "seed": "Seed for the optional MCDA jitter (unset = fully analytic).",
    "scale_preview": "Nearest-neighbor upscale factor for *_preview.png.",
})

# The allowed values of each enum-like string field (outline also takes "#rrggbb").
FIELD_CHOICES = MappingProxyType({
    "preset": tuple(PRESETS), "method": METHODS, "denoise": DENOISERS,
    "dither": DITHERS, "palette_source": PALETTE_SOURCES, "fit": FITS,
    "transparent_index": TRANSPARENT_INDICES})


def rembg_available() -> bool:
    """Whether rembg is importable (used for the error message; never changes a default)."""
    return importlib.util.find_spec("rembg") is not None


@dataclass(frozen=True, init=False)
class Config:
    # --- target ---
    preset: str = "sprite"            # "sprite" | "background"
    out_width: int | None = None      # if None, derived from out_height and aspect
    out_height: int | None = None     # if both None: sprite→64 on longest edge, background→256
    scale: float | None = None        # fixed downscale: output = round(cropped input / scale)
    canvas: str | None = None         # "WxH": fit the subject inside, centered (outline inside)
    # --- preprocess ---
    remove_bg: bool = False           # rembg AI matting; opt-in only (never environment-derived)
    # DEVIATION: Section 4 — key_bg, key_bg_tolerance and key_bg_fringe are not in the spec.
    key_bg: bool = False              # key out a flat opaque background (sprite default True)
    key_bg_tolerance: float = 0.08    # max per-channel sRGB distance from the background color
    key_bg_fringe: int = 3            # max passes keying out the anti-aliased edge (0 = off)
    crop_to_alpha: bool = True        # crop to alpha bbox before downscale (only if alpha present)
    alpha_threshold: int = 128        # input alpha < this → transparent
    denoise: str = "bilateral"        # "none" | "bilateral" | "median"
    denoise_sigma_color: float = 0.08  # bilateral, in [0,1] sRGB units
    denoise_sigma_spatial: float = 2.0  # in input pixels, after any pre-reduction
    prereduce_max_ratio: int = 8      # box-reduce inputs > this × output first; 0 = never
    # DEVIATION: Section 4 — enhance, enhance_radius and ink are not in the spec.
    enhance: float = 0.0              # levels stretch + local contrast before downscaling; 0 = off
    enhance_radius: float = 2.0       # local-contrast Gaussian sigma, in output pixels
    ink: float = 0.0                  # 0..1, darkening of thin dark features; 0 = off
    # --- downscale ---
    method: str = "gerstner"          # "box" | "kopf" | "gerstner"
    kopf_max_iters: int = 50
    kopf_tol: float = 1e-3            # RMS Δν (unit-cube color); RMS Δμ < 10× this, output px
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
    fit: str = "pad"                  # how a derived output size is reconciled with tile_size:
                                      # "pad" (keep aspect, pad transparent, centered) |
                                      # "stretch" (round up, aspect changes) | "crop" (centered)
    tile_dedupe_tolerance: float = 2.0  # mean LAB distance between tiles to consider identical
    seamless: bool = False            # wrap-around filtering so output tiles seamlessly
    tileset_columns: int = 0          # tiles per tileset row; 0 = ceil(sqrt(tile count))
    # --- output ---
    transparent_index: str = "last"   # PNG palette slot of transparency: "last" | "first"
    # --- misc ---
    seed: int | None = None           # only used for optional MCDA jitter; None = fully analytic
    scale_preview: int = 8            # nearest-neighbor upscale factor for *_preview.png

    def __init__(self, **kwargs):
        """Build a config as: field defaults ← preset ← explicit keyword arguments."""
        specs = {f.name: f for f in fields(self)}
        unknown = sorted(set(kwargs) - set(specs))
        if unknown:
            raise ConfigError(f"unknown config field(s): {', '.join(unknown)}")
        preset = kwargs.get("preset", specs["preset"].default)
        if preset not in PRESETS:
            raise ConfigError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
        values = {name: f.default for name, f in specs.items()}
        values.update(dict(PRESETS[preset]))
        values.update(kwargs)
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

    def palette_hex_lines(self) -> list[str] | None:
        """The resolved palette_name colors as 'rrggbb' lines (cached), or None."""
        if self.palette_name is None:
            return None
        cached = self.__dict__.get("_palette_lines")
        if cached is None:
            from .palette import parse_hex_lines   # palette imports io, which must not cycle
            cached = tuple(parse_hex_lines(self.palette_name))
            object.__setattr__(self, "_palette_lines", cached)
        return list(cached)

    def palette_sha256(self) -> str | None:
        """SHA-256 of the resolved palette's newline-joined hex lines, or None."""
        lines = self.palette_hex_lines()
        if lines is None:
            return None
        return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()

    def hash(self) -> str:
        """SHA-256 of the canonical JSON, the version and, with palette_name, the palette's
        contents, so rewriting a palette file (or a bundled palette) changes the hash."""
        payload = self.canonical_json() + "\n" + __version__
        if self.palette_name is not None:
            payload += "\npalette:" + self.palette_sha256()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def provenance(self) -> dict:
        """Everything needed to reproduce a run besides the input image."""
        return {"config": self.to_dict(), "palette_sha256": self.palette_sha256(),
                "palette": self.palette_hex_lines(), "version": __version__}

    @property
    def canvas_size(self) -> tuple[int, int] | None:
        """(width, height) of `canvas`, or None."""
        if self.canvas is None:
            return None
        w, h = _CANVAS.fullmatch(self.canvas).groups()
        return int(w), int(h)

    def replace(self, **changes) -> "Config":
        """A copy with `changes` applied; a changed preset re-applies to non-explicit fields."""
        values = {name: getattr(self, name) for name in self._explicit}
        values.update(changes)
        return Config(**values)

    def validate_palette(self) -> None:
        """Load ``palette_name`` once (file read and parse); ConfigError if it is unusable.

        ``Config(...)`` itself only checks the name's form, so constructing one stays free of
        file reads; ``pipeline.run_loaded`` calls this before preprocessing.
        """
        if self.palette_name is None:
            return
        from . import palette   # local: palette pulls in io and Pillow

        try:
            palette.load_palette(self.palette_name)
        except ConfigError:
            raise
        except (ValueError, OSError, PixelforgeError) as exc:
            raise ConfigError(str(exc)) from exc

    def _validate(self) -> None:
        if self.palette_name is not None:
            from .palette import INLINE_PREFIX, PALETTE_SUFFIXES, bundled_palettes

            if (self.palette_name not in bundled_palettes()
                    and not self.palette_name.lower().endswith(PALETTE_SUFFIXES)
                    and not self.palette_name.startswith(INLINE_PREFIX)):
                raise ConfigError(
                    f"unknown palette {self.palette_name!r}: not one of {bundled_palettes()}, "
                    "not a path to a .hex or .gpl file and not 'hex:rrggbb,...'")
        for name in ("out_width", "out_height"):
            v = getattr(self, name)
            if v is not None and v < 8:
                raise ConfigError(f"{name} must be >= 8, got {v}")
        if self.scale is not None and not self.scale > 0:
            raise ConfigError(f"scale must be > 0, got {self.scale}")
        if self.canvas is not None and not (_CANVAS.fullmatch(self.canvas) and all(
                8 <= v <= 4096 for v in map(int, _CANVAS.fullmatch(self.canvas).groups()))):
            raise ConfigError(f'canvas must be "WxH" with W, H in [8, 4096], got {self.canvas!r}')
        if (self.scale is not None or self.canvas is not None) and (
                self.out_width is not None or self.out_height is not None):
            raise ConfigError("scale and canvas cannot be combined with out_width/out_height")
        if not 2 <= self.palette_size <= 256:
            raise ConfigError(f"palette_size must be in [2, 256], got {self.palette_size}")
        _choice("method", self.method, METHODS)
        _choice("denoise", self.denoise, DENOISERS)
        _choice("dither", self.dither, DITHERS)
        _choice("palette_source", self.palette_source, PALETTE_SOURCES)
        _choice("transparent_index", self.transparent_index, TRANSPARENT_INDICES)
        _choice("fit", self.fit, FITS)
        if self.outline not in ("none", "auto") and not _HEX_COLOR.fullmatch(self.outline):
            raise ConfigError(f'outline must be "none", "auto" or "#rrggbb", got {self.outline!r}')
        if not 0.0 < self.g_alpha < 1.0:
            raise ConfigError(f"g_alpha must be in (0, 1), got {self.g_alpha}")
        if not 0 <= self.alpha_threshold <= 255:
            raise ConfigError(f"alpha_threshold must be in [0, 255], got {self.alpha_threshold}")
        if not 0.0 <= self.dither_strength <= 1.0:
            raise ConfigError(f"dither_strength must be in [0, 1], got {self.dither_strength}")
        if not 0.0 <= self.key_bg_tolerance <= 1.0:
            raise ConfigError(
                f"key_bg_tolerance must be in [0, 1], got {self.key_bg_tolerance}")
        if not (math.isfinite(self.orphan_max_delta) and self.orphan_max_delta >= 0):
            raise ConfigError(
                f"orphan_max_delta must be finite and >= 0, got {self.orphan_max_delta}")
        if not 0 <= self.key_bg_fringe <= 8:
            raise ConfigError(f"key_bg_fringe must be in [0, 8], got {self.key_bg_fringe}")
        if self.prereduce_max_ratio < 0:
            raise ConfigError(f"prereduce_max_ratio must be >= 0, got {self.prereduce_max_ratio}")
        if not 0.0 <= self.outline_darken <= 1.0:
            raise ConfigError(f"outline_darken must be in [0, 1], got {self.outline_darken}")
        for name in ("kopf_max_iters", "g_max_iters", "tile_size", "scale_preview",
                     "orphan_min_region"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("g_T_final", "g_m", "kopf_tol", "saturation_beta", "denoise_sigma_color",
                     "denoise_sigma_spatial", "g_bilateral_sigma_color",
                     "g_bilateral_sigma_spatial", "enhance_radius"):
            if not getattr(self, name) > 0:
                raise ConfigError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("dither_variance_threshold", "tile_dedupe_tolerance", "enhance", "ink"):
            if not getattr(self, name) >= 0:
                raise ConfigError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0 <= self.tileset_columns <= 256:
            raise ConfigError(f"tileset_columns must be 0 (automatic) or in [1, 256], "
                              f"got {self.tileset_columns}")
        if self.seed is not None and self.seed < 0:
            raise ConfigError(f"seed must be >= 0, got {self.seed}")
        for name, bound in UPPER_BOUNDS.items():
            v = getattr(self, name)
            if v is not None and v > bound:
                raise ConfigError(f"{name} must be <= {bound}, got {v}")
        if self.tileset:
            for name in ("out_width", "out_height"):
                v = getattr(self, name)
                if v is not None and v % self.tile_size != 0:
                    raise ConfigError(
                        f"tile_size ({self.tile_size}) must divide {name} ({v}) when tileset=True")
            if self.canvas is not None and any(v % self.tile_size for v in self.canvas_size):
                raise ConfigError(f"tile_size ({self.tile_size}) must divide canvas "
                                 f"({self.canvas}) when tileset=True")


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _coerce(name: str, annotation: str, value):
    """Coerce a value to the field's declared type so equal configs hash equally.

    Non-finite floats and integers outside the signed 64-bit range are rejected for every
    field, so no bound check can be bypassed with NaN or Infinity.
    """
    parts = [p.strip() for p in annotation.split("|")]
    if value is None:
        if "None" in parts:
            return None
        raise ConfigError(f"{name} must not be None")
    base = parts[0]
    if base not in ("bool", "int", "float", "str"):
        raise ConfigError(f"unsupported field type {annotation!r} for {name}")
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
        raise ConfigError(f"{name} must be of type {annotation}, got {value!r}") from None
    if base == "int" and not -2**63 <= result <= 2**63 - 1:
        raise ConfigError(f"{name} must fit in a signed 64-bit integer, got {value!r}")
    if base == "float" and not math.isfinite(result):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    return result
