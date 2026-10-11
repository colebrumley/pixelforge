"""pixelforge: deterministic conversion of raster images into 16-bit-style pixel art.

The Python binding of the pixelforge Rust library (the extension module
``pixelforge._native``). The command-line program is the Rust ``pixelforge`` binary; this
package is optional and only needed to call the library from Python::

    from pixelforge import Config, run

    res = run("hero.png", Config(preset="sprite", palette_size=12, out_height=48))
    res.save("out/hero")    # out/hero.png, hero_preview.png, hero_palette.json, ...

Outputs are byte-identical to ``pixelforge convert`` with the same settings.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


class PixelforgeError(RuntimeError):
    """A runtime failure that is not a config validation error (undecodable, oversized or
    fully transparent input, ...)."""


class ConfigError(PixelforgeError, ValueError):
    """An invalid configuration: a bad field value, an unknown or unreadable palette.

    Also a ValueError, so ``except ValueError`` around ``Config(...)`` works.
    """


# After the exception classes: the extension module raises them by name.
import numpy as np  # noqa: E402

from . import _native  # noqa: E402

__version__: str = _native.native_version()

# Library convention: never configure output; the caller adds a handler.
logging.getLogger("pixelforge").addHandler(logging.NullHandler())

_FIELDS: tuple[str, ...] = tuple(_native.config_fields())

PRESETS = MappingProxyType({name: MappingProxyType(values)
                            for name, values in _native.config_presets().items()})
"""Each preset's overrides of the field defaults (read-only)."""

MAX_INPUT_PIXELS: int = _native.MAX_INPUT_PIXELS
"""The default input pixel budget (width × height), checked before decoding."""


class Config:
    """A validated, immutable configuration: field defaults ← preset ← keyword arguments.

    ``Config(preset="background", palette_size=24)``; every field is an attribute
    (``Config.field_names()`` lists them). Raises ``ConfigError`` for unknown fields and
    invalid values, with the same messages as the CLI.
    """

    __slots__ = ("_native", "_values")

    def __init__(self, **kwargs: Any) -> None:
        self._init(_native.NativeConfig(kwargs))

    def _init(self, native: Any) -> None:
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_values", native.to_dict())

    @classmethod
    def _wrap(cls, native: Any) -> Config:
        config = cls.__new__(cls)
        config._init(native)
        return config

    @classmethod
    def field_names(cls) -> list[str]:
        return list(_FIELDS)

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("_"):
            values = object.__getattribute__(self, "_values")
            if name in values:
                return values[name]
        raise AttributeError(f"'Config' object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is immutable; use replace(**changes)")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Config is immutable; use replace(**changes)")

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(_FIELDS))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self._values == other._values

    def __hash__(self) -> int:
        return hash(self.canonical_json())

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={value!r}" for name, value in self._values.items())
        return f"Config({fields})"

    def __reduce__(self) -> tuple:
        return (_from_dict, (self.to_dict(),))

    def to_dict(self) -> dict[str, Any]:
        """Every field and its value, in field order."""
        return dict(self._values)

    def canonical_json(self) -> str:
        """The config as embedded in every output PNG (sorted keys, no whitespace)."""
        return self._native.canonical_json()

    def hash(self) -> str:
        """SHA-256 of the canonical JSON, the version and, with ``palette_name``, the
        palette's colors (``config_hash`` in ``_meta.json``)."""
        return self._native.hash()

    def provenance(self) -> dict[str, Any]:
        """Everything needed to reproduce a run besides the input image: ``config``,
        ``palette_sha256``, ``palette`` (its 'rrggbb' lines) and ``version``."""
        return self._native.provenance()

    def replace(self, **changes: Any) -> Config:
        """A copy with ``changes`` applied; a changed preset re-applies to every field that
        was not given explicitly."""
        return Config._wrap(self._native.replace(changes))

    def validate_palette(self) -> None:
        """Read ``palette_name`` once; ``ConfigError`` if it is unusable. ``run`` calls this
        before any expensive stage; ``Config(...)`` itself reads no file."""
        self._native.validate_palette()

    @property
    def canvas_size(self) -> tuple[int, int] | None:
        """(width, height) of ``canvas``, or None."""
        return self._native.canvas_size()


def _from_dict(values: dict[str, Any]) -> Config:
    return Config(**values)


@dataclass
class Loaded:
    """A decoded image: what ``load`` and ``from_image`` return and ``run_loaded`` takes."""

    rgb: np.ndarray        # (H, W, 3) float64 sRGB in [0, 1]
    alpha: np.ndarray      # (H, W) uint8
    sha256: str            # SHA-256 of the file bytes, or of shape + RGBA bytes for arrays


@dataclass(eq=False)
class Result:
    """The output of a run. ``save(prefix)`` writes the files ``pixelforge convert`` writes."""

    image: np.ndarray                 # (H, W, 4) uint8 RGBA at native pixel-art resolution
    palette: np.ndarray               # (K, 3) uint8
    indices: np.ndarray               # (H, W) int64, -1 for transparent
    config: Config
    input_sha256: str
    palette_lab: np.ndarray           # (K, 3) float64
    stats: dict = field(default_factory=dict)
    tileset: np.ndarray | None = None  # (h, w, 4) uint8 RGBA tileset sheet
    tilemap: dict | None = None
    tileset_indices: np.ndarray | None = None
    _output: Any = field(default=None, repr=False)

    def save(self, prefix: str | os.PathLike) -> dict[str, str]:
        """Write <prefix>.png, _preview.png, _palette.json, _palette.hex, _meta.json and,
        for tilesets, _tileset.png, _tilemap.json, .tmj (Tiled) and _tilemap.csv, from the
        result as ``run`` returned it. Returns {name: path}."""
        if self._output is None:
            raise PixelforgeError("Result.save needs a Result returned by run or run_loaded")
        return dict(self._output.save(os.fspath(prefix)))


def _budget(max_pixels: int | None) -> int:
    return MAX_INPUT_PIXELS if max_pixels is None else max_pixels


def load(path: str | os.PathLike, max_pixels: int | None = None) -> Loaded:
    """Decode an image file (PNG, JPEG, GIF, WEBP, BMP, TIFF; regular files only).

    EXIF orientation is applied; animations use their first frame (with a warning).
    ``PixelforgeError`` for undecodable or unsupported files and above ``max_pixels``
    (default ``MAX_INPUT_PIXELS``); ``OSError`` if the file cannot be opened.
    """
    return Loaded(*_native.load(os.fspath(path), _budget(max_pixels)))


_HIGH_BIT_MODES = ("I;16", "I;16B", "I;16L", "I")


def _pil_to_rgba(image: Any, max_pixels: int) -> np.ndarray:
    """A PIL image as (H, W, 4) uint8: EXIF orientation applied, 16-bit grey rescaled."""
    from PIL import Image, ImageOps

    width, height = image.size
    if width * height > max_pixels:
        raise PixelforgeError(
            f"image: {width}x{height} = {width * height} pixels exceeds the "
            f"input budget of {max_pixels} pixels (--max-input-pixels)")
    im = ImageOps.exif_transpose(image)
    if im.mode in _HIGH_BIT_MODES:      # Pillow would clip 16-bit grey at 255
        values = np.asarray(im, dtype=np.float64)
        grey = np.round(np.clip(values, 0.0, 65535.0) / 257.0).astype(np.uint8)
        im = Image.fromarray(grey, mode="L")
    return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def _is_pil_image(image: Any) -> bool:
    if not type(image).__module__.startswith("PIL."):
        return False
    try:
        from PIL import Image
    except ImportError as exc:
        raise PixelforgeError(
            "PIL images need Pillow installed (pip install Pillow); "
            "or pass a path or a numpy array") from exc
    return isinstance(image, Image.Image)


def from_image(image: Any, max_pixels: int | None = None) -> Loaded:
    """An in-memory image: an (H, W, 3) / (H, W, 4) numpy array of uint8 or of float in
    [0, 1], or a ``PIL.Image.Image`` (needs Pillow). The sha256 covers the shape and the
    RGBA bytes, so equal pixels give equal provenance. ``PixelforgeError`` for other
    shapes or dtypes and above ``max_pixels``."""
    budget = _budget(max_pixels)
    if _is_pil_image(image):
        image = _pil_to_rgba(image, budget)
    return Loaded(*_native.from_array(image, budget))


def run_loaded(loaded: Loaded, config: Config, timings: dict[str, float] | None = None
               ) -> Result:
    """Convert an already loaded image (``load`` / ``from_image``)."""
    if not isinstance(config, Config):
        raise TypeError(f"config must be a pixelforge.Config, got {type(config).__name__}")
    out = _native.run_loaded(np.ascontiguousarray(loaded.rgb, dtype=np.float64),
                             np.ascontiguousarray(loaded.alpha, dtype=np.uint8),
                             loaded.sha256, config._native, dict(timings or {}))
    return Result(image=out.image, palette=out.palette, indices=out.indices, config=config,
                  input_sha256=out.input_sha256, palette_lab=out.palette_lab,
                  stats=out.stats, tileset=out.tileset, tilemap=out.tilemap,
                  tileset_indices=out.tileset_indices, _output=out)


def run(image: Any, config: Config, max_pixels: int | None = None) -> Result:
    """Load and convert one image.

    ``image`` is a path (str or os.PathLike), an (H, W, 3) / (H, W, 4) numpy array of
    uint8 or of float in [0, 1], or a ``PIL.Image.Image`` (needs Pillow). For paths
    ``input_sha256`` hashes the file bytes; for in-memory images it hashes shape and RGBA
    bytes. ``max_pixels`` defaults to ``MAX_INPUT_PIXELS`` and applies to every kind.
    """
    start = time.perf_counter()
    if isinstance(image, (str, os.PathLike)):
        loaded = load(image, max_pixels)
    else:
        loaded = from_image(image, max_pixels)
    return run_loaded(loaded, config, {"load": time.perf_counter() - start})


def thread_count() -> int:
    """Worker threads the pipeline uses (``PIXELFORGE_THREADS``, else every core)."""
    return _native.thread_count()


__all__ = ["Config", "ConfigError", "Loaded", "MAX_INPUT_PIXELS", "PRESETS", "PixelforgeError",
           "Result", "__version__", "from_image", "load", "run", "run_loaded", "thread_count"]

