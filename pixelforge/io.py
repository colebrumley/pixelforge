"""Image load/save, alpha handling and PNG metadata embedding.

The work is done by the native module (crates/pixelforge/src/io.rs, through
``pixelforge._native``): decoding the six accepted formats as Pillow 12 did, the input hash,
the array inputs and the PNG writer. Pillow is only imported here for ``PIL.Image.Image``
inputs to :func:`from_image`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from . import _native
from .errors import PixelforgeError

_CONSTANTS = _native.io_constants()

META_CONFIG_KEY = _CONSTANTS["META_CONFIG_KEY"]
META_INPUT_KEY = _CONSTANTS["META_INPUT_KEY"]
META_PALETTE_SHA_KEY = _CONSTANTS["META_PALETTE_SHA_KEY"]   # only when palette_name is set
META_PALETTE_KEY = _CONSTANTS["META_PALETTE_KEY"]           # resolved 'rrggbb' lines, newline-joined

# Default input pixel budget (width × height), checked before any pixel data is decoded.
MAX_INPUT_PIXELS = _CONSTANTS["MAX_INPUT_PIXELS"]
# Only these formats are decoded.
INPUT_FORMATS = _CONSTANTS["INPUT_FORMATS"]
_HIGH_BIT_MODES = ("I;16", "I;16B", "I;16L", "I")


@dataclass
class Loaded:
    rgb: np.ndarray        # (H, W, 3) float64 sRGB in [0, 1]
    alpha: np.ndarray      # (H, W) uint8
    sha256: str            # SHA-256 of the input file bytes (see from_image for arrays)

    @property
    def alpha_mask(self) -> np.ndarray:
        """Alpha as a float mask in [0, 1]."""
        return self.alpha.astype(np.float64) / 255.0


def open_regular(path):
    """Open a regular file for binary reading; refuse FIFOs, devices and directories.

    O_NONBLOCK keeps the open itself from hanging on a FIFO; the check runs on the opened
    descriptor, so the file cannot be swapped between check and read.
    """
    return os.fdopen(_native.io_open_regular(path), "rb")


def sha256_file(path) -> str:
    return _native.io_sha256_file(path)


def load(path, max_pixels: int = MAX_INPUT_PIXELS) -> Loaded:
    """Decode an image file to RGBA.

    Raises PixelforgeError for non-regular files, unsupported or undecodable data and images
    with more than ``max_pixels`` pixels. Multi-frame images use their first frame (with a
    warning); EXIF orientation is applied.
    """
    return Loaded(*_native.io_load(path, max_pixels))


def _pil_to_rgba(image) -> np.ndarray:
    """A PIL image as (H, W, 4) uint8: EXIF orientation applied, 16-bit gray rescaled."""
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(image)
    if im.mode in _HIGH_BIT_MODES:      # Pillow would clip 16-bit gray at 255
        values = np.asarray(im, dtype=np.float64)
        gray = np.round(np.clip(values, 0.0, 65535.0) / 257.0).astype(np.uint8)
        im = Image.fromarray(gray, mode="L")
    return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def from_image(image, max_pixels: int = MAX_INPUT_PIXELS) -> Loaded:
    """An in-memory image as Loaded: a ``PIL.Image.Image`` or an (H, W, 3) / (H, W, 4) array.

    Arrays are uint8, or float in [0, 1] (rounded to 8 bits, like every decoded file). PIL
    images get the same conversion as files (EXIF orientation, 16-bit gray rescaled). The
    sha256 covers the shape and the RGBA bytes, so equal pixels give equal provenance.
    Raises PixelforgeError for other shapes or dtypes and above ``max_pixels``.
    """
    pil = sys.modules.get("PIL.Image")
    if pil is not None and isinstance(image, pil.Image):
        width, height = image.size
        _check_budget("image", width, height, max_pixels)
        image = _pil_to_rgba(image)
    return Loaded(*_native.io_from_image_array(image, max_pixels))


def _check_budget(name, width: int, height: int, max_pixels: int) -> None:
    if width * height > max_pixels:
        raise PixelforgeError(
            f"{name}: {width}x{height} = {width * height} pixels exceeds the "
            f"input budget of {max_pixels} pixels (--max-input-pixels)")


def from_rgba(rgba: np.ndarray, sha256: str = "") -> Loaded:
    rgba = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8)[..., :4])
    return Loaded(*_native.io_from_rgba(rgba, sha256))


def indices_to_rgba(indices: np.ndarray, palette_rgb8: np.ndarray) -> np.ndarray:
    """Index image (-1 = transparent) + palette -> (H, W, 4) uint8 RGBA."""
    return _native.io_indices_to_rgba(np.ascontiguousarray(indices, dtype=np.int64),
                                      _palette(palette_rgb8))


def upscale_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in (np.uint8, np.int64, np.float64):
        return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    return _native.io_upscale_nearest(arr, factor)


def _palette(palette_rgb8) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(palette_rgb8, dtype=np.uint8).reshape(-1, 3))


def save_png(path, indices: np.ndarray, palette_rgb8: np.ndarray, text: dict | None = None,
             scale: int = 1, transparent_index: str = "last") -> None:
    """Write an index image as PNG: palette-indexed if it fits in 256 entries, else RGBA.

    ``transparent_index="last"`` appends a transparent entry after the K colors, only when
    some pixel is transparent. ``"first"`` always puts it at index 0 and shifts every color
    up by one, so every PNG written with the same palette has the same PLTE.
    """
    _native.io_save_png(path, np.ascontiguousarray(indices, dtype=np.int64),
                        _palette(palette_rgb8), text, scale, transparent_index)


def read_png_text(path) -> dict:
    return _native.io_read_png_text(path)
