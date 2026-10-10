"""Image load/save, alpha handling and PNG metadata embedding."""

from __future__ import annotations

import hashlib
import os
import stat
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

from .errors import PixelforgeError

META_CONFIG_KEY = "pixelforge:config"
META_INPUT_KEY = "pixelforge:input_sha256"
META_PALETTE_SHA_KEY = "pixelforge:palette_sha256"   # only when palette_name is set
META_PALETTE_KEY = "pixelforge:palette"              # resolved 'rrggbb' lines, newline-joined

# Default input pixel budget (width × height), checked before any pixel data is decoded.
MAX_INPUT_PIXELS = 24_000_000
# Only these decoders are reachable; Pillow's other plugins (EPS → Ghostscript, ...) are not.
INPUT_FORMATS = ("PNG", "JPEG", "GIF", "WEBP", "BMP", "TIFF")
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
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except IsADirectoryError:
        raise PixelforgeError(f"{path} is not a regular file") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PixelforgeError(f"{path} is not a regular file")
        if hasattr(os, "set_blocking"):
            os.set_blocking(fd, True)
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def sha256_file(path) -> str:
    with open_regular(path) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _to_8bit(im: Image.Image) -> Image.Image:
    """Rescale 16-bit (and 32-bit integer) grayscale to 8-bit; Pillow would clip it at 255."""
    if im.mode not in _HIGH_BIT_MODES:
        return im
    values = np.asarray(im, dtype=np.float64)
    gray = np.round(np.clip(values, 0.0, 65535.0) / 257.0).astype(np.uint8)
    return Image.fromarray(gray, mode="L")


def load(path, max_pixels: int = MAX_INPUT_PIXELS) -> Loaded:
    """Decode an image file to RGBA.

    Raises PixelforgeError for non-regular files, unsupported or undecodable data and images
    with more than ``max_pixels`` pixels. Multi-frame images use their first frame (with a
    warning); EXIF orientation is applied.
    """
    with open_regular(path) as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(f, formats=list(INPUT_FORMATS)) as im:
                    width, height = im.size
                    _check_budget(path, width, height, max_pixels)
                    if getattr(im, "n_frames", 1) > 1:
                        warnings.warn(f"{path} has {im.n_frames} frames; using the first",
                                      stacklevel=2)
                    frame = _to_8bit(ImageOps.exif_transpose(im))
                    rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8)
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise PixelforgeError(f"{path}: {exc}") from None
        except UnidentifiedImageError:
            raise PixelforgeError(
                f"{path}: not a supported image (expected one of "
                f"{', '.join(INPUT_FORMATS)})") from None
        except (OSError, SyntaxError, ValueError) as exc:
            raise PixelforgeError(f"{path}: could not decode image ({exc})") from None
    return from_rgba(rgba, digest)


def from_image(image, max_pixels: int = MAX_INPUT_PIXELS) -> Loaded:
    """An in-memory image as Loaded: a ``PIL.Image.Image`` or an (H, W, 3) / (H, W, 4) array.

    Arrays are uint8, or float in [0, 1] (rounded to 8 bits, like every decoded file). PIL
    images get the same conversion as files (EXIF orientation, 16-bit gray rescaled). The
    sha256 covers the shape and the RGBA bytes, so equal pixels give equal provenance.
    Raises PixelforgeError for other shapes or dtypes and above ``max_pixels``.
    """
    if isinstance(image, Image.Image):
        width, height = image.size
        _check_budget("image", width, height, max_pixels)
        rgba = np.asarray(_to_8bit(ImageOps.exif_transpose(image)).convert("RGBA"),
                          dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise PixelforgeError(f"image array must have shape (H, W, 3) or (H, W, 4), "
                                  f"got {arr.shape}")
        _check_budget("image", arr.shape[1], arr.shape[0], max_pixels)
        if np.issubdtype(arr.dtype, np.floating):
            if not (np.isfinite(arr).all() and arr.min(initial=0.0) >= 0.0
                    and arr.max(initial=0.0) <= 1.0):
                raise PixelforgeError("float image arrays must be finite and in [0, 1]")
            arr = np.round(arr.astype(np.float64) * 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            raise PixelforgeError(f"image array must be uint8 or float, got {arr.dtype}")
        if arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], 255, dtype=np.uint8)])
        rgba = np.ascontiguousarray(arr)
    digest = hashlib.sha256(f"rgba8 {rgba.shape[0]}x{rgba.shape[1]}\n".encode("ascii"))
    digest.update(rgba.tobytes())
    return from_rgba(rgba, digest.hexdigest())


def _check_budget(name, width: int, height: int, max_pixels: int) -> None:
    if width * height > max_pixels:
        raise PixelforgeError(
            f"{name}: {width}x{height} = {width * height} pixels exceeds the "
            f"input budget of {max_pixels} pixels (--max-input-pixels)")


def from_rgba(rgba: np.ndarray, sha256: str = "") -> Loaded:
    rgba = np.asarray(rgba, dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    return Loaded(rgb=rgb, alpha=rgba[..., 3].copy(), sha256=sha256)


def indices_to_rgba(indices: np.ndarray, palette_rgb8: np.ndarray) -> np.ndarray:
    """Index image (-1 = transparent) + palette -> (H, W, 4) uint8 RGBA."""
    indices = np.asarray(indices)
    opaque = indices >= 0
    out = np.zeros(indices.shape + (4,), dtype=np.uint8)
    out[opaque, :3] = np.asarray(palette_rgb8, dtype=np.uint8)[indices[opaque]]
    out[opaque, 3] = 255
    return out


def upscale_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


def save_png(path, indices: np.ndarray, palette_rgb8: np.ndarray, text: dict | None = None,
             scale: int = 1, transparent_index: str = "last") -> None:
    """Write an index image as PNG: palette-indexed if it fits in 256 entries, else RGBA.

    ``transparent_index="last"`` appends a transparent entry after the K colors, only when
    some pixel is transparent. ``"first"`` always puts it at index 0 and shifts every color
    up by one, so every PNG written with the same palette has the same PLTE.
    """
    indices = np.asarray(indices)
    palette_rgb8 = np.asarray(palette_rgb8, dtype=np.uint8).reshape(-1, 3)
    if scale > 1:
        indices = upscale_nearest(indices, scale)
    info = PngInfo()
    for key in sorted(text or {}):
        info.add_text(key, text[key])
    first = transparent_index == "first"
    has_transparency = first or bool((indices < 0).any())
    n_entries = len(palette_rgb8) + (1 if has_transparency else 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if n_entries <= 256:
        flat = palette_rgb8.reshape(-1).tolist()
        if first:
            clear = 0
            data = (indices + 1).astype(np.uint8)          # -1 (transparent) becomes 0
            flat = [0, 0, 0] + flat
        else:
            clear = len(palette_rgb8)
            data = np.where(indices < 0, clear, indices).astype(np.uint8)
            if has_transparency:
                flat += [0, 0, 0]
        im = Image.fromarray(data, mode="P")
        im.putpalette(flat)
        if has_transparency:
            im.save(path, format="PNG", pnginfo=info, transparency=clear)
        else:
            im.save(path, format="PNG", pnginfo=info)
    else:
        im = Image.fromarray(indices_to_rgba(indices, palette_rgb8), mode="RGBA")
        im.save(path, format="PNG", pnginfo=info)


def read_png_text(path) -> dict:
    with Image.open(path) as im:
        return dict(getattr(im, "text", {}))
