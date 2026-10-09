"""Image load/save, alpha handling and PNG metadata embedding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

META_CONFIG_KEY = "pixelforge:config"
META_INPUT_KEY = "pixelforge:input_sha256"


@dataclass
class Loaded:
    rgb: np.ndarray        # (H, W, 3) float64 sRGB in [0, 1]
    alpha: np.ndarray      # (H, W) uint8
    sha256: str            # SHA-256 of the input file bytes

    @property
    def alpha_mask(self) -> np.ndarray:
        """Alpha as a float mask in [0, 1]."""
        return self.alpha.astype(np.float64) / 255.0


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path) -> Loaded:
    digest = sha256_file(path)
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    return from_rgba(rgba, digest)


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
             scale: int = 1) -> None:
    """Write an index image as PNG: palette-indexed if it fits in 256 entries, else RGBA."""
    indices = np.asarray(indices)
    palette_rgb8 = np.asarray(palette_rgb8, dtype=np.uint8).reshape(-1, 3)
    if scale > 1:
        indices = upscale_nearest(indices, scale)
    info = PngInfo()
    for key in sorted(text or {}):
        info.add_text(key, text[key])
    has_transparency = bool((indices < 0).any())
    n_entries = len(palette_rgb8) + (1 if has_transparency else 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if n_entries <= 256:
        transparent_index = len(palette_rgb8)
        data = np.where(indices < 0, transparent_index, indices).astype(np.uint8)
        im = Image.fromarray(data, mode="P")
        flat = palette_rgb8.reshape(-1).tolist()
        if has_transparency:
            flat += [0, 0, 0]
        im.putpalette(flat)
        if has_transparency:
            im.save(path, format="PNG", pnginfo=info, transparency=transparent_index)
        else:
            im.save(path, format="PNG", pnginfo=info)
    else:
        im = Image.fromarray(indices_to_rgba(indices, palette_rgb8), mode="RGBA")
        im.save(path, format="PNG", pnginfo=info)


def read_png_text(path) -> dict:
    with Image.open(path) as im:
        return dict(getattr(im, "text", {}))
