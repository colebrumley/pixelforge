"""Preprocessing: background removal, alpha mask, crop, denoise, LAB, output dims."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import color
from .config import PRESET_LONGEST_EDGE, Config

# Output pixels of wrapped context added on every side in seamless mode (Section 9:
# the input is wrap-padded by 2·r input pixels, i.e. 2 output pixels).
SEAMLESS_PAD_OUT = 2


# A flat background is assumed when at least this fraction of the image's border pixels
# share one color (within key_bg_tolerance).
KEY_BG_MIN_BORDER_FRACTION = 0.9


class PixelforgeError(RuntimeError):
    """A runtime failure that is not a config validation error."""


@dataclass
class Preprocessed:
    lab: np.ndarray          # (H, W, 3) float64 CIELAB
    mask: np.ndarray         # (H, W) bool, True = opaque
    out_width: int           # final output size (without seamless padding)
    out_height: int
    pad_out: int             # output pixels of wrap padding per side (0 unless seamless)
    background_keyed: bool = False   # a flat opaque background was made transparent
    prereduce_factor: int = 1        # integer box pre-reduction applied to the input (1 = none)

    @property
    def target_width(self) -> int:
        return self.out_width + 2 * self.pad_out

    @property
    def target_height(self) -> int:
        return self.out_height + 2 * self.pad_out


def resolve_dims(width: int, height: int, config: Config) -> tuple[int, int]:
    """Section 5 step 2f: output (width, height) for an input of the given size."""
    ow, oh = config.out_width, config.out_height
    if ow is None and oh is None:
        longest = PRESET_LONGEST_EDGE[config.preset]
        if width >= height:
            ow, oh = longest, max(8, round(longest * height / width))
        else:
            ow, oh = max(8, round(longest * width / height)), longest
    elif ow is None:
        ow = max(8, round(oh * width / height))
    elif oh is None:
        oh = max(8, round(ow * height / width))
    if config.tileset:
        t = config.tile_size
        ow, oh = math.ceil(ow / t) * t, math.ceil(oh / t) * t
    return int(ow), int(oh)


def remove_background(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return a new uint8 alpha channel computed by rembg."""
    try:
        from rembg import remove
    except ImportError:
        raise PixelforgeError(
            "--remove-bg requires the optional 'rembg' dependency; "
            "install it with: pip install 'pixelforge[bg]'") from None
    from PIL import Image

    rgba = np.dstack([np.round(rgb * 255.0).astype(np.uint8), alpha])
    out = remove(Image.fromarray(rgba, mode="RGBA"))
    return np.asarray(out.convert("RGBA"), dtype=np.uint8)[..., 3].copy()


def flat_background(rgb: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Mask of a flat background connected to the image border, or None if there is none.

    The background color is the per-channel median of the border pixels. If nearly the whole
    border has that color, every pixel of that color that is 4-connected to the border is
    background; regions of the same color enclosed by the subject are kept.
    """
    border = np.concatenate([rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]])
    reference = np.median(border, axis=0)
    border_close = np.abs(border - reference).max(axis=1) <= tolerance
    if border_close.mean() < KEY_BG_MIN_BORDER_FRACTION:
        return None
    close = np.abs(rgb - reference).max(axis=-1) <= tolerance
    labels, _ = ndimage.label(close)
    on_border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, on_border[on_border > 0])
    if background.all():
        return None
    return background


def _fill_transparent(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Give transparent pixels the color of the nearest opaque pixel.

    Their RGB is otherwise arbitrary and would bleed into the opaque edge during denoising.
    """
    if mask.all():
        return rgb
    nearest = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return rgb[nearest[0], nearest[1]]


def prereduce_factor(h: int, w: int, out_h: int, out_w: int, max_ratio: int) -> int:
    """Largest integer box factor that keeps the input >= max_ratio × the output on both axes."""
    if max_ratio <= 0:
        return 1
    return max(1, int(min(h // out_h, w // out_w) // max_ratio))


def prereduce(rgb: np.ndarray, mask: np.ndarray,
              factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact area mean over factor×factor blocks; trailing partial blocks are dropped.

    Transparent pixels first take the nearest opaque color so their arbitrary RGB does not
    bleed into edge blocks. A block is opaque when at least half of its pixels are.
    """
    h, w = (mask.shape[0] // factor) * factor, (mask.shape[1] // factor) * factor
    hb, wb = h // factor, w // factor
    filled = _fill_transparent(rgb, mask)[:h, :w]
    small = filled.reshape(hb, factor, wb, factor, 3).mean(axis=(1, 3))
    coverage = mask[:h, :w].reshape(hb, factor, wb, factor).mean(axis=(1, 3))
    return small, coverage >= 0.5


def denoise(rgb: np.ndarray, config: Config) -> np.ndarray:
    """Denoise the (possibly pre-reduced) image; the bilateral sigmas are in its pixels."""
    if config.denoise == "none":
        return rgb
    if config.denoise == "median":
        return np.stack([ndimage.median_filter(rgb[..., c], size=3, mode="nearest")
                         for c in range(3)], axis=-1)
    from skimage.restoration import denoise_bilateral

    # DEVIATION: Section 5 — mode="edge" instead of skimage's default zero padding, which
    # darkens the image border. Transparent pixels were filled with the nearest opaque color
    # by the caller for the same reason.
    out = denoise_bilateral(rgb, sigma_color=config.denoise_sigma_color,
                            sigma_spatial=config.denoise_sigma_spatial, channel_axis=-1,
                            mode="edge")
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def run(rgb: np.ndarray, alpha: np.ndarray, config: Config) -> Preprocessed:
    rgb = np.asarray(rgb, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.uint8)

    if config.remove_bg:
        alpha = remove_background(rgb, alpha)

    mask = alpha >= config.alpha_threshold
    if not mask.any():
        raise PixelforgeError("image is fully transparent after alpha thresholding")

    # Not in the spec: image generators return sprites on an opaque flat background, which
    # would otherwise take part in every statistic and swallow most of the palette.
    background_keyed = False
    if config.key_bg and mask.all():
        background = flat_background(rgb, config.key_bg_tolerance)
        if background is not None:
            mask = ~background
            background_keyed = True

    if config.crop_to_alpha and not mask.all():
        ys, xs = np.nonzero(mask)
        y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 2, mask.shape[0])
        x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, mask.shape[1])
        rgb, mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    h, w = mask.shape
    out_w, out_h = resolve_dims(w, h, config)

    # DEVIATION: Section 5 — inputs smaller than the output are repeated by an integer factor
    # first, so that every output pixel is backed by at least one input pixel in all three
    # downscalers (otherwise gerstner would leave superpixels without pixels).
    factor = max(math.ceil(out_w / w), math.ceil(out_h / h))
    if factor > 1:
        rgb = np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)
        mask = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
        h, w = mask.shape

    # DEVIATION: Section 5 — inputs far larger than the output are first box-reduced by an
    # integer factor (exact area mean), so that denoise and the downscalers run on less than
    # 2 × prereduce_max_ratio × the output resolution. Kopf's kernels span 4×4 output pixels, so
    # at these ratios the reduction is nearly lossless, and it bounds time and memory. Up to
    # factor − 1 trailing rows/columns that do not fill a block are dropped.
    reduce_by = prereduce_factor(h, w, out_h, out_w, config.prereduce_max_ratio)
    if reduce_by > 1:
        small_rgb, small_mask = prereduce(rgb, mask, reduce_by)
        if small_mask.any():     # else a sparse subject would vanish: keep full resolution
            rgb, mask = small_rgb, small_mask
            h, w = mask.shape
        else:
            reduce_by = 1

    pad_out = 0
    if config.seamless:
        pad_out = SEAMLESS_PAD_OUT
        pad_x = math.ceil(pad_out * w / out_w)
        pad_y = math.ceil(pad_out * h / out_h)
        rgb = np.pad(rgb, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="wrap")
        mask = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="wrap")

    rgb = denoise(_fill_transparent(rgb, mask), config)
    lab = color.rgb_to_lab(rgb)
    return Preprocessed(lab=np.ascontiguousarray(lab), mask=np.ascontiguousarray(mask),
                        out_width=out_w, out_height=out_h, pad_out=pad_out,
                        background_keyed=background_keyed, prereduce_factor=reduce_by)
