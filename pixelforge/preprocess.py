"""Preprocessing: background removal, alpha mask, crop, enhance, denoise, LAB, output dims.

Runs natively (`pixelforge._native`, from crates/pixelforge/src/preprocess.rs and the
`mask`, `layout`, `enhance`, `filters` and `color` modules it assembles). The masks, keying,
distance transform, layout, repeat, pre-reduction, median filter and percentiles are bitwise
identical to the former Python version (kept as tests/preprocess_reference.py, which the
parity tests run); the Gaussian blurs of `enhance` and the bilateral denoise agree with it to
the last bit or two (see CHANGELOG.md). Only `remove_background` (rembg) still runs in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _native
from .config import Config
from .errors import ConfigError, PixelforgeError  # noqa: F401 - re-exported for compatibility

# Output pixels of wrapped context added on every side in seamless mode (Section 9:
# the input is wrap-padded by 2·r input pixels, i.e. 2 output pixels).
# DEVIATION: Section 9 — the pad is fixed in output pixels (2·rx × 2·ry input pixels).
SEAMLESS_PAD_OUT = 2


# A flat background is assumed when at least this fraction of the image's border pixels
# share one color (within key_bg_tolerance).
KEY_BG_MIN_BORDER_FRACTION = 0.9

# fit="stretch" warns when it changes the aspect ratio by more than this fraction.
STRETCH_WARN_ASPECT = 0.02

# Width in pixels of the widest anti-aliasing fringe the background keying expects.
FRINGE_DEPTH = 3

# Percentiles of the subject's brightness (largest channel) that `enhance` maps to black and
# white. Brightness, not luminance: a saturated color reaches its ceiling at a luminance far
# below 1, and stretching its luminance to "white" would clip it flat.
ENHANCE_LEVELS = (1.0, 99.0)
# The levels stretch never amplifies by more than 1 / this: a subject whose luminance range is
# narrower (a flat color with sensor noise) is stretched to part of the range only.
ENHANCE_MIN_RANGE = 0.2
# `ink` finds thin dark features as a difference of Gaussians at these sigmas (output pixels):
# a response of INK_ONSET (luminance, 0..1) starts the darkening and INK_ONSET + INK_RANGE
# gives all of it.
INK_SIGMAS = (0.5, 1.6)
INK_ONSET, INK_RANGE = 0.02, 0.04
# Blurs at least this wide (input pixels) use running means instead of a Gaussian kernel.
BLUR_BOX_SIGMA = 4.0


@dataclass
class Preprocessed:
    lab: np.ndarray          # (H, W, 3) float64 CIELAB
    mask: np.ndarray         # (H, W) bool, True = opaque
    out_width: int           # final output size (without seamless padding)
    out_height: int
    pad_out: int             # output pixels of wrap padding per side (0 unless seamless)
    background_keyed: bool = False   # a flat opaque background was made transparent
    outline_margin: int = 0  # output pixels per side reserved for the outline ring (0 or 1)
    layout: tuple[int, int, int, int] | None = None   # canvas: (inner w, inner h, left, top)
    fit: str | None = None   # fit_mode(config): how the size was reconciled with tile_size
    prereduce_factor: int = 1        # integer box pre-reduction applied to the input (1 = none)
    # (H, W) float64 in [0, 1]: how much each pixel counts in the downscalers' averages (its
    # alpha / 255 inside the mask, 0 outside); None is read as 1 over the mask.
    weight: np.ndarray | None = None

    def __post_init__(self):
        if self.weight is None:
            self.weight = self.mask.astype(np.float64)

    @property
    def inner_width(self) -> int:
        """Width the image itself is downscaled to (the outline margin is added after)."""
        if self.layout is not None:
            return self.layout[0]
        return self.out_width - 2 * self.outline_margin

    @property
    def inner_height(self) -> int:
        if self.layout is not None:
            return self.layout[1]
        return self.out_height - 2 * self.outline_margin

    def canvas_pad(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Transparent ((top, bottom), (left, right)) padding from the inner size to the canvas,
        outline margin included."""
        left, top = (self.layout[2], self.layout[3]) if self.layout is not None else (0, 0)
        left, top = left + self.outline_margin, top + self.outline_margin
        return ((top, self.out_height - self.inner_height - top),
                (left, self.out_width - self.inner_width - left))

    @property
    def target_width(self) -> int:
        return self.inner_width + 2 * self.pad_out

    @property
    def target_height(self) -> int:
        return self.inner_height + 2 * self.pad_out


def _rgb(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb, dtype=np.float64)


def _bool(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask, dtype=bool)


def _f64(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else np.ascontiguousarray(values, dtype=np.float64)


def outline_margin(config: Config) -> int:
    """Output pixels per side reserved inside the canvas for the outline ring.

    1 whenever an outline is drawn and tileset=False (whether or not the silhouette ends up
    touching the edge), so the final canvas is always exactly the resolved size. Tilesets
    reserve nothing: the canvas must stay a multiple of tile_size.
    """
    return _native.outline_margin(config.to_dict())


def resolve_dims(width: int, height: int, config: Config) -> tuple[int, int]:
    """Section 5 step 2f: final output (width, height) for an input of the given size.

    The result includes the outline margin (see outline_margin). A derived dimension keeps the
    input aspect ratio of the inner (image) area, i.e. it is computed from the given size minus
    the margin and the margin is added back; it is never below 8. With `scale` the size is
    round(input / scale) per axis (margin inside); with `canvas` it is the canvas size. With
    `tileset` a derived size is reconciled with tile_size as `fit` says (see fit_mode).
    """
    return resolve_layout(width, height, config)[:2]


def fit_mode(config: Config) -> str | None:
    """How a derived size is made a multiple of tile_size: config.fit, or "stretch" in
    seamless mode (padding or cropping would break the wrap-around); None when nothing is
    derived (tileset=False, a canvas, or both out_width and out_height given)."""
    return _native.fit_mode(config.to_dict())


def resolve_layout(width: int, height: int,
                   config: Config) -> tuple[int, int, tuple[int, int, int, int] | None]:
    """(out_width, out_height, layout); layout is None unless the image is smaller than the
    canvas (`canvas`, or fit="pad" with a tileset), else (inner width, inner height, left,
    top): the image's size and offset inside the area left by the outline margin.

    With a tileset the derived size is rounded to tile_size (see fit_mode): the target edge
    down, the canvas up; ConfigError when tile_size exceeds the longest derived edge, and a
    UserWarning when fit="stretch" changes the aspect ratio by more than STRETCH_WARN_ASPECT.
    """
    return _native.resolve_layout(width, height, config.to_dict())


def fit_crop(width: int, height: int, out_width: int, out_height: int) -> tuple[int, int, int, int]:
    """Centered (y0, y1, x0, x1) of a width×height input with the aspect of the output; the
    axis in excess is cropped, the other is kept whole."""
    return _native.fit_crop(width, height, out_width, out_height)


def remove_background(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return a new uint8 alpha channel computed by rembg."""
    try:
        from rembg import remove
    except ImportError:
        raise ConfigError(
            "--remove-bg requires the optional 'rembg' dependency; "
            "install it with: pip install 'pixelforge[bg]'") from None
    from PIL import Image

    rgba = np.dstack([np.round(rgb * 255.0).astype(np.uint8), alpha])
    out = remove(Image.fromarray(rgba, mode="RGBA"))
    return np.asarray(out.convert("RGBA"), dtype=np.uint8)[..., 3].copy()


def flat_background(rgb: np.ndarray, tolerance: float,
                    fringe: int = 0) -> np.ndarray | None:
    """Mask of a flat background connected to the image border, or None if there is none.

    The background color is the per-channel median of the border pixels. If nearly the whole
    border has that color, every pixel of that color that is 4-connected to the border is
    background; regions of the same color enclosed by the subject are kept. Up to ``fringe``
    passes then key out the anti-aliased edge between subject and backdrop (pixels covering
    less than half of the subject color, estimated against a nearby fully opaque pixel).
    """
    return _native.flat_background(_rgb(rgb), tolerance, fringe)


def prereduce_factor(h: int, w: int, out_h: int, out_w: int, max_ratio: int) -> int:
    """Largest integer box factor that keeps the input >= max_ratio × the output on both axes."""
    return _native.prereduce_factor(h, w, out_h, out_w, max_ratio)


def prereduce(rgb: np.ndarray, mask: np.ndarray, factor: int,
              weight: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact area mean over factor×factor blocks; trailing partial blocks are dropped.

    Transparent pixels first take the nearest opaque color so their arbitrary RGB does not
    bleed into edge blocks. A block is opaque when at least half of its pixels are. With
    ``weight`` its area mean (0 on transparent blocks) is returned as a third array.
    """
    return _native.prereduce(_rgb(rgb), _bool(mask), factor, _f64(weight))


def denoise(rgb: np.ndarray, config: Config) -> np.ndarray:
    """Denoise the (possibly pre-reduced) image; the bilateral sigmas are in its pixels.

    "median" is a 3×3 median per channel (edges repeated); "bilateral" is scikit-image 0.26's
    `denoise_bilateral` with edge padding, clipped to [0, 1].
    """
    if config.denoise == "none":
        return rgb
    return _native.denoise(_rgb(rgb), config.to_dict())


def enhance_levels(value: np.ndarray, config: Config) -> tuple[float, float]:
    """(black, white) brightness for the levels stretch: `enhance_black` / `enhance_white`
    where given, else the ENHANCE_LEVELS percentiles of ``value``."""
    return _native.enhance_levels(_f64(np.ravel(value)), config.to_dict())


def _gaussian(image: np.ndarray, sigma: tuple[float, float], mode: str) -> np.ndarray:
    """Gaussian blur of a 2-D array whose cost does not grow with sigma.

    From BLUR_BOX_SIGMA pixels up, each axis is filtered with three running means whose widths
    give the same variance (the usual box approximation, within a few percent of a Gaussian);
    below it the exact kernel is used. ``mode`` is "nearest" or "wrap".
    """
    return _native.enhance_gaussian(_f64(image), tuple(sigma), mode)


def enhance(rgb: np.ndarray, mask: np.ndarray, cell: tuple[float, float], config: Config,
            weight: np.ndarray | None = None, wrap: bool = False) -> np.ndarray:
    """Exaggerate features about one output pixel in size, so the downscalers keep them.

    Photos have soft gradients and no lines: a feature smaller than an output pixel (an eye, a
    mouth, a chain) is averaged into its surroundings. ``cell`` is the (y, x) number of input
    pixels per output pixel, so every radius here is in output pixels. Three steps, the first
    two scaled by ``config.enhance`` and the last by ``config.ink``:

    1. levels: the subject's brightness range (largest channel, see enhance_levels) is
       stretched to black and white, with a capped gain (1 / ENHANCE_MIN_RANGE);
    2. local contrast: an unsharp mask of radius ``enhance_radius``, which amplifies features
       up to that size and leaves larger gradients alone;
    3. ink: pixels darker than their surroundings at about one output pixel are darkened. The
       darkness is measured on the fully stretched image, whatever ``enhance`` is.

    Steps 2 and 3 work on luminance. Each pixel's RGB is scaled by one factor, so hue is kept:
    a pixel is brightened only as far as its brightest channel allows, approaching that
    ceiling smoothly instead of clipping.
    Pixels count in the blurs by ``weight`` (default: the mask), so transparent ones neither
    contribute nor change. ``wrap`` treats the image as periodic (seamless mode). Sources that
    already have outlines and a full tonal range (illustrations, pixel art) lose their
    mid-tone shading; it is off by default.
    """
    if config.enhance <= 0 and config.ink <= 0:
        return rgb
    return _native.enhance(_rgb(rgb), _bool(mask), (float(cell[0]), float(cell[1])),
                           config.to_dict(), _f64(weight), bool(wrap))


def _inputs(rgb: np.ndarray, alpha: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """The image as contiguous float64 and its alpha, replaced by rembg when remove_bg."""
    rgb = _rgb(rgb)
    alpha = np.asarray(alpha, dtype=np.uint8)
    if config.remove_bg:
        alpha = remove_background(rgb, alpha)
    return rgb, np.ascontiguousarray(alpha)


def subject_brightness(rgb: np.ndarray, alpha: np.ndarray, config: Config) -> np.ndarray:
    """Brightness (largest channel) of the subject's pixels after keying, for levels shared
    between images."""
    rgb, alpha = _inputs(rgb, alpha, config)
    return _native.subject_brightness(rgb, alpha, config.to_dict())


def subject_bbox(rgb: np.ndarray, alpha: np.ndarray, config: Config) -> tuple[int, int, int, int]:
    """The crop box (y0, y1, x0, x1) run() would use: after keying, before denoising."""
    rgb, alpha = _inputs(rgb, alpha, config)
    return _native.subject_bbox(rgb, alpha, config.to_dict())


def run(rgb: np.ndarray, alpha: np.ndarray, config: Config) -> Preprocessed:
    """Alpha mask (rembg first when remove_bg), keying, crop, output size, fit="crop", integer
    repeat of small inputs, box pre-reduction of large ones, enhance, seamless wrap padding,
    transparent fill, denoise and CIELAB: the image the downscalers work on."""
    rgb, alpha = _inputs(rgb, alpha, config)
    return Preprocessed(**_native.preprocess_run(rgb, alpha, config.to_dict()))
