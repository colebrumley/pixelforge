"""Preprocessing: background removal, alpha mask, crop, denoise, LAB, output dims."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import color
from .config import PRESET_LONGEST_EDGE, Config
from .errors import ConfigError, PixelforgeError  # noqa: F401 - re-exported for compatibility
from .io import MAX_INPUT_PIXELS

# Output pixels of wrapped context added on every side in seamless mode (Section 9:
# the input is wrap-padded by 2·r input pixels, i.e. 2 output pixels).
SEAMLESS_PAD_OUT = 2


# A flat background is assumed when at least this fraction of the image's border pixels
# share one color (within key_bg_tolerance).
KEY_BG_MIN_BORDER_FRACTION = 0.9

# fit="stretch" warns when it changes the aspect ratio by more than this fraction.
STRETCH_WARN_ASPECT = 0.02


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


def outline_margin(config: Config) -> int:
    """Output pixels per side reserved inside the canvas for the outline ring.

    1 whenever an outline is drawn and tileset=False (whether or not the silhouette ends up
    touching the edge), so the final canvas is always exactly the resolved size. Tilesets
    reserve nothing: the canvas must stay a multiple of tile_size.
    """
    return 1 if config.outline != "none" and not config.tileset else 0


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
    if not config.tileset or config.canvas is not None or (
            config.out_width is not None and config.out_height is not None):
        return None
    return "stretch" if config.seamless else config.fit


def resolve_layout(width: int, height: int,
                   config: Config) -> tuple[int, int, tuple[int, int, int, int] | None]:
    """(out_width, out_height, layout); layout is None unless the image is smaller than the
    canvas (`canvas`, or fit="pad" with a tileset), else (inner width, inner height, left,
    top): the image's size and offset inside the area left by the outline margin."""
    m2 = 2 * outline_margin(config)
    if config.canvas is not None:
        cw, ch = config.canvas_size
        aw, ah = cw - m2, ch - m2
        if config.scale is not None:
            iw, ih = max(1, round(width / config.scale)), max(1, round(height / config.scale))
        if config.scale is None or iw > aw or ih > ah:
            # Fit inside, preserving aspect; the longer relative side fills the canvas exactly.
            if width * ah >= height * aw:
                iw, ih = aw, min(ah, max(1, round(aw * height / width)))
            else:
                iw, ih = min(aw, max(1, round(ah * width / height))), ah
        return cw, ch, (iw, ih, (aw - iw) // 2, (ah - ih) // 2)
    ow, oh = config.out_width, config.out_height
    fixed = None   # the axis whose size is a target: 0 = width, 1 = height
    if config.scale is not None:
        ow = max(8, round(width / config.scale))
        oh = max(8, round(height / config.scale))
    elif ow is None and oh is None:
        longest = PRESET_LONGEST_EDGE[config.preset]
        if width >= height:
            ow, oh, fixed = longest, max(8, round((longest - m2) * height / width) + m2), 0
        else:
            ow, oh, fixed = max(8, round((longest - m2) * width / height) + m2), longest, 1
    elif ow is None:
        ow, fixed = max(8, round((oh - m2) * width / height) + m2), 1
    elif oh is None:
        oh, fixed = max(8, round((ow - m2) * height / width) + m2), 0
    mode = fit_mode(config)
    if mode is None:
        return int(ow), int(oh), None
    return _fit_tiles(width, height, (int(ow), int(oh)), fixed, mode, config.tile_size)


def _fit_tiles(width: int, height: int, size: tuple[int, int], fixed: int | None, mode: str,
               t: int) -> tuple[int, int, tuple[int, int, int, int] | None]:
    """resolve_layout for a derived tileset size (outline margin is 0 with tilesets).

    The target axis (`fixed`: the given out_width/out_height, or the preset's longest edge) is
    rounded *down* to a multiple of t so it never exceeds the target, and the other axis is
    re-derived from it; with `scale` both sizes are kept. The canvas rounds each size up to a
    multiple. "pad" centers the aspect-preserving image on it with transparent pixels,
    "stretch" scales the image to the whole canvas and "crop" has run() crop the input to the
    canvas aspect first.
    """
    # DEVIATION: Section 5 step 2f rounds both dims up, which stretches the image and can exceed
    # the target edge; that is fit="stretch" now, and the default "pad" keeps the aspect.
    if t > max(size):
        raise ConfigError(
            f"tile_size ({t}) exceeds the longest output edge ({max(size)}) derived for a "
            f"{width}x{height} input; lower tile_size or raise the output size")
    inner = list(size)
    if fixed is not None:
        dims = (width, height)
        inner[fixed] = size[fixed] // t * t
        inner[1 - fixed] = max(1, round(inner[fixed] * dims[1 - fixed] / dims[fixed]))
    cw, ch = (math.ceil(v / t) * t for v in inner)
    if mode == "stretch":
        source, result = width / height, cw / ch
        if abs(result / source - 1.0) > STRETCH_WARN_ASPECT:
            warnings.warn(f"fit='stretch' changes the aspect ratio from {source:.3f} "
                          f"({width}x{height}) to {result:.3f} ({cw}x{ch}) to fit "
                          f"tile_size={t}; use fit='pad' or 'crop' to keep it",
                          UserWarning, stacklevel=3)
    if mode != "pad" or (inner[0], inner[1]) == (cw, ch):
        return cw, ch, None
    iw, ih = inner
    return cw, ch, (iw, ih, (cw - iw) // 2, (ch - ih) // 2)


def fit_crop(width: int, height: int, out_width: int, out_height: int) -> tuple[int, int, int, int]:
    """Centered (y0, y1, x0, x1) of a width×height input with the aspect of the output; the
    axis in excess is cropped, the other is kept whole."""
    if width * out_height > height * out_width:
        cw = max(1, round(height * out_width / out_height))
        x0 = (width - cw) // 2
        return 0, height, x0, x0 + cw
    ch = max(1, round(width * out_height / out_width))
    y0 = (height - ch) // 2
    return y0, y0 + ch, 0, width


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
    passes of `_unmix_fringe` then key out the anti-aliased edge between subject and backdrop.
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
    background = _unmix_fringe(rgb, background, reference, fringe)
    if background.all():
        return None
    return background


_EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=bool)
# Width in pixels of the widest anti-aliasing fringe `_unmix_fringe` expects.
FRINGE_DEPTH = 3


def _unmix_fringe(rgb: np.ndarray, background: np.ndarray, reference: np.ndarray,
                  passes: int) -> np.ndarray:
    """Grow ``background`` into the anti-aliased edge of the subject.

    Each pass looks at the opaque pixels 8-adjacent to the background and estimates how much of
    the subject they cover as ``max_c |p - bg| / max_c |fg - bg|``, where ``fg`` is the color of
    a nearby opaque pixel that is not adjacent to the background (see below). Pixels covering
    less than half are keyed out. Anti-aliasing fringes are 1-3 px wide, hence a few passes; it
    stops early once a pass keys out nothing. Regions enclosed by the subject never touch the
    background and are left alone.
    """
    background = background.copy()
    for _ in range(passes):
        edge = ndimage.binary_dilation(background, structure=_EIGHT_NEIGHBOURS) & ~background
        shallow = ~background & ~edge
        if not edge.any() or not shallow.any():
            break
        ey, ex = np.nonzero(edge)
        _, near = ndimage.distance_transform_edt(~shallow, return_indices=True)
        fg = rgb[near[0][ey, ex], near[1][ey, ex]]
        # The pixel just inside a fringe is itself partly mixed, so where the subject is thick
        # enough the reference color comes from beyond the widest fringe instead (a pixel at
        # coverage 0.4 must not be compared against one at 0.7). Thin parts keep the nearest
        # non-edge pixel rather than borrowing the color of some distant region.
        deep = ~ndimage.binary_dilation(background, structure=_EIGHT_NEIGHBOURS,
                                        iterations=FRINGE_DEPTH + 1)
        if deep.any():
            dist, far = ndimage.distance_transform_edt(~deep, return_indices=True)
            use = dist[ey, ex] <= 2 * FRINGE_DEPTH
            fg[use] = rgb[far[0][ey[use], ex[use]], far[1][ey[use], ex[use]]]
        p = rgb[ey, ex]
        mixed = np.abs(p - reference).max(axis=-1)
        full = np.abs(fg - reference).max(axis=-1)
        # A subject color indistinguishable from the backdrop gives no coverage estimate;
        # such pixels are kept.
        keyed = (full > 1e-6) & (2.0 * mixed < full)
        if not keyed.any():
            break
        background[ey[keyed], ex[keyed]] = True
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


def prereduce(rgb: np.ndarray, mask: np.ndarray, factor: int,
              weight: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact area mean over factor×factor blocks; trailing partial blocks are dropped.

    Transparent pixels first take the nearest opaque color so their arbitrary RGB does not
    bleed into edge blocks. A block is opaque when at least half of its pixels are. With
    ``weight`` its area mean (0 on transparent blocks) is returned as a third array.
    """
    h, w = (mask.shape[0] // factor) * factor, (mask.shape[1] // factor) * factor
    hb, wb = h // factor, w // factor
    filled = _fill_transparent(rgb, mask)[:h, :w]
    small = filled.reshape(hb, factor, wb, factor, 3).mean(axis=(1, 3))
    coverage = mask[:h, :w].reshape(hb, factor, wb, factor).mean(axis=(1, 3))
    small_mask = coverage >= 0.5
    if weight is None:
        return small, small_mask
    small_weight = weight[:h, :w].reshape(hb, factor, wb, factor).mean(axis=(1, 3))
    return small, small_mask, np.where(small_mask, small_weight, 0.0)


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


def _subject(rgb: np.ndarray, alpha: np.ndarray, config: Config):
    """(rgb, mask, weight, background_keyed, crop box (y0, y1, x0, x1)) before any denoising.

    ``weight`` is alpha / 255 on the mask and 0 elsewhere (see Preprocessed.weight).
    """
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
        background = flat_background(rgb, config.key_bg_tolerance, config.key_bg_fringe)
        if background is not None:
            mask = ~background
            background_keyed = True

    # Membership stays binary (mask), but semi-transparent pixels count in proportion to their
    # alpha, so an anti-aliased fringe just above alpha_threshold does not come out as a solid
    # band of its own color. The floor of 1/255 keeps every opaque pixel's weight positive
    # (alpha_threshold=0 admits alpha-0 pixels).
    weight = np.where(mask, np.maximum(alpha, 1).astype(np.float64) / 255.0, 0.0)

    box = (0, mask.shape[0], 0, mask.shape[1])
    if config.crop_to_alpha and not mask.all():
        ys, xs = np.nonzero(mask)
        y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 2, mask.shape[0])
        x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, mask.shape[1])
        box = (y0, y1, x0, x1)
    return rgb, mask, weight, background_keyed, box


def subject_bbox(rgb: np.ndarray, alpha: np.ndarray, config: Config) -> tuple[int, int, int, int]:
    """The crop box (y0, y1, x0, x1) run() would use: after keying, before denoising."""
    return _subject(rgb, alpha, config)[4]


def run(rgb: np.ndarray, alpha: np.ndarray, config: Config) -> Preprocessed:
    rgb, mask, weight, background_keyed, (y0, y1, x0, x1) = _subject(rgb, alpha, config)
    rgb, mask, weight = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1], weight[y0:y1, x0:x1]

    h, w = mask.shape
    out_w, out_h, layout = resolve_layout(w, h, config)
    if fit_mode(config) == "crop":
        # The downscalers always fill the whole target, so the cropped part of the canvas is
        # removed from the input instead (centered, to the canvas aspect).
        cy0, cy1, cx0, cx1 = fit_crop(w, h, out_w, out_h)
        rgb, mask = rgb[cy0:cy1, cx0:cx1], mask[cy0:cy1, cx0:cx1]
        weight = weight[cy0:cy1, cx0:cx1]
        h, w = mask.shape
        if not mask.any():
            raise PixelforgeError("no opaque pixels left after fit='crop'")
    margin = outline_margin(config)
    inner_w, inner_h = (layout[0], layout[1]) if layout else (out_w - 2 * margin,
                                                               out_h - 2 * margin)

    # DEVIATION: Section 5 — inputs smaller than the output are repeated by an integer factor
    # first, so that every output pixel is backed by at least one input pixel in all three
    # downscalers (otherwise gerstner would leave superpixels without pixels).
    factor = max(math.ceil(out_w / w), math.ceil(out_h / h))
    if factor > 1:
        # Checked before allocating: a 2000×1 input with a square output would otherwise be
        # repeated into hundreds of millions of float64 pixels.
        if h * w * factor * factor > MAX_INPUT_PIXELS:
            raise PixelforgeError(
                f"a {w}x{h} input is too small for a {out_w}x{out_h} output: repeating it "
                f"{factor}x per axis would exceed {MAX_INPUT_PIXELS} pixels")
        rgb = np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)
        mask = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
        weight = np.repeat(np.repeat(weight, factor, axis=0), factor, axis=1)
        h, w = mask.shape

    # DEVIATION: Section 5 — inputs far larger than the output are first box-reduced by an
    # integer factor (exact area mean), so that denoise and the downscalers run on less than
    # 2 × prereduce_max_ratio × the output resolution. Kopf's kernels span 4×4 output pixels, so
    # at these ratios the reduction is nearly lossless, and it bounds time and memory. Up to
    # factor − 1 trailing rows/columns that do not fill a block are dropped.
    reduce_by = prereduce_factor(h, w, out_h, out_w, config.prereduce_max_ratio)
    if reduce_by > 1:
        small_rgb, small_mask, small_weight = prereduce(rgb, mask, reduce_by, weight)
        if small_mask.any():     # else a sparse subject would vanish: keep full resolution
            rgb, mask, weight = small_rgb, small_mask, small_weight
            h, w = mask.shape
        else:
            reduce_by = 1

    pad_out = 0
    if config.seamless:
        pad_out = SEAMLESS_PAD_OUT
        pad_x = math.ceil(pad_out * w / inner_w)
        pad_y = math.ceil(pad_out * h / inner_h)
        rgb = np.pad(rgb, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="wrap")
        mask = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="wrap")
        weight = np.pad(weight, ((pad_y, pad_y), (pad_x, pad_x)), mode="wrap")

    rgb = denoise(_fill_transparent(rgb, mask), config)
    lab = color.rgb_to_lab(rgb)
    return Preprocessed(lab=np.ascontiguousarray(lab), mask=np.ascontiguousarray(mask),
                        out_width=out_w, out_height=out_h, pad_out=pad_out,
                        background_keyed=background_keyed, outline_margin=margin,
                        layout=layout, fit=fit_mode(config), prereduce_factor=reduce_by,
                        weight=np.ascontiguousarray(weight))
