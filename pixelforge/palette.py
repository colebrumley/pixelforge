"""Palette construction: median cut, standalone MCDA, hardware palettes, hue ramps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import color

PALETTE_DIR = Path(__file__).parent / "palettes"

# Boxes spanning less than this (LAB units; ΔE ≈ 2.3 is one just-noticeable difference) are not split.
MEDIAN_CUT_MIN_RANGE = 2.3

RAMP_BIN_DEGREES = 30.0
RAMP_MAX_SHIFT_DEGREES = 6.0
RAMP_ACHROMATIC_CHROMA = 8.0
# L distance from the bin's mean L at which the hue shift reaches its 6° maximum.
RAMP_FULL_SHIFT_L = 30.0
RAMP_DARK_HUE = 270.0    # blue/purple
RAMP_LIGHT_HUE = 90.0    # yellow


# ----------------------------------------------------------------------------- clustering

def median_cut(lab_pixels: np.ndarray, K: int) -> np.ndarray:
    """Classic median cut in LAB. Returns at most K colors (fewer if the input has fewer).

    The box with the most pixels is split along its axis of largest range at the lower
    median; pixels equal to the median value stay together so a box is never split through
    a run of identical values. Boxes that are already a single color (range below
    MEDIAN_CUT_MIN_RANGE) are left alone.
    """
    pixels = np.ascontiguousarray(lab_pixels, dtype=np.float64).reshape(-1, 3)
    if len(pixels) == 0:
        raise ValueError("median_cut needs at least one pixel")
    boxes = [pixels]
    while len(boxes) < K:
        # Largest pixel count first; ties → earliest box.
        # DEVIATION: Section 8.1 — boxes whose largest range is below MEDIAN_CUT_MIN_RANGE are
        # treated as a single color and never split. Choosing purely by pixel count would
        # otherwise spend the whole palette on imperceptible variations of a large flat
        # background and leave a small feature (a thin dark line) averaged into it.
        chosen = None
        for i in sorted(range(len(boxes)), key=lambda i: (-len(boxes[i]), i)):
            ranges = boxes[i].max(axis=0) - boxes[i].min(axis=0)
            if ranges.max() > MEDIAN_CUT_MIN_RANGE:
                chosen = (i, int(np.argmax(ranges)))   # ties → lowest axis
                break
        if chosen is None:
            break
        i, axis = chosen
        box = boxes[i]
        values = box[:, axis]
        # DEVIATION: Section 8.1 — the split is by value (<= lower median), not by position
        # in the sorted list, and stops early when no box has two distinct colors.
        median = np.sort(values, kind="stable")[(len(values) - 1) // 2]
        lower = values <= median
        if lower.all():
            lower = values < median
        boxes[i:i + 1] = [box[lower], box[~lower]]
    palette = np.array([b.mean(axis=0) for b in boxes])
    order = np.lexsort((palette[:, 2], palette[:, 1], palette[:, 0]))
    return palette[order]


def mcda(lab_pixels: np.ndarray, K: int, config) -> np.ndarray:
    """The palette half of Section 7 applied directly to pixel colors (no superpixels)."""
    from .downscale.gerstner import PaletteAnnealer

    pixels = np.ascontiguousarray(lab_pixels, dtype=np.float64).reshape(-1, 3)
    if len(pixels) == 0:
        raise ValueError("mcda needs at least one pixel")
    annealer = PaletteAnnealer(pixels, K, config)
    weights = np.full(len(pixels), 1.0 / len(pixels))
    for _ in range(config.g_max_iters):
        annealer.step(pixels, weights)
        if annealer.done:
            break
    return annealer.colors.copy()


# ----------------------------------------------------------------------- hardware palettes

def bundled_palettes() -> list[str]:
    return sorted(p.stem for p in PALETTE_DIR.glob("*.hex"))


def parse_hex(text: str) -> np.ndarray:
    colors = []
    for line in text.splitlines():
        line = line.split(";")[0].strip()
        if not line or line.startswith("//"):
            continue
        colors.append(color.hex_to_rgb8(line))
    return np.array(colors, dtype=np.uint8).reshape(-1, 3)


def parse_gpl(text: str) -> np.ndarray:
    colors = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            colors.append([int(p) for p in parts[:3]])
    return np.array(colors, dtype=np.uint8).reshape(-1, 3)


def load_palette(name: str) -> np.ndarray:
    """Load a bundled palette by name, or a .hex / .gpl file by path. Returns (K, 3) uint8."""
    bundled = PALETTE_DIR / f"{name}.hex"
    path = bundled if bundled.is_file() else Path(name)
    if not path.is_file():
        raise ValueError(f"unknown palette {name!r}: not one of {bundled_palettes()} "
                         "and not a path to a .hex or .gpl file")
    text = path.read_text()
    rgb8 = parse_gpl(text) if path.suffix.lower() == ".gpl" else parse_hex(text)
    if not 1 <= len(rgb8) <= 256:
        raise ValueError(f"palette {name!r} must have between 1 and 256 colors, "
                         f"got {len(rgb8)}")
    return rgb8


def parse_hex_lines(name: str) -> list[str]:
    """The colors of a palette as lowercase 'rrggbb' strings."""
    return [color.rgb8_to_hex(c)[1:] for c in load_palette(name)]


def write_hex(path, palette_rgb8: np.ndarray) -> None:
    lines = [color.rgb8_to_hex(c)[1:] for c in np.asarray(palette_rgb8).reshape(-1, 3)]
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------- ramps

def regularize_ramps(palette_lab: np.ndarray) -> np.ndarray:
    """Hue-shifted ramps (Section 8.1): darker colors lean blue/purple, lighter ones yellow.

    Colors are binned by hue angle (30° bins, achromatic colors separately). Within a bin a
    color's hue moves toward 270° if it is darker than the bin's mean L and toward 90° if
    lighter, by 6°·min(1, |L − mean L| / 30), never past the target. L and chroma are kept.
    """
    palette = np.array(palette_lab, dtype=np.float64).reshape(-1, 3)
    L, a, b = palette[:, 0], palette[:, 1], palette[:, 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    bins = np.floor(hue / RAMP_BIN_DEGREES).astype(np.int64)
    bins[chroma < RAMP_ACHROMATIC_CHROMA] = -1

    new_hue = hue.copy()
    for bin_id in np.unique(bins):
        members = np.nonzero(bins == bin_id)[0]
        offset = L[members] - L[members].mean()
        amount = RAMP_MAX_SHIFT_DEGREES * np.minimum(1.0, np.abs(offset) / RAMP_FULL_SHIFT_L)
        target = np.where(offset < 0, RAMP_DARK_HUE, RAMP_LIGHT_HUE)
        # Signed shortest rotation to the target in (-180, 180]; +180 when exactly opposite.
        to_target = (target - hue[members] + 180.0) % 360.0 - 180.0
        to_target[to_target == -180.0] = 180.0
        new_hue[members] = hue[members] + np.sign(to_target) * np.minimum(amount,
                                                                          np.abs(to_target))
    rad = np.radians(new_hue)
    out = palette.copy()
    moved = new_hue != hue
    out[moved, 1] = chroma[moved] * np.cos(rad[moved])
    out[moved, 2] = chroma[moved] * np.sin(rad[moved])
    return out


# --------------------------------------------------------------------------------- build

def resolve_source(config) -> str:
    if config.palette_source != "auto":
        return config.palette_source
    return "mcda" if config.method == "gerstner" else "median_cut"


def build(lab_pixels: np.ndarray, config) -> np.ndarray:
    """Palette (K, 3) in LAB for the given pixels according to the config."""
    if config.palette_name is not None:
        return color.rgb8_to_lab(load_palette(config.palette_name))
    if resolve_source(config) == "mcda":
        return mcda(lab_pixels, config.palette_size, config)
    return median_cut(lab_pixels, config.palette_size)
