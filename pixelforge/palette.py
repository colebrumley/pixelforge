"""Palette construction: median cut, standalone MCDA, hardware palettes, hue ramps.

Median cut, MCDA, the hue ramps, `build` and the `.hex` text run natively
(`pixelforge._native`, from crates/pixelforge/src/palette.rs and annealer.rs): median cut
bitwise identical to the former numpy version, the ramps within one ulp (libm's hypot,
atan2, cos and sin), MCDA within the annealer's replaced BLAS/LAPACK routines; the file I/O
and the parsers are still here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import _native, color, io
from .errors import PixelforgeError

PALETTE_DIR = Path(__file__).parent / "palettes"
PALETTE_SUFFIXES = (".hex", ".gpl")
MAX_PALETTE_FILE_BYTES = 64 * 1024
INLINE_PREFIX = "hex:"
_HEX_LINE = re.compile(r"#?[0-9a-fA-F]{6}")

# Boxes spanning less than this (LAB units; ΔE ≈ 2.3 is one just-noticeable difference) are not split.
MEDIAN_CUT_MIN_RANGE = 2.3

RAMP_BIN_DEGREES = 30.0
RAMP_MAX_SHIFT_DEGREES = 6.0
RAMP_ACHROMATIC_CHROMA = 8.0
# L distance from the bin's mean L at which the hue shift reaches its 6° maximum.
# DEVIATION: Section 8.1 — the spec gives the 6° maximum but no L scale for it.
RAMP_FULL_SHIFT_L = 30.0
RAMP_DARK_HUE = 270.0    # blue/purple
RAMP_LIGHT_HUE = 90.0    # yellow


# ----------------------------------------------------------------------------- clustering

def median_cut(lab_pixels: np.ndarray, K: int) -> np.ndarray:
    """Box-splitting palette in LAB. Returns at most K colors (fewer if the input has fewer).

    Boxes are split along their axis of largest range and the palette color of a box is its
    mean, as in classic median cut, but the two choices that matter are error-driven.

    DEVIATION: Section 8.1 — the spec picks the box with the most pixels and cuts it at the
    median. On an image dominated by one color that keeps halving the dominant color while
    the few pixels of every small feature (highlights, eyes, an outline) share one leftover
    box and are averaged into a color none of them has. Here the box with the largest summed
    squared error is split, at the cut that minimizes the summed squared error of the two
    halves. On six generated test images this lowered the 99th-percentile quantization error
    in every case (slime sprite: ΔE 45 → 11) and never raised the mean by more than 0.1.
    Boxes that are already a single color (range below MEDIAN_CUT_MIN_RANGE) are not split;
    otherwise the palette is spent on imperceptible variations of a flat background.
    """
    pixels = np.ascontiguousarray(lab_pixels, dtype=np.float64).reshape(-1, 3)
    return _native.median_cut(pixels, K)


def mcda(lab_pixels: np.ndarray, K: int, config) -> np.ndarray:
    """The palette half of Section 7 applied directly to pixel colors (no superpixels):
    gerstner's annealer on the pixels with a uniform prior, at most g_max_iters steps."""
    pixels = np.ascontiguousarray(lab_pixels, dtype=np.float64).reshape(-1, 3)
    return _native.mcda(pixels, int(K), config.to_dict())


# ----------------------------------------------------------------------- hardware palettes

def bundled_palettes() -> list[str]:
    return sorted(p.stem for p in PALETTE_DIR.glob("*.hex"))


def parse_hex(text: str) -> np.ndarray:
    """One 'rrggbb' or '#rrggbb' per line; ';' starts a comment. Errors name the line number
    only, never its content."""
    colors = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.split(";")[0].strip()
        if not line or line.startswith("//"):
            continue
        if not _HEX_LINE.fullmatch(line):
            raise ValueError(f"line {number}: not a hex color")
        colors.append(color.hex_to_rgb8(line))
    return np.array(colors, dtype=np.uint8).reshape(-1, 3)


def parse_gpl(text: str) -> np.ndarray:
    colors = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            rgb = [int(p) for p in parts[:3]]
            if max(rgb) > 255:
                raise ValueError(f"line {number}: color channel above 255")
            colors.append(rgb)
    return np.array(colors, dtype=np.uint8).reshape(-1, 3)


def _read_palette_text(path: Path) -> str:
    """Read a palette file: regular files of at most MAX_PALETTE_FILE_BYTES, UTF-8 text."""
    try:
        with io.open_regular(path) as f:
            data = f.read(MAX_PALETTE_FILE_BYTES + 1)
    except PixelforgeError as exc:
        raise ValueError(str(exc)) from None
    if len(data) > MAX_PALETTE_FILE_BYTES:
        raise ValueError(f"palette file {path} is larger than {MAX_PALETTE_FILE_BYTES} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"palette file {path} is not a UTF-8 text file") from None


def inline_name(palette_rgb8: np.ndarray) -> str:
    """A palette_name that carries the colors themselves: 'hex:rrggbb,rrggbb,...'."""
    lines = [color.rgb8_to_hex(c)[1:] for c in np.asarray(palette_rgb8).reshape(-1, 3)]
    return INLINE_PREFIX + ",".join(lines)


def load_palette(name: str) -> np.ndarray:
    """Load a bundled palette by name, a .hex / .gpl file by path, or an inline
    'hex:rrggbb,rrggbb,...' palette. Returns (K, 3) uint8."""
    if name.startswith(INLINE_PREFIX):
        try:
            rgb8 = parse_hex("\n".join(name[len(INLINE_PREFIX):].split(",")))
        except ValueError as exc:
            raise ValueError(f"inline palette: {exc}") from None
        if not 1 <= len(rgb8) <= 256:
            raise ValueError(f"inline palette must have between 1 and 256 colors, "
                             f"got {len(rgb8)}")
        return rgb8
    if name in bundled_palettes():
        path = PALETTE_DIR / f"{name}.hex"
    else:
        path = Path(name)
        if path.suffix.lower() not in PALETTE_SUFFIXES or not path.exists():
            raise ValueError(f"unknown palette {name!r}: not one of {bundled_palettes()} "
                             "and not a path to a .hex or .gpl file")
    text = _read_palette_text(path)
    try:
        rgb8 = parse_gpl(text) if path.suffix.lower() == ".gpl" else parse_hex(text)
    except ValueError as exc:
        raise ValueError(f"palette file {path}: {exc}") from None
    if not 1 <= len(rgb8) <= 256:
        raise ValueError(f"palette {name!r} must have between 1 and 256 colors, "
                         f"got {len(rgb8)}")
    return rgb8


def parse_hex_lines(name: str) -> list[str]:
    """The colors of a palette as lowercase 'rrggbb' strings."""
    return [color.rgb8_to_hex(c)[1:] for c in load_palette(name)]


TRANSPARENT_HEX_LINE = "; transparent"


def _rgb8(palette_rgb8) -> np.ndarray:
    """(K, 3) uint8; other integer dtypes must hold 8-bit values."""
    rgb8 = np.asarray(palette_rgb8).reshape(-1, 3)
    if rgb8.dtype != np.uint8:
        if rgb8.size and (rgb8.min() < 0 or rgb8.max() > 255):
            raise ValueError("palette colors must be 8-bit RGB values")
        rgb8 = rgb8.astype(np.uint8)
    return np.ascontiguousarray(rgb8)


def write_hex(path, palette_rgb8: np.ndarray, transparent_first: bool = False) -> None:
    """One 'rrggbb' per line. ``transparent_first`` adds a leading TRANSPARENT_HEX_LINE, a
    comment that parse_hex skips, so the file still loads as the K colors."""
    Path(path).write_text(_native.palette_hex_text(_rgb8(palette_rgb8), transparent_first))


# --------------------------------------------------------------------------------- ramps

def regularize_ramps(palette_lab: np.ndarray) -> np.ndarray:
    """Hue-shifted ramps (Section 8.1): darker colors lean blue/purple, lighter ones yellow.

    Colors are binned by hue angle (30° bins, achromatic colors separately). Within a bin a
    color's hue moves toward 270° if it is darker than the bin's mean L and toward 90° if
    lighter, by 6°·min(1, |L − mean L| / 30), never past the target. L and chroma are kept.
    """
    palette = np.ascontiguousarray(palette_lab, dtype=np.float64).reshape(-1, 3)
    return _native.regularize_ramps(palette)


# --------------------------------------------------------------------------------- build

def resolve_source(config) -> str:
    if config.palette_source != "auto":
        return config.palette_source
    return "mcda" if config.method == "gerstner" else "median_cut"


def build(lab_pixels: np.ndarray, config) -> np.ndarray:
    """Palette (K, 3) in LAB for the given pixels according to the config."""
    settings = config.to_dict()
    if config.palette_name is not None:
        # The cached lines, so the palette used is exactly the one Config.hash() covers: they
        # reach the native build as an inline palette, never re-read from a file.
        settings["palette_name"] = INLINE_PREFIX + ",".join(config.palette_hex_lines())
    elif resolve_source(config) == "mcda":
        return mcda(lab_pixels, config.palette_size, config)
    pixels = np.ascontiguousarray(lab_pixels, dtype=np.float64).reshape(-1, 3)
    return _native.palette_build(pixels, settings)
