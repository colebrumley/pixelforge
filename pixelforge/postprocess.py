"""Post-processing on the index image (Section 8.3).

Order: ramps → saturation → orphans → jaggies → outline. Every pass works on the index image
`idx` (-1 = transparent) plus the LAB palette and is deterministic. The passes run in the
native library (crates/pixelforge/src/postprocess.rs), bit for bit as the numpy version did;
palette ramps (`palette.regularize_ramps`) are applied here before the native call.
"""

from __future__ import annotations

import numpy as np

from . import _native
from .palette import regularize_ramps

ORPHAN_MAX_PASSES = 5
ORPHAN_MAX_DELTA = 25.0


def _indices(idx) -> np.ndarray:
    return np.ascontiguousarray(idx, dtype=np.int64)


def _palette(palette_lab) -> np.ndarray:
    return np.ascontiguousarray(palette_lab, dtype=np.float64).reshape(-1, 3)


def apply_saturation(palette_lab: np.ndarray, beta: float) -> np.ndarray:
    out = np.array(palette_lab, dtype=np.float64)
    out[:, 1:] *= beta
    return out


def remove_orphans(idx: np.ndarray, palette_lab: np.ndarray, min_region: int = 2,
                   max_passes: int = ORPHAN_MAX_PASSES,
                   max_delta: float = ORPHAN_MAX_DELTA) -> np.ndarray:
    """Merge 8-connected regions with fewer than min_region pixels into their surroundings.

    Transparent pixels are never reassigned and never count as neighbors. A region is merged
    only if its color is within ΔE max_delta (CIE76) of the replacement, so high-contrast
    singles such as eye pixels and highlights are kept. Passes repeat until nothing changes,
    a pass reproduces an earlier index image, or max_passes is reached.
    """
    return _native.remove_orphans(_indices(idx), _palette(palette_lab), int(min_region),
                                  int(max_passes), float(max_delta))


def fix_jaggies(idx: np.ndarray) -> np.ndarray:
    """Remove "doubles" from 1-px diagonal staircases.

    A double is the corner pixel p of an L-shaped triple on a diagonal line of index A over a
    background B: p has exactly two A neighbors, one horizontal (n_h) and one vertical (n_v),
    its six other neighbors are all the same opaque index B, and the line continues with a
    diagonal 1-px step beyond both n_h and n_v. Removing p (setting it to B) leaves n_h and
    n_v connected diagonally. Anything that does not match exactly is left alone, so a clean
    45° line is unchanged.
    """
    return _native.fix_jaggies(_indices(idx))


def add_outline(idx: np.ndarray, palette_lab: np.ndarray, outline: str, darken: float = 0.55,
                allow_pad: bool = True, *, fixed_palette: bool = False,
                stats: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Set every transparent pixel with an opaque 4-neighbor to the outline color.

    The outline is drawn outside the silhouette. With `allow_pad=True` the canvas is padded by
    1 px first if any opaque pixel touches its edge; with `allow_pad=False` the size is kept
    and the ring is clipped on those edges (stats["outline_clipped"] = True).

    The outline color ("auto": the darkest entry blended toward black; or an explicit
    #rrggbb) reuses an entry with the same RGB8 value, else is appended as a new entry. With
    `fixed_palette=True` (a named hardware palette) the palette is never changed: the color,
    auto or explicit, snaps to the nearest existing entry in LAB (CIE76, ties → lowest index).
    A non-fixed palette that is already full (256 entries) has its nearest entry overwritten
    with the outline color instead. If `stats` is given, stats["outline_index"] is set to the
    index used, or None when no outline pixel was drawn.
    """
    out, palette_lab, index, clipped = _native.add_outline(
        _indices(idx), _palette(palette_lab), outline, float(darken), bool(allow_pad),
        bool(fixed_palette))
    if stats is not None:
        stats["outline_index"] = index
        stats["outline_clipped"] = clipped
    return out, palette_lab


def run(indices: np.ndarray, palette_lab: np.ndarray, config, *, saturated: bool = False,
        fixed_palette: bool = False, outline_margin: int = 0,
        stats: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply all passes. `saturated`: β was already applied (gerstner). `fixed_palette`: a
    named hardware palette is in use, whose colors are left untouched (the outline snaps to
    one of them). `outline_margin`: the caller already reserved a transparent margin for the
    outline (see preprocess.outline_margin), so the canvas is never padded here. If `stats`
    is given, stats["outline_index"] is the outline's palette index, or None when no outline
    was drawn, and stats["outline_clipped"] is True when the outline could not be drawn on an
    edge the silhouette touches (no padding allowed)."""
    palette_lab = np.array(palette_lab, dtype=np.float64)
    # DEVIATION: Section 8.3 — ramps and saturation are skipped for a named palette
    # (palette_name): hardware colors are fixed by definition and stay exactly as loaded.
    # Saturation and the remaining passes run natively.
    if not fixed_palette and config.palette_ramps:
        palette_lab = regularize_ramps(palette_lab)
    idx, palette_lab, index, clipped = _native.postprocess_run(
        _indices(indices), _palette(palette_lab), config.to_dict(), bool(saturated),
        bool(fixed_palette), int(outline_margin))
    if stats is not None:
        stats["outline_index"] = index
        stats["outline_clipped"] = clipped
    return idx, palette_lab
