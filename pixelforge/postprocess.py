"""Post-processing on the index image (Section 8.3).

Order: ramps → saturation → orphans → jaggies → outline. Every pass works on the index image
`idx` (-1 = transparent) plus the LAB palette and is deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import color
from .palette import regularize_ramps

_EIGHT = np.ones((3, 3), dtype=bool)
_OFFSETS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ORPHAN_MAX_PASSES = 5


def apply_saturation(palette_lab: np.ndarray, beta: float) -> np.ndarray:
    out = np.array(palette_lab, dtype=np.float64)
    out[:, 1:] *= beta
    return out


# -------------------------------------------------------------------------------- orphans

def _label_regions(idx: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected regions of equal index. 0 = transparent, regions are 1..n."""
    labels = np.zeros(idx.shape, dtype=np.int64)
    n_total = 0
    for value in np.unique(idx[idx >= 0]):
        lab, n = ndimage.label(idx == value, structure=_EIGHT)
        inside = lab > 0
        labels[inside] = lab[inside] + n_total
        n_total += n
    return labels, n_total


def _orphan_pass(idx: np.ndarray, distances: np.ndarray, min_region: int) -> bool:
    """Merge every region smaller than min_region into its most frequent neighbor index."""
    h, w = idx.shape
    labels, n_regions = _label_regions(idx)
    sizes = np.bincount(labels.ravel(), minlength=n_regions + 1)
    small = sizes < min_region
    small[0] = False
    ys, xs = np.nonzero(small[labels])
    if len(ys) == 0:
        return False
    region = labels[ys, xs]

    pair_region, pair_pos = [], []
    for dy, dx in _OFFSETS8:
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ny, nx, r = ny[inside], nx[inside], region[inside]
        neighbor_label = labels[ny, nx]
        ok = (neighbor_label > 0) & (neighbor_label != r)
        pair_region.append(r[ok])
        pair_pos.append(ny[ok] * w + nx[ok])
    pair_region = np.concatenate(pair_region)
    pair_pos = np.concatenate(pair_pos)
    if len(pair_region) == 0:
        return False

    # Each neighboring pixel counts once per region, however many region pixels touch it.
    pairs = np.unique(pair_region * (h * w) + pair_pos)
    pair_region, pair_pos = pairs // (h * w), pairs % (h * w)
    region_ids, compact = np.unique(pair_region, return_inverse=True)
    n_colors = distances.shape[0]
    counts = np.bincount(compact * n_colors + idx.ravel()[pair_pos],
                         minlength=len(region_ids) * n_colors).reshape(len(region_ids), n_colors)

    # Region's own index, looked up through any one of its pixels.
    first_pixel = np.zeros(n_regions + 1, dtype=np.int64)
    first_pixel[region[::-1]] = (ys * w + xs)[::-1]
    own = idx.ravel()[first_pixel[region_ids]]

    # Most frequent neighbor; ties → nearest in LAB to the orphan's color; then lowest index.
    is_max = counts == counts.max(axis=1, keepdims=True)
    target = np.argmin(np.where(is_max, distances[own], np.inf), axis=1)

    new_index = np.full(n_regions + 1, -1, dtype=np.int64)
    new_index[region_ids] = target
    replacement = new_index[region]
    change = replacement >= 0
    idx[ys[change], xs[change]] = replacement[change]
    return bool(change.any())


def remove_orphans(idx: np.ndarray, palette_lab: np.ndarray, min_region: int = 2,
                   max_passes: int = ORPHAN_MAX_PASSES) -> np.ndarray:
    """Merge 8-connected regions with fewer than min_region pixels into their surroundings.

    Transparent pixels are never reassigned and never count as neighbors.
    """
    idx = np.array(idx, dtype=np.int64)
    if min_region <= 1:
        return idx
    distances = color.palette_distance_matrix(palette_lab)
    for _ in range(max_passes):
        if not _orphan_pass(idx, distances, min_region):
            break
    return idx


# -------------------------------------------------------------------------------- jaggies

def fix_jaggies(idx: np.ndarray) -> np.ndarray:
    """Remove "doubles" from 1-px diagonal staircases.

    A double is the corner pixel p of an L-shaped triple on a diagonal line of index A over a
    background B: p has exactly two A neighbors, one horizontal (n_h) and one vertical (n_v),
    its six other neighbors are all the same opaque index B, and the line continues with a
    diagonal 1-px step beyond both n_h and n_v. Removing p (setting it to B) leaves n_h and
    n_v connected diagonally. Anything that does not match exactly is left alone, so a clean
    45° line is unchanged.
    """
    idx = np.array(idx, dtype=np.int64)
    h, w = idx.shape
    if h < 4 or w < 4:
        return idx
    pad = 2
    p = np.full((h + 2 * pad, w + 2 * pad), -2, dtype=np.int64)   # -2 = outside the canvas
    p[pad:pad + h, pad:pad + w] = idx

    def at(dy: int, dx: int) -> np.ndarray:
        return p[pad + dy:pad + dy + h, pad + dx:pad + dx + w]

    center = at(0, 0)
    replace_with = np.full((h, w), -1, dtype=np.int64)
    found = np.zeros((h, w), dtype=bool)
    for sy in (-1, 1):
        for sx in (-1, 1):
            others = [o for o in _OFFSETS8 if o not in ((0, sx), (sy, 0))]
            background = at(*others[0])
            match = (center >= 0) & (at(0, sx) == center) & (at(sy, 0) == center)
            match &= (background >= 0) & (background != center)
            for o in others[1:]:
                match &= at(*o) == background
            # The staircase continues diagonally beyond both arms of the L.
            match &= (at(-sy, 2 * sx) == center) & (at(2 * sy, -sx) == center)
            match &= ~found
            replace_with[match] = background[match]
            found |= match
    idx[found] = replace_with[found]
    return idx


# -------------------------------------------------------------------------------- outline

def outline_color_lab(palette_lab: np.ndarray, outline: str, darken: float) -> np.ndarray:
    if outline == "auto":
        darkest = palette_lab[int(np.argmin(palette_lab[:, 0]))]
        return darkest * (1.0 - darken)   # blend toward LAB black (0, 0, 0)
    return color.rgb8_to_lab(color.hex_to_rgb8(outline))


def add_outline(idx: np.ndarray, palette_lab: np.ndarray, outline: str, darken: float = 0.55,
                allow_pad: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Set every transparent pixel with an opaque 4-neighbor to the outline color.

    The outline is drawn outside the silhouette; the canvas is padded by 1 px first if any
    opaque pixel touches its edge.
    """
    idx = np.array(idx, dtype=np.int64)
    palette_lab = np.array(palette_lab, dtype=np.float64)
    opaque = idx >= 0
    touches_edge = opaque[0].any() or opaque[-1].any() or opaque[:, 0].any() or opaque[:, -1].any()
    if allow_pad and touches_edge:
        idx = np.pad(idx, 1, mode="constant", constant_values=-1)
        opaque = idx >= 0

    near = np.zeros_like(opaque)
    near[1:] |= opaque[:-1]
    near[:-1] |= opaque[1:]
    near[:, 1:] |= opaque[:, :-1]
    near[:, :-1] |= opaque[:, 1:]
    ring = near & ~opaque
    if not ring.any():
        return idx, palette_lab

    line_lab = outline_color_lab(palette_lab, outline, darken)
    line_rgb8 = color.lab_to_rgb8(line_lab)
    same = np.nonzero((color.lab_to_rgb8(palette_lab) == line_rgb8).all(axis=1))[0]
    if len(same):
        line_index = int(same[0])
    elif len(palette_lab) < 256:
        line_index = len(palette_lab)
        palette_lab = np.vstack([palette_lab, line_lab[None, :]])
    else:
        line_index = int(color.nearest_index(line_lab[None, :], palette_lab)[0])
        palette_lab[line_index] = line_lab
    idx[ring] = line_index
    return idx, palette_lab


# ------------------------------------------------------------------------------------ run

def run(indices: np.ndarray, palette_lab: np.ndarray, config, *, saturated: bool = False,
        fixed_palette: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Apply all passes. `saturated`: β was already applied (gerstner). `fixed_palette`: a
    named hardware palette is in use, whose colors are left untouched."""
    idx = np.array(indices, dtype=np.int64)
    palette_lab = np.array(palette_lab, dtype=np.float64)

    # DEVIATION: Section 8.3 — ramps and saturation are skipped for a named palette
    # (palette_name): hardware colors are fixed by definition and stay exactly as loaded.
    if not fixed_palette:
        if config.palette_ramps:
            palette_lab = regularize_ramps(palette_lab)
        if not saturated:
            palette_lab = apply_saturation(palette_lab, config.saturation_beta)
    if config.remove_orphans:
        idx = remove_orphans(idx, palette_lab, config.orphan_min_region)
    if config.fix_jaggies:
        idx = fix_jaggies(idx)
    if config.outline != "none":
        # DEVIATION: Section 8.3 — with tileset=True the canvas is not padded for the
        # outline, because that would break the tile_size | width, height requirement.
        idx, palette_lab = add_outline(idx, palette_lab, config.outline, config.outline_darken,
                                       allow_pad=not config.tileset)
        if config.remove_orphans:
            # The outline can create 1-px nubs.
            idx = remove_orphans(idx, palette_lab, config.orphan_min_region, max_passes=1)
    return idx, palette_lab
