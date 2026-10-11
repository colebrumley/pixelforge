"""The native post-processing (crates/pixelforge/src/postprocess.rs) reproduces the numpy
implementation it replaced, bit for bit. The numpy code is kept below as the reference."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy import ndimage

from pixelforge import Config, color, pipeline, postprocess
from pixelforge.errors import PixelforgeError
from pixelforge.palette import regularize_ramps

# ------------------------------------------------------------------ reference (numpy)

_EIGHT = np.ones((3, 3), dtype=bool)
_OFFSETS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def ref_apply_saturation(palette_lab, beta):
    out = np.array(palette_lab, dtype=np.float64)
    out[:, 1:] *= beta
    return out


def ref_label_regions(idx):
    labels = np.zeros(idx.shape, dtype=np.int64)
    n_total = 0
    for value in np.unique(idx[idx >= 0]):
        lab, n = ndimage.label(idx == value, structure=_EIGHT)
        inside = lab > 0
        labels[inside] = lab[inside] + n_total
        n_total += n
    return labels, n_total


def ref_orphan_pass(idx, distances, min_region, max_delta, frozen):
    h, w = idx.shape
    labels, n_regions = ref_label_regions(idx)
    sizes = np.bincount(labels.ravel(), minlength=n_regions + 1)
    small = sizes < min_region
    small[0] = False
    ys, xs = np.nonzero(small[labels])
    if len(ys) == 0:
        return False
    region = labels[ys, xs]
    candidate = np.ones(n_regions + 1, dtype=bool)
    candidate[labels[frozen]] = False
    pair_region, pair_pos = [], []
    for dy, dx in _OFFSETS8:
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ny, nx, r = ny[inside], nx[inside], region[inside]
        neighbor_label = labels[ny, nx]
        ok = (neighbor_label > 0) & ~small[neighbor_label] & candidate[r]
        pair_region.append(r[ok])
        pair_pos.append(ny[ok] * w + nx[ok])
    pair_region = np.concatenate(pair_region)
    pair_pos = np.concatenate(pair_pos)
    if len(pair_region) == 0:
        return False
    pairs = np.unique(pair_region * (h * w) + pair_pos)
    pair_region, pair_pos = pairs // (h * w), pairs % (h * w)
    region_ids, compact = np.unique(pair_region, return_inverse=True)
    n_colors = distances.shape[0]
    counts = np.bincount(compact * n_colors + idx.ravel()[pair_pos],
                         minlength=len(region_ids) * n_colors).reshape(len(region_ids), n_colors)
    first_pixel = np.zeros(n_regions + 1, dtype=np.int64)
    first_pixel[region[::-1]] = (ys * w + xs)[::-1]
    own = idx.ravel()[first_pixel[region_ids]]
    is_max = counts == counts.max(axis=1, keepdims=True)
    target = np.argmin(np.where(is_max, distances[own], np.inf), axis=1)
    merge = distances[own, target] < max_delta
    keep = np.zeros(n_regions + 1, dtype=bool)
    keep[region_ids[~merge]] = True
    frozen[ys[keep[region]], xs[keep[region]]] = True
    new_index = np.full(n_regions + 1, -1, dtype=np.int64)
    new_index[region_ids[merge]] = target[merge]
    replacement = new_index[region]
    change = replacement >= 0
    idx[ys[change], xs[change]] = replacement[change]
    return bool(change.any())


def ref_remove_orphans(idx, palette_lab, min_region=2, max_passes=5, max_delta=25.0):
    idx = np.array(idx, dtype=np.int64)
    if min_region <= 1:
        return idx
    distances = color.palette_distance_matrix(palette_lab)
    frozen = np.zeros(idx.shape, dtype=bool)
    seen = [hashlib.sha256(idx.tobytes()).digest()]
    for _ in range(max_passes):
        if not ref_orphan_pass(idx, distances, min_region, max_delta, frozen):
            break
        digest = hashlib.sha256(idx.tobytes()).digest()
        if digest in seen:
            break
        seen.append(digest)
    return idx


def ref_fix_jaggies(idx):
    idx = np.array(idx, dtype=np.int64)
    h, w = idx.shape
    if h < 4 or w < 4:
        return idx
    pad = 2
    p = np.full((h + 2 * pad, w + 2 * pad), -2, dtype=np.int64)
    p[pad:pad + h, pad:pad + w] = idx

    def at(dy, dx):
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
            match &= (at(-sy, 2 * sx) == center) & (at(2 * sy, -sx) == center)
            match &= ~found
            replace_with[match] = background[match]
            found |= match
    idx[found] = replace_with[found]
    return idx


def ref_outline_color_lab(palette_lab, outline, darken):
    if outline == "auto":
        darkest = palette_lab[int(np.argmin(palette_lab[:, 0]))]
        return darkest * (1.0 - darken)
    return color.rgb8_to_lab(color.hex_to_rgb8(outline))


def ref_nearest_entry(lab, palette_lab):
    return int(np.argmin(color.delta_e(palette_lab, lab[None, :])))


def ref_add_outline(idx, palette_lab, outline, darken=0.55, allow_pad=True, *,
                    fixed_palette=False, stats=None):
    idx = np.array(idx, dtype=np.int64)
    palette_lab = np.array(palette_lab, dtype=np.float64)
    opaque = idx >= 0
    touches_edge = bool(opaque[0].any() or opaque[-1].any() or opaque[:, 0].any()
                        or opaque[:, -1].any())
    if stats is not None:
        stats["outline_index"] = None
        stats["outline_clipped"] = touches_edge and not allow_pad
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
    line_lab = ref_outline_color_lab(palette_lab, outline, darken)
    if fixed_palette:
        line_index = ref_nearest_entry(line_lab, palette_lab)
    else:
        line_rgb8 = color.lab_to_rgb8(line_lab)
        same_rgb = np.nonzero((color.lab_to_rgb8(palette_lab) == line_rgb8).all(axis=1))[0]
        if len(same_rgb):
            line_index = int(same_rgb[0])
        elif len(palette_lab) < 256:
            line_index = len(palette_lab)
            palette_lab = np.vstack([palette_lab, line_lab[None, :]])
        else:
            line_index = ref_nearest_entry(line_lab, palette_lab)
            palette_lab[line_index] = line_lab
    idx[ring] = line_index
    if stats is not None:
        stats["outline_index"] = line_index
    return idx, palette_lab


def ref_run(indices, palette_lab, config, *, saturated=False, fixed_palette=False,
            outline_margin=0, stats=None):
    idx = np.array(indices, dtype=np.int64)
    palette_lab = np.array(palette_lab, dtype=np.float64)
    if not fixed_palette:
        if config.palette_ramps:
            palette_lab = regularize_ramps(palette_lab)
        if not saturated:
            palette_lab = ref_apply_saturation(palette_lab, config.saturation_beta)
    if config.remove_orphans:
        idx = ref_remove_orphans(idx, palette_lab, config.orphan_min_region,
                                 max_delta=config.orphan_max_delta)
    if config.fix_jaggies:
        idx = ref_fix_jaggies(idx)
    if stats is not None:
        stats["outline_index"] = None
        stats["outline_clipped"] = False
    if config.outline != "none":
        idx, palette_lab = ref_add_outline(idx, palette_lab, config.outline,
                                           config.outline_darken,
                                           allow_pad=not config.tileset and not outline_margin,
                                           fixed_palette=fixed_palette, stats=stats)
        if config.remove_orphans:
            idx = ref_remove_orphans(idx, palette_lab, config.orphan_min_region, max_passes=1,
                                     max_delta=config.orphan_max_delta)
    return idx, palette_lab


# ------------------------------------------------------------------------- helpers

def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def random_palette(rng, k):
    return np.column_stack([rng.uniform(0, 100, k), rng.uniform(-60, 60, k),
                            rng.uniform(-60, 60, k)])


def random_image(rng, h, w, k, transparent=0.2):
    idx = rng.integers(0, k, (h, w))
    idx[rng.random((h, w)) < transparent] = -1
    return idx.astype(np.int64)


def blobby_image(rng, h, w, k):
    """Mostly large regions with sprinkled orphans and holes."""
    idx = np.repeat(np.repeat(rng.integers(0, k, (h // 4 + 1, w // 4 + 1)), 4, 0), 4, 1)[:h, :w]
    noise = rng.random((h, w))
    idx = np.where(noise < 0.08, rng.integers(0, k, (h, w)), idx)
    idx[noise > 0.95] = -1
    return np.ascontiguousarray(idx, dtype=np.int64)


def special_images(k):
    out = [np.full((1, 1), -1, dtype=np.int64), np.zeros((1, 1), dtype=np.int64),
           np.full((5, 7), -1, dtype=np.int64), np.full((64, 64), -1, dtype=np.int64)]
    single = np.full((6, 6), -1, dtype=np.int64)
    single[2, 3] = k - 1
    out.append(single)
    out.append((np.indices((9, 8)).sum(axis=0) % 2).astype(np.int64))      # checkerboard
    n = 16
    line = np.zeros((n, n), dtype=np.int64)
    line[np.arange(n), np.arange(n)] = 1
    out += [line, np.ascontiguousarray(line[:, ::-1])]
    for dy, dx in ((0, 1), (1, 0)):                                        # injected doubles
        doubled = line.copy()
        doubled[7 + dy, 7 + dx] = 1
        out += [doubled, np.ascontiguousarray(doubled[:, ::-1])]
        clear = np.where(doubled == 0, -1, doubled)                         # over transparency
        out.append(clear)
    edge = np.full((8, 8), -1, dtype=np.int64)                             # touching edges
    edge[0:3, 0:3] = 0
    edge[5:8, 6:8] = k - 1
    out.append(edge)
    sparkle = np.zeros((7, 7), dtype=np.int64)                             # 2-px sparkle
    sparkle[3, 3], sparkle[3, 4] = 1, k - 1
    out.append(sparkle)
    return out


def check_orphans(idx, palette):
    for min_region in (1, 2, 3, 5):
        for max_passes in (1, 5):
            for max_delta in (0.0, 25.0, np.inf):
                want = ref_remove_orphans(idx, palette, min_region, max_passes, max_delta)
                got = postprocess.remove_orphans(idx, palette, min_region, max_passes, max_delta)
                assert same(got, want), (min_region, max_passes, max_delta)


def check_outline(idx, palette, outline="auto", darken=0.55):
    for allow_pad in (True, False):
        for fixed in (False, True):
            ws, gs = {}, {}
            want = ref_add_outline(idx, palette, outline, darken, allow_pad,
                                   fixed_palette=fixed, stats=ws)
            got = postprocess.add_outline(idx, palette, outline, darken, allow_pad,
                                          fixed_palette=fixed, stats=gs)
            assert same(got[0], want[0]) and same(got[1], want[1]), (allow_pad, fixed)
            assert gs == ws


# --------------------------------------------------------------------------- tests

@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_random_images(k):
    rng = np.random.default_rng(1000 + k)
    sizes = [(1, 1), (1, 5), (4, 4), (5, 3), (9, 11), (16, 16), (31, 17), (64, 64)]
    for h, w in sizes:
        palette = random_palette(rng, k)
        for idx in (random_image(rng, h, w, k), blobby_image(rng, h, w, k),
                    random_image(rng, h, w, k, transparent=0.0)):
            check_orphans(idx, palette)
            assert same(postprocess.fix_jaggies(idx), ref_fix_jaggies(idx))
            check_outline(idx, palette)
            check_outline(idx, palette, "#c82828", 0.3)


@pytest.mark.parametrize("k", [2, 3, 6])
def test_special_images(k):
    rng = np.random.default_rng(7 + k)
    palette = random_palette(rng, k)
    for idx in special_images(k):
        check_orphans(idx, palette)
        assert same(postprocess.fix_jaggies(idx), ref_fix_jaggies(idx))
        check_outline(idx, palette)
        check_outline(idx, palette, "#000000")


def test_close_colors_and_ties():
    # Equal and near-equal distances between entries: the tie rules must agree.
    palette = np.array([[50.0, 0, 0], [52.0, 0, 0], [48.0, 0, 0], [50.0, 2.0, 0],
                        [50.0, 0, 2.0], [50.0, 0, -2.0]])
    rng = np.random.default_rng(3)
    for _ in range(40):
        idx = blobby_image(rng, 24, 24, len(palette))
        check_orphans(idx, palette)


def test_full_palette():
    rng = np.random.default_rng(11)
    palette = random_palette(rng, 256)
    idx = np.full((10, 10), -1, dtype=np.int64)
    idx[2:8, 3:7] = rng.integers(0, 256, (6, 4))
    for outline in ("auto", "#102030", "#c82828"):
        check_outline(idx, palette, outline)
    # Without a match a full palette has its nearest entry overwritten.
    out, pal = ref_add_outline(idx, palette, "#c82828")
    assert len(pal) == 256 and not same(pal, palette)
    # An exact RGB8 match of the outline colour in a full palette is reused.
    with_match = palette.copy()
    with_match[100] = color.rgb8_to_lab(np.array([200, 40, 40], dtype=np.uint8))
    check_outline(idx, with_match, "#c82828")
    out, pal = ref_add_outline(idx, with_match, "#c82828")
    assert out[1, 3] == 100 and same(pal, with_match)


@pytest.mark.parametrize("preset", ["sprite", "background"])
@pytest.mark.parametrize("method", ["box", "kopf"])
@pytest.mark.parametrize("name", ["checker_tiles", "circle_alpha", "gradient6", "line_diag",
                                  "noisy_gradient", "seamless_src", "two_color"])
def test_pipeline_inputs(fixture_path, monkeypatch, name, preset, method):
    captured = []
    real = postprocess.run

    def spy(indices, palette_lab, config, **kw):
        captured.append((np.array(indices), np.array(palette_lab), config, dict(kw)))
        return real(indices, palette_lab, config, **kw)

    monkeypatch.setattr(pipeline.postprocess, "run", spy)
    size = {"out_width": 32, "out_height": 32} if preset == "background" else {"out_width": 32}
    try:
        pipeline.run(fixture_path(name), Config(preset=preset, method=method, **size))
    except PixelforgeError as exc:      # e.g. a 1-px line keyed out entirely at 32 px
        pytest.skip(str(exc))
    assert len(captured) == 1
    indices, palette_lab, config, kw = captured[0]
    kw.pop("stats", None)
    variants = [config, config.replace(palette_ramps=False),
                config.replace(outline="auto", tileset=False),
                config.replace(outline="#204060"), config.replace(remove_orphans=False)]
    for cfg in variants:
        for fixed in (False, True):
            for saturated in (False, True):
                args = dict(kw, fixed_palette=fixed, saturated=saturated)
                ws, gs = {}, {}
                want = ref_run(indices, palette_lab, cfg, stats=ws, **args)
                got = postprocess.run(indices, palette_lab, cfg, stats=gs, **args)
                assert same(got[0], want[0]) and same(got[1], want[1])
                assert gs == ws
