import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from pixelforge import Config, preprocess
from pixelforge.pipeline import run_loaded
from pixelforge import io


def _sprite_on_white(size=96):
    rgb = np.ones((size, size, 3))
    rgb[20:76, 30:66] = (0.8, 0.2, 0.1)        # the sprite
    rgb[40:50, 40:56] = 1.0                    # a white patch enclosed by the sprite
    return rgb, np.full((size, size), 255, dtype=np.uint8)


def test_flat_background_is_keyed_but_enclosed_regions_are_kept():
    rgb, alpha = _sprite_on_white()
    pre = preprocess.run(rgb, alpha, Config(preset="sprite", denoise="none", out_height=16))
    assert pre.background_keyed
    # Cropped to the sprite's bounding box padded by 1 px; only that ring is transparent.
    assert pre.mask.shape == (58, 38)
    assert pre.mask[1:-1, 1:-1].all() and not pre.mask[0].any() and not pre.mask[:, 0].any()


def test_keying_is_off_for_backgrounds_and_when_disabled():
    rgb, alpha = _sprite_on_white()
    for config in (Config(preset="background"), Config(preset="sprite", key_bg=False)):
        pre = preprocess.run(rgb, alpha, config)
        assert not pre.background_keyed and pre.mask.all()


def test_images_without_a_flat_border_are_left_alone(fixture_path):
    for name in ("two_color", "gradient6", "noisy_gradient", "checker_tiles"):
        loaded = io.load(fixture_path(name))
        pre = preprocess.run(loaded.rgb, loaded.alpha, Config(preset="sprite"))
        assert not pre.background_keyed and pre.mask.all(), name


def test_existing_alpha_takes_precedence(fixture_path):
    loaded = io.load(fixture_path("circle_alpha"))
    pre = preprocess.run(loaded.rgb, loaded.alpha, Config(preset="sprite"))
    assert not pre.background_keyed


def test_solid_image_is_not_keyed_away():
    rgb = np.full((32, 32, 3), 0.5)
    pre = preprocess.run(rgb, np.full((32, 32), 255, dtype=np.uint8), Config(preset="sprite"))
    assert not pre.background_keyed and pre.mask.all()


def test_tolerance_and_validation():
    rgb, alpha = _sprite_on_white()
    rgb[:, :48] = 0.9                          # two-tone background, 0.1 apart
    rgb[20:76, 30:66] = (0.8, 0.2, 0.1)
    strict = preprocess.run(rgb, alpha, Config(preset="sprite", key_bg_tolerance=0.02))
    assert not strict.background_keyed
    loose = preprocess.run(rgb, alpha, Config(preset="sprite", key_bg_tolerance=0.15))
    assert loose.background_keyed
    with pytest.raises(ValueError):
        Config(key_bg_tolerance=1.5)


def test_pipeline_reports_keying_and_palette_ignores_background():
    rgb, alpha = _sprite_on_white()
    rgba = np.dstack([np.round(rgb * 255).astype(np.uint8), alpha])
    result = run_loaded(io.from_rgba(rgba, "test"), Config(preset="sprite", method="box"))
    assert result.stats["background_keyed"]
    # The background is gone: the canvas is the sprite plus its outline ring, whose corners
    # stay transparent.
    assert result.indices[0, 0] == -1 and result.indices[-1, -1] == -1
    assert (result.indices[2:-2, 2:-2] >= 0).all()
    assert result.indices.shape[0] == 64 and result.indices.shape[1] < 50


def test_resolve_dims_reserves_the_outline_margin_inside_the_canvas():
    assert preprocess.outline_margin(Config(preset="sprite")) == 1
    assert preprocess.outline_margin(Config(preset="sprite", outline="none")) == 0
    assert preprocess.outline_margin(Config(preset="sprite", tileset=True)) == 0
    sprite = Config(preset="sprite")
    # Longest edge is exactly 64; the other side keeps the aspect of the 62-px inner area.
    assert preprocess.resolve_dims(100, 50, sprite) == (64, 33)        # round(62 / 2) + 2
    assert preprocess.resolve_dims(50, 100, sprite) == (33, 64)
    assert preprocess.resolve_dims(64, 64, sprite.replace(out_width=32, out_height=32)) == \
        (32, 32)
    assert preprocess.resolve_dims(64, 48, sprite.replace(out_height=24)) == (31, 24)  # 22·4/3
    assert preprocess.resolve_dims(100, 50, sprite.replace(outline="none")) == (64, 32)
    assert preprocess.resolve_dims(1000, 10, sprite) == (64, 8)       # never below 8
    pre = preprocess.run(np.full((64, 64, 3), 0.5), np.full((64, 64), 255, dtype=np.uint8),
                         sprite.replace(out_width=32, out_height=32, key_bg=False))
    assert pre.outline_margin == 1 and (pre.out_width, pre.out_height) == (32, 32)
    assert (pre.target_width, pre.target_height) == (30, 30)


_RED = np.array([0.85, 0.0, 0.07])


def _disc_on_white(ramp=3.0, size=128, radius=40.0):
    """A red disc on white whose edge is a linear anti-aliasing ramp ``ramp`` px wide.

    Returns the RGB image and the true subject coverage of every pixel.
    """
    yy, xx = np.mgrid[:size, :size]
    d = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    if ramp > 0:
        coverage = np.clip((radius + ramp / 2 - d) / ramp, 0.0, 1.0)
    else:
        coverage = (d <= radius).astype(np.float64)
    rgb = coverage[..., None] * _RED + (1.0 - coverage[..., None])
    return rgb, coverage


def test_anti_aliased_fringe_is_keyed_out():
    rgb, coverage = _disc_on_white()
    halo = (coverage > 0) & (coverage < 0.5)
    before = preprocess.flat_background(rgb, 0.08, fringe=0)
    after = preprocess.flat_background(rgb, 0.08, fringe=3)
    assert (halo & ~before).sum() > 0.5 * halo.sum()          # the bug: most of it stays
    assert (halo & ~after).sum() <= 0.1 * halo.sum()
    assert not (after & (coverage >= 0.5)).any()               # no subject pixel is lost


def test_hard_edges_and_enclosed_regions_are_unchanged_by_unmixing():
    rgb, coverage = _disc_on_white(ramp=0)
    np.testing.assert_array_equal(preprocess.flat_background(rgb, 0.08, fringe=3),
                                  coverage == 0)
    rgb, alpha = _sprite_on_white()
    keyed = preprocess.flat_background(rgb, 0.08, fringe=3)
    assert not keyed[40:50, 40:56].any() and not keyed[20:76, 30:66].any()


def test_fringe_zero_reproduces_plain_keying():
    rgb, _ = _disc_on_white()
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    plain = preprocess.flat_background(rgb, 0.08)
    pre = preprocess.run(rgb, alpha, Config(preset="sprite", denoise="none", out_height=128,
                                            crop_to_alpha=False, key_bg_fringe=0))
    np.testing.assert_array_equal(pre.mask, ~plain)
    for bad in (-1, 9):
        with pytest.raises(ValueError):
            Config(key_bg_fringe=bad)


def test_palette_is_not_spent_on_the_halo():
    from pixelforge import color

    rgb, _ = _disc_on_white()
    rgba = np.dstack([np.round(rgb * 255).astype(np.uint8),
                      np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    red_l, red_a, red_b = color.rgb_to_lab(_RED[None, None])[0, 0]
    red_c = np.hypot(red_a, red_b)

    def pinkish(fringe):
        result = run_loaded(io.from_rgba(rgba, "disc"),
                            Config(preset="sprite", method="box", palette_size=4,
                                   key_bg_fringe=fringe))
        lab = color.rgb_to_lab(np.asarray(result.palette, dtype=np.float64)[None] / 255.0)[0]
        # Red mixed with white: lighter than the red by > 5 L and below 85 % of its chroma
        # (the dark auto-outline entry is darker, so it does not count).
        return int(((lab[:, 0] > red_l + 5)
                    & (np.hypot(lab[:, 1], lab[:, 2]) < 0.85 * red_c)).sum())

    assert pinkish(0) >= 3          # without unmixing three of four entries are pinks
    assert pinkish(3) <= 1


def test_resolve_dims_with_scale():
    sprite = Config(preset="sprite")
    assert preprocess.resolve_dims(64, 64, sprite.replace(scale=2)) == (32, 32)  # margin inside
    assert preprocess.resolve_dims(100, 50, sprite.replace(scale=0.5)) == (200, 100)
    assert preprocess.resolve_dims(30, 10, sprite.replace(scale=4)) == (8, 8)     # never below 8
    tiles = Config(preset="background", scale=3)
    assert preprocess.resolve_dims(100, 50, tiles) == (48, 32)           # ceil(33, 17) to 16


def test_resolve_dims_with_canvas():
    sprite = Config(preset="sprite", canvas="48x32")
    # The subject fits inside the 46x30 area left by the outline margin, centered.
    assert preprocess.resolve_layout(100, 50, sprite) == (48, 32, (46, 23, 0, 3))
    assert preprocess.resolve_layout(50, 100, sprite) == (48, 32, (15, 30, 15, 0))
    assert preprocess.resolve_dims(100, 50, sprite) == (48, 32)
    plain = sprite.replace(outline="none")
    assert preprocess.resolve_layout(100, 50, plain) == (48, 32, (48, 24, 0, 4))
    # With a scale the subject keeps it, and is only shrunk when it would not fit.
    assert preprocess.resolve_layout(20, 10, sprite.replace(scale=1)) == (48, 32, (20, 10, 13, 10))
    assert preprocess.resolve_layout(200, 10, sprite.replace(scale=1)) == (48, 32, (46, 2, 0, 14))
    pre = preprocess.run(np.full((64, 64, 3), 0.5), np.full((64, 64), 255, dtype=np.uint8),
                         sprite.replace(key_bg=False))
    assert (pre.inner_width, pre.inner_height, pre.target_width) == (30, 30, 30)
    assert pre.canvas_pad() == ((1, 1), (9, 9))


def test_tileset_fit_pad_keeps_aspect_and_pads_centered():
    pad = Config(preset="background", out_width=64)
    assert pad.fit == "pad"
    assert preprocess.resolve_layout(160, 96, pad) == (64, 48, (64, 38, 0, 5))
    # Default 256 longest edge: content 256x171 (rounded), canvas padded up to 176.
    assert preprocess.resolve_layout(300, 200, Config(preset="background")) == \
        (256, 176, (256, 171, 0, 2))
    assert preprocess.resolve_layout(200, 300, Config(preset="background")) == \
        (176, 256, (171, 256, 2, 0))
    assert preprocess.fit_mode(Config(preset="background", out_width=64, out_height=48)) is None
    assert preprocess.fit_mode(Config(preset="sprite")) is None


@pytest.mark.filterwarnings("ignore:fit='stretch'")
def test_tileset_fit_stretch_and_crop_fill_the_canvas():
    for fit in ("stretch", "crop"):
        config = Config(preset="background", out_width=64, fit=fit)
        assert preprocess.resolve_layout(160, 96, config) == (64, 48, None)
    assert preprocess.fit_crop(160, 96, 64, 48) == (0, 96, 16, 144)
    assert preprocess.fit_crop(96, 160, 64, 48) == (44, 116, 0, 96)
    with pytest.raises(ValueError, match="fit"):
        Config(fit="zoom")


@pytest.mark.filterwarnings("ignore:fit='stretch'")
def test_tileset_longest_edge_never_exceeds_the_target():
    for fit in ("pad", "stretch", "crop"):
        config = Config(preset="background", tile_size=24, fit=fit)
        out_w, out_h, _ = preprocess.resolve_layout(300, 200, config)
        assert (out_w, out_h) == (240, 168) and max(out_w, out_h) <= 256
    with pytest.raises(ValueError, match="tile_size"):
        preprocess.resolve_layout(300, 200, Config(preset="background", tile_size=1000))


def test_stretch_warns_about_aspect_change_and_pad_does_not():
    import warnings
    with pytest.warns(UserWarning, match="aspect ratio from 1.667"):
        preprocess.resolve_layout(160, 96, Config(preset="background", out_width=64,
                                                  fit="stretch"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fit in ("pad", "crop"):
            preprocess.resolve_layout(160, 96, Config(preset="background", out_width=64, fit=fit))
        # Within 2 %: no warning.
        preprocess.resolve_layout(128, 95, Config(preset="background", out_width=64,
                                                 fit="stretch"))


def _reduce_config(**overrides):
    kw = dict(preset="sprite", key_bg=False, crop_to_alpha=False, denoise="none",
              out_width=16, out_height=16, prereduce_max_ratio=8)
    kw.update(overrides)
    return Config(**kw)


def test_prereduce_factor_and_disable():
    rgb = np.full((256, 256, 3), 0.25)
    alpha = np.full((256, 256), 255, dtype=np.uint8)
    pre = preprocess.run(rgb, alpha, _reduce_config())
    assert pre.prereduce_factor == 2 and pre.mask.shape == (128, 128)
    off = preprocess.run(rgb, alpha, _reduce_config(prereduce_max_ratio=0))
    assert off.prereduce_factor == 1 and off.mask.shape == (256, 256)
    # Below 2 × ratio × output on either axis there is nothing to gain.
    narrow = preprocess.run(rgb[:, :250], alpha[:, :250], _reduce_config())
    assert narrow.prereduce_factor == 1
    with pytest.raises(ValueError):
        Config(prereduce_max_ratio=-1)


def test_prereduce_block_mean_is_exact_and_remainder_is_trimmed():
    rgb = (np.arange(5 * 7 * 3).reshape(5, 7, 3) % 16) / 16.0   # dyadic: exact in float64
    small, mask = preprocess.prereduce(rgb, np.ones((5, 7), dtype=bool), 2)
    assert small.shape == (2, 3, 3) and mask.all()
    expected = (rgb[0:4:2, 0:6:2] + rgb[1:4:2, 0:6:2] + rgb[0:4:2, 1:6:2]
                + rgb[1:4:2, 1:6:2]) / 4.0
    assert np.array_equal(small, expected)


def test_prereduce_mask_threshold_and_transparent_fill_does_not_bleed():
    rgb = np.zeros((4, 8, 3))                  # transparent pixels are black ...
    rgb[:, :3] = (0.8, 0.2, 0.1)               # ... the opaque part is red
    mask = np.zeros((4, 8), dtype=bool)
    mask[:, :3] = True                         # columns 0-2 opaque
    mask[0, 4] = True                          # one opaque pixel in block (0, 2)
    rgb[0, 4] = (0.8, 0.2, 0.1)
    small, small_mask = preprocess.prereduce(rgb, mask, 2)
    # Block columns: 0 fully opaque, 1 half opaque (kept), 2 a quarter (dropped), 3 empty.
    assert small_mask.tolist() == [[True, True, False, False], [True, True, False, False]]
    assert np.array_equal(small[:, :2], np.broadcast_to((0.8, 0.2, 0.1), (2, 2, 3)))


def test_pipeline_reports_prereduce_factor():
    rgba = np.full((256, 256, 4), 255, dtype=np.uint8)
    rgba[:, :128, :3] = 40
    result = run_loaded(io.from_rgba(rgba, "test"), _reduce_config(method="box"))
    assert result.stats["prereduce_factor"] == 2


def test_opaque_input_has_unit_weight():
    rgb = np.random.default_rng(0).random((40, 24, 3))
    pre = preprocess.run(rgb, np.full((40, 24), 255, dtype=np.uint8),
                         Config(out_width=16, out_height=16, key_bg=False, seamless=True))
    assert pre.weight.dtype == np.float64 and pre.weight.shape == pre.mask.shape
    assert np.array_equal(pre.weight, np.ones(pre.mask.shape))


def test_weight_is_alpha_on_the_mask_through_repeat_and_prereduce():
    rgb = np.full((8, 8, 3), 0.5)
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    alpha[:, 6:] = 200
    alpha[:, 7] = 100                          # below alpha_threshold: transparent
    pre = preprocess.run(rgb, alpha, Config(out_width=16, out_height=16, denoise="none",
                                            crop_to_alpha=False, outline="none"))
    assert pre.mask.shape == (16, 16)          # repeated 2x
    assert np.array_equal(pre.weight[:, :12], np.ones((16, 12)))
    assert np.allclose(pre.weight[:, 12:14], 200 / 255) and not pre.weight[:, 14:].any()
    assert not pre.weight[~pre.mask].any()
    big = np.repeat(np.repeat(alpha, 32, axis=0), 32, axis=1)
    reduced = preprocess.run(np.full((256, 256, 3), 0.5), big, _reduce_config())
    assert reduced.prereduce_factor == 2
    assert np.array_equal(reduced.weight[:, :96], np.ones((128, 96)))
    assert np.allclose(reduced.weight[:, 96:112], 200 / 255)
    assert not reduced.weight[:, 112:].any()


def test_prereduce_weight_is_the_block_mean():
    mask = np.ones((2, 4), dtype=bool)
    mask[:, 3] = False
    weight = np.where(mask, 0.5, 0.0)
    weight[:, 0] = 1.0
    _, small_mask, small_weight = preprocess.prereduce(np.zeros((2, 4, 3)), mask, 2, weight)
    assert small_mask.all()
    assert np.allclose(small_weight, [[0.75, 0.25]])


def _soft_feature():
    """A mid-gray field with a faint dark dot the size of one output pixel (8 px at 64 -> 8)."""
    y, x = np.mgrid[:64, :64]
    luma = 0.6 - 0.15 * np.exp(-((y - 32) ** 2 + (x - 32) ** 2) / (2 * 3.0 ** 2))
    luma += 0.1 * (x / 63.0)                      # a gradient much larger than the feature
    return np.repeat(luma[..., None], 3, axis=-1), np.ones((64, 64), dtype=bool)


def test_enhance_off_is_identity():
    rgb, mask = _soft_feature()
    assert preprocess.enhance(rgb, mask, (8.0, 8.0), Config()) is rgb


@pytest.mark.parametrize("overrides", [dict(enhance=1.0), dict(ink=1.0),
                                       dict(enhance=1.0, ink=0.6)])
def test_enhance_deepens_a_small_dark_feature(overrides):
    rgb, mask = _soft_feature()
    out = preprocess.enhance(rgb, mask, (8.0, 8.0), Config(**overrides))
    assert out.shape == rgb.shape and out.min() >= 0.0 and out.max() <= 1.0

    def contrast(image):
        return image[4:12, 28:36].mean() - image[32, 32].mean()   # same column, off the dot

    assert contrast(out) > 1.5 * contrast(rgb)


def test_enhance_keeps_hue_and_ignores_transparent_pixels():
    rgb, mask = _soft_feature()
    rgb = rgb * np.array([1.0, 0.6, 0.4])
    mask = mask.copy()
    mask[:, :16] = False
    rgb[:, :16] = (0.0, 1.0, 0.0)                 # arbitrary color under alpha 0
    out = preprocess.enhance(rgb, mask, (8.0, 8.0), Config(enhance=1.0, ink=0.5))
    assert np.array_equal(out[:, :16], rgb[:, :16])
    inside = out[mask]
    lit = inside[:, 0] > 0.0
    assert lit.any()
    assert np.allclose(inside[lit, 1] / inside[lit, 0], 0.6)
    assert np.allclose(inside[lit, 2] / inside[lit, 0], 0.4)
    # The green under the mask did not leak into the opaque edge.
    other = rgb.copy()
    other[:, :16] = (1.0, 0.0, 1.0)
    assert np.array_equal(out[mask], preprocess.enhance(
        other, mask, (8.0, 8.0), Config(enhance=1.0, ink=0.5))[mask])


@pytest.mark.parametrize("color_", [(0.05, 0.1, 1.0), (1.0, 0.08, 0.05), (1.0, 0.75, 0.6)])
@pytest.mark.parametrize("strength", [0.5, 1.0, 4.0])
def test_enhance_does_not_clip_saturated_colors(color_, strength):
    """Brightening stops at the pixel's brightest channel: shading and hue both survive."""
    shade = np.linspace(0.2, 0.95, 64)[None, :, None] * np.ones((64, 1, 1))
    rgb = shade * np.array(color_)
    out = preprocess.enhance(rgb, np.ones((64, 64), dtype=bool), (8.0, 8.0),
                             Config(enhance=strength))
    top = int(np.argmax(color_))
    row = out[32, :, top]
    assert (np.diff(row) >= -1e-9).all() and len(np.unique(np.round(row, 3))) > 20
    lit = out[..., top] > 0.0
    assert np.allclose(out[lit] / out[lit][:, top:top + 1], np.array(color_) / max(color_))


def test_enhance_leaves_a_nearly_flat_subject_nearly_flat():
    """One 8-bit step of difference is not stretched to black and white."""
    rgb = np.full((32, 32, 3), 128 / 255)
    rgb[:, 16:] = 129 / 255
    mask = np.ones((32, 32), dtype=bool)
    out = preprocess.enhance(rgb, mask, (2.0, 2.0), Config(enhance=1.0))
    assert out.max() - out.min() < 0.05 and abs(out.mean() - rgb.mean()) < 0.02
    flat = np.full((32, 32, 3), 0.3)
    assert np.allclose(preprocess.enhance(flat, mask, (2.0, 2.0), Config(enhance=1.0)), flat)


def test_ink_does_not_depend_on_enhance():
    """A faint dark dot is inked the same with or without the levels stretch."""
    y, x = np.mgrid[:64, :64]
    luma = 0.5 + 0.05 * (x / 63.0) - 0.05 * np.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 18.0)
    rgb, mask = np.repeat(luma[..., None], 3, axis=-1), np.ones((64, 64), dtype=bool)
    touched = []
    for strength in (0.0, 0.25, 1.0):
        plain = preprocess.enhance(rgb, mask, (8.0, 8.0), Config(enhance=strength))
        inked = preprocess.enhance(rgb, mask, (8.0, 8.0), Config(enhance=strength, ink=1.0))
        assert inked[32, 32, 0] < 0.5 * plain[32, 32, 0]
        touched.append(np.abs(inked - plain).max(axis=-1) > 1e-6)
    assert np.array_equal(touched[0], touched[1]) and np.array_equal(touched[0], touched[2])


def test_enhance_weighs_pixels_by_alpha():
    """alpha_threshold=0 admits alpha-0 pixels; their RGB must barely count."""
    rng = np.random.default_rng(0)
    rgb = ndimage.gaussian_filter(rng.random((40, 40, 3)), (2, 2, 0))
    mask, weight = np.ones((40, 40), dtype=bool), np.ones((40, 40))
    weight[:, :10] = 1 / 255
    outs = []
    for hidden in ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)):
        image = rgb.copy()
        image[:, :10] = hidden
        outs.append(preprocess.enhance(image, mask, (4.0, 4.0), Config(enhance=1.0, ink=0.5),
                                       weight)[:, 10:])
    assert np.abs(outs[0] - outs[1]).max() < 0.03
    unweighted = [preprocess.enhance(np.where(np.arange(40)[None, :, None] < 10, hidden, rgb),
                                     mask, (4.0, 4.0), Config(enhance=1.0, ink=0.5))[:, 10:]
                  for hidden in ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0))]
    assert np.abs(unweighted[0] - unweighted[1]).max() > 0.1


@pytest.mark.parametrize("overrides", [dict(enhance=1.0), dict(ink=1.0),
                                       dict(enhance=1.0, enhance_radius=16.0, ink=1.0)])
def test_enhance_keeps_a_seamless_tile_periodic(overrides):
    rng = np.random.default_rng(1)
    rgb = ndimage.gaussian_filter(rng.random((48, 60, 3)), (3, 3, 0), mode="wrap")
    mask = np.ones((48, 60), dtype=bool)
    config = Config(**overrides)
    out = preprocess.enhance(rgb, mask, (6.0, 6.0), config, wrap=True)
    assert np.abs(out - rgb).max() > 0.1
    rolled = preprocess.enhance(np.roll(rgb, (7, 31), axis=(0, 1)), mask, (6.0, 6.0), config,
                                wrap=True)
    assert np.allclose(np.roll(out, (7, 31), axis=(0, 1)), rolled, atol=1e-9)
    # Without wrap the same tile is filtered as if it ended at its border.
    assert not np.allclose(out, preprocess.enhance(rgb, mask, (6.0, 6.0), config), atol=1e-3)


@pytest.mark.parametrize("overrides", [dict(enhance=1.0), dict(ink=1.0)])
def test_seamless_padding_is_the_wrapped_enhanced_image(fixture_path, overrides):
    loaded = io.load(fixture_path("seamless_src"))
    config = Config(preset="background", tileset=False, seamless=True, out_width=16,
                    out_height=16, method="box", denoise="none", dither="none", **overrides)
    pre = preprocess.run(loaded.rgb, loaded.alpha, config)
    assert pre.prereduce_factor == 1
    py = (pre.lab.shape[0] - loaded.rgb.shape[0]) // 2
    px = (pre.lab.shape[1] - loaded.rgb.shape[1]) // 2
    core = pre.lab[py:-py, px:-px]
    assert np.array_equal(np.pad(core, ((py, py), (px, px), (0, 0)), mode="wrap"), pre.lab)
    plain = preprocess.run(loaded.rgb, loaded.alpha, config.replace(enhance=0.0, ink=0.0))
    assert not np.allclose(plain.lab, pre.lab)


@pytest.mark.parametrize("sigma", [4.0, 7.3, 25.0])
def test_wide_blur_has_the_requested_width_at_constant_cost(sigma):
    impulse = np.zeros((1, 2001))
    impulse[0, 1000] = 1.0
    kernel = preprocess._gaussian(impulse, (0.1, sigma), "nearest")[0]
    x = np.arange(2001) - 1000
    assert kernel.sum() == pytest.approx(1.0)
    assert np.sqrt((kernel * x * x).sum()) == pytest.approx(sigma, rel=0.05)


def test_explicit_levels_replace_the_percentiles():
    rgb, mask = _soft_feature()
    auto = preprocess.enhance(rgb, mask, (8.0, 8.0), Config(enhance=1.0))
    fixed = preprocess.enhance(rgb, mask, (8.0, 8.0),
                               Config(enhance=1.0, enhance_black=0.0, enhance_white=1.0))
    assert not np.allclose(auto, fixed)
    # With black 0 and white 1 the stretch is the identity: only the local contrast remains.
    brighter = preprocess.enhance(rgb + 0.05, mask, (8.0, 8.0),
                                  Config(enhance=1.0, enhance_black=0.0, enhance_white=1.0))
    assert np.allclose(brighter[8:-8, 8:-8] - fixed[8:-8, 8:-8], 0.05, atol=5e-3)


def test_enhance_radius_is_in_output_pixels(fixture_path):
    """The same picture at twice the resolution gives (nearly) the same output."""
    from pixelforge import run

    small = np.asarray(Image.open(fixture_path("noisy_gradient")).convert("RGB"))
    big = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    config = Config(preset="background", method="box", out_width=32, out_height=32,
                    denoise="none", prereduce_max_ratio=0, enhance=1.5, ink=0.5, dither="none")
    a, b = run(small, config).image.astype(int), run(big, config).image.astype(int)
    assert np.abs(a - b).mean() < 2.0
    assert not np.array_equal(a, run(small, config.replace(enhance=0.0, ink=0.0)).image)
