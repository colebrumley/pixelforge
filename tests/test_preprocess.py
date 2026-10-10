import numpy as np
import pytest

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
