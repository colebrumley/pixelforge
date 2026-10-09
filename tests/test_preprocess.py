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
    assert result.indices.shape[0] in (64, 66) and result.indices.shape[1] < 50
