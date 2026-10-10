import numpy as np
from scipy import ndimage

from pixelforge import Config, color
from pixelforge.downscale import box, kopf

DARK_L = 50.0   # "dark" = closer to black than to white


def test_thin_line_stays_connected(preprocessed):
    pre, config = preprocessed("line_diag", method="kopf", out_width=32, out_height=32,
                               key_bg=False)
    out = kopf.run(pre.lab, pre.mask, 32, 32, config)
    dark = out.small_lab[..., 0] < DARK_L
    # Every step of the diagonal is dark, and the dark pixels form one 8-connected line.
    assert dark[np.arange(32), np.arange(32)].all()
    _, n_components = ndimage.label(dark, structure=np.ones((3, 3)))
    assert n_components == 1
    assert dark.sum() <= 3 * 32   # a line, not a smear

    # The box baseline averages the line into the white background: nothing dark survives.
    baseline = box.run(pre.lab, pre.mask, 32, 32, config)
    assert not (baseline.small_lab[..., 0] < DARK_L).any()
    assert out.small_lab[..., 0].min() < baseline.small_lab[..., 0].min() - 30


def test_converges(preprocessed):
    pre, config = preprocessed("two_color", method="kopf", out_width=8, out_height=8)
    out = kopf.run(pre.lab, pre.mask, 8, 8, config)
    assert out.stats["converged"]
    assert out.stats["iterations"] < config.kopf_max_iters
    # The vertical edge stays sharp: 4 red columns, 4 blue columns, no blended column.
    assert color.delta_e(out.small_lab[:, :4], pre.lab[0, 0]).max() < 3
    assert color.delta_e(out.small_lab[:, 4:], pre.lab[0, -1]).max() < 3


def test_mask_respected(preprocessed):
    pre, config = preprocessed("circle_alpha", method="kopf", out_width=32, out_height=32,
                               crop_to_alpha=False)
    out = kopf.run(pre.lab, pre.mask, 32, 32, config)
    # Support of the disc at 32×32: output cells that contain at least one opaque input pixel.
    support = pre.mask.reshape(32, 8, 32, 8).any(axis=(1, 3))
    assert out.small_mask.any()
    assert not (out.small_mask & ~support).any()
    assert np.isfinite(out.small_lab).all()


def test_max_iters_is_respected(preprocessed):
    pre, config = preprocessed("noisy_gradient", method="kopf", out_width=16, out_height=16,
                               kopf_max_iters=3)
    out = kopf.run(pre.lab, pre.mask, 16, 16, config)
    assert out.stats["iterations"] == 3 and not out.stats["converged"]
    assert out.small_lab.shape == (16, 16, 3) and out.small_mask.all()


def test_solid_color_converges_quickly():
    # A non-integer ratio, where border truncation used to keep max |Δμ| above kopf_tol.
    lab = np.broadcast_to(color.rgb_to_lab(np.array([[[0.3, 0.6, 0.2]]])), (50, 50, 3)).copy()
    config = Config(method="kopf", out_width=32, out_height=32)
    out = kopf.run(lab, np.ones((50, 50), dtype=bool), 32, 32, config)
    assert out.stats["converged"] and out.stats["iterations"] < 10
    assert color.delta_e(out.small_lab, lab[0, 0]).max() < 1e-6


def test_thin_line_converges(preprocessed):
    pre, config = preprocessed("line_diag", method="kopf", out_width=32, out_height=32,
                               key_bg=False)
    out = kopf.run(pre.lab, pre.mask, 32, 32, config)
    assert out.stats["converged"] and out.stats["iterations"] < config.kopf_max_iters


def test_all_ones_weight_matches_unweighted(preprocessed):
    for name in ("line_diag", "circle_alpha"):
        pre, config = preprocessed(name, method="kopf", out_width=16, out_height=16)
        plain = kopf.run(pre.lab, pre.mask, 16, 16, config)
        ones = kopf.run(pre.lab, pre.mask, 16, 16, config, weight=np.ones(pre.mask.shape))
        assert np.array_equal(ones.small_lab, plain.small_lab)
        assert np.array_equal(ones.small_mask, plain.small_mask)


def test_low_weight_pixels_pull_kernels_less():
    # Columns alternate dark and light; with the light columns weighted down, every kernel's
    # color moves toward the dark ones while the mask is unchanged.
    lab = np.zeros((32, 32, 3))
    lab[:, 1::2, 0] = 80.0
    lab[:, 0::2, 0] = 40.0
    mask = np.ones((32, 32), dtype=bool)
    weight = np.ones((32, 32))
    weight[:, 1::2] = 0.25
    config = Config(method="kopf", denoise="none", kopf_max_iters=5)
    plain = kopf.run(lab, mask, 8, 8, config)
    out = kopf.run(lab, mask, 8, 8, config, weight=weight)
    assert np.array_equal(out.small_mask, plain.small_mask)
    assert out.small_lab[..., 0].mean() < plain.small_lab[..., 0].mean() - 1.0
