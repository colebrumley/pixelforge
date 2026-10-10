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


def _line_near_silhouette(edge: int, offset: int, n: int = 64):
    """Orange (L≈68) right of a transparent margin of `edge` px, with a 2-px dark (L≈6)
    vertical line `offset` px inside the silhouette."""
    x = np.broadcast_to(np.arange(n), (n, n))
    rgb = np.empty((n, n, 3))
    rgb[:] = np.array([240, 140, 30]) / 255
    rgb[(x >= edge + offset) & (x < edge + offset + 2)] = 20 / 255
    return color.rgb_to_lab(rgb), x >= edge


def test_line_near_silhouette_not_worse_than_box():
    # Kernels whose window the silhouette cuts used to fire the shape constraint on every
    # iteration and saturate σ, so a line 4 px inside the edge came out fainter than with box.
    lab, mask = _line_near_silhouette(edge=4, offset=4)
    config = Config(method="kopf", out_width=16, out_height=16)
    out = kopf.run(lab, mask, 16, 16, config)
    baseline = box.run(lab, mask, 16, 16, config)
    kopf_l = np.where(out.small_mask, out.small_lab[..., 0], np.inf)
    box_l = np.where(baseline.small_mask, baseline.small_lab[..., 0], np.inf)
    assert kopf_l.min() <= box_l.min() + 2
    assert (kopf_l.min(axis=1) < DARK_L).all()   # the line is visible on every row


def test_silhouette_does_not_saturate_sigma(preprocessed):
    pre, config = preprocessed("circle_alpha", method="kopf", out_width=16, out_height=16,
                               crop_to_alpha=False, prereduce_max_ratio=4)
    assert pre.mask.shape == (64, 64)
    out = kopf.run(pre.lab, pre.mask, 16, 16, config)
    assert out.stats["sigma_capped"] < 0.1 * 16 * 16
