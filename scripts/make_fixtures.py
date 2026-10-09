"""Generate the synthetic test images in tests/fixtures deterministically (no RNG).

Usage: python scripts/make_fixtures.py [OUTDIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

FIXTURES = ["line_diag", "two_color", "gradient6", "circle_alpha", "checker_tiles",
            "noisy_gradient", "seamless_src"]


def _to_u8(values: np.ndarray) -> np.ndarray:
    return np.round(np.clip(values, 0.0, 255.0)).astype(np.uint8)


def line_diag() -> np.ndarray:
    """256×256 white, 2-px black diagonal line corner to corner."""
    y, x = np.mgrid[0:256, 0:256]
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    img[np.abs(x - y) <= 1] = 0    # perpendicular thickness 3/√2 ≈ 2 px
    return img


def two_color() -> np.ndarray:
    """128×128, left half pure red, right half pure blue."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, :64] = (255, 0, 0)
    img[:, 64:] = (0, 0, 255)
    return img


def gradient6() -> np.ndarray:
    """256×256, six vertical bands, each a smooth dark-to-light gradient of one hue."""
    y, x = np.mgrid[0:256, 0:256]
    band = np.minimum(x * 6 // 256, 5)
    hue = band / 6.0
    t = y / 255.0
    value = 0.25 + 0.75 * t
    saturation = 1.0 - 0.5 * t
    # HSV → RGB
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    u = value * (1.0 - saturation * (1.0 - f))
    choices = [(value, u, p), (q, value, p), (p, value, u), (p, q, value), (u, p, value),
               (value, p, q)]
    rgb = np.zeros((256, 256, 3))
    for i, triple in enumerate(choices):
        sel = sector == i
        for c in range(3):
            rgb[..., c][sel] = triple[c][sel]
    return _to_u8(rgb * 255.0)


def circle_alpha() -> np.ndarray:
    """256×256 RGBA: orange disc (r=100) with a 4-px feathered alpha edge and a darker eye."""
    y, x = np.mgrid[0:256, 0:256]
    cx = cy = 127.5
    dist = np.hypot(x - cx, y - cy)
    alpha = np.clip((102.0 - dist) / 4.0, 0.0, 1.0)
    eye_dist = np.hypot(x - (cx - 32.0), y - (cy - 28.0))
    eye = np.clip(20.5 - eye_dist, 0.0, 1.0)
    orange = np.array([240.0, 140.0, 30.0])
    dark = np.array([110.0, 50.0, 10.0])
    rgb = orange[None, None, :] * (1.0 - eye[..., None]) + dark[None, None, :] * eye[..., None]
    return np.dstack([_to_u8(rgb), _to_u8(alpha * 255.0)])


def checker_tiles() -> np.ndarray:
    """256×256 of 16-px tiles, each a 2×2 checker of 8-px blocks; odd tile rows are hflipped."""
    y, x = np.mgrid[0:256, 0:256]
    odd_row = (y // 16) % 2 == 1
    local_x = np.where(odd_row, 15 - (x % 16), x % 16)
    is_a = ((local_x // 8) + ((y % 16) // 8)) % 2 == 0
    img = np.empty((256, 256, 3), dtype=np.uint8)
    img[is_a] = (60, 90, 160)
    img[~is_a] = (230, 200, 90)
    return img


def noisy_gradient() -> np.ndarray:
    """256×256 horizontal gray gradient plus a deterministic high-frequency pattern."""
    y, x = np.mgrid[0:256, 0:256]
    gray = x.astype(np.float64) + (np.sin(37.0 * x) + np.cos(53.0 * y)) * 8.0
    return np.repeat(_to_u8(gray)[..., None], 3, axis=2)


def seamless_src() -> np.ndarray:
    """128×128 smooth 2-D sine color field with period 128, so it wraps."""
    y, x = np.mgrid[0:128, 0:128]
    w = 2.0 * np.pi / 128.0
    rgb = np.stack([0.5 + 0.4 * np.sin(w * x), 0.5 + 0.4 * np.sin(w * y),
                    0.5 + 0.4 * np.cos(w * (x + y))], axis=-1)
    return _to_u8(rgb * 255.0)


def main(outdir=DEFAULT_DIR) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in FIXTURES:
        data = globals()[name]()
        path = outdir / f"{name}.png"
        Image.fromarray(data, mode="RGBA" if data.shape[2] == 4 else "RGB").save(path)
        written.append(path)
    return written


if __name__ == "__main__":
    for written_path in main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR):
        print(written_path)
