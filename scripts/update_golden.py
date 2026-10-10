"""Regenerate tests/golden.json: SHA-256 of the pixels each method/preset produces.

Run this only when an output change is intentional (an algorithm fix, a new default), then
review and commit the diff together with the change.

Usage: python scripts/update_golden.py [--check]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "tests" / "golden.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures"

GOLDEN_FIXTURES = ("two_color", "circle_alpha", "gradient6")
GOLDEN_METHODS = ("box", "kopf", "gerstner")
GOLDEN_PRESETS = ("sprite", "background")


def golden_config(method: str, preset: str):
    """32 × 32 output without denoising (fast) or background keying (fixtures are synthetic)."""
    from pixelforge import Config

    return Config(preset=preset, method=method, out_width=32, out_height=32, denoise="none",
                  key_bg=False)


def pixel_digest(image: np.ndarray, indices: np.ndarray) -> str:
    """SHA-256 of the decoded RGBA pixels and the palette indices.

    Hashes arrays, not PNG bytes: the encoded bytes vary between Pillow and zlib builds.
    """
    digest = hashlib.sha256()
    for name, arr in (("rgba", np.asarray(image, dtype=np.uint8)),
                      ("indices", np.asarray(indices).astype("<i4"))):
        digest.update(f"{name} {'x'.join(map(str, arr.shape))}\n".encode("ascii"))
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def compute_entry(fixture: str, method: str, preset: str) -> dict:
    from pixelforge import run

    config = golden_config(method, preset)
    result = run(FIXTURE_DIR / f"{fixture}.png", config)
    return {"config_hash": config.hash(), "pixels_sha256": pixel_digest(result.image,
                                                                         result.indices)}


def golden_keys() -> list[str]:
    return [f"{fixture}/{method}/{preset}" for fixture in GOLDEN_FIXTURES
            for method in GOLDEN_METHODS for preset in GOLDEN_PRESETS]


def compute_all() -> dict:
    return {key: compute_entry(*key.split("/")) for key in golden_keys()}


def dumps(golden: dict) -> str:
    return json.dumps(golden, indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT))
    text = dumps(compute_all())
    if "--check" in argv:
        if GOLDEN_PATH.read_text() != text:
            print(f"{GOLDEN_PATH} is out of date", file=sys.stderr)
            return 1
        return 0
    GOLDEN_PATH.write_text(text)
    print(GOLDEN_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
