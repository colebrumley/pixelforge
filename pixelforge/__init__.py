"""pixelforge — deterministic conversion of raster images into 16-bit-style pixel art."""

from .config import PRESETS, Config
from .pipeline import Result, run
from .version import __version__

__all__ = ["Config", "PRESETS", "Result", "run", "__version__"]
