"""pixelforge — deterministic conversion of raster images into 16-bit-style pixel art."""

from .config import PRESETS, Config
from .errors import ConfigError, PixelforgeError
from .pipeline import Result, run, run_loaded
from .version import __version__

__all__ = ["Config", "ConfigError", "PRESETS", "PixelforgeError", "Result", "run", "run_loaded",
           "__version__"]
