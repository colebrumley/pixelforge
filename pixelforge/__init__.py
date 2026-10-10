"""pixelforge — deterministic conversion of raster images into 16-bit-style pixel art."""

import logging

from .config import PRESETS, Config
from .errors import ConfigError, PixelforgeError
from .pipeline import Result, run, run_loaded
from .version import __version__

# Library convention: never configure output; the CLI (or the caller) adds a handler.
logging.getLogger("pixelforge").addHandler(logging.NullHandler())

__all__ = ["Config", "ConfigError", "PRESETS", "PixelforgeError", "Result", "run", "run_loaded",
           "__version__"]
