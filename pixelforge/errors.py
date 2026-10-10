"""Exception types shared across modules."""

from __future__ import annotations


class PixelforgeError(RuntimeError):
    """A runtime failure that is not a config validation error."""


class ConfigError(PixelforgeError, ValueError):
    """An invalid configuration (bad field value, unknown palette, missing optional dependency).

    Also a ValueError, so ``except ValueError`` around ``Config(...)`` keeps working.
    """
