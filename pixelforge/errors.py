"""Exception types shared across modules."""

from __future__ import annotations


class PixelforgeError(RuntimeError):
    """A runtime failure that is not a config validation error."""
