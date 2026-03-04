"""Cleo-specific exceptions.

This module defines project-level exceptions that wrap or abstract
third-party exceptions, keeping the public API independent of
library internals.
"""

from __future__ import annotations


class ClipNoDataInBounds(ValueError):
    """Clipping geometry does not overlap raster bounds.

    Raised when a clip operation fails because the geometry
    has no intersection with the raster data extent.
    """

    pass
