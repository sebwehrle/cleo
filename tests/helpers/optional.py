# tests/helpers/optional.py
from __future__ import annotations

from typing import Any

import pytest


__all__ = [
    "requires",
    "requires_rasterio",
    "requires_rioxarray",
    "requires_dask",
    "requires_xrspatial",
    "requires_geopandas",
]


def requires(module_name: str, *, reason: str | None = None) -> Any:
    """
    Standardised optional dependency gate for tests.

    Usage:
        rasterio = requires("rasterio")
        rxr = requires("rioxarray")

    This returns the imported module if available; otherwise skips the test module.
    """
    return pytest.importorskip(
        module_name,
        reason=reason or f"optional dependency '{module_name}' is not installed",
    )


def requires_rasterio(*, reason: str | None = None):
    return requires("rasterio", reason=reason)


def requires_rioxarray(*, reason: str | None = None):
    # Note: importing rioxarray typically activates the .rio accessor on xarray objects.
    return requires("rioxarray", reason=reason)


def requires_dask(*, reason: str | None = None):
    return requires("dask", reason=reason)


def requires_xrspatial(*, reason: str | None = None):
    return requires("xrspatial", reason=reason)


def requires_geopandas(*, reason: str | None = None):
    return requires("geopandas", reason=reason)
