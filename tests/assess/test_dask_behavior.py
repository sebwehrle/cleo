"""assess: test_dask_behavior.

Tests for cleo.assess dask detection helpers.

Contract:
- Tests must pass whether or not dask is installed.
- If dask is not installed, this module is skipped cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.assess import _is_dask_backed

dask = pytest.importorskip("dask")
import dask.array as da  # noqa: E402


def _da_numpy(*, shape: tuple[int, int] = (3, 3)) -> xr.DataArray:
    """Small numpy-backed DataArray."""
    return xr.DataArray(np.ones(shape, dtype=float), dims=("y", "x"))


def _da_dask(*, shape: tuple[int, int] = (3, 3), chunks: int | tuple[int, int] = 2) -> xr.DataArray:
    """Small dask-backed DataArray."""
    darr = da.from_array(np.ones(shape, dtype=float), chunks=chunks)
    return xr.DataArray(darr, dims=("y", "x"))


def test_is_dask_backed_detects_dask_array() -> None:
    """_is_dask_backed identifies dask-backed DataArrays."""
    assert _is_dask_backed(_da_dask()) is True
    assert _is_dask_backed(_da_numpy()) is False
