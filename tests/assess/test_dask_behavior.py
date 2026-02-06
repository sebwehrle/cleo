"""assess: test_dask_behavior.

Tests for cleo.assess dask-guard helpers.

Contract:
- Tests must pass whether or not dask is installed.
- If dask is not installed, this module is skipped cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.assess import _is_dask_backed, _require_not_dask

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


def test_require_not_dask_raises_typeerror() -> None:
    """_require_not_dask raises TypeError with clear message for dask arrays."""
    with pytest.raises(TypeError, match=r"Dask arrays are not supported"):
        _require_not_dask(_da_dask())


def test_require_not_dask_allows_numpy() -> None:
    """_require_not_dask is a no-op for numpy-backed arrays."""
    _require_not_dask(_da_numpy())


def test_require_not_dask_checks_all_inputs() -> None:
    """_require_not_dask checks all provided arrays, not just the first."""
    with pytest.raises(TypeError, match=r"Dask arrays are not supported"):
        _require_not_dask(_da_numpy(), _da_dask())
