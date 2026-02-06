"""
Test that dask-backed arrays are rejected with a clear error message.
Issue 53: D3=B - Dask unsupported => clear early error.
"""
import numpy as np
import pytest
import xarray as xr

dask = pytest.importorskip("dask")
import dask.array as da

from cleo.assess import _is_dask_backed, _require_not_dask


def test_is_dask_backed_detects_dask_array():
    """_is_dask_backed correctly identifies dask-backed DataArrays."""
    # Dask-backed
    dask_arr = da.from_array(np.ones((3, 3)), chunks=2)
    dask_da = xr.DataArray(dask_arr, dims=("y", "x"))
    assert _is_dask_backed(dask_da), "Should detect dask-backed DataArray"

    # Numpy-backed
    numpy_da = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
    assert not _is_dask_backed(numpy_da), "Should not flag numpy-backed DataArray"


def test_require_not_dask_raises_typeerror():
    """_require_not_dask raises TypeError with clear message for dask arrays."""
    dask_arr = da.from_array(np.ones((3, 3)), chunks=2)
    dask_da = xr.DataArray(dask_arr, dims=("y", "x"))

    with pytest.raises(TypeError, match="Dask arrays are not supported"):
        _require_not_dask(dask_da)


def test_require_not_dask_allows_numpy():
    """_require_not_dask passes for numpy-backed arrays."""
    numpy_da = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
    # Should not raise
    _require_not_dask(numpy_da)


def test_require_not_dask_checks_all_inputs():
    """_require_not_dask checks all provided arrays."""
    numpy_da = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
    dask_arr = da.from_array(np.ones((3, 3)), chunks=2)
    dask_da = xr.DataArray(dask_arr, dims=("y", "x"))

    # First is numpy, second is dask - should still raise
    with pytest.raises(TypeError, match="Dask arrays are not supported"):
        _require_not_dask(numpy_da, dask_da)
