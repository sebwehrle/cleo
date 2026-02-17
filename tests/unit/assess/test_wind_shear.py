"""assess: test_wind_shear.

Contracts for compute_wind_shear_coefficient():
- accepts dask-backed inputs (output remains lazy)
- masks invalid cells (<=0, NaN) as NaN in output
- never produces inf
- does not emit RuntimeWarnings from log/divide on invalid inputs
- computes valid cells according to the log-law shear formula
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from tests.helpers.oracles import assert_close

from cleo.assess import compute_wind_shear_coefficient


@dataclass
class _Parent:
    country: str = "AUT"


@dataclass
class _Self:
    data: xr.Dataset
    parent: _Parent

    def _set_var(self, name, da):
        """Simple mock for _set_var - just assigns directly."""
        self.data[name] = da


def _mean_wind_speed(u50: np.ndarray, u100: np.ndarray) -> xr.DataArray:
    """
    Build mean_wind_speed(height, x) with heights exactly [50, 100].

    We keep it 1D in space to make masking assertions trivial and robust.
    """
    u50 = np.asarray(u50, dtype=float)
    u100 = np.asarray(u100, dtype=float)
    if u50.shape != u100.shape:
        raise ValueError("u50 and u100 must have same shape")

    x = np.arange(u50.size, dtype=float)
    return xr.DataArray(
        data=np.stack([u50, u100], axis=0),
        dims=("height", "x"),
        coords={"height": [50.0, 100.0], "x": x},
        name="mean_wind_speed",
    )


def _make_self(u50: np.ndarray, u100: np.ndarray) -> _Self:
    mean = _mean_wind_speed(u50, u100)
    return _Self(data=xr.Dataset({"mean_wind_speed": mean}), parent=_Parent(country="AUT"))


def test_wind_shear_accepts_dask_arrays() -> None:
    """
    Dask arrays are now supported; output should remain dask-backed (lazy).
    """
    da = pytest.importorskip("dask.array")
    from cleo.assess import _is_dask_backed

    u50 = xr.DataArray(da.ones((2, 2), chunks=(2, 2)) * 5.0, dims=("y", "x"))
    u100 = xr.DataArray(da.ones((2, 2), chunks=(2, 2)) * 10.0, dims=("y", "x"))

    mean = xr.concat(
        [u50.expand_dims(height=[50.0]), u100.expand_dims(height=[100.0])],
        dim="height",
        join="exact",
    ).rename("mean_wind_speed")

    self = _Self(data=xr.Dataset({"mean_wind_speed": mean}), parent=_Parent(country="AUT"))

    # Should NOT raise - dask is now supported
    compute_wind_shear_coefficient(self)

    # Output should be dask-backed (lazy)
    assert _is_dask_backed(self.data["wind_shear"]), "wind_shear should remain dask-backed"

    # Compute to verify correctness
    ws = self.data["wind_shear"].compute().values
    expected = float(np.log(10.0 / 5.0) / np.log(100.0 / 50.0))  # = 1.0
    np.testing.assert_allclose(ws, expected, rtol=1e-12)


def test_wind_shear_masks_zero_wind_speed() -> None:
    self = _make_self(u50=[0.0, 1.0, 2.0], u100=[2.0, 2.0, 2.0])
    compute_wind_shear_coefficient(self)

    out = self.data["wind_shear"].values
    assert np.isnan(out[0])
    assert np.isfinite(out[1])
    assert np.isfinite(out[2])


def test_wind_shear_masks_negative_wind_speed() -> None:
    self = _make_self(u50=[-1.0, 1.0], u100=[2.0, 2.0])
    compute_wind_shear_coefficient(self)

    out = self.data["wind_shear"].values
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_wind_shear_masks_nan_input() -> None:
    self = _make_self(u50=[np.nan, 1.0], u100=[2.0, 2.0])
    compute_wind_shear_coefficient(self)

    out = self.data["wind_shear"].values
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_wind_shear_never_contains_inf() -> None:
    self = _make_self(u50=[0.0, -1.0, np.nan, 1.0, 5.0], u100=[2.0, 2.0, 2.0, 2.0, 10.0])
    compute_wind_shear_coefficient(self)

    out = self.data["wind_shear"].values
    assert not np.any(np.isinf(out))


def test_wind_shear_valid_cells_match_oracle() -> None:
    """
    Oracle (log-law shear):
      alpha = (log(u100) - log(u50)) / (log(100) - log(50))
            = log(u100/u50) / log(2)
    """
    self = _make_self(u50=[5.0, 5.0], u100=[10.0, 10.0])
    compute_wind_shear_coefficient(self)

    got = self.data["wind_shear"].values
    expected = float(np.log(10.0 / 5.0) / np.log(100.0 / 50.0))  # = 1.0
    assert_close(got[0], expected, rtol=0.0, atol=1e-12)
    assert_close(got[1], expected, rtol=0.0, atol=1e-12)


def test_wind_shear_no_runtime_warnings_from_log_on_invalid_cells() -> None:
    """
    Must not emit RuntimeWarnings ("divide by zero"/"invalid value") when u50 has zeros.
    """
    self = _make_self(u50=[0.0, 5.0, 6.0], u100=[2.0, 10.0, 12.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_wind_shear_coefficient(self)

    runtime_log_warnings = [
        ww
        for ww in w
        if issubclass(ww.category, RuntimeWarning)
        and ("divide by zero" in str(ww.message) or "invalid value" in str(ww.message))
    ]
    assert runtime_log_warnings == []


def test_wind_shear_nan_on_invalid_cells_even_when_warnings_are_errors() -> None:
    self = _make_self(u50=[0.0, 5.0, 6.0], u100=[2.0, 10.0, 12.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_wind_shear_coefficient(self)

    out = self.data["wind_shear"].values
    assert np.isnan(out[0])
    assert np.isfinite(out[1])
    assert np.isfinite(out[2])
