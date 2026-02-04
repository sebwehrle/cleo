from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.assess import compute_wind_shear_coefficient


def test_wind_shear_dask_safe_logging():
    da = pytest.importorskip("dask.array")

    u50 = xr.DataArray(da.ones((2, 2), chunks=(2, 2)), dims=("y", "x"))
    u100 = xr.DataArray(da.ones((2, 2), chunks=(2, 2)), dims=("y", "x"))

    mean = xr.concat(
        [u50.expand_dims(height=[50]), u100.expand_dims(height=[100])],
        dim="height",
        join="exact",
    ).rename("mean_wind_speed")

    self = type("S", (), {})()
    self.data = xr.Dataset({"mean_wind_speed": mean})
    self.parent = type("P", (), {"country": "AUT"})()

    compute_wind_shear_coefficient(self, chunk_size=None)

    assert "wind_shear" in self.data.data_vars
    assert self.data["wind_shear"].shape == (2, 2)
