from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.assess import capacity_factor


def test_capacity_factor_allows_windspeed_reordering_but_requires_exact_labels():
    pdf = xr.DataArray(
        np.ones((3, 1, 1), dtype=np.float64),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": np.array([2.0, 1.0, 0.0]), "y": [0], "x": [0]},  # descending
    )
    wind_shear = xr.DataArray(
        np.zeros((1, 1), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0], "x": [0]},
    )

    u = np.array([0.0, 1.0, 2.0])  # ascending
    p = np.array([0.0, 0.5, 1.0])

    # Order mismatch must still work (alignment by label)
    out = capacity_factor(pdf, wind_shear, u, p, h_turbine=100)
    assert out.shape == (1, 1)

    # Missing label must raise (exact label contract)
    u_missing = np.array([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="wind_speed grid mismatch"):
        capacity_factor(pdf, wind_shear, u_missing, p, h_turbine=100)
