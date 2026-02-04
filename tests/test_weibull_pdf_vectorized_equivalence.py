from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import weibull_probability_density


def test_weibull_probability_density_vectorized_oracle_small_grid():
    # Small raster
    a = xr.DataArray(
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    k = xr.DataArray(
        np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    u = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    pdf = weibull_probability_density(u, k, a)

    assert "wind_speed" in pdf.dims
    assert np.array_equal(pdf.coords["wind_speed"].values, u)

    # Oracle: u=0 -> 0 everywhere
    assert np.allclose(pdf.sel(wind_speed=0.0).values, 0.0)

    # Oracle formula check for u=2 at one cell (y=0,x=0): a=2,k=2
    # f(2) = (k/a) * (u/a)^(k-1) * exp(-(u/a)^k)
    u_val = 2.0
    a_val = 2.0
    k_val = 2.0
    expected = (k_val / a_val) * (u_val / a_val) ** (k_val - 1) * np.exp(-((u_val / a_val) ** k_val))
    got = float(pdf.sel(wind_speed=2.0).isel(y=0, x=0).values)
    assert abs(got - expected) < 1e-12
