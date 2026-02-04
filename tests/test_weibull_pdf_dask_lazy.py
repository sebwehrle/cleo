from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.assess import weibull_probability_density


def test_weibull_pdf_is_dask_lazy_when_inputs_are_dask():
    da = pytest.importorskip("dask.array")

    a = xr.DataArray(da.ones((10, 10), chunks=(5, 5)) * 7.0, dims=("y", "x"))
    k = xr.DataArray(da.ones((10, 10), chunks=(5, 5)) * 2.0, dims=("y", "x"))
    u = np.array([0.0, 5.0, 10.0, 15.0])

    pdf = weibull_probability_density(u, k, a)

    # Should still be lazy (dask-backed)
    assert hasattr(pdf.data, "compute")
    # Shape contract
    assert pdf.sizes["wind_speed"] == len(u)
