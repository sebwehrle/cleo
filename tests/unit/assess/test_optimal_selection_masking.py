"""Tests for invalid-cell masking in optimization helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.economics import min_lcoe_turbine_idx, optimal_power_kw


def _sample_lcoe() -> xr.DataArray:
    data = np.array(
        [
            [[10.0, np.nan], [5.0, 2.0]],  # turbine t0
            [[3.0, np.nan], [7.0, 1.0]],  # turbine t1
        ],
        dtype=np.float64,
    )
    return xr.DataArray(
        data,
        dims=("turbine", "y", "x"),
        coords={"turbine": ["t0", "t1"], "y": [0, 1], "x": [0, 1]},
        name="lcoe",
    )


def test_min_lcoe_turbine_idx_marks_all_nan_pixels_with_nodata_index() -> None:
    lcoe = _sample_lcoe()

    got = min_lcoe_turbine_idx(lcoe=lcoe, turbine_ids=("t0", "t1"))

    expected = np.array([[1, -1], [0, 1]], dtype=np.int32)
    np.testing.assert_array_equal(got.values, expected)
    assert got.attrs.get("cleo:nodata_index") == -1


def test_optimal_power_kw_keeps_all_nan_pixels_as_nan() -> None:
    lcoe = _sample_lcoe()
    power_kw = np.array([1000.0, 2000.0], dtype=np.float64)

    got = optimal_power_kw(lcoe=lcoe, power_kw=power_kw)

    expected = np.array([[2000.0, np.nan], [1000.0, 2000.0]], dtype=np.float64)
    np.testing.assert_allclose(got.values, expected, rtol=0.0, atol=0.0, equal_nan=True)
