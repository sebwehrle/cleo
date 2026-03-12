"""economics: optimal selection helper tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from cleo.economics import min_lcoe_turbine_idx, optimal_energy_gwh_a, optimal_power_kw


def _sample_lcoe() -> xr.DataArray:
    data = np.array(
        [
            [[10.0, np.nan], [5.0, 2.0]],
            [[3.0, np.nan], [7.0, 1.0]],
        ],
        dtype=np.float64,
    )
    da = xr.DataArray(
        data,
        dims=("turbine", "y", "x"),
        coords={"turbine": ["t0", "t1"], "y": [0, 1], "x": [0, 1]},
        name="lcoe",
    )
    da.attrs["cleo:cf_method"] = "hub_height_weibull"
    da.attrs["cleo:economics_json"] = json.dumps({"discount_rate": 0.05}, ensure_ascii=True)
    da.attrs["cleo:turbine_ids_json"] = json.dumps(["t0", "t1"], ensure_ascii=True)
    return da


def _sample_cf() -> xr.DataArray:
    data = np.array(
        [
            [[0.2, np.nan], [0.3, 0.25]],
            [[0.35, np.nan], [0.28, 0.4]],
        ],
        dtype=np.float64,
    )
    return xr.DataArray(
        data,
        dims=("turbine", "y", "x"),
        coords={"turbine": ["t0", "t1"], "y": [0, 1], "x": [0, 1]},
        name="capacity_factors",
    )


def test_min_lcoe_turbine_idx_and_attrs() -> None:
    lcoe = _sample_lcoe()
    got = min_lcoe_turbine_idx(lcoe=lcoe, turbine_ids=("t0", "t1"))

    expected = np.array([[1, -1], [0, 1]], dtype=np.int32)
    np.testing.assert_array_equal(got.values, expected)
    assert got.attrs["cleo:nodata_index"] == -1
    assert json.loads(got.attrs["cleo:turbine_ids_json"]) == ["t0", "t1"]
    assert "cleo:economics_json" in got.attrs


def test_optimal_power_kw_keeps_all_nan_pixels_as_nan_and_lineage_attrs() -> None:
    lcoe = _sample_lcoe()
    power_kw = np.array([1000.0, 2000.0], dtype=np.float64)

    got = optimal_power_kw(lcoe=lcoe, power_kw=power_kw)

    expected = np.array([[2000.0, np.nan], [1000.0, 2000.0]], dtype=np.float64)
    np.testing.assert_allclose(got.values, expected, rtol=0.0, atol=0.0, equal_nan=True)
    assert got.attrs["units"] == "kW"
    assert got.attrs["cleo:selection_basis"] == "min_lcoe_turbine_idx"
    assert "cleo:economics_json" in got.attrs
    assert "cleo:turbine_ids_json" in got.attrs


def test_optimal_energy_gwh_a_uses_selected_turbine_cf_and_power() -> None:
    lcoe = _sample_lcoe()
    cf = _sample_cf()
    power_kw = np.array([1000.0, 2000.0], dtype=np.float64)

    got = optimal_energy_gwh_a(lcoe=lcoe, cf=cf, power_kw=power_kw, hours_per_year=8766.0)

    expected = np.array(
        [
            [0.35 * 2000.0 * 8766.0 / 1e6, np.nan],
            [0.3 * 1000.0 * 8766.0 / 1e6, 0.4 * 2000.0 * 8766.0 / 1e6],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got.values, expected, rtol=0.0, atol=1e-12, equal_nan=True)
    assert got.attrs["units"] == "GWh/a"
    assert got.attrs["cleo:selection_basis"] == "min_lcoe_turbine_idx"
    assert "cleo:economics_json" in got.attrs


def test_optimal_energy_gwh_a_aligns_full_axis_cf_to_lcoe_turbines() -> None:
    """Full-axis CF inputs are sliced back to the LCOE turbine order by label."""
    lcoe = xr.DataArray(
        np.array([[[5.0]], [[1.0]]], dtype=np.float64),
        dims=("turbine", "y", "x"),
        coords={"turbine": ["t1", "t3"], "y": [0], "x": [0]},
        name="lcoe",
    )
    lcoe.attrs["cleo:turbine_ids_json"] = json.dumps(["t1", "t3"], ensure_ascii=True)
    cf = xr.DataArray(
        np.array([[[np.nan]], [[0.2]], [[np.nan]], [[0.4]]], dtype=np.float64),
        dims=("turbine", "y", "x"),
        coords={"turbine": ["t0", "t1", "t2", "t3"], "y": [0], "x": [0]},
        name="capacity_factors",
    )

    got = optimal_energy_gwh_a(
        lcoe=lcoe,
        cf=cf,
        power_kw=np.array([1000.0, 2000.0], dtype=np.float64),
        hours_per_year=1.0,
    )

    np.testing.assert_allclose(got.values, np.array([[0.0008]], dtype=np.float64), rtol=0.0, atol=1e-12)


def test_optimal_selection_is_dask_safe_with_chunked_argmin_indexer() -> None:
    dask_array = pytest.importorskip("dask.array")

    lcoe = _sample_lcoe().chunk({"y": 1, "x": 1})
    cf = _sample_cf().chunk({"y": 1, "x": 1})
    power_kw = np.array([1000.0, 2000.0], dtype=np.float64)

    got_power = optimal_power_kw(lcoe=lcoe, power_kw=power_kw)
    got_energy = optimal_energy_gwh_a(lcoe=lcoe, cf=cf, power_kw=power_kw, hours_per_year=8766.0)

    assert isinstance(got_power.data, dask_array.Array)
    assert isinstance(got_energy.data, dask_array.Array)

    expected_power = np.array([[2000.0, np.nan], [1000.0, 2000.0]], dtype=np.float64)
    np.testing.assert_allclose(got_power.compute().values, expected_power, rtol=0.0, atol=0.0, equal_nan=True)

    expected_energy = np.array(
        [
            [0.35 * 2000.0 * 8766.0 / 1e6, np.nan],
            [0.3 * 1000.0 * 8766.0 / 1e6, 0.4 * 2000.0 * 8766.0 / 1e6],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got_energy.compute().values, expected_energy, rtol=0.0, atol=1e-12, equal_nan=True)
