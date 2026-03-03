"""Phase 5 wind_metrics coverage tests (success + extra error branches)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from cleo.wind_metrics import (
    _wind_metric_capacity_factors,
    _wind_metric_lcoe,
    _wind_metric_mean_wind_speed,
    _wind_metric_min_lcoe_turbine,
    _wind_metric_optimal_power,
)


def _make_wind_land() -> tuple[xr.Dataset, xr.Dataset]:
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 1.0], dtype=np.float64)
    height = np.array([50.0, 100.0], dtype=np.float64)
    wind_speed = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    turbines_meta = [
        {"id": "T1", "capacity": 3000.0, "overnight_cost_eur_per_kw": 1300.0, "rotor_diameter": 120.0},
        {"id": "T2", "capacity_kw": 3500.0, "overnight_cost": 1400.0, "rotor_diameter_m": 130.0},
    ]

    wind = xr.Dataset(
        coords={
            "y": y,
            "x": x,
            "height": height,
            "wind_speed": wind_speed,
            "turbine": np.array([0, 1], dtype=np.int64),
        },
        data_vars={
            "weibull_A": (("height", "y", "x"), np.full((2, 2, 2), 8.0, dtype=np.float64)),
            "weibull_k": (("height", "y", "x"), np.full((2, 2, 2), 2.0, dtype=np.float64)),
            "power_curve": (
                ("turbine", "wind_speed"),
                np.vstack(
                    [
                        np.clip((wind_speed - 3.0) / 10.0, 0.0, 1.0),
                        np.clip((wind_speed - 4.0) / 9.0, 0.0, 1.0),
                    ]
                ).astype(np.float64),
            ),
            "turbine_hub_height": (("turbine",), np.array([100.0, 100.0], dtype=np.float64)),
            "turbine_rotor_diameter": (("turbine",), np.array([120.0, 130.0], dtype=np.float64)),
            "rho": (("height", "y", "x"), np.full((2, 2, 2), 1.225, dtype=np.float64)),
        },
    )
    wind.attrs["cleo_turbines_json"] = json.dumps(turbines_meta)
    land = xr.Dataset(
        {"valid_mask": (("y", "x"), np.array([[True, False], [True, True]]))},
        coords={"y": y, "x": x},
    )
    return wind, land


def _lcoe_params() -> dict:
    return {
        "om_fixed_eur_per_kw_a": 25.0,
        "om_variable_eur_per_kwh": 0.01,
        "discount_rate": 0.05,
        "lifetime_a": 20,
        "hours_per_year": 8766.0,
    }


def test_mean_wind_speed_invalid_height_raises() -> None:
    wind, land = _make_wind_land()
    with pytest.raises(ValueError, match="height=200 not in wind store"):
        _wind_metric_mean_wind_speed(wind, land, height=200)


def test_mean_wind_speed_applies_valid_mask() -> None:
    wind, land = _make_wind_land()
    out = _wind_metric_mean_wind_speed(wind, land, height=100)
    assert out.dims == ("height", "y", "x")
    assert out.sizes["height"] == 1
    assert float(out.coords["height"].values[0]) == 100.0
    assert np.isnan(out.values[0, 0, 1])
    assert np.isfinite(out.values[0, 0, 0])


def test_capacity_factors_missing_required_vars_raise() -> None:
    wind, land = _make_wind_land()
    with pytest.raises(ValueError, match="must have weibull_[aA] and weibull_k"):
        _wind_metric_capacity_factors(wind.drop_vars("weibull_A"), land, turbines=("T1",))
    with pytest.raises(ValueError, match="must have power_curve"):
        _wind_metric_capacity_factors(wind.drop_vars("power_curve"), land, turbines=("T1",))


def test_capacity_factors_unknown_turbine_raises() -> None:
    wind, land = _make_wind_land()
    with pytest.raises(ValueError, match="turbine 'T9' not in wind store"):
        _wind_metric_capacity_factors(wind, land, turbines=("T9",))


def test_lcoe_min_lcoe_and_optimal_power_success_paths() -> None:
    wind, land = _make_wind_land()
    params = _lcoe_params()
    lcoe = _wind_metric_lcoe(wind, land, turbines=("T1", "T2"), **params)
    idx = _wind_metric_min_lcoe_turbine(wind, land, turbines=("T1", "T2"), **params)
    p = _wind_metric_optimal_power(wind, land, turbines=("T1", "T2"), **params)

    assert lcoe.dims == ("turbine", "y", "x")
    assert idx.dims == ("y", "x")
    assert p.dims == ("y", "x")
    assert np.isfinite(lcoe.values[np.isfinite(lcoe.values)]).all()
