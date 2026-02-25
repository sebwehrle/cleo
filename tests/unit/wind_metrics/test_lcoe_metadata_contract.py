"""Contract tests for LCOE/optimal metric metadata lineage."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr

from cleo.wind_metrics import (
    _wind_metric_lcoe,
    _wind_metric_min_lcoe_turbine,
    _wind_metric_optimal_energy,
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
        },
    )
    wind.attrs["cleo_turbines_json"] = json.dumps(turbines_meta)
    land = xr.Dataset(
        {"valid_mask": (("y", "x"), np.array([[True, False], [True, True]]))},
        coords={"y": y, "x": x},
    )
    return wind, land


def _lcoe_params() -> dict[str, float | int]:
    return {
        "om_fixed_eur_per_kw_a": 25.0,
        "om_variable_eur_per_kwh": 0.01,
        "discount_rate": 0.05,
        "lifetime_a": 20,
        "hours_per_year": 8766.0,
    }


def test_lcoe_and_optimal_metrics_emit_metadata_lineage_contract() -> None:
    wind, land = _make_wind_land()
    params = _lcoe_params()

    lcoe = _wind_metric_lcoe(wind, land, turbines=("T1", "T2"), **params)
    idx = _wind_metric_min_lcoe_turbine(wind, land, turbines=("T1", "T2"), **params)
    power = _wind_metric_optimal_power(wind, land, turbines=("T1", "T2"), **params)
    energy = _wind_metric_optimal_energy(wind, land, turbines=("T1", "T2"), **params)

    assert lcoe.attrs["units"] == "EUR/MWh"
    assert "cleo:economics_json" in lcoe.attrs
    assert json.loads(lcoe.attrs["cleo:turbine_ids_json"]) == ["T1", "T2"]

    economics = json.loads(lcoe.attrs["cleo:economics_json"])
    assert economics["discount_rate"] == 0.05
    assert economics["lifetime_a"] == 20
    assert economics["om_fixed_eur_per_kw_a"] == 25.0
    assert economics["om_variable_eur_per_kwh"] == 0.01
    assert economics["bos_cost_share"] == 0.0

    assert idx.attrs["cleo:nodata_index"] == -1
    assert idx.attrs["cleo:turbine_ids_json"] == lcoe.attrs["cleo:turbine_ids_json"]
    assert idx.attrs["cleo:economics_json"] == lcoe.attrs["cleo:economics_json"]

    assert power.attrs["units"] == "kW"
    assert power.attrs["cleo:selection_basis"] == "min_lcoe_turbine_idx"
    assert power.attrs["cleo:turbine_ids_json"] == lcoe.attrs["cleo:turbine_ids_json"]
    assert power.attrs["cleo:economics_json"] == lcoe.attrs["cleo:economics_json"]

    assert energy.attrs["units"] == "GWh/a"
    assert energy.attrs["cleo:selection_basis"] == "min_lcoe_turbine_idx"
    assert energy.attrs["cleo:turbine_ids_json"] == lcoe.attrs["cleo:turbine_ids_json"]
    assert energy.attrs["cleo:economics_json"] == lcoe.attrs["cleo:economics_json"]
