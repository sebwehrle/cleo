"""Unit tests for wind metric orchestration helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from cleo.wind_metrics import (
    _extract_overnight_cost_eur_per_kw,
    _extract_turbine_power_kw,
    _wind_metric_capacity_factors,
    _wind_metric_lcoe,
    _wind_metric_optimal_energy,
    _wind_metric_rews_mps,
)


def _make_wind_land(
    *,
    with_rho: bool = True,
    with_cleo_turbines_json: bool = True,
    turbines_meta: list[dict] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 1.0], dtype=np.float64)
    height = np.array([10.0, 50.0, 100.0, 150.0, 200.0], dtype=np.float64)
    wind_speed = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    turbine = np.array(["T1", "T2"], dtype=object)

    wind = xr.Dataset(
        coords={
            "y": y,
            "x": x,
            "height": height,
            "wind_speed": wind_speed,
            "turbine": turbine,
        },
        data_vars={
            "weibull_A": (("height", "y", "x"), np.full((5, 2, 2), 8.0, dtype=np.float64)),
            "weibull_k": (("height", "y", "x"), np.full((5, 2, 2), 2.0, dtype=np.float64)),
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
        },
    )
    if with_rho:
        wind["rho"] = xr.DataArray(
            np.full((5, 2, 2), 1.225, dtype=np.float64),
            dims=("height", "y", "x"),
            coords={"height": height, "y": y, "x": x},
        )

    if turbines_meta is None:
        turbines_meta = [
            {
                "id": "T1",
                "capacity": 3000.0,
                "overnight_cost_eur_per_kw": 1300.0,
                "rotor_diameter": 120.0,
            },
            {
                "id": "T2",
                "capacity_kw": 3500.0,
                "overnight_cost": 1400.0,
                "rotor_diameter_m": 130.0,
            },
        ]
    if with_cleo_turbines_json:
        wind.attrs["cleo_turbines_json"] = json.dumps(turbines_meta)

    land = xr.Dataset(
        coords={"y": y, "x": x},
        data_vars={"valid_mask": (("y", "x"), np.array([[True, False], [True, True]]))},
    )
    return wind, land


def test_extract_turbine_power_kw_mixed_supported_keys() -> None:
    meta = [
        {"id": "A", "capacity": 3000},
        {"id": "B", "capacity_kw": 3200},
        {"id": "C", "capacity_mw": 3.4},
    ]
    out = _extract_turbine_power_kw(meta, ("A", "B", "C"))
    assert np.allclose(out, np.array([3000.0, 3200.0, 3400.0]))


def test_extract_turbine_power_kw_missing_capacity_raises() -> None:
    meta = [{"id": "A"}]
    with pytest.raises(ValueError, match="missing capacity info"):
        _extract_turbine_power_kw(meta, ("A",))


def test_extract_overnight_cost_supported_keys() -> None:
    meta = [
        {"id": "A", "overnight_cost_eur_per_kw": 1200},
        {"id": "B", "overnight_cost": 1300},
    ]
    out = _extract_overnight_cost_eur_per_kw(meta, ("A", "B"))
    assert np.allclose(out, np.array([1200.0, 1300.0]))


def test_extract_overnight_cost_missing_raises() -> None:
    meta = [{"id": "A"}]
    with pytest.raises(ValueError, match="missing overnight cost"):
        _extract_overnight_cost_eur_per_kw(meta, ("A",))


def test_capacity_factors_requires_valid_mask() -> None:
    wind, _land = _make_wind_land()
    with pytest.raises(ValueError, match="valid_mask required"):
        _wind_metric_capacity_factors(wind, None, turbines=("T1",))


def test_capacity_factors_requires_cleo_turbines_json_attr() -> None:
    wind, land = _make_wind_land(with_cleo_turbines_json=False)
    with pytest.raises(ValueError, match="cleo_turbines_json"):
        _wind_metric_capacity_factors(wind, land, turbines=("T1",))


def test_capacity_factors_rews_missing_rotor_diameter_raises() -> None:
    bad_meta = [
        {"id": "T1", "capacity": 3000.0, "overnight_cost_eur_per_kw": 1300.0},
        {"id": "T2", "capacity_kw": 3500.0, "overnight_cost": 1400.0},
    ]
    wind, land = _make_wind_land(turbines_meta=bad_meta)
    with pytest.raises(ValueError, match="requires rotor_diameter"):
        _wind_metric_capacity_factors(wind, land, turbines=("T1",), mode="rews")


def test_capacity_factors_rews_from_turbine_metadata_succeeds() -> None:
    wind, land = _make_wind_land()
    out = _wind_metric_capacity_factors(wind, land, turbines=("T1",), mode="rews", rews_n=3)
    assert out.name == "capacity_factors"
    assert out.dims == ("turbine", "y", "x")
    assert "cleo:cf_mode" in out.attrs
    assert out.attrs["cleo:cf_mode"] == "rews"


def test_capacity_factors_direct_cf_quadrature_is_available() -> None:
    wind, land = _make_wind_land()
    out = _wind_metric_capacity_factors(wind, land, turbines=("T1",), mode="direct_cf_quadrature", rews_n=4)
    assert out.name == "capacity_factors"
    assert out.attrs["cleo:cf_mode"] == "direct_cf_quadrature"


def test_capacity_factors_momentmatch_weibull_is_available() -> None:
    wind, land = _make_wind_land()
    out = _wind_metric_capacity_factors(wind, land, turbines=("T1",), mode="momentmatch_weibull", rews_n=4)
    assert out.name == "capacity_factors"
    assert out.attrs["cleo:cf_mode"] == "momentmatch_weibull"


def test_rews_mps_metric_returns_expected_shape() -> None:
    wind, land = _make_wind_land()
    out = _wind_metric_rews_mps(wind, land, turbines=("T1",), rews_n=4)
    assert out.name == "rews_mps"
    assert out.dims == ("turbine", "y", "x")
    assert out.attrs["units"] == "m/s"


def test_capacity_factors_air_density_requires_rho() -> None:
    wind, land = _make_wind_land(with_rho=False)
    with pytest.raises(ValueError, match="air_density=True but wind store missing 'rho'"):
        _wind_metric_capacity_factors(wind, land, turbines=("T1",), air_density=True)


def test_lcoe_missing_required_params_raises() -> None:
    wind, land = _make_wind_land()
    with pytest.raises(ValueError, match="LCOE requires parameters"):
        _wind_metric_lcoe(wind, land, turbines=("T1",), hours_per_year=8766.0)


def test_optimal_energy_missing_required_params_raises() -> None:
    wind, land = _make_wind_land()
    with pytest.raises(ValueError, match="optimal_energy requires LCOE parameters"):
        _wind_metric_optimal_energy(wind, land, turbines=("T1",), hours_per_year=8766.0)
