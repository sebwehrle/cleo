"""economics: levelized_cost tests."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.economics import levelized_cost


def _npv_factor(*, r: float, n: int) -> float:
    if r == 0.0:
        return float(n)
    return (1.0 - (1.0 + r) ** (-n)) / r


def test_levelized_cost_closed_form_per_mwh() -> None:
    cf_val = 0.3
    cf = xr.DataArray([[cf_val]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    power_kw = 2000.0
    hours = 8766
    lifetime = 25
    discount_rate = 0.05

    overnight_cost = 3_000_000.0
    grid_cost = xr.DataArray([[200_000.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    om_fixed = 30.0
    om_variable = 0.002

    got = levelized_cost(
        power=power_kw,
        capacity_factors=cf,
        overnight_cost=overnight_cost,
        grid_cost=grid_cost,
        om_fixed=om_fixed,
        om_variable=om_variable,
        discount_rate=discount_rate,
        lifetime=lifetime,
        hours_per_year=hours,
        per_mwh=True,
    )

    npv = _npv_factor(r=discount_rate, n=lifetime)
    npv_electricity_kwh = cf_val * hours * power_kw * npv
    npv_cost_eur = ((om_variable * cf_val * hours + om_fixed) * power_kw * npv) + overnight_cost + 200_000.0
    expected = (npv_cost_eur / npv_electricity_kwh) * 1000.0

    np.testing.assert_allclose(got.values, expected, rtol=0.0, atol=1e-10)


def test_levelized_cost_returns_kwh_units_when_requested() -> None:
    got_kwh = levelized_cost(
        power=1000.0,
        capacity_factors=xr.DataArray(0.5),
        overnight_cost=1_000_000.0,
        grid_cost=xr.DataArray(0.0),
        om_fixed=0.0,
        om_variable=0.0,
        discount_rate=0.0,
        lifetime=20,
        hours_per_year=8766,
        per_mwh=False,
    )
    got_mwh = levelized_cost(
        power=1000.0,
        capacity_factors=xr.DataArray(0.5),
        overnight_cost=1_000_000.0,
        grid_cost=xr.DataArray(0.0),
        om_fixed=0.0,
        om_variable=0.0,
        discount_rate=0.0,
        lifetime=20,
        hours_per_year=8766,
        per_mwh=True,
    )

    np.testing.assert_allclose(float(got_mwh), float(got_kwh) * 1000.0, rtol=0.0, atol=1e-12)
