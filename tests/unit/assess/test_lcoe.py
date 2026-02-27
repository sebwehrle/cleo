"""economics: test_lcoe.

Tests for cleo.economics.levelized_cost, with analytic oracles.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.economics import levelized_cost


def _npv_factor(*, r: float, n: int) -> float:
    """
    Oracle: present value factor of a unit annuity paid at t=1..n.

    r = 0  ->  n
    r > 0  ->  sum_{t=1..n} (1+r)^(-t) = (1 - (1+r)^(-n)) / r
    """
    if r == 0.0:
        return float(n)
    return (1.0 - (1.0 + r) ** (-n)) / r


def test_npv_factor_positive_discount_rate() -> None:
    """NPV factor matches analytic closed form for r>0 (isolated via LCOE algebra)."""
    r = 0.05
    n = 20
    expected = _npv_factor(r=r, n=n)

    # isolate npv_factor via: lcoe = overnight_cost / npv_factor (per_mwh=False, CF=1, power=1, hours=1)
    overnight_cost = 100.0
    lcoe = levelized_cost(
        power=1.0,
        capacity_factors=xr.DataArray(1.0),
        overnight_cost=overnight_cost,
        grid_cost=xr.DataArray(0.0),
        om_fixed=0.0,
        om_variable=0.0,
        discount_rate=r,
        lifetime=n,
        hours_per_year=1,
        per_mwh=False,
    )

    got = overnight_cost / float(lcoe)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-10)


def test_npv_factor_zero_discount_rate() -> None:
    """NPV factor equals n for r=0 (isolated via LCOE algebra)."""
    r = 0.0
    n = 20
    expected = _npv_factor(r=r, n=n)

    overnight_cost = 100.0
    lcoe = levelized_cost(
        power=1.0,
        capacity_factors=xr.DataArray(1.0),
        overnight_cost=overnight_cost,
        grid_cost=xr.DataArray(0.0),
        om_fixed=0.0,
        om_variable=0.0,
        discount_rate=r,
        lifetime=n,
        hours_per_year=1,
        per_mwh=False,
    )

    got = overnight_cost / float(lcoe)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-10)


def test_levelized_cost_oracle_r0_only_capex_per_mwh() -> None:
    """
    Oracle with r=0 and only CAPEX:
      LCOE [EUR/MWh] = overnight_cost / (CF * hours * power * lifetime) * 1000
    where power is kW and energy is kWh.
    """
    cf = xr.DataArray([[0.5]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    power_kw = 1000.0
    hours = 8766
    lifetime = 20

    overnight_cost_eur = 1_000_000.0
    grid_cost = xr.DataArray([[0.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    got = levelized_cost(
        power=power_kw,
        capacity_factors=cf,
        overnight_cost=overnight_cost_eur,
        grid_cost=grid_cost,
        om_fixed=0.0,
        om_variable=0.0,
        discount_rate=0.0,
        lifetime=lifetime,
        hours_per_year=hours,
        per_mwh=True,
    )

    lifetime_kwh = float(cf.values.item()) * hours * power_kw * lifetime
    expected = overnight_cost_eur / lifetime_kwh * 1000.0

    assert got.shape == (1, 1)
    np.testing.assert_allclose(got.values, expected, rtol=0.0, atol=1e-12)


def test_levelized_cost_oracle_with_om_and_discounting() -> None:
    """
    Oracle (general):
      npv_electricity_kwh = CF * hours * power_kw * npv_factor
      npv_cost_eur =
          ((om_variable * CF * hours + om_fixed) * power_kw * npv_factor)
          + overnight_cost + grid_cost
      LCOE [EUR/MWh] = (npv_cost_eur / npv_electricity_kwh) * 1000
    """
    cf_val = 0.3
    cf = xr.DataArray([[cf_val]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    power_kw = 2000.0
    hours = 8766
    lifetime = 25
    r = 0.05

    overnight_cost_eur = 3_000_000.0
    grid_cost = xr.DataArray([[200_000.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    om_fixed_eur_per_kw = 30.0  # EUR/kW-year equivalent in model's NPV structure
    om_variable_eur_per_kwh = 0.002  # EUR/kWh

    got = levelized_cost(
        power=power_kw,
        capacity_factors=cf,
        overnight_cost=overnight_cost_eur,
        grid_cost=grid_cost,
        om_fixed=om_fixed_eur_per_kw,
        om_variable=om_variable_eur_per_kwh,
        discount_rate=r,
        lifetime=lifetime,
        hours_per_year=hours,
        per_mwh=True,
    )

    npv = _npv_factor(r=r, n=lifetime)
    npv_electricity_kwh = cf_val * hours * power_kw * npv
    npv_cost_eur = (
        (om_variable_eur_per_kwh * cf_val * hours + om_fixed_eur_per_kw) * power_kw * npv
        + overnight_cost_eur
        + float(grid_cost.values.item())
    )
    expected = (npv_cost_eur / npv_electricity_kwh) * 1000.0

    np.testing.assert_allclose(got.values, expected, rtol=0.0, atol=1e-10)


def test_levelized_cost_scaling_invariance_power_and_costs() -> None:
    """
    Scaling invariance: if you scale project size by s (power and absolute EUR costs),
    LCOE should remain unchanged (given identical CF and per-kW/per-kWh O&M rates).
    """
    cf = xr.DataArray([[0.4]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    grid_cost = xr.DataArray([[50_000.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    common = dict(
        capacity_factors=cf,
        om_fixed=25.0,  # EUR/kW-year
        om_variable=0.0015,  # EUR/kWh
        discount_rate=0.05,
        lifetime=20,
        hours_per_year=8766,
        per_mwh=True,
    )

    l1 = levelized_cost(power=1000.0, overnight_cost=1_000_000.0, grid_cost=grid_cost, **common)
    l2 = levelized_cost(power=2000.0, overnight_cost=2_000_000.0, grid_cost=grid_cost * 2.0, **common)

    np.testing.assert_allclose(l1.values, l2.values, rtol=0.0, atol=1e-10)
