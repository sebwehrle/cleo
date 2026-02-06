from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import levelized_cost


def _npv_factor(r: float, n: int) -> float:
    """
    Oracle discount factor (present value factor of an annuity).
    For r=0: sum_{t=1..n} 1 = n
    For r>0: sum_{t=1..n} (1+r)^(-t) = (1 - (1+r)^(-n)) / r
    """
    if r == 0:
        return float(n)
    return (1.0 - (1.0 + r) ** (-n)) / r


def test_levelized_cost_oracle_r0_only_capex_per_mwh():
    """
    Oracle: with r=0 and only overnight_cost non-zero:
      LCOE [EUR/MWh] = overnight_cost / (CF * hours * power * lifetime) * 1000
    where power is kW, hours per year is hours, CF dimensionless.
    """
    cf = xr.DataArray([[0.5]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    power_kw = 1000.0
    hours = 8766
    lifetime = 20
    r = 0.0

    overnight_cost_eur = 1_000_000.0  # absolute EUR lump sum
    grid_cost = xr.DataArray([[0.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    om_fixed_eur_per_kw = 0.0
    om_variable_eur_per_kwh = 0.0

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

    # Oracle expected value
    lifetime_kwh = float(cf.values.item()) * hours * power_kw * lifetime
    expected_eur_per_mwh = overnight_cost_eur / lifetime_kwh * 1000.0

    assert got.shape == (1, 1)
    np.testing.assert_allclose(got.values, expected_eur_per_mwh, rtol=0, atol=1e-12)


def test_levelized_cost_oracle_with_om_and_discounting():
    """
    Oracle: general formula
      NPV_electricity = CF * hours * power * npv_factor
      NPV_cost = ((om_variable * CF * hours + om_fixed) * power * npv_factor) + overnight_cost + grid_cost
      LCOE = NPV_cost / NPV_electricity * (1000 if per_mwh else 1)
    """
    cf = xr.DataArray([[0.3]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    power_kw = 2000.0
    hours = 8766
    lifetime = 25
    r = 0.05

    overnight_cost_eur = 3_000_000.0  # absolute EUR
    grid_cost = xr.DataArray([[200_000.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    om_fixed_eur_per_kw = 30.0          # EUR/kW-year equivalent (model uses EUR/kW inside NPV with npv_factor)
    om_variable_eur_per_kwh = 0.002     # EUR/kWh

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

    npv = _npv_factor(r, lifetime)
    npv_electricity_kwh = float(cf.values.item()) * hours * power_kw * npv
    npv_cost_eur = ((om_variable_eur_per_kwh * float(cf.values.item()) * hours + om_fixed_eur_per_kw) * power_kw * npv) \
                   + overnight_cost_eur + float(grid_cost.values.item())

    expected_eur_per_mwh = (npv_cost_eur / npv_electricity_kwh) * 1000.0

    np.testing.assert_allclose(got.values, expected_eur_per_mwh, rtol=0, atol=1e-10)
