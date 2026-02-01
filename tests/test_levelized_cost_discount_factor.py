"""Oracle-driven tests for NPV factor in levelized_cost()."""
import pytest
import xarray as xr
from cleo.assess import levelized_cost


def test_npv_factor_positive_discount_rate():
    """Verify NPV factor equals sum_{t=1..N} (1+r)^(-t) = (1 - (1+r)^(-N)) / r."""
    discount_rate = 0.05
    lifetime = 20

    # Expected NPV factor: (1 - (1+r)^(-N)) / r
    expected_npv_factor = (1 - (1 + discount_rate) ** (-lifetime)) / discount_rate

    # Setup that isolates NPV factor: lcoe = overnight_cost / npv_factor
    capacity_factors = xr.DataArray(1.0)
    power = 1.0
    grid_cost = xr.DataArray(0.0)
    om_fixed = 0.0
    om_variable = 0.0
    overnight_cost = 100.0
    hours_per_year = 1

    lcoe = levelized_cost(
        power=power,
        capacity_factors=capacity_factors,
        overnight_cost=overnight_cost,
        grid_cost=grid_cost,
        om_fixed=om_fixed,
        om_variable=om_variable,
        discount_rate=discount_rate,
        lifetime=lifetime,
        hours_per_year=hours_per_year,
        per_mwh=False,
    )

    # lcoe = overnight_cost / npv_factor => npv_factor = overnight_cost / lcoe
    computed_npv_factor = overnight_cost / float(lcoe)

    assert abs(computed_npv_factor - expected_npv_factor) <= 1e-10


def test_npv_factor_zero_discount_rate():
    """Verify NPV factor equals N when discount_rate = 0."""
    discount_rate = 0.0
    lifetime = 20

    # Expected NPV factor: N
    expected_npv_factor = float(lifetime)

    # Setup that isolates NPV factor: lcoe = overnight_cost / npv_factor
    capacity_factors = xr.DataArray(1.0)
    power = 1.0
    grid_cost = xr.DataArray(0.0)
    om_fixed = 0.0
    om_variable = 0.0
    overnight_cost = 100.0
    hours_per_year = 1

    lcoe = levelized_cost(
        power=power,
        capacity_factors=capacity_factors,
        overnight_cost=overnight_cost,
        grid_cost=grid_cost,
        om_fixed=om_fixed,
        om_variable=om_variable,
        discount_rate=discount_rate,
        lifetime=lifetime,
        hours_per_year=hours_per_year,
        per_mwh=False,
    )

    # lcoe = overnight_cost / npv_factor => npv_factor = overnight_cost / lcoe
    computed_npv_factor = overnight_cost / float(lcoe)

    assert abs(computed_npv_factor - expected_npv_factor) <= 1e-10
