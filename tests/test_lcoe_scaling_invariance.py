from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import levelized_cost


def test_levelized_cost_scaling_invariance_power_and_costs():
    # Same CF everywhere
    cf = xr.DataArray([[0.4]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    grid_cost = xr.DataArray([[50_000.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    params = dict(
        capacity_factors=cf,
        om_fixed=25.0,              # EUR/kW-year (as used in production)
        om_variable=0.0015,         # EUR/kWh
        discount_rate=0.05,
        lifetime=20,
        hours_per_year=8766,
        per_mwh=True,
    )

    # Baseline
    l1 = levelized_cost(
        power=1000.0,
        overnight_cost=1_000_000.0,
        grid_cost=grid_cost,
        **params,
    )

    # Scale power by 2 and scale absolute costs that scale with project size:
    # overnight_cost and grid_cost are absolute EUR -> scale by 2 to represent a project twice as big.
    l2 = levelized_cost(
        power=2000.0,
        overnight_cost=2_000_000.0,
        grid_cost=grid_cost * 2.0,
        **params,
    )

    np.testing.assert_allclose(l1.values, l2.values, rtol=0, atol=1e-10)
