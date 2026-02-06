"""assess: test_energy_aggregation.

Tests for cleo.assess energy aggregation helpers.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import compute_optimal_power_energy


def test_annual_energy_uses_8766_hours() -> None:
    """
    Verify annual energy uses 8766 hours/year.

    Oracle:
        optimal_energy [GWh] = capacity_factor * rated_power_kw * 8766 / 1e6
    """
    rated_power_kw = 1000.0
    turbine = f"TestTurbine.{int(rated_power_kw)}"

    # Minimal dataset: idxmin over turbine works if lcoe exists.
    lcoe = xr.DataArray(
        data=[[[50.0]]],
        dims=("turbine", "y", "x"),
        coords={"turbine": [turbine], "y": [0], "x": [0]},
    )
    capacity_factors = xr.DataArray(
        data=[[[1.0]]],
        dims=("turbine", "y", "x"),
        coords={"turbine": [turbine], "y": [0], "x": [0]},
    )

    class _Self:
        data = xr.Dataset({"lcoe": lcoe, "capacity_factors": capacity_factors})

        @staticmethod
        def get_turbine_attribute(_turbine: str, attr: str) -> float | None:
            return rated_power_kw if attr == "capacity" else None

    self = _Self()

    compute_optimal_power_energy(self)

    expected = 1.0 * rated_power_kw * 8766.0 / 1e6  # GWh
    got = self.data["optimal_energy"].values.item()

    assert np.isclose(got, expected, rtol=0.0, atol=1e-9), (got, expected)
