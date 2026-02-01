"""Test that annual energy calculation uses exactly 8766 hours/year."""

import numpy as np
import xarray as xr
from unittest.mock import MagicMock

from cleo.assess import compute_optimal_power_energy


def test_annual_energy_uses_8766_hours():
    """
    Verify that annual energy is computed using 8766 hours/year.

    Oracle: expected_energy = capacity_factor * rated_power_kw * 8766 / 1e6
    (output is in GWh given the /1e6 conversion in the code)
    """
    # Create mock self object
    mock_self = MagicMock()

    # Rated power encoded in turbine name (parse_capacity splits by '.' and takes last)
    rated_power_kw = 1000.0
    turbine_name = f"TestTurbine.{int(rated_power_kw)}"

    # lcoe with single turbine - minimal structure for idxmin to work
    lcoe = xr.DataArray(
        data=[[[50.0]]],
        dims=["turbine", "y", "x"],
        coords={"turbine": [turbine_name], "y": [0], "x": [0]},
    )

    # capacity_factors with value 1.0 (full capacity)
    capacity_factors = xr.DataArray(
        data=[[[1.0]]],
        dims=["turbine", "y", "x"],
        coords={"turbine": [turbine_name], "y": [0], "x": [0]},
    )

    # Set up mock data - include 'lcoe' so @requires decorator passes
    mock_self.data = xr.Dataset({
        "lcoe": lcoe,
        "capacity_factors": capacity_factors,
    })

    # Call the function
    compute_optimal_power_energy(mock_self)

    # Oracle: capacity_factor * rated_power_kw * 8766 / 1e6
    # With capacity_factor = 1.0, rated_power_kw = 1000:
    # expected = 1.0 * 1000 * 8766 / 1e6 = 8.766
    expected_energy = 1.0 * rated_power_kw * 8766.0 / 1e6

    # Get the result
    result_energy = float(mock_self.data["optimal_energy"].values.item())

    # Assert exact match within tight tolerance
    assert np.isclose(result_energy, expected_energy, rtol=0, atol=1e-9), (
        f"Annual energy {result_energy} does not match oracle {expected_energy}. "
        f"Difference: {abs(result_energy - expected_energy)}"
    )
