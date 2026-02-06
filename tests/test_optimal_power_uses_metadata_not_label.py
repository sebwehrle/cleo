"""Test that compute_optimal_power_energy uses turbine metadata, not label parsing."""

import numpy as np
import xarray as xr
from unittest.mock import MagicMock


def test_optimal_power_uses_metadata_not_label():
    """
    optimal_power should use get_turbine_attribute(..., "capacity"),
    not parse the turbine label string.

    Example: "Vestas.V112.3.45" should NOT result in 45.0 (string parse).
    If metadata says capacity=3.45, that's what we should get.
    """
    # Create mock self (WindAtlas)
    mock_self = MagicMock()

    # Turbine label that would give wrong result with string parsing
    turbine_label = "Vestas.V112.3.45"
    # String parsing would give: parts[-1] = "45" -> 45.0
    # But actual capacity from metadata is 3.45

    # Create lcoe DataArray where this turbine is always the minimum
    lcoe = xr.DataArray(
        np.array([[[100.0]]]),  # Single value
        dims=["turbine", "y", "x"],
        coords={"turbine": [turbine_label], "y": [0.0], "x": [0.0]},
    )

    # Create capacity_factors
    capacity_factors = xr.DataArray(
        np.array([[[0.35]]]),
        dims=["turbine", "y", "x"],
        coords={"turbine": [turbine_label], "y": [0.0], "x": [0.0]},
    )

    mock_self.data = xr.Dataset({
        "lcoe": lcoe,
        "capacity_factors": capacity_factors,
    })
    mock_self.data = mock_self.data.assign_coords(turbine=[turbine_label])

    # Mock get_turbine_attribute to return the correct capacity
    def mock_get_turbine_attribute(turbine_name, attr):
        if attr == "capacity":
            return 3.45  # NOT 45.0
        return None

    mock_self.get_turbine_attribute = mock_get_turbine_attribute

    from cleo.assess import compute_optimal_power_energy
    compute_optimal_power_energy(mock_self)

    optimal_power = mock_self.data["optimal_power"]

    # Should be 3.45 (from metadata), NOT 45.0 (from string parsing)
    assert float(optimal_power.values.flatten()[0]) == 3.45, (
        f"Expected 3.45 from metadata, got {optimal_power.values.flatten()[0]}. "
        "String parsing bug still present?"
    )


def test_optimal_power_multiple_turbines():
    """Test with multiple turbines where best varies by location."""
    mock_self = MagicMock()

    turbines = ["Enercon.E126.7.58", "Vestas.V90.3.0"]

    # Create lcoe where different turbines win at different locations
    lcoe_data = np.array([
        [[80.0, 120.0]],  # Enercon wins at x=0
        [[100.0, 70.0]],  # Vestas wins at x=1
    ])
    lcoe = xr.DataArray(
        lcoe_data,
        dims=["turbine", "y", "x"],
        coords={"turbine": turbines, "y": [0.0], "x": [0.0, 1.0]},
    )

    capacity_factors = xr.DataArray(
        np.ones((2, 1, 2)) * 0.3,
        dims=["turbine", "y", "x"],
        coords={"turbine": turbines, "y": [0.0], "x": [0.0, 1.0]},
    )

    mock_self.data = xr.Dataset({
        "lcoe": lcoe,
        "capacity_factors": capacity_factors,
    })

    # Metadata: Enercon=7.58 MW, Vestas=3.0 MW
    capacity_map = {
        "Enercon.E126.7.58": 7.58,
        "Vestas.V90.3.0": 3.0,
    }

    def mock_get_turbine_attribute(turbine_name, attr):
        if attr == "capacity":
            return capacity_map.get(turbine_name, np.nan)
        return None

    mock_self.get_turbine_attribute = mock_get_turbine_attribute

    from cleo.assess import compute_optimal_power_energy
    compute_optimal_power_energy(mock_self)

    optimal_power = mock_self.data["optimal_power"]

    # At x=0: Enercon wins -> 7.58
    # At x=1: Vestas wins -> 3.0
    assert float(optimal_power.isel(y=0, x=0).values) == 7.58
    assert float(optimal_power.isel(y=0, x=1).values) == 3.0
