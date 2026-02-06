"""Test that compute_wind_shear_coefficient masks invalid cells (log(<=0) protection)."""

import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch


def _create_mock_wind_atlas_with_wind_speed(u50_values, u100_values):
    """Create a mock _WindAtlas with mean_wind_speed data."""
    x = np.arange(len(u50_values), dtype=float)

    # Create mean_wind_speed with height dimension
    mean_wind_speed = xr.DataArray(
        data=np.array([u50_values, u100_values]),
        dims=["height", "x"],
        coords={"height": [50, 100], "x": x},
        name="mean_wind_speed",
    )

    mock_self = MagicMock()
    mock_self.data = xr.Dataset({"mean_wind_speed": mean_wind_speed})
    mock_self.parent = MagicMock()
    mock_self.parent.country = "AUT"

    return mock_self


def test_wind_shear_masks_zero_wind_speed():
    """Wind shear should be NaN where u50 == 0."""
    # u50=[0, 1, 2], u100=[2, 2, 2]
    # Cell 0: u50=0 -> invalid (log(0) = -inf)
    # Cells 1,2: valid
    mock_self = _create_mock_wind_atlas_with_wind_speed(
        u50_values=[0.0, 1.0, 2.0],
        u100_values=[2.0, 2.0, 2.0],
    )

    from cleo.assess import compute_wind_shear_coefficient
    compute_wind_shear_coefficient(mock_self)

    wind_shear = mock_self.data["wind_shear"]

    # Cell 0 should be NaN
    assert np.isnan(wind_shear.values[0]), "wind_shear[0] should be NaN (u50=0)"

    # Cells 1 and 2 should be finite
    assert np.isfinite(wind_shear.values[1]), "wind_shear[1] should be finite"
    assert np.isfinite(wind_shear.values[2]), "wind_shear[2] should be finite"


def test_wind_shear_masks_negative_wind_speed():
    """Wind shear should be NaN where wind speed is negative."""
    mock_self = _create_mock_wind_atlas_with_wind_speed(
        u50_values=[-1.0, 1.0],
        u100_values=[2.0, 2.0],
    )

    from cleo.assess import compute_wind_shear_coefficient
    compute_wind_shear_coefficient(mock_self)

    wind_shear = mock_self.data["wind_shear"]

    assert np.isnan(wind_shear.values[0]), "wind_shear[0] should be NaN (u50<0)"
    assert np.isfinite(wind_shear.values[1]), "wind_shear[1] should be finite"


def test_wind_shear_no_inf_values():
    """Wind shear should never contain inf values."""
    # Include edge cases that could produce inf
    mock_self = _create_mock_wind_atlas_with_wind_speed(
        u50_values=[0.0, -1.0, np.nan, 1.0, 5.0],
        u100_values=[2.0, 2.0, 2.0, 2.0, 10.0],
    )

    from cleo.assess import compute_wind_shear_coefficient
    compute_wind_shear_coefficient(mock_self)

    wind_shear = mock_self.data["wind_shear"]

    # No inf values should exist
    assert not np.any(np.isinf(wind_shear.values)), "wind_shear should not contain inf"


def test_wind_shear_valid_cells_computed_correctly():
    """Valid cells should have correctly computed wind shear."""
    # u50=5, u100=10 -> alpha = (log(10) - log(5)) / (log(100) - log(50))
    # = log(2) / log(2) = 1.0
    mock_self = _create_mock_wind_atlas_with_wind_speed(
        u50_values=[5.0, 5.0],  # Use 2 cells to avoid 0-d squeeze
        u100_values=[10.0, 10.0],
    )

    from cleo.assess import compute_wind_shear_coefficient
    compute_wind_shear_coefficient(mock_self)

    wind_shear = mock_self.data["wind_shear"]

    expected = np.log(2) / np.log(2)  # = 1.0
    # Handle both 0-d and 1-d arrays
    values = wind_shear.values.flatten()
    assert np.isclose(values[0], expected), f"Expected {expected}, got {values[0]}"
    assert np.isclose(values[1], expected), f"Expected {expected}, got {values[1]}"


def test_wind_shear_masks_nan_input():
    """Wind shear should be NaN where input wind speed is NaN."""
    mock_self = _create_mock_wind_atlas_with_wind_speed(
        u50_values=[np.nan, 1.0],
        u100_values=[2.0, 2.0],
    )

    from cleo.assess import compute_wind_shear_coefficient
    compute_wind_shear_coefficient(mock_self)

    wind_shear = mock_self.data["wind_shear"]

    assert np.isnan(wind_shear.values[0]), "wind_shear[0] should be NaN (u50=NaN)"
    assert np.isfinite(wind_shear.values[1]), "wind_shear[1] should be finite"
