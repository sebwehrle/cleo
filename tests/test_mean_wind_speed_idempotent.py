"""Test compute_mean_wind_speed idempotency."""

import numpy as np
import xarray as xr
from unittest.mock import MagicMock

from cleo.assess import compute_mean_wind_speed


def test_compute_mean_wind_speed_idempotent_no_recompute():
    """
    Test that compute_mean_wind_speed does not recompute when height already exists.
    """
    # Create mock self with existing mean_wind_speed at height=100
    mock_self = MagicMock()

    existing_data = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.float32) * 7.5,
        dims=["height", "y", "x"],
        coords={"height": [100], "y": range(4), "x": range(4)},
        name="mean_wind_speed",
    )

    mock_self.data = xr.Dataset({"mean_wind_speed": existing_data})

    # Track if load_weibull_parameters is called (indicates computation happened)
    load_calls = {"count": 0}

    def mock_load_weibull(height):
        load_calls["count"] += 1
        return (
            xr.DataArray(np.ones((4, 4)) * 8.0),
            xr.DataArray(np.ones((4, 4)) * 2.0),
        )

    mock_self.load_weibull_parameters = mock_load_weibull

    # Call compute_mean_wind_speed for height=100 (already exists)
    compute_mean_wind_speed(mock_self, 100)

    # Assert load_weibull_parameters was NOT called (no computation)
    assert load_calls["count"] == 0, "Should not compute when height already exists"

    # Assert height=100 appears exactly once
    heights = mock_self.data["mean_wind_speed"].coords["height"].values
    assert list(heights) == [100], f"Height should be [100], got {list(heights)}"

    # Call again to verify idempotency
    compute_mean_wind_speed(mock_self, 100)

    assert load_calls["count"] == 0, "Should still not compute on second call"
    heights = mock_self.data["mean_wind_speed"].coords["height"].values
    assert list(heights) == [100], "Height should still be [100]"


def test_compute_mean_wind_speed_adds_new_height():
    """
    Test that compute_mean_wind_speed correctly adds a new height.
    """
    # Create mock self with existing mean_wind_speed at height=50
    mock_self = MagicMock()

    existing_data = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.float32) * 6.0,
        dims=["height", "y", "x"],
        coords={"height": [50], "y": range(4), "x": range(4)},
        name="mean_wind_speed",
    )

    mock_self.data = xr.Dataset({"mean_wind_speed": existing_data})

    # Track calls
    load_calls = {"count": 0}

    def mock_load_weibull(height):
        load_calls["count"] += 1
        return (
            xr.DataArray(np.ones((4, 4)) * 8.0, dims=["y", "x"]),
            xr.DataArray(np.ones((4, 4)) * 2.0, dims=["y", "x"]),
        )

    mock_self.load_weibull_parameters = mock_load_weibull

    # Call compute_mean_wind_speed for height=100 (new)
    compute_mean_wind_speed(mock_self, 100)

    # Assert load_weibull_parameters WAS called (computation happened)
    assert load_calls["count"] == 1, "Should compute for new height"

    # Assert heights are now [50, 100] (unique, both present)
    heights = sorted(mock_self.data["mean_wind_speed"].coords["height"].values)
    assert heights == [50, 100], f"Heights should be [50, 100], got {heights}"


def test_compute_mean_wind_speed_no_duplicates_after_multiple_calls():
    """
    Test that calling compute_mean_wind_speed multiple times for same height
    does not create duplicate height entries.
    """
    mock_self = MagicMock()

    existing_data = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.float32) * 7.5,
        dims=["height", "y", "x"],
        coords={"height": [100], "y": range(4), "x": range(4)},
        name="mean_wind_speed",
    )

    mock_self.data = xr.Dataset({"mean_wind_speed": existing_data})

    def mock_load_weibull(height):
        return (
            xr.DataArray(np.ones((4, 4)) * 8.0, dims=["y", "x"]),
            xr.DataArray(np.ones((4, 4)) * 2.0, dims=["y", "x"]),
        )

    mock_self.load_weibull_parameters = mock_load_weibull

    # Call multiple times for the same height
    for _ in range(5):
        compute_mean_wind_speed(mock_self, 100)

    # Assert no duplicates
    heights = list(mock_self.data["mean_wind_speed"].coords["height"].values)
    assert heights == [100], f"Should have exactly one height=100, got {heights}"
    assert len(heights) == 1, f"Should have exactly 1 height entry, got {len(heights)}"
