"""Test that compute_chunked preserves all Dataset variables."""

import numpy as np
import xarray as xr

from cleo.utils import compute_chunked


class MockSelf:
    """Minimal mock for compute_chunked."""

    def __init__(self, x_size=4, y_size=4):
        self.data = xr.Dataset(
            coords={
                "x": np.arange(x_size, dtype=float),
                "y": np.arange(y_size, dtype=float),
            }
        )


def test_compute_chunked_preserves_dataset_vars():
    """compute_chunked should preserve all Dataset variables when func returns Dataset."""

    def processing_func(**kwargs):
        """Return a Dataset with two variables."""
        # Get chunk coords from first kwarg that's a DataArray
        for v in kwargs.values():
            if isinstance(v, xr.DataArray):
                y_coords = v.coords["y"].values
                x_coords = v.coords["x"].values
                break
        else:
            y_coords = np.array([0.0, 1.0])
            x_coords = np.array([0.0, 1.0])

        return xr.Dataset({
            "a": xr.DataArray(
                np.ones((len(y_coords), len(x_coords))),
                dims=["y", "x"],
                coords={"y": y_coords, "x": x_coords},
            ),
            "b": xr.DataArray(
                np.ones((len(y_coords), len(x_coords))) * 2,
                dims=["y", "x"],
                coords={"y": y_coords, "x": x_coords},
            ),
        })

    mock_self = MockSelf(x_size=4, y_size=4)

    # Create a dummy input DataArray to pass through
    dummy_input = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": mock_self.data.coords["y"], "x": mock_self.data.coords["x"]},
    )

    result = compute_chunked(mock_self, processing_func, chunk_size=2, dummy=dummy_input)

    # Assert result is a Dataset
    assert isinstance(result, xr.Dataset), f"Expected Dataset, got {type(result)}"

    # Assert both variables are present
    assert set(result.data_vars) == {"a", "b"}, f"Expected vars {{'a', 'b'}}, got {set(result.data_vars)}"


def test_compute_chunked_returns_dataarray():
    """compute_chunked should return DataArray when func returns DataArray."""

    def processing_func(**kwargs):
        """Return a DataArray."""
        for v in kwargs.values():
            if isinstance(v, xr.DataArray):
                y_coords = v.coords["y"].values
                x_coords = v.coords["x"].values
                break
        else:
            y_coords = np.array([0.0, 1.0])
            x_coords = np.array([0.0, 1.0])

        return xr.DataArray(
            np.ones((len(y_coords), len(x_coords))) * 3,
            dims=["y", "x"],
            coords={"y": y_coords, "x": x_coords},
            name="result",
        )

    mock_self = MockSelf(x_size=4, y_size=4)

    dummy_input = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": mock_self.data.coords["y"], "x": mock_self.data.coords["x"]},
    )

    result = compute_chunked(mock_self, processing_func, chunk_size=2, dummy=dummy_input)

    # Assert result is a DataArray
    assert isinstance(result, xr.DataArray), f"Expected DataArray, got {type(result)}"
    assert result.name == "result"
