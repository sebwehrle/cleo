"""Test that save() does not update index when to_netcdf fails."""

import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import MagicMock, patch


def _create_mock_dataset():
    """Create a minimal xarray Dataset for testing."""
    ds = xr.Dataset(
        coords={
            "x": np.arange(3, dtype=float),
            "y": np.arange(3, dtype=float),
        }
    )
    ds["template"] = xr.DataArray(
        np.zeros((3, 3)),
        dims=["y", "x"],
    )
    ds.attrs["country"] = "AUT"
    ds = ds.rio.write_crs("EPSG:4326")
    return ds


def test_save_does_not_write_index_on_failure(tmp_path):
    """If to_netcdf fails, index should not be updated."""
    with patch("cleo.classes.Atlas.__init__", return_value=None):
        from cleo.classes import Atlas

        atlas = Atlas.__new__(Atlas)
        atlas._path = tmp_path
        atlas.country = "AUT"
        atlas._region = None
        atlas._wind_turbines = []

        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
        atlas.index_file = tmp_path / "data" / "index.txt"
        atlas.index_file.touch()

        # Create datasets
        atlas.wind = MagicMock()
        atlas.wind.data = _create_mock_dataset()
        atlas.landscape = MagicMock()
        atlas.landscape.data = _create_mock_dataset()

        # Patch to_netcdf to raise an error
        original_to_netcdf = xr.Dataset.to_netcdf

        def failing_to_netcdf(self, *args, **kwargs):
            raise RuntimeError("Simulated write failure")

        with patch.object(xr.Dataset, "to_netcdf", failing_to_netcdf):
            # save() should log error but not raise
            atlas.save()

        # Verify index was NOT updated (should be empty)
        with open(atlas.index_file) as f:
            index_content = f.read()

        assert index_content.strip() == "", f"Index should be empty after failed save, got: {index_content}"

        # Verify no .nc files were created
        nc_files = list((tmp_path / "data" / "processed").glob("*.nc"))
        assert len(nc_files) == 0, f"No .nc files should exist after failed save, got {len(nc_files)}"


def test_save_partial_failure_does_not_update_index(tmp_path):
    """If first to_netcdf succeeds but second fails, index should not be updated."""
    with patch("cleo.classes.Atlas.__init__", return_value=None):
        from cleo.classes import Atlas

        atlas = Atlas.__new__(Atlas)
        atlas._path = tmp_path
        atlas.country = "AUT"
        atlas._region = None
        atlas._wind_turbines = []

        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
        atlas.index_file = tmp_path / "data" / "index.txt"
        atlas.index_file.touch()

        # Create datasets
        atlas.wind = MagicMock()
        atlas.wind.data = _create_mock_dataset()
        atlas.landscape = MagicMock()
        atlas.landscape.data = _create_mock_dataset()

        call_count = [0]
        original_to_netcdf = xr.Dataset.to_netcdf

        def fail_on_second_call(self, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated write failure on second file")
            return original_to_netcdf(self, *args, **kwargs)

        with patch.object(xr.Dataset, "to_netcdf", fail_on_second_call):
            atlas.save()

        # Verify index was NOT updated (should be empty)
        with open(atlas.index_file) as f:
            index_content = f.read()

        assert index_content.strip() == "", f"Index should be empty after partial save failure, got: {index_content}"
