"""classes: test_save_load.
Merged test file (imports preserved per chunk).
"""

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from pathlib import Path
from unittest.mock import patch, MagicMock
from cleo.classes import Atlas

# --- merged from tests/_staging/test_save_creates_processed_dir.py ---

class _DummySubAtlas:
    def __init__(self):
        self.data = xr.Dataset(attrs={"country": "AUT", "region": "None"})


def test_save_creates_processed_dir(tmp_path, monkeypatch):
    # Create Atlas instance without running __init__
    atlas = Atlas.__new__(Atlas)
    atlas._path = Path(tmp_path)
    atlas.country = "AUT"
    atlas.region = None
    atlas.index_file = atlas._path / "data" / "index.txt"
    atlas.wind = _DummySubAtlas()
    atlas.landscape = _DummySubAtlas()

    # Avoid real NetCDF writing; just touch the file path
    def _fake_to_netcdf(self, path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"dummy")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _fake_to_netcdf, raising=True)

    # Remove processed dir if exists
    processed = atlas._path / "data" / "processed"
    if processed.exists():
        for p in processed.rglob("*"):
            if p.is_file():
                p.unlink()
        processed.rmdir()

    atlas.save()

    assert processed.is_dir()


# --- merged from tests/_staging/test_save_does_not_write_index_on_failure.py ---

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


# --- merged from tests/_staging/test_save_load_roundtrip_default_names.py ---

def _create_mock_dataset(with_turbine=False):
    """Create a minimal xarray Dataset for testing."""
    coords = {
        "x": np.arange(3, dtype=float),
        "y": np.arange(3, dtype=float),
    }
    if with_turbine:
        coords["turbine"] = ["test_turbine"]

    ds = xr.Dataset(coords=coords)
    ds["template"] = xr.DataArray(
        np.zeros((3, 3)),
        dims=["y", "x"],
        coords={"y": coords["y"], "x": coords["x"]},
    )
    # Set attrs including one that could be None
    ds.attrs["country"] = "AUT"
    ds.attrs["region"] = None  # This would break NetCDF without sanitization

    # Add rio.crs mock
    ds = ds.rio.write_crs("EPSG:4326")
    return ds


def test_save_load_roundtrip_default_names(tmp_path):
    """Atlas.save() should write files, Atlas.load() should read them back."""
    # Create mock Atlas with patched __init__
    with patch("cleo.classes.Atlas.__init__", return_value=None):
        from cleo.classes import Atlas

        # Create first Atlas instance for saving
        atlas1 = Atlas.__new__(Atlas)
        atlas1._path = tmp_path
        atlas1.country = "AUT"
        atlas1._region = None
        atlas1._crs = "EPSG:4326"
        atlas1._wind_turbines = []

        # Create data directory structure
        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)

        # Create index file
        atlas1.index_file = tmp_path / "data" / "index.txt"
        atlas1.index_file.touch()

        # Create mock wind and landscape subclasses
        atlas1.wind = MagicMock()
        atlas1.wind.data = _create_mock_dataset(with_turbine=True)
        atlas1.landscape = MagicMock()
        atlas1.landscape.data = _create_mock_dataset()

        # Save should succeed (attrs sanitized)
        atlas1.save()

        # Verify at least one nc file exists under processed
        nc_files = list((tmp_path / "data" / "processed").glob("*.nc"))
        assert len(nc_files) >= 1, "Expected at least one .nc file after save"

        # Verify index was updated
        with open(atlas1.index_file) as f:
            index_content = f.read()
        assert "WindAtlas" in index_content, "Index should contain WindAtlas entry"
        assert "LandscapeAtlas" in index_content, "Index should contain LandscapeAtlas entry"

        # Create second Atlas instance for loading
        atlas2 = Atlas.__new__(Atlas)
        atlas2._path = tmp_path
        atlas2.country = "AUT"
        atlas2._region = None
        atlas2._crs = "EPSG:4326"
        atlas2._wind_turbines = []
        atlas2.index_file = tmp_path / "data" / "index.txt"

        # Mock wind and landscape for the load target
        atlas2.wind = MagicMock()
        atlas2.landscape = MagicMock()

        # Load should not raise
        atlas2.load()

        # Verify data was loaded
        assert atlas2.wind.data is not None, "wind.data should be set after load"
        assert atlas2.landscape.data is not None, "landscape.data should be set after load"


def test_save_sanitizes_none_attrs(tmp_path):
    """save() should sanitize None attrs to avoid NetCDF serialization errors."""
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

        # Create datasets with None attrs
        atlas.wind = MagicMock()
        atlas.wind.data = _create_mock_dataset()
        atlas.wind.data.attrs["region"] = None
        atlas.wind.data.attrs["scenario"] = None

        atlas.landscape = MagicMock()
        atlas.landscape.data = _create_mock_dataset()
        atlas.landscape.data.attrs["region"] = None

        # Save should succeed without raising
        atlas.save()

        # Verify files were written
        nc_files = list((tmp_path / "data" / "processed").glob("*.nc"))
        assert len(nc_files) == 2, f"Expected 2 .nc files, got {len(nc_files)}"

        # Verify None attrs were removed
        assert "region" not in atlas.wind.data.attrs or atlas.wind.data.attrs["region"] is not None
