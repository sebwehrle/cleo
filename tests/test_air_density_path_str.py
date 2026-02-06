"""Test that compute_air_density_correction handles str path without TypeError."""

import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import MagicMock, patch


def _create_tiny_raster():
    """Create a tiny DataArray for testing."""
    da = xr.DataArray(
        np.ones((3, 3)) * 500,  # 500m elevation
        dims=["y", "x"],
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
    )
    da = da.rio.write_crs("EPSG:4326")
    return da


def test_compute_air_density_correction_with_str_path(tmp_path):
    """compute_air_density_correction should work when parent.path is a str."""
    # Create mock self (WindAtlas) with parent.path as str
    mock_self = MagicMock()
    mock_self.parent = MagicMock()
    mock_self.parent.path = str(tmp_path)  # str, not Path
    mock_self.parent.country = "AUT"
    mock_self.parent.crs = "EPSG:4326"
    mock_self.parent.region = None
    mock_self.data = xr.Dataset()

    # Create fake reference file path (needed for FileNotFoundError check)
    raw_dir = tmp_path / "data" / "raw" / "AUT"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ref_file = raw_dir / "AUT_combined-Weibull-A_100.tif"
    ref_file.touch()  # Create empty file

    tiny_raster = _create_tiny_raster()

    # Mock rxr.open_rasterio to return tiny raster
    def mock_open_rasterio(path, *args, **kwargs):
        return tiny_raster

    # Mock load_elevation to return tiny raster
    def mock_load_elevation(base_dir, iso3, reference_da):
        return tiny_raster.copy()

    with patch("cleo.assess.rxr.open_rasterio", side_effect=mock_open_rasterio):
        with patch("cleo.loaders.load_elevation", side_effect=mock_load_elevation):
            from cleo.assess import compute_air_density_correction

            # Should not raise TypeError: unsupported operand type(s) for /: 'str' and 'str'
            compute_air_density_correction(mock_self)

    # Assert result was stored
    assert "air_density_correction" in mock_self.data.data_vars
    assert np.all(np.isfinite(mock_self.data["air_density_correction"].values))


def test_compute_air_density_correction_missing_file_error(tmp_path):
    """compute_air_density_correction should raise FileNotFoundError with clear message."""
    import pytest

    mock_self = MagicMock()
    mock_self.parent = MagicMock()
    mock_self.parent.path = str(tmp_path)
    mock_self.parent.country = "AUT"
    mock_self.parent.crs = "EPSG:4326"
    mock_self.parent.region = None
    mock_self.data = xr.Dataset()

    # Don't create the raw directory - file will be missing

    from cleo.assess import compute_air_density_correction

    with pytest.raises(FileNotFoundError) as exc_info:
        compute_air_density_correction(mock_self)

    # Error message should be actionable
    assert "Raw GWA files missing" in str(exc_info.value)
    assert "AUT" in str(exc_info.value)
