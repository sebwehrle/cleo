"""Test elevation reference fallback to air-density file when reference_da lacks CRS."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import xarray as xr
from pathlib import Path

from cleo.loaders import load_elevation


def test_load_elevation_uses_air_density_when_reference_lacks_crs(monkeypatch, tmp_path):
    """
    Test that load_elevation uses air-density file as reference when
    the provided reference_da lacks CRS.
    """
    iso3 = "AUT"

    # Create directory structure
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True)

    # Create air-density file with CRS
    air_density_path = raw_dir / f"{iso3}_air-density_100.tif"
    air_density_data = np.ones((4, 8), dtype=np.float32)
    air_density_transform = from_bounds(9.0, 46.0, 17.0, 49.0, 8, 4)

    with rasterio.open(
        air_density_path,
        "w",
        driver="GTiff",
        height=4,
        width=8,
        count=1,
        dtype=air_density_data.dtype,
        crs="EPSG:4326",
        transform=air_density_transform,
    ) as dst:
        dst.write(air_density_data, 1)

    # Create a reference DataArray WITHOUT CRS
    reference_da_no_crs = xr.DataArray(
        np.zeros((4, 8), dtype=np.float32),
        dims=["y", "x"],
    )

    # Sentinel value to verify correct DataArray is returned
    sentinel_elevation = xr.DataArray(
        np.full((4, 8), 999.0, dtype=np.float32),
        dims=["y", "x"],
        name="elevation",
    )
    sentinel_elevation = sentinel_elevation.rio.write_crs("EPSG:4326")

    # Track calls
    build_calls = []

    def mock_download(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat, overwrite=False):
        return [Path("/fake/tile1.tif"), Path("/fake/tile2.tif")]

    def mock_build(ref_da, tile_paths):
        # Assert that the reference_da passed to build has CRS
        assert ref_da.rio.crs is not None, "reference_da passed to build should have CRS"
        build_calls.append({"ref_crs": str(ref_da.rio.crs)})
        return sentinel_elevation

    monkeypatch.setattr("cleo.copdem.download_copdem_tiles_for_bbox", mock_download)
    monkeypatch.setattr("cleo.copdem.build_copdem_elevation_like", mock_build)

    # Call load_elevation with a reference that has no CRS
    result = load_elevation(tmp_path, iso3, reference_da_no_crs)

    # Assert build was called
    assert len(build_calls) == 1, "build_copdem_elevation_like should be called once"

    # Assert the reference passed to build had CRS
    assert "4326" in build_calls[0]["ref_crs"]

    # Assert result is the sentinel
    assert np.allclose(result.values, 999.0)
    assert result.name == "elevation"


def test_load_elevation_raises_when_no_crs_and_no_air_density(monkeypatch, tmp_path):
    """
    Test that load_elevation raises ValueError when reference_da lacks CRS
    and air-density file does not exist.
    """
    import pytest

    iso3 = "AUT"

    # Create directory structure but NO air-density file
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True)

    # Create a reference DataArray WITHOUT CRS
    reference_da_no_crs = xr.DataArray(
        np.zeros((4, 8), dtype=np.float32),
        dims=["y", "x"],
    )

    with pytest.raises(ValueError) as exc_info:
        load_elevation(tmp_path, iso3, reference_da_no_crs)

    assert "no CRS" in str(exc_info.value)
    assert "air-density" in str(exc_info.value)
