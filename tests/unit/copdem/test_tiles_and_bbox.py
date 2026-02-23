"""copdem: test_tiles_and_bbox.
Merged test file (imports preserved per chunk).
"""

from tests.helpers.optional import requires_rioxarray

requires_rioxarray()

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from pathlib import Path
from affine import Affine
from cleo.copdem import tiles_for_bbox


def _create_reference_da_with_crs(crs, x_range, y_range):
    """Create a minimal DataArray with specified CRS and coordinate ranges."""
    x = np.linspace(x_range[0], x_range[1], 10)
    y = np.linspace(y_range[0], y_range[1], 10)
    data = np.ones((len(y), len(x)))

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    da = da.rio.write_crs(crs)

    # Compute transform from coords
    res_x = (x[-1] - x[0]) / (len(x) - 1)
    res_y = (y[0] - y[-1]) / (len(y) - 1)  # y typically decreasing
    transform = Affine.translation(x[0] - res_x / 2, y[0] - res_y / 2) * Affine.scale(res_x, res_y)
    da = da.rio.write_transform(transform)

    return da


def test_projected_crs_bounds_transformed_to_lonlat(tmp_path, monkeypatch):
    """Projected CRS (EPSG:3035) bounds should be transformed to lon/lat degrees."""
    # EPSG:3035 is ETRS89-extended / LAEA Europe - coordinates are in meters
    # Create reference in EPSG:3035 with typical European extent in meters
    # (roughly center of Europe: ~4000000 E, ~3000000 N)
    ref = _create_reference_da_with_crs(
        "EPSG:3035",
        x_range=(4000000, 4100000),  # ~100km in meters
        y_range=(2900000, 3000000),  # ~100km in meters
    )

    captured = {}

    def mock_download_tiles(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat):
        captured["min_lon"] = min_lon
        captured["min_lat"] = min_lat
        captured["max_lon"] = max_lon
        captured["max_lat"] = max_lat
        # Return dummy tile paths (no actual download)
        return [Path("/fake/tile1.tif")]

    def mock_build_elevation(reference_da, tile_paths):
        # Return a trivial DataArray aligned to reference
        return xr.DataArray(
            np.ones_like(reference_da.values),
            dims=reference_da.dims,
            coords=reference_da.coords,
        ).rio.write_crs(reference_da.rio.crs)

    monkeypatch.setattr("cleo.copdem.download_copdem_tiles_for_bbox", mock_download_tiles)
    monkeypatch.setattr("cleo.copdem.build_copdem_elevation_like", mock_build_elevation)

    # Create minimal directory structure
    (tmp_path / "data" / "raw" / "AUT").mkdir(parents=True, exist_ok=True)

    from cleo.loaders import load_elevation
    load_elevation(tmp_path, "AUT", ref)

    # Assert bounds were captured and are valid lon/lat degrees
    assert "min_lon" in captured, "download_copdem_tiles_for_bbox should have been called"

    min_lon = captured["min_lon"]
    min_lat = captured["min_lat"]
    max_lon = captured["max_lon"]
    max_lat = captured["max_lat"]

    # Check bounds are finite
    assert np.isfinite(min_lon), f"min_lon should be finite, got {min_lon}"
    assert np.isfinite(min_lat), f"min_lat should be finite, got {min_lat}"
    assert np.isfinite(max_lon), f"max_lon should be finite, got {max_lon}"
    assert np.isfinite(max_lat), f"max_lat should be finite, got {max_lat}"

    # Check bounds are in valid lon/lat range
    assert -180 <= min_lon < max_lon <= 180, f"lon range invalid: {min_lon} to {max_lon}"
    assert -90 <= min_lat < max_lat <= 90, f"lat range invalid: {min_lat} to {max_lat}"

    # The original bounds in EPSG:3035 were ~4 million meters
    # After transformation, they should be reasonable European coordinates
    # (roughly 10-20°E, 45-55°N for central Europe)
    assert min_lon < 100, f"min_lon should be degrees not meters: {min_lon}"
    assert max_lon < 100, f"max_lon should be degrees not meters: {max_lon}"


def test_epsg4326_bounds_passed_unchanged(tmp_path, monkeypatch):
    """EPSG:4326 bounds should be passed through unchanged."""
    # Create reference in EPSG:4326 with known bounds
    expected_bounds = (10.0, 46.0, 17.0, 49.0)  # Austria-ish
    ref = _create_reference_da_with_crs(
        "EPSG:4326",
        x_range=(expected_bounds[0], expected_bounds[2]),
        y_range=(expected_bounds[1], expected_bounds[3]),
    )

    captured = {}

    def mock_download_tiles(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat):
        captured["min_lon"] = min_lon
        captured["min_lat"] = min_lat
        captured["max_lon"] = max_lon
        captured["max_lat"] = max_lat
        return [Path("/fake/tile1.tif")]

    def mock_build_elevation(reference_da, tile_paths):
        return xr.DataArray(
            np.ones_like(reference_da.values),
            dims=reference_da.dims,
            coords=reference_da.coords,
        ).rio.write_crs(reference_da.rio.crs)

    monkeypatch.setattr("cleo.copdem.download_copdem_tiles_for_bbox", mock_download_tiles)
    monkeypatch.setattr("cleo.copdem.build_copdem_elevation_like", mock_build_elevation)

    (tmp_path / "data" / "raw" / "AUT").mkdir(parents=True, exist_ok=True)

    from cleo.loaders import load_elevation
    load_elevation(tmp_path, "AUT", ref)

    # Get actual bounds from reference
    actual_bounds = ref.rio.bounds()

    # Assert captured bounds match reference bounds exactly
    assert captured["min_lon"] == actual_bounds[0], f"min_lon mismatch: {captured['min_lon']} != {actual_bounds[0]}"
    assert captured["min_lat"] == actual_bounds[1], f"min_lat mismatch: {captured['min_lat']} != {actual_bounds[1]}"
    assert captured["max_lon"] == actual_bounds[2], f"max_lon mismatch: {captured['max_lon']} != {actual_bounds[2]}"
    assert captured["max_lat"] == actual_bounds[3], f"max_lat mismatch: {captured['max_lat']} != {actual_bounds[3]}"



def test_tiles_for_bbox_austria():
    """
    Test tile selection for Austria bounding box.

    Oracle bbox (from PO's Austria raster bounds):
        min_lon=9.50875, max_lon=17.18125, min_lat=46.35125, max_lat=49.04125

    Expected integer degree ranges:
        lon_deg = 9..17 inclusive (9 values)
        lat_deg = 46..49 inclusive (4 values)
        expected count = 36
    """
    min_lon = 9.50875
    max_lon = 17.18125
    min_lat = 46.35125
    max_lat = 49.04125

    # Oracle: inline formatting function (do not import from cleo.copdem)
    def oracle_tile_id(lat_deg, lon_deg):
        ns = "N" if lat_deg >= 0 else "S"
        ew = "E" if lon_deg >= 0 else "W"
        abs_lat = abs(lat_deg)
        abs_lon = abs(lon_deg)
        return f"Copernicus_DSM_COG_10_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"

    # Build expected list using oracle
    expected_tiles = []
    for lat_deg in range(46, 50):  # 46..49 inclusive
        for lon_deg in range(9, 18):  # 9..17 inclusive
            expected_tiles.append(oracle_tile_id(lat_deg, lon_deg))
    expected_tiles.sort()

    # Call the function under test
    result = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)

    # Assert correct count
    assert len(result) == 36, f"Expected 36 tiles, got {len(result)}"

    # Assert exact equality with oracle
    assert result == expected_tiles, f"Tile list mismatch:\nExpected: {expected_tiles}\nGot: {result}"


def test_tiles_for_bbox_determinism():
    """Test that tiles_for_bbox returns identical results on repeated calls."""
    min_lon = 9.50875
    max_lon = 17.18125
    min_lat = 46.35125
    max_lat = 49.04125

    result1 = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)
    result2 = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)

    assert result1 == result2, "tiles_for_bbox should be deterministic"
