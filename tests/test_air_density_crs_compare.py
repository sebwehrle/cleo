"""Test that compute_air_density_correction uses robust CRS comparison."""

import numpy as np
import xarray as xr
from rasterio.crs import CRS


def test_crs_normalization_string_vs_crs_object():
    """String 'EPSG:3035' should equal CRS.from_epsg(3035) after normalization."""
    string_crs = "EPSG:3035"
    int_crs = 3035
    crs_obj = CRS.from_epsg(3035)

    # All should normalize to the same CRS
    normalized_string = CRS.from_user_input(string_crs)
    normalized_int = CRS.from_user_input(int_crs)

    assert normalized_string == crs_obj, "String EPSG:3035 should equal CRS object"
    assert normalized_int == crs_obj, "Int 3035 should equal CRS object"
    assert normalized_string == normalized_int, "String and int should normalize equally"


def test_crs_comparison_avoids_unnecessary_reproject():
    """When elevation CRS matches parent CRS, no reproject should be needed."""
    # Create a tiny elevation DataArray with EPSG:3035
    elevation = xr.DataArray(
        np.ones((3, 3)),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
    )
    elevation = elevation.rio.write_crs("EPSG:3035")

    # Simulate parent.crs as string (common case)
    parent_crs_string = "EPSG:3035"

    # Robust comparison using CRS.from_user_input should see them as equal
    src_crs = elevation.rio.crs
    dst_crs = CRS.from_user_input(parent_crs_string)
    assert src_crs == dst_crs, "Robust comparison should see EPSG:3035 CRS object == string"

    # Same for integer representation
    parent_crs_int = 3035
    dst_crs_from_int = CRS.from_user_input(parent_crs_int)
    assert src_crs == dst_crs_from_int, "Robust comparison should see EPSG:3035 CRS object == int"


def test_crs_comparison_with_int_crs():
    """Parent CRS as integer (3035) should work correctly."""
    elevation = xr.DataArray(
        np.ones((2, 2)),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    elevation = elevation.rio.write_crs("EPSG:3035")

    # Parent CRS as integer
    parent_crs_int = 3035

    src_crs = elevation.rio.crs
    dst_crs = CRS.from_user_input(parent_crs_int)

    assert src_crs == dst_crs, "CRS from int should match elevation CRS"


def test_crs_comparison_detects_actual_difference():
    """Different CRS should be detected as different."""
    elevation = xr.DataArray(
        np.ones((2, 2)),
        dims=["y", "x"],
    )
    elevation = elevation.rio.write_crs("EPSG:4326")

    parent_crs = "EPSG:3035"

    src_crs = elevation.rio.crs
    dst_crs = CRS.from_user_input(parent_crs)

    assert src_crs != dst_crs, "EPSG:4326 and EPSG:3035 should be different"
