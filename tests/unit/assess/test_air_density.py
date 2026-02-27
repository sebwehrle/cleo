"""assess: test_air_density.

Contract tests for air density correction pure compute function:
- output formula is correct
- output preserves input coords
- output is lazy when input is lazy
- NaN propagation works correctly
"""

from __future__ import annotations

from tests.helpers.optional import requires_rioxarray, requires_rasterio

requires_rioxarray()
requires_rasterio()


import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from cleo.assess import compute_air_density_correction_core
from tests.helpers.factories import da_xy_with_crs


def test_air_density_correction_formula() -> None:
    """Air density correction formula should match expected barometric calculation."""
    # Create simple elevation array
    elevation_values = np.array([[0.0, 100.0], [500.0, 1000.0]])
    elevation = xr.DataArray(
        elevation_values,
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
        name="elevation",
    )
    template = elevation.copy()

    result = compute_air_density_correction_core(elevation=elevation, template=template)

    # Expected formula: 1.247015 * exp(-0.000104 * elevation) / 1.225
    expected = 1.247015 * np.exp(-0.000104 * elevation_values) / 1.225

    np.testing.assert_allclose(result.values, expected, rtol=1e-6)
    assert result.name == "air_density_correction"


def test_air_density_correction_preserves_coords() -> None:
    """Air density correction output must preserve input coordinates."""
    y_coords = np.array([10.0, 20.0, 30.0])
    x_coords = np.array([100.0, 200.0])

    elevation = xr.DataArray(
        np.ones((3, 2)) * 500.0,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        name="elevation",
    )
    template = elevation.copy()

    result = compute_air_density_correction_core(elevation=elevation, template=template)

    np.testing.assert_array_equal(result.coords["y"].values, y_coords)
    np.testing.assert_array_equal(result.coords["x"].values, x_coords)
    assert result.dims == ("y", "x")


def test_air_density_correction_propagates_nan() -> None:
    """NaN values in elevation must propagate to output."""
    elevation_values = np.array([[100.0, np.nan], [500.0, 1000.0]])
    elevation = xr.DataArray(
        elevation_values,
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
        name="elevation",
    )
    template = elevation.copy()

    result = compute_air_density_correction_core(elevation=elevation, template=template)

    assert np.isnan(result.values[0, 1]), "NaN should propagate from elevation to output"
    assert np.isfinite(result.values[0, 0]), "Finite values should remain finite"
    assert np.isfinite(result.values[1, 0]), "Finite values should remain finite"
    assert np.isfinite(result.values[1, 1]), "Finite values should remain finite"


def test_air_density_correction_lazy_dask() -> None:
    """Output should be lazy (dask-backed) when input is dask-backed."""
    elevation_values = da.ones((10, 10), chunks=(5, 5)) * 500.0
    elevation = xr.DataArray(
        elevation_values,
        dims=["y", "x"],
        coords={"y": range(10), "x": range(10)},
        name="elevation",
    )
    template = elevation.copy()

    result = compute_air_density_correction_core(elevation=elevation, template=template)

    # Check that result is dask-backed (lazy)
    assert hasattr(result.data, "compute"), "Result should be dask-backed"
    assert result.data.__class__.__module__.startswith("dask"), "Result should be dask array"


def test_air_density_correction_sea_level() -> None:
    """At sea level (elevation=0), correction factor should be close to 1.018."""
    elevation = xr.DataArray(
        np.zeros((2, 2)),
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
        name="elevation",
    )
    template = elevation.copy()

    result = compute_air_density_correction_core(elevation=elevation, template=template)

    # At sea level: 1.247015 * exp(0) / 1.225 = 1.247015 / 1.225 ≈ 1.018
    expected_sea_level = 1.247015 / 1.225
    np.testing.assert_allclose(result.values, expected_sea_level, rtol=1e-6)


def test_air_density_correction_high_altitude() -> None:
    """At high altitude, correction factor should decrease."""
    elevation_low = xr.DataArray(
        np.zeros((2, 2)),
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
        name="elevation",
    )
    elevation_high = xr.DataArray(
        np.ones((2, 2)) * 3000.0,  # 3000m
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
        name="elevation",
    )
    template = elevation_low.copy()

    result_low = compute_air_density_correction_core(elevation=elevation_low, template=template)
    result_high = compute_air_density_correction_core(elevation=elevation_high, template=template)

    # Higher altitude should have lower air density correction factor
    assert np.all(result_high.values < result_low.values), "Higher altitude should have lower air density correction"


@pytest.mark.parametrize("user_input", ["EPSG:3035", 3035])
def test_crs_from_user_input_normalizes_to_expected(user_input) -> None:
    """We rely on CRS.from_user_input for robust comparison across string/int forms."""
    expected = CRS.from_epsg(3035)
    assert CRS.from_user_input(user_input) == expected


def test_crs_comparison_detects_actual_difference() -> None:
    """Different CRS should be detected as different."""
    elevation = da_xy_with_crs(values=np.ones((2, 2), dtype=float), n=2, name="elev", crs="EPSG:4326")
    assert elevation.rio.crs != CRS.from_user_input("EPSG:3035")
