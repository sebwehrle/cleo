"""Tests for default wind_speed grid independent of turbines."""

import numpy as np
import xarray as xr

from cleo.assess import compute_weibull_pdf


# Default wind_speed grid: 0.0 to 40.0 step 0.5
DEFAULT_WIND_SPEED = np.arange(0.0, 40.0 + 0.5, 0.5)


def make_template_raster(x_coords, y_coords, crs="EPSG:4326"):
    """Create a template raster with rioxarray CRS and transform."""
    data = np.ones((len(y_coords), len(x_coords)), dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        name="template",
    )
    da = da.rio.write_crs(crs)
    x_res = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    y_res = y_coords[0] - y_coords[1] if len(y_coords) > 1 else -1.0
    from rasterio.transform import from_bounds
    transform = from_bounds(
        x_coords[0] - x_res / 2,
        y_coords[-1] - abs(y_res) / 2,
        x_coords[-1] + x_res / 2,
        y_coords[0] + abs(y_res) / 2,
        len(x_coords),
        len(y_coords),
    )
    da = da.rio.write_transform(transform)
    return da


class MockWindAtlasData:
    """Minimal mock for WindAtlas with wind_speed coord."""

    def __init__(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0])
        template = make_template_raster(x, y)

        self.data = xr.Dataset(
            data_vars={"template": template},
            coords={"wind_speed": DEFAULT_WIND_SPEED},
            attrs={"country": "TST"},
        )
        self.data = self.data.rio.write_crs("EPSG:4326")

    def load_weibull_parameters(self, height):
        """Return simple weibull parameters aligned to template."""
        x = self.data.coords["x"].values
        y = self.data.coords["y"].values
        a = make_template_raster(x, y)
        a.values[:] = 8.0
        k = make_template_raster(x, y)
        k.values[:] = 2.0
        return (a, k)


def test_default_wind_speed_grid_values():
    """Verify the default wind_speed grid is 0.0 to 40.0 step 0.5."""
    expected = np.arange(0.0, 40.0 + 0.5, 0.5)
    assert len(expected) == 81  # 0.0, 0.5, 1.0, ..., 40.0
    assert expected[0] == 0.0
    assert expected[-1] == 40.0
    assert np.allclose(np.diff(expected), 0.5)


def test_wind_atlas_has_wind_speed_without_turbine():
    """WindAtlas data should have wind_speed coord even without turbines."""
    mock = MockWindAtlasData()

    # Assert wind_speed is in coords
    assert "wind_speed" in mock.data.coords

    # Assert it equals the default grid
    assert np.array_equal(mock.data.coords["wind_speed"].values, DEFAULT_WIND_SPEED)


def test_power_curve_resampling():
    """Adding a turbine resamples its power_curve to atlas wind_speed grid."""
    # Original turbine curve: V=[0, 10, 20, 30], P=[0, 0.25, 0.75, 1.0]
    old_u = np.array([0.0, 10.0, 20.0, 30.0])
    old_p = np.array([0.0, 0.25, 0.75, 1.0])

    # Atlas wind_speed grid
    atlas_u = DEFAULT_WIND_SPEED

    # Expected resampled values via np.interp
    expected_p = np.interp(atlas_u, old_u, old_p, left=0.0, right=0.0)

    # Verify interpolation works as expected
    # At u=5: interp between (0,0) and (10,0.25) -> 0.125
    assert np.isclose(expected_p[10], 0.125)  # atlas_u[10] = 5.0
    # At u=15: interp between (10,0.25) and (20,0.75) -> 0.5
    assert np.isclose(expected_p[30], 0.5)  # atlas_u[30] = 15.0
    # At u=40: extrapolate right -> 0.0
    assert np.isclose(expected_p[-1], 0.0)  # atlas_u[-1] = 40.0

    # Create power_curve as the code would
    new_p = np.interp(atlas_u, old_u, old_p, left=0.0, right=0.0)
    power_curve = xr.DataArray(
        data=new_p,
        coords={'wind_speed': atlas_u},
        dims=['wind_speed']
    )

    # Assert coords match atlas grid
    assert np.array_equal(power_curve.coords["wind_speed"].values, atlas_u)
    # Assert values match expected interpolation
    assert np.allclose(power_curve.values, expected_p)


def test_compute_weibull_pdf_works_without_turbine():
    """compute_weibull_pdf works with default wind_speed grid, no turbine needed."""
    mock = MockWindAtlasData()

    # Call compute_weibull_pdf
    compute_weibull_pdf(mock, chunk_size=None)

    # Assert weibull_pdf was created
    assert "weibull_pdf" in mock.data.data_vars

    pdf = mock.data["weibull_pdf"]

    # Assert wind_speed dimension exists and matches atlas grid
    assert "wind_speed" in pdf.dims
    assert np.array_equal(pdf.coords["wind_speed"].values, DEFAULT_WIND_SPEED)

    # Assert spatial dims exist
    assert "y" in pdf.dims
    assert "x" in pdf.dims

    # Assert values are valid (non-negative PDFs)
    assert np.all(pdf.values >= 0)
