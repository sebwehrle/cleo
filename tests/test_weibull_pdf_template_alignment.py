"""Test compute_weibull_pdf aligns to template grid."""

import numpy as np
import xarray as xr

from cleo.assess import compute_weibull_pdf


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


class DummySelf:
    """Minimal mock for compute_weibull_pdf."""

    def __init__(self, data, weibull_a, weibull_k):
        self.data = data
        self._weibull_a = weibull_a
        self._weibull_k = weibull_k

    def load_weibull_parameters(self, height):
        return (self._weibull_a, self._weibull_k)


def test_compute_weibull_pdf_aligns_to_template():
    """Verify compute_weibull_pdf aligns output to template grid."""
    # Template grid
    x_template = np.array([0.0, 1.0, 2.0, 3.0])
    y_template = np.array([2.0, 1.0, 0.0])
    template = make_template_raster(x_template, y_template)

    # Shifted grid for weibull parameters (simulate mismatch)
    x_shifted = np.array([0.1, 1.1, 2.1, 3.1])
    y_shifted = np.array([2.1, 1.1, 0.1])

    # Create weibull_a and weibull_k on shifted grid
    weibull_a = make_template_raster(x_shifted, y_shifted)
    weibull_a.values[:] = 8.0  # Typical scale parameter

    weibull_k = make_template_raster(x_shifted, y_shifted)
    weibull_k.values[:] = 2.0  # Typical shape parameter

    # Create dataset with template and wind_speed coord
    wind_speeds = np.array([0.0, 5.0, 10.0])
    ds = xr.Dataset(
        data_vars={"template": template},
        coords={"wind_speed": wind_speeds},
        attrs={"country": "TST"},
    )
    ds = ds.rio.write_crs("EPSG:4326")

    # Create dummy self
    dummy = DummySelf(ds, weibull_a, weibull_k)

    # Call compute_weibull_pdf
    compute_weibull_pdf(dummy, chunk_size=None)

    # Assertions
    assert "weibull_pdf" in dummy.data.data_vars, "weibull_pdf should be in dataset"

    pdf = dummy.data["weibull_pdf"]

    # Check coords match template exactly
    assert np.array_equal(pdf.coords["x"].values, x_template), (
        f"x coords mismatch: {pdf.coords['x'].values} vs {x_template}"
    )
    assert np.array_equal(pdf.coords["y"].values, y_template), (
        f"y coords mismatch: {pdf.coords['y'].values} vs {y_template}"
    )

    # Check wind_speed dimension
    assert "wind_speed" in pdf.dims
    assert np.array_equal(pdf.coords["wind_speed"].values, wind_speeds)

    # Stripe metric: ensure no column/row is >=95% NaN within valid area
    valid_mask = ~np.isnan(template.values)
    stripe_cols = 0
    for ws_idx in range(len(wind_speeds)):
        pdf_slice = pdf.isel(wind_speed=ws_idx).values
        for col_idx in range(pdf_slice.shape[1]):
            col_data = pdf_slice[:, col_idx]
            col_valid = valid_mask[:, col_idx]
            if col_valid.sum() > 0:
                nan_frac = np.isnan(col_data[col_valid]).sum() / col_valid.sum()
                if nan_frac >= 0.95:
                    stripe_cols += 1

    assert stripe_cols == 0, f"Found {stripe_cols} stripe columns with >=95% NaN"
