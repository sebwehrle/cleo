"""
utils: _match_to_template tests (grid/template alignment).
Split from merged test_downloads_and_templates.py
"""

from tests.helpers.optional import requires_rioxarray, requires_rasterio

requires_rioxarray()
requires_rasterio()

import pytest
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.transform import from_bounds

from cleo.utils import _match_to_template


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


def test_match_to_template_already_aligned():
    """If coords already match, return unchanged."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0])

    template = make_template_raster(x, y)
    other = make_template_raster(x, y)
    other.values[:] = 42.0

    result = _match_to_template(other, template)

    assert np.array_equal(result.coords["x"].values, template.coords["x"].values)
    assert np.array_equal(result.coords["y"].values, template.coords["y"].values)
    assert np.all(result.values == 42.0)


def test_match_to_template_misaligned_coords():
    """Misaligned coords get reprojected to match template."""
    x_template = np.array([0.0, 1.0, 2.0])
    y_template = np.array([1.0, 0.0])

    x_other = np.array([0.1, 1.1, 2.1])
    y_other = np.array([1.1, 0.1])

    template = make_template_raster(x_template, y_template)
    other = make_template_raster(x_other, y_other)
    other.values[:] = 99.0

    result = _match_to_template(other, template)

    assert np.array_equal(result.coords["x"].values, template.coords["x"].values)
    assert np.array_equal(result.coords["y"].values, template.coords["y"].values)


def test_concat_with_join_exact_after_alignment():
    """After alignment, concat with join='exact' succeeds without NaN stripes."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0])

    template = make_template_raster(x, y)

    mws_50 = template.copy()
    mws_50.values[:] = 5.0
    mws_50 = mws_50.expand_dims(height=[50])

    x_shifted = np.array([0.05, 1.05, 2.05])
    y_shifted = np.array([1.05, 0.05])
    mws_100_misaligned = make_template_raster(x_shifted, y_shifted)
    mws_100_misaligned.values[:] = 10.0

    mws_100_aligned = _match_to_template(mws_100_misaligned, template)
    mws_100_aligned = mws_100_aligned.expand_dims(height=[100])

    result = xr.concat([mws_50, mws_100_aligned], dim="height", join="exact")

    assert result.dims == ("height", "y", "x")
    assert list(result.coords["height"].values) == [50, 100]
    assert np.array_equal(result.coords["x"].values, x)
    assert np.array_equal(result.coords["y"].values, y)


def test_no_nan_stripes_after_alignment():
    """Verify aligned output has no NaN stripes within valid mask."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 1.0, 0.0])

    template = make_template_raster(x, y)
    template.values[:] = 0.0

    x_shifted = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
    y_shifted = np.array([2.1, 1.1, 0.1])
    other = make_template_raster(x_shifted, y_shifted)
    other.values[:] = 1.0

    aligned = _match_to_template(other, template)

    valid_mask = ~np.isnan(template.values)
    aligned_vals = aligned.values

    stripe_cols = 0
    for col_idx in range(aligned_vals.shape[1]):
        col_data = aligned_vals[:, col_idx]
        col_valid = valid_mask[:, col_idx]
        if col_valid.sum() > 0:
            nan_frac = np.isnan(col_data[col_valid]).sum() / col_valid.sum()
            if nan_frac >= 0.95:
                stripe_cols += 1

    stripe_score = stripe_cols / aligned_vals.shape[1] if aligned_vals.shape[1] > 0 else 0

    assert stripe_score == 0, f"Stripe score {stripe_score} > 0 indicates NaN stripes"


def test_merge_with_join_exact_requires_alignment():
    """Merge with join='exact' raises if coords don't match - alignment is required."""
    x1 = np.array([0.0, 1.0, 2.0])
    y1 = np.array([1.0, 0.0])
    x2 = np.array([0.1, 1.1, 2.1])
    y2 = np.array([1.0, 0.0])

    da1 = xr.DataArray(
        np.ones((2, 3)),
        dims=["y", "x"],
        coords={"y": y1, "x": x1},
        name="var1",
    )
    da2 = xr.DataArray(
        np.ones((2, 3)),
        dims=["y", "x"],
        coords={"y": y2, "x": x2},
        name="var2",
    )

    with pytest.raises(ValueError, match="cannot align.*join='exact'"):
        xr.merge([da1, da2], join="exact")


def test_match_to_template_raises_without_xy_coords():
    """Calling _match_to_template without x/y coords raises ValueError."""
    da_no_xy = xr.DataArray(
        np.ones((3, 4)),
        dims=["a", "b"],
        coords={"a": [0, 1, 2], "b": [0, 1, 2, 3]},
    )

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0])
    template = make_template_raster(x, y)

    with pytest.raises(ValueError, match="x/y coords"):
        _match_to_template(da_no_xy, template)
