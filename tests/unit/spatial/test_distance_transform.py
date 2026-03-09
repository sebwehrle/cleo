from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.spatial import distance_to_positive_mask


def _with_crs(da: xr.DataArray, crs: str = "epsg:3035") -> xr.DataArray:
    return da.rio.write_crs(crs)


def test_distance_to_positive_mask_basic_2d() -> None:
    source = xr.DataArray(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((3, 3), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
        name="valid_mask",
    )
    source = _with_crs(source)
    valid_mask = _with_crs(valid_mask)

    out = distance_to_positive_mask(source, valid_mask)
    expected = np.array(
        [
            [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
            [1.0, 0.0, 1.0],
            [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
        ]
    )
    np.testing.assert_allclose(out.values, expected, rtol=0, atol=1e-9)
    assert out.attrs["units"] == "m"


def test_distance_to_positive_mask_no_targets_returns_nan() -> None:
    source = xr.DataArray(
        np.zeros((2, 2), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    source = _with_crs(source)
    valid_mask = _with_crs(valid_mask)

    out = distance_to_positive_mask(source, valid_mask)
    assert np.isnan(out.values).all()


def test_distance_to_positive_mask_all_targets_returns_zero() -> None:
    source = xr.DataArray(
        np.ones((2, 2), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    source = _with_crs(source)
    valid_mask = _with_crs(valid_mask)

    out = distance_to_positive_mask(source, valid_mask)
    np.testing.assert_allclose(out.values, 0.0, rtol=0, atol=0)


def test_distance_to_positive_mask_supports_one_extra_dim() -> None:
    source = xr.DataArray(
        np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float64,
        ),
        dims=("band", "y", "x"),
        coords={"band": [10, 20], "y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((3, 3), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
    )
    source = _with_crs(source)
    valid_mask = _with_crs(valid_mask)

    out = distance_to_positive_mask(source, valid_mask)
    assert out.dims == ("band", "y", "x")
    assert out.sizes["band"] == 2
    np.testing.assert_allclose(out.sel(band=10).values[1, 1], 0.0)
    np.testing.assert_allclose(out.sel(band=20).values[0, 0], 0.0)


def test_distance_to_positive_mask_rejects_irregular_grid() -> None:
    source = xr.DataArray(
        np.ones((2, 3), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0, 3.0]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((2, 3), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0, 3.0]},
    )
    source = _with_crs(source)
    valid_mask = _with_crs(valid_mask)

    with pytest.raises(ValueError, match="regularly spaced"):
        distance_to_positive_mask(source, valid_mask)


def test_distance_to_positive_mask_rejects_mismatched_coords() -> None:
    source = xr.DataArray(
        np.ones((2, 2), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 2.0]},
    )
    source = _with_crs(source)
    valid_mask = _with_crs(valid_mask)

    with pytest.raises(ValueError, match="x coordinates must match"):
        distance_to_positive_mask(source, valid_mask)


def test_distance_to_positive_mask_rejects_multiple_extra_dims() -> None:
    source = xr.DataArray(
        np.ones((2, 2, 2, 2), dtype=np.float64),
        dims=("band", "time", "y", "x"),
        coords={"band": [1, 2], "time": [0, 1], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    source = _with_crs(source)
    valid_mask = _with_crs(valid_mask)

    with pytest.raises(ValueError, match="at most one non-spatial dimension"):
        distance_to_positive_mask(source, valid_mask)


def test_distance_to_positive_mask_rejects_geographic_crs() -> None:
    source = xr.DataArray(
        np.ones((2, 2), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [47.0, 47.1], "x": [15.0, 15.1]},
        name="roads",
    )
    valid_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [47.0, 47.1], "x": [15.0, 15.1]},
    )
    source = _with_crs(source, "epsg:4326")
    valid_mask = _with_crs(valid_mask, "epsg:4326")

    with pytest.raises(ValueError, match="projected CRS"):
        distance_to_positive_mask(source, valid_mask)
