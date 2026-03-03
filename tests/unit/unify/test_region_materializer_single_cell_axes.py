"""Regression tests for region materialization with single-cell axes."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from cleo.unification.materializers.region import materialize_region


class _AtlasStub:
    def __init__(self, path: Path, region_gdf: gpd.GeoDataFrame) -> None:
        self.path = path
        self._region_gdf = region_gdf

    def get_nuts_region(self, region_id: str) -> gpd.GeoDataFrame:  # noqa: ARG002
        return self._region_gdf


class _UnifierStub:
    chunk_policy = {"y": 2, "x": 2}
    fingerprint_method = "path_mtime_size"


def _write_base_stores(root: Path) -> None:
    """Write tiny canonical base stores with chunking aligned to test policy.

    This fixture intentionally writes ``y/x`` chunks of size ``2`` so opening
    with ``chunk_policy={"y": 2, "x": 2}`` does not trigger xarray
    split-chunk performance warnings during region materializer tests.

    :param root: Temporary atlas root directory.
    :type root: pathlib.Path
    :returns: ``None``.
    :rtype: None
    """
    y = np.array([30.0, 20.0, 10.0], dtype=np.float64)
    x = np.array([0.0, 10.0, 20.0], dtype=np.float64)

    wind_base = xr.Dataset(
        data_vars={
            "weibull_A": (("height", "y", "x"), np.arange(9, dtype=np.float32).reshape(1, 3, 3)),
            "weibull_k": (("height", "y", "x"), np.full((1, 3, 3), 2.0, dtype=np.float32)),
        },
        coords={
            "height": np.array([100], dtype=np.int32),
            "y": y,
            "x": x,
        },
        attrs={
            "store_state": "complete",
            "inputs_id": "wind-inputs-id",
            "grid_id": "base-grid-id",
        },
    )

    land_base = xr.Dataset(
        data_vars={
            "valid_mask": (("y", "x"), np.ones((3, 3), dtype=bool)),
            "elevation": (("y", "x"), np.full((3, 3), 500.0, dtype=np.float32)),
        },
        coords={"y": y, "x": x},
        attrs={
            "store_state": "complete",
            "inputs_id": "land-inputs-id",
            "grid_id": "base-grid-id",
        },
    )

    wind_base = wind_base.chunk({"height": 1, "y": 2, "x": 2})
    land_base = land_base.chunk({"y": 2, "x": 2})

    wind_base.to_zarr(root / "wind.zarr", mode="w", consolidated=False)
    land_base.to_zarr(root / "landscape.zarr", mode="w", consolidated=False)


@pytest.mark.parametrize(
    ("bounds", "expected_shape"),
    [
        ((9.0, 9.0, 11.0, 31.0), (3, 1)),  # x-axis single cell
        ((-1.0, 19.0, 21.0, 21.0), (1, 3)),  # y-axis single cell
        ((9.0, 19.0, 11.0, 21.0), (1, 1)),  # both axes single cell
    ],
)
def test_region_materialization_handles_single_cell_axes(
    tmp_path: Path,
    bounds: tuple[float, float, float, float],
    expected_shape: tuple[int, int],
) -> None:
    """Single-cell axis subsets should materialize without transform-index errors."""
    _write_base_stores(tmp_path)

    region_gdf = gpd.GeoDataFrame({"geometry": [box(*bounds)]}, crs="EPSG:3035")
    atlas = _AtlasStub(path=tmp_path, region_gdf=region_gdf)
    unifier = _UnifierStub()

    materialize_region(unifier, atlas, "ATX2")

    wind_region_path = tmp_path / "regions" / "ATX2" / "wind.zarr"
    land_region_path = tmp_path / "regions" / "ATX2" / "landscape.zarr"
    wind_region = xr.open_zarr(wind_region_path, consolidated=False)
    land_region = xr.open_zarr(land_region_path, consolidated=False)

    try:
        assert (wind_region.sizes["y"], wind_region.sizes["x"]) == expected_shape
        assert (land_region.sizes["y"], land_region.sizes["x"]) == expected_shape
        assert wind_region.attrs["store_state"] == "complete"
        assert land_region.attrs["store_state"] == "complete"
        assert wind_region.attrs["region_id"] == "ATX2"
        assert land_region.attrs["region_id"] == "ATX2"
        assert wind_region.attrs["base_wind_inputs_id"] == "wind-inputs-id"
        assert land_region.attrs["base_land_inputs_id"] == "land-inputs-id"

        # Regression guard: at least part of the selected window remains valid after masking.
        assert bool(wind_region["weibull_A"].notnull().any().compute()) is True
    finally:
        wind_region.close()
        land_region.close()
