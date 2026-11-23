"""prepare_clc_to_wind_grid reads only a window of a large CLC source without changing the result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from cleo.unification.clc_io import prepare_clc_to_wind_grid


def _write_source(path: Path) -> None:
    """600 x 600 cells of 100 m in EPSG:3035 with codes > 44 (no compact-id expansion), nodata -1."""
    src = np.random.default_rng(0).integers(100, 500, size=(600, 600), dtype=np.int16)
    src[0, :] = -1
    src[201, 201] = -1  # sampled by the first EPSG:3035 reference cell: nodata must stay NaN
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=600,
        width=600,
        count=1,
        dtype="int16",
        crs="EPSG:3035",
        transform=from_origin(4_000_000, 2_700_000, 100, 100),
        nodata=-1,
    ) as dst:
        dst.write(src, 1)


def _reference(crs: str) -> xr.DataArray:
    """20 x 20 cells of 250 m inside the source, not aligned to source cell edges."""
    x0, y0 = Transformer.from_crs("EPSG:3035", crs, always_xy=True).transform(4_020_175, 2_679_825)
    x = x0 + 250 * np.arange(20)
    y = y0 - 250 * np.arange(20)
    ref = xr.DataArray(np.ones((20, 20), np.float32), dims=("y", "x"), coords={"y": y, "x": x})
    return ref.rio.write_crs(crs)


@pytest.mark.parametrize("crs", ["EPSG:3035", "EPSG:32633"])
def test_prepare_from_large_source_matches_full_reprojection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crs: str
) -> None:
    source_path = tmp_path / "clc_source.tif"
    _write_source(source_path)
    ref = _reference(crs)
    full = xr.open_dataarray(source_path, engine="rasterio").squeeze(drop=True).astype(np.float32)
    expected = full.rio.reproject_match(ref, resampling=Resampling.nearest, nodata=np.nan)

    # The float32 cast is where the full source used to be materialized; it must only see a window.
    cast_sizes: list[int] = []
    astype = xr.DataArray.astype

    def _spy(self, *args, **kwargs):  # noqa: ANN001, ANN202
        cast_sizes.append(self.size)
        return astype(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "astype", _spy)
    prepared = prepare_clc_to_wind_grid(
        source_path=source_path,
        prepared_path=tmp_path / "prepared.tif",
        ref=ref,
        atlas_path=tmp_path,
        source_crs="EPSG:3035",
    )
    monkeypatch.undo()

    assert max(cast_sizes) < full.size // 20
    if crs == "EPSG:3035":
        assert np.isnan(expected.values).any()
    with rasterio.open(prepared) as ds:
        np.testing.assert_array_equal(ds.read(1), expected.values)


def test_prepare_rejects_source_far_from_wind_grid(tmp_path: Path) -> None:
    source_path = tmp_path / "clc_source.tif"
    _write_source(source_path)
    ref = _reference("EPSG:3035")
    with pytest.raises(RuntimeError, match="does not overlap"):
        prepare_clc_to_wind_grid(
            source_path=source_path,
            prepared_path=tmp_path / "disjoint.tif",
            ref=ref.assign_coords(x=ref.x + 1_000_000),
            atlas_path=tmp_path,
            source_crs="EPSG:3035",
        )
