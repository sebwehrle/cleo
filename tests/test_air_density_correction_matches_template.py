from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import rioxarray as rxr

import cleo.assess as A


def _write_geotiff(path, arr2d, *, crs="EPSG:4326", transform, nodata=None, dtype=None):
    if dtype is None:
        dtype = arr2d.dtype
    profile = {
        "driver": "GTiff",
        "height": int(arr2d.shape[0]),
        "width": int(arr2d.shape[1]),
        "count": 1,
        "dtype": str(np.dtype(dtype)),
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(dtype), 1)


def test_air_density_correction_matches_template(monkeypatch, tmp_path):
    country = "AUT"

    # Create reference file at the exact path the function expects
    raw_dir = tmp_path / "data" / "raw" / country
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_file = raw_dir / f"{country}_combined-Weibull-A_100.tif"

    # Reference raster: 2x2
    ref_transform = from_origin(0, 2, 1, 1)
    _write_geotiff(reference_file, np.ones((2, 2), dtype=np.float32), transform=ref_transform)

    # Template in self.data: 4x4 (different resolution/coords)
    tpl_transform = from_origin(0, 2, 0.5, 0.5)
    tpl_path = tmp_path / "tpl.tif"
    _write_geotiff(tpl_path, np.zeros((4, 4), dtype=np.float32), transform=tpl_transform)
    template = rxr.open_rasterio(tpl_path).squeeze(drop=True)

    # Misaligned elevation returned by load_elevation: 3x3 on another grid
    elev_transform = from_origin(0, 3, 1, 1)
    elev_path = tmp_path / "elev.tif"
    _write_geotiff(elev_path, np.ones((3, 3), dtype=np.float32) * 100.0, transform=elev_transform)

    elev_da = rxr.open_rasterio(elev_path).squeeze(drop=True)

    def fake_load_elevation(base_dir, iso3, reference_da):
        return elev_da

    monkeypatch.setattr("cleo.loaders.load_elevation", fake_load_elevation)

    # Minimal self object
    self = type("S", (), {})()
    self.parent = type("P", (), {})()
    self.parent.path = tmp_path
    self.parent.country = country
    self.parent.crs = "EPSG:4326"
    self.parent.region = None
    self.parent.get_nuts_region = lambda region: None

    self.data = xr.Dataset({"template": template})

    A.compute_air_density_correction(self, chunk_size=None)

    out = self.data["air_density_correction"]
    assert np.array_equal(out.coords["x"].values, template.coords["x"].values)
    assert np.array_equal(out.coords["y"].values, template.coords["y"].values)
