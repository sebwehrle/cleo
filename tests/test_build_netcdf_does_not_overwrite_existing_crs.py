import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from pathlib import Path

from cleo.class_helpers import build_netcdf


class MockParent:
    def __init__(self, path, country, region, crs):
        self.path = Path(path)
        self.country = country
        self.region = region
        self.crs = crs


class MockSelf:
    def __init__(self, parent):
        self.parent = parent
        self.data = None


def test_build_netcdf_reprojects_when_existing_crs_conflicts(tmp_path):
    parent = MockParent(tmp_path, "AUT", None, "EPSG:3035")
    self = MockSelf(parent)

    processed = Path(tmp_path) / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    fname = processed / "WindAtlas_AUT.nc"

    # Existing dataset in EPSG:4326 with degree-like coordinates
    ds = xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={"x": [10.0, 11.0], "y": [45.0, 46.0]},
    )
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_crs("EPSG:4326")
    ds.to_netcdf(fname)

    build_netcdf(self, "WindAtlas")

    assert str(self.data.rio.crs) == str(parent.crs)
    # If CRS was merely overwritten without reprojection, x would still be ~10..11
    assert float(np.max(np.abs(self.data["x"].values))) > 1000.0
