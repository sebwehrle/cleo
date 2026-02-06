import pytest
import numpy as np
import xarray as xr
from types import SimpleNamespace
import rioxarray  # noqa: F401
from cleo.spatial import clip_to_geometry

def test_clip_to_geometry_missing_path_raises():
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2)))}, coords={"x": [0, 1], "y": [0, 1]})
    ds = ds.rio.write_crs("EPSG:4326")
    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="EPSG:4326"))

    with pytest.raises(ValueError, match="Cannot read clipping geometry"):
        clip_to_geometry(dummy, "nonexistent.shp")
