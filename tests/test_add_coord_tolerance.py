import numpy as np
import xarray as xr
from types import SimpleNamespace
from cleo.utils import add

def test_add_does_not_clip_for_tiny_coord_noise():
    a = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"x": [0.0, 1.0], "y": [0.0, 1.0]}, name="a").rio.write_crs("EPSG:4326")
    b = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"x": [1e-12, 1.0 + 1e-12], "y": [0.0, 1.0]}, name="b").rio.write_crs("EPSG:4326")

    class _Parent:
        region = None
        def get_nuts_country(self):
            raise AssertionError("Clipping should not be triggered for tiny coord noise")

    self = SimpleNamespace(
        data=xr.Dataset({"a": a}).rio.write_crs("EPSG:4326"),
        crs="EPSG:4326",
        parent=_Parent(),
    )

    add(self, b, "b")
    assert "b" in self.data.data_vars
