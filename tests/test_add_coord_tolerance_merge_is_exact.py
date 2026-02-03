import numpy as np
import xarray as xr
from types import SimpleNamespace

from cleo.utils import add


def test_add_coord_tolerance_merge_is_exact_grid():
    a = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="a",
    ).rio.write_crs("EPSG:4326")

    # Tiny noise in x coords (should be treated as equivalent; no clipping path)
    b = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
        coords={"x": [1e-12, 1.0 + 1e-12], "y": [0.0, 1.0]},
        name="b",
    ).rio.write_crs("EPSG:4326")

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

    assert "a" in self.data.data_vars
    assert "b" in self.data.data_vars
    assert np.array_equal(self.data["b"].coords["x"].values, self.data["a"].coords["x"].values)
    assert np.array_equal(self.data["b"].coords["y"].values, self.data["a"].coords["y"].values)
