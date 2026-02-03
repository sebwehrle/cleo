import numpy as np
import xarray as xr
from types import SimpleNamespace

from cleo.utils import compute_chunked


def test_compute_chunked_supports_extra_dims_time_y_x():
    rng = np.random.default_rng(0)

    time = np.arange(3, dtype=int)
    x = np.arange(6, dtype=float)
    y = np.arange(5, dtype=float)

    v = xr.DataArray(
        rng.normal(size=(len(time), len(y), len(x))),
        dims=("time", "y", "x"),
        coords={"time": time, "x": x, "y": y},
        name="v",
    )

    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    def identity(v):
        # Must preserve coords exactly for strict-grid reassembly
        return v

    out = compute_chunked(self, processing_func=identity, chunk_size=2, v=v)

    assert isinstance(out, xr.DataArray)
    assert out.dims == v.dims
    assert np.array_equal(out.coords["time"].values, v.coords["time"].values)
    assert np.array_equal(out.coords["x"].values, v.coords["x"].values)
    assert np.array_equal(out.coords["y"].values, v.coords["y"].values)
    np.testing.assert_allclose(out.values, v.values)
