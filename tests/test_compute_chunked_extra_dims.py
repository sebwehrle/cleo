import numpy as np
import xarray as xr
from types import SimpleNamespace

from cleo.chunk import compute_chunked


def test_compute_chunked_stitches_extra_dims():
    x = np.arange(6, dtype=float)
    y = np.arange(5, dtype=float)
    h = np.array([50.0, 100.0], dtype=float)

    v = xr.DataArray(
        np.arange(len(h) * len(y) * len(x), dtype=float).reshape(len(h), len(y), len(x)),
        dims=("height", "y", "x"),
        coords={"height": h, "x": x, "y": y},
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    def f(v):
        # keep dims, just transform values
        return v * 2.0

    out = compute_chunked(self, processing_func=f, chunk_size=2, v=v)

    assert out.dims == ("height", "y", "x")
    np.testing.assert_array_equal(out.coords["x"].values, x)
    np.testing.assert_array_equal(out.coords["y"].values, y)
    np.testing.assert_array_equal(out.coords["height"].values, h)

    np.testing.assert_allclose(out.values, (v * 2.0).values)
