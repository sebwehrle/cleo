import numpy as np
import xarray as xr
from types import SimpleNamespace

from cleo.chunk import compute_chunked


def test_compute_chunked_reduction_allowed_returns_scalar():
    x = np.arange(6, dtype=float)
    y = np.arange(5, dtype=float)
    v = xr.DataArray(
        np.ones((len(y), len(x)), dtype=float),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    out = compute_chunked(self, processing_func=lambda v: v.sum(), chunk_size=2, allow_reduction=True, reduction="sum", v=v)
    assert out.dims == ()
    assert float(out.item()) == float(v.sum().item())
