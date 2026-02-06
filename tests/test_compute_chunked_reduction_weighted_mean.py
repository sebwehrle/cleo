import numpy as np
import xarray as xr
import pytest
from types import SimpleNamespace

from cleo.chunk import compute_chunked


def test_compute_chunked_weighted_mean_reduction_matches_global_mean():
    x = np.arange(6, dtype=float)
    y = np.arange(5, dtype=float)
    v = xr.DataArray(
        np.arange(len(y) * len(x), dtype=float).reshape(len(y), len(x)),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    out = compute_chunked(
        self,
        processing_func=lambda v: v.mean(),
        chunk_size=2,
        allow_reduction=True,
        v=v,
    )

    assert isinstance(out, xr.DataArray)
    assert out.dims == ()
    assert float(out.item()) == pytest.approx(float(v.mean().item()))
