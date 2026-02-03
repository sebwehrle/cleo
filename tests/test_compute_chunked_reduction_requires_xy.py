import numpy as np
import xarray as xr
import pytest
from types import SimpleNamespace

from cleo.utils import compute_chunked


def test_compute_chunked_reduction_requires_xy_dims():
    x = np.arange(6, dtype=float)
    y = np.arange(5, dtype=float)
    v = xr.DataArray(
        np.arange(len(y) * len(x), dtype=float).reshape(len(y), len(x)),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    with pytest.raises(ValueError, match=r"requires results to include dims 'x' and 'y'"):
        compute_chunked(self, processing_func=lambda v: v.mean(), chunk_size=2, v=v)
