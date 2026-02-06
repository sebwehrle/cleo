import numpy as np
import xarray as xr
import pytest
from types import SimpleNamespace
from cleo.chunk import compute_chunked

def test_compute_chunked_reassembles_exact_dataarray():
    # Small deterministic grid
    x = np.arange(6, dtype=float)
    y = np.arange(5, dtype=float)
    v = xr.DataArray(
        np.arange(len(y) * len(x), dtype=float).reshape(len(y), len(x)),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="v",
    )
    ds = xr.Dataset({"v": v})

    self = SimpleNamespace(data=ds)

    def identity_chunk(v):
        # Must preserve coords and name
        return v

    out = compute_chunked(self, processing_func=identity_chunk, chunk_size=2, v=ds["v"])
    assert isinstance(out, xr.DataArray)
    xr.testing.assert_identical(out, ds["v"])


def test_compute_chunked_raises_on_coord_drift():
    x = np.arange(6, dtype=float)
    y = np.arange(5, dtype=float)
    v = xr.DataArray(
        np.ones((len(y), len(x)), dtype=float),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="v",
    )
    ds = xr.Dataset({"v": v})
    self = SimpleNamespace(data=ds)

    def drift_x(v):
        # Introduce a tiny coord drift to simulate misalignment/striping risk
        v2 = v.copy()
        v2 = v2.assign_coords(x=v2.coords["x"] + 1e-6)
        return v2

    with pytest.raises(ValueError, match="x-coords mismatch"):
        compute_chunked(self, processing_func=drift_x, chunk_size=2, v=ds["v"])
