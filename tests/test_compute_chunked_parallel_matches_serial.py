import numpy as np
import xarray as xr
from types import SimpleNamespace

from cleo.chunk import compute_chunked


def test_compute_chunked_parallel_matches_serial_map_mode_with_extra_dim():
    # Deterministic grid that does NOT divide evenly by chunk_size (tests edge chunks)
    x = np.arange(7, dtype=float)
    y = np.arange(5, dtype=float)

    rng = np.random.default_rng(0)
    v = xr.DataArray(
        rng.normal(size=(len(y), len(x))).astype(float),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    def processing_func(v: xr.DataArray) -> xr.DataArray:
        # Return something assemble-able with an extra dim (non-spatial)
        out0 = (v + 1.0).astype(float)
        out1 = (v * 2.0).astype(float)
        out = xr.concat([out0, out1], dim="band")
        out = out.assign_coords(band=np.array([0, 1], dtype=int))
        out.name = "out"
        return out.transpose("band", "y", "x")

    chunk_size = 3

    serial = compute_chunked(
        self,
        processing_func=processing_func,
        chunk_size=chunk_size,
        max_workers=1,
        show_progress=False,
        v=v,
    )

    parallel = compute_chunked(
        self,
        processing_func=processing_func,
        chunk_size=chunk_size,
        max_workers=4,
        max_in_flight=8,
        show_progress=False,
        v=v,
    )

    # Exact identity: values + coords + attrs + name
    xr.testing.assert_identical(serial, parallel)
