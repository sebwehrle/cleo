from __future__ import annotations

import numpy as np
import xarray as xr


def assert_same_coords(a: xr.DataArray | xr.Dataset, b: xr.DataArray | xr.Dataset):
    for d in a.dims:
        if d not in b.dims:
            raise AssertionError(f"Missing dim {d}")
    for c in a.coords:
        if c in b.coords:
            if not np.array_equal(a[c].values, b[c].values):
                raise AssertionError(f"Coord mismatch: {c}")


def assert_has_crs(x):
    crs = getattr(x, "rio", None)
    if crs is None:
        raise AssertionError("Object has no .rio accessor; expected rioxarray.")
