from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr


def xy_coords(n: int = 5, x0: float = 0.0, y0: float = 0.0, dx: float = 1.0, dy: float = 1.0):
    x = x0 + dx * np.arange(n)
    y = y0 + dy * np.arange(n)
    return x, y


def da_xy(values: np.ndarray | None = None, *, n: int = 5, name: str = "v", attrs: dict[str, Any] | None = None):
    x, y = xy_coords(n=n)
    if values is None:
        values = np.zeros((n, n), dtype=float)
    da = xr.DataArray(values, dims=("y", "x"), coords={"x": x, "y": y}, name=name)
    if attrs:
        da.attrs.update(attrs)
    return da


def ds_wind_speed(*, n: int = 5, ws: float = 7.0, name: str = "wind_speed"):
    da = da_xy(n=n, name=name)
    da[:] = ws
    return da.to_dataset()


def da_xy_with_crs(
    values: np.ndarray | None = None,
    *,
    n: int = 5,
    name: str = "v",
    crs: str = "EPSG:4326",
    attrs: dict[str, Any] | None = None,
):
    da = da_xy(values, n=n, name=name, attrs=attrs)
    return da.rio.write_crs(crs)
