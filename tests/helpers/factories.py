# tests/helpers/factories.py
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import xarray as xr


__all__ = [
    # existing public API (kept stable)
    "xy_coords",
    "da_xy",
    "ds_wind_speed",
    "da_xy_with_crs",
    # new helpers
    "wind_speed_axis",
    "heights_axis",
    "template_da",
    "ds_template_xy",
    "weibull_params_da",
    "ds_weibull_params",
    "power_curve_step",
    "power_curve_top_hat",
    "power_curve_linear",
    "power_curve_da",
]


# -----------------------------------------------------------------------------
# Existing helpers (kept as-is in spirit and signature)
# -----------------------------------------------------------------------------


def xy_coords(n: int = 5, x0: float = 0.0, y0: float = 0.0, dx: float = 1.0, dy: float = 1.0):
    x = x0 + dx * np.arange(n)
    y = y0 + dy * np.arange(n)
    return x, y


def da_xy(
    values: np.ndarray | None = None,
    *,
    n: int = 5,
    name: str = "v",
    attrs: dict[str, Any] | None = None,
):
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
    """
    Same intent as before, but now imports rioxarray lazily so importing this
    module does not require the optional dependency.
    """
    da = da_xy(values, n=n, name=name, attrs=attrs)
    try:
        # noqa: F401 - activates .rio accessor
        pass  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "da_xy_with_crs requires optional dependency 'rioxarray'. "
            "In tests, prefer pytest.importorskip('rioxarray') or a shared optional helper."
        ) from e
    return da.rio.write_crs(crs)


# -----------------------------------------------------------------------------
# New shared factories
# -----------------------------------------------------------------------------


def wind_speed_axis(
    u_max: float = 40.0,
    du: float = 0.5,
    *,
    u_min: float = 0.0,
    inclusive: bool = True,
) -> np.ndarray:
    """
    Common wind-speed grid factory.

    - Default matches the common pattern: 0.0..40.0 in 0.5 steps (inclusive).
    - Uses stable arithmetic so that labels match exactly (important for cleo contracts).
    """
    if du <= 0:
        raise ValueError("du must be positive.")
    if u_max < u_min:
        raise ValueError("u_max must be >= u_min.")

    n_steps = int(round((u_max - u_min) / du))
    end = u_min + n_steps * du
    if inclusive:
        # include endpoint if it's (numerically) reachable
        # ensure exact labels via integer-step construction
        return u_min + du * np.arange(n_steps + 1)
    return u_min + du * np.arange(n_steps if end == u_max else n_steps + 1)


def heights_axis(heights: Iterable[float] | None = None) -> np.ndarray:
    """
    Canonical height axis for test fixtures. Defaults to a sensible, widely used set.
    """
    if heights is None:
        heights = (10.0, 50.0, 100.0, 200.0)
    h = np.asarray(list(heights), dtype=float)
    if h.ndim != 1 or h.size == 0:
        raise ValueError("heights must be a non-empty 1D iterable.")
    return h


def template_da(
    *,
    n: int = 5,
    fill: float = 1.0,
    name: str = "template",
    dtype: Any = np.float32,
    x0: float = 0.0,
    y0: float = 0.0,
    dx: float = 1.0,
    dy: float = 1.0,
) -> xr.DataArray:
    x, y = xy_coords(n=n, x0=x0, y0=y0, dx=dx, dy=dy)
    values = np.full((n, n), fill, dtype=dtype)
    return xr.DataArray(values, dims=("y", "x"), coords={"x": x, "y": y}, name=name)


def ds_template_xy(
    *,
    n: int = 5,
    fill: float = 1.0,
    with_crs: bool = False,
    crs: str = "EPSG:4326",
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """
    Dataset containing only a 'template' raster (y/x grid).
    """
    tmpl = template_da(n=n, fill=fill)
    if with_crs:
        try:
            import rioxarray  # type: ignore  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError("ds_template_xy(with_crs=True) requires optional dependency 'rioxarray'.") from e
        tmpl = tmpl.rio.write_crs(crs)
    ds = xr.Dataset(data_vars={"template": tmpl})
    if attrs:
        ds.attrs.update(attrs)
    return ds


def _field_3d(
    value: float | np.ndarray | Callable[[float], float] | Callable[[float, int, int], float],
    *,
    heights: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Normalize a scalar/array/callable into (n_heights, ny, nx).
    """
    nh = int(heights.size)

    if isinstance(value, (int, float, np.floating)):
        return np.full((nh, ny, nx), float(value), dtype=float)

    arr = np.asarray(value)
    if arr.shape == (nh, ny, nx):
        return arr.astype(float, copy=False)

    if callable(value):
        out = np.empty((nh, ny, nx), dtype=float)
        # Detect callable signature by trying 1-arg first (height), then 3-arg (h,y,x)
        try:
            for i, h in enumerate(heights):
                out[i, :, :] = float(value(float(h)))  # type: ignore[misc]
            return out
        except TypeError:
            for i, h in enumerate(heights):
                for iy in range(ny):
                    for ix in range(nx):
                        out[i, iy, ix] = float(value(float(h), iy, ix))  # type: ignore[misc]
            return out

    raise ValueError("Unsupported value for 3D field. Provide scalar, ndarray(shape=(nh,ny,nx)), or callable.")


def weibull_params_da(
    *,
    heights: Iterable[float] | None = None,
    n: int = 5,
    A: float | np.ndarray | Callable[[float], float] | Callable[[float, int, int], float] = 8.0,
    k: float | np.ndarray | Callable[[float], float] | Callable[[float, int, int], float] = 2.0,
    A_name: str = "weibull_A",
    k_name: str = "weibull_k",
    x0: float = 0.0,
    y0: float = 0.0,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Build Weibull parameter DataArrays with dims ('height','y','x') and names
    matching cleo’s conventions: 'weibull_A' and 'weibull_k'.
    """
    h = heights_axis(heights)
    x, y = xy_coords(n=n, x0=x0, y0=y0, dx=dx, dy=dy)

    A3 = _field_3d(A, heights=h, ny=n, nx=n)
    k3 = _field_3d(k, heights=h, ny=n, nx=n)

    A_da = xr.DataArray(A3, dims=("height", "y", "x"), coords={"height": h, "y": y, "x": x}, name=A_name)
    k_da = xr.DataArray(k3, dims=("height", "y", "x"), coords={"height": h, "y": y, "x": x}, name=k_name)
    return A_da, k_da


def ds_weibull_params(
    *,
    heights: Iterable[float] | None = None,
    n: int = 5,
    A: float | np.ndarray | Callable[[float], float] | Callable[[float, int, int], float] = 8.0,
    k: float | np.ndarray | Callable[[float], float] | Callable[[float, int, int], float] = 2.0,
    include_template: bool = True,
    template_fill: float = 1.0,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """
    Dataset containing:
      - weibull_A(height,y,x)
      - weibull_k(height,y,x)
      - optionally template(y,x)
    """
    A_da, k_da = weibull_params_da(heights=heights, n=n, A=A, k=k)
    data_vars: dict[str, xr.DataArray] = {A_da.name: A_da, k_da.name: k_da}
    if include_template:
        data_vars["template"] = template_da(n=n, fill=template_fill)
    ds = xr.Dataset(data_vars=data_vars)
    if attrs:
        ds.attrs.update(attrs)
    return ds


def power_curve_step(u: np.ndarray, *, u0: float) -> np.ndarray:
    """
    Step curve: 0 below u0, 1 at/above u0.
    """
    u = np.asarray(u, dtype=float)
    return np.where(u >= u0, 1.0, 0.0)


def power_curve_top_hat(u: np.ndarray, *, u_in: float, u_out: float) -> np.ndarray:
    """
    Top-hat: 1 for u_in <= u < u_out, else 0.
    """
    u = np.asarray(u, dtype=float)
    return np.where((u >= u_in) & (u < u_out), 1.0, 0.0)


def power_curve_linear(u: np.ndarray, *, u_cut_in: float = 3.0, u_rated: float = 12.0, u_cut_out: float = 25.0):
    """
    Simple linear ramp:
      - 0 for u < u_cut_in
      - ramps to 1 between [u_cut_in, u_rated]
      - 1 for [u_rated, u_cut_out]
      - 0 for u > u_cut_out
    """
    u = np.asarray(u, dtype=float)
    p = np.zeros_like(u, dtype=float)

    ramp = (u >= u_cut_in) & (u < u_rated)
    p[ramp] = (u[ramp] - u_cut_in) / max(u_rated - u_cut_in, 1e-12)

    rated = (u >= u_rated) & (u <= u_cut_out)
    p[rated] = 1.0

    return p


def power_curve_da(
    *,
    u: np.ndarray | None = None,
    p: np.ndarray | None = None,
    turbine_id: str = "T1",
    name: str = "power_curve",
) -> xr.DataArray:
    """
    Create a 'power_curve' DataArray with dims ('turbine','wind_speed'), matching cleo usage.
    """
    if u is None:
        u = wind_speed_axis()
    u = np.asarray(u, dtype=float)

    if p is None:
        # default: a gentle linear curve
        p = power_curve_linear(u)

    p = np.asarray(p, dtype=float)
    if p.shape != u.shape:
        raise ValueError(f"p must have same shape as u. Got p{p.shape} vs u{u.shape}.")

    return xr.DataArray(
        p.reshape(1, -1),
        dims=("turbine", "wind_speed"),
        coords={"turbine": [turbine_id], "wind_speed": u},
        name=name,
    )
