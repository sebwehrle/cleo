# tests/helpers/mocks.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import xarray as xr

from .factories import ds_weibull_params, ds_wind_speed, power_curve_da, template_da, wind_speed_axis


__all__ = [
    # existing public API (kept stable)
    "DummyParent",
    "DummyWindAtlas",
    # new builders / richer mock
    "make_parent",
    "make_atlas",
    "SimpleMockAtlas",
    "make_minimal_atlas",
    "make_assess_atlas",
]


@dataclass
class DummyParent:
    path: Path | str
    country: str
    crs: str
    area: str | None = None

    def get_nuts_area(self, area):
        return None


@dataclass
class DummyWindAtlas:
    parent: DummyParent
    data: xr.Dataset


def make_parent(
    path: Path | str,
    *,
    country: str = "TEST",
    crs: str = "EPSG:4326",
    area: str | None = None,
) -> DummyParent:
    return DummyParent(path=path, country=country, crs=crs, area=area)


def make_atlas(data: xr.Dataset, *, parent: DummyParent | None = None) -> DummyWindAtlas:
    if parent is None:
        parent = make_parent(path=".", country=str(data.attrs.get("country", "TEST")), crs="EPSG:4326")
    return DummyWindAtlas(parent=parent, data=data)


@dataclass
class SimpleMockAtlas:
    """
    A test-focused mock that mimics the subset of Atlas/WindAtlas behaviour used by assess tests.

    Provides:
      - .data: xr.Dataset
      - .parent: object with .country and .crs
      - get_turbine_attribute(turbine_id, key)
      - _set_var(name, da)
      - load_weibull_parameters(height)
    """

    data: xr.Dataset
    country: str = "TEST"
    crs: str = "EPSG:4326"
    weibull_A: xr.DataArray | None = None
    weibull_k: xr.DataArray | None = None
    turbine_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # match what some tests expect (.parent.country / .parent.crs)
        self.parent = SimpleNamespace(country=self.country, crs=self.crs)

    def get_turbine_attribute(self, turbine_id: str, key: str):
        try:
            return self.turbine_attrs[turbine_id][key]
        except KeyError as e:
            raise KeyError(f"Unknown turbine attribute: turbine_id={turbine_id!r}, key={key!r}") from e

    def _set_var(self, name: str, da: xr.DataArray) -> None:
        self.data[name] = da

    def load_weibull_parameters(self, height: float) -> tuple[xr.DataArray, xr.DataArray]:
        if self.weibull_A is None or self.weibull_k is None:
            raise ValueError("MockAtlas missing weibull_A/weibull_k. Provide them when constructing the mock.")
        return (self.weibull_A.sel(height=height), self.weibull_k.sel(height=height))


def make_minimal_atlas(
    *,
    tmp_path: Path | str = ".",
    n: int = 5,
    country: str = "TEST",
    crs: str = "EPSG:4326",
    area: str | None = None,
    wind_speed: float = 7.0,
    include_template: bool = True,
    include_power_curve: bool = False,
    turbine_id: str = "T1",
) -> DummyWindAtlas:
    """
    Minimal DummyWindAtlas:
      - wind_speed(y,x)
      - optionally template(y,x)
      - optionally power_curve(turbine,wind_speed)
    """
    ds = ds_wind_speed(n=n, ws=wind_speed, name="wind_speed")
    ds.attrs["country"] = country

    if include_template:
        ds["template"] = template_da(n=n, fill=1.0)

    if include_power_curve:
        ds["power_curve"] = power_curve_da(u=wind_speed_axis(), turbine_id=turbine_id)

    parent = make_parent(path=tmp_path, country=country, crs=crs, area=area)
    return make_atlas(ds, parent=parent)


def make_assess_atlas(
    *,
    n: int = 5,
    country: str = "TEST",
    crs: str = "EPSG:4326",
    turbine_id: str = "T1",
    hub_height: float = 100.0,
    heights: tuple[float, ...] = (10.0, 50.0, 100.0, 200.0),
    A: float = 8.0,
    k: float = 2.0,
    include_air_density: bool = False,
    air_density_value: float = 1.225,
    include_power_curve: bool = True,
) -> SimpleMockAtlas:
    """
    Convenience constructor for assess-level tests (CF, density correction, Weibull, etc.).
    """
    base = ds_weibull_params(heights=heights, n=n, A=A, k=k, include_template=True, attrs={"country": country})

    if include_power_curve and "power_curve" not in base.data_vars:
        base["power_curve"] = power_curve_da(u=wind_speed_axis(), turbine_id=turbine_id)

    if include_air_density:
        # dims ('height','y','x') to match GWA-style layers used in tests
        rho = xr.full_like(base["weibull_A"], float(air_density_value)).rename("air_density")
        base["air_density"] = rho

    A_da = base["weibull_A"]
    k_da = base["weibull_k"]

    turbine_attrs = {turbine_id: {"hub_height": hub_height}}

    return SimpleMockAtlas(
        data=base,
        country=country,
        crs=crs,
        weibull_A=A_da,
        weibull_k=k_da,
        turbine_attrs=turbine_attrs,
    )
