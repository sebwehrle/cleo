"""Unit tests for direct rotor CF quadrature and REWS outputs."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import capacity_factors, rews_mps


def _make_stacks() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    heights = np.array([10.0, 50.0, 100.0, 150.0, 200.0], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 1.0], dtype=np.float64)
    A = xr.DataArray(
        np.full((5, 2, 2), 8.0, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_A",
    )
    k = xr.DataArray(
        np.full((5, 2, 2), 2.0, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_k",
    )
    rho = xr.DataArray(
        np.full((5, 2, 2), 1.225, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="rho",
    )
    return A, k, rho


def test_capacity_factors_rotor_node_average_runs() -> None:
    A, k, rho = _make_stacks()
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    power_curves = np.array([np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)], dtype=np.float64)
    out = capacity_factors(
        A_stack=A,
        k_stack=k,
        u_grid=u_grid,
        turbine_ids=("T1",),
        hub_heights_m=np.array([120.0], dtype=np.float64),
        power_curves=power_curves,
        method="rotor_node_average",
        rotor_diameters_m=np.array([220.0], dtype=np.float64),
        rho_stack=rho,
        air_density=True,
        rews_n=6,
    )
    assert out.name == "capacity_factors"
    assert out.attrs["cleo:cf_method"] == "rotor_node_average"
    assert np.all(np.isfinite(out.values))


def test_rews_mps_runs_and_returns_units() -> None:
    A, k, rho = _make_stacks()
    out = rews_mps(
        A_stack=A,
        k_stack=k,
        turbine_ids=("T1",),
        hub_heights_m=np.array([120.0], dtype=np.float64),
        rotor_diameters_m=np.array([220.0], dtype=np.float64),
        rho_stack=rho,
        air_density=True,
        rews_n=6,
    )
    assert out.name == "rews_mps"
    assert out.attrs["units"] == "m/s"
    assert np.all(np.isfinite(out.values))


def test_capacity_factors_rotor_moment_matched_weibull_runs() -> None:
    A, k, rho = _make_stacks()
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    power_curves = np.array([np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)], dtype=np.float64)
    out = capacity_factors(
        A_stack=A,
        k_stack=k,
        u_grid=u_grid,
        turbine_ids=("T1",),
        hub_heights_m=np.array([120.0], dtype=np.float64),
        power_curves=power_curves,
        method="rotor_moment_matched_weibull",
        rotor_diameters_m=np.array([220.0], dtype=np.float64),
        rho_stack=rho,
        air_density=True,
        rews_n=6,
    )
    assert out.name == "capacity_factors"
    assert out.attrs["cleo:cf_method"] == "rotor_moment_matched_weibull"
    assert np.all(np.isfinite(out.values))


def test_capacity_factors_momentmatch_vs_direct_close_on_constant_profile() -> None:
    A, k, rho = _make_stacks()
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    power_curves = np.array([np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)], dtype=np.float64)
    kwargs = dict(
        A_stack=A,
        k_stack=k,
        u_grid=u_grid,
        turbine_ids=("T1",),
        hub_heights_m=np.array([120.0], dtype=np.float64),
        power_curves=power_curves,
        rotor_diameters_m=np.array([220.0], dtype=np.float64),
        rho_stack=rho,
        air_density=True,
        rews_n=12,
    )
    direct = capacity_factors(method="rotor_node_average", **kwargs)
    mm = capacity_factors(method="rotor_moment_matched_weibull", **kwargs)
    diff = np.abs((direct - mm).values)
    assert float(np.nanmax(diff)) <= 0.02
