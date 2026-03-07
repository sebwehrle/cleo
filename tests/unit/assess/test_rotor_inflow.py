"""Seam-level tests for rotor inflow factories and inflow integration."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import (
    RHO_0,
    _build_direct_inflow,
    _build_hub_inflow,
    _build_momentmatch_inflow,
    _build_rews_inflow,
    _direct_cf_and_rews_for_turbine,
    _integrate_cf_with_density_correction,
    _integrate_cf_no_density,
    _momentmatch_cf_and_rews_for_turbine,
    _rews_moment_factor,
    capacity_factors_v1,
    integrate_cf_from_inflow,
    interpolate_weibull_params_to_height,
    weibull_probability_density,
)
from cleo.domains import _cf_spec_matches
from cleo.wind_metrics import resolve_cf_spec


def _make_stacks(
    *,
    a_value: float = 8.0,
    k_value: float = 2.0,
    rho_value: float = RHO_0,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Create constant test stacks for Weibull A/k and density."""
    heights = np.array([50.0, 100.0, 150.0], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 1.0], dtype=np.float64)
    A = xr.DataArray(
        np.full((3, 2, 2), a_value, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_A",
    )
    k = xr.DataArray(
        np.full((3, 2, 2), k_value, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_k",
    )
    rho = xr.DataArray(
        np.full((3, 2, 2), rho_value, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="rho",
    )
    return A, k, rho


def test_hub_inflow_preserves_weibull_params() -> None:
    """Hub inflow keeps hub-interpolated A/k when air density is disabled."""
    A, k, _rho = _make_stacks()
    H = 100.0
    inflow = _build_hub_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=H,
        rho_stack=None,
        air_density=False,
    )
    A_hub, k_hub = interpolate_weibull_params_to_height(A, k, H)
    np.testing.assert_allclose(inflow.A_eff.values, A_hub.values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(inflow.k_eff.values, k_hub.values, rtol=0.0, atol=0.0)


def test_hub_inflow_sets_density_scale_when_enabled() -> None:
    """Hub inflow stores legacy density scale without mutating ``A_eff``."""
    A, k, rho = _make_stacks(rho_value=1.5)
    H = 100.0
    inflow = _build_hub_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=H,
        rho_stack=rho,
        air_density=True,
    )
    expected_scale = (1.5 / RHO_0) ** (1.0 / 3.0)
    assert inflow.density_scale is not None
    np.testing.assert_allclose(inflow.A_eff.values, A.sel(height=H).values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(inflow.density_scale.values, expected_scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inflow.k_eff.values, k.sel(height=H).values, rtol=0.0, atol=0.0)


def test_rews_inflow_applies_moment_factor() -> None:
    """REWS inflow scale matches ``A_hub * rews_factor`` when density is disabled."""
    A, k, _rho = _make_stacks()
    H = 100.0
    D = 80.0
    rews_n = 12
    inflow = _build_rews_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=H,
        rotor_diameter=D,
        rews_n=rews_n,
        rho_stack=None,
        air_density=False,
    )
    A_hub, _k_hub = interpolate_weibull_params_to_height(A, k, H)
    factor = _rews_moment_factor(
        A_stack=A,
        k_stack=k,
        hub_height=H,
        rotor_diameter=D,
        n=rews_n,
    )
    np.testing.assert_allclose(inflow.A_eff.values, (A_hub * factor).values, rtol=1e-12, atol=1e-12)


def test_direct_inflow_has_expected_nodes_and_weights() -> None:
    """Direct inflow returns node arrays with normalized weights."""
    A, k, _rho = _make_stacks()
    inflow = _build_direct_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=100.0,
        rotor_diameter=80.0,
        rews_n=10,
        rho_stack=None,
        air_density=False,
        vertical_policy=None,
    )
    assert inflow.A_nodes.sizes["node"] == 10
    assert inflow.k_nodes.sizes["node"] == 10
    np.testing.assert_allclose(np.sum(inflow.weights), 1.0, rtol=1e-12, atol=1e-12)


def test_integrate_single_weibull_matches_current_integrator() -> None:
    """Single inflow integration matches the existing no-density primitive."""
    A, k, _rho = _make_stacks()
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    p_curve = np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)
    inflow = _build_hub_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=100.0,
        rho_stack=None,
        air_density=False,
    )
    got = integrate_cf_from_inflow(
        inflow=inflow,
        u_grid=u_grid,
        p_curve=p_curve,
        loss_factor=0.95,
    )
    pdf = weibull_probability_density(u_grid, inflow.k_eff, inflow.A_eff)
    expected = _integrate_cf_no_density(
        pdf=pdf,
        u_grid=u_grid,
        p_curve=p_curve,
        loss_factor=0.95,
    )
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12, atol=1e-12)


def test_integrate_single_weibull_with_density_matches_legacy_integrator() -> None:
    """Single inflow density path matches existing density-corrected primitive."""
    A, k, rho = _make_stacks(rho_value=1.4)
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    p_curve = np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)
    inflow = _build_hub_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=100.0,
        rho_stack=rho,
        air_density=True,
    )
    assert inflow.density_scale is not None
    got = integrate_cf_from_inflow(
        inflow=inflow,
        u_grid=u_grid,
        p_curve=p_curve,
        loss_factor=0.95,
    )
    pdf = weibull_probability_density(u_grid, inflow.k_eff, inflow.A_eff)
    expected = _integrate_cf_with_density_correction(
        pdf=pdf,
        u_grid=u_grid,
        p_curve=p_curve,
        c=inflow.density_scale,
        loss_factor=0.95,
    )
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12, atol=1e-12)


def test_integrate_weighted_nodes_matches_direct_helper() -> None:
    """Weighted-node integration equals existing direct helper output."""
    A, k, _rho = _make_stacks()
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    p_curve = np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)
    H = 100.0
    D = 80.0
    rews_n = 8
    loss_factor = 0.93

    inflow = _build_direct_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=H,
        rotor_diameter=D,
        rews_n=rews_n,
        rho_stack=None,
        air_density=False,
        vertical_policy=None,
    )
    got = integrate_cf_from_inflow(
        inflow=inflow,
        u_grid=u_grid,
        p_curve=p_curve,
        loss_factor=loss_factor,
    )
    expected, _rews = _direct_cf_and_rews_for_turbine(
        A_stack=A,
        k_stack=k,
        rho_stack=None,
        u_grid=u_grid,
        p_curve=p_curve,
        hub_height=H,
        rotor_diameter=D,
        rews_n=rews_n,
        loss_factor=loss_factor,
        vertical_policy=None,
        compute_cf=True,
    )
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12, atol=1e-12)


def test_momentmatch_inflow_matches_current_helper() -> None:
    """Moment-match inflow integration equals existing moment-match helper output."""
    A, k, rho = _make_stacks(rho_value=1.3)
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    p_curve = np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)
    H = 100.0
    D = 80.0
    rews_n = 8
    loss_factor = 0.97

    inflow = _build_momentmatch_inflow(
        A_stack=A,
        k_stack=k,
        hub_height=H,
        rotor_diameter=D,
        rews_n=rews_n,
        rho_stack=rho,
        air_density=True,
        vertical_policy=None,
    )
    got = integrate_cf_from_inflow(
        inflow=inflow,
        u_grid=u_grid,
        p_curve=p_curve,
        loss_factor=loss_factor,
    )
    expected_cf, expected_rews = _momentmatch_cf_and_rews_for_turbine(
        A_stack=A,
        k_stack=k,
        rho_stack=rho,
        u_grid=u_grid,
        p_curve=p_curve,
        hub_height=H,
        rotor_diameter=D,
        rews_n=rews_n,
        loss_factor=loss_factor,
        vertical_policy=None,
    )
    np.testing.assert_allclose(got.values, expected_cf.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inflow.rews_mps.values, expected_rews.values, rtol=1e-12, atol=1e-12)


def test_inflow_based_cf_preserves_algo_version_attr() -> None:
    """CF output keeps ``cleo:algo_version`` for downstream reuse checks."""
    A, k, rho = _make_stacks()
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    power_curves = np.array([np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)], dtype=np.float64)
    out = capacity_factors_v1(
        A_stack=A,
        k_stack=k,
        u_grid=u_grid,
        turbine_ids=("T1",),
        hub_heights_m=np.array([100.0], dtype=np.float64),
        power_curves=power_curves,
        mode="direct_cf_quadrature",
        rotor_diameters_m=np.array([80.0], dtype=np.float64),
        rho_stack=rho,
        air_density=True,
        rews_n=8,
    )
    assert out.attrs["cleo:algo_version"] == "3"


def test_cf_reuse_check_works_with_inflow_based_cf() -> None:
    """Existing CF reuse matcher accepts metadata from computed CF outputs."""
    A, k, rho = _make_stacks()
    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    power_curves = np.array([np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0)], dtype=np.float64)
    out = capacity_factors_v1(
        A_stack=A,
        k_stack=k,
        u_grid=u_grid,
        turbine_ids=("T1",),
        hub_heights_m=np.array([100.0], dtype=np.float64),
        power_curves=power_curves,
        mode="direct_cf_quadrature",
        rotor_diameters_m=np.array([80.0], dtype=np.float64),
        rho_stack=rho,
        air_density=False,
        loss_factor=1.0,
        rews_n=12,
    )
    assert _cf_spec_matches(out, resolve_cf_spec(None), ("T1",))
