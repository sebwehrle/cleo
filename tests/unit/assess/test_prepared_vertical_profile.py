"""Tests for prepared vertical-profile seam in wind assessment."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import (
    CF_METHOD_HUB_HEIGHT,
    CF_METHOD_HUB_HEIGHT_REWS,
    CF_METHOD_ROTOR_NODE_AVERAGE,
    RHO_0,
    _build_direct_inflow_from_profile,
    _build_hub_inflow_from_profile,
    _prepare_vertical_profile,
    _rotor_nodes_and_weights,
)


def _make_stacks(
    *,
    a_value: float = 8.0,
    k_value: float = 2.0,
    rho_value: float = RHO_0,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Create constant Weibull and density stacks for seam tests."""
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


def test_prepare_vertical_profile_hub_height_has_single_query_height() -> None:
    """Hub-height profile keeps exactly one query height and raw Weibull fields."""
    A, k, _rho = _make_stacks()
    profile = _prepare_vertical_profile(
        A_stack=A,
        k_stack=k,
        hub_height_m=100.0,
        rotor_diameter_m=None,
        rho_stack=None,
        air_density=False,
        method=CF_METHOD_HUB_HEIGHT,
        rews_n=12,
        interpolation="ak_logz",
        vertical_policy=None,
    )

    assert profile.query_heights_m.tolist() == [100.0]
    assert profile.A.sizes["query_height"] == 1
    assert profile.k.sizes["query_height"] == 1
    assert profile.A_density_corrected is None
    assert profile.density_scale is None


def test_prepare_vertical_profile_hub_height_rews_includes_hub_and_rotor_nodes() -> None:
    """REWS-scaled hub profile includes hub height plus rotor node heights."""
    A, k, _rho = _make_stacks()
    hub_height = 100.0
    rotor_diameter = 80.0
    rews_n = 6
    z_nodes, _weights = _rotor_nodes_and_weights(
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )
    profile = _prepare_vertical_profile(
        A_stack=A,
        k_stack=k,
        hub_height_m=hub_height,
        rotor_diameter_m=rotor_diameter,
        rho_stack=None,
        air_density=False,
        method=CF_METHOD_HUB_HEIGHT_REWS,
        rews_n=rews_n,
        interpolation="ak_logz",
        vertical_policy=None,
    )

    expected = np.concatenate((np.array([hub_height], dtype=np.float64), z_nodes))
    np.testing.assert_allclose(profile.query_heights_m, expected, rtol=0.0, atol=0.0)
    assert profile.A.sizes["query_height"] == rews_n + 1


def test_prepare_vertical_profile_carries_density_corrected_scale() -> None:
    """Vertical profile prepares upstream density-corrected ``A'`` for rotor-aware methods."""
    A, k, rho = _make_stacks(rho_value=1.4)
    profile = _prepare_vertical_profile(
        A_stack=A,
        k_stack=k,
        hub_height_m=100.0,
        rotor_diameter_m=80.0,
        rews_n=5,
        rho_stack=rho,
        air_density=True,
        method=CF_METHOD_ROTOR_NODE_AVERAGE,
        vertical_policy=None,
        interpolation="mu_cv_loglog",
    )

    assert profile.A.sizes["query_height"] == 5
    assert profile.A_density_corrected is not None
    assert profile.rho is not None
    assert profile.density_scale is not None


def test_build_hub_inflow_from_profile_preserves_density_scale_semantics() -> None:
    """Hub inflow built from prepared profile keeps raw ``A`` and integration-time density scaling."""
    A, k, rho = _make_stacks(rho_value=1.5)
    profile = _prepare_vertical_profile(
        A_stack=A,
        k_stack=k,
        hub_height_m=100.0,
        rotor_diameter_m=None,
        rho_stack=rho,
        air_density=True,
        method=CF_METHOD_HUB_HEIGHT,
        rews_n=12,
        interpolation="ak_logz",
        vertical_policy=None,
    )
    inflow = _build_hub_inflow_from_profile(profile=profile)

    np.testing.assert_allclose(inflow.A_eff.values, A.sel(height=100.0).values, rtol=1e-12, atol=1e-12)
    assert inflow.density_scale is not None
    np.testing.assert_allclose(inflow.density_scale.values, (1.5 / RHO_0) ** (1.0 / 3.0), rtol=1e-12, atol=1e-12)


def test_build_direct_inflow_from_profile_uses_prepared_node_count() -> None:
    """Direct inflow builder consumes the prepared rotor-node profile without re-querying heights."""
    A, k, rho = _make_stacks(rho_value=1.3)
    profile = _prepare_vertical_profile(
        A_stack=A,
        k_stack=k,
        hub_height_m=100.0,
        rotor_diameter_m=90.0,
        rews_n=7,
        rho_stack=rho,
        air_density=True,
        method=CF_METHOD_ROTOR_NODE_AVERAGE,
        vertical_policy=None,
        interpolation="mu_cv_loglog",
    )
    inflow = _build_direct_inflow_from_profile(
        profile=profile,
        hub_height=100.0,
        rotor_diameter=90.0,
        rews_n=7,
    )

    assert inflow.A_nodes.sizes["node"] == 7
    assert inflow.k_nodes.sizes["node"] == 7
    np.testing.assert_allclose(np.sum(inflow.weights), 1.0, rtol=1e-12, atol=1e-12)
    assert inflow.rews_mps is not None
