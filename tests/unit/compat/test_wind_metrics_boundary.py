"""Boundary regression tests for wind_metrics module interface.

These tests verify that cleo.domains does not import underscore-prefixed
internals from cleo.wind_metrics, ensuring a stable boundary interface.

Uses shared utilities from tools.arch_check.
"""

from __future__ import annotations

from tools.arch_check import check_domains_wind_metrics_boundary


def test_domains_no_underscore_imports_from_wind_metrics() -> None:
    """cleo/domains.py must not import underscore internals from cleo.wind_metrics."""
    violations = check_domains_wind_metrics_boundary()
    assert violations == [], (
        "cleo.domains imports underscore internals from cleo.wind_metrics:\n"
        + "\n".join(violations)
        + "\n\nUse the stable interface functions instead:\n"
        + "  list_wind_metrics(), get_wind_metric_spec(), resolve_cf_spec(),\n"
        + "  required_economics_fields(), flat_cf_kwargs(), flat_economics_kwargs()"
    )


def test_wind_metrics_interface_functions_exist() -> None:
    """Wind metrics module must expose stable interface functions."""
    from cleo import wind_metrics

    # Verify all interface functions exist and are callable
    assert callable(wind_metrics.list_wind_metrics)
    assert callable(wind_metrics.get_wind_metric_spec)
    assert callable(wind_metrics.resolve_cf_spec)
    assert callable(wind_metrics.required_economics_fields)
    assert callable(wind_metrics.flat_cf_kwargs)
    assert callable(wind_metrics.flat_economics_kwargs)


def test_list_wind_metrics_returns_tuple() -> None:
    """list_wind_metrics() must return a tuple of metric names."""
    from cleo.wind_metrics import list_wind_metrics

    result = list_wind_metrics()
    assert isinstance(result, tuple)
    assert len(result) > 0
    assert all(isinstance(name, str) for name in result)
    assert "capacity_factors" in result
    assert "lcoe" in result


def test_get_wind_metric_spec_returns_immutable() -> None:
    """get_wind_metric_spec() must return an immutable mapping."""
    import types
    from cleo.wind_metrics import get_wind_metric_spec

    spec = get_wind_metric_spec("capacity_factors")
    assert isinstance(spec, types.MappingProxyType)

    # Verify expected keys exist
    assert "fn" in spec
    assert "requires_turbines" in spec
    assert "required" in spec
    assert "allowed" in spec
    assert "composed" in spec

    # Verify sets are frozensets
    assert isinstance(spec["required"], frozenset)
    assert isinstance(spec["allowed"], frozenset)


def test_required_economics_fields_returns_frozenset() -> None:
    """required_economics_fields() must return a frozenset."""
    from cleo.wind_metrics import required_economics_fields

    result = required_economics_fields()
    assert isinstance(result, frozenset)
    assert "discount_rate" in result
    assert "lifetime_a" in result


def test_flat_cf_kwargs_returns_frozenset() -> None:
    """flat_cf_kwargs() must return a frozenset."""
    from cleo.wind_metrics import flat_cf_kwargs

    result = flat_cf_kwargs()
    assert isinstance(result, frozenset)
    assert "mode" in result
    assert "air_density" in result


def test_flat_economics_kwargs_returns_frozenset() -> None:
    """flat_economics_kwargs() must return a frozenset."""
    from cleo.wind_metrics import flat_economics_kwargs

    result = flat_economics_kwargs()
    assert isinstance(result, frozenset)
    assert "discount_rate" in result
    assert "bos_cost_share" in result


def test_resolve_cf_spec_with_none() -> None:
    """resolve_cf_spec(None) must return defaults."""
    from cleo.wind_metrics import resolve_cf_spec

    result = resolve_cf_spec(None)
    assert isinstance(result, dict)
    assert result["mode"] == "direct_cf_quadrature"
    assert result["air_density"] is False
    assert result["rews_n"] == 12
    assert result["loss_factor"] == 1.0


def test_resolve_cf_spec_with_overrides() -> None:
    """resolve_cf_spec() must apply overrides."""
    from cleo.wind_metrics import resolve_cf_spec

    result = resolve_cf_spec({"mode": "hub", "air_density": True})
    assert result["mode"] == "hub"
    assert result["air_density"] is True
    assert result["rews_n"] == 12  # default preserved
    assert result["loss_factor"] == 1.0  # default preserved
