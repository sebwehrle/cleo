"""Unit tests for DomainResult string representation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from cleo.results import DomainResult


class _DomainStub:
    """Minimal domain stub exposing atlas and staged overlay map."""

    def __init__(self, metric: str, *, staged: bool):
        self._atlas = SimpleNamespace()
        self._computed_overlays = {metric: object()} if staged else {}


def test_domain_result_repr_staged_guides_next_steps() -> None:
    metric = "mean_wind_speed"
    domain = _DomainStub(metric, staged=True)
    da = xr.DataArray(
        np.array([[1.0]], dtype=np.float32),
        dims=("y", "x"),
        name=metric,
    )
    result = DomainResult(domain, metric, da, {})

    text = repr(result)
    assert "DomainResult(metric='mean_wind_speed', state='staged'" in text
    assert 'target=\'atlas.wind.data["mean_wind_speed"]\'' in text
    assert "Lazy data: .data" in text
    assert ".materialize(overwrite=True, allow_mode_change=False)" in text
    assert ".persist(run_id=None, metric_name=None)" in text


def test_domain_result_repr_capacity_factors_includes_mode_when_present() -> None:
    metric = "capacity_factors"
    domain = _DomainStub(metric, staged=False)
    da = xr.DataArray(
        np.array([[[0.3]]], dtype=np.float32),
        dims=("turbine", "y", "x"),
        coords={"turbine": ["Enercon.E40.500"]},
        name=metric,
        attrs={"cleo:cf_mode": "direct_cf_quadrature"},
    )
    result = DomainResult(domain, metric, da, {})

    text = repr(result)
    assert "DomainResult(metric='capacity_factors', state='computed'" in text
    assert "mode='direct_cf_quadrature'" in text
