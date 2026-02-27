"""economics: grid_connect_cost tests."""

from __future__ import annotations

import numpy as np

from cleo.economics import grid_connect_cost


def test_grid_connect_cost_scalar() -> None:
    """Default rate of 50 EUR/kW is applied."""
    assert grid_connect_cost(0.0) == 0.0
    assert grid_connect_cost(1000.0) == 50_000.0


def test_grid_connect_cost_vectorized() -> None:
    """Default rate works with arrays."""
    power_kw = np.array([500.0, 1500.0, 2000.0], dtype=np.float64)
    expected = np.array([25_000.0, 75_000.0, 100_000.0], dtype=np.float64)
    got = grid_connect_cost(power_kw)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


def test_grid_connect_cost_custom_rate() -> None:
    """Custom rate can be specified."""
    # 100 EUR/kW rate
    assert grid_connect_cost(1000.0, rate_eur_per_kw=100.0) == 100_000.0
    # 25 EUR/kW rate
    assert grid_connect_cost(1000.0, rate_eur_per_kw=25.0) == 25_000.0


def test_grid_connect_cost_zero_rate() -> None:
    """Zero rate eliminates grid connection cost (paper qLCOE mode)."""
    assert grid_connect_cost(1000.0, rate_eur_per_kw=0.0) == 0.0
    assert grid_connect_cost(5000.0, rate_eur_per_kw=0.0) == 0.0


def test_grid_connect_cost_zero_rate_vectorized() -> None:
    """Zero rate works with arrays."""
    power_kw = np.array([500.0, 1500.0, 2000.0], dtype=np.float64)
    expected = np.zeros(3, dtype=np.float64)
    got = grid_connect_cost(power_kw, rate_eur_per_kw=0.0)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
