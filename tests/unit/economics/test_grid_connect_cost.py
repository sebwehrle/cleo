"""economics: grid_connect_cost tests."""

from __future__ import annotations

import numpy as np

from cleo.economics import grid_connect_cost


def test_grid_connect_cost_scalar() -> None:
    assert grid_connect_cost(0.0) == 0.0
    assert grid_connect_cost(1000.0) == 50_000.0


def test_grid_connect_cost_vectorized() -> None:
    power_kw = np.array([500.0, 1500.0, 2000.0], dtype=np.float64)
    expected = np.array([25_000.0, 75_000.0, 100_000.0], dtype=np.float64)
    got = grid_connect_cost(power_kw)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
