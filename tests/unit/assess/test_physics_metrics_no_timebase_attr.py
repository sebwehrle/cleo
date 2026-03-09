"""Contract: physics metrics carry no timebase metadata.

This test documents and enforces that physics metrics (capacity_factors,
wind_speed, wind_speed) do NOT carry cleo:hours_per_year attrs.

This is important because:
1. Physics outputs are timebase-independent (CF is a fraction, not energy)
2. Changing timebase should not invalidate cached CF results
3. Only economics metrics (LCOE, optimal_energy, etc.) depend on timebase
"""

import numpy as np
import xarray as xr

from cleo.assess import (
    capacity_factors,
    mean_wind_speed_from_weibull,
    rews_mps,
)


def _make_minimal_inputs():
    """Create minimal inputs for physics metric functions."""
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 1.0], dtype=np.float64)
    height = np.array([50.0, 100.0, 150.0], dtype=np.float64)

    A_stack = xr.DataArray(
        np.full((3, 2, 2), 8.0, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": height, "y": y, "x": x},
    )
    k_stack = xr.DataArray(
        np.full((3, 2, 2), 2.0, dtype=np.float64),
        dims=("height", "y", "x"),
        coords={"height": height, "y": y, "x": x},
    )

    u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
    power_curves = np.vstack(
        [
            np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0),
        ]
    ).astype(np.float64)

    return {
        "A_stack": A_stack,
        "k_stack": k_stack,
        "u_grid": u_grid,
        "power_curves": power_curves,
        "turbine_ids": ("T1",),
        "hub_heights_m": np.array([100.0], dtype=np.float64),
        "rotor_diameters_m": np.array([120.0], dtype=np.float64),
    }


def test_capacity_factors_has_no_hours_per_year_attr():
    """capacity_factors must NOT set cleo:hours_per_year attr."""
    inputs = _make_minimal_inputs()

    result = capacity_factors(
        A_stack=inputs["A_stack"],
        k_stack=inputs["k_stack"],
        u_grid=inputs["u_grid"],
        turbine_ids=inputs["turbine_ids"],
        hub_heights_m=inputs["hub_heights_m"],
        power_curves=inputs["power_curves"],
        method="hub_height_weibull",
    )

    assert "cleo:hours_per_year" not in result.attrs, "capacity_factors must NOT carry timebase metadata"


def test_mean_wind_speed_has_no_hours_per_year_attr():
    """mean_wind_speed_from_weibull must NOT set cleo:hours_per_year attr."""
    A = xr.DataArray(
        np.full((2, 2), 8.0, dtype=np.float64),
        dims=("y", "x"),
    )
    k = xr.DataArray(
        np.full((2, 2), 2.0, dtype=np.float64),
        dims=("y", "x"),
    )

    result = mean_wind_speed_from_weibull(A=A, k=k)

    assert "cleo:hours_per_year" not in result.attrs, "wind_speed must NOT carry timebase metadata"


def test_rews_mps_has_no_hours_per_year_attr():
    """rews_mps must NOT set cleo:hours_per_year attr."""
    inputs = _make_minimal_inputs()

    # Use smaller rotor to stay within height stack bounds
    result = rews_mps(
        A_stack=inputs["A_stack"],
        k_stack=inputs["k_stack"],
        turbine_ids=inputs["turbine_ids"],
        hub_heights_m=np.array([100.0], dtype=np.float64),
        rotor_diameters_m=np.array([80.0], dtype=np.float64),  # Smaller to fit in [50, 150]
        rews_n=5,
    )

    assert "cleo:hours_per_year" not in result.attrs, "wind_speed must NOT carry timebase metadata"


def test_physics_metrics_have_no_timebase_related_attrs():
    """Physics metrics should have no timebase-related attrs at all."""
    inputs = _make_minimal_inputs()

    cf = capacity_factors(
        A_stack=inputs["A_stack"],
        k_stack=inputs["k_stack"],
        u_grid=inputs["u_grid"],
        turbine_ids=inputs["turbine_ids"],
        hub_heights_m=inputs["hub_heights_m"],
        power_curves=inputs["power_curves"],
        method="hub_height_weibull",
    )

    # Check no timebase-related attrs
    timebase_attrs = [key for key in cf.attrs.keys() if "hours" in key.lower() or "timebase" in key.lower()]
    assert not timebase_attrs, f"capacity_factors should have no timebase-related attrs, found: {timebase_attrs}"
