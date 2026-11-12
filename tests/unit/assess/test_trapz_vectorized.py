"""Equivalence of the vectorized wind-speed trapezoid with the former per-pixel implementation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo import assess
from cleo.assess import _trapz_over_wind_speed


def _legacy_trapz_over_wind_speed(y: xr.DataArray, x: xr.DataArray) -> xr.DataArray:
    """Former implementation (per-pixel np.trapezoid via vectorize=True), kept as the test oracle."""
    return xr.apply_ufunc(
        np.trapezoid,
        y,
        x,
        input_core_dims=[["wind_speed"], ["wind_speed"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float64],
    )


def _integrand(wind_speed: np.ndarray, *, dims: tuple[str, ...], seed: int = 0) -> xr.DataArray:
    sizes = {"wind_speed": wind_speed.size, "turbine": 2, "y": 5, "x": 4}
    rng = np.random.default_rng(seed)
    values = rng.random(tuple(sizes[d] for d in dims))
    da = xr.DataArray(values, dims=dims, coords={"wind_speed": wind_speed})
    # NaN in single samples and an entirely NaN pixel.
    da = da.where(~((da["wind_speed"] == wind_speed[3]) & (xr.DataArray(np.arange(5), dims="y") == 1)))
    return da.where(~((xr.DataArray(np.arange(5), dims="y") == 4) & (xr.DataArray(np.arange(4), dims="x") == 0)))


@pytest.mark.parametrize(
    "wind_speed",
    [np.arange(0.0, 10.5, 0.5), np.array([0.0, 0.25, 1.0, 2.5, 3.0, 7.0, 7.5, 12.0, 25.0])],
    ids=["uniform", "non-uniform"],
)
@pytest.mark.parametrize(
    "dims",
    [("wind_speed", "y", "x"), ("y", "wind_speed", "x"), ("turbine", "wind_speed", "y", "x")],
    ids=["leading", "permuted", "extra-dim"],
)
@pytest.mark.parametrize("chunked", [False, True], ids=["numpy", "dask"])
def test_matches_legacy_per_pixel_integration(wind_speed: np.ndarray, dims: tuple[str, ...], chunked: bool) -> None:
    y = _integrand(wind_speed, dims=dims)
    if chunked:
        y = y.chunk({"y": 2, "x": 3})
    x = xr.DataArray(wind_speed, dims=("wind_speed",), coords={"wind_speed": wind_speed})

    expected = _legacy_trapz_over_wind_speed(y, x).compute()
    actual = _trapz_over_wind_speed(y, x).compute()

    assert actual.dims == expected.dims
    assert actual.dtype == expected.dtype == np.float64
    if chunked:
        np.testing.assert_array_equal(actual.values, expected.values)
    else:
        # Summation order can differ in the last bit for non-contiguous numpy inputs; NaN positions must match.
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-14, atol=0.0, equal_nan=True)
    assert np.isnan(actual.values).any()  # the NaN cases are exercised


def test_float32_input_still_returns_float64() -> None:
    wind_speed = np.arange(0.0, 5.0, 0.5, dtype=np.float32)
    y = xr.DataArray(np.ones((wind_speed.size, 2), dtype=np.float32), dims=("wind_speed", "x"))
    x = xr.DataArray(wind_speed, dims=("wind_speed",))
    assert _trapz_over_wind_speed(y, x).dtype == np.float64


def test_capacity_factors_unchanged_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    heights = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
    ny, nx = 6, 5
    rng = np.random.default_rng(1)
    base = rng.uniform(4.0, 9.0, (ny, nx))
    base[0, 0] = np.nan
    coords = {"height": heights, "y": np.arange(ny, dtype=float), "x": np.arange(nx, dtype=float)}

    def stack(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, dims=("height", "y", "x"), coords=coords).chunk({"y": 3, "x": 3})

    a_stack = stack(np.stack([base * (h / 100.0) ** 0.15 for h in heights]))
    k_stack = stack(np.stack([np.full((ny, nx), 1.8 + 0.002 * h) for h in heights]))
    rho_stack = stack(np.stack([np.full((ny, nx), 1.2) * np.exp(-h / 8000.0) for h in heights]))
    u = np.arange(0.0, 40.5, 0.5)
    power_curve = np.clip((u - 3.0) / 9.0, 0.0, 1.0) ** 3
    power_curve[u > 25.0] = 0.0

    def run() -> np.ndarray:
        return assess.capacity_factors(
            A_stack=a_stack,
            k_stack=k_stack,
            u_grid=u,
            turbine_ids=("T",),
            hub_heights_m=np.array([120.0]),
            power_curves=power_curve[None, :],
            method="rotor_node_average",
            rotor_diameters_m=np.array([100.0]),
            rho_stack=rho_stack,
            air_density=True,
        ).values

    actual = run()
    monkeypatch.setattr(assess, "_trapz_over_wind_speed", _legacy_trapz_over_wind_speed)
    expected = run()

    np.testing.assert_array_equal(actual, expected)
    assert np.isnan(actual).any() and np.isfinite(actual).any()
