"""Unit tests for vertical policy primitives (phase 3)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.vertical import (
    apply_alpha_fallback,
    build_cv_k_lut,
    enforce_query_height_bounds,
    evaluate_weibull_at_heights,
    interp_log_linear_ln_z,
    invert_cv_to_k,
    weibull_cv_from_k,
    weibull_mean_from_a_k,
    weighted_top_shear_alpha,
)


def _stack_from_heights(values: np.ndarray, heights: list[float]) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": [0], "x": [0]},
    )


def test_weibull_mean_matches_definition() -> None:
    A = xr.DataArray(np.array([[8.0]], dtype=np.float64), dims=("y", "x"))
    k = xr.DataArray(np.array([[2.0]], dtype=np.float64), dims=("y", "x"))
    out = weibull_mean_from_a_k(A, k)
    expected = 8.0 * np.sqrt(np.pi) / 2.0
    assert np.isclose(float(out.values.item()), expected, rtol=1e-10, atol=1e-12)


def test_cv_monotone_and_lut_roundtrip() -> None:
    k_grid = xr.DataArray(np.linspace(0.5, 10.0, 256), dims=("k",))
    cv = weibull_cv_from_k(k_grid).values
    assert np.all(np.diff(cv) < 0)

    cv_lut, k_lut = build_cv_k_lut(k_min=0.5, k_max=10.0, size=4096)
    k_test = xr.DataArray(np.array([0.7, 1.2, 2.0, 4.5, 8.0], dtype=np.float64), dims=("n",))
    cv_test = weibull_cv_from_k(k_test)
    k_back = invert_cv_to_k(cv_test, cv_lut=cv_lut, k_lut=k_lut)
    assert np.allclose(k_back.values, k_test.values, rtol=1e-3, atol=1e-3)


def test_log_interp_exact_at_base_heights() -> None:
    heights = [10.0, 50.0, 100.0, 150.0, 200.0]
    data = np.array([[[2.0]], [[4.0]], [[6.0]], [[8.0]], [[10.0]]], dtype=np.float64)
    da = _stack_from_heights(data, heights)
    out = interp_log_linear_ln_z(da, query_heights_m=np.array(heights, dtype=np.float64))
    assert np.allclose(out.values[:, 0, 0], np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64))


def test_weighted_alpha_regression_and_clamp() -> None:
    heights = np.array([50.0, 100.0, 150.0, 200.0], dtype=np.float64)
    alpha_true = 0.2
    c = 3.0
    mu = c * (heights ** alpha_true)
    mu_stack = _stack_from_heights(mu[:, None, None], heights.tolist())
    alpha = weighted_top_shear_alpha(mu_stack, alpha_min=-0.05, alpha_max=0.35)
    assert np.isclose(float(alpha.values.item()), alpha_true, rtol=1e-5, atol=1e-5)

    # Strong shear should be clipped.
    mu2 = c * (heights ** 0.8)
    mu2_stack = _stack_from_heights(mu2[:, None, None], heights.tolist())
    alpha2 = weighted_top_shear_alpha(mu2_stack, alpha_min=-0.05, alpha_max=0.35)
    assert np.isclose(float(alpha2.values.item()), 0.35, atol=1e-12)


def test_alpha_fallback_modes() -> None:
    alpha = xr.DataArray(
        np.array([[np.nan, 0.2], [0.3, np.nan]], dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )

    out_const = apply_alpha_fallback(alpha, mode="constant", constant=0.1)
    assert np.isclose(float(out_const.sel(y=0, x=0).values.item()), 0.1)
    assert np.isclose(float(out_const.sel(y=1, x=1).values.item()), 0.1)

    out_tile = apply_alpha_fallback(alpha, mode="tile_median", tile_shape=(2, 2), min_count=2)
    # median([0.2, 0.3]) = 0.25 fills NaNs
    assert np.isclose(float(out_tile.sel(y=0, x=0).values.item()), 0.25)
    assert np.isclose(float(out_tile.sel(y=1, x=1).values.item()), 0.25)


def test_enforce_query_height_bounds() -> None:
    q = enforce_query_height_bounds([10.0, 50.0, 250.0], max_supported_m=300.0)
    assert np.allclose(q, np.array([10.0, 50.0, 250.0], dtype=np.float64))


def test_evaluate_weibull_at_heights_runs_above_200() -> None:
    heights = [10.0, 50.0, 100.0, 150.0, 200.0]
    A = _stack_from_heights(
        np.array([[[6.0]], [[7.0]], [[8.0]], [[8.5]], [[9.0]]], dtype=np.float64),
        heights,
    ).rename("weibull_A")
    k = _stack_from_heights(
        np.array([[[2.0]], [[2.1]], [[2.2]], [[2.3]], [[2.4]]], dtype=np.float64),
        heights,
    ).rename("weibull_k")
    rho = _stack_from_heights(
        np.array([[[1.20]], [[1.19]], [[1.18]], [[1.17]], [[1.16]]], dtype=np.float64),
        heights,
    ).rename("rho")

    mu_q, k_q, A_q, A_p = evaluate_weibull_at_heights(
        A,
        k,
        query_heights_m=np.array([100.0, 220.0, 260.0], dtype=np.float64),
        rho_stack=rho,
    )
    assert mu_q.sizes["query_height"] == 3
    assert np.all(np.isfinite(mu_q.values))
    assert np.all(np.isfinite(k_q.values))
    assert np.all(np.isfinite(A_q.values))
    assert np.all(np.isfinite(A_p.values))
