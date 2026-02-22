"""Height-continuous vertical policy primitives for wind/REWS computations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr
from scipy.special import gammaln

from cleo.unification.vertical_policy import (
    DEFAULT_VERTICAL_POLICY,
    resolve_vertical_policy,
)


def enforce_query_height_bounds(
    query_heights_m: Sequence[float] | np.ndarray,
    *,
    min_supported_m: float = 10.0,
    max_supported_m: float,
    min_policy: Literal["error", "clamp_to_10"] = "error",
    max_policy: Literal["error"] = "error",
) -> np.ndarray:
    """Validate (or minimally clamp) query heights per policy."""
    q = np.asarray(query_heights_m, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError("query_heights_m must be 1D")

    if np.any(q <= 0):
        raise ValueError("query heights must be strictly positive")

    q_out = q.copy()
    if np.min(q_out) < min_supported_m:
        if min_policy == "error":
            raise ValueError(
                f"query height below supported minimum {min_supported_m} m: min={float(np.min(q_out))}"
            )
        if min_policy == "clamp_to_10":
            q_out = np.maximum(q_out, min_supported_m)
        else:
            raise ValueError(f"Unsupported min_policy: {min_policy!r}")

    if np.max(q_out) > max_supported_m:
        if max_policy == "error":
            raise ValueError(
                f"query height above supported maximum {max_supported_m} m: max={float(np.max(q_out))}"
            )
        raise ValueError(f"Unsupported max_policy: {max_policy!r}")

    return q_out


def weibull_mean_from_a_k(A: xr.DataArray, k: xr.DataArray) -> xr.DataArray:
    """Compute Weibull mean wind speed ``mu = A * Gamma(1 + 1/k)`` using log-gamma numerics."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_mu = np.log(A) + xr.apply_ufunc(gammaln, 1.0 + (1.0 / k), dask="parallelized")
        mu = np.exp(log_mu)
    return mu.rename("mean_wind_speed")


def weibull_cv_from_k(k: xr.DataArray) -> xr.DataArray:
    """Compute Weibull coefficient of variation from ``k`` using log-gamma numerics."""
    g1 = xr.apply_ufunc(gammaln, 1.0 + (1.0 / k), dask="parallelized")
    g2 = xr.apply_ufunc(gammaln, 1.0 + (2.0 / k), dask="parallelized")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # var / mean^2 = exp(g2 - 2*g1) - 1
        cv2 = np.exp(g2 - 2.0 * g1) - 1.0
        cv2 = xr.where(cv2 < 0.0, 0.0, cv2)
        cv = np.sqrt(cv2)
    return cv.rename("weibull_cv")


def build_cv_k_lut(
    *,
    k_min: float = 0.5,
    k_max: float = 10.0,
    size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Build monotone LUT for deterministic ``CV -> k`` inversion."""
    if size < 8:
        raise ValueError("LUT size must be >= 8")
    if k_min <= 0 or k_max <= k_min:
        raise ValueError("Invalid k-range for LUT")

    k_grid = np.linspace(float(k_min), float(k_max), int(size), dtype=np.float64)
    cv_grid = np.sqrt(np.exp(gammaln(1.0 + (2.0 / k_grid)) - 2.0 * gammaln(1.0 + (1.0 / k_grid))) - 1.0)

    # CV decreases with k on practical ranges. Store ascending CV for searchsorted/interp.
    order = np.argsort(cv_grid)
    cv_sorted = cv_grid[order]
    k_sorted = k_grid[order]

    if not np.all(np.diff(cv_sorted) >= 0):
        raise RuntimeError("CV LUT is not monotone; cannot invert deterministically")
    return cv_sorted, k_sorted


def invert_cv_to_k(
    cv: xr.DataArray,
    *,
    cv_lut: np.ndarray,
    k_lut: np.ndarray,
) -> xr.DataArray:
    """Invert CV values to Weibull ``k`` by monotone LUT interpolation with clamping."""
    cv_lut_arr = np.asarray(cv_lut, dtype=np.float64)
    k_lut_arr = np.asarray(k_lut, dtype=np.float64)
    if cv_lut_arr.ndim != 1 or k_lut_arr.ndim != 1 or cv_lut_arr.size != k_lut_arr.size:
        raise ValueError("cv_lut and k_lut must be 1D arrays of equal length")

    if not np.all(np.diff(cv_lut_arr) >= 0):
        raise ValueError("cv_lut must be sorted ascending")

    cv_min = float(cv_lut_arr[0])
    cv_max = float(cv_lut_arr[-1])

    def _interp(arr: np.ndarray) -> np.ndarray:
        arr_clipped = np.clip(arr, cv_min, cv_max)
        return np.interp(arr_clipped, cv_lut_arr, k_lut_arr)

    return xr.apply_ufunc(
        _interp,
        cv,
        dask="parallelized",
        output_dtypes=[np.float64],
    ).rename("weibull_k")


def interp_log_linear_ln_z(
    q_stack: xr.DataArray,
    *,
    query_heights_m: np.ndarray,
) -> xr.DataArray:
    """Interpolate positive quantity in ``ln(q)`` vs ``ln(z)`` space."""
    if "height" not in q_stack.dims:
        raise ValueError("q_stack must have a 'height' dimension")

    z = np.asarray(q_stack["height"].values, dtype=np.float64)
    if np.any(z <= 0):
        raise ValueError("height coordinates must be strictly positive")

    qh = np.asarray(query_heights_m, dtype=np.float64)
    if np.any(qh <= 0):
        raise ValueError("query heights must be strictly positive")

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if float(np.min(qh)) < z_min or float(np.max(qh)) > z_max:
        raise ValueError(
            f"query heights [{float(np.min(qh))}, {float(np.max(qh))}] exceed stack range [{z_min}, {z_max}]"
        )

    lnz = np.log(z)
    lnq = np.log(q_stack).assign_coords(ln_height=("height", lnz)).swap_dims({"height": "ln_height"}).sortby("ln_height")

    q_coord = xr.DataArray(qh, dims=("query_height",), coords={"query_height": qh})
    ln_target = xr.DataArray(np.log(qh), dims=("query_height",), coords={"query_height": qh})
    lnq_out = lnq.interp(ln_height=ln_target, method="linear")
    # interp introduces ln_height coordinate; normalize to query_height only.
    if "ln_height" in lnq_out.coords:
        lnq_out = lnq_out.drop_vars("ln_height")
    return np.exp(lnq_out).rename(q_stack.name)


def weighted_top_shear_alpha(
    mu_stack: xr.DataArray,
    *,
    heights_m: Sequence[float] = (50.0, 100.0, 150.0, 200.0),
    weights: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
    alpha_min: float = -0.05,
    alpha_max: float = 0.35,
) -> xr.DataArray:
    """Estimate top-layer shear ``alpha`` by weighted regression in ``ln(mu)`` vs ``ln(z)``."""
    if "height" not in mu_stack.dims:
        raise ValueError("mu_stack must have a 'height' dimension")
    if len(heights_m) != len(weights):
        raise ValueError("heights_m and weights lengths must match")

    available_heights = set(float(h) for h in np.asarray(mu_stack["height"].values, dtype=np.float64))
    selected = [float(h) for h in heights_m if float(h) in available_heights]
    if len(selected) < 2:
        raise ValueError("At least two requested heights must exist in mu_stack")

    selected_weights = [float(w) for h, w in zip(heights_m, weights) if float(h) in available_heights]
    mu_top = mu_stack.sel(height=selected)

    x = xr.DataArray(np.log(np.asarray(selected, dtype=np.float64)), dims=("height",), coords={"height": selected})
    w = xr.DataArray(np.asarray(selected_weights, dtype=np.float64), dims=("height",), coords={"height": selected})

    valid = np.isfinite(mu_top) & (mu_top > 0.0)
    y = xr.where(valid, np.log(mu_top), 0.0)

    w_valid = xr.where(valid, w, 0.0)
    w_sum = w_valid.sum(dim="height")
    valid_count = valid.sum(dim="height")

    # Avoid runtime divide-by-zero warnings in dask graphs by dividing by a safe
    # denominator, then masking invalid tiles back to NaN.
    safe_w_sum = xr.where(w_sum > 0, w_sum, 1.0)
    x_bar = ((w_valid * x).sum(dim="height") / safe_w_sum).where(w_sum > 0)
    y_bar = ((w_valid * y).sum(dim="height") / safe_w_sum).where(w_sum > 0)

    num = (w_valid * (x - x_bar) * (y - y_bar)).sum(dim="height")
    den = (w_valid * (x - x_bar) ** 2).sum(dim="height")

    safe_den = xr.where(den > 0, den, 1.0)
    alpha_raw = (num / safe_den).where((valid_count >= 2) & (den > 0))
    alpha = alpha_raw.clip(min=float(alpha_min), max=float(alpha_max))
    return alpha.rename("alpha_top")


def apply_alpha_fallback(
    alpha: xr.DataArray,
    *,
    mode: Literal["none", "constant", "tile_median"] = "none",
    constant: float = 0.14,
    tile_shape: tuple[int, int] = (256, 256),
    min_count: int = 2000,
) -> xr.DataArray:
    """Apply configured alpha fallback for missing shear estimates."""
    if mode == "none":
        return alpha
    if mode == "constant":
        return alpha.fillna(float(constant))
    if mode != "tile_median":
        raise ValueError(f"Unsupported alpha fallback mode: {mode!r}")

    if "y" not in alpha.dims or "x" not in alpha.dims:
        raise ValueError("tile_median fallback requires y/x dimensions")

    tile_y, tile_x = int(tile_shape[0]), int(tile_shape[1])
    if tile_y <= 0 or tile_x <= 0:
        raise ValueError("tile_shape entries must be > 0")

    def _fill(arr_2d: np.ndarray) -> np.ndarray:
        out = arr_2d.copy()
        ny, nx = arr_2d.shape
        for y0 in range(0, ny, tile_y):
            y1 = min(y0 + tile_y, ny)
            for x0 in range(0, nx, tile_x):
                x1 = min(x0 + tile_x, nx)
                tile = arr_2d[y0:y1, x0:x1]
                valid = np.isfinite(tile)
                if int(valid.sum()) < int(min_count):
                    continue
                med = float(np.nanmedian(tile))
                if np.isnan(med):
                    continue
                mask = ~valid
                if np.any(mask):
                    patched = out[y0:y1, x0:x1]
                    patched[mask] = med
                    out[y0:y1, x0:x1] = patched
        return out

    return xr.apply_ufunc(
        _fill,
        alpha,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float64],
    )


def evaluate_density_at_heights(
    rho_stack: xr.DataArray,
    *,
    query_heights_m: np.ndarray,
    policy: dict | None = None,
) -> xr.DataArray:
    """Evaluate air density at query heights using policy defaults."""
    cfg = resolve_vertical_policy(policy)
    qh = np.asarray(query_heights_m, dtype=np.float64)
    q_coord = xr.DataArray(qh, dims=("query_height",), coords={"query_height": qh})

    if "height" not in rho_stack.dims:
        return rho_stack.expand_dims(query_height=q_coord)

    z_top = float(np.max(np.asarray(rho_stack["height"].values, dtype=np.float64)))
    rho_interp = interp_log_linear_ln_z(rho_stack, query_heights_m=np.minimum(qh, z_top))
    rho_interp = rho_interp.assign_coords(query_height=qh)
    if float(np.max(qh)) <= z_top:
        return rho_interp

    rho_top = interp_log_linear_ln_z(rho_stack, query_heights_m=np.array([z_top], dtype=np.float64)).squeeze("query_height")
    if cfg["rho_extrap_above_200"] == "constant_at_200":
        rho_above = rho_top.expand_dims(query_height=q_coord)
    elif cfg["rho_extrap_above_200"] == "exp_scale_height":
        H = float(cfg["rho_scale_height_m"])
        rho_above = rho_top.expand_dims(query_height=q_coord) * np.exp(-(q_coord - z_top) / H)
    else:
        raise ValueError(f"Unsupported rho_extrap_above_200 policy: {cfg['rho_extrap_above_200']!r}")

    return xr.where(q_coord <= z_top, rho_interp, rho_above)


def evaluate_weibull_at_heights(
    weibull_A_stack: xr.DataArray,
    weibull_k_stack: xr.DataArray,
    *,
    query_heights_m: Sequence[float] | np.ndarray,
    rho_stack: xr.DataArray | None = None,
    policy: dict | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Evaluate ``mu(z), k(z), A(z), A'(z)`` at arbitrary query heights."""
    cfg = resolve_vertical_policy(policy)
    z_top = float(np.max(np.asarray(weibull_A_stack["height"].values, dtype=np.float64)))
    qh = enforce_query_height_bounds(
        query_heights_m,
        min_supported_m=10.0,
        max_supported_m=float(cfg["max_query_height_m"]),
        min_policy=cfg["min_query_height_policy"],
        max_policy=cfg["max_query_height_policy"],
    )

    mu_stack = weibull_mean_from_a_k(weibull_A_stack, weibull_k_stack)
    cv_stack = weibull_cv_from_k(weibull_k_stack)
    cv_lut, k_lut = build_cv_k_lut(
        k_min=float(cfg["k_lut_min"]),
        k_max=float(cfg["k_lut_max"]),
        size=int(cfg["k_lut_size"]),
    )

    qh_clamped = np.minimum(qh, z_top)
    mu_q_in = interp_log_linear_ln_z(mu_stack, query_heights_m=qh_clamped).assign_coords(query_height=qh)
    cv_q_in = interp_log_linear_ln_z(cv_stack, query_heights_m=qh_clamped).assign_coords(query_height=qh)
    k_q_in = invert_cv_to_k(cv_q_in, cv_lut=cv_lut, k_lut=k_lut)

    alpha = weighted_top_shear_alpha(
        mu_stack,
        heights_m=tuple(cfg["mu_extrap_heights_m"]),
        weights=tuple(cfg["mu_extrap_weights"]),
        alpha_min=float(cfg["alpha_min"]),
        alpha_max=float(cfg["alpha_max"]),
    )
    alpha = apply_alpha_fallback(
        alpha,
        mode=str(cfg["alpha_fallback"]),
        constant=float(cfg["alpha_fallback_constant"]),
        tile_shape=tuple(int(v) for v in cfg["alpha_fallback_tile_shape"]),
        min_count=int(cfg["alpha_fallback_min_count"]),
    )

    q_coord = xr.DataArray(qh, dims=("query_height",), coords={"query_height": qh})
    if float(np.max(qh)) > z_top:
        mu_top = interp_log_linear_ln_z(mu_stack, query_heights_m=np.array([z_top], dtype=np.float64)).squeeze("query_height")
        mu_above = mu_top.expand_dims(query_height=q_coord) * (q_coord / z_top) ** alpha.expand_dims(query_height=q_coord)
        mu_q = xr.where(q_coord <= z_top, mu_q_in, mu_above).rename("mean_wind_speed")
    else:
        mu_q = mu_q_in.rename("mean_wind_speed")

    k_top = interp_log_linear_ln_z(cv_stack, query_heights_m=np.array([z_top], dtype=np.float64)).squeeze("query_height")
    k_top = invert_cv_to_k(k_top, cv_lut=cv_lut, k_lut=k_lut)
    k_q = xr.where(q_coord <= z_top, k_q_in, k_top.expand_dims(query_height=q_coord)).rename("weibull_k")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        A_q = np.exp(np.log(mu_q) - xr.apply_ufunc(gammaln, 1.0 + (1.0 / k_q), dask="parallelized")).rename("weibull_A")

    if rho_stack is not None:
        rho_q = evaluate_density_at_heights(rho_stack, query_heights_m=qh, policy=cfg)
        c = (rho_q / float(cfg["rho0"])) ** (1.0 / 3.0)
        A_prime = (A_q * c).rename("weibull_A_density_corrected")
    else:
        A_prime = A_q.rename("weibull_A_density_corrected")

    return mu_q, k_q, A_q, A_prime


def default_vertical_policy() -> dict:
    """Return a copy of default vertical policy."""
    return dict(DEFAULT_VERTICAL_POLICY)
