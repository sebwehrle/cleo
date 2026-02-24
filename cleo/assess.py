# Pure numeric assessment functions for wind resource analysis.
# Contract: No I/O, no network, no store access, no mutation of self.data.
# All spatial arrays stay lazy (dask-compatible).
import numpy as np
import xarray as xr
import logging

from scipy.special import gamma, gammaln

from cleo.vertical import evaluate_weibull_at_heights

logger = logging.getLogger(__name__)


# %% Constants
RHO_0 = 1.225  # Reference air density at sea level (kg/m³)


# %% Pure compute primitives


def compute_air_density_correction_core(
    *,
    elevation: xr.DataArray,
    template: xr.DataArray,
) -> xr.DataArray:
    """
    Pure compute function: air density correction factor based on elevation.

    This is a stateless, I/O-free computation primitive. The caller is responsible
    for providing elevation and template DataArrays that are already aligned to the
    canonical grid.

    Formula: rho_correction = 1.247015 * exp(-0.000104 * elevation) / 1.225

    Contract:
    - elevation and template MUST have identical y/x coords (caller must verify alignment).
    - Output is NaN where elevation is NaN (propagates nodata).
    - Remains lazy: never forces eager evaluation (compute/load/values).
    - Returns a DataArray named "air_density_correction" on the same coords as elevation.

    :param elevation: Elevation DataArray (meters a.s.l.) on canonical grid.
    :param template: Template DataArray for output coords/dims alignment.
    :return: DataArray "air_density_correction" factor, lazy if input is lazy.
    """
    # Core formula: barometric-like correction
    rho_correction_factor = 1.247015 * np.exp(-0.000104 * elevation) / 1.225

    # Ensure output has correct name and preserves coords from elevation
    result = rho_correction_factor.rename("air_density_correction")

    # Squeeze any singleton dimensions (e.g., band) but keep y, x
    if "band" in result.dims:
        result = result.squeeze("band", drop=True)

    return result


def mean_wind_speed_from_weibull(
    *,
    A: xr.DataArray,
    k: xr.DataArray,
) -> xr.DataArray:
    """
    Compute mean wind speed from Weibull A and k parameters.

    Pure compute primitive: no I/O, dask-friendly, returns same shape as inputs.

    Mathematical definition:
        mean_wind_speed = A * gamma(1 + 1/k)

    :param A: Weibull A (scale) parameter, DataArray with dims (y, x) or (height, y, x)
    :param k: Weibull k (shape) parameter, DataArray with same dims as A
    :return: DataArray with mean wind speed (m/s), same dims as inputs
    """
    mean_ws = A * gamma(1 / k + 1)
    return mean_ws.rename("mean_wind_speed").assign_attrs(units="m/s")



def _interp_power_curve(u_eq: xr.DataArray, u: np.ndarray, p: np.ndarray) -> xr.DataArray:
    """
    Interpolate power curve at equivalent wind speeds (dask-friendly).

    Uses xr.apply_ufunc to wrap np.interp with dask="parallelized".

    :param u_eq: Equivalent wind speeds, any shape (dask-backed OK)
    :param u: 1D wind speed grid (numpy)
    :param p: 1D power curve values (numpy)
    :return: Power curve values at u_eq, same shape as u_eq
    """
    return xr.apply_ufunc(
        lambda x: np.interp(x, u, p, left=0.0, right=0.0),
        u_eq,
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _trapz_over_wind_speed(y: xr.DataArray, x: xr.DataArray) -> xr.DataArray:
    """
    Trapezoidal integration over the wind_speed dimension (dask-friendly).

    Uses np.trapezoid via xr.apply_ufunc with dask="parallelized".

    :param y: Integrand with dims ("wind_speed", ...spatial...), dask-backed OK
    :param x: 1D wind_speed coordinate DataArray (same for all pixels)
    :return: Integrated values with wind_speed dim removed
    """
    return xr.apply_ufunc(
        np.trapezoid,
        y, x,
        input_core_dims=[["wind_speed"], ["wind_speed"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float64],
    )


def _align_pdf_to_wind_speed_exact(
    pdf: xr.DataArray,
    u: np.ndarray,
) -> xr.DataArray:
    """Align PDF to the integration wind-speed grid with exact-match semantics.

    Contract:
    - Reindex to requested wind-speed coordinate values.
    - Fail if any coordinate is missing (NaN introduced by reindex), i.e. no
      nearest-neighbor tolerance and no silent drift.
    """
    if "wind_speed" not in pdf.dims:
        return pdf

    src = np.asarray(pdf["wind_speed"].data, dtype=np.float64)
    tgt = np.asarray(u, dtype=np.float64)

    # Fast path: same logical grid (allow float32/float64 representation noise).
    if src.shape == tgt.shape and np.allclose(src, tgt, rtol=0.0, atol=1e-6):
        return pdf.assign_coords(wind_speed=tgt).transpose("wind_speed", ...)

    # Strict contract: no nearest interpolation. Target labels must exist exactly.
    missing_labels = ~np.isin(tgt, src)
    if bool(np.any(missing_labels)):
        raise ValueError(
            "PDF wind_speed coordinates do not exactly match integration grid. "
            "No nearest alignment is allowed."
        )

    return pdf.reindex(wind_speed=tgt).transpose("wind_speed", ...)


def _integrate_cf_with_density_correction(
    pdf: xr.DataArray,
    u_grid: np.ndarray,
    p_curve: np.ndarray,
    c: xr.DataArray,
    loss_factor: float = 1.0,
) -> xr.DataArray:
    """
    Integrate capacity factor with air density correction.

    CF = loss_factor * ∫ PC(u * c) * pdf(u) du

    The correction factor c = (rho/rho0)^(1/3) scales wind speed to equivalent
    wind speed at reference density, applying the correction INSIDE the power
    curve evaluation, NOT as a post-hoc CF multiplier.

    This implementation is fully vectorized and dask-friendly (no Python loops,
    no eager array evaluation). Uses np.trapezoid via xr.apply_ufunc.

    :param pdf: Weibull PDF at hub height, dims (wind_speed, y, x)
    :param u_grid: 1D wind speed grid
    :param p_curve: 1D power curve values (same length as u_grid)
    :param c: Density correction factor (rho/rho0)^(1/3), dims (y, x)
    :param loss_factor: Additional correction factor
    :return: Capacity factor DataArray with dims (y, x)
    """
    u = np.asarray(u_grid, dtype=np.float64)
    p = np.asarray(p_curve, dtype=np.float64)

    # Align PDF to integration grid with strict exact-match semantics.
    pdf_aligned = _align_pdf_to_wind_speed_exact(pdf, u)

    # Build wind_speed as 1D DataArray for vectorized operations
    u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})

    # Compute equivalent wind speed: u_eq = u * c
    # Broadcasting: u_da (wind_speed,) * c (y, x) => (wind_speed, y, x)
    u_eq = u_da * c

    # Evaluate power curve at all equivalent wind speeds (vectorized, dask-friendly)
    p_eq = _interp_power_curve(u_eq, u, p)

    # CF-integrator endpoint stability fix:
    # For k<1, Weibull PDF at u=0 can be +inf while P(0)=0; forcing pdf(0)=0
    # prevents 0*inf -> NaN in the integrand.
    if "wind_speed" in pdf_aligned.dims:
        pdf_aligned = xr.where(pdf_aligned["wind_speed"] == 0.0, 0.0, pdf_aligned)

    # Build integrand: pdf(u) * PC(u*c)
    integrand = pdf_aligned * p_eq

    # Vectorized trapezoidal integration over wind_speed (dask-friendly)
    cf = _trapz_over_wind_speed(integrand, u_da)

    # Apply loss factor and set name
    result = cf * loss_factor
    result.name = "capacity_factor"
    return result


def _integrate_cf_no_density(
    *,
    pdf: xr.DataArray,
    u_grid: np.ndarray,
    p_curve: np.ndarray,
    loss_factor: float,
) -> xr.DataArray:
    """
    Integrate capacity factor WITHOUT air density correction (pure).

    CF = loss_factor * ∫ PC(u) * pdf(u) du

    Fully vectorized and dask-friendly (no Python loops, no eager array evaluation).
    Uses np.trapezoid via xr.apply_ufunc.

    :param pdf: Weibull PDF at hub height, dims (wind_speed, y, x)
    :param u_grid: 1D wind speed grid
    :param p_curve: 1D power curve values (same length as u_grid)
    :param loss_factor: Additional correction factor
    :return: Capacity factor DataArray with dims (y, x)
    """
    u = np.asarray(u_grid, dtype=np.float64)
    p = np.asarray(p_curve, dtype=np.float64)

    # Create DataArrays for integration
    u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})
    pc_da = xr.DataArray(p, dims=("wind_speed",), coords={"wind_speed": u})

    # Align PDF to integration grid with strict exact-match semantics.
    pdf_aligned = _align_pdf_to_wind_speed_exact(pdf, u)

    # CF-integrator endpoint stability fix (see density path for rationale).
    if "wind_speed" in pdf_aligned.dims:
        pdf_aligned = xr.where(pdf_aligned["wind_speed"] == 0.0, 0.0, pdf_aligned)

    # Build integrand: pdf(u) * PC(u)
    integrand = pdf_aligned * pc_da

    # Vectorized trapezoidal integration over wind_speed (dask-friendly)
    cf = _trapz_over_wind_speed(integrand, u_da)

    return cf * loss_factor


def air_density_at_height(rho_stack: xr.DataArray, height_m: float) -> xr.DataArray:
    """
    Interpolate air density to a specific height (pure).

    Uses linear interpolation if height dimension exists.

    :param rho_stack: Air density DataArray, optionally with 'height' dim
    :param height_m: Target height in meters
    :return: Air density at target height (y, x)
    """
    if "height" in rho_stack.dims:
        return rho_stack.interp(height=height_m, method="linear")
    return rho_stack


def capacity_factors_v1(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    u_grid: np.ndarray,
    turbine_ids: tuple[str, ...],
    hub_heights_m: np.ndarray,
    power_curves: np.ndarray,
    mode: str = "direct_cf_quadrature",
    rotor_diameters_m: np.ndarray | None = None,
    rho_stack: xr.DataArray | None = None,
    air_density: bool = False,
    loss_factor: float = 1.0,
    rews_n: int = 12,
    vertical_policy: dict | None = None,
) -> xr.DataArray:
    """
    Compute capacity factors for multiple turbines (pure numerics, turbine-loop).

    This is the v1 capacity factor algorithm with support for:
    - direct rotor-node quadrature mode (default)
    - hub-height mode
    - legacy REWS-factor mode

    Contract: No I/O, no eager evaluation on spatial arrays (stays lazy/dask).
    Turbine-loop implementation (benchmarked best).

    :param A_stack: Weibull A parameter, dims (height, y, x)
    :param k_stack: Weibull k parameter, dims (height, y, x)
    :param u_grid: 1D wind speed grid for PDF/power curve
    :param turbine_ids: Tuple of turbine ID strings
    :param hub_heights_m: Hub heights for each turbine (n_turb,)
    :param power_curves: Power curves for each turbine (n_turb, n_ws)
    :param mode: "direct_cf_quadrature" (default), "hub", or legacy "rews"
    :param rotor_diameters_m: Rotor diameters (required for direct/rews modes)
    :param rho_stack: Air density DataArray (required if air_density=True)
    :param air_density: If True, apply air density correction
    :param loss_factor: Loss correction factor (default 1.0)
    :param rews_n: Number of quadrature points for REWS integration
    :return: DataArray with capacity factors, dims (turbine, y, x)
    """
    if mode not in ("hub", "rews", "direct_cf_quadrature", "momentmatch_weibull"):
        raise ValueError(
            "mode must be 'hub', 'rews', 'direct_cf_quadrature', or "
            f"'momentmatch_weibull', got {mode!r}"
        )

    if mode in ("rews", "direct_cf_quadrature", "momentmatch_weibull") and rotor_diameters_m is None:
        raise ValueError(f"rotor_diameters_m required when mode={mode!r}")

    if air_density and rho_stack is None:
        raise ValueError("rho_stack required when air_density=True")

    cf_list = []
    for i, turbine_id in enumerate(turbine_ids):
        H = float(hub_heights_m[i])

        if mode == "direct_cf_quadrature":
            D = float(rotor_diameters_m[i])
            cf, _rews = _direct_cf_and_rews_for_turbine(
                A_stack=A_stack,
                k_stack=k_stack,
                rho_stack=rho_stack if air_density else None,
                u_grid=u_grid,
                p_curve=power_curves[i],
                hub_height=H,
                rotor_diameter=D,
                rews_n=rews_n,
                loss_factor=loss_factor,
                vertical_policy=vertical_policy,
                compute_cf=True,
            )
        elif mode == "momentmatch_weibull":
            D = float(rotor_diameters_m[i])
            cf, _rews = _momentmatch_cf_and_rews_for_turbine(
                A_stack=A_stack,
                k_stack=k_stack,
                rho_stack=rho_stack if air_density else None,
                u_grid=u_grid,
                p_curve=power_curves[i],
                hub_height=H,
                rotor_diameter=D,
                rews_n=rews_n,
                loss_factor=loss_factor,
                vertical_policy=vertical_policy,
            )
        else:
            # Legacy paths retained for explicit comparability.
            A_hub, k_hub = interpolate_weibull_params_to_height(A_stack, k_stack, H)
            if mode == "rews":
                D = float(rotor_diameters_m[i])
                f = _rews_moment_factor(
                    A_stack=A_stack,
                    k_stack=k_stack,
                    hub_height=H,
                    rotor_diameter=D,
                    n=rews_n,
                )
                A_eff = A_hub * f
            else:
                A_eff = A_hub

            pdf = weibull_probability_density(u_grid, k_hub, A_eff)
            if air_density:
                rho_hub = air_density_at_height(rho_stack, H)
                c = (rho_hub / RHO_0) ** (1.0 / 3.0)
                cf = _integrate_cf_with_density_correction(
                    pdf=pdf,
                    u_grid=u_grid,
                    p_curve=power_curves[i],
                    c=c,
                    loss_factor=loss_factor,
                )
            else:
                cf = _integrate_cf_no_density(
                    pdf=pdf,
                    u_grid=u_grid,
                    p_curve=power_curves[i],
                    loss_factor=loss_factor,
                )

        # Expand with turbine dimension
        cf = cf.expand_dims(turbine=[turbine_id])
        cf_list.append(cf)

    # Concatenate along turbine dimension (explicit coords policy avoids xarray FutureWarning).
    out = xr.concat(cf_list, dim="turbine", coords="different", compat="equals")
    out = out.rename("capacity_factors")

    # Set attrs (no compute)
    out.attrs["cleo:cf_mode"] = mode
    out.attrs["cleo:algo"] = "capacity_factors_v1"
    out.attrs["cleo:algo_version"] = "2"
    if mode in ("rews", "direct_cf_quadrature", "momentmatch_weibull"):
        out.attrs["cleo:rews_n"] = int(rews_n)
    out.attrs["cleo:air_density"] = int(air_density)  # int for netCDF4 compat

    return out


def rews_mps_v1(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    turbine_ids: tuple[str, ...],
    hub_heights_m: np.ndarray,
    rotor_diameters_m: np.ndarray,
    rho_stack: xr.DataArray | None = None,
    air_density: bool = False,
    rews_n: int = 12,
    vertical_policy: dict | None = None,
) -> xr.DataArray:
    """Compute first-class REWS output in m/s, dims ``(turbine, y, x)``."""
    if air_density and rho_stack is None:
        raise ValueError("rho_stack required when air_density=True")

    rews_list = []
    for i, turbine_id in enumerate(turbine_ids):
        H = float(hub_heights_m[i])
        D = float(rotor_diameters_m[i])
        _cf, rews = _direct_cf_and_rews_for_turbine(
            A_stack=A_stack,
            k_stack=k_stack,
            rho_stack=rho_stack if air_density else None,
            u_grid=np.array([0.0, 1.0], dtype=np.float64),
            p_curve=np.array([0.0, 0.0], dtype=np.float64),
            hub_height=H,
            rotor_diameter=D,
            rews_n=rews_n,
            loss_factor=1.0,
            vertical_policy=vertical_policy,
            compute_cf=False,
        )
        rews_list.append(rews.expand_dims(turbine=[turbine_id]))

    out = xr.concat(rews_list, dim="turbine", coords="different", compat="equals")
    out = out.rename("rews_mps")
    out.attrs["units"] = "m/s"
    out.attrs["cleo:algo"] = "rews_mps_v1"
    out.attrs["cleo:algo_version"] = "1"
    out.attrs["cleo:rews_n"] = int(rews_n)
    out.attrs["cleo:air_density"] = int(air_density)
    return out


def _direct_cf_and_rews_for_turbine(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    rho_stack: xr.DataArray | None,
    u_grid: np.ndarray,
    p_curve: np.ndarray,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
    loss_factor: float,
    vertical_policy: dict | None,
    compute_cf: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute direct rotor CF and REWS for one turbine."""
    if rews_n < 2:
        raise ValueError(f"rews_n must be >= 2, got {rews_n}")
    if rotor_diameter <= 0:
        raise ValueError(f"rotor_diameter must be > 0, got {rotor_diameter!r}")

    t, g = np.polynomial.legendre.leggauss(int(rews_n))
    chord = (2.0 / np.pi) * np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0))
    weights = g * chord
    weights = weights / np.sum(weights)

    z_nodes = float(hub_height) + (float(rotor_diameter) / 2.0) * t

    _mu, k_nodes, _A_nodes, A_prime_nodes = evaluate_weibull_at_heights(
        A_stack,
        k_stack,
        query_heights_m=z_nodes,
        rho_stack=rho_stack,
        policy=vertical_policy,
    )

    cf_acc: xr.DataArray | None = None
    m3_acc: xr.DataArray | None = None
    for j, zq in enumerate(z_nodes):
        wj = float(weights[j])
        A_j = A_prime_nodes.sel(query_height=float(zq))
        k_j = k_nodes.sel(query_height=float(zq))

        m3_j = np.exp(3.0 * np.log(A_j) + xr.apply_ufunc(gammaln, 1.0 + (3.0 / k_j), dask="parallelized"))
        m3_acc = (wj * m3_j) if m3_acc is None else (m3_acc + (wj * m3_j))

        if compute_cf:
            pdf_j = weibull_probability_density(u_grid, k_j, A_j)
            cf_j = _integrate_cf_no_density(
                pdf=pdf_j,
                u_grid=u_grid,
                p_curve=p_curve,
                loss_factor=loss_factor,
            )
            cf_acc = (wj * cf_j) if cf_acc is None else (cf_acc + (wj * cf_j))

    assert m3_acc is not None
    rews = (m3_acc ** (1.0 / 3.0)).rename("rews_mps")
    if cf_acc is None:
        cf_acc = xr.zeros_like(rews)
    return cf_acc.rename("capacity_factor"), rews


def _momentmatch_cf_and_rews_for_turbine(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    rho_stack: xr.DataArray | None,
    u_grid: np.ndarray,
    p_curve: np.ndarray,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
    loss_factor: float,
    vertical_policy: dict | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute rotor CF via moment-matched Weibull and return REWS."""
    if rews_n < 2:
        raise ValueError(f"rews_n must be >= 2, got {rews_n}")
    if rotor_diameter <= 0:
        raise ValueError(f"rotor_diameter must be > 0, got {rotor_diameter!r}")

    t, g = np.polynomial.legendre.leggauss(int(rews_n))
    chord = (2.0 / np.pi) * np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0))
    weights = g * chord
    weights = weights / np.sum(weights)
    z_nodes = float(hub_height) + (float(rotor_diameter) / 2.0) * t

    _mu, k_nodes, _A_nodes, A_prime_nodes = evaluate_weibull_at_heights(
        A_stack,
        k_stack,
        query_heights_m=z_nodes,
        rho_stack=rho_stack,
        policy=vertical_policy,
    )

    m1_acc: xr.DataArray | None = None
    m3_acc: xr.DataArray | None = None
    for j, zq in enumerate(z_nodes):
        wj = float(weights[j])
        A_j = A_prime_nodes.sel(query_height=float(zq))
        k_j = k_nodes.sel(query_height=float(zq))

        log_m1_j = np.log(A_j) + xr.apply_ufunc(gammaln, 1.0 + (1.0 / k_j), dask="parallelized")
        log_m3_j = 3.0 * np.log(A_j) + xr.apply_ufunc(gammaln, 1.0 + (3.0 / k_j), dask="parallelized")
        m1_j = np.exp(log_m1_j)
        m3_j = np.exp(log_m3_j)

        m1_acc = (wj * m1_j) if m1_acc is None else (m1_acc + (wj * m1_j))
        m3_acc = (wj * m3_j) if m3_acc is None else (m3_acc + (wj * m3_j))

    assert m1_acc is not None and m3_acc is not None
    rews = (m3_acc ** (1.0 / 3.0)).rename("rews_mps")

    # Moment match Weibull via r = m3 / m1^3.
    r = m3_acc / (m1_acc ** 3)
    k_rot = _solve_k_from_moment_ratio(r)

    log_A_rot = np.log(m1_acc) - xr.apply_ufunc(gammaln, 1.0 + (1.0 / k_rot), dask="parallelized")
    A_rot = np.exp(log_A_rot)

    pdf_rot = weibull_probability_density(u_grid, k_rot, A_rot)
    cf = _integrate_cf_no_density(
        pdf=pdf_rot,
        u_grid=u_grid,
        p_curve=p_curve,
        loss_factor=loss_factor,
    ).rename("capacity_factor")
    return cf, rews


def _solve_k_from_moment_ratio(
    r: xr.DataArray,
    *,
    k_lo: float = 0.6,
    k_hi: float = 12.0,
    iterations: int = 32,
) -> xr.DataArray:
    """Solve ``Gamma(1+3/k)/Gamma(1+1/k)^3 = r`` by vectorized bisection."""
    if k_lo <= 0 or k_hi <= k_lo:
        raise ValueError("Invalid k bracket")
    if iterations < 8:
        raise ValueError("iterations must be >= 8")

    def ratio_from_k(k_da: xr.DataArray) -> xr.DataArray:
        return np.exp(
            xr.apply_ufunc(gammaln, 1.0 + (3.0 / k_da), dask="parallelized")
            - 3.0 * xr.apply_ufunc(gammaln, 1.0 + (1.0 / k_da), dask="parallelized")
        )

    k_left = xr.full_like(r, float(k_lo), dtype=np.float64)
    k_right = xr.full_like(r, float(k_hi), dtype=np.float64)
    r_left = ratio_from_k(k_left)
    r_right = ratio_from_k(k_right)

    r_min = xr.where(r_left < r_right, r_left, r_right)
    r_max = xr.where(r_left > r_right, r_left, r_right)
    r_clamped = xr.where(r < r_min, r_min, xr.where(r > r_max, r_max, r))

    for _ in range(int(iterations)):
        k_mid = 0.5 * (k_left + k_right)
        r_mid = ratio_from_k(k_mid)
        go_right = r_mid > r_clamped
        k_left = xr.where(go_right, k_mid, k_left)
        k_right = xr.where(go_right, k_right, k_mid)

    return (0.5 * (k_left + k_right)).rename("k_rot")


# %% Height interpolation functions
def _interp_da_to_height_log(
    da: xr.DataArray,
    target_height: float | int,
) -> xr.DataArray:
    """
    Interpolate a DataArray with "height" dimension to a target height using log-height-linear interpolation.

    Internal helper for height interpolation. Uses x = ln(height) space.
    No extrapolation: target_height must be within [min(height), max(height)].

    This implementation is xarray-native and dask-friendly (no Python loops, no eager evaluation).

    :param da: DataArray with dim "height" (e.g., Weibull A/k at multiple heights)
    :param target_height: Target height for interpolation
    :return: DataArray without "height" dim, with coord hub_height=target_height
    :raises ValueError: If target_height is outside available height range or insufficient heights
    """
    # Validate height dimension exists
    if "height" not in da.dims:
        raise ValueError("DataArray must have a 'height' dimension")

    heights = np.asarray(da.coords["height"].data)
    if len(heights) < 2:
        raise ValueError(f"At least 2 heights required for interpolation, got {len(heights)}")

    # No extrapolation: check bounds
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    if target_height < h_min or target_height > h_max:
        raise ValueError(
            f"target_height={target_height} is outside available height range [{h_min}, {h_max}]. "
            "Extrapolation is not supported."
        )

    # Create ln_height coordinate and transform to log-space
    ln_heights = np.log(np.asarray(heights, dtype=np.float64))
    da_log = da.assign_coords(ln_height=("height", ln_heights))

    # Swap dims to use ln_height, sort, and interpolate
    da_log = da_log.swap_dims({"height": "ln_height"})
    da_log = da_log.sortby("ln_height")

    # Interpolate in log-height space (xarray-native, dask-friendly)
    ln_target = np.log(float(target_height))
    result = da_log.interp(ln_height=ln_target, method="linear")

    # Drop the ln_height coordinate (now scalar after interp)
    if "ln_height" in result.coords:
        result = result.drop_vars("ln_height")
    if "height" in result.coords:
        result = result.drop_vars("height")

    # Assign hub_height coordinate
    result = result.assign_coords(hub_height=target_height)
    result.name = da.name

    return result


def interpolate_weibull_params_to_height(
    weibull_A: xr.DataArray,
    weibull_k: xr.DataArray,
    target_height: float | int,
    *,
    method: str = "log_height_linear",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Interpolate Weibull A and k parameters to an arbitrary target height.

    Uses log-height-linear interpolation: x = ln(height).
    For each cell: A* = interp(x*, x_i, A_i), k* = interp(x*, x_i, k_i).

    :param weibull_A: DataArray with dim "height" containing A parameter at multiple heights
    :param weibull_k: DataArray with dim "height" containing k parameter at multiple heights
    :param target_height: Target height for interpolation (e.g., turbine hub height)
    :param method: Interpolation method. Only "log_height_linear" supported in v1.
    :return: Tuple (A_hub, k_hub) as DataArrays without "height" dim, with coord hub_height=target_height
    :raises ValueError: If method is not supported, or if target_height is outside available height range
    """
    if method != "log_height_linear":
        raise ValueError(f"Unsupported interpolation method: {method!r}. Only 'log_height_linear' is supported.")

    # Validate height dimension exists in both
    if "height" not in weibull_A.dims or "height" not in weibull_k.dims:
        raise ValueError("weibull_A and weibull_k must have a 'height' dimension")

    # Use shared interpolation helper
    A_hub = _interp_da_to_height_log(weibull_A, target_height)
    A_hub.name = "weibull_A"

    k_hub = _interp_da_to_height_log(weibull_k, target_height)
    k_hub.name = "weibull_k"

    return A_hub, k_hub


def _interp_da_to_heights_log(
    da: xr.DataArray,
    target_heights: np.ndarray,
    *,
    dim_name: str = "sample",
) -> xr.DataArray:
    """
    Log-height-linear interpolation for multiple target heights in one xarray-native call.

    Returns a DataArray with a new dimension `dim_name` (length = len(target_heights)).
    No extrapolation: all target heights must lie within the available height range.
    """
    if "height" not in da.dims:
        raise ValueError("DataArray must have a 'height' dimension")

    heights = np.asarray(da.coords["height"].data, dtype=np.float64)
    if heights.size < 2:
        raise ValueError(f"At least 2 heights required for interpolation, got {heights.size}")

    th = np.asarray(target_heights, dtype=np.float64)
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    if float(np.min(th)) < h_min or float(np.max(th)) > h_max:
        raise ValueError(
            f"target_heights outside available height range [{h_min}, {h_max}]. "
            "Extrapolation is not supported."
        )

    ln_heights = np.log(heights)
    da_log = da.assign_coords(ln_height=("height", ln_heights)).swap_dims({"height": "ln_height"}).sortby("ln_height")

    ln_targets = np.log(th)
    out = da_log.interp(ln_height=ln_targets, method="linear")

    # Rename the interpolation dimension and attach the physical heights as a coordinate.
    out = out.rename({"ln_height": dim_name})
    out = out.assign_coords({dim_name: np.arange(th.size, dtype=np.int64), f"{dim_name}_height": (dim_name, th)})
    return out


def _weibull_moment(
    *,
    A: xr.DataArray,
    k: xr.DataArray,
    p: int,
) -> xr.DataArray:
    """
    Compute the p-th raw moment of a Weibull(A, k) distribution:

        E[U^p] = A^p * Gamma(1 + p/k)

    This is vectorized and dask-friendly.
    """
    return (A ** p) * gamma(1 + (p / k))


def _rews_moment_factor(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    hub_height: float,
    rotor_diameter: float,
    method: str = "log_height_linear",
    n: int = 9,
) -> xr.DataArray:
    """
    Rotor-Equivalent Wind Speed (REWS) factor based on the cubic moment.

    Let U(z) ~ Weibull(A(z), k(z)).
    Define the rotor-area-averaged cubic moment:

        m3_rotor = (2/pi) * ∫_{-1}^{1} m3(H + R*t) * sqrt(1 - t^2) dt,
        where m3(z) = E[U(z)^3] = A(z)^3 * Gamma(1 + 3/k(z)),
              H is hub height, R = rotor_diameter/2.

    The REWS factor is:
        f = (m3_rotor / m3_hub)^(1/3)

    Numerical integration:
    Uses Gauss-Chebyshev quadrature of the second kind with `n` nodes, which is efficient
    for the weight sqrt(1 - t^2) and needs only `n` interpolations of A/k.

    Contract:
    - No extrapolation: rotor top/bottom must lie within available Weibull height range.
    - Returns a dask-friendly DataArray on the spatial grid (y, x).
    """
    if method != "log_height_linear":
        raise ValueError(f"Unsupported interpolation method: {method!r}. Only 'log_height_linear' is supported.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    R = float(rotor_diameter) / 2.0
    if R <= 0:
        raise ValueError(f"rotor_diameter must be > 0, got {rotor_diameter!r}")

    # Quadrature nodes for Chebyshev (2nd kind): t_i = cos(i*pi/(n+1)), weights ∝ sin^2(...)
    i = np.arange(1, n + 1, dtype=np.float64)
    theta = i * np.pi / (n + 1.0)
    t = np.cos(theta)  # in [-1, 1]
    w = np.sin(theta) ** 2  # positive weights

    # Physical sample heights along the rotor
    z = hub_height + R * t  # meters

    # Interpolate A(z), k(z) for all samples in one go
    A_s = _interp_da_to_heights_log(A_stack, z, dim_name="sample")
    k_s = _interp_da_to_heights_log(k_stack, z, dim_name="sample")

    # Cubic moment at samples and hub
    m3_s = _weibull_moment(A=A_s, k=k_s, p=3)
    A_hub = _interp_da_to_height_log(A_stack, hub_height)
    k_hub = _interp_da_to_height_log(k_stack, hub_height)
    m3_hub = _weibull_moment(A=A_hub, k=k_hub, p=3)

    # Weighted quadrature for (2/pi)∫ f(t) sqrt(1-t^2) dt
    # Using the identity: (2/pi)∫ ≈ 2/(n+1) * Σ sin^2(theta_i) * f(cos(theta_i))
    w_da = xr.DataArray(w, dims=("sample",), coords={"sample": np.arange(n, dtype=np.int64)})
    m3_rotor = (2.0 / (n + 1.0)) * (m3_s * w_da).sum(dim="sample")

    # REWS factor
    f = (m3_rotor / m3_hub) ** (1.0 / 3.0)
    f = f.rename("rews_factor")
    return f




# %% Weibull and cost functions

def weibull_probability_density(u_power_curve, weibull_k, weibull_a):
    """
    Calculate Weibull probability density at wind-speed grid u_power_curve.

    Performance contract:
    - Vectorized / broadcasted computation (dask-friendly).
    - Produces dims ('wind_speed', 'y', 'x') (plus any extra non-spatial dims from inputs),
      with wind_speed coordinate exactly equal to u_power_curve.

    Mathematical definition (for u > 0):
        f(u) = (k/a) * (u/a)^(k-1) * exp(-(u/a)^k)

    At u = 0:
        f(0) := 0 (continuous Weibull; avoids inf when k < 1)

    :param u_power_curve: 1D array-like wind speed grid (exact labels contract)
    :param weibull_k: raster k parameter
    :param weibull_a: raster a parameter
    :return: DataArray with wind_speed dimension
    """
    u = np.asarray(u_power_curve)
    u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})

    # Broadcasting: u_da (wind_speed) over raster (y,x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = u_da / weibull_a
        # Avoid the u==0 singular branch for k<1 (0**negative), which is later
        # explicitly set to 0 by contract.
        z_safe = xr.where(u_da == 0, 1.0, z)
        pdf = (weibull_k / weibull_a) * (z_safe ** (weibull_k - 1)) * np.exp(-(z ** weibull_k))

    # Exact u=0 handling (avoids inf for k<1); keep dtype float
    pdf = xr.where(u_da == 0, 0.0, pdf)

    pdf = pdf.transpose("wind_speed", ...)  # stable dim order
    return pdf.squeeze().rename("weibull_probability_density")

