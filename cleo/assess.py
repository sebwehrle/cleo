# Pure numeric assessment functions for wind resource analysis.
# Contract: No I/O, no network, no store access, no mutation of self.data.
# All spatial arrays stay lazy (dask-compatible).
import json
import numpy as np
import xarray as xr
import logging

from scipy.special import gamma

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

    # Align PDF to wind_speed order
    if "wind_speed" in pdf.dims:
        pdf_aligned = pdf.sel(wind_speed=u).transpose("wind_speed", ...)
    else:
        pdf_aligned = pdf

    # Build wind_speed as 1D DataArray for vectorized operations
    u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})

    # Compute equivalent wind speed: u_eq = u * c
    # Broadcasting: u_da (wind_speed,) * c (y, x) => (wind_speed, y, x)
    u_eq = u_da * c

    # Evaluate power curve at all equivalent wind speeds (vectorized, dask-friendly)
    p_eq = _interp_power_curve(u_eq, u, p)

    # Build integrand: pdf(u) * PC(u*c)
    # Both have dims (wind_speed, y, x) after alignment
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

    # Align PDF to wind_speed order
    if "wind_speed" in pdf.dims:
        pdf_aligned = pdf.sel(wind_speed=u).transpose("wind_speed", ...)
    else:
        pdf_aligned = pdf

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
    mode: str = "hub",
    rotor_diameters_m: np.ndarray | None = None,
    rho_stack: xr.DataArray | None = None,
    air_density: bool = False,
    loss_factor: float = 1.0,
    rews_n: int = 9,
) -> xr.DataArray:
    """
    Compute capacity factors for multiple turbines (pure numerics, turbine-loop).

    This is the v1 capacity factor algorithm with support for hub-height mode
    and REWS (rotor-equivalent wind speed) mode.

    Contract: No I/O, no eager evaluation on spatial arrays (stays lazy/dask).
    Turbine-loop implementation (benchmarked best).

    :param A_stack: Weibull A parameter, dims (height, y, x)
    :param k_stack: Weibull k parameter, dims (height, y, x)
    :param u_grid: 1D wind speed grid for PDF/power curve
    :param turbine_ids: Tuple of turbine ID strings
    :param hub_heights_m: Hub heights for each turbine (n_turb,)
    :param power_curves: Power curves for each turbine (n_turb, n_ws)
    :param mode: "hub" for hub-height, "rews" for rotor-equivalent wind speed
    :param rotor_diameters_m: Rotor diameters (required if mode="rews")
    :param rho_stack: Air density DataArray (required if air_density=True)
    :param air_density: If True, apply air density correction
    :param loss_factor: Loss correction factor (default 1.0)
    :param rews_n: Number of quadrature points for REWS integration
    :return: DataArray with capacity factors, dims (turbine, y, x)
    """
    # Validate mode
    if mode not in ("hub", "rews"):
        raise ValueError(f"mode must be 'hub' or 'rews', got {mode!r}")

    if mode == "rews" and rotor_diameters_m is None:
        raise ValueError("rotor_diameters_m required when mode='rews'")

    if air_density and rho_stack is None:
        raise ValueError("rho_stack required when air_density=True")

    cf_list = []
    for i, turbine_id in enumerate(turbine_ids):
        H = float(hub_heights_m[i])

        # Interpolate Weibull params to hub height
        A_hub, k_hub = interpolate_weibull_params_to_height(A_stack, k_stack, H)

        # Apply REWS correction if requested
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

        # Compute Weibull PDF
        pdf = weibull_probability_density(u_grid, k_hub, A_eff)

        # Compute capacity factor
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
    out = xr.concat(cf_list, dim="turbine", coords="different")
    out = out.rename("capacity_factors")

    # Set attrs (no compute)
    out.attrs["cleo:cf_mode"] = mode
    out.attrs["cleo:algo"] = "capacity_factors_v1"
    out.attrs["cleo:algo_version"] = "1"
    if mode == "rews":
        out.attrs["cleo:rews_n"] = int(rews_n)
    out.attrs["cleo:air_density"] = int(air_density)  # int for netCDF4 compat

    return out


def lcoe_v1_from_capacity_factors(
    *,
    cf: xr.DataArray,
    turbine_ids: tuple[str, ...],
    power_kw: np.ndarray,
    overnight_cost_eur_per_kw: np.ndarray,
    turbine_cost_share: float,
    om_fixed_eur_per_kw_a: float,
    om_variable_eur_per_kwh: float,
    discount_rate: float,
    lifetime_a: int,
    hours_per_year: float = 8766.0,
) -> xr.DataArray:
    """
    Compute LCOE from capacity factors (pure numerics, turbine-loop).

    Contract: No I/O, stays lazy/dask-friendly.

    :param cf: Capacity factors DataArray, dims (turbine, y, x)
    :param turbine_ids: Tuple of turbine ID strings
    :param power_kw: Rated power in kW for each turbine (n_turb,)
    :param overnight_cost_eur_per_kw: EUR/kW overnight cost for each turbine (n_turb,)
    :param turbine_cost_share: Share of location-independent investment cost
    :param om_fixed_eur_per_kw_a: Fixed O&M cost in EUR/kW/year
    :param om_variable_eur_per_kwh: Variable O&M cost in EUR/kWh
    :param discount_rate: Discount rate (fraction)
    :param lifetime_a: Lifetime in years
    :param hours_per_year: Hours per year (default 8766)
    :return: LCOE DataArray with dims (turbine, y, x), units EUR/MWh
    """
    lcoe_list = []
    for i, turbine_id in enumerate(turbine_ids):
        p_kw = float(power_kw[i])
        oc_eur_per_kw = float(overnight_cost_eur_per_kw[i])

        # Absolute overnight cost for this turbine
        oc_abs = oc_eur_per_kw * p_kw * float(turbine_cost_share)

        # Grid connection cost (EUR, scalar)
        gc_abs = float(grid_connect_cost(p_kw))

        # Get CF for this turbine
        cf_turb = cf.sel(turbine=turbine_id)

        # Compute LCOE using existing helper
        l = levelized_cost(
            power=p_kw,
            capacity_factors=cf_turb,
            overnight_cost=oc_abs,
            grid_cost=gc_abs,
            om_fixed=float(om_fixed_eur_per_kw_a),
            om_variable=float(om_variable_eur_per_kwh),
            discount_rate=float(discount_rate),
            lifetime=int(lifetime_a),
            hours_per_year=float(hours_per_year),
            per_mwh=True,
        )
        l = l.expand_dims(turbine=[turbine_id])
        lcoe_list.append(l)

    out = xr.concat(lcoe_list, dim="turbine")
    out = out.rename("lcoe")
    out.attrs["units"] = "EUR/MWh"
    out.attrs["cleo:cf_mode"] = cf.attrs.get("cleo:cf_mode")
    out.attrs["cleo:hours_per_year"] = float(hours_per_year)
    out.attrs["cleo:algo"] = "lcoe_v1"
    out.attrs["cleo:algo_version"] = "1"

    return out


def min_lcoe_turbine_idx(
    *,
    lcoe: xr.DataArray,
    turbine_ids: tuple[str, ...],
) -> xr.DataArray:
    """
    Find turbine index with minimum LCOE at each pixel (pure numerics).

    Returns int32 index (Zarr v3 safe, no string arrays).

    :param lcoe: LCOE DataArray with dims (turbine, y, x)
    :param turbine_ids: Tuple of turbine ID strings
    :return: DataArray with turbine index (int32), dims (y, x)
    """
    lcoe_f = lcoe.fillna(np.inf)
    idx = lcoe_f.argmin(dim="turbine").astype(np.int32)
    idx = idx.rename("min_lcoe_turbine")

    idx.attrs["cleo:turbine_ids_json"] = json.dumps(list(turbine_ids), ensure_ascii=True)
    idx.attrs["cleo:cf_mode"] = lcoe.attrs.get("cleo:cf_mode")
    idx.attrs["cleo:algo"] = "min_lcoe_turbine_idx"
    idx.attrs["cleo:algo_version"] = "1"

    return idx


def optimal_power_kw(
    *,
    lcoe: xr.DataArray,
    power_kw: np.ndarray,
) -> xr.DataArray:
    """
    Get rated power of minimum-LCOE turbine at each pixel (pure numerics).

    Uses argmin index selection (avoids float equality comparison).

    :param lcoe: LCOE DataArray with dims (turbine, y, x)
    :param power_kw: Rated power in kW for each turbine (n_turb,)
    :return: DataArray with optimal power in kW, dims (y, x)
    """
    # Use argmin to find the index of minimum LCOE (avoids float equality)
    idx = lcoe.fillna(np.inf).argmin(dim="turbine")

    p_da = xr.DataArray(
        power_kw.astype(np.float64),
        dims=("turbine",),
        coords={"turbine": lcoe.coords["turbine"]},
    )
    # Select power at the min-LCOE turbine index
    p_sel = p_da.isel(turbine=idx).rename("optimal_power")
    p_sel.attrs["units"] = "kW"
    p_sel.attrs["cleo:cf_mode"] = lcoe.attrs.get("cleo:cf_mode")
    p_sel.attrs["cleo:algo"] = "optimal_power_kw"
    p_sel.attrs["cleo:algo_version"] = "2"

    return p_sel


def optimal_energy_gwh_a(
    *,
    lcoe: xr.DataArray,
    cf: xr.DataArray,
    power_kw: np.ndarray,
    hours_per_year: float = 8766.0,
) -> xr.DataArray:
    """
    Get annual energy output of minimum-LCOE turbine at each pixel (pure numerics).

    Uses argmin index selection (avoids float equality comparison).

    :param lcoe: LCOE DataArray with dims (turbine, y, x)
    :param cf: Capacity factors DataArray with dims (turbine, y, x)
    :param power_kw: Rated power in kW for each turbine (n_turb,)
    :param hours_per_year: Hours per year (default 8766)
    :return: DataArray with optimal energy in GWh/year, dims (y, x)
    """
    # Use argmin to find the index of minimum LCOE (avoids float equality)
    idx = lcoe.fillna(np.inf).argmin(dim="turbine")

    # Select CF at the min-LCOE turbine index
    cf_sel = cf.isel(turbine=idx)

    # Select power at the min-LCOE turbine index
    p_da = xr.DataArray(
        power_kw.astype(np.float64),
        dims=("turbine",),
        coords={"turbine": lcoe.coords["turbine"]},
    )
    p_sel_kw = p_da.isel(turbine=idx)

    # Energy = CF * Power * hours / 1e6 (kWh -> GWh)
    energy = (cf_sel * p_sel_kw * float(hours_per_year) / 1e6)
    energy = energy.rename("optimal_energy")
    energy.attrs["units"] = "GWh/a"
    energy.attrs["cleo:cf_mode"] = lcoe.attrs.get("cleo:cf_mode")
    energy.attrs["cleo:hours_per_year"] = float(hours_per_year)
    energy.attrs["cleo:algo"] = "optimal_energy_gwh_a"
    energy.attrs["cleo:algo_version"] = "2"

    return energy


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

    heights = da.coords["height"].values
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

    heights = np.asarray(da.coords["height"].values, dtype=np.float64)
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


def turbine_overnight_cost(power, hub_height, rotor_diameter, year):
    """
    calculates wind turbine investment cost in EUR per MW based on >>Rinne et al. (2018): Effects of turbine technology
    and land use on wind power resource potential, Nature Energy<<
    :param power: rated power in MW
    :param hub_height: hub height in meters
    :param rotor_diameter: rotor diameter in meters
    :param year: year of first commercial deployment
    :return: overnight investment cost in EUR per kW
    """
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    spec_power = power * 10 ** 6 / rotor_area
    cost = ((620 * np.log(hub_height)) - (1.68 * spec_power) + (182 * (2016 - year) ** 0.5) - 1005)
    return cost.astype('float')


def grid_connect_cost(power):
    """
    Calculates grid connection cost according to §54 (3,4) ElWOG
    https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=20007045
    :param power: power in kW
    :return:
    """
    cost = 50 * power
    return cost


def levelized_cost(power, capacity_factors, overnight_cost, grid_cost, om_fixed, om_variable, discount_rate, lifetime,
                   hours_per_year=8766, per_mwh=True):
    """
    Calculates wind turbines' levelized cost of electricity in EUR per MWh
    :param per_mwh: Returns LCOE in currency per megawatt hour if true (default). Else returns LCOE in currency per kwh.
    :type per_mwh: bool
    :param hours_per_year: Number of hours per year. Default is 8766 to account for leap years.
    :param power: rated power in kW
    :type power: float
    :param capacity_factors: wind turbine capacity factor (share of year)
    :type capacity_factors: xarray.DataArray
    :param overnight_cost: absolute overnight cost in EUR (lump sum for this turbine at this location)
    :type overnight_cost: float
    :param grid_cost: cost for connecting to the electricity grid
    :type grid_cost: xarray.DataArray
    :param om_fixed: EUR/kW
    :type om_fixed: float
    :param om_variable: EUR/kWh
    :type om_variable: float
    :param discount_rate: percent
    :type discount_rate: float
    :param lifetime: years
    :type lifetime: int
    :return: lcoe in EUR/kWh
    """

    def discount_factor(discount_rate, period):
        """
        Calculate the discount factor for a given discount rate and period.
        :param discount_rate: discount rate (fraction of 1)
        :type discount_rate: float
        :param period: Number of years
        :type period: int
        :return: Discount factor
        :rtype: float
        """
        if discount_rate == 0:
            return float(period)
        return (1 - (1 + discount_rate) ** (-period)) / discount_rate

    npv_factor = discount_factor(discount_rate, lifetime)

    # calculate net present amount of electricity generated over lifetime
    npv_electricity = capacity_factors * hours_per_year * power * npv_factor

    # calculate net present value of cost
    npv_cost = (om_variable * capacity_factors * hours_per_year + om_fixed) * power * npv_factor
    npv_cost = npv_cost + overnight_cost + grid_cost

    lcoe = npv_cost / npv_electricity

    if per_mwh:
        return lcoe * 1000
    else:
        return lcoe
