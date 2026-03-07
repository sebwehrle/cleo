# Pure numeric assessment functions for wind resource analysis.
# Contract: No I/O, no network, no store access, no mutation of self.data.
# All spatial arrays stay lazy (dask-compatible).
from dataclasses import dataclass
import json
import logging
import numpy as np
import xarray as xr

from scipy.special import gamma, gammaln

from cleo.vertical import (
    evaluate_density_at_heights,
    evaluate_weibull_at_heights,
    weibull_mean_from_a_k,
)

logger = logging.getLogger(__name__)


# %% Constants
RHO_0 = 1.225  # Reference air density at sea level (kg/m³)
CF_METHOD_HUB_HEIGHT = "hub_height_weibull"
CF_METHOD_HUB_HEIGHT_REWS = "hub_height_weibull_rews_scaled"
CF_METHOD_ROTOR_NODE_AVERAGE = "rotor_node_average"
CF_METHOD_ROTOR_MOMENT_MATCHED = "rotor_moment_matched_weibull"
CF_METHODS = frozenset(
    {
        CF_METHOD_HUB_HEIGHT,
        CF_METHOD_HUB_HEIGHT_REWS,
        CF_METHOD_ROTOR_NODE_AVERAGE,
        CF_METHOD_ROTOR_MOMENT_MATCHED,
    }
)
CF_INTERPOLATIONS = frozenset({"auto", "ak_logz", "mu_cv_loglog"})


@dataclass(frozen=True)
class SingleWeibullInflow:
    """Rotor inflow represented by one effective Weibull distribution.

    :param A_eff: Effective Weibull scale parameter on ``(y, x)``.
    :type A_eff: xarray.DataArray
    :param k_eff: Effective Weibull shape parameter on ``(y, x)``.
    :type k_eff: xarray.DataArray
    :param rews_mps: Optional REWS diagnostic on ``(y, x)`` in m/s.
    :type rews_mps: xarray.DataArray | None
    :param density_scale: Optional air-density speed scale ``c=(rho/rho0)**(1/3)``
        on ``(y, x)``. When present, turbine integration applies density
        correction inside power-curve evaluation.
    :type density_scale: xarray.DataArray | None
    """

    A_eff: xr.DataArray
    k_eff: xr.DataArray
    rews_mps: xr.DataArray | None = None
    density_scale: xr.DataArray | None = None


@dataclass(frozen=True)
class WeightedNodeInflow:
    """Rotor inflow represented by weighted node-wise Weibull distributions.

    :param weights: Normalized quadrature weights ``(n_nodes,)``.
    :type weights: numpy.ndarray
    :param A_nodes: Node Weibull scale parameters ``(node, y, x)``.
    :type A_nodes: xarray.DataArray
    :param k_nodes: Node Weibull shape parameters ``(node, y, x)``.
    :type k_nodes: xarray.DataArray
    :param rews_mps: Optional REWS diagnostic on ``(y, x)`` in m/s.
    :type rews_mps: xarray.DataArray | None
    """

    weights: np.ndarray
    A_nodes: xr.DataArray
    k_nodes: xr.DataArray
    rews_mps: xr.DataArray | None = None


@dataclass(frozen=True)
class PreparedVerticalProfile:
    """Prepared vertical Weibull and density quantities for rotor approximation.

    :param query_heights_m: Query heights corresponding to the
        ``query_height`` axis, in meters.
    :type query_heights_m: numpy.ndarray
    :param A: Raw Weibull scale parameter evaluated at query heights, with dims
        ``(query_height, y, x)``.
    :type A: xarray.DataArray
    :param k: Weibull shape parameter evaluated at query heights, with dims
        ``(query_height, y, x)``.
    :type k: xarray.DataArray
    :param A_density_corrected: Optional density-corrected scale parameter
        ``A'`` on ``(query_height, y, x)`` for paths that prepare density
        upstream.
    :type A_density_corrected: xarray.DataArray | None
    :param rho: Optional air density at query heights on
        ``(query_height, y, x)``.
    :type rho: xarray.DataArray | None
    :param density_scale: Optional integration-time density correction factor
        ``c=(rho/rho0)**(1/3)`` for hub-height semantics, broadcastable to
        ``(y, x)``.
    :type density_scale: xarray.DataArray | None
    """

    query_heights_m: np.ndarray
    A: xr.DataArray
    k: xr.DataArray
    A_density_corrected: xr.DataArray | None
    rho: xr.DataArray | None
    density_scale: xr.DataArray | None = None


# %% Pure compute primitives


def compute_air_density_correction_core(
    *,
    elevation: xr.DataArray,
) -> xr.DataArray:
    """
    Pure compute function: air density correction factor based on elevation.

    This is a stateless, I/O-free computation primitive. The caller is responsible
    for providing an elevation DataArray that is already aligned to the canonical
    grid.

    Formula: rho_correction = 1.247015 * exp(-0.000104 * elevation) / 1.225

    Contract:
    - Output is NaN where elevation is NaN (propagates nodata).
    - Remains lazy: never forces eager evaluation (compute/load/values).
    - Returns a DataArray named "air_density_correction" on the same coords as elevation.

    :param elevation: Elevation DataArray (meters a.s.l.) on canonical grid.
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

    This compatibility wrapper delegates to the single canonical implementation
    in :mod:`cleo.vertical`.

    :param A: Weibull A (scale) parameter with dims such as ``(y, x)`` or
        ``(height, y, x)``.
    :type A: xarray.DataArray
    :param k: Weibull k (shape) parameter with the same dims as ``A``.
    :type k: xarray.DataArray
    :return: Mean wind speed in ``m/s`` with the same dims as ``A`` and ``k``.
    :rtype: xarray.DataArray
    """
    return weibull_mean_from_a_k(A, k)


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
        y,
        x,
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
            "PDF wind_speed coordinates do not exactly match integration grid. No nearest alignment is allowed."
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


def _rotor_nodes_and_weights(
    *,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rotor quadrature node heights and normalized weights.

    Uses Gauss-Legendre nodes ``t`` on ``[-1, 1]`` and rotor-chord weighting
    ``(2/pi) * sqrt(1 - t^2)``.

    :param hub_height: Turbine hub height in meters.
    :type hub_height: float
    :param rotor_diameter: Turbine rotor diameter in meters.
    :type rotor_diameter: float
    :param rews_n: Number of quadrature nodes, must be >= 2.
    :type rews_n: int
    :returns: Tuple ``(z_nodes, weights)`` where ``z_nodes`` are physical
        sample heights in meters and ``weights`` sum to one.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If ``rews_n < 2`` or ``rotor_diameter <= 0``.
    """
    if rews_n < 2:
        raise ValueError(f"rews_n must be >= 2, got {rews_n}")
    if rotor_diameter <= 0:
        raise ValueError(f"rotor_diameter must be > 0, got {rotor_diameter!r}")

    t, g = np.polynomial.legendre.leggauss(int(rews_n))
    chord = (2.0 / np.pi) * np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0))
    weights = g * chord
    weights = weights / np.sum(weights)
    z_nodes = float(hub_height) + (float(rotor_diameter) / 2.0) * t
    return z_nodes, weights


def resolve_capacity_factor_interpolation(
    *,
    method: str,
    interpolation: str,
) -> str:
    """Resolve the effective interpolation backend for a capacity-factor method.

    :param method: Public capacity-factor method family name.
    :type method: str
    :param interpolation: Public interpolation selector.
    :type interpolation: str
    :returns: Effective interpolation backend name.
    :rtype: str
    :raises ValueError: If ``method`` or ``interpolation`` is unsupported.
    """
    if method not in CF_METHODS:
        raise ValueError(f"Unknown capacity-factor method {method!r}. Supported: {sorted(CF_METHODS)!r}")
    if interpolation not in CF_INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation {interpolation!r}. Supported: {sorted(CF_INTERPOLATIONS)!r}")

    if method in {CF_METHOD_ROTOR_NODE_AVERAGE, CF_METHOD_ROTOR_MOMENT_MATCHED}:
        return "mu_cv_loglog" if interpolation == "auto" else interpolation

    if interpolation == "auto":
        return "ak_logz"
    return interpolation


def _query_heights_for_method(
    *,
    method: str,
    hub_height_m: float,
    rotor_diameter_m: float | None,
    rews_n: int,
) -> np.ndarray:
    """Resolve vertical query heights required by a capacity-factor method.

    :param method: Capacity-factor method family.
    :type method: str
    :param hub_height_m: Turbine hub height in meters.
    :type hub_height_m: float
    :param rotor_diameter_m: Rotor diameter in meters when rotor information is
        required by the method.
    :type rotor_diameter_m: float | None
    :param rews_n: Rotor quadrature resolution.
    :type rews_n: int
    :returns: Query heights in meters.
    :rtype: numpy.ndarray
    :raises ValueError: If the method requires rotor geometry but
        ``rotor_diameter_m`` is missing.
    """
    if method == CF_METHOD_HUB_HEIGHT:
        return np.array([float(hub_height_m)], dtype=np.float64)

    if rotor_diameter_m is None:
        raise ValueError(f"rotor_diameter_m required when method={method!r}")

    z_nodes, _weights = _rotor_nodes_and_weights(
        hub_height=float(hub_height_m),
        rotor_diameter=float(rotor_diameter_m),
        rews_n=int(rews_n),
    )
    if method == CF_METHOD_HUB_HEIGHT_REWS:
        return np.concatenate((np.array([float(hub_height_m)], dtype=np.float64), z_nodes))
    return z_nodes


def _prepare_vertical_profile(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    hub_height_m: float,
    rotor_diameter_m: float | None,
    rews_n: int,
    rho_stack: xr.DataArray | None,
    air_density: bool,
    method: str,
    vertical_policy: dict | None,
    interpolation: str,
) -> PreparedVerticalProfile:
    """Prepare a vertical profile for one capacity-factor method.

    :param A_stack: Weibull scale stack ``(height, y, x)``.
    :type A_stack: xarray.DataArray
    :param k_stack: Weibull shape stack ``(height, y, x)``.
    :type k_stack: xarray.DataArray
    :param hub_height_m: Turbine hub height in meters.
    :type hub_height_m: float
    :param rotor_diameter_m: Optional rotor diameter in meters.
    :type rotor_diameter_m: float | None
    :param rews_n: Rotor quadrature resolution.
    :type rews_n: int
    :param rho_stack: Optional air-density stack.
    :type rho_stack: xarray.DataArray | None
    :param air_density: Whether density-aware quantities should be prepared.
    :type air_density: bool
    :param method: Capacity-factor method family.
    :type method: str
    :param interpolation: Effective interpolation backend.
    :type interpolation: str
    :param vertical_policy: Optional vertical-policy overrides.
    :type vertical_policy: dict | None
    :returns: Prepared vertical profile.
    :rtype: PreparedVerticalProfile
    """
    query_heights_m = _query_heights_for_method(
        method=method,
        hub_height_m=hub_height_m,
        rotor_diameter_m=rotor_diameter_m,
        rews_n=rews_n,
    )
    _mu_q, k_q, A_q, A_prime_q = evaluate_weibull_at_heights(
        A_stack,
        k_stack,
        query_heights_m=query_heights_m,
        rho_stack=rho_stack if air_density else None,
        policy=vertical_policy,
        interpolation=interpolation,
    )
    rho_q = None
    density_scale = None
    if rho_stack is not None:
        rho_q = evaluate_density_at_heights(rho_stack, query_heights_m=query_heights_m, policy=vertical_policy)
        rho_q = rho_q.rename("rho")
    if air_density:
        if rho_q is None:
            raise ValueError("rho_stack required when air_density=True")
        density_scale = ((rho_q.isel(query_height=0, drop=True)) / RHO_0) ** (1.0 / 3.0)
        A_density_corrected = A_prime_q.rename("weibull_A_density_corrected")
    else:
        A_density_corrected = None

    return PreparedVerticalProfile(
        query_heights_m=query_heights_m,
        A=A_q.rename("weibull_A"),
        k=k_q.rename("weibull_k"),
        A_density_corrected=A_density_corrected,
        rho=rho_q,
        density_scale=density_scale,
    )


def _build_hub_inflow(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    hub_height: float,
    rho_stack: xr.DataArray | None,
    air_density: bool,
    interpolation: str = "ak_logz",
    vertical_policy: dict | None = None,
) -> SingleWeibullInflow:
    """Build single-Weibull inflow for ``hub_height_weibull`` method.

    :param A_stack: Weibull scale stack ``(height, y, x)``.
    :type A_stack: xarray.DataArray
    :param k_stack: Weibull shape stack ``(height, y, x)``.
    :type k_stack: xarray.DataArray
    :param hub_height: Hub height in meters.
    :type hub_height: float
    :param rho_stack: Air-density stack ``(height, y, x)`` when density is enabled.
    :type rho_stack: xarray.DataArray | None
    :param air_density: Whether to apply density correction.
    :type air_density: bool
    :param interpolation: Effective interpolation backend.
    :type interpolation: str
    :param vertical_policy: Optional vertical-policy overrides.
    :type vertical_policy: dict | None
    :returns: Effective single-Weibull inflow with density already prepared in
        ``A_eff`` when ``air_density=True``.
    :rtype: SingleWeibullInflow
    """
    profile = _prepare_vertical_profile(
        A_stack=A_stack,
        k_stack=k_stack,
        hub_height_m=hub_height,
        rotor_diameter_m=None,
        rho_stack=rho_stack,
        air_density=air_density,
        method=CF_METHOD_HUB_HEIGHT,
        rews_n=12,
        interpolation=interpolation,
        vertical_policy=vertical_policy,
    )
    return _build_hub_inflow_from_profile(profile=profile)


def _build_rews_inflow(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
    rho_stack: xr.DataArray | None,
    air_density: bool,
    interpolation: str = "ak_logz",
    vertical_policy: dict | None = None,
) -> SingleWeibullInflow:
    """Build single-Weibull inflow for ``hub_height_weibull_rews_scaled``.

    :param A_stack: Weibull scale stack ``(height, y, x)``.
    :type A_stack: xarray.DataArray
    :param k_stack: Weibull shape stack ``(height, y, x)``.
    :type k_stack: xarray.DataArray
    :param hub_height: Hub height in meters.
    :type hub_height: float
    :param rotor_diameter: Rotor diameter in meters.
    :type rotor_diameter: float
    :param rews_n: Number of rotor samples for REWS-factor integration.
    :type rews_n: int
    :param rho_stack: Air-density stack ``(height, y, x)`` when density is enabled.
    :type rho_stack: xarray.DataArray | None
    :param air_density: Whether to apply density correction.
    :type air_density: bool
    :param interpolation: Effective interpolation backend.
    :type interpolation: str
    :param vertical_policy: Optional vertical-policy overrides.
    :type vertical_policy: dict | None
    :returns: Effective single-Weibull inflow with density already prepared in
        ``A_eff`` when ``air_density=True``.
    :rtype: SingleWeibullInflow
    """
    profile = _prepare_vertical_profile(
        A_stack=A_stack,
        k_stack=k_stack,
        hub_height_m=hub_height,
        rotor_diameter_m=rotor_diameter,
        rho_stack=rho_stack,
        air_density=air_density,
        method=CF_METHOD_HUB_HEIGHT_REWS,
        rews_n=rews_n,
        interpolation=interpolation,
        vertical_policy=vertical_policy,
    )
    return _build_rews_inflow_from_profile(
        profile=profile,
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )


def _build_direct_inflow(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
    rho_stack: xr.DataArray | None,
    air_density: bool,
    vertical_policy: dict | None,
) -> WeightedNodeInflow:
    """Build weighted node-wise inflow for ``rotor_node_average`` method.

    :param A_stack: Weibull scale stack ``(height, y, x)``.
    :type A_stack: xarray.DataArray
    :param k_stack: Weibull shape stack ``(height, y, x)``.
    :type k_stack: xarray.DataArray
    :param hub_height: Hub height in meters.
    :type hub_height: float
    :param rotor_diameter: Rotor diameter in meters.
    :type rotor_diameter: float
    :param rews_n: Number of rotor nodes.
    :type rews_n: int
    :param rho_stack: Air-density stack ``(height, y, x)`` when density is enabled.
    :type rho_stack: xarray.DataArray | None
    :param air_density: Whether to apply density correction.
    :type air_density: bool
    :param vertical_policy: Optional vertical interpolation/extrapolation policy.
    :type vertical_policy: dict | None
    :returns: Weighted node-wise rotor inflow.
    :rtype: WeightedNodeInflow
    :raises ValueError: If density correction is requested without ``rho_stack``.
    """
    profile = _prepare_vertical_profile(
        A_stack=A_stack,
        k_stack=k_stack,
        hub_height_m=hub_height,
        rotor_diameter_m=rotor_diameter,
        rho_stack=rho_stack,
        air_density=air_density,
        method=CF_METHOD_ROTOR_NODE_AVERAGE,
        rews_n=rews_n,
        interpolation="mu_cv_loglog",
        vertical_policy=vertical_policy,
    )
    return _build_direct_inflow_from_profile(
        profile=profile,
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )


def _build_momentmatch_inflow(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
    rho_stack: xr.DataArray | None,
    air_density: bool,
    vertical_policy: dict | None,
) -> SingleWeibullInflow:
    """Build single-Weibull inflow for ``rotor_moment_matched_weibull``.

    :param A_stack: Weibull scale stack ``(height, y, x)``.
    :type A_stack: xarray.DataArray
    :param k_stack: Weibull shape stack ``(height, y, x)``.
    :type k_stack: xarray.DataArray
    :param hub_height: Hub height in meters.
    :type hub_height: float
    :param rotor_diameter: Rotor diameter in meters.
    :type rotor_diameter: float
    :param rews_n: Number of rotor nodes.
    :type rews_n: int
    :param rho_stack: Air-density stack ``(height, y, x)`` when density is enabled.
    :type rho_stack: xarray.DataArray | None
    :param air_density: Whether to apply density correction.
    :type air_density: bool
    :param vertical_policy: Optional vertical interpolation/extrapolation policy.
    :type vertical_policy: dict | None
    :returns: Effective moment-matched single-Weibull inflow.
    :rtype: SingleWeibullInflow
    :raises ValueError: If density correction is requested without ``rho_stack``.
    """
    profile = _prepare_vertical_profile(
        A_stack=A_stack,
        k_stack=k_stack,
        hub_height_m=hub_height,
        rotor_diameter_m=rotor_diameter,
        rho_stack=rho_stack,
        air_density=air_density,
        method=CF_METHOD_ROTOR_MOMENT_MATCHED,
        rews_n=rews_n,
        interpolation="mu_cv_loglog",
        vertical_policy=vertical_policy,
    )
    return _build_momentmatch_inflow_from_profile(
        profile=profile,
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )


def _build_hub_inflow_from_profile(*, profile: PreparedVerticalProfile) -> SingleWeibullInflow:
    """Build hub-height inflow from a prepared vertical profile.

    :param profile: Prepared single-height hub profile.
    :type profile: PreparedVerticalProfile
    :returns: Effective single-Weibull inflow.
    :rtype: SingleWeibullInflow
    :raises ValueError: If the profile does not contain exactly one query height.
    """
    if profile.A.sizes.get("query_height") != 1 or profile.k.sizes.get("query_height") != 1:
        raise ValueError("Hub-height inflow requires a prepared profile with exactly one query height.")
    A_hub = profile.A.isel(query_height=0, drop=True).rename("weibull_A")
    k_hub = profile.k.isel(query_height=0, drop=True).rename("weibull_k")
    return SingleWeibullInflow(A_eff=A_hub, k_eff=k_hub, rews_mps=None, density_scale=profile.density_scale)


def _build_rews_inflow_from_profile(
    *,
    profile: PreparedVerticalProfile,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
) -> SingleWeibullInflow:
    """Build REWS-scaled hub inflow from a prepared vertical profile.

    :param profile: Prepared profile containing hub height followed by rotor
        nodes on the ``query_height`` axis.
    :type profile: PreparedVerticalProfile
    :param hub_height: Turbine hub height in meters.
    :type hub_height: float
    :param rotor_diameter: Rotor diameter in meters.
    :type rotor_diameter: float
    :param rews_n: Rotor quadrature resolution.
    :type rews_n: int
    :returns: Effective single-Weibull inflow.
    :rtype: SingleWeibullInflow
    """
    _z_nodes, weights = _rotor_nodes_and_weights(
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )
    if profile.A.sizes.get("query_height") != int(rews_n) + 1:
        raise ValueError("REWS-scaled hub inflow requires hub height plus rotor nodes in prepared profile.")

    A_hub = profile.A.isel(query_height=0, drop=True).rename("weibull_A")
    k_hub = profile.k.isel(query_height=0, drop=True).rename("weibull_k")
    A_nodes = profile.A.isel(query_height=slice(1, None)).rename({"query_height": "node"})
    k_nodes = profile.k.isel(query_height=slice(1, None)).rename({"query_height": "node"})
    node_coord = np.arange(int(rews_n), dtype=np.int64)
    A_nodes = A_nodes.assign_coords(node=node_coord)
    k_nodes = k_nodes.assign_coords(node=node_coord)

    log_hub_m3 = 3.0 * np.log(A_hub) + xr.apply_ufunc(gammaln, 1.0 + (3.0 / k_hub), dask="parallelized")
    hub_m3 = np.exp(log_hub_m3)
    log_node_m3 = 3.0 * np.log(A_nodes) + xr.apply_ufunc(gammaln, 1.0 + (3.0 / k_nodes), dask="parallelized")
    node_m3 = np.exp(log_node_m3)
    w_da = xr.DataArray(weights, dims=("node",), coords={"node": node_coord})
    rotor_m3 = (node_m3 * w_da).sum(dim="node")
    rews_factor = (rotor_m3 / hub_m3) ** (1.0 / 3.0)

    A_eff = (A_hub * rews_factor).rename("weibull_A")
    return SingleWeibullInflow(A_eff=A_eff, k_eff=k_hub, rews_mps=None, density_scale=profile.density_scale)


def _build_direct_inflow_from_profile(
    *,
    profile: PreparedVerticalProfile,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
) -> WeightedNodeInflow:
    """Build rotor-node inflow from a prepared vertical profile.

    :param profile: Prepared rotor-node profile.
    :type profile: PreparedVerticalProfile
    :param hub_height: Turbine hub height in meters.
    :type hub_height: float
    :param rotor_diameter: Rotor diameter in meters.
    :type rotor_diameter: float
    :param rews_n: Rotor quadrature resolution.
    :type rews_n: int
    :returns: Weighted node-wise inflow.
    :rtype: WeightedNodeInflow
    """
    _z_nodes, weights = _rotor_nodes_and_weights(
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )
    if profile.A.sizes.get("query_height") != int(rews_n) or profile.k.sizes.get("query_height") != int(rews_n):
        raise ValueError("Rotor-node inflow requires one prepared profile slice per rotor node.")

    node_coord = np.arange(int(rews_n), dtype=np.int64)
    source_A = profile.A_density_corrected if profile.A_density_corrected is not None else profile.A
    A_nodes = source_A.rename({"query_height": "node"}).assign_coords(node=node_coord)
    k_nodes_out = profile.k.rename({"query_height": "node"}).assign_coords(node=node_coord)

    m3_nodes = np.exp(3.0 * np.log(A_nodes) + xr.apply_ufunc(gammaln, 1.0 + (3.0 / k_nodes_out), dask="parallelized"))
    w_da = xr.DataArray(weights, dims=("node",), coords={"node": node_coord})
    rews = ((m3_nodes * w_da).sum(dim="node") ** (1.0 / 3.0)).rename("rews_mps")
    return WeightedNodeInflow(weights=weights, A_nodes=A_nodes, k_nodes=k_nodes_out, rews_mps=rews)


def _build_momentmatch_inflow_from_profile(
    *,
    profile: PreparedVerticalProfile,
    hub_height: float,
    rotor_diameter: float,
    rews_n: int,
) -> SingleWeibullInflow:
    """Build moment-matched rotor inflow from a prepared vertical profile.

    :param profile: Prepared rotor-node profile.
    :type profile: PreparedVerticalProfile
    :param hub_height: Turbine hub height in meters.
    :type hub_height: float
    :param rotor_diameter: Rotor diameter in meters.
    :type rotor_diameter: float
    :param rews_n: Rotor quadrature resolution.
    :type rews_n: int
    :returns: Moment-matched single-Weibull inflow.
    :rtype: SingleWeibullInflow
    """
    direct_inflow = _build_direct_inflow_from_profile(
        profile=profile,
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )
    rews = direct_inflow.rews_mps

    m1_acc: xr.DataArray | None = None
    m3_acc: xr.DataArray | None = None
    for j, wj in enumerate(direct_inflow.weights):
        A_j = direct_inflow.A_nodes.isel(node=j)
        k_j = direct_inflow.k_nodes.isel(node=j)

        log_m1_j = np.log(A_j) + xr.apply_ufunc(gammaln, 1.0 + (1.0 / k_j), dask="parallelized")
        log_m3_j = 3.0 * np.log(A_j) + xr.apply_ufunc(gammaln, 1.0 + (3.0 / k_j), dask="parallelized")
        m1_j = np.exp(log_m1_j)
        m3_j = np.exp(log_m3_j)

        m1_acc = (wj * m1_j) if m1_acc is None else (m1_acc + (wj * m1_j))
        m3_acc = (wj * m3_j) if m3_acc is None else (m3_acc + (wj * m3_j))

    if rews is None or m1_acc is None or m3_acc is None:
        raise ValueError("Moment-match inflow requires non-empty rotor nodes and REWS diagnostic.")

    r = m3_acc / (m1_acc**3)
    k_rot = _solve_k_from_moment_ratio(r)
    log_A_rot = np.log(m1_acc) - xr.apply_ufunc(gammaln, 1.0 + (1.0 / k_rot), dask="parallelized")
    A_rot = np.exp(log_A_rot)

    return SingleWeibullInflow(A_eff=A_rot, k_eff=k_rot, rews_mps=rews)


def integrate_cf_from_inflow(
    *,
    inflow: SingleWeibullInflow | WeightedNodeInflow,
    u_grid: np.ndarray,
    p_curve: np.ndarray,
    loss_factor: float,
) -> xr.DataArray:
    """Integrate capacity factor from a rotor inflow surrogate.

    :param inflow: Rotor inflow surrogate, either single-Weibull or weighted nodes.
    :type inflow: SingleWeibullInflow | WeightedNodeInflow
    :param u_grid: Wind-speed integration grid in m/s.
    :type u_grid: numpy.ndarray
    :param p_curve: Turbine power curve sampled at ``u_grid`` (per-unit output).
    :type p_curve: numpy.ndarray
    :param loss_factor: Multiplicative loss factor applied to the integrated CF.
    :type loss_factor: float
    :returns: Capacity factor on ``(y, x)``.
    :rtype: xarray.DataArray
    :raises ValueError: If weighted inflow has inconsistent node shapes.
    """
    if isinstance(inflow, SingleWeibullInflow):
        pdf = weibull_probability_density(u_grid, inflow.k_eff, inflow.A_eff)
        if inflow.density_scale is None:
            cf = _integrate_cf_no_density(
                pdf=pdf,
                u_grid=u_grid,
                p_curve=p_curve,
                loss_factor=loss_factor,
            )
        else:
            cf = _integrate_cf_with_density_correction(
                pdf=pdf,
                u_grid=u_grid,
                p_curve=p_curve,
                c=inflow.density_scale,
                loss_factor=loss_factor,
            )
        return cf.rename("capacity_factor")

    n_nodes = int(inflow.weights.shape[0])
    if inflow.A_nodes.sizes.get("node") != n_nodes or inflow.k_nodes.sizes.get("node") != n_nodes:
        raise ValueError("WeightedNodeInflow node dimension must match weights length.")

    cf_acc: xr.DataArray | None = None
    for j, w_j in enumerate(inflow.weights):
        A_j = inflow.A_nodes.isel(node=j)
        k_j = inflow.k_nodes.isel(node=j)
        pdf_j = weibull_probability_density(u_grid, k_j, A_j)
        cf_j = _integrate_cf_no_density(
            pdf=pdf_j,
            u_grid=u_grid,
            p_curve=p_curve,
            loss_factor=loss_factor,
        )
        cf_acc = (float(w_j) * cf_j) if cf_acc is None else (cf_acc + (float(w_j) * cf_j))

    if cf_acc is None:
        raise ValueError("WeightedNodeInflow must contain at least one node.")
    return cf_acc.rename("capacity_factor")


def capacity_factors_v1(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    u_grid: np.ndarray,
    turbine_ids: tuple[str, ...],
    hub_heights_m: np.ndarray,
    power_curves: np.ndarray,
    method: str = CF_METHOD_ROTOR_NODE_AVERAGE,
    rotor_diameters_m: np.ndarray | None = None,
    rho_stack: xr.DataArray | None = None,
    air_density: bool = False,
    loss_factor: float = 1.0,
    rews_n: int = 12,
    interpolation: str = "auto",
    vertical_policy: dict | None = None,
) -> xr.DataArray:
    """
    Compute capacity factors for multiple turbines (pure numerics, turbine-loop).

    This is the v1 capacity factor algorithm with support for:
    - rotor node average (default)
    - rotor moment-matched Weibull
    - hub-height Weibull
    - hub-height Weibull with REWS scaling

    Contract: No I/O, no eager evaluation on spatial arrays (stays lazy/dask).
    Turbine-loop implementation (benchmarked best).

    :param A_stack: Weibull A parameter, dims (height, y, x)
    :param k_stack: Weibull k parameter, dims (height, y, x)
    :param u_grid: 1D wind speed grid for PDF/power curve
    :param turbine_ids: Tuple of turbine ID strings
    :param hub_heights_m: Hub heights for each turbine (n_turb,)
    :param power_curves: Power curves for each turbine (n_turb, n_ws)
    :param method: Public method family name.
    :type method: str
    :param rotor_diameters_m: Rotor diameters (required for rotor-aware methods
        and REWS-scaled hub-height method).
    :type rotor_diameters_m: numpy.ndarray | None
    :param rho_stack: Air density DataArray (required if air_density=True)
    :param air_density: If True, apply air density correction
    :param loss_factor: Loss correction factor (default 1.0)
    :param rews_n: Number of quadrature points for REWS integration
    :param interpolation: Interpolation selector. ``"auto"`` resolves by method
        family and explicit backends remain available for all method families.
    :type interpolation: str
    :return: DataArray with capacity factors, dims (turbine, y, x)
    """
    resolved_interpolation = resolve_capacity_factor_interpolation(method=method, interpolation=interpolation)

    if (
        method
        in {
            CF_METHOD_HUB_HEIGHT_REWS,
            CF_METHOD_ROTOR_NODE_AVERAGE,
            CF_METHOD_ROTOR_MOMENT_MATCHED,
        }
        and rotor_diameters_m is None
    ):
        raise ValueError(f"rotor_diameters_m required when method={method!r}")

    if air_density and rho_stack is None:
        raise ValueError("rho_stack required when air_density=True")

    cf_list = []
    for i, turbine_id in enumerate(turbine_ids):
        H = float(hub_heights_m[i])
        D = None if rotor_diameters_m is None else float(rotor_diameters_m[i])
        profile = _prepare_vertical_profile(
            A_stack=A_stack,
            k_stack=k_stack,
            hub_height_m=H,
            rotor_diameter_m=D,
            rho_stack=rho_stack,
            air_density=air_density,
            method=method,
            rews_n=rews_n,
            interpolation=resolved_interpolation,
            vertical_policy=vertical_policy,
        )

        if method == CF_METHOD_HUB_HEIGHT:
            inflow = _build_hub_inflow_from_profile(profile=profile)
        elif method == CF_METHOD_HUB_HEIGHT_REWS:
            assert D is not None
            inflow = _build_rews_inflow_from_profile(
                profile=profile,
                hub_height=H,
                rotor_diameter=D,
                rews_n=rews_n,
            )
        elif method == CF_METHOD_ROTOR_NODE_AVERAGE:
            assert D is not None
            inflow = _build_direct_inflow_from_profile(
                profile=profile,
                hub_height=H,
                rotor_diameter=D,
                rews_n=rews_n,
            )
        else:
            assert D is not None
            inflow = _build_momentmatch_inflow_from_profile(
                profile=profile,
                hub_height=H,
                rotor_diameter=D,
                rews_n=rews_n,
            )

        cf = integrate_cf_from_inflow(
            inflow=inflow,
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
    out.attrs["units"] = "1"  # dimensionless fraction
    out.attrs["cleo:cf_method"] = method
    out.attrs["cleo:interpolation"] = resolved_interpolation
    out.attrs["cleo:algo"] = "capacity_factors_v1"
    out.attrs["cleo:algo_version"] = "4"  # v4: renamed public method metadata and stored interpolation
    out.attrs["cleo:rews_n"] = int(rews_n)
    out.attrs["cleo:air_density"] = int(air_density)  # int for netCDF4 compat
    out.attrs["cleo:loss_factor"] = float(loss_factor)
    out.attrs["cleo:turbines_json"] = json.dumps(list(turbine_ids), ensure_ascii=True)

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
    interpolation: str = "auto",
    vertical_policy: dict | None = None,
) -> xr.DataArray:
    """Compute first-class REWS output in m/s.

    :param A_stack: Weibull scale stack on ``(height, y, x)``.
    :type A_stack: xarray.DataArray
    :param k_stack: Weibull shape stack on ``(height, y, x)``.
    :type k_stack: xarray.DataArray
    :param turbine_ids: Turbine identifiers for the output ``turbine`` axis.
    :type turbine_ids: tuple[str, ...]
    :param hub_heights_m: Hub heights in meters for each turbine.
    :type hub_heights_m: numpy.ndarray
    :param rotor_diameters_m: Rotor diameters in meters for each turbine.
    :type rotor_diameters_m: numpy.ndarray
    :param rho_stack: Optional air-density stack on ``(height, y, x)``.
    :type rho_stack: xarray.DataArray | None
    :param air_density: Whether to apply density-aware speed scaling.
    :type air_density: bool
    :param rews_n: Rotor quadrature resolution.
    :type rews_n: int
    :param interpolation: Interpolation selector for the rotor profile.
        ``"auto"`` resolves to the rotor-aware default backend.
    :type interpolation: str
    :param vertical_policy: Optional vertical-policy overrides.
    :type vertical_policy: dict | None
    :returns: Rotor-equivalent wind speed with dims ``(turbine, y, x)``.
    :rtype: xarray.DataArray
    :raises ValueError: If density correction is requested without ``rho_stack``.
    """
    if air_density and rho_stack is None:
        raise ValueError("rho_stack required when air_density=True")
    resolved_interpolation = resolve_capacity_factor_interpolation(
        method=CF_METHOD_ROTOR_NODE_AVERAGE,
        interpolation=interpolation,
    )

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
            interpolation=resolved_interpolation,
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
    out.attrs["cleo:interpolation"] = resolved_interpolation
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
    interpolation: str,
    vertical_policy: dict | None,
    compute_cf: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute direct rotor CF and REWS for one turbine.

    Delegates rotor inflow construction to the prepared-profile seam and
    integrates CF through ``integrate_cf_from_inflow(...)`` when requested.
    """
    profile = _prepare_vertical_profile(
        A_stack=A_stack,
        k_stack=k_stack,
        hub_height_m=hub_height,
        rotor_diameter_m=rotor_diameter,
        rews_n=rews_n,
        rho_stack=rho_stack,
        air_density=rho_stack is not None,
        method=CF_METHOD_ROTOR_NODE_AVERAGE,
        vertical_policy=vertical_policy,
        interpolation=interpolation,
    )
    inflow = _build_direct_inflow_from_profile(
        profile=profile,
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )
    rews = inflow.rews_mps
    if rews is None:
        raise ValueError("Direct inflow must provide REWS diagnostic.")

    if compute_cf:
        cf = integrate_cf_from_inflow(
            inflow=inflow,
            u_grid=u_grid,
            p_curve=p_curve,
            loss_factor=loss_factor,
        )
    else:
        cf = xr.zeros_like(rews).rename("capacity_factor")
    return cf.rename("capacity_factor"), rews.rename("rews_mps")


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
    """Compute rotor CF via moment-matched Weibull and return REWS.

    Delegates rotor inflow construction to the prepared-profile seam
    and integrates CF through ``integrate_cf_from_inflow(...)``.
    """
    profile = _prepare_vertical_profile(
        A_stack=A_stack,
        k_stack=k_stack,
        hub_height_m=hub_height,
        rotor_diameter_m=rotor_diameter,
        rews_n=rews_n,
        rho_stack=rho_stack,
        air_density=rho_stack is not None,
        method=CF_METHOD_ROTOR_MOMENT_MATCHED,
        vertical_policy=vertical_policy,
        interpolation="mu_cv_loglog",
    )
    inflow = _build_momentmatch_inflow_from_profile(
        profile=profile,
        hub_height=hub_height,
        rotor_diameter=rotor_diameter,
        rews_n=rews_n,
    )
    rews = inflow.rews_mps
    if rews is None:
        raise ValueError("Moment-match inflow must provide REWS diagnostic.")

    cf = integrate_cf_from_inflow(
        inflow=inflow,
        u_grid=u_grid,
        p_curve=p_curve,
        loss_factor=loss_factor,
    )
    return cf.rename("capacity_factor"), rews.rename("rews_mps")


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
            f"target_heights outside available height range [{h_min}, {h_max}]. Extrapolation is not supported."
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
    return (A**p) * gamma(1 + (p / k))


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
        pdf = (weibull_k / weibull_a) * (z_safe ** (weibull_k - 1)) * np.exp(-(z**weibull_k))

    # Exact u=0 handling (avoids inf for k<1); keep dtype float
    pdf = xr.where(u_da == 0, 0.0, pdf)

    pdf = pdf.transpose("wind_speed", ...)  # stable dim order
    return pdf.squeeze().rename("weibull_probability_density")
