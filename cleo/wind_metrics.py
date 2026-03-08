# %% imports
from dataclasses import dataclass
import json
import types
import numpy as np
import xarray as xr

from cleo.assess import (
    capacity_factors,
    rews_mps,
    resolve_capacity_factor_interpolation,
)
from cleo.economics import (
    lcoe_from_capacity_factors,
    min_lcoe_turbine_idx,
    optimal_power_kw,
    optimal_energy_gwh_a,
)
from cleo.vertical import evaluate_weibull_at_heights


def _extract_turbine_power_kw(
    turbines_meta: list[dict],
    turbine_ids: tuple[str, ...],
    wind: xr.Dataset | None = None,
) -> np.ndarray:
    """Extract rated power in kW from turbine metadata or wind dataset.

    Checks turbines_meta first for capacity/capacity_kw/capacity_mw keys.
    Falls back to wind["turbine_capacity"] data variable if metadata lacks capacity.
    """
    id_to_meta = {t["id"]: t for t in turbines_meta}
    id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}
    power_list = []
    for tid in turbine_ids:
        meta = id_to_meta[tid]
        if "capacity" in meta:
            power_list.append(float(meta["capacity"]))
        elif "capacity_kw" in meta:
            power_list.append(float(meta["capacity_kw"]))
        elif "capacity_mw" in meta:
            power_list.append(float(meta["capacity_mw"]) * 1000.0)
        elif wind is not None and "turbine_capacity" in wind:
            # Fall back to turbine_capacity data variable
            tidx = id_to_idx[tid]
            cap_val = wind["turbine_capacity"].isel(turbine=tidx).values
            power_list.append(float(cap_val))
        else:
            raise ValueError(f"Turbine {tid!r} missing capacity info; need 'capacity', 'capacity_kw', or 'capacity_mw'")
    return np.array(power_list, dtype=np.float64)


def _extract_overnight_cost_eur_per_kw(
    turbines_meta: list[dict],
    turbine_ids: tuple[str, ...],
    wind: xr.Dataset | None = None,
) -> np.ndarray:
    """Extract overnight cost in EUR/kW from turbine metadata or compute via Rinne model.

    Checks turbines_meta first for overnight_cost_eur_per_kw or overnight_cost keys.
    Falls back to Rinne cost model using wind store data variables if metadata lacks cost.
    """
    from cleo.economics import turbine_overnight_cost

    id_to_meta = {t["id"]: t for t in turbines_meta}
    id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}
    cost_list = []
    for tid in turbine_ids:
        meta = id_to_meta[tid]
        if "overnight_cost_eur_per_kw" in meta:
            cost_list.append(float(meta["overnight_cost_eur_per_kw"]))
        elif "overnight_cost" in meta:
            cost_list.append(float(meta["overnight_cost"]))
        elif wind is not None and all(
            v in wind
            for v in ("turbine_capacity", "turbine_hub_height", "turbine_rotor_diameter", "turbine_commissioning_year")
        ):
            # Fall back to Rinne cost model using wind store data
            tidx = id_to_idx[tid]
            power = float(wind["turbine_capacity"].isel(turbine=tidx).values)
            hub_height = float(wind["turbine_hub_height"].isel(turbine=tidx).values)
            rotor_diameter = float(wind["turbine_rotor_diameter"].isel(turbine=tidx).values)
            year = int(wind["turbine_commissioning_year"].isel(turbine=tidx).values)
            # Rinne model returns EUR/kW
            cost_eur_per_kw = turbine_overnight_cost(power / 1000.0, hub_height, rotor_diameter, year)
            cost_list.append(float(cost_eur_per_kw))
        else:
            raise ValueError(
                f"Turbine {tid!r} missing overnight cost; need 'overnight_cost_eur_per_kw' or 'overnight_cost'"
            )
    return np.array(cost_list, dtype=np.float64)


_CF_METHOD_DEFAULT = "rotor_node_average"
_WIND_SPEED_METHOD_DEFAULT = "height_weibull_mean"
_WIND_SPEED_METHOD_HEIGHT = "height_weibull_mean"
_WIND_SPEED_METHOD_ROTOR = "rotor_equivalent"
_WIND_SPEED_METHODS = frozenset({_WIND_SPEED_METHOD_HEIGHT, _WIND_SPEED_METHOD_ROTOR})
_INTERPOLATIONS = frozenset({"auto", "ak_logz", "mu_cv_loglog"})


@dataclass(frozen=True)
class _ResolvedWindAssessmentInputs:
    """Resolved store-backed inputs for wind assessment kernels.

    :param A_stack: Weibull scale stack on ``(height, y, x)``.
    :type A_stack: xarray.DataArray
    :param k_stack: Weibull shape stack on ``(height, y, x)``.
    :type k_stack: xarray.DataArray
    :param turbines_meta: Decoded turbine metadata from ``cleo_turbines_json``.
    :type turbines_meta: list[dict]
    :param hub_heights_m: Hub heights in meters for the selected turbines.
    :type hub_heights_m: numpy.ndarray
    :param rotor_diameters_m: Rotor diameters in meters for the selected turbines,
        when required by the caller.
    :type rotor_diameters_m: numpy.ndarray | None
    :param rho_stack: Optional air-density stack on ``(height, y, x)``.
    :type rho_stack: xarray.DataArray | None
    :param vertical_policy: Optional decoded vertical-policy overrides.
    :type vertical_policy: dict | None
    :param u_grid: Optional wind-speed integration grid.
    :type u_grid: numpy.ndarray | None
    :param power_curves: Optional selected turbine power curves.
    :type power_curves: numpy.ndarray | None
    """

    A_stack: xr.DataArray
    k_stack: xr.DataArray
    turbines_meta: list[dict]
    hub_heights_m: np.ndarray
    rotor_diameters_m: np.ndarray | None
    rho_stack: xr.DataArray | None
    vertical_policy: dict | None
    u_grid: np.ndarray | None = None
    power_curves: np.ndarray | None = None


def _require_land_valid_mask(land: xr.Dataset | None, *, metric_name: str) -> xr.DataArray:
    """Return ``valid_mask`` for a wind metric or raise a metric-specific error.

    :param land: Active landscape dataset.
    :type land: xarray.Dataset | None
    :param metric_name: Wind metric name used for error text.
    :type metric_name: str
    :returns: Valid-mask data array.
    :rtype: xarray.DataArray
    :raises ValueError: If the landscape store or ``valid_mask`` is missing.
    """
    if land is None or "valid_mask" not in land:
        raise ValueError(f"landscape store with valid_mask required for {metric_name}")
    return land["valid_mask"]


def _resolve_weibull_stacks(wind: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Resolve canonical Weibull stacks from the active wind dataset.

    :param wind: Active wind dataset.
    :type wind: xarray.Dataset
    :returns: Tuple ``(A_stack, k_stack)`` with ``height`` dimension.
    :rtype: tuple[xarray.DataArray, xarray.DataArray]
    :raises ValueError: If required Weibull variables are missing or malformed.
    """
    var_k = "weibull_k"
    if "weibull_A" not in wind or var_k not in wind:
        raise ValueError("wind store must have weibull_A and weibull_k")
    A_stack = wind["weibull_A"]
    if "height" not in A_stack.dims:
        raise ValueError("weibull_A must have height dimension")
    return A_stack, wind[var_k]


def _load_turbines_meta(wind: xr.Dataset) -> list[dict]:
    """Decode turbine metadata from the active wind store.

    :param wind: Active wind dataset.
    :type wind: xarray.Dataset
    :returns: Turbine metadata list from ``cleo_turbines_json``.
    :rtype: list[dict]
    :raises ValueError: If required turbine metadata is missing.
    """
    if "cleo_turbines_json" not in wind.attrs:
        raise ValueError("wind store must have cleo_turbines_json attr")
    if "turbine" not in wind.coords:
        raise ValueError("wind store must have turbine coordinate")
    return json.loads(wind.attrs["cleo_turbines_json"])


def _resolve_turbine_indices(turbines_meta: list[dict], turbines: tuple[str, ...]) -> list[int]:
    """Resolve selected turbine IDs to store indices.

    :param turbines_meta: Decoded turbine metadata.
    :type turbines_meta: list[dict]
    :param turbines: Requested turbine identifiers.
    :type turbines: tuple[str, ...]
    :returns: Indices into the active wind store turbine axis.
    :rtype: list[int]
    :raises ValueError: If any turbine ID is missing from the wind store.
    """
    turbine_id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}
    missing = [tid for tid in turbines if tid not in turbine_id_to_idx]
    if missing:
        raise ValueError(f"turbine {missing[0]!r} not in wind store")
    return [turbine_id_to_idx[tid] for tid in turbines]


def _resolve_rotor_diameters(
    wind: xr.Dataset,
    *,
    turbines: tuple[str, ...],
    turbines_meta: list[dict],
    tidx: list[int],
    metric_name: str,
) -> np.ndarray:
    """Resolve rotor diameters for selected turbines.

    :param wind: Active wind dataset.
    :type wind: xarray.Dataset
    :param turbines: Requested turbine identifiers.
    :type turbines: tuple[str, ...]
    :param turbines_meta: Decoded turbine metadata.
    :type turbines_meta: list[dict]
    :param tidx: Selected turbine indices into the wind store.
    :type tidx: list[int]
    :param metric_name: Metric/method name used for error text.
    :type metric_name: str
    :returns: Rotor diameters in meters.
    :rtype: numpy.ndarray
    :raises ValueError: If rotor diameter metadata is unavailable.
    """
    if "turbine_rotor_diameter" in wind:
        return wind["turbine_rotor_diameter"].isel(turbine=tidx).to_numpy()
    if "rotor_diameter" in wind:
        return wind["rotor_diameter"].isel(turbine=tidx).to_numpy()

    turbine_id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}
    diameters: list[float] = []
    for tid in turbines:
        meta = turbines_meta[turbine_id_to_idx[tid]]
        diameter = meta.get("rotor_diameter") or meta.get("rotor_diameter_m")
        if diameter is None:
            raise ValueError(
                f"{metric_name} requires rotor_diameter for turbine {tid!r}; "
                "not found in wind store or turbine metadata"
            )
        diameters.append(float(diameter))
    return np.array(diameters, dtype=np.float64)


def _load_vertical_policy(wind: xr.Dataset) -> dict | None:
    """Decode optional vertical-policy metadata from the active wind store.

    :param wind: Active wind dataset.
    :type wind: xarray.Dataset
    :returns: Decoded vertical policy or ``None`` when absent/invalid.
    :rtype: dict | None
    """
    policy_json = wind.attrs.get("cleo_vertical_policy_json")
    if not isinstance(policy_json, str) or not policy_json:
        return None
    try:
        return json.loads(policy_json)
    except json.JSONDecodeError:
        return None


def _resolve_wind_assessment_inputs(
    wind: xr.Dataset,
    *,
    turbines: tuple[str, ...],
    air_density: bool,
    metric_name: str,
    require_power_curve: bool,
    require_rotor_diameter: bool,
) -> _ResolvedWindAssessmentInputs:
    """Resolve store-backed inputs shared by wind assessment orchestrators.

    :param wind: Active wind dataset.
    :type wind: xarray.Dataset
    :param turbines: Requested turbine identifiers.
    :type turbines: tuple[str, ...]
    :param air_density: Whether density-aware inputs are required.
    :type air_density: bool
    :param metric_name: Metric/method label used for error text.
    :type metric_name: str
    :param require_power_curve: Whether power-curve data must be resolved.
    :type require_power_curve: bool
    :param require_rotor_diameter: Whether rotor diameters must be resolved.
    :type require_rotor_diameter: bool
    :returns: Resolved input bundle for the requested assessment path.
    :rtype: _ResolvedWindAssessmentInputs
    :raises ValueError: If required wind-store variables are missing.
    """
    A_stack, k_stack = _resolve_weibull_stacks(wind)
    turbines_meta = _load_turbines_meta(wind)
    tidx = _resolve_turbine_indices(turbines_meta, turbines)

    if require_power_curve and "power_curve" not in wind:
        raise ValueError("wind store must have power_curve variable")

    rho_stack = None
    if air_density:
        if "rho" not in wind:
            raise ValueError("air_density=True but wind store missing 'rho' variable")
        rho_stack = wind["rho"]

    power_curves = None
    u_grid = None
    if require_power_curve:
        power_curves = wind["power_curve"].isel(turbine=tidx).to_numpy()
        u_grid = wind.coords["wind_speed"].to_numpy()

    rotor_diameters_m = None
    if require_rotor_diameter:
        rotor_diameters_m = _resolve_rotor_diameters(
            wind,
            turbines=turbines,
            turbines_meta=turbines_meta,
            tidx=tidx,
            metric_name=metric_name,
        )

    return _ResolvedWindAssessmentInputs(
        A_stack=A_stack,
        k_stack=k_stack,
        turbines_meta=turbines_meta,
        hub_heights_m=wind["turbine_hub_height"].isel(turbine=tidx).to_numpy(),
        rotor_diameters_m=rotor_diameters_m,
        rho_stack=rho_stack,
        vertical_policy=_load_vertical_policy(wind),
        u_grid=u_grid,
        power_curves=power_curves,
    )


def resolved_wind_output_name(*, metric: str, params: dict, data: xr.DataArray | None = None) -> str:
    """Return the staged/materialized variable name for a public wind metric.

    :param metric: Public metric key passed to ``compute(...)``.
    :type metric: str
    :param params: Metric parameters used for this result.
    :type params: dict
    :param data: Optional computed data array. When provided and named, its
        variable name takes precedence.
    :type data: xarray.DataArray | None
    :returns: Output variable name to use in ``atlas.wind.data`` and stores.
    :rtype: str
    """
    if data is not None and data.name:
        return str(data.name)
    if metric != "wind_speed":
        return metric
    method = str(params.get("method", _WIND_SPEED_METHOD_DEFAULT))
    if method == _WIND_SPEED_METHOD_ROTOR:
        return "rotor_equivalent_wind_speed"
    return "mean_wind_speed"


def _normalize_cf_method_options(
    *,
    method: str = _CF_METHOD_DEFAULT,
    interpolation: str = "auto",
    air_density: bool = False,
    rews_n: int = 12,
    loss_factor: float = 1.0,
) -> dict[str, object]:
    """Normalize public capacity-factor method options to canonical internal form."""
    resolved_interpolation = resolve_capacity_factor_interpolation(method=method, interpolation=interpolation)
    normalized_rews_n = int(rews_n)
    if method == "hub_height_weibull":
        normalized_rews_n = 12
    return {
        "method": method,
        "interpolation": resolved_interpolation,
        "air_density": bool(air_density),
        "rews_n": normalized_rews_n,
        "loss_factor": float(loss_factor),
    }


def _normalize_wind_speed_options(
    *,
    method: str = _WIND_SPEED_METHOD_DEFAULT,
    interpolation: str = "auto",
    air_density: bool = False,
    rews_n: int = 12,
) -> dict[str, object]:
    """Normalize public wind-speed method options to canonical internal form.

    :param method: Public wind-speed method name.
    :type method: str
    :param interpolation: Public interpolation selector.
    :type interpolation: str
    :param air_density: Whether density-aware scaling is requested.
    :type air_density: bool
    :param rews_n: Rotor quadrature resolution for rotor-equivalent wind speed.
    :type rews_n: int
    :returns: Canonicalized wind-speed option dictionary.
    :rtype: dict[str, object]
    :raises ValueError: If ``method`` or ``interpolation`` is unsupported.
    """
    if method not in _WIND_SPEED_METHODS:
        raise ValueError(f"Unknown wind_speed method {method!r}. Supported: {sorted(_WIND_SPEED_METHODS)!r}")
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation {interpolation!r}. Supported: {sorted(_INTERPOLATIONS)!r}")

    if method == _WIND_SPEED_METHOD_ROTOR:
        resolved_interpolation = "mu_cv_loglog" if interpolation == "auto" else interpolation
    else:
        resolved_interpolation = "ak_logz" if interpolation == "auto" else interpolation

    return {
        "method": method,
        "interpolation": resolved_interpolation,
        "air_density": bool(air_density),
        "rews_n": int(rews_n),
    }


def _wind_metric_height_weibull_mean(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    height: int | float,
    interpolation: str = "auto",
) -> xr.DataArray:
    """Compute height-specific mean wind speed via the unified vertical evaluator."""
    var_k = "weibull_k"
    A = wind["weibull_A"]
    k = wind[var_k]

    options = _normalize_wind_speed_options(method=_WIND_SPEED_METHOD_HEIGHT, interpolation=interpolation)
    resolved_interpolation = str(options["interpolation"])
    query_height = float(height)
    if resolved_interpolation == "ak_logz":
        heights = np.asarray(A.coords["height"].values, dtype=np.float64)
        h_min = float(np.min(heights))
        h_max = float(np.max(heights))
        if query_height < h_min or query_height > h_max:
            raise ValueError(
                f"target_height={query_height} is outside available height range [{h_min}, {h_max}]. "
                "Extrapolation is not supported."
            )

    mu_q, _k_q, _A_q, _A_prime_q = evaluate_weibull_at_heights(
        A,
        k,
        query_heights_m=np.array([query_height], dtype=np.float64),
        policy=None,
        interpolation=resolved_interpolation,
    )
    da = (
        mu_q.rename("mean_wind_speed")
        .assign_coords(height=("query_height", mu_q.coords["query_height"].values))
        .swap_dims({"query_height": "height"})
        .drop_vars("query_height")
    )

    da = da.rename("mean_wind_speed")
    da.attrs["units"] = "m/s"
    da.attrs["cleo:wind_speed_method"] = _WIND_SPEED_METHOD_HEIGHT
    da.attrs["cleo:interpolation"] = resolved_interpolation

    if land is not None and "valid_mask" in land:
        da = da.where(land["valid_mask"])

    return da


def _wind_metric_wind_speed(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    method: str = _WIND_SPEED_METHOD_DEFAULT,
    interpolation: str = "auto",
    height: int | float | None = None,
    turbines: tuple[str, ...] | None = None,
    air_density: bool = False,
    rews_n: int = 12,
) -> xr.DataArray:
    """Compute public ``wind_speed`` metric via method-dependent semantics."""
    options = _normalize_wind_speed_options(
        method=method,
        interpolation=interpolation,
        air_density=air_density,
        rews_n=rews_n,
    )
    resolved_interpolation = str(options["interpolation"])

    if method == _WIND_SPEED_METHOD_HEIGHT:
        if height is None:
            raise ValueError("wind_speed(method='height_weibull_mean') requires height=...")
        if turbines is not None:
            raise ValueError("wind_speed(method='height_weibull_mean') does not accept turbines=...")
        if air_density:
            raise ValueError("wind_speed(method='height_weibull_mean') does not accept air_density=True")
        return _wind_metric_height_weibull_mean(
            wind,
            land,
            height=height,
            interpolation=resolved_interpolation,
        )

    if height is not None:
        raise ValueError("wind_speed(method='rotor_equivalent') does not accept height=...")
    if turbines is None:
        raise ValueError("wind_speed(method='rotor_equivalent') requires turbines=... or atlas.wind.select(...)")

    out = _wind_metric_rews_mps(
        wind,
        land,
        turbines=turbines,
        air_density=bool(options["air_density"]),
        rews_n=int(options["rews_n"]),
        interpolation=resolved_interpolation,
    )
    out = out.rename("rotor_equivalent_wind_speed")
    out.attrs["cleo:wind_speed_method"] = method
    out.attrs["cleo:interpolation"] = resolved_interpolation
    return out


def _wind_metric_capacity_factors(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    method: str = _CF_METHOD_DEFAULT,
    interpolation: str = "auto",
    air_density: bool = False,
    loss_factor: float = 1.0,
    rews_n: int = 12,
) -> xr.DataArray:
    """
    Orchestration for capacity_factors metric (calls assess.capacity_factors).

    Args:
        wind: Canonical wind dataset (must have weibull_A, weibull_k, power_curve).
        land: Canonical landscape dataset (must have valid_mask).
        turbines: Tuple of turbine IDs to compute.
        method: Public capacity-factor method family name.
        interpolation: Public interpolation selector.
        air_density: If True, apply air density correction using rho.
        loss_factor: Loss correction factor (default 1.0).
        rews_n: Number of quadrature points for REWS integration.

    Returns:
        DataArray with capacity factors, dims (turbine, y, x).
    """
    valid_mask = _require_land_valid_mask(land, metric_name="capacity_factors")
    options = _normalize_cf_method_options(
        method=method,
        interpolation=interpolation,
        air_density=air_density,
        rews_n=rews_n,
        loss_factor=loss_factor,
    )
    resolved_method = str(options["method"])
    resolved_interpolation = str(options["interpolation"])
    inputs = _resolve_wind_assessment_inputs(
        wind,
        turbines=turbines,
        air_density=bool(options["air_density"]),
        metric_name=f"method={resolved_method!r}",
        require_power_curve=True,
        require_rotor_diameter=resolved_method != "hub_height_weibull",
    )

    # Call pure numerics function
    result = capacity_factors(
        A_stack=inputs.A_stack,
        k_stack=inputs.k_stack,
        u_grid=inputs.u_grid,
        turbine_ids=turbines,
        hub_heights_m=inputs.hub_heights_m,
        power_curves=inputs.power_curves,
        method=resolved_method,
        rotor_diameters_m=inputs.rotor_diameters_m,
        rho_stack=inputs.rho_stack,
        air_density=bool(options["air_density"]),
        loss_factor=float(options["loss_factor"]),
        rews_n=int(options["rews_n"]),
        interpolation=resolved_interpolation,
        vertical_policy=inputs.vertical_policy,
    )

    # Apply valid_mask AFTER assess call
    result = result.where(valid_mask)

    return result


def _wind_metric_rews_mps(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    air_density: bool = False,
    rews_n: int = 12,
    interpolation: str = "auto",
) -> xr.DataArray:
    """Compute first-class rotor-equivalent wind speed.

    :param wind: Active wind dataset.
    :type wind: xarray.Dataset
    :param land: Active landscape dataset containing ``valid_mask``.
    :type land: xarray.Dataset | None
    :param turbines: Selected turbine identifiers.
    :type turbines: tuple[str, ...]
    :param air_density: Whether to apply density-aware scaling.
    :type air_density: bool
    :param rews_n: Rotor quadrature resolution.
    :type rews_n: int
    :param interpolation: Interpolation selector. ``"auto"`` resolves to the
        rotor-aware default backend.
    :type interpolation: str
    :returns: Rotor-equivalent wind speed with dims ``(turbine, y, x)``.
    :rtype: xarray.DataArray
    :raises ValueError: If required inputs are missing.
    """
    valid_mask = _require_land_valid_mask(land, metric_name="rews_mps")
    inputs = _resolve_wind_assessment_inputs(
        wind,
        turbines=turbines,
        air_density=air_density,
        metric_name="rews_mps",
        require_power_curve=False,
        require_rotor_diameter=True,
    )

    out = rews_mps(
        A_stack=inputs.A_stack,
        k_stack=inputs.k_stack,
        turbine_ids=turbines,
        hub_heights_m=inputs.hub_heights_m,
        rotor_diameters_m=inputs.rotor_diameters_m,
        rho_stack=inputs.rho_stack,
        air_density=air_density,
        rews_n=rews_n,
        interpolation=interpolation,
        vertical_policy=inputs.vertical_policy,
    )
    return out.where(valid_mask)


def _wind_metric_lcoe(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    method: str = _CF_METHOD_DEFAULT,
    interpolation: str = "auto",
    rews_n: int = 12,
    air_density: bool = False,
    loss_factor: float = 1.0,
    bos_cost_share: float = 0.0,
    grid_connect_cost_eur_per_kw: float = 50.0,
    om_fixed_eur_per_kw_a: float | None = None,
    om_variable_eur_per_kwh: float | None = None,
    discount_rate: float | None = None,
    lifetime_a: int | None = None,
    hours_per_year: float,
    _precomputed_cf: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Orchestration for LCOE metric.

    Computes capacity factors internally (unless _precomputed_cf is provided),
    then derives LCOE.

    Args:
        _precomputed_cf: Optional pre-computed CF DataArray for reuse.
            If provided, skips internal CF computation.
    """
    # Validate required params
    missing = []
    if om_fixed_eur_per_kw_a is None:
        missing.append("om_fixed_eur_per_kw_a")
    if om_variable_eur_per_kwh is None:
        missing.append("om_variable_eur_per_kwh")
    if discount_rate is None:
        missing.append("discount_rate")
    if lifetime_a is None:
        missing.append("lifetime_a")
    if missing:
        raise ValueError(
            f"LCOE requires parameters: {', '.join(missing)}. "
            f"Pass them to compute(), e.g., compute('lcoe', om_fixed_eur_per_kw_a=25, ...)"
        )

    # Use precomputed CF if provided, otherwise compute
    if _precomputed_cf is not None:
        cf = _precomputed_cf
    else:
        cf = _wind_metric_capacity_factors(
            wind,
            land,
            turbines=turbines,
            method=method,
            interpolation=interpolation,
            rews_n=rews_n,
            air_density=air_density,
            loss_factor=loss_factor,
        )

    # Extract turbine metadata
    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    power_kw = _extract_turbine_power_kw(turbines_meta, turbines, wind)
    overnight_cost = _extract_overnight_cost_eur_per_kw(turbines_meta, turbines, wind)

    # Compute LCOE
    lcoe = lcoe_from_capacity_factors(
        cf=cf,
        turbine_ids=turbines,
        power_kw=power_kw,
        overnight_cost_eur_per_kw=overnight_cost,
        bos_cost_share=bos_cost_share,
        grid_connect_cost_eur_per_kw=grid_connect_cost_eur_per_kw,
        om_fixed_eur_per_kw_a=om_fixed_eur_per_kw_a,
        om_variable_eur_per_kwh=om_variable_eur_per_kwh,
        discount_rate=discount_rate,
        lifetime_a=lifetime_a,
        hours_per_year=hours_per_year,
    )

    # valid_mask already applied via CF - NaN propagates through LCOE computation
    return lcoe


def _wind_metric_min_lcoe_turbine(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    method: str = _CF_METHOD_DEFAULT,
    interpolation: str = "auto",
    rews_n: int = 12,
    air_density: bool = False,
    loss_factor: float = 1.0,
    bos_cost_share: float = 0.0,
    grid_connect_cost_eur_per_kw: float = 50.0,
    om_fixed_eur_per_kw_a: float | None = None,
    om_variable_eur_per_kwh: float | None = None,
    discount_rate: float | None = None,
    lifetime_a: int | None = None,
    hours_per_year: float,
    _precomputed_cf: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Orchestration for min_lcoe_turbine metric.

    Returns the minimum-LCOE turbine index at each valid pixel.

    Invalid pixels are masked in the public result for consistency with the
    other clipped wind metrics, even though the low-level economics helper uses
    ``-1`` as an internal nodata sentinel.
    """
    lcoe = _wind_metric_lcoe(
        wind,
        land,
        turbines=turbines,
        method=method,
        interpolation=interpolation,
        rews_n=rews_n,
        air_density=air_density,
        loss_factor=loss_factor,
        bos_cost_share=bos_cost_share,
        grid_connect_cost_eur_per_kw=grid_connect_cost_eur_per_kw,
        om_fixed_eur_per_kw_a=om_fixed_eur_per_kw_a,
        om_variable_eur_per_kwh=om_variable_eur_per_kwh,
        discount_rate=discount_rate,
        lifetime_a=lifetime_a,
        hours_per_year=hours_per_year,
        _precomputed_cf=_precomputed_cf,
    )

    idx = min_lcoe_turbine_idx(lcoe=lcoe, turbine_ids=turbines)
    valid_mask = _require_land_valid_mask(land, metric_name="min_lcoe_turbine")
    public_mask = valid_mask & ~lcoe.isnull().all(dim="turbine")
    idx_public = idx.where(public_mask)
    idx_public.attrs = idx.attrs.copy()

    return idx_public


def _wind_metric_optimal_power(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    method: str = _CF_METHOD_DEFAULT,
    interpolation: str = "auto",
    rews_n: int = 12,
    air_density: bool = False,
    loss_factor: float = 1.0,
    bos_cost_share: float = 0.0,
    grid_connect_cost_eur_per_kw: float = 50.0,
    om_fixed_eur_per_kw_a: float | None = None,
    om_variable_eur_per_kwh: float | None = None,
    discount_rate: float | None = None,
    lifetime_a: int | None = None,
    hours_per_year: float,
    _precomputed_cf: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Orchestration for optimal_power metric.

    Returns rated power (kW) of minimum-LCOE turbine at each pixel.
    """
    lcoe = _wind_metric_lcoe(
        wind,
        land,
        turbines=turbines,
        method=method,
        interpolation=interpolation,
        rews_n=rews_n,
        air_density=air_density,
        loss_factor=loss_factor,
        bos_cost_share=bos_cost_share,
        grid_connect_cost_eur_per_kw=grid_connect_cost_eur_per_kw,
        om_fixed_eur_per_kw_a=om_fixed_eur_per_kw_a,
        om_variable_eur_per_kwh=om_variable_eur_per_kwh,
        discount_rate=discount_rate,
        lifetime_a=lifetime_a,
        hours_per_year=hours_per_year,
        _precomputed_cf=_precomputed_cf,
    )

    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    power_kw = _extract_turbine_power_kw(turbines_meta, turbines, wind)

    p = optimal_power_kw(lcoe=lcoe, power_kw=power_kw)

    # valid_mask already applied via CF -> LCOE; all-NaN pixels remain NaN.
    return p


def _wind_metric_optimal_energy(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    method: str = _CF_METHOD_DEFAULT,
    interpolation: str = "auto",
    rews_n: int = 12,
    air_density: bool = False,
    loss_factor: float = 1.0,
    bos_cost_share: float = 0.0,
    grid_connect_cost_eur_per_kw: float = 50.0,
    om_fixed_eur_per_kw_a: float | None = None,
    om_variable_eur_per_kwh: float | None = None,
    discount_rate: float | None = None,
    lifetime_a: int | None = None,
    hours_per_year: float,
    _precomputed_cf: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Orchestration for optimal_energy metric.

    Returns annual energy (GWh/a) of minimum-LCOE turbine at each pixel.
    Computes CF once (unless _precomputed_cf is provided), then derives LCOE directly.
    """
    # Validate LCOE params
    missing = []
    if om_fixed_eur_per_kw_a is None:
        missing.append("om_fixed_eur_per_kw_a")
    if om_variable_eur_per_kwh is None:
        missing.append("om_variable_eur_per_kwh")
    if discount_rate is None:
        missing.append("discount_rate")
    if lifetime_a is None:
        missing.append("lifetime_a")
    if missing:
        raise ValueError(
            f"optimal_energy requires LCOE parameters: {', '.join(missing)}. "
            f"Pass them to compute(), e.g., compute('optimal_energy', om_fixed_eur_per_kw_a=25, ...)"
        )

    # Use precomputed CF if provided, otherwise compute
    if _precomputed_cf is not None:
        cf = _precomputed_cf
    else:
        cf = _wind_metric_capacity_factors(
            wind,
            land,
            turbines=turbines,
            method=method,
            interpolation=interpolation,
            rews_n=rews_n,
            air_density=air_density,
            loss_factor=loss_factor,
        )

    # Extract turbine metadata
    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    power_kw = _extract_turbine_power_kw(turbines_meta, turbines, wind)
    overnight_cost = _extract_overnight_cost_eur_per_kw(turbines_meta, turbines, wind)

    # Compute LCOE directly from CF (avoids redundant CF computation)
    lcoe = lcoe_from_capacity_factors(
        cf=cf,
        turbine_ids=turbines,
        power_kw=power_kw,
        overnight_cost_eur_per_kw=overnight_cost,
        bos_cost_share=bos_cost_share,
        grid_connect_cost_eur_per_kw=grid_connect_cost_eur_per_kw,
        om_fixed_eur_per_kw_a=om_fixed_eur_per_kw_a,
        om_variable_eur_per_kwh=om_variable_eur_per_kwh,
        discount_rate=discount_rate,
        lifetime_a=lifetime_a,
        hours_per_year=hours_per_year,
    )

    e = optimal_energy_gwh_a(
        lcoe=lcoe,
        cf=cf,
        power_kw=power_kw,
        hours_per_year=hours_per_year,
    )

    # valid_mask already applied via CF -> LCOE -> e (NaN propagates)
    return e


# Wind metrics registry
_WIND_METRICS = {
    "wind_speed": {
        "fn": _wind_metric_wind_speed,
        "requires_turbines": False,
        "required": set(),
        "allowed": {"method", "interpolation", "height", "turbines", "air_density", "rews_n"},
    },
    "capacity_factors": {
        "fn": _wind_metric_capacity_factors,
        "requires_turbines": True,
        "required": set(),
        "allowed": {"turbines", "method", "interpolation", "air_density", "loss_factor", "rews_n"},
    },
    "lcoe": {
        "fn": _wind_metric_lcoe,
        "requires_turbines": True,
        "required": set(),
        "composed": True,  # accepts grouped cf={} and economics={} specs
        "allowed": {
            "turbines",
            "cf",
            "economics",
            "hours_per_year",  # injected internally by domain
        },
    },
    "min_lcoe_turbine": {
        "fn": _wind_metric_min_lcoe_turbine,
        "requires_turbines": True,
        "required": set(),
        "composed": True,
        "allowed": {
            "turbines",
            "cf",
            "economics",
            "hours_per_year",  # injected internally by domain
        },
    },
    "optimal_power": {
        "fn": _wind_metric_optimal_power,
        "requires_turbines": True,
        "required": set(),
        "composed": True,
        "allowed": {
            "turbines",
            "cf",
            "economics",
            "hours_per_year",  # injected internally by domain
        },
    },
    "optimal_energy": {
        "fn": _wind_metric_optimal_energy,
        "requires_turbines": True,
        "required": set(),
        "composed": True,
        "allowed": {
            "turbines",
            "cf",
            "economics",
            "hours_per_year",  # injected internally by domain
        },
    },
}


# CF spec defaults for composed metrics
_CF_SPEC_DEFAULTS = {
    "method": _CF_METHOD_DEFAULT,
    "interpolation": "auto",
    "air_density": False,
    "rews_n": 12,
    "loss_factor": 1.0,
}

# Economics fields that are required for LCOE computation
_REQUIRED_ECONOMICS_FIELDS = frozenset(
    {
        "discount_rate",
        "lifetime_a",
        "om_fixed_eur_per_kw_a",
        "om_variable_eur_per_kwh",
    }
)

# Flat kwargs that are rejected for composed metrics (must use grouped specs)
_FLAT_CF_KWARGS = frozenset({"method", "interpolation", "air_density", "loss_factor", "rews_n"})
_FLAT_ECONOMICS_KWARGS = frozenset(
    {
        "discount_rate",
        "lifetime_a",
        "om_fixed_eur_per_kw_a",
        "om_variable_eur_per_kwh",
        "bos_cost_share",
        "grid_connect_cost_eur_per_kw",
    }
)


# =============================================================================
# Stable internal interface functions
# =============================================================================
# These functions provide a stable boundary interface for cleo.domains to access
# wind metric registry internals without importing underscore-prefixed symbols.


def list_wind_metrics() -> tuple[str, ...]:
    """Return tuple of available wind metric names.

    Returns:
        Tuple of metric names in sorted order.
    """
    return tuple(sorted(_WIND_METRICS.keys()))


def get_wind_metric_spec(name: str) -> types.MappingProxyType:
    """Return immutable metric specification for the given metric name.

    Args:
        name: The metric name (e.g., "capacity_factors", "lcoe").

    Returns:
        Immutable dict-like object with keys:
        - fn: callable metric function
        - requires_turbines: bool
        - required: frozenset of required parameter names
        - allowed: frozenset of allowed parameter names
        - composed: bool (True for metrics using grouped cf/economics specs)

    Raises:
        KeyError: If metric name is not found.
    """
    if name not in _WIND_METRICS:
        raise KeyError(f"Unknown wind metric: {name!r}")

    spec = _WIND_METRICS[name]
    # Normalize to immutable types
    required_params: set[str] = spec["required"]  # type: ignore[assignment]
    allowed_params: set[str] = spec["allowed"]  # type: ignore[assignment]
    normalized = {
        "fn": spec["fn"],
        "requires_turbines": spec["requires_turbines"],
        "required": frozenset(required_params),
        "allowed": frozenset(allowed_params),
        "composed": spec.get("composed", False),
    }
    return types.MappingProxyType(normalized)


def resolve_cf_spec(cf: dict | None) -> dict:
    """Resolve CF spec dict with defaults for composed metrics.

    Args:
        cf: User-provided CF spec dict, or None for defaults.

    Returns:
        Complete CF spec dict with all required keys:
        - method: str (default "rotor_node_average")
        - interpolation: str (default "mu_cv_loglog" after auto resolution)
        - air_density: bool (default False)
        - rews_n: int (default 12)
        - loss_factor: float (default 1.0)
    """
    result = dict(_CF_SPEC_DEFAULTS)
    if cf is not None:
        result.update(cf)
    return _normalize_cf_method_options(
        method=str(result["method"]),
        interpolation=str(result["interpolation"]),
        air_density=bool(result["air_density"]),
        rews_n=int(result["rews_n"]),
        loss_factor=float(result["loss_factor"]),
    )


def required_economics_fields() -> frozenset[str]:
    """Return frozenset of required economics field names for LCOE computation.

    Returns:
        Frozenset containing: discount_rate, lifetime_a,
        om_fixed_eur_per_kw_a, om_variable_eur_per_kwh.
    """
    return frozenset(_REQUIRED_ECONOMICS_FIELDS)


def flat_cf_kwargs() -> frozenset[str]:
    """Return frozenset of CF kwargs that are rejected for composed metrics.

    These kwargs must be passed via the grouped cf={} spec for composed
    metrics (lcoe, min_lcoe_turbine, optimal_power, optimal_energy).

    Returns:
        Frozenset containing: method, interpolation, air_density, loss_factor, rews_n.
    """
    return _FLAT_CF_KWARGS


def flat_economics_kwargs() -> frozenset[str]:
    """Return frozenset of economics kwargs that are rejected for composed metrics.

    These kwargs must be passed via the grouped economics={} spec for composed
    metrics (lcoe, min_lcoe_turbine, optimal_power, optimal_energy).

    Returns:
        Frozenset containing: discount_rate, lifetime_a, om_fixed_eur_per_kw_a,
        om_variable_eur_per_kwh, bos_cost_share, grid_connect_cost_eur_per_kw.
    """
    return _FLAT_ECONOMICS_KWARGS
