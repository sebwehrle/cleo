# %% imports
import json
import types
import numpy as np
import xarray as xr

from cleo.assess import (
    mean_wind_speed_from_weibull,
    capacity_factors_v1,
    rews_mps_v1,
)
from cleo.economics import (
    lcoe_v1_from_capacity_factors,
    min_lcoe_turbine_idx,
    optimal_power_kw,
    optimal_energy_gwh_a,
)


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


def _wind_metric_mean_wind_speed(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    height: int,
) -> xr.DataArray:
    """Compute mean wind speed from canonical wind Weibull parameters.

    :param wind: Canonical wind dataset with ``weibull_A``/``weibull_k`` and
        a ``height`` coordinate in meters.
    :type wind: xarray.Dataset
    :param land: Optional landscape dataset. When provided with
        ``valid_mask(y,x)``, invalid pixels are masked to ``NaN``.
    :type land: xarray.Dataset | None
    :param height: Query height in meters above ground.
    :type height: int
    :returns: Mean wind speed data array with dims ``("height", "y", "x")``.
        The returned ``height`` axis is a singleton containing the requested
        level, preserving explicit height semantics for downstream aggregation.
    :rtype: xarray.DataArray
    :raises ValueError: If requested height is not available in wind-store
        Weibull parameter coordinates.
    """
    # Get weibull params (canonical store uses weibull_A)
    var_A = "weibull_A" if "weibull_A" in wind else "weibull_a"
    var_k = "weibull_k"

    A = wind[var_A]
    k = wind[var_k]

    # Validate height exists
    if "height" not in A.coords:
        raise ValueError(f"No height dimension in wind store {var_A}")
    available = [int(h) for h in A.coords["height"].values]
    if height not in available:
        raise ValueError(f"height={height} not in wind store; available: {sorted(available)}")

    # Keep height as an explicit singleton axis for downstream aggregation.
    da = mean_wind_speed_from_weibull(A=A.sel(height=[height]), k=k.sel(height=[height]))

    # Apply valid_mask if available
    if land is not None and "valid_mask" in land:
        da = da.where(land["valid_mask"])

    return da


def _wind_metric_capacity_factors(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    air_density: bool = False,
    loss_factor: float = 1.0,
    mode: str = "direct_cf_quadrature",
    rews_n: int = 12,
) -> xr.DataArray:
    """
    Orchestration for capacity_factors metric (calls assess.capacity_factors_v1).

    Args:
        wind: Canonical wind dataset (must have weibull_A, weibull_k, power_curve).
        land: Canonical landscape dataset (must have valid_mask).
        turbines: Tuple of turbine IDs to compute.
        air_density: If True, apply air density correction using rho.
        loss_factor: Loss correction factor (default 1.0).
        mode: "direct_cf_quadrature" (default), "momentmatch_weibull",
            "hub", or legacy "rews".
        rews_n: Number of quadrature points for REWS integration.

    Returns:
        DataArray with capacity factors, dims (turbine, y, x).
    """
    # Validate inputs
    if land is None or "valid_mask" not in land:
        raise ValueError("landscape store with valid_mask required for capacity_factors")

    var_A = "weibull_A" if "weibull_A" in wind else "weibull_a"
    var_k = "weibull_k"

    if var_A not in wind or var_k not in wind:
        raise ValueError(f"wind store must have {var_A} and {var_k}")
    if "power_curve" not in wind:
        raise ValueError("wind store must have power_curve variable")
    if "turbine" not in wind.coords:
        raise ValueError("wind store must have turbine coordinate")

    # Get Weibull stacks
    A_stack = wind[var_A]
    k_stack = wind[var_k]

    if "height" not in A_stack.dims:
        raise ValueError(f"{var_A} must have height dimension")

    # Get wind speed grid
    u_grid = wind.coords["wind_speed"].to_numpy()

    # Build turbine ID to index mapping from cleo_turbines_json attr
    if "cleo_turbines_json" not in wind.attrs:
        raise ValueError("wind store must have cleo_turbines_json attr")
    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    turbine_id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}

    # Validate turbines exist
    available_turbines = set(turbine_id_to_idx.keys())
    for tid in turbines:
        if tid not in available_turbines:
            raise ValueError(f"turbine {tid!r} not in wind store")

    # Get turbine indices
    tidx = [turbine_id_to_idx[tid] for tid in turbines]

    # Gather turbine parameters (small arrays OK to load eagerly)
    hub_heights_m = wind["turbine_hub_height"].isel(turbine=tidx).to_numpy()
    power_curves = wind["power_curve"].isel(turbine=tidx).to_numpy()

    # Air density: get rho_stack if needed
    rho_stack = None
    if air_density:
        if "rho" not in wind:
            raise ValueError("air_density=True but wind store missing 'rho' variable")
        rho_stack = wind["rho"]

    # Rotor diameters required for rotor-aware modes.
    rotor_diameters_m = None
    if mode in ("rews", "direct_cf_quadrature", "momentmatch_weibull"):
        # Try wind var first
        if "turbine_rotor_diameter" in wind:
            rotor_diameters_m = wind["turbine_rotor_diameter"].isel(turbine=tidx).to_numpy()
        elif "rotor_diameter" in wind:
            rotor_diameters_m = wind["rotor_diameter"].isel(turbine=tidx).to_numpy()
        else:
            # Try attrs turbines meta
            diameters = []
            for tid in turbines:
                meta = turbines_meta[turbine_id_to_idx[tid]]
                d = meta.get("rotor_diameter") or meta.get("rotor_diameter_m")
                if d is None:
                    raise ValueError(
                        f"mode={mode!r} requires rotor_diameter for turbine {tid!r}; "
                        "not found in wind store or turbine metadata"
                    )
                diameters.append(float(d))
            rotor_diameters_m = np.array(diameters, dtype=np.float64)

    vertical_policy = None
    policy_json = wind.attrs.get("cleo_vertical_policy_json")
    if isinstance(policy_json, str) and policy_json:
        try:
            vertical_policy = json.loads(policy_json)
        except json.JSONDecodeError:
            vertical_policy = None

    # Call pure numerics function
    result = capacity_factors_v1(
        A_stack=A_stack,
        k_stack=k_stack,
        u_grid=u_grid,
        turbine_ids=turbines,
        hub_heights_m=hub_heights_m,
        power_curves=power_curves,
        mode=mode,
        rotor_diameters_m=rotor_diameters_m,
        rho_stack=rho_stack,
        air_density=air_density,
        loss_factor=loss_factor,
        rews_n=rews_n,
        vertical_policy=vertical_policy,
    )

    # Apply valid_mask AFTER assess call
    result = result.where(land["valid_mask"])

    return result


def _wind_metric_rews_mps(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    air_density: bool = False,
    rews_n: int = 12,
) -> xr.DataArray:
    """Compute first-class REWS output (m/s), dims ``(turbine, y, x)``."""
    if land is None or "valid_mask" not in land:
        raise ValueError("landscape store with valid_mask required for rews_mps")
    if "cleo_turbines_json" not in wind.attrs:
        raise ValueError("wind store must have cleo_turbines_json attr")

    var_A = "weibull_A" if "weibull_A" in wind else "weibull_a"
    var_k = "weibull_k"
    if var_A not in wind or var_k not in wind:
        raise ValueError(f"wind store must have {var_A} and {var_k}")
    if "turbine" not in wind.coords:
        raise ValueError("wind store must have turbine coordinate")

    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    turbine_id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}
    for tid in turbines:
        if tid not in turbine_id_to_idx:
            raise ValueError(f"turbine {tid!r} not in wind store")
    tidx = [turbine_id_to_idx[tid] for tid in turbines]

    hub_heights_m = wind["turbine_hub_height"].isel(turbine=tidx).to_numpy()
    if "turbine_rotor_diameter" in wind:
        rotor_diameters_m = wind["turbine_rotor_diameter"].isel(turbine=tidx).to_numpy()
    elif "rotor_diameter" in wind:
        rotor_diameters_m = wind["rotor_diameter"].isel(turbine=tidx).to_numpy()
    else:
        diameters = []
        for tid in turbines:
            meta = turbines_meta[turbine_id_to_idx[tid]]
            d = meta.get("rotor_diameter") or meta.get("rotor_diameter_m")
            if d is None:
                raise ValueError(
                    f"rews_mps requires rotor_diameter for turbine {tid!r}; not found in wind store or turbine metadata"
                )
            diameters.append(float(d))
        rotor_diameters_m = np.array(diameters, dtype=np.float64)

    rho_stack = None
    if air_density:
        if "rho" not in wind:
            raise ValueError("air_density=True but wind store missing 'rho' variable")
        rho_stack = wind["rho"]

    vertical_policy = None
    policy_json = wind.attrs.get("cleo_vertical_policy_json")
    if isinstance(policy_json, str) and policy_json:
        try:
            vertical_policy = json.loads(policy_json)
        except json.JSONDecodeError:
            vertical_policy = None

    out = rews_mps_v1(
        A_stack=wind[var_A],
        k_stack=wind[var_k],
        turbine_ids=turbines,
        hub_heights_m=hub_heights_m,
        rotor_diameters_m=rotor_diameters_m,
        rho_stack=rho_stack,
        air_density=air_density,
        rews_n=rews_n,
        vertical_policy=vertical_policy,
    )
    return out.where(land["valid_mask"])


def _wind_metric_lcoe(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    mode: str = "hub",
    rews_n: int = 9,
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
            mode=mode,
            rews_n=rews_n,
            air_density=air_density,
            loss_factor=loss_factor,
        )

    # Extract turbine metadata
    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    power_kw = _extract_turbine_power_kw(turbines_meta, turbines, wind)
    overnight_cost = _extract_overnight_cost_eur_per_kw(turbines_meta, turbines, wind)

    # Compute LCOE
    lcoe = lcoe_v1_from_capacity_factors(
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
    mode: str = "hub",
    rews_n: int = 9,
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

    Returns turbine index (int32) with minimum LCOE at each pixel.
    """
    lcoe = _wind_metric_lcoe(
        wind,
        land,
        turbines=turbines,
        mode=mode,
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

    # valid_mask already applied via CF -> LCOE.
    # All-NaN pixels are encoded as nodata index (-1) in economics.min_lcoe_turbine_idx.
    return idx


def _wind_metric_optimal_power(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    mode: str = "hub",
    rews_n: int = 9,
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
        mode=mode,
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
    mode: str = "hub",
    rews_n: int = 9,
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
            mode=mode,
            rews_n=rews_n,
            air_density=air_density,
            loss_factor=loss_factor,
        )

    # Extract turbine metadata
    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    power_kw = _extract_turbine_power_kw(turbines_meta, turbines, wind)
    overnight_cost = _extract_overnight_cost_eur_per_kw(turbines_meta, turbines, wind)

    # Compute LCOE directly from CF (avoids redundant CF computation)
    lcoe = lcoe_v1_from_capacity_factors(
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
    "mean_wind_speed": {
        "fn": _wind_metric_mean_wind_speed,
        "requires_turbines": False,
        "required": {"height"},
        "allowed": {"height"},
    },
    "capacity_factors": {
        "fn": _wind_metric_capacity_factors,
        "requires_turbines": True,
        "required": set(),
        "allowed": {"turbines", "air_density", "loss_factor", "mode", "rews_n"},
    },
    "rews_mps": {
        "fn": _wind_metric_rews_mps,
        "requires_turbines": True,
        "required": set(),
        "allowed": {"turbines", "air_density", "rews_n"},
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
    "mode": "direct_cf_quadrature",
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
_FLAT_CF_KWARGS = frozenset({"mode", "air_density", "loss_factor", "rews_n"})
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
    normalized = {
        "fn": spec["fn"],
        "requires_turbines": spec["requires_turbines"],
        "required": frozenset(spec["required"]),
        "allowed": frozenset(spec["allowed"]),
        "composed": spec.get("composed", False),
    }
    return types.MappingProxyType(normalized)


def resolve_cf_spec(cf: dict | None) -> dict:
    """Resolve CF spec dict with defaults for composed metrics.

    Args:
        cf: User-provided CF spec dict, or None for defaults.

    Returns:
        Complete CF spec dict with all required keys:
        - mode: str (default "direct_cf_quadrature")
        - air_density: bool (default False)
        - rews_n: int (default 12)
        - loss_factor: float (default 1.0)
    """
    result = dict(_CF_SPEC_DEFAULTS)
    if cf is not None:
        result.update(cf)
    return result


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
        Frozenset containing: mode, air_density, loss_factor, rews_n.
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
