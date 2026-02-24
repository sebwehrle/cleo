"""Neutral cost helper ownership for cross-layer reuse."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr


def turbine_overnight_cost(power, hub_height, rotor_diameter, year):
    """Estimate turbine overnight investment cost in EUR per kW."""
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    spec_power = power * 10 ** 6 / rotor_area
    cost = ((620 * np.log(hub_height)) - (1.68 * spec_power) + (182 * (2016 - year) ** 0.5) - 1005)
    return cost.astype("float")


def grid_connect_cost(power):
    """
    Calculate grid connection cost according to §54 (3,4) ElWOG.
    https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=20007045
    :param power: power in kW
    :return: absolute connection cost in EUR
    """
    cost = 50 * power
    return cost


def levelized_cost(
    power,
    capacity_factors,
    overnight_cost,
    grid_cost,
    om_fixed,
    om_variable,
    discount_rate,
    lifetime,
    hours_per_year=8766,
    per_mwh=True,
):
    """
    Calculate turbines' levelized cost of electricity.

    :param per_mwh: Return LCOE in currency per megawatt hour if true (default).
        Else returns LCOE in currency per kWh.
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
    :return: lcoe in EUR/kWh if per_mwh=False, otherwise EUR/MWh
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

    # Calculate net present amount of electricity generated over lifetime.
    npv_electricity = capacity_factors * hours_per_year * power * npv_factor

    # Calculate net present value of costs.
    npv_cost = (om_variable * capacity_factors * hours_per_year + om_fixed) * power * npv_factor
    npv_cost = npv_cost + overnight_cost + grid_cost

    lcoe = npv_cost / npv_electricity

    if per_mwh:
        return lcoe * 1000
    else:
        return lcoe


def _stable_json(value: dict) -> str:
    """Serialize dict deterministically for attrs provenance."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _turbine_ids_json(turbine_ids: tuple[str, ...]) -> str:
    return json.dumps(list(turbine_ids), ensure_ascii=True)


def _turbine_ids_json_from_lcoe(lcoe: xr.DataArray) -> str:
    raw = lcoe.attrs.get("cleo:turbine_ids_json")
    if isinstance(raw, str) and raw:
        return raw
    if "turbine" in lcoe.coords:
        tids = tuple(str(v) for v in np.asarray(lcoe.coords["turbine"].values))
        return _turbine_ids_json(tids)
    return json.dumps([], ensure_ascii=True)


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

    Contract: no I/O, stays lazy/dask-friendly.
    """
    economics_payload = {
        "discount_rate": float(discount_rate),
        "lifetime_a": int(lifetime_a),
        "om_fixed_eur_per_kw_a": float(om_fixed_eur_per_kw_a),
        "om_variable_eur_per_kwh": float(om_variable_eur_per_kwh),
        "turbine_cost_share": float(turbine_cost_share),
    }

    lcoe_list = []
    for i, turbine_id in enumerate(turbine_ids):
        p_kw = float(power_kw[i])
        oc_eur_per_kw = float(overnight_cost_eur_per_kw[i])

        # Absolute overnight cost for this turbine.
        oc_abs = oc_eur_per_kw * p_kw * float(turbine_cost_share)

        # Grid connection cost (EUR, scalar).
        gc_abs = float(grid_connect_cost(p_kw))

        cf_turb = cf.sel(turbine=turbine_id)
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

    out = xr.concat(lcoe_list, dim="turbine").rename("lcoe")
    out.attrs["units"] = "EUR/MWh"
    out.attrs["cleo:cf_mode"] = cf.attrs.get("cleo:cf_mode")
    out.attrs["cleo:hours_per_year"] = float(hours_per_year)
    out.attrs["cleo:turbine_ids_json"] = _turbine_ids_json(turbine_ids)
    out.attrs["cleo:economics_json"] = _stable_json(economics_payload)
    out.attrs["cleo:algo"] = "lcoe_v1"
    out.attrs["cleo:algo_version"] = "2"
    return out


def min_lcoe_turbine_idx(
    *,
    lcoe: xr.DataArray,
    turbine_ids: tuple[str, ...],
) -> xr.DataArray:
    """
    Find turbine index with minimum LCOE at each pixel (pure numerics).

    Returns int32 index (Zarr v3 safe, no string arrays).
    """
    lcoe_f = lcoe.fillna(np.inf)
    idx = lcoe_f.argmin(dim="turbine").astype(np.int32)
    all_invalid = lcoe.isnull().all(dim="turbine")
    idx = xr.where(all_invalid, np.int32(-1), idx).astype(np.int32).rename("min_lcoe_turbine")

    idx.attrs["cleo:turbine_ids_json"] = _turbine_ids_json(turbine_ids)
    idx.attrs["cleo:cf_mode"] = lcoe.attrs.get("cleo:cf_mode")
    idx.attrs["cleo:hours_per_year"] = lcoe.attrs.get("cleo:hours_per_year")
    if "cleo:economics_json" in lcoe.attrs:
        idx.attrs["cleo:economics_json"] = lcoe.attrs["cleo:economics_json"]
    idx.attrs["cleo:algo"] = "min_lcoe_turbine_idx"
    idx.attrs["cleo:algo_version"] = "2"
    idx.attrs["cleo:nodata_index"] = -1
    return idx


def optimal_power_kw(
    *,
    lcoe: xr.DataArray,
    power_kw: np.ndarray,
) -> xr.DataArray:
    """
    Get rated power of minimum-LCOE turbine at each pixel (pure numerics).

    Uses argmin index selection (avoids float equality comparison).
    """
    idx = lcoe.fillna(np.inf).argmin(dim="turbine")
    all_invalid = lcoe.isnull().all(dim="turbine")

    p_da = xr.DataArray(
        power_kw.astype(np.float64),
        dims=("turbine",),
        coords={"turbine": lcoe.coords["turbine"]},
    )
    p_sel = p_da.isel(turbine=idx).rename("optimal_power")
    p_sel = p_sel.where(~all_invalid)
    p_sel.attrs["units"] = "kW"
    p_sel.attrs["cleo:cf_mode"] = lcoe.attrs.get("cleo:cf_mode")
    p_sel.attrs["cleo:turbine_ids_json"] = _turbine_ids_json_from_lcoe(lcoe)
    p_sel.attrs["cleo:selection_basis"] = "min_lcoe_turbine_idx"
    if "cleo:economics_json" in lcoe.attrs:
        p_sel.attrs["cleo:economics_json"] = lcoe.attrs["cleo:economics_json"]
    p_sel.attrs["cleo:algo"] = "optimal_power_kw"
    p_sel.attrs["cleo:algo_version"] = "3"
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
    """
    idx = lcoe.fillna(np.inf).argmin(dim="turbine")

    cf_sel = cf.isel(turbine=idx)
    p_da = xr.DataArray(
        power_kw.astype(np.float64),
        dims=("turbine",),
        coords={"turbine": lcoe.coords["turbine"]},
    )
    p_sel_kw = p_da.isel(turbine=idx)

    energy = (cf_sel * p_sel_kw * float(hours_per_year) / 1e6).rename("optimal_energy")
    energy.attrs["units"] = "GWh/a"
    energy.attrs["cleo:cf_mode"] = lcoe.attrs.get("cleo:cf_mode")
    energy.attrs["cleo:hours_per_year"] = float(hours_per_year)
    energy.attrs["cleo:turbine_ids_json"] = _turbine_ids_json_from_lcoe(lcoe)
    energy.attrs["cleo:selection_basis"] = "min_lcoe_turbine_idx"
    if "cleo:economics_json" in lcoe.attrs:
        energy.attrs["cleo:economics_json"] = lcoe.attrs["cleo:economics_json"]
    energy.attrs["cleo:algo"] = "optimal_energy_gwh_a"
    energy.attrs["cleo:algo_version"] = "3"
    return energy
