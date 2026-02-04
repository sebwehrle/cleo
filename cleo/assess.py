# resource assessment methods of the Atlas class
import inspect
import numpy as np
import xarray as xr
import rioxarray as rxr
import logging

from scipy.special import gamma
from cleo.utils import _match_to_template
from cleo.chunk import compute_chunked


# %% decorator
def accepts_parameter(func, parameter):
    """
    Check if a function accepts a specific parameter.

    :param func: Function to check.
    :type func: callable
    :param parameter: Parameter name to check for.
    :type parameter: str
    :return: True if the function accepts the parameter, False otherwise.
    :rtype: bool
    """
    signature = inspect.signature(func)
    return parameter in signature.parameters


def requires(dep_var_mapping):
    """
    Decorator to ensure dependencies are processed before executing the main function.

    :param dep_var_mapping: Dictionary mapping dependency method names to data variable names.
    :type dep_var_mapping: dict
    :return: Decorated function.
    :rtype: callable
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            """
            Wrapper function to check and process dependencies.

            :param self: Instance of the class.
            :type self: object
            :param args: Positional arguments for the main function.
            :param kwargs: Keyword arguments for the main function.
            :return: Result of the main function.
            """
            # Get the arguments passed to the main function
            func_signature = inspect.signature(func)
            provided_args = {k: v for k, v in kwargs.items() if k in func_signature.parameters}

            for dependency, data_var in dep_var_mapping.items():
                if data_var not in self.data.data_vars:
                    dep_func = getattr(self, dependency)
                    # Filter args to pass only those that the dependency accepts
                    dep_args = {k: v for k, v in provided_args.items() if accepts_parameter(dep_func, k)}
                    dep_func(**dep_args)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# %% independent methods
def compute_air_density_correction(self, chunk_size=None):
    """
    Compute air density correction factor rho based on elevation data.

    Contract:
    - Output is guaranteed to be on the same x/y grid as the Atlas template (if present),
      otherwise on the reference GWA raster grid.
    - CRS is enforced to self.parent.crs (Atlas CRS discipline).
    - Final assignment to self.data is exact-grid safe (prevents NaN stripes from implicit alignment).

    :param self: an instance of the _WindAtlas class
    :param chunk_size: size of chunks in pixels for chunked computation. If None, computation is not chunked
    """
    from cleo.loaders import load_elevation
    from pathlib import Path
    from rasterio.enums import Resampling
    from rasterio.crs import CRS

    def rho_correction(elevation):
        """
        Compute air density correction factor based on elevation (meters a.s.l.)
        """
        rho_correction_factor = 1.247015 * np.exp(-0.000104 * elevation) / 1.225
        return rho_correction_factor.squeeze().rename("air_density_correction_factor")

    # Normalize path to Path object (handles str input)
    parent_path = Path(self.parent.path) if not isinstance(self.parent.path, Path) else self.parent.path

    # Use A_100 raster as reference for grid alignment (raw GWA raster)
    path_raw_country = parent_path / "data" / "raw" / f"{self.parent.country}"
    reference_file = path_raw_country / f"{self.parent.country}_combined-Weibull-A_100.tif"

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Raw GWA files missing: {reference_file}. "
            f"Run Atlas init/download first to fetch data for {self.parent.country}."
        )

    reference_da = rxr.open_rasterio(reference_file).squeeze()

    # Determine template grid to enforce exact alignment
    if "template" in self.data.data_vars:
        template = self.data["template"]
        # Hard fail if template CRS contradicts Atlas CRS discipline
        tpl_crs = template.rio.crs
        if tpl_crs is None:
            raise ValueError("template DataArray missing CRS (.rio.crs is None)")
        if CRS.from_user_input(tpl_crs) != CRS.from_user_input(self.parent.crs):
            raise ValueError(
                f"template CRS ({tpl_crs}) does not match Atlas CRS ({self.parent.crs}). "
                "Refuse to compute air_density_correction on inconsistent grids."
            )
    else:
        # Fallback: enforce Atlas CRS on the reference grid for a consistent output CRS
        ref_crs = reference_da.rio.crs
        if ref_crs is None:
            raise ValueError("reference GWA raster missing CRS (.rio.crs is None)")
        if CRS.from_user_input(ref_crs) != CRS.from_user_input(self.parent.crs):
            reference_da = reference_da.rio.reproject(self.parent.crs, nodata=np.nan).squeeze()
        template = reference_da

    # Load elevation (legacy preferred; CopDEM fallback). Contract: must be made to match template.
    elevation = load_elevation(self.parent.path, self.parent.country, reference_da)

    src_crs = elevation.rio.crs
    if src_crs is None:
        raise ValueError("elevation DataArray missing CRS (.rio.crs is None)")

    # Optional clip, but ALWAYS reproject-match to template afterwards (exact grid contract)
    if self.parent.region is not None:
        clip_shape = self.parent.get_nuts_region(self.parent.region)
        if clip_shape is None:
            raise ValueError(f"region={self.parent.region!r} produced no geometry from get_nuts_region()")
        if clip_shape.crs != self.parent.crs:
            clip_shape = clip_shape.to_crs(self.parent.crs)
        elevation = elevation.rio.clip(clip_shape.geometry)

    # Enforce exact template grid + Atlas CRS with bilinear resampling (continuous elevation)
    elevation = elevation.rio.reproject_match(template, resampling=Resampling.bilinear, nodata=np.nan).squeeze()

    if chunk_size is None:
        rho = rho_correction(elevation)
    else:
        rho = compute_chunked(self, rho_correction, chunk_size, elevation=elevation)

    # Final exact-grid enforcement before storing (prevents silent alignment/NaN stripes)
    rho = rho.rio.reproject_match(template, resampling=Resampling.bilinear, nodata=np.nan).squeeze()

    # Hard check: x/y coords identical
    if not (np.array_equal(rho.coords["x"].values, template.coords["x"].values) and
            np.array_equal(rho.coords["y"].values, template.coords["y"].values)):
        raise ValueError("air_density_correction grid does not match template grid exactly (x/y coord mismatch)")

    self.data["air_density_correction"] = rho.rename("air_density_correction")
    logging.info(f"Air density correction for {self.parent.country} computed.")


def compute_mean_wind_speed(self, height, chunk_size=None, inplace=True):
    """
    Compute mean wind speed for a given height.

    Contract:
    - Idempotent: if height already exists, do nothing.
    - Preserves dataset-level coords (never rebuild self.data from data_vars).
    - If template exists, all height slices are exact-grid matched to template and concat uses join="exact".
      Otherwise concat uses join="outer" (legacy/no-template mode).

    :param self: An instance of the Atlas-class
    :param height: Height for which to compute mean wind speed
    :type height: int
    :param chunk_size: number of chunks for chunked computation. If None, computation is not chunked.
    :type chunk_size: int
    :param inplace: If True, add mean wind speed to Dataset (kept for backward compatibility; function always updates self.data)
    """
    # Early return if height already exists (idempotent)
    if "mean_wind_speed" in self.data and height in self.data["mean_wind_speed"].coords["height"].values:
        logging.info(f"Mean wind speed at height '{height} m' already exists, skipping computation")
        return

    def calculate_mean_wind_speed(weibull_a, weibull_k):
        mean_wind_speed = weibull_a * gamma(1 / weibull_k + 1)
        return mean_wind_speed.rename("mean_wind_speed").assign_attrs(units="m/s").squeeze()

    weibull_a, weibull_k = self.load_weibull_parameters(height)
    mean_wind_speed_data = (
        calculate_mean_wind_speed(weibull_a=weibull_a, weibull_k=weibull_k)
        if chunk_size is None
        else compute_chunked(self, calculate_mean_wind_speed, chunk_size, weibull_a=weibull_a, weibull_k=weibull_k)
    )

    # Align new slice to template grid if available
    has_template = "template" in self.data.data_vars
    if has_template:
        template = self.data["template"]
        mean_wind_speed_data = _match_to_template(mean_wind_speed_data, template)

    mean_wind_speed_data = mean_wind_speed_data.expand_dims(height=[height])

    if "mean_wind_speed" not in self.data:
        self.data["mean_wind_speed"] = mean_wind_speed_data
        return

    existing = self.data["mean_wind_speed"]

    # If template exists, align all existing slices to template before concatenation
    if has_template:
        template = self.data["template"]
        aligned_slices = []
        for h in existing.coords["height"].values:
            slice_2d = existing.sel(height=h).drop_vars("height")
            aligned_slice = _match_to_template(slice_2d, template)
            aligned_slices.append(aligned_slice.expand_dims(height=[h]))
        existing_aligned = xr.concat(aligned_slices, dim="height", join="exact")
        join_mode = "exact"
    else:
        existing_aligned = existing
        join_mode = "outer"

    combined = xr.concat([existing_aligned, mean_wind_speed_data], dim="height", join=join_mode)

    # Deterministic ordering and uniqueness
    combined = combined.sortby("height")

    # Update Dataset height coordinate safely:
    # drop the old variable first (it carries the old height dimension size), then set coord and reattach.
    ds = self.data.drop_vars("mean_wind_speed")
    ds = ds.assign_coords(height=combined.coords["height"])
    ds["mean_wind_speed"] = combined
    self.data = ds


def compute_wind_shear_coefficient(self, chunk_size=None):
    """
    Compute wind shear coefficient

    :param self: An instance of the Atlas-class
    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :type chunk_size: int
    """
    if "mean_wind_speed" not in self.data.data_vars or 50 not in self.data["mean_wind_speed"].coords["height"]:
        self.compute_mean_wind_speed(50, chunk_size)
    if 100 not in self.data["mean_wind_speed"].coords["height"]:
        self.compute_mean_wind_speed(100, chunk_size)

    u_mean_50 = self.data.mean_wind_speed.sel(height=50)
    u_mean_100 = self.data.mean_wind_speed.sel(height=100)

    # Mask invalid cells to avoid log(<=0) which produces inf/NaN
    valid = (u_mean_50 > 0) & (u_mean_100 > 0) & np.isfinite(u_mean_50) & np.isfinite(u_mean_100)

    # Suppress RuntimeWarning from log(0) - we handle invalid values via xr.where
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = xr.where(
            valid,
            (np.log(u_mean_100) - np.log(u_mean_50)) / (np.log(100) - np.log(50)),
            np.nan,
        )

    # Log warning if invalid cells exist (dask-safe scalar reduction)
    invalid_sum = (~valid).sum()
    if hasattr(invalid_sum.data, "compute"):
        invalid_sum = invalid_sum.compute()
    invalid_count = int(invalid_sum.values.item())
    if invalid_count > 0:
        total_count = int(valid.size)
        logging.warning(
            f"wind_shear: {invalid_count}/{total_count} cells masked (non-positive or non-finite wind speed)"
        )

    self.data['wind_shear'] = alpha.squeeze().rename("wind_shear")
    logging.info(f'Wind shear coefficient for {self.parent.country} computed.')


def compute_weibull_pdf(self, chunk_size=None):
    """
    Compute weibull probability density function for reference wind speeds

    :param self: an instance of the _WindAtlas class
    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :type chunk_size: int
    """
    a100, k100 = self.load_weibull_parameters(100)
    u = self.data.coords["wind_speed"].values

    # Align to template grid if available
    if "template" in self.data.data_vars:
        tpl = self.data["template"]
        a100 = _match_to_template(a100, tpl)
        k100 = _match_to_template(k100, tpl)

    inputs = {
        "weibull_a": a100,
        "weibull_k": k100,
        "u_power_curve": u
    }

    if chunk_size is None:
        p = weibull_probability_density(**inputs)
    else:
        p = compute_chunked(self, weibull_probability_density, chunk_size, **inputs)

    # Align result to template if available
    if "template" in self.data.data_vars:
        tpl = self.data["template"]
        p = _match_to_template(p, tpl)

    self.data["weibull_pdf"] = p
    logging.info(f'Weibull probability density function of wind speeds in {self.data.attrs["country"]} computed.')


# %% dependent methods
@requires({'compute_wind_shear_coefficient': 'wind_shear', 'compute_weibull_pdf': 'weibull_pdf'})
def simulate_capacity_factors(self, chunk_size=None, bias_correction=1):
    """
    Simulate capacity factors for specified wind turbine models

    :param self: An instance of the Atlas-class
    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :type chunk_size: int
    :param bias_correction: Bias correction factor for simulated capacity factors.
    :type bias_correction: float
    """
    cap_factor = []

    for turbine_model in self.data.coords["turbine"].values:
        inputs = {
            "weibull_pdf": self.data["weibull_pdf"],
            "wind_shear": self.data["wind_shear"],
            "u_power_curve": self.data.coords["wind_speed"].values,
            "p_power_curve": self.data.power_curve.sel(turbine=turbine_model).values,
            "h_turbine": self.get_turbine_attribute(turbine_model, "hub_height"),
            "correction_factor": bias_correction
        }

        if chunk_size is None:
            cf = capacity_factor(**inputs)
        else:
            cf = compute_chunked(self, capacity_factor, chunk_size, **inputs)

        # cf = cf * self.data["air_density_correction"]
        cf = cf.expand_dims(turbine=[turbine_model])
        logging.info(f"Capacity factors for {turbine_model} in {self.data.attrs['country']} computed.")
        # cap_factor = xr.merge([cap_factor, cf.rename("capacity_factor")])
        cap_factor.append(cf)

    cap_factor = xr.concat(cap_factor, dim="turbine")
    # cap_factor = cap_factor.assign_coords(turbine=self.wind_turbines)
    cap_factor = cap_factor.rename("capacity_factor")

    if "capacity_factors" not in self.data:
        self.data["capacity_factors"] = cap_factor
    else:
        self.data = self.data.combine_first(cap_factor)


@requires({'simulate_capacity_factors': 'capacity_factors'})
def compute_lcoe(self, chunk_size=None, turbine_cost_share=1):
    """
    Compute levelized cost of electricity (LCOE) for specified wind turbine models

    :param self: an instance of the Atlas-class
    :param chunk_size: Size of chunk in pixels. If None, computation is not chunked.
    :type chunk_size: int
    :param turbine_cost_share: Share of location-independent investment cost to compute pseudo-LCOE (see
    Wehrle et al. 2023, Inferring Local Social Cost of Wind Power. Evidence from Lower Austria)
    :type turbine_cost_share: float
    """
    lcoe_list = []
    for turbine_model in self.data.coords["turbine"].values:

        inputs = {
            "power": self.get_turbine_attribute(turbine_model, "capacity"),
            "capacity_factors": self.data["capacity_factors"].sel(turbine=turbine_model),
            "overnight_cost": self.get_overnight_cost(turbine_model) *
                              self.get_turbine_attribute(turbine_model, "capacity") * turbine_cost_share,
            "grid_cost": grid_connect_cost(self.get_turbine_attribute(turbine_model, "capacity")),
            "om_fixed": self.get_cost_assumptions("fixed_om_cost"),
            "om_variable": self.get_cost_assumptions("variable_om_cost"),
            "discount_rate": self.get_cost_assumptions("discount_rate"),
            "lifetime": self.get_cost_assumptions("turbine_lifetime")
        }

        if chunk_size is None:
            lcoe = levelized_cost(**inputs)
        else:
            lcoe = compute_chunked(self, levelized_cost, chunk_size, **inputs)

        lcoe = lcoe.expand_dims(turbine=[turbine_model])
        lcoe_list.append(lcoe)

    self.data["lcoe"] = xr.concat(lcoe_list, dim='turbine').rename("lcoe").assign_attrs(units="EUR/MWh")
    logging.info(f"Levelized Cost of Electricity in {self.data.attrs['country']} computed.")


@requires({'compute_lcoe': 'lcoe'})
def compute_optimal_power_energy(self):
    # Compute optimal power using turbine metadata (not string parsing)
    least_cost_turbine = self.data["lcoe"].idxmin(dim='turbine').compute()

    # Build lookup dict from turbine labels to capacity using metadata
    turbine_labels = self.data.coords["turbine"].values
    capacity_lookup = {
        label: self.get_turbine_attribute(label, "capacity")
        for label in turbine_labels
    }

    def get_capacity(turbine_label):
        if isinstance(turbine_label, str) and turbine_label in capacity_lookup:
            return float(capacity_lookup[turbine_label])
        else:
            return np.nan

    power = xr.apply_ufunc(get_capacity, least_cost_turbine, vectorize=True)

    self.data["optimal_power"] = power
    # compute optimal energy
    least_cost_index = self.data["lcoe"].fillna(9999).argmin(dim='turbine').compute()
    energy = self.data["capacity_factors"].isel(turbine=least_cost_index).drop_vars("turbine")
    energy = energy.assign_coords({'turbine': "min_lcoe"})
    # power is in kW; CF is dimensionless; 8766 h/year -> kWh/year; divide by 1e6 -> GWh/year
    energy = (energy * power * 8766 / 10 ** 6).rename("optimal_energy").assign_attrs(units="GWh/a")
    self.data["optimal_energy"] = energy


@requires({'compute_lcoe': 'lcoe'})
def minimum_lcoe(self):
    """
    Calculate the minimum lcoe for each location among all turbines
    """
    lcoe = self.data["lcoe"]
    lc_min = lcoe.min(dim='turbine', keep_attrs=True)
    lc_min = lc_min.assign_coords({'turbine': 'min_lcoe'})
    self.data["min_lcoe"] = lc_min.rename("minimal_lcoe")


# %% functions
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
        pdf = (weibull_k / weibull_a) * (z ** (weibull_k - 1)) * np.exp(-(z ** weibull_k))

    # Exact u=0 handling (avoids inf for k<1); keep dtype float
    pdf = xr.where(u_da == 0, 0.0, pdf)

    pdf = pdf.transpose("wind_speed", ...)  # stable dim order
    return pdf.squeeze().rename("weibull_probability_density")


def capacity_factor(weibull_pdf, wind_shear, u_power_curve, p_power_curve, h_turbine, h_reference=100,
                    correction_factor=1):
    """
    calculates wind turbine capacity factors given Weibull probability density pdf, roughness factor alpha, wind turbine
    power curve data in u_power_curve and p_power_curve, turbine height h_turbine and reference height of wind speed
    modelling h_reference
    :param correction_factor:
    :param weibull_pdf: probability density function from weibull_probability_density()
    :param wind_shear: wind shear coefficient
    :param u_power_curve: power curve wind speed
    :param p_power_curve: power curve output
    :param h_turbine: hub height of wind turbine in m
    :param h_reference: reference height at which weibull pdf is computed
    :return:
    """
    # Ensure u and p are numpy arrays
    u = np.asarray(u_power_curve)
    p = np.asarray(p_power_curve)

    # Validate input length
    if len(u) < 2:
        raise ValueError(
            f"u_power_curve must have at least 2 points for integration, got {len(u)}"
        )
    if len(p) != len(u):
        raise ValueError(
            f"p_power_curve length ({len(p)}) must match u_power_curve length ({len(u)})"
        )

    # Compute shear scaling factor s(y,x) = (h_turbine / h_reference) ** wind_shear
    s = (h_turbine / h_reference) ** wind_shear.values

    # Get spatial shape from wind_shear
    spatial_shape = wind_shear.shape

    # Helper to evaluate power curve at hub-height wind speed (2D)
    def eval_power_curve(u_ref_scalar, s_2d):
        u_hub_2d = u_ref_scalar * s_2d
        flat = u_hub_2d.ravel()
        p_flat = np.interp(flat, u, p, left=0.0, right=0.0)
        return p_flat.reshape(spatial_shape)

    # Streaming trapezoidal integration over u_ref
    acc = np.zeros(spatial_shape, dtype=np.float64)

    # Exact wind_speed grid contract (no tolerant float matching).
    # Order may differ; alignment MUST be by label (prevents positional bugs).
    if "wind_speed" in weibull_pdf.dims:
        ws = np.asarray(weibull_pdf.coords["wind_speed"].values)

        # Require exact same labels (no tolerance), but allow different order.
        if ws.shape != u.shape or not np.array_equal(np.sort(ws), np.sort(u)):
            raise ValueError(
                "wind_speed grid mismatch: weibull_pdf.coords['wind_speed'] must contain exactly the same "
                "labels as u_power_curve (order can differ). "
                f"Got coords={ws!r} vs u_power_curve={u!r}."
            )

        # Reorder by label to match u_power_curve order (contract: label-based)
        weibull_pdf_aligned = weibull_pdf.sel(wind_speed=u)
        pdf_vals = weibull_pdf_aligned.transpose("wind_speed", ...).values
    else:
        pdf_vals = weibull_pdf.values

    n = len(u)
    for i in range(n - 1):
        du = u[i + 1] - u[i]
        pdf_i = pdf_vals[i]
        pdf_ip1 = pdf_vals[i + 1]
        pc_i = eval_power_curve(u[i], s)
        pc_ip1 = eval_power_curve(u[i + 1], s)
        term_i = pdf_i * pc_i
        term_ip1 = pdf_ip1 * pc_ip1
        acc += 0.5 * (term_i + term_ip1) * du

    # Apply correction factor and build result
    cap_factor = wind_shear.copy()
    cap_factor.values = acc * correction_factor
    cap_factor.name = "capacity_factor"
    return cap_factor


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
