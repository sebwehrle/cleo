# resource assessment methods of the Atlas class
import numpy as np
import xarray as xr
import rioxarray as rxr
import logging

from scipy.special import gamma
from cleo.utils import compute_chunked


def weibull_probability_density(u_power_curve, weibull_k, weibull_a):
    """
    Calculates probability density at points in u_power_curve given Weibull parameters in k and A
    :param u_power_curve: Power curve wind speeds
    :param weibull_k: k-parameter of the Weibull distribution of wind speed
    :param weibull_a: A-parameter of the Weibull distribution of wind speed
    :return:
    """
    uar = np.asarray(u_power_curve)
    prb = [(weibull_k / weibull_a * (z / weibull_a) ** (weibull_k - 1)) * (np.exp(-(z / weibull_a) ** weibull_k)) for z
           in uar]
    pdf = xr.concat(prb, dim='wind_speed')
    pdf = pdf.assign_coords({'wind_speed': u_power_curve})
    pdf = pdf.squeeze().rename("weibull_probability_density")
    return pdf


def capacity_factor(weibull_pdf, terrain_roughness_length, u_power_curve, p_power_curve, h_turbine, h_reference=100,
                    correction_factor=1):
    """
    calculates wind turbine capacity factors given Weibull probability density pdf, roughness factor alpha, wind turbine
    power curve data in u_power_curve and p_power_curve, turbine height h_turbine and reference height of wind speed
    modelling h_reference
    :param correction_factor:
    :param weibull_pdf: probability density function from weibull_probability_density()
    :param terrain_roughness_length: terrain roughness length
    :param u_power_curve: power curve wind speed
    :param p_power_curve: power curve output
    :param h_turbine: hub height of wind turbine in m
    :param h_reference: reference height at which weibull pdf is computed
    :return:
    """
    power_curve = xr.DataArray(data=p_power_curve, coords={'wind_speed': u_power_curve})
    u_adjusted = xr.DataArray(data=u_power_curve, coords={'wind_speed': u_power_curve}) @ (
            h_turbine / h_reference) ** terrain_roughness_length
    cap_factor_values = np.trapz(weibull_pdf * power_curve, u_adjusted, axis=0)
    cap_factor = terrain_roughness_length.copy()
    cap_factor.values = cap_factor_values * correction_factor
    cap_factor.name = "capacity_factor"
    return cap_factor


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
    dcf_numerator = 1 - (1 + discount_rate) ** (-period)
    dcf_denominator = 1 - (1 + discount_rate) ** (-1)
    dcf = dcf_numerator / dcf_denominator
    return dcf


def turbine_overnight_cost(power, hub_height, rotor_diameter, year):
    """
    calculates wind turbine investment cost in EUR per MW based on >>Rinne et al. (2018): Effects of turbine technology
    and land use on wind power resource potential, Nature Energy<<
    :param power: rated power in MW
    :param hub_height: hub height in meters
    :param rotor_diameter: rotor diameter in meters
    :return: overnight investment cost in EUR per kW
    """
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    spec_power = power * 10 ** 6 / rotor_area
    cost = ((620 * np.log(hub_height)) - (1.68 * spec_power) + (182 * (2016 - year) ** 0.5) - 1005)
    return cost.astype('float')


def grid_connect_cost(power):
    """
    Calculates grid connection cost according to §54 (3,4) ElWOG https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=20007045
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
    :param overnight_cost: in EUR/MW
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


def compute_air_density_correction(self, chunk_size=None):
    """
    Compute air density correction factor rho based on elevation data.
    See https://wind-data.ch/tools/luftdichte.php for methodology details

    :param chunk_size: size of chunks in pixels for chunked computation. If None, computation is not chunked
    """

    def rho_correction(elevation):
        """
        Compute air density correction factor based on elevation
        :param elevation: Elevation in m above sea level
        :return: Air density correction factor rho
        """
        rho_correction_factor = 1.247015 * np.exp(-0.000104 * elevation) / 1.225
        rho_correction_factor = rho_correction_factor.squeeze().rename("air_density_correction_factor")
        return rho_correction_factor

    path_raw_coutry = self.path / "data" / "raw" / f"{self.country}"
    elevation = rxr.open_rasterio(path_raw_coutry / f"{self.country}_elevation_w_bathymetry.tif").chunk("auto")
    elevation = elevation.rename("elevation").squeeze()

    if elevation.rio.crs != self.crs:
        elevation = elevation.rio.reproject(self.crs)

    if self.clip_shape is not None:
        elevation = elevation.rio.clip(self.clip_shape.geometry)

    if chunk_size is None:
        rho = rho_correction(elevation)
    else:
        rho = compute_chunked(self, rho_correction, chunk_size, elevation=elevation)

    self.data["air_density_correction"] = rho
    logging.info(f'Air density correction for {self.country} computed.')


def compute_mean_wind_speed(self, height, chunk_size=None, inplace=True):
    """
    Compute mean wind speed for a given height

    :param self: An instance of the Atlas-class
    :param height: Height for which to compute mean wind speed
    :type height: int
    :param chunk_size: number of chunks for chunked computation. If None, computation is not chunked.
    :type chunk_size: int
    :param replace: if True, replace existing wind speed data. Defaults to False.
    :type replace: bool
    """

    def calculate_mean_wind_speed(weibull_a, weibull_k):
        """
        Compute mean wind speed from given weibull distribution of wind speeds

        :param weibull_a: Weibull parameter A
        :type weibull_a: xr.DataArray
        :param weibull_k: Weibull parameter k
        :type weibull_k: xr.DataArray
        :return: mean wind speed
        :rtype: xr.DataArray
        """
        mean_wind_speed = weibull_a * gamma(1 / weibull_k + 1)
        return mean_wind_speed.rename("mean_wind_speed").assign_attrs(units="m/s").squeeze()

    inputs = dict(zip(["weibull_a", "weibull_k"], self.load_weibull_parameters(height)))

    mean_wind_speed_data = (
        calculate_mean_wind_speed(**inputs)
        if chunk_size is None
        else compute_chunked(self, calculate_mean_wind_speed, chunk_size, **inputs)
    )

    mean_wind_speed_data = mean_wind_speed_data.expand_dims(height=[height])

    if "mean_wind_speed" not in self.data:
        self.data["mean_wind_speed"] = mean_wind_speed_data
    elif height not in self.data["mean_wind_speed"].coords["height"].values:
        mean_wind_speed_concatenated = xr.concat([self.data["mean_wind_speed"], mean_wind_speed_data], dim="height")
        self.data = self.data.drop_vars(["mean_wind_speed", "height"])
        self.data["mean_wind_speed"] = mean_wind_speed_concatenated
    elif height in self.data["mean_wind_speed"].coords["height"].values and inplace:
        mean_wind_speed_concatenated = xr.concat([self.data["mean_wind_speed"], mean_wind_speed_data], dim="height")
        self.data = self.data.drop_vars(["mean_wind_speed", "height"])
        self.data["mean_wind_speed"] = mean_wind_speed_concatenated
    else:
        logging.warning(f"Mean wind speed at height '{height} m' already exists")
        return mean_wind_speed_data


def compute_terrain_roughness_length(self, chunk_size=None):
    """
    Compute terrain roughness length

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

    alpha = (np.log(u_mean_100) - np.log(u_mean_50)) / (np.log(100) - np.log(50))
    self.data['terrain_roughness_length'] = alpha.squeeze().rename("terrain_roughness_length")
    logging.info(f'Terrain roughness length for {self.country} computed.')


def compute_weibull_pdf(self, chunk_size=None):
    """
    Compute weibull probability density function for reference wind speeds

    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :type chunk_size: int
    """
    inputs = {
        "weibull_a": self.load_weibull_parameters(100)[0],
        "weibull_k": self.load_weibull_parameters(100)[1],
        "u_power_curve": self.power_curves.index.values
    }

    if chunk_size is None:
        p = weibull_probability_density(**inputs)
    else:
        p = compute_chunked(self, weibull_probability_density, chunk_size, **inputs)

    self.data["weibull_pdf"] = p
    logging.info(f'Weibull probability density function of wind speeds in {self.country} computed.')


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

    for turbine_model in self.wind_turbines:
        inputs = {
            "weibull_pdf": self.data["weibull_pdf"],
            "terrain_roughness_length": self.data["terrain_roughness_length"],
            "u_power_curve": self.power_curves.index.values,
            "p_power_curve": self.power_curves[turbine_model].values,
            "h_turbine": self.get_turbine_attribute(turbine_model, "hub_height"),
            "correction_factor": bias_correction
        }

        if chunk_size is None:
            cf = capacity_factor(**inputs)
        else:
            cf = compute_chunked(self, capacity_factor, chunk_size, **inputs)

        # cf = cf * self.data["air_density_correction"]
        cf = cf.expand_dims(turbine_models=[turbine_model])
        logging.info(f"Capacity factors for {turbine_model} in {self.country} computed.")
        # cap_factor = xr.merge([cap_factor, cf.rename("capacity_factor")])
        cap_factor.append(cf)

    cap_factor = xr.concat(cap_factor, dim="turbine_models")
    cap_factor = cap_factor.assign_coords(turbine_models=self.wind_turbines)
    cap_factor = cap_factor.rename("capacity_factor")

    if "capacity_factors" not in self.data:
        self.data["capacity_factors"] = cap_factor
    else:
        self.data = self.data.combine_first(cap_factor)

    return cap_factor


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
    for turbine_model in self.wind_turbines:

        inputs = {
            "power": self.get_turbine_attribute(turbine_model, "capacity"),
            "capacity_factors": self.data["capacity_factors"].sel(turbine_models=turbine_model),
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

        lcoe = lcoe.expand_dims(turbine_models=[turbine_model])
        lcoe_list.append(lcoe)

    self.data["lcoe"] = xr.concat(lcoe_list, dim='turbine_models').rename("lcoe").assign_attrs(units="EUR/MWh")
    logging.info(f"Levelized Cost of Electricity in {self.country} computed.")


def minimum_lcoe(self):
    """
    Calculate the minimum lcoe for each location among all turbines
    """
    lcoe = self.data["lcoe"]
    lc_min = lcoe.min(dim='turbine_models', keep_attrs=True)
    lc_min = lc_min.assign_coords({'turbine_models': 'min_lcoe'})
    self.data["min_lcoe"] = lc_min.rename("minimal_lcoe")
