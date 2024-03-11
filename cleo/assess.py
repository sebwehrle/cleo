# resource assessment methods of the Atlas class
import numpy as np
import xarray as xr
import rioxarray as rxr
import logging

from scipy.special import gamma
from cleo.utils import weibull_probability_density, capacity_factor, turbine_overnight_cost, grid_connect_cost, \
    levelized_cost


def compute_air_density_correction(self):
    """
    Compute air density correction factor rho based on elevation data.
    See https://wind-data.ch/tools/luftdichte.php for methodology details
    """
    path_raw_coutry = self._path / "data" / "raw" / f"{self.country}"
    elevation = rxr.open_rasterio(path_raw_coutry / f"{self.country}_elevation_w_bathymetry.tif")
    rho_correction = 1.247015 * np.exp(-0.000104 * elevation) / 1.225
    self.data["air_density_correction"] = rho_correction.squeeze().rename("air_density_correction_factor")
    logging.info(f'Air density correction computed for {self.country}')


def compute_mean_wind_speed(self, height):
    """
    Compute mean wind speed for a given height

    :param height: Height for which to compute mean wind speed
    :type height: int
    """
    a, k = self.load_weibull_parameters(height)
    mean_wind_speed = a * gamma(1 / k + 1)
    mean_wind_speed = mean_wind_speed.rename("mean_wind_speed").assign_attrs(units="m/s")
    mean_wind_speed = mean_wind_speed.squeeze().expand_dims(height=[height])
    # add to xarray.Dataset
    if "mean_wind_speed" in self.data:
        mws = xr.concat([self.data["mean_wind_speed"], mean_wind_speed], dim="height")
        self.data = self.data.drop_vars(["mean_wind_speed", "height"])
        self.data["mean_wind_speed"] = mws
    else:
        self.data["mean_wind_speed"] = mean_wind_speed

    logging.info(f"Computation of mean wind speed at height {height} completed")


def compute_wind_shear(self):
    """
    Compute wind shear factor alpha
    """
    self.compute_mean_wind_speed(50)
    self.compute_mean_wind_speed(100)

    u_mean_50 = self.data.mean_wind_speed.sel(height=50)
    u_mean_100 = self.data.mean_wind_speed.sel(height=100)

    alpha = (np.log(u_mean_100) - np.log(u_mean_50)) / (np.log(100) - np.log(50))
    self.data['wind_shear'] = alpha.squeeze().rename("wind_shear_factor")
    logging.info(f'Computation of wind shear for {self.country} complete')


def compute_weibull_pdf(self):
    """
    Compute weibull probability density function for reference wind speeds
    """
    # load wind speed Weibull parameters
    A100, k100 = self.load_weibull_parameters(100)

    # Reference wind speeds from power curves
    u_pwrcrv = self.power_curves.index.values

    # Calculate Weibull wind speed probability density
    p = weibull_probability_density(u_pwrcrv, k100, A100)
    self.data["weibull_pdf"] = p
    logging.info('Computation of Weibull probability density function complete')


def simulate_capacity_factors(self, bias_correction=1):
    """
    Simulate capacity factors for specified wind turbine models
    :param bias_correction: Bias correction factor for simulated capacity factors
    :type bias_correction: float
    """

    def compute_cf(atlas, turbine_type):
        h_turbine = atlas.get_turbine_attribute(turbine_type, "hub_height")
        p_pwrcrv = atlas.power_curves[turbine_type].values
        cf = capacity_factor(
            atlas.data["weibull_pdf"],
            atlas.data["wind_shear"],
            atlas.power_curves.index.values,
            p_pwrcrv,
            h_turbine
        )
        cf = cf * atlas.data["air_density_correction"]
        cf = cf * bias_correction
        cf = cf.assign_coords({"turbine_models": turbine_type})
        logging.info(f"Capacity factor for {turbine_type} computed")
        return cf

    cf_list = [compute_cf(self, ttype) for ttype in self.wind_turbine]
    cap_fac = xr.concat(cf_list, dim="turbine_models")
    self.data["capacity_factors"] = cap_fac
    logging.info(f"Computation of capacity factors for {self.country} complete")


def compute_lcoe(self, turbine_cost_share=1):
    """
    Compute levelized cost of electricity (LCOE) for specified wind turbine models
    :param turbine_cost_share: Share of location-independent investment cost
    :type turbine_cost_share: float
    """

    om_fixed = self.get_cost_assumptions("fixed_om_cost")
    om_variable = self.get_cost_assumptions("variable_om_cost")
    lifetime = self.get_cost_assumptions("turbine_lifetime")
    discount_rate = self.get_cost_assumptions("discount_rate")

    LCOE = []
    for turbine in self.wind_turbine:
        power = self.get_turbine_attribute(turbine, "capacity")
        hub_height = self.get_turbine_attribute(turbine, "hub_height")
        rotor_diameter = self.get_turbine_attribute(turbine, "rotor_diameter")
        year = self.get_turbine_attribute(turbine, "commissioning_year")
        overnight_cost = turbine_overnight_cost(power / 1000, hub_height, rotor_diameter,
                                                year) * power * turbine_cost_share
        grid_cost = grid_connect_cost(power)
        cap_factor = self.data["capacity_factors"].sel(turbine_models=turbine)

        lcoe = levelized_cost(power,
                              cap_factor,
                              overnight_cost,
                              grid_cost,
                              om_fixed,
                              om_variable,
                              discount_rate,
                              lifetime,
                              )

        LCOE.append(lcoe)

    lcoe = xr.concat(LCOE, dim='turbine_models')
    lcoe = lcoe.assign_coords({"turbine_models": self.wind_turbine})
    self.data["lcoe"] = lcoe.rename("lcoe").assign_attrs(units="EUR/MWh")


def minimum_lcoe(self):
    """
    Calculate the minimum lcoe for each location among all turbines
    """
    lcoe = self.data["lcoe"]
    lc_min = lcoe.min(dim='turbine_models', keep_attrs=True)
    lc_min = lc_min.assign_coords({'turbine_models': 'min_lcoe'})
    self.data["min_lcoe"] = lc_min.rename("minimal_lcoe")
