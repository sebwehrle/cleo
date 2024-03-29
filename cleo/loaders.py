# helpers for the Atlas class
# %% imports
import yaml
import pandas as pd
import rioxarray as rxr
import logging
from cleo.assess import turbine_overnight_cost


def get_cost_assumptions(self, attribute_name):
    """
    Retrieve cost assumptions from a yaml-file in ./resources

    :param self: an instance of the Atlas class
    :param attribute_name: Name of the cost assumption attribute to retrieve
    :type attribute_name: str
    :return: Value of the specific cost assumption
    """
    with open(str(self.path / "resources/cost_assumptions.yml")) as f:
        data = yaml.safe_load(f)
    return data[attribute_name]


def get_turbine_attribute(self, turbine, attribute_name):
    """
    Retrieve turbine attribute from a yaml-file in ./resources

    :param turbine: Name of the wind turbine in the format "Manufacturer.Type.Power_in_kW"
    :type turbine: str
    :param attribute_name: Name of the turbine attribute to retrieve
    :type attribute_name: str
    :return: Value of the specific turbine attribute
    """
    with open(str(self.path / "resources" / turbine) + ".yml") as f:
        data = yaml.safe_load(f)
    return data[attribute_name]


def load_powercurves(self):
    """
    Load power curves from yaml-file in ./resources
    Loads a power curve for each wind turbine in self.wind_turbine
    """
    file_paths = [str(self.path / "resources" / turbine) + ".yml" for turbine in self.wind_turbines]

    power_curves = [
        pd.DataFrame(
            data=data["cf"],
            index=data["V"],
            columns=[f"{data['manufacturer']}.{data['model']}.{data['capacity']}"])
        for data in (yaml.safe_load(open(path, "r")) for path in file_paths)]

    self.power_curves = pd.concat(power_curves, axis=1)
    logging.info(f"Power curves for {self.wind_turbines} loaded.")


def load_weibull_parameters(self, height):
    """
    Load weibull parameters for a specific height
    :param self: an  instance of the Atlas class
    :param height: Height for which to load Weibull parameters. Possible values are [50, 100, 150].
    GWA also provides 10 and 200 m data, which, however, is not loaded by Atlas class currently.
    : type height: int
    :return: Tuple containing Weibull parameter rasters (a, k)
    :rtype: Tuple[xarray.DataArray, xarray.DataArray]
    """
    path_raw_country = self.path / "data" / "raw" / f"{self.country}"
    try:
        a = rxr.open_rasterio(path_raw_country / f"{self.country}_combined-Weibull-A_{height}.tif").chunk("auto")
        k = rxr.open_rasterio(path_raw_country / f"{self.country}_combined-Weibull-k_{height}.tif").chunk("auto")
        a.name = "weibull_a"
        k.name = "weibull_k"
        return a, k
    except Exception as e:
        logging.error(f"Error loading weibull parameters for height {height}: {e}")
        return None, None


def get_overnight_cost(self, turbine_model):
    power = self.get_turbine_attribute(turbine_model, "capacity") / 1000
    hub_height = self.get_turbine_attribute(turbine_model, "hub_height")
    rotor_diameter = self.get_turbine_attribute(turbine_model, "rotor_diameter")
    year = self.get_turbine_attribute(turbine_model, "commissioning_year")
    return turbine_overnight_cost(power=power, hub_height=hub_height, rotor_diameter=rotor_diameter, year=year)
