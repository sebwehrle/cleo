# helpers for the Atlas class
# %% imports
import yaml
import zipfile
import requests
import pandas as pd
import rioxarray as rxr
import logging
from cleo.assess import turbine_overnight_cost
from cleo.utils import download_file


# %% methods
def get_cost_assumptions(self, attribute_name):
    """
    Retrieve cost assumptions from a yaml-file in ./resources

    :param self: an instance of the Atlas class
    :param attribute_name: Name of the cost assumption attribute to retrieve
    :type attribute_name: str
    :return: Value of the specific cost assumption
    """
    with open(str(self.parent.path / "resources/cost_assumptions.yml")) as f:
        data = yaml.safe_load(f)
    return data[attribute_name]


def get_overnight_cost(self, turbine_model):
    power = self.get_turbine_attribute(turbine_model, "capacity") / 1000
    hub_height = self.get_turbine_attribute(turbine_model, "hub_height")
    rotor_diameter = self.get_turbine_attribute(turbine_model, "rotor_diameter")
    year = self.get_turbine_attribute(turbine_model, "commissioning_year")
    return turbine_overnight_cost(power=power, hub_height=hub_height, rotor_diameter=rotor_diameter, year=year)


def get_powercurves(self):
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


def get_turbine_attribute(self, turbine, attribute_name):
    """
    Retrieve turbine attribute from a yaml-file in ./resources

    :param turbine: Name of the wind turbine in the format "Manufacturer.Type.Power_in_kW"
    :type turbine: str
    :param attribute_name: Name of the turbine attribute to retrieve
    :type attribute_name: str
    :return: Value of the specific turbine attribute
    """
    with open(str(self.parent.path / "resources" / turbine) + ".yml") as f:
        data = yaml.safe_load(f)
    return data[attribute_name]


def load_weibull_parameters(self, height):
    """
    Load weibull parameters for a specific height
    :param self: an  instance of the Atlas class
    :param height: Height for which to load Weibull parameters. Possible values are [50, 100, 150].
    GWA also provides 10 and 200 m data, which, however, is not loaded by Atlas class currently.
    :type height: int
    :return: Tuple containing Weibull parameter rasters (a, k)
    :rtype: Tuple[xarray.DataArray, xarray.DataArray]
    """
    path_raw_country = self.parent.path / "data" / "raw" / f"{self.parent.country}"
    try:
        a = rxr.open_rasterio(path_raw_country / f"{self.parent.country}_combined-Weibull-A_{height}.tif").chunk("auto")
        k = rxr.open_rasterio(path_raw_country / f"{self.parent.country}_combined-Weibull-k_{height}.tif").chunk("auto")
        a.name = "weibull_a"
        k.name = "weibull_k"

        if a.rio.crs != self.parent.crs:
            a = a.rio.reproject(self.parent.crs)

        if k.rio.crs != self.parent.crs:
            k = k.rio.reproject(self.parent.crs)

        if self.parent.region is not None:
            clip_shape = self.parent.get_nuts_region(self.parent.region)
            if clip_shape.crs != self.parent.crs:
                clip_shape = clip_shape.to_crs(self.parent.crs)
            a = a.rio.clip(clip_shape.geometry)
            k = k.rio.clip(clip_shape.geometry)

        return a, k
    except Exception as e:
        logging.error(f"Error loading weibull parameters for height {height}: {e}")
        return None, None


# %% methods
# def get_nuts_borders(self):
#     alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
#     border = gpd.read_file(self.path / "data" / "nuts" / "NUTS_RG_03M_2021_4326.shp")
#     if any(border["CNTR_CODE"].str.contains(alpha_2)):
#         # border = border.loc[(border["CNTR_CODE"].str.contains(alpha_2)) & (border["LEVL_CODE"] == 0)]
#         border = border.loc[border["CNTR_CODE"].str.contains(alpha_2)]
#     else:
#         raise ValueError(f"'{alpha_2}' is not a valid NUTS country code")
#
#     return border


def load_gwa(self):
    """
    Download wind resource data for the specified country from GWA API
    Downloads air density, combined Weibull parameters, and ground elevation data for multiple heights
    """
    url = "https://globalwindatlas.info/api/gis/country"
    layers = ['air-density', 'combined-Weibull-A', 'combined-Weibull-k']
    ground = ['elevation_w_bathymetry']
    height = ['50', '100', '150', '200']

    c = self.parent.country
    path_raw = self.parent.path / "data" / "raw" / self.parent.country
    logging.info(f"Initializing WindScape with Global Wind Atlas data")

    for ly in layers:
        for h in height:
            fname = f'{c}_{ly}_{h}.tif'
            fpath = path_raw / fname

            if not fpath.is_file():
                try:
                    if not fpath.is_file():
                        durl = f"{url}/{c}/{ly}/{h}"
                        download_file(durl, fpath)
                        logging.info(f'Download of {fname} from {durl} complete')
                except requests.RequestException as e:
                    logging.error(f'Error downloading {fname}: {e}')

    for g in ground:
        fname = f'{c}_{g}.tif'
        fpath = path_raw / fname
        if not fpath.is_file():
            try:
                if not fpath.is_file():
                    durl = f'{url}/{c}/{g}'
                    download_file(durl, fpath)
                    logging.info(f'Download of {fname} from {durl} complete')
            except requests.RequestException as e:
                logging.error(f'Error downloading {fname}: {e}')

    logging.info(f'Global Wind Atlas data for {c} initialized.')


def load_nuts(self, resolution="03M", year=2021, crs=4326):
    RESOLUTION = ["01M", "03M", "10M", "20M", "60M"]
    YEAR = [2021, 2016, 2013, 2010, 2006, 2003]
    CRS = [3035, 4326, 3857]

    if resolution not in RESOLUTION:
        raise ValueError(f'Invalid resolution: {resolution}')

    if year not in YEAR:
        raise ValueError(f'Invalid year: {year}')

    if crs not in CRS:
        raise ValueError(f'Invalid crs: {crs}')

    nuts_path = self.parent.path / "data" / "nuts"

    if not nuts_path.is_dir():
        nuts_path.mkdir()

    url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/"
    file_collection = f"ref-nuts-{year}-{resolution}.shp.zip"
    file_name = f"NUTS_RG_{resolution}_{year}_{crs}.shp.zip"

    if not (nuts_path / file_name).is_file():
        download_file(url + file_collection, nuts_path / file_collection)
        logging.info(f"Downloaded {file_collection}")

        with zipfile.ZipFile(str(nuts_path / file_collection), "r") as zip_ref:
            if file_name in zip_ref.namelist():
                zip_ref.extract(file_name, nuts_path)

                with zipfile.ZipFile(str(nuts_path / file_name), "r") as zip_inner:
                    zip_inner.extractall(nuts_path)

            else:
                raise FileNotFoundError(f"File {file_name}")

        logging.info(f"Extracted {file_name}")
    else:
        logging.info(f"NUTS borders initialised.")
