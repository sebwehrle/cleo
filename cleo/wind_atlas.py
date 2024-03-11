# %% try to build a class for wind power resource assessment with GWA
import logging
from pathlib import Path
from typing import List

import pandas as pd
import xarray as xr
import rioxarray as rxr

from dataclasses import dataclass
from cleo.spatial import reproject, clip_to_geometry
from cleo.utils import download_file
from cleo.loaders import get_cost_assumptions, get_turbine_attribute, load_powercurves
from cleo.assess import load_weibull_parameters, compute_air_density_correction, compute_mean_wind_speed, \
    compute_wind_shear, compute_weibull_pdf, simulate_capacity_factors, compute_lcoe, minimum_lcoe
import matplotlib.pyplot as plt

# from logging_config import setup_logging
# setup_logging()


@dataclass
class AtlasState:
    path: Path
    country: str
    wind_turbine: List[str]
    _crs: str = None  # Private attribute for CRS, use property for access
    data: xr.Dataset = None  # Optional attribute to hold loaded data
    power_curves: pd.DataFrame = None  # Optional attribute for power curves


class Atlas:
    """
    Atlas base class for windpower resource assessment with GWA data
    """

    def __init__(self, path: str, country: str, **atlas_params):
        """
        Provides an Atlas object

        :param path: Path to data directory. Subdirectories "raw" and "processed" will be created.
        :type path: str
        :param country: 3-digit ISO code of country to download
        :type country: str
        :param atlas_params: Optional additional parameters
        """
        self.state = AtlasState(path, country, atlas_params.get("wind_turbine", ["Vestas.V112.3075"]))
        logging.info(f"Atlas instance created with {self.state.path}")
        self.data = None
        self._path = Path(path)
        self._country = str(country)  # TODO: check if 3-digit ISO code
        self._wind_turbine = ["Vestas.V112.3075"]
        self._crs = None
        self.power_curves = None
        self.download_file = download_file

        logging.info(f"Atlas instance created with path: {self._path}, country: {self._country}")

        self._setup_directories()
        self._load_data()
        self._build_netcdf()

    def __repr__(self):
        return (
            f"<WindAtlas {self.country}>\n"
            f"Wind turbine(s): {self._wind_turbine}\n"
            f"{self.data.data_vars}"
        )


    @property
    def country(self) -> str:
        return self.state.country

    @property
    def path(self) -> Path:
        return self.state.path

    @property
    def wind_turbine(self) -> List[str]:
        return self.state.wind_turbine

    @wind_turbine.setter
    def wind_turbine(self, value: List[str]) -> None:
        if not isinstance(value, list):
            raise TypeError(f"Input must be a list of strings")
        self.state.wind_turbine = value

    @property
    def crs(self) -> str:
        return self.state._crs

    @crs.setter
    def crs(self, crs: str) -> None:
        self.state._crs = crs

    def _setup_directories(self) -> None:
        """
        Create directories for raw and processed data if they do not exist
        """
        path_raw = self._path / "data" / "raw" / self._country
        path_processed = self._path / "data" / "processed"

        if not path_raw.is_dir():
            path_raw.mkdir()

        if not path_processed.is_dir():
            path_processed.mkdir()

    def _load_data(self) -> None:
        """
        Download wind resource data for the specified country from GWA API
        Downloads air density, combined Weibull parameters, and ground elevation data for multiple heights
        """
        url = "https://globalwindatlas.info/api/gis/country"
        layers = ['air-density', 'combined-Weibull-A', 'combined-Weibull-k']
        ground = ['elevation_w_bathymetry']
        height = ['50', '100', '150']

        c = self.country
        path_raw = self._path / "data" / "raw" / self._country

        logging.info(f"Starting data download for country {c} from {url}")

        for l in layers:
            for h in height:
                fname = f'{c}_{l}_{h}.tif'
                fpath = path_raw / fname

                # check if file already exists before downloading
                try:
                    if not fpath.is_file():
                        durl = f"{url}/{c}/{l}/{h}"
                        self.download_file(durl, fpath)
                        logging.info(f'Download of {fname} complete')
                except Exception as e:
                    logging.error(f'Error downloading {fname}: {e}')

        for g in ground:
            fname = f'{c}_{g}.tif'
            fpath = path_raw / fname
            try:
                if not fpath.is_file():
                    self.download_file(f'{url}/{c}/{g}', fpath)
                    logging.info(f'Download of {fname} complete')
            except Exception as e:
                logging.error(f'Error downloading {fname}: {e}')

        logging.info(f'Raw data of resource atlas for {c} loaded.')

    def _build_netcdf(self) -> None:
        """
        Build a NetCDF file from the downloaded data or open an existing one.
        The NetCDF file stores the wind resource data in a structured format.
        """
        path_raw = self._path / "data" / "raw" / self.country
        path_netcdf = self._path / "data" / "processed"
        fname_netcdf = path_netcdf / f"atlas_{self.country}.nc"

        if not fname_netcdf.is_file():
            logging.info(f"Building new resource atlas at {str(path_netcdf)}")
            # get coords from GWA
            weibull_a_100 = rxr.open_rasterio(path_raw / f"{self.country}_combined-Weibull-A_100.tif",
                                              parse_coordinates=True).squeeze()
            self.data = xr.Dataset(coords=weibull_a_100.coords)
            self.data = self.data.rio.write_crs(weibull_a_100.rio.crs)
            self.data.to_netcdf(path_raw / fname_netcdf)
        else:
            self.data = xr.open_dataset(fname_netcdf)
            logging.info(f"Opened pre-existing resource atlas at {str(path_netcdf)}")

        self.crs = self.data.rio.crs

    def to_file(self):
        """
        Save the NetCDF data to a file
        """
        path_netcdf = self._path / "data" / "processed"
        fname_netcdf = path_netcdf / f"atlas_{self.country}.nc"
        self.data.to_netcdf(fname_netcdf)
        logging.info(f"Atlas data saved to {fname_netcdf}")

    def clip_atlas(self, clip_shape, inplace=True):
        """
        Wrapper function to enable inplace-operations for clip_to_geometry-function
        :param clip_shape: path-string or Geopandas GeoDataFrame containing the clipping shape
        :param inplace: boolean flag indicating whether clipped data should be updated inplace. Default is True
        :return:
        """
        data_clipped = clip_to_geometry(self, clip_shape)
        # Update the data in the Atlas object based on the inplace argument
        if inplace:
            self.data = data_clipped
        else:
            clipped_atlas = Atlas(str(self.path), self.country)
            clipped_atlas.data = data_clipped
            return clipped_atlas

    # data handling
    load_weibull_parameters = load_weibull_parameters
    get_cost_assumptions = get_cost_assumptions
    get_turbine_attribute = get_turbine_attribute
    load_powercurves = load_powercurves
    # spatial
    reproject = reproject
    # resource assessment
    compute_air_density_correction = compute_air_density_correction
    compute_mean_wind_speed = compute_mean_wind_speed
    compute_wind_shear = compute_wind_shear
    compute_weibull_pdf = compute_weibull_pdf
    simulate_capacity_factors = simulate_capacity_factors
    compute_lcoe = compute_lcoe
    minimum_lcoe = minimum_lcoe

    def process(self):
        """
        Process wind resource data through all necessary computations
        This method executes all necessary computations to generate the final wind resource assessment.
        :return:
        """
        self.compute_wind_shear()
        self.compute_air_density_correction()
        self.load_powercurves()
        self.compute_weibull_pdf()
        self.simulate_capacity_factors()
        self.compute_lcoe()
        self.minimum_lcoe()
