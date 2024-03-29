# %% try to build a class for wind power resource assessment with GWA
import logging
from pathlib import Path
from typing import List
from tempfile import NamedTemporaryFile

import requests
from dask.diagnostics import ProgressBar

import pandas as pd
import netCDF4  # import required for proper writing of netcdf-files
import xarray as xr
import rioxarray as rxr

from dataclasses import dataclass

from cleo.spatial import (
    reproject,
    clip_to_geometry
)
from cleo.utils import (
    download_file,
    _setup_logging,
)
from cleo.loaders import (
    get_cost_assumptions,
    get_turbine_attribute,
    get_overnight_cost,
    load_powercurves,
    load_weibull_parameters
)
from cleo.assess import (
    compute_air_density_correction,
    compute_mean_wind_speed,
    compute_terrain_roughness_length,
    compute_weibull_pdf,
    simulate_capacity_factors,
    compute_lcoe,
    minimum_lcoe,
)

# from logging_config import setup_logging
# setup_logging()


@dataclass
class WindResourceAtlas:
    """
    Atlas base class for wind power resource assessment with GWA data
    """
    path: Path
    country: str
    wind_turbines: List[str]
    crs: str = None  # Private attribute for CRS, use property for access
    data: xr.Dataset = None  # Optional attribute to hold loaded data
    power_curves: pd.DataFrame = None  # Optional attribute for power curves

    def __init__(self, path: str, country: str, **atlas_params):
        """
        Provides an Atlas object

        :param path: Path to data directory. Subdirectories "raw" and "processed" will be created.
        :type path: str
        :param country: 3-digit ISO code of country to download
        :type country: str
        :param atlas_params: Optional additional parameters
        """
        self._init_params(path, country)
        self._setup_directories()
        self._setup_logging()
        self._load_data()
        self._build_netcdf()
        logging.info(f"WindResourceAtlas for {self.country} initialized at {self.path}")

    def __repr__(self):
        return (
            f"<WindResourceAtlas {self.country}>\n"
            f"Wind turbine(s): {self.wind_turbines}\n"
            f"{self.data.data_vars}"
        )

    def _init_params(self, path, country):
        """
        Initialize the path and country properties
        """
        if isinstance(path, str):
            self.path = Path(path)
        elif isinstance(path, Path):
            self.path = path
        else:
            raise TypeError("Path must be a string or pathlib.Path object")

        if isinstance(country, str):
            self.country = country
        else:
            raise TypeError("Country must be a string")

        self.wind_turbines = ["Vestas.V112.3075"]

    def _setup_directories(self) -> None:
        """
        Create directories for raw and processed data if they do not exist
        """
        path_raw = self.path / "data" / "raw" / self.country
        path_processed = self.path / "data" / "processed"
        path_logging = self.path / "logs"

        for path in [path_raw, path_processed, path_logging]:
            if not path.is_dir():
                path.mkdir()

    def _load_data(self) -> None:
        """
        Download wind resource data for the specified country from GWA API
        Downloads air density, combined Weibull parameters, and ground elevation data for multiple heights
        """
        url = "https://globalwindatlas.info/api/gis/country"
        layers = ['air-density', 'combined-Weibull-A', 'combined-Weibull-k']
        ground = ['elevation_w_bathymetry']
        height = ['50', '100', '150', '200']

        c = self.country
        path_raw = self.path / "data" / "raw" / self.country

        logging.info(f"Initializing WindResourceAtlas with Global Wind Atlas data")

        for l in layers:
            for h in height:
                fname = f'{c}_{l}_{h}.tif'
                fpath = path_raw / fname

                if not fpath.is_file():
                    try:
                        if not fpath.is_file():
                            durl = f"{url}/{c}/{l}/{h}"
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

    def _build_netcdf(self) -> None:
        """
        Build a NetCDF file from the downloaded data or open an existing one.
        The NetCDF file stores the wind resource data in a structured format.
        """
        path_raw = self.path / "data" / "raw" / self.country
        path_netcdf = self.path / "data" / "processed"
        fname_netcdf = path_netcdf / f"atlas_{self.country}.nc"

        if not fname_netcdf.is_file():
            logging.info(f"Building new resource atlas at {str(path_netcdf)}")
            # get coords from GWA
            with rxr.open_rasterio(path_raw / f"{self.country}_combined-Weibull-A_100.tif",
                                   parse_coordinates=True).squeeze() as weibull_a_100:
                self.data = xr.Dataset(coords=weibull_a_100.coords)
                self.data = self.data.rio.write_crs(weibull_a_100.rio.crs)
                self.data.to_netcdf(path_raw / fname_netcdf)
        else:
            with xr.open_dataset(fname_netcdf, chunks="auto") as dataset:
                self.data = dataset
            logging.info(f"Existing WindResourceAtlas at {str(path_netcdf)} opened.")

        self.crs = self.data.rio.crs

    def to_file(self, complevel=4):
        """
        Save NetCDF data safely to a file
        """
        with NamedTemporaryFile(suffix=".tmp", dir=self.path / "data" / "processed", delete=False) as tmp_file:
            tmp_file_path = Path(tmp_file.name)

            logging.debug(f"Writing data to {tmp_file_path} ...")
            encoding = {}
            total_size = self.data.nbytes

            for var_name, var in self.data.variables.items():
                encoding[var_name] = {"zlib": True, "complevel": complevel}

            write_job = self.data.to_netcdf(str(tmp_file_path),
                                                  compute=False,
                                                  format="NETCDF4",
                                                  engine="netcdf4",
                                                  encoding=encoding,
                                                  )
            with ProgressBar():
                write_job.compute()

            if (self.path / "data" / "processed" / f"atlas_{self.country}.nc").exists():
                self.data.close()
                (self.path / "data" / "processed" / f"atlas_{self.country}.nc").unlink()

            tmp_file_path.rename(self.path / "data" / "processed" / f"atlas_{self.country}.nc")
        logging.info(f"WindResourceAtlas data saved to {str(self.path / 'data' / 'processed' / f'atlas_{self.country}.nc')}")

    def clip_to(self, clip_shape, inplace=True):
        """
        Wrapper function to enable inplace-operations for clip_to_geometry-function
        :param clip_shape: path-string or Geopandas GeoDataFrame containing the clipping shape
        :param inplace: boolean flag indicating whether clipped data should be updated inplace. Default is True
        :return:
        """
        data_clipped = clip_to_geometry(self.data, clip_shape)
        # Update the data in the Atlas object based on the inplace argument
        if inplace:
            self.data = data_clipped
        else:
            clipped_atlas = WindResourceAtlas(str(self.path), self.country)
            clipped_atlas.data = data_clipped
            return clipped_atlas

    # utils
    _setup_logging = _setup_logging
    # data handling: get from yaml, load from GWA-tiffs
    get_cost_assumptions = get_cost_assumptions
    get_turbine_attribute = get_turbine_attribute
    get_powercurves = load_powercurves
    get_overnight_cost = get_overnight_cost
    load_weibull_parameters = load_weibull_parameters

    # spatial operations
    reproject = reproject

    # methods for resource assessment
    compute_air_density_correction = compute_air_density_correction
    compute_mean_wind_speed = compute_mean_wind_speed
    compute_terrain_roughness_length = compute_terrain_roughness_length
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
        compute_terrain_roughness_length(self)
        compute_air_density_correction(self)
        load_powercurves(self)
        compute_weibull_pdf(self)
        simulate_capacity_factors(self)
        compute_lcoe(self)
        minimum_lcoe(self)
