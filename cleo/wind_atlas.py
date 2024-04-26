# %% try to build a class for wind power resource assessment with GWA
import shutil
import logging
import zipfile
from pathlib import Path
from typing import List
from tempfile import NamedTemporaryFile

import requests
from dask.diagnostics import ProgressBar

import pandas as pd
import geopandas as gpd
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
    compute_optimal_power_energy,
)

# from logging_config import setup_logging
# setup_logging()
# TODO: write properties to attributes of the netcdf file
# TODO: use load_dataset instead of open_dataset?
# TODO: unify WindScape and GeoScape classes and save two netcdf files, one for each thing. Second netcdf can be initialized from first

@dataclass
class WindScape:
    """
    WindScape base class for wind power resource assessment with GWA data
    """
    path: Path
    country: str
    wind_turbines: List[str]
    region: str = None
    crs: str = None  # Private attribute for CRS, use property for access
    data: xr.Dataset = None  # Optional attribute to hold loaded data
    power_curves: pd.DataFrame = None  # Optional attribute for power curves

    def __init__(self, path: str, country: str, *wind_scape_params):
        """
        Provides a WindScape object

        :param path: Path to data directory. Subdirectories "raw" and "processed" will be created.
        :type path: str
        :param country: 3-digit ISO code of country to download
        :type country: str
        :param wind_scape_params: Optional additional parameters
        """
        self._init_params(path, country)
        self._setup_directories()
        self._setup_logging()
        self._load_data()
        self._load_nuts()
        self._build_netcdf()
        self._load_clip_shape()
        self._copy_resource_files()
        logging.info(f"WindScape for {self.country} initialized at {self.path}")

    def __repr__(self):
        return (
            f"<WindScape {self.country}>\n"
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
        self.clip_shape = None

    def _setup_directories(self) -> None:
        """
        Create directories for raw and processed data if they do not exist
        """
        path_raw = self.path / "data" / "raw" / self.country
        path_processed = self.path / "data" / "processed"
        path_logging = self.path / "logs"

        for path in [path_raw, path_processed, path_logging]:
            if not path.is_dir():
                path.mkdir(parents=True)

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

        logging.info(f"Initializing WindScape with Global Wind Atlas data")

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

    def _load_nuts(self, resolution="03M", year=2021, crs=4326):

        RESOLUTION = ["01M", "03M", "10M", "20M", "60M"]
        YEAR = [2021, 2016, 2013, 2010, 2006, 2003]
        CRS = [3035, 4326, 3857]

        if resolution not in RESOLUTION:
            raise ValueError(f'Invalid resolution: {resolution}')

        if year not in YEAR:
            raise ValueError(f'Invalid year: {year}')

        if crs not in CRS:
            raise ValueError(f'Invalid crs: {crs}')

        nuts_path = self.path / "data" / "nuts"

        if not nuts_path.is_dir():
            nuts_path.mkdir()

        url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/"
        file = f"NUTS_RG_{resolution}_{year}_{crs}.shp.zip"
        if not (nuts_path / file).is_file():
            download_file(url + file, nuts_path / file)
            logging.info(f"Downloaded {file}")

            with zipfile.ZipFile(str(nuts_path / file), "r") as zip_ref:
                zip_ref.extractall(nuts_path)

            logging.info(f"Extracted {file}")
        else:
            logging.info(f"NUTS borders initialised.")

    def _build_netcdf(self) -> None:
        """
        Build a NetCDF file from the downloaded data or open an existing one.
        The NetCDF file stores the wind resource data in a structured format.
        """
        path_raw = self.path / "data" / "raw" / self.country
        path_netcdf = self.path / "data" / "processed"
        fname_netcdf = path_netcdf / f"windscape_{self.country}.nc"

        if not fname_netcdf.is_file():
            logging.info(f"Building new WindScape object at {str(path_netcdf)}")
            # get coords from GWA
            with rxr.open_rasterio(path_raw / f"{self.country}_combined-Weibull-A_100.tif",
                                   parse_coordinates=True).squeeze() as weibull_a_100:
                self.data = xr.Dataset(coords=weibull_a_100.coords)
                self.data = self.data.rio.write_crs(weibull_a_100.rio.crs)
                self.data.to_netcdf(path_raw / fname_netcdf)
        else:
            with xr.open_dataset(fname_netcdf, chunks="auto") as dataset:
                self.data = dataset
            logging.info(f"Existing WindScape at {str(path_netcdf)} opened.")

        self.crs = self.data.rio.crs

    def _load_clip_shape(self) -> None:
        """
        Build the clip shape to which the WindScape was clipped
        """
        fname_clipshape = self.path / "data" / "processed" / f"clip_shape_{self.country}.shp"
        if fname_clipshape.is_file():
            self.clip_shape = gpd.read_file(fname_clipshape)
            logging.info(f"Existing clip shape {fname_clipshape} loaded.")

    def _copy_resource_files(self):
        """
        Copy yaml-resource files to the destination directory
        """
        # Path to the directory containing YAML files within the package
        source_dir = Path(__file__).parent.parent / 'resources'
        # create destination directory
        (self.path / "resources").mkdir(parents=True, exist_ok=True)
        # Iterate over each YAML file in the source folder
        for file_path in source_dir.glob('*.yml'):
            # Copy the YAML file to the destination folder
            shutil.copy(file_path, self.path / "resources")
        logging.info(f"Resource files copied to {self.path / 'resources'}.")

    def to_file(self, complevel=4):
        """
        Save NetCDF data safely to a file
        """
        with NamedTemporaryFile(suffix=".tmp", dir=self.path / "data" / "processed", delete=False) as tmp_file:
            tmp_file_path = Path(tmp_file.name)

            logging.debug(f"Writing data to {tmp_file_path} ...")
            encoding = {}

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

            if (self.path / "data" / "processed" / f"windscape_{self.country}.nc").exists():
                self.data.close()
                (self.path / "data" / "processed" / f"windscape_{self.country}.nc").unlink()

            tmp_file.close()

        tmp_file_path.rename(self.path / "data" / "processed" / f"windscape_{self.country}.nc")

        logging.info(
            f"WindScape data saved to {str(self.path / 'data' / 'processed' / f'windscape_{self.country}.nc')}")

        fname_clip_shape = self.path / "data" / "processed" / f"clip_shape_{self.country}.shp"
        if isinstance(self.clip_shape, gpd.GeoDataFrame) and not fname_clip_shape.is_file():
            self.clip_shape.to_file(fname_clip_shape)
            logging.info(f"Clip shape saved to '{fname_clip_shape}'.")

    def clip_to(self, clip_shape, region=None, inplace=True):
        """
        Wrapper function to enable inplace-operations for clip_to_geometry-function
        :param clip_shape: path-string or Geopandas GeoDataFrame containing the clipping shape
        :param inplace: boolean flag indicating whether clipped data should be updated inplace. Default is True
        :return:
        """
        # check whether WindScape is clipped already
        if self.clip_shape is None:
            data_clipped, clip_shape_used = clip_to_geometry(self, clip_shape)
            # add clip shape to WindScape
            self.clip_shape = clip_shape_used

            # Update the data in the WindScape object based on the inplace argument
            if inplace:
                self.data = data_clipped
            else:
                clipped_scape = WindScape(str(self.path), self.country)
                clipped_scape.data = data_clipped
                return clipped_scape
        elif self.clip_shape.equals(clip_shape):
            logging.info("WindScape already clipped to clip shape.")
        else:
            logging.warning("WindScape already clipped to another clip shape. Operation aborted.")

    # utils
    _setup_logging = _setup_logging
    # data handling: get from yaml, load from GWA-tiffs
    get_cost_assumptions = get_cost_assumptions
    get_turbine_attribute = get_turbine_attribute
    get_powercurves = load_powercurves
    get_overnight_cost = get_overnight_cost
    load_weibull_parameters = load_weibull_parameters
    compute_optimal_power_energy = compute_optimal_power_energy

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
