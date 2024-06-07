# %% imports
import logging
import shutil
import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path


# %% methods
def build_netcdf(self, atlas_type):
    """
    Build a NetCDF file from the downloaded data or open an existing one.
    The NetCDF file stores the wind resource data in a structured format.
    """
    path_raw = self.parent.path / "data" / "raw" / self.parent.country
    path_netcdf = self.parent.path / "data" / "processed"
    if self.parent.region is not None:
        fname_netcdf = path_netcdf / f"{atlas_type}_{self.parent.country}_{self.parent.region}.nc"
    else:
        fname_netcdf = path_netcdf / f"{atlas_type}_{self.parent.country}.nc"

    if not fname_netcdf.is_file():
        logging.info(f"Building new {atlas_type} object at {str(path_netcdf)}")
        # get coords from GWA
        with rxr.open_rasterio(path_raw / f"{self.parent.country}_combined-Weibull-A_100.tif",
                               parse_coordinates=True).squeeze() as weibull_a_100:
            self.data = xr.Dataset(coords=weibull_a_100.coords)
            self.data = self.data.rio.write_crs(weibull_a_100.rio.crs)

            if atlas_type == "LandscapeAtlas":
                nan_mask = np.isnan(weibull_a_100)
                self.data["template"] = xr.where(nan_mask, np.nan, 0)

            self.data.to_netcdf(path_raw / fname_netcdf)

    else:
        with xr.open_dataset(fname_netcdf) as dataset:
            self.data = dataset
        logging.info(f"Existing {atlas_type} at {str(path_netcdf)} opened.")

    if self.data.rio.crs != self.parent.crs:
        self.data = self.data.rio.reproject(self.parent.crs)

    if self.data.rio.crs is None:
        self.data = self.data.rio.write_crs(self.parent.crs)
        logging.warning(f"Coordinate reference system of {self} set to {self.parent.crs}")


def deploy_resources(self):
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


def set_attributes(self):
    self.data.attrs['country'] = self.parent.country
    self.data.attrs['region'] = self.parent.region
    if self.data.rio.crs is None:
        raise AttributeError(f"{self.data} does not have a coordinate reference system.")
    if self.data.rio.crs != self.parent.crs:
        raise ValueError(f"Coordinate reference system mismatch: {self.parent.crs} and {self.data.rio.crs}")


def setup_logging(self):
    """
    Setup logging in the logs directory
    :param self: an instance of the Atlas class
    """
    fname = self.path / "logs" / "logfile.log"
    colors = {
        'reset': '\33[m',
        'green': '\33[32m',
        'purple': '\33[35m',
        'orange': '\33[33m',
    }

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)-4s - %(name)-4s - %(message)s'
            },
            'color': {
                'format': f"{colors['green']}[%(asctime)s]{colors['reset']} {colors['purple']}%(levelname)-5s{colors['reset']} - {colors['orange']}%(name)-5s{colors['reset']}: %(message)s"
            }
        },
        'handlers': {
            'stream': {
                'class': 'logging.StreamHandler',
                'formatter': 'color',
            }
        },
        'root': {
            'handlers': ['stream'],
            'level': logging.INFO,
        },
    }

    if fname:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'level': logging.DEBUG,
            'filename': fname,
        }
        logging_config['root']['handlers'].append('file')

    logging.config.dictConfig(logging_config)
