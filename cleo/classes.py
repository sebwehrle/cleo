# %% imports
import os
import re
import yaml
import pyproj
import logging
import zipfile
import datetime
import rasterio.crs
import numpy as np
import xarray as xr
import geopandas as gpd
import pycountry as pct
from pathlib import Path
from xrspatial import proximity

from cleo.class_helpers import (
    build_netcdf,
    deploy_resources,
    set_attributes,
    setup_logging,
)

from cleo.utils import (
    add,
    flatten,
    convert,
    download_file,
)

from cleo.loaders import (
    get_cost_assumptions,
    get_overnight_cost,
    get_turbine_attribute,
    get_clc_codes,
    load_weibull_parameters,
    load_gwa,
    load_nuts,
    add_corine_land_cover,
)

from cleo.spatial import (
    clip_to_geometry, reproject,
)

from cleo.assess import (
    compute_air_density_correction,
    compute_lcoe,
    compute_mean_wind_speed,
    compute_optimal_power_energy,
    compute_wind_shear_coefficient,
    compute_weibull_pdf,
    minimum_lcoe,
    simulate_capacity_factors,
)


# %% classes
class Atlas:
    def __init__(self, path, country, crs):
        self.path = path
        self.country = country
        self.region = None
        self.crs = crs
        self._wind_turbines = []
        self._setup_directories()
        self._setup_logging()
        self._deploy_resources()
        self.index_file = self.path / "data" / "index.txt"
        # automatically instantiate WindAtlas and LandscapeAtlas
        self.wind = _WindAtlas(self)
        self.landscape = _LandscapeAtlas(self)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        value = Path(value)
        self._path = value

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, value):
        self._region = value

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        if pyproj.CRS(value):
            self._crs = value
        else:
            raise ValueError(f"{value} is not a valid coordinate reference system")

    @property
    def wind_turbines(self):
        return self._wind_turbines

    @wind_turbines.setter
    def wind_turbines(self, turbine_names):
        if isinstance(turbine_names, str):
            turbine_names = [turbine_names]
        elif not isinstance(turbine_names, list):
            raise ValueError(f"Turbine names must be provided as list or as string")

        for name in turbine_names:
            self.add_turbine(name)

    _setup_logging = setup_logging
    _deploy_resources = deploy_resources

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

        # Create the index file in the data directory if it doesn't exist
        index_file_path = self.path / "data" / "index.txt"
        if not index_file_path.exists():
            index_file_path.touch()  # Create an empty file
            logging.info(f"Created new index file: {index_file_path}")

    def load(self, user_string = None, *, region='None', scenario='default', timestamp='latest'):
        """
        Load the datasets for the specified country, region, and scenario. If no region is specified, the datasets for
        the entire country are loaded. If no scenario is specified, the default scenario is loaded.
        """
        filename_pattern = re.compile(
            r"(?P<type>[A-Za-z]+Atlas)_(?P<country>[A-Z]+)_(?P<region>[^\d_]+)_(?P<scenario>[A-Za-z0-9]+)_(?P<timestamp>\d{8}T\d{6})\.nc"
        )
        # Parse the user string
        if user_string:
            match = filename_pattern.match(user_string)
            if not match:
                raise ValueError(f"{user_string} does not match the expected format.")
            metadata = match.groupdict()
            region = metadata["region"]
            scenario = metadata["scenario"]
            timestamp = metadata["timestamp"]
            if metadata["country"] != self.country:
                raise ValueError(f"Country code in {user_string} does not match the country code of the Atlas.")

        # locate the appropriate directory
        if scenario != 'default':
            directory = self.path / "data" / "processed" / scenario
            if not directory.exists():
                raise FileNotFoundError(f"Scenario directory for {scenario} does not exist.")
        else:
            directory = self.path / "data" / "processed"

        # find matching files
        matching_files = []
        for file in directory.glob(f"*.nc"):
            match = filename_pattern.match(file.name)
            if match:
                file_metadata = match.groupdict()
                if (
                        file_metadata["country"] == self.country
                        and file_metadata["region"] == region
                        and file_metadata["scenario"] == scenario
                        and (timestamp == 'latest' or file_metadata["timestamp"] == timestamp)
                ):
                    matching_files.append((file, file_metadata))

        if not matching_files:
            raise FileNotFoundError(f"No matching files found in {directory}.")

        if timestamp == 'latest':
            matching_files.sort(key=lambda x: x[1]["timestamp"], reverse=True)

        for file, metadata in matching_files:
            subclass = metadata["type"]
            if subclass == "WindAtlas":
                wind_data = xr.open_dataset(file, engine="netcdf4")
                self.wind.data = wind_data
                self._wind_turbines = self.wind.data.coords['turbine'].values.tolist()
                self._crs = f"epsg:{self.wind.data.rio.crs.to_epsg()}"
                if region != 'None':
                    self._region = region
                logging.info(f"Wind dataset loaded successfully: {file}")
            elif subclass == "LandscapeAtlas":
                landscape_data = xr.open_dataset(file, engine="netcdf4")
                self.landscape.data = landscape_data
                logging.info(f"Landscape dataset loaded successfully: {file}")
            else:
                logging.warning(f"Unknown subclass: {subclass}")

        return self

    def add_turbine(self, turbine_name):
        # Check if the YAML file exists
        yaml_file = self.path / "resources" / f"{turbine_name}.yml"
        if not yaml_file.is_file():
            raise FileNotFoundError(f"The YAML file for {turbine_name} does not exist.")
        if turbine_name not in self._wind_turbines:
            # Add the turbine data to the wind atlas
            self.wind.add_turbine_data(yaml_file)
            # Add the turbine name to the list of wind turbines
            self._wind_turbines.append(turbine_name)
        else:
            logging.warning(f"Turbine {turbine_name} already added.")

    def get_nuts_region(self, region, merged_name=None, to_atlascrs=True):
        nuts_dir = self.path / "data" / "nuts"
        nuts_shape = next(nuts_dir.rglob('*.shp'))
        nuts = gpd.read_file(nuts_shape)

        # Convert three-digit country code to two-digit country code
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2

        # Filter regions by country code
        feasible_regions = nuts[nuts['CNTR_CODE'] == alpha_2]

        if isinstance(region, str):
            region_list = [region]
        elif isinstance(region, list):
            region_list = region
        else:
            raise TypeError("Region must be a string or a list of strings.")

        # Find invalid regions
        invalid_regions = [r for r in region_list if r not in feasible_regions["NAME_LATN"].values]
        if invalid_regions:
            raise ValueError(f"{', '.join(invalid_regions)} are not valid regions in {self.country}.")

        # Select and merge shapes
        selected_shapes = feasible_regions[feasible_regions["NAME_LATN"].isin(region_list)]
        merged_shape = selected_shapes.dissolve()

        # Set the name for the merged region
        merged_shape["NAME_LATN"] = merged_name if merged_name else ", ".join(region_list)
        merged_shape = merged_shape.reset_index(drop=True)

        if to_atlascrs:
            merged_shape = merged_shape.to_crs(self.crs)

        return merged_shape

    def get_nuts_country(self):
        nuts_dir = self.path / "data" / "nuts"
        nuts_shape = list(nuts_dir.rglob('*.shp'))[0]
        nuts = gpd.read_file(nuts_shape)
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
        clip_shape = nuts.loc[(nuts["CNTR_CODE"] == alpha_2) & (nuts["LEVL_CODE"] == 0), :]
        return clip_shape

    def clip_to_nuts(self, region, merged_name=None, inplace=True):
        """
        Clips all Atlas datasets to the specified NUTS region
        :param merged_name: name string for union of merged shapes
        :param region: latin name of a NUTS region.
        :param inplace: boolean flag indicating whether clipped data should be updated inplace. Default is True
        :return:
        """
        clip_shape = self.get_nuts_region(region, merged_name)
        # clip both Datasets to clip_shape
        wind_dataset, _ = clip_to_geometry(self.wind, clip_shape)
        landscape_dataset, _ = clip_to_geometry(self.landscape, clip_shape)
        if inplace:
            # update Datasets in subclasses
            self.wind.data = wind_dataset
            self.landscape.data = landscape_dataset
            # update attributes in Datasets
            self.wind.data.attrs["country"] = self.country
            self.wind.data.attrs['region'] = region
            self.landscape.data.attrs["country"] = self.country
            self.landscape.data.attrs['region'] = region
            # update region property in Atlas class
            self.region = region
            logging.info(f"Atlas clipped to {region}")
        else:
            return wind_dataset, landscape_dataset

    def save(self, scenario=None):
        """
        Save NetCDF files for WindAtlas and LandscapeAtlas, adding a timestamp for versioning.
        """
        # create directory if non-existent
        if not (self.path / 'data' / 'processed').is_dir():
            self.path.mkdir(parents=True, exist_ok=True)
        if scenario:
            savepath = self.path / 'data' / 'processed' / scenario
            savepath.mkdir(parents=True, exist_ok=True)
        else:
            savepath = self.path / 'data' / 'processed'

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        # Define file paths with scenario and timestamp
        scenario_suffix = f"_{scenario}" if scenario else ""
        wind_file = savepath / f"WindAtlas_{self.country}_{self.region or 'None'}{scenario_suffix}_{timestamp}.nc"
        landscape_file = savepath / f"LandscapeAtlas_{self.country}_{self.region or 'None'}{scenario_suffix}_{timestamp}.nc"

        # Save datasets
        try:
            self.wind.data.to_netcdf(wind_file, format="NETCDF4", engine="netcdf4")
            self.landscape.data.to_netcdf(landscape_file, format="NETCDF4", engine="netcdf4")
            logging.info(f"Datasets saved successfully: {wind_file}, {landscape_file}")
        except Exception as e:
            logging.error(f"Failed to save dataset: {e}")

        # Log to index
        index_entries = [
            f"WindAtlas:{self.country}:{self.region or 'None'}:{scenario or 'default'}:{wind_file}:{timestamp}\n",
            f"LandscapeAtlas:{self.country}:{self.region or 'None'}:{scenario or 'default'}:{landscape_file}:{timestamp}\n",
        ]
        # write to index file
        try:
            with open(self.index_file, "a") as f:
                f.writelines(index_entries)
            logging.info(f"index updated with entries: {index_entries}")
        except Exception as e:
            logging.error(f"Failed to update index file: {e}")
            raise

    def _read_index(self):
        """Read the index file into a list of tuples."""
        if not isinstance(self.index_file, Path):
            self.index_file = Path(self.index_file)

        if not self.index_file.is_file():
            logging.warning("index file not found.")
            return []

        with open(self.index_file, "r") as f:
            index_list = [line.strip().split(":") for line in f]
            logging.info(f"index list: {index_list}")

        return [tuple(entry) for entry in index_list]

    def _write_index(self, entries):
        """Write updated entries to the index file."""
        with open(self.index_file, "w") as f:
            for entry in entries:
                f.write(":".join(entry) + "\n")

    def cleanup_datasets(self, scenario=None):
        """
        Retain only the most recent version of the dataset for the current country, region, and scenario.
        """
        entries = self._read_index()

        # Filter entries for the current country, region, and scenario
        filtered_entries = [
            entry for entry in entries
            if entry[1] == self.country and entry[2] == (self.region or "None") and
               (scenario is None or entry[3] == scenario)
        ]

        # Group by subclass (WindAtlas, LandscapeAtlas) and keep the latest
        latest_entries = {}
        for entry in filtered_entries:
            subclass = entry[0]
            if subclass not in latest_entries or entry[5] > latest_entries[subclass][5]:
                latest_entries[subclass] = entry

        # Delete older versions
        for entry in filtered_entries:
            if entry not in latest_entries.values():
                file_path = Path(entry[4])
                if file_path.exists():
                    file_path.unlink()

        # Update the index
        remaining_entries = [
                                entry for entry in entries if entry not in filtered_entries
                            ] + list(latest_entries.values())
        self._write_index(remaining_entries)

        logging.info(
            f"Cleanup complete for {self.country}, region {self.region or 'None'}, scenario {scenario or 'all'}.")


class _WindAtlas:
    def __init__(self, parent):
        self.parent = parent
        self.data = None
        self._load_gwa()
        self._build_netcdf("WindAtlas")
        self._set_attributes()

    def add_turbine_data(self, yaml_file):
        """
        Add wind turbine data to the xarray Dataset wrapped by the _WindAtlas class.
        Parameters:
        yaml_content (str): The YAML content containing the wind turbine data.
        Returns:
        None
        """
        # # Load the YAML file
        with yaml_file.open('r') as f:
            turbine_data = yaml.safe_load(f)
        # extract wind turbine data
        turbine_name = f"{turbine_data['manufacturer']}.{turbine_data['model']}.{turbine_data['capacity']}"
        wind_speed = list(map(float, turbine_data['V']))
        power_output = list(map(float, turbine_data['cf']))

        # initialize wind turbine power curves
        power_curve = xr.DataArray(data=power_output, coords={'wind_speed': wind_speed}, dims=['wind_speed'])
        power_curve = power_curve.assign_coords(turbine=turbine_name).expand_dims('turbine')

        if 'power_curve' in self.data:
            power_curve = xr.concat([self.data['power_curve'], power_curve], dim='turbine')

        # merge into xarray dataset
        power_curve.name = 'power_curve'
        self.data = xr.merge([self.data, power_curve])

    _load_gwa = load_gwa
    _build_netcdf = build_netcdf
    _set_attributes = set_attributes

    # loaders
    load_weibull_parameters = load_weibull_parameters
    get_turbine_attribute = get_turbine_attribute
    get_cost_assumptions = get_cost_assumptions
    get_overnight_cost = get_overnight_cost

    # methods for resource assessment
    compute_air_density_correction = compute_air_density_correction
    compute_mean_wind_speed = compute_mean_wind_speed
    compute_wind_shear_coefficient = compute_wind_shear_coefficient
    compute_weibull_pdf = compute_weibull_pdf
    simulate_capacity_factors = simulate_capacity_factors
    compute_lcoe = compute_lcoe
    compute_optimal_power_energy = compute_optimal_power_energy
    minimum_lcoe = minimum_lcoe


class _LandscapeAtlas:
    def __init__(self, parent):
        self.parent = parent
        self.data = None
        self._load_nuts()
        self._build_netcdf("LandscapeAtlas")
        self._set_attributes()

    _load_nuts = load_nuts
    _build_netcdf = build_netcdf
    _set_attributes = set_attributes
    add = add
    flatten = flatten
    convert = convert
    get_clc_codes = get_clc_codes
    add_corine_land_cover = add_corine_land_cover

    @staticmethod
    def load_and_extract_from_dict(source_dict, proxy=None, proxy_user=None, proxy_pass=None):

        for file, (directory, url) in source_dict.items():
            directory_path = Path(directory)
            directory_path.mkdir(parents=True, exist_ok=True)

            os.chdir(directory_path)

            download_path = directory_path / file
            dnld = download_file(url, download_path, proxy=proxy, proxy_user=proxy_user, proxy_pass=proxy_pass)
            logging.info(f'Download of {file} complete')

            if dnld and file.endswith(('.zip', '.kmz')):
                with zipfile.ZipFile(download_path) as zip_ref:
                    zip_file_info = zip_ref.infolist()
                    zip_extensions = [info.filename[-3:] for info in zip_file_info]

                    if 'shp' in zip_extensions:
                        for info in zip_file_info:
                            info.filename = f'{file[:-3]}{info.filename[-3:]}'
                            zip_ref.extract(info)
                    else:
                        zip_ref.extractall(directory_path)

                for index, nested_zip_path in enumerate(directory_path.glob('*.zip')):
                    with zipfile.ZipFile(nested_zip_path) as nested_zip:
                        nested_zip_info = nested_zip.infolist()
                        nested_extensions = [info.filename[-3:] for info in nested_zip_info]

                        if 'shp' in nested_extensions:
                            for info in nested_zip_info:
                                info.filename = f'{info.filename[:-4]}_{index}.{info.filename[-3:]}'
                                nested_zip.extract(info)

    def rasterize(self, shape, column=None, name=None, all_touched=False, inplace=True):
        """
        Transform geodata, for example polygons, to a xarray DataArray.
        :param shape: string of geodatafile to read or GeoDataFrame to rasterize
        :type shape: str or gpd.GeoDataFrame
        :param column: column of GeoDataFrame to use for value-assignment in rasterization
        :type column: str
        :param name: name of the xarray DataArray
        :type name: str
        :param all_touched: if True, all pixels touched by polygon are assigned. If False, only pixels whose center
        point is inside the polygon is assigned. Default is False.
        :param inplace: adds raster to `self.data` if True. Default is True.
        :type inplace: bool
        :return: merges rasterized DataArray into `self.data`
        """
        # check whether column input is sensible
        if column is not None:
            if column not in shape.columns:
                raise ValueError(f"column {column} is not in shape")
        # load shapefile if shape is string
        if isinstance(shape, str):
            shape = gpd.read_file(shape)
        elif not isinstance(shape, gpd.GeoDataFrame):
            raise TypeError(f"shape must be a GeoDataFrame or a string")
        # project shape to unique crs
        if shape.crs != self.data.rio.crs:
            shape = shape.to_crs(self.data.rio.crs)
        # reproject template to single crs if required
        if self.data["template"].rio.crs != self.data.rio.crs:
            self.data["template"] = self.data["template"].rio.reproject(self.data.rio.crs, nodata=np.nan)

        raster = self.data["template"].copy()
        for _, row in shape.iterrows():
            # mask is np.nan where not in shape and 0 where in shape
            mask = self.data["template"].rio.clip([row.geometry], self.data.rio.crs, drop=False,
                                                  all_touched=all_touched)

            if column is None:
                raster = xr.where(mask == 0, 1, raster, keep_attrs=True)
            else:
                raster = xr.where(mask == 0, row[column], raster, keep_attrs=True)

        if name is not None:
            raster.name = name
        elif name is None and column is not None:
            raster.name = column
        else:
            raise ValueError(f"'name' or 'column' must be specified.")

        if inplace:
            self.data[raster.name] = raster
            logging.info(f"rasterized {name} and added to data for {self.data.attrs['country']}")
        else:
            logging.info(f"returning rasterized {name}")
            return raster

    def compute_distance(self, data_var, inplace=False):
        """
        Compute distance from non-zero values in data_var to closest non-zero value in data_var
        :param data_var: name of a data variable in self.data
        :type data_var: str
        :param inplace: adds distance to `self.data` if True. Default is True.
        :type inplace: bool
        :return: DataArray with distances
        :rtype: xarray.DataArray
        """
        if isinstance(data_var, str):
            data_var = [data_var]
        elif not isinstance(data_var, list):
            raise TypeError("'data_var' must be a string or a list of strings.")

        for var in data_var:
            if not isinstance(var, str):
                raise TypeError("'data_var' must be a string or a list of strings.")

            if var not in self.data:
                raise ValueError(f"'{data_var}' is not a data variable in self.data")

        # check whether coordinate reference system is suitable for computing distance in meters
        if isinstance(self.data.rio.crs, rasterio.crs.CRS):
            if self.data.rio.crs.linear_units != 'metre':
                raise ValueError(f"Coordinate reference system with metric units must be used. "
                                 f"Got {str(self.data.rio.crs.linear_units)}")
        else:
            raise ValueError(f"Coordinate reference system not recognized. Must be an instance of rasterio.crs.CRS")

        distances = {}
        for var in data_var:
            # ensure xrraster is same as template
            xrraster = self.data[var]  # xrraster.interp_like(self.data["template"])

            if len(xrraster.dims) == 2:
                distance = proximity(xr.where(xrraster > 0, 1, 0), x="x", y="y")
                # re-introduce np.nan-values where template has no data
                distance = xr.where(self.data["template"].isnull(), np.nan, distance)
                # set crs of distance dataarray
                distance = distance.rio.write_crs(self.data.rio.crs)
            elif len(xrraster.dims) == 3:
                non_spatial_dim = [dim for dim in list(xrraster.dims) if dim not in ["x", "y"]]
                distance = []

                for coord in xrraster[non_spatial_dim[0]]:
                    raster_slice = xrraster[xrraster[non_spatial_dim[0]] == coord].squeeze()
                    distance_slice = proximity(xr.where(raster_slice > 0, 1, 0), x="x", y="y")
                    distance_slice = xr.where(self.data["template"].isnull(), np.nan, distance_slice)
                    distance_slice = distance_slice.rio.write_crs(self.data.rio.crs)
                    distance.append(distance_slice)

                distance = xr.concat(distance, dim=non_spatial_dim[0])

            else:
                raise ValueError('More than 3 dimensions are not supported.')
            distance.name = f"distance_{xrraster.name}"
            distance.attrs["unit"] = distance.rio.crs.linear_units

            distances[distance.name] = distance

        if inplace:
            self.data.update({var: data for var, data in distances.items()})
        else:
            return distances
