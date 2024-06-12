# %% imports
import os
import numpy as np
import xarray as xr
import pyproj
import rasterio.crs
import geopandas as gpd
import yaml
import logging
import zipfile
import pycountry as pct
from pathlib import Path
from xrspatial import proximity
from tempfile import NamedTemporaryFile

from dask.diagnostics import ProgressBar

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
    clip_to_geometry,
)

from cleo.assess import (
    compute_air_density_correction,
    compute_lcoe,
    compute_mean_wind_speed,
    compute_optimal_power_energy,
    compute_terrain_roughness_length,
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
        # automatically instantiate WindAtlas and LandscapeAtlas
        self.wind = _WindAtlas(self)
        self.landscape = _LandscapeAtlas(self)
        self._check_and_load_datasets()

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

    def _check_and_load_datasets(self) -> None:
        """
        Check for the existence of saved NetCDF files for WindAtlas and LandscapeAtlas. If they exist, load the datasets
        and set the properties of the Atlas class.
        """
        wind_atlas_path = self.path / "data" / "processed" / f"WindAtlas_{self.country}.nc"
        landscape_atlas_path = self.path / "data" / "processed" / f"LandscapeAtlas_{self.country}.nc"

        # check if datasets exist and open
        if wind_atlas_path.is_file():
            wind_dataset = xr.open_dataset(wind_atlas_path)

        if landscape_atlas_path.is_file():
            landscape_dataset = xr.open_dataset(landscape_atlas_path)

        if wind_dataset and landscape_dataset:
            wind_attrs = wind_dataset.attrs
            landscape_attrs = landscape_dataset.attrs

            if wind_attrs != landscape_attrs:
                raise ValueError("Attributes of WindAtlas and LandscapeAtlas do not match")

        self.country = wind_attrs.get('country')
        self.region = wind_attrs.get('region')
        self.crs = wind_dataset.rio.crs.to_string()

        if wind_dataset:
            self.wind.data = wind_dataset
            wind_dataset.close()
        if landscape_dataset:
            self.landscape.data = landscape_dataset
            landscape_dataset.close()

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

    def get_nuts_region(self, region):
        nuts_dir = self.path / "data" / "nuts"
        nuts_shape = list(nuts_dir.rglob('*.shp'))[0]
        nuts = gpd.read_file(nuts_shape)
        # convert three-digit country code to two-digit country code
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
        # pre-filter possible region names based on two-digit country code
        feasible_regions = nuts.loc[nuts['CNTR_CODE'] == alpha_2, ["LEVL_CODE", "NAME_LATN"]]
        if feasible_regions["NAME_LATN"].str.contains(region).any():
            clip_shape = nuts.loc[nuts["NAME_LATN"] == region, :]
            clip_shape = clip_shape.to_crs(self.crs)
            return clip_shape
        else:
            raise ValueError(f"{region} is not a valid region in {self.country}.")

    def get_nuts_country(self):
        nuts_dir = self.path / "data" / "nuts"
        nuts_shape = list(nuts_dir.rglob('*.shp'))[0]
        nuts = gpd.read_file(nuts_shape)
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
        clip_shape = nuts.loc[(nuts["CNTR_CODE"] == alpha_2) & (nuts["LEVL_CODE"] == 0), :]
        return clip_shape

    def clip_to_nuts(self, region, inplace=True):
        """
        Clips all Atlas datasets to the specified NUTS region
        :param region: latin name of a NUTS region.
        :param inplace: boolean flag indicating whether clipped data should be updated inplace. Default is True
        :return:
        """
        clip_shape = self.get_nuts_region(region)
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

    def write_netcdfs(self, complevel=4):
        """
        Save NetCDF data safely to a file
        """
        def save_tempfile(dataset, folder):

            with NamedTemporaryFile(suffix=".tmp", dir=folder, delete=False) as tmp_file:
                tmp_file_path = Path(tmp_file.name)

            logging.debug(f"Writing data to {tmp_file_path} ...")
            encoding = {}

            for var_name, var in dataset.variables.items():
                encoding[var_name] = {"zlib": True, "complevel": complevel}

            write_job = dataset.to_netcdf(
                str(tmp_file_path), compute=False, format="NETCDF4", engine="netcdf4", encoding=encoding)

            with ProgressBar():
                write_job.compute()

            return tmp_file_path

        tmp_wind = save_tempfile(self.wind.data, self.path / "data" / "processed")
        tmp_landscape = save_tempfile(self.landscape.data, self.path / "data" / "processed")

        # close and unlink target files if they exist
        if self.region is None:
            wind_atlas_file = self.path / "data" / "processed" / f"WindAtlas_{self.country}.nc"
            landscape_atlas_file = self.path / "data" / "processed" / f"LandscapeAtlas_{self.country}.nc"
        else:
            wind_atlas_file = self.path / "data" / "processed" / f"WindAtlas_{self.country}_{self.region}.nc"
            landscape_atlas_file = self.path / "data" / "processed" / f"LandscapeAtlas_{self.country}_{self.region}.nc"

        if wind_atlas_file.exists():
            self.wind.data.close()
            wind_atlas_file.unlink()

        if landscape_atlas_file.exists():
            self.landscape.data.close()
            landscape_atlas_file.unlink()

        # rename temporary files
        tmp_wind.rename(wind_atlas_file)
        tmp_landscape.rename(landscape_atlas_file)

        logging.info(
            f"Atlas saved to {str(self.path / 'data' / 'processed')}")


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
    compute_terrain_roughness_length = compute_terrain_roughness_length
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
            self.data["template"] = self.data["template"].rio.reproject(self.data.rio.crs)

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
