# %% imports
import os
import zipfile
import logging
import numpy as np
import geopandas as gpd
import xarray as xr
import rasterio.crs
from pathlib import Path
from xrspatial import proximity
from tempfile import NamedTemporaryFile
from dask.diagnostics import ProgressBar

from cleo.utils import download_file, add, convert
from cleo.spatial import clip_to_geometry
from cleo.clc_codes import clc_codes


def _init_geoscape_data(atlas):
    """
    Initialize the geoscape data either from disk or from a provided atlas.
    :param atlas: a cleo.WindScape object
    :return:
    """
    fname = atlas.path / "data" / "processed" / f"geoscape_{atlas.country}.nc"
    if fname.exists():
        dataset = xr.open_dataset(fname)
    else:
        # Get the coordinates
        spatial_coords = {coord: atlas.data.coords[coord] for coord in ['x', 'y']}
        # first data variable
        first_data_var = [next(iter(atlas.data.data_vars))]
        # Find the name of the non-spatial dimension
        non_spatial_dim = [dim for dim in atlas.data[first_data_var].dims if dim not in ['x', 'y']][0]
        # Generate mask of nans
        nan_mask = np.isnan(atlas.data[first_data_var]).all(dim=non_spatial_dim)
        # Create a new dataset with only the spatial coordinates and a new data variable
        dataset = xr.where(nan_mask, np.nan, 0).rename({first_data_var[0]: "template"})

    return dataset


class GeoScape:
    """
    Represents geospatial data describing potential wind turbine sites
    """
    def __init__(self, atlas):
        self.path = atlas.path
        self.country = atlas.country
        self.crs = atlas.crs
        self.data = _init_geoscape_data(atlas)
        self.clip_shape = atlas.clip_shape

    def __repr__(self):
        return (
            f"<SiteData {self.country}>\n"
            f"CRS: {self.crs}\n"
            f"{self.data.data_vars}"
        )

    add = add
    convert = convert

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
        :return: merges rasterized DataArray into self.data
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
        if shape.crs != self.crs:
            shape = shape.to_crs(self.crs)
        # reproject template to single crs if required
        if self.data["template"].rio.crs != self.crs:
            self.data["template"] = self.data["template"].rio.reproject(self.crs)

        raster = self.data["template"].copy()
        for _, row in shape.iterrows():
            # mask is np.nan where not in shape and 0 where in shape
            mask = self.data["template"].rio.clip([row.geometry], self.crs, drop=False, all_touched=all_touched)

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
            logging.info(f"rasterized {name} and added to data for {self.country}")
        else:
            logging.info(f"returning rasterized {name}")
            return raster

    def compute_distance(self, data_var, inplace=False):
        """
        Compute distance from non-zero values in data_var to closest non-zero value in data_var
        :param data_var: name of a data variable in self.data
        :type data_var: str
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
        if isinstance(self.crs, rasterio.crs.CRS):
            if self.crs.linear_units != 'metre':
                raise ValueError(f"Coordinate reference system with metric units must be used. "
                                 f"Got {str(self.crs.linear_units)}")
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
                distance = distance.rio.write_crs(self.crs)
            elif len(xrraster.dims) == 3:
                non_spatial_dim = [dim for dim in list(xrraster.dims) if dim not in ["x", "y"]]
                distance = []

                for coord in xrraster[non_spatial_dim[0]]:
                    raster_slice = xrraster[xrraster[non_spatial_dim[0]] == coord].squeeze()
                    distance_slice = proximity(xr.where(raster_slice > 0, 1, 0), x="x", y="y")
                    distance_slice = xr.where(self.data["template"].isnull(), np.nan, distance_slice)
                    distance_slice = distance_slice.rio.write_crs(self.crs)
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

    def load_and_extract_from_dict(self, source_dict, proxy=None, proxy_user=None, proxy_pass=None):

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
    
    def clip_to(self, clip_shape, region=None, inplace=True):
        """
        Wrapper function to enable inplace-operations for clip_to_geometry-function
        :param clip_shape: path-string or Geopandas GeoDataFrame containing the clipping shape
        :param inplace: boolean flag indicating whether clipped data should be updated inplace. Default is True
        :return:
        """
        data_clipped, clip_shape_used = clip_to_geometry(self, clip_shape)
        
        if self.clip_shape is None:    
            # add clip shape to atlas
            self.clip_shape = clip_shape_used
        
        if inplace:
            self.data = data_clipped
        else:
            return data_clipped

        logging.info("SiteData already clipped to clip shape.")

    def load_corine_land_cover(self):
        """
        add Corine Land Cover to site.data
        :return:
        """
        # TODO: download corine land cover data for Europe
        # TODO: clip corine land cover data to country with get_nuts_borders() from cleo.loaders
        # TODO: merge corine land cover data into SiteData.data (with add method?) as site.data.corine_land_cover
        # TODO: current code is very slow

        # Corine Land Cover - pastures and crop area
        clc = gpd.read_file(self.path / 'data' / 'site' / 'clc' / 'CLC_2018_AT.shp')
        clc['CODE_18'] = clc['CODE_18'].astype('int')
        clc = clc.to_crs(self.crs)
        if self.clip_shape is not None:
            clc = clc.clip(self.clip_shape.geometry)
        clc = clc.dissolve(by="CODE_18")

        clc_array = []
        for key, cat in clc_codes.items():
            if key in clc.index:
                cat_layer = clc.loc[[key]]
                cat_raster = self.rasterize(cat_layer, name="corine_land_cover", all_touched=False, inplace=False)
                cat_raster = cat_raster.expand_dims(dim="clc_class", axis=0)
                cat_raster.coords["clc_class"] = [cat]
                clc_array.append(cat_raster)

        clc_3d = xr.concat(clc_array, dim="clc_class")
        clc_3d = clc_3d.rio.write_crs(self.crs)
        self.add(clc_3d, name="corine_land_cover")
        logging.info(f"Corine Land Cover loaded.")

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

            if (self.path / "data" / "processed" / f"geoscape_{self.country}.nc").exists():
                self.data.close()
                (self.path / "data" / "processed" / f"geoscape_{self.country}.nc").unlink()

            tmp_file.close()

        tmp_file_path.rename(self.path / "data" / "processed" / f"geoscape_{self.country}.nc")

        logging.info(
            f"GeoScape data saved to {str(self.path / 'data' / 'processed' / f'geoscape_{self.country}.nc')}")
