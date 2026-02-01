# helpers for the Atlas class
# %% imports
import json
import yaml
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import logging
from cleo.assess import turbine_overnight_cost
from cleo.utils import download_file


# %% CRS inference from GWA
def fetch_gwa_crs(iso3):
    """
    Fetch CRS from Global Wind Atlas GeoJSON API for a given country.

    :param iso3: ISO 3166-1 alpha-3 country code
    :type iso3: str
    :return: CRS string from the GeoJSON response
    :rtype: str
    :raises ValueError: If CRS is missing from the GeoJSON response
    :raises requests.RequestException: If the HTTP request fails
    """
    url = f"https://globalwindatlas.info/api/gdal/country/geojson?areaId={iso3}"
    headers = {
        "Accept": "application/geo+json,application/json;q=0.9,*/*;q=0.8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://globalwindatlas.info",
        "Referer": "https://globalwindatlas.info/en/download/gis-files",
        "User-Agent": "Mozilla/5.0",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    payload = response.json()

    # Handle double-encoded JSON (response.json() returns a string containing JSON)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GWA GeoJSON response for country {iso3}: {e}")

    try:
        crs = payload["crs"]["properties"]["name"]
    except (KeyError, TypeError):
        raise ValueError(f"CRS missing from GWA GeoJSON response for country {iso3}")

    return crs


def ensure_crs_from_gwa(ds, iso3):
    """
    Ensure a raster DataArray has a CRS set, fetching from GWA if needed.

    If ds.rio.crs is already set, returns ds unchanged.
    If ds.rio.crs is None, fetches CRS from GWA GeoJSON API and writes it.

    :param ds: Raster DataArray or Dataset
    :type ds: xarray.DataArray or xarray.Dataset
    :param iso3: ISO 3166-1 alpha-3 country code
    :type iso3: str
    :return: DataArray/Dataset with CRS set
    :rtype: xarray.DataArray or xarray.Dataset
    :raises ValueError: If CRS is missing from the GeoJSON response
    """
    if ds.rio.crs is not None:
        return ds

    crs = fetch_gwa_crs(iso3)
    return ds.rio.write_crs(crs)


def load_elevation(base_dir, iso3, reference_da):
    """
    Load elevation data with legacy-file preference.

    If local elevation_w_bathymetry.tif exists and can be opened, use it.
    Otherwise, build elevation from Copernicus DEM tiles.

    :param base_dir: Base directory (Path) containing country data folders
    :param iso3: ISO 3166-1 alpha-3 country code
    :param reference_da: Reference DataArray for CRS/bounds/grid alignment
    :return: Elevation DataArray aligned to reference grid
    :rtype: xarray.DataArray
    :raises FileNotFoundError: If neither legacy file nor CopDEM tiles available
    """
    from pathlib import Path
    from cleo.copdem import download_copdem_tiles_for_bbox, build_copdem_elevation_like

    path_raw_country = Path(base_dir) / "data" / "raw" / iso3
    legacy_path = path_raw_country / f"{iso3}_elevation_w_bathymetry.tif"

    # Try legacy file first
    if legacy_path.exists():
        try:
            elevation = rxr.open_rasterio(legacy_path)
            elevation = elevation.rename("elevation").squeeze()
            elevation = ensure_crs_from_gwa(elevation, iso3)
            logging.info(f"Loaded legacy elevation from {legacy_path}")
            return elevation
        except Exception as e:
            logging.warning(f"Legacy elevation file exists but failed to open: {e}")

    # Fall back to CopDEM
    logging.info(f"Legacy elevation not available, building from Copernicus DEM")

    # If reference_da lacks CRS, try to use air-density file as reference
    if reference_da.rio.crs is None:
        air_density_path = path_raw_country / f"{iso3}_air-density_100.tif"
        if air_density_path.exists():
            try:
                reference_da = rxr.open_rasterio(air_density_path).squeeze(drop=True)
                logging.info(f"Using air-density raster as CRS reference: {air_density_path}")
            except Exception as e:
                raise ValueError(
                    f"Reference DataArray has no CRS and air-density file failed to open: {e}"
                )
        else:
            raise ValueError(
                f"Reference DataArray has no CRS and air-density reference file not found: {air_density_path}"
            )

        if reference_da.rio.crs is None:
            raise ValueError(
                f"Reference DataArray has no CRS and air-density file also lacks CRS: {air_density_path}"
            )

    # Get bounds from reference raster and transform to EPSG:4326 for CopDEM tile selection
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    bounds = reference_da.rio.bounds()
    src_crs = reference_da.rio.crs
    if src_crs is None:
        raise ValueError("reference_da has no CRS (.rio.crs is None)")

    src = CRS.from_user_input(src_crs)
    dst = CRS.from_epsg(4326)

    if src != dst:
        min_lon, min_lat, max_lon, max_lat = transform_bounds(src, dst, *bounds, densify_pts=21)
    else:
        min_lon, min_lat, max_lon, max_lat = bounds

    # Download tiles
    tile_paths = download_copdem_tiles_for_bbox(
        base_dir,
        iso3,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )

    # Build elevation matched to reference
    elevation = build_copdem_elevation_like(reference_da, tile_paths)
    logging.info(f"Built elevation from {len(tile_paths)} Copernicus DEM tiles")

    return elevation


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

        # Ensure CRS is set before reprojection
        a = ensure_crs_from_gwa(a, self.parent.country)
        k = ensure_crs_from_gwa(k, self.parent.country)

        if a.rio.crs != self.parent.crs:
            a = a.rio.reproject(self.parent.crs, nodata=np.nan)

        if k.rio.crs != self.parent.crs:
            k = k.rio.reproject(self.parent.crs, nodata=np.nan)

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
                        success = download_file(durl, fpath)
                        if success:
                            logging.info(f'Download of {fname} from {durl} complete')
                        else:
                            logging.info(f'Download of {fname} from {durl} failed')
                except requests.RequestException as e:
                    logging.error(f'Error downloading {fname}: {e}')

    # Skip GWA elevation download - elevation is handled via legacy file or Copernicus DEM
    logging.info("Skipping GWA elevation download; handled via legacy file or Copernicus DEM")

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


def get_clc_codes(self, reverse=False):
    with open(str(self.parent.path / 'resources' / 'clc_codes.yml')) as f:
        data = yaml.safe_load(f)
        if not reverse:
            return data['clc_codes']
        else:
            return data['clc_reverse']


def add_corine_land_cover(self, clc_class=None):
    """
    loads Corine Land Cover
    :return:
    """
    # TODO: add corine land cover codes to resources
    # TODO: download corine land cover data for Europe
    # TODO: clip corine land cover data to country with get_nuts_borders() from cleo.loaders
    # TODO: merge corine land cover data into landscape.data (with add method?) as landscape.data.corine_land_cover
    # TODO: current code is very slow

    # Corine Land Cover - pastures and crop area
    clc = gpd.read_file(self.parent.path / 'data' / 'site' / 'clc' / 'CLC_2018_AT.shp')
    clc['CODE_18'] = clc['CODE_18'].astype('int')
    clc = clc.to_crs(self.parent.crs)

    if self.parent.region is not None:
        clip_shape = self.parent.get_nuts_region(self.parent.region)
        clc = clc.clip(clip_shape.geometry)
    clc = clc.dissolve(by="CODE_18")

    clc_array = []
    clc_codes = get_clc_codes(self, reverse=False)

    # determine clc classes to process
    classes_to_process = clc_codes.keys() if clc_class is None else (clc_class if
                                                                     isinstance(clc_class, list) else [clc_class])
    # process classes
    for clc_code in classes_to_process:
        if clc_code in clc.index:
            cat_layer = clc.loc[[clc_code]]
            cat_raster = self.rasterize(cat_layer, name="corine_land_cover", all_touched=False, inplace=False)

            if len(classes_to_process) == 1:
                cat_raster = cat_raster.rio.write_crs(self.parent.crs)
                self.add(cat_raster, name=clc_codes[clc_code].lower())
                logging.info(f"Corine Land Cover class {clc_codes[clc_code].lower()} added.")
            else:
                cat_raster = cat_raster.expand_dims(dim="clc_class", axis=0)
                cat_raster.coords["clc_class"] = [clc_codes[clc_code]]
                clc_array.append(cat_raster)

    if len(classes_to_process) > 1:
        clc_3d = xr.concat(clc_array, dim="clc_class")
        clc_3d = clc_3d.rio.write_crs(self.parent.crs)
        self.add(clc_3d, name="corine_land_cover")
        logging.info(f"Corine Land Cover added.")
