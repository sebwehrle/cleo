# helpers for the Atlas class
# %% imports
import re
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
from pathlib import Path
from cleo.assess import turbine_overnight_cost
from cleo.utils import download_file
from cleo.spatial import reproject_raster_if_needed, to_crs_if_needed

logger = logging.getLogger(__name__)


def _maybe_chunk_auto(x):
    """
    Chunk a DataArray with 'auto' chunking if dask is available.

    If dask is not installed or chunking fails, returns the input unchanged.
    This allows the codebase to work with or without dask installed.
    """
    try:
        import dask.array  # noqa: F401
    except ImportError:
        return x
    try:
        return x.chunk("auto")
    except Exception:
        return x


def _load_yaml_file(path: Path, *, context: str) -> dict:
    """Load a YAML file with consistent UTF-8 handling and good error context."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing YAML resource for {context}: {path}. "
            "If this is a packaged resource, run `atlas.deploy_resources()` to populate "
            "`<atlas.path>/resources/` (workdir override surface)."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path} for {context}: {e}") from e
    return data


def _safe_extractall(zip_ref: zipfile.ZipFile, dest_dir: Path) -> None:
    """Safely extract a ZipFile, rejecting path traversal (zip-slip)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()

    for member in zip_ref.infolist():
        name = member.filename

        # Reject absolute paths and Windows drive letters
        if name.startswith("/") or name.startswith("\\") or re.match(r"^[A-Za-z]:", name):
            raise ValueError(f"Unsafe zip member (absolute path): {name}")

        target = (dest_dir / name).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe zip member (path traversal): {name}")

    zip_ref.extractall(dest_dir)



# %% CRS inference from GWA
def fetch_gwa_crs(iso3):
    """
    Fetch CRS from Global Wind Atlas GeoJSON API for a given country.

    Network contract:
    - Must not hang indefinitely: uses a bounded timeout.
    - Must raise with actionable context (ISO3 + URL) on request failures.

    :param iso3: ISO 3166-1 alpha-3 country code
    :type iso3: str
    :return: CRS string from the GeoJSON response
    :rtype: str
    :raises ValueError: If CRS is missing from the GeoJSON response
    :raises RuntimeError: If the HTTP request fails (timeout, DNS, etc.)
    """
    url = f"https://globalwindatlas.info/api/gdal/country/geojson?areaId={iso3}"
    headers = {
        "Accept": "application/geo+json,application/json;q=0.9,*/*;q=0.8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://globalwindatlas.info",
        "Referer": "https://globalwindatlas.info/en/download/gis-files",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=(10, 60))
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch GWA CRS for {iso3} from {url}: {e}") from e

    # Handle double-encoded JSON (response.json() returns a string containing JSON)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GWA GeoJSON response for country {iso3}: {e}") from e

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
    Load elevation data with local GeoTIFF preference.

    Contract:
    - Returned elevation MUST match the reference grid exactly (dims/coords/transform/CRS),
      to prevent silent misalignment and downstream NaN stripes.

    If local elevation_w_bathymetry.tif exists and can be opened, use it (but still reproject_match).
    Otherwise, build elevation from Copernicus DEM tiles.

    :param base_dir: Base directory (Path) containing country data folders
    :param iso3: ISO 3166-1 alpha-3 country code
    :param reference_da: Reference DataArray for CRS/bounds/grid alignment
    :return: Elevation DataArray aligned to reference grid
    :rtype: xarray.DataArray
    :raises FileNotFoundError: If neither local GeoTIFF nor CopDEM tiles available
    """
    from pathlib import Path
    from rasterio.enums import Resampling
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    from cleo.copdem import download_copdem_tiles_for_bbox, build_copdem_elevation_like

    path_raw_country = Path(base_dir) / "data" / "raw" / iso3
    local_path = path_raw_country / f"{iso3}_elevation_w_bathymetry.tif"

    # Ensure reference_da has a CRS (needed for reproject_match and bounds transforms)
    if reference_da.rio.crs is None:
        air_density_path = path_raw_country / f"{iso3}_air-density_100.tif"
        if air_density_path.exists():
            reference_da = rxr.open_rasterio(air_density_path).squeeze(drop=True)
            reference_da = ensure_crs_from_gwa(reference_da, iso3)
            logger.info(f"Using air-density raster as CRS reference: {air_density_path}")
        else:
            raise ValueError(
                f"Reference DataArray has no CRS and air-density reference file not found: {air_density_path}"
            )

        if reference_da.rio.crs is None:
            raise ValueError(
                f"Reference DataArray has no CRS and air-density file also lacks CRS: {air_density_path}"
            )

    # Try local GeoTIFF first (but always align to reference grid)
    if local_path.exists():
        try:
            elevation = rxr.open_rasterio(local_path).rename("elevation").squeeze()
            elevation = ensure_crs_from_gwa(elevation, iso3)

            elevation = elevation.rio.reproject_match(
                reference_da,
                resampling=Resampling.bilinear,
                nodata=np.nan,
            )
            elevation.name = "elevation"

            # Hard alignment guardrail: coords must match exactly
            if set(elevation.dims) != set(reference_da.dims):
                raise ValueError(
                    f"Elevation dims mismatch after reproject_match: {elevation.dims} vs {reference_da.dims}"
                )
            for dim in reference_da.dims:
                if not np.array_equal(elevation[dim].values, reference_da[dim].values):
                    raise ValueError(f"Elevation coordinate mismatch on dim '{dim}' after reproject_match")

            logger.info(f"Loaded local elevation from {local_path} and matched to reference grid.")
            return elevation
        except Exception as e:
            logger.warning(f"Local elevation file exists but failed to open/match: {e}")

    # Fall back to CopDEM
    logger.info("Local elevation not available, building from Copernicus DEM")

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

    tile_paths = download_copdem_tiles_for_bbox(
        base_dir,
        iso3,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )

    elevation = build_copdem_elevation_like(reference_da, tile_paths)
    logger.info(f"Built elevation from {len(tile_paths)} Copernicus DEM tiles")

    return elevation


# %% methods
def get_cost_assumptions(self, attribute_name):
    """
    Retrieve cost assumptions from YAML in `<atlas.path>/resources/`.

    :param self: an instance of the Atlas class
    :param attribute_name: Name of the cost assumption attribute to retrieve
    :type attribute_name: str
    :return: Value of the specific cost assumption
    """
    path = Path(self.parent.path) / "resources" / "cost_assumptions.yml"
    data = _load_yaml_file(path, context="cost assumptions")
    return data[attribute_name]


def get_overnight_cost(self, turbine_id):
    power = self.get_turbine_attribute(turbine_id, "capacity") / 1000
    hub_height = self.get_turbine_attribute(turbine_id, "hub_height")
    rotor_diameter = self.get_turbine_attribute(turbine_id, "rotor_diameter")
    year = self.get_turbine_attribute(turbine_id, "commissioning_year")
    return turbine_overnight_cost(power=power, hub_height=hub_height, rotor_diameter=rotor_diameter, year=year)


def get_powercurves(self):
    """
    Load power curves from YAML files in `<atlas.path>/resources/`.

    Loads a power curve for each turbine listed in `self.wind_turbines`.

    Contract:
    - YAML files must be read via context managers (no leaked file handles).
    - Errors must identify the turbine / path that failed.
    """
    file_paths = [Path(self.path) / "resources" / f"{t}.yml" for t in self.wind_turbines]

    frames = []
    for path, turbine in zip(file_paths, self.wind_turbines):
        data = _load_yaml_file(path, context=f"power curve turbine={turbine}")
        try:
            frames.append(
                pd.DataFrame(
                    data=data["cf"],
                    index=data["V"],
                    columns=[f"{data['manufacturer']}.{data['model']}.{data['capacity']}"],
                )
            )
        except Exception as e:
            raise ValueError(f"Invalid power curve YAML schema in {path}: {e}") from e

    self.power_curves = pd.concat(frames, axis=1)
    logger.info(f"Power curves for {self.wind_turbines} loaded.")


def get_turbine_attribute(self, turbine_id, attribute_name):
    """
    Retrieve turbine attribute from dataset metadata.

    String metadata (manufacturer, model, model_key) is stored in cleo_turbines_json attr.
    Numeric metadata (capacity, hub_height, etc.) is stored in dataset variables.

    :param turbine_id: Turbine config ID (YAML file stem)
    :type turbine_id: str
    :param attribute_name: Name of the turbine attribute to retrieve
        (e.g., "hub_height", "capacity", "rotor_diameter", "commissioning_year",
         "manufacturer", "model", "model_key")
    :type attribute_name: str
    :return: Value of the specific turbine attribute
    :raises ValueError: If turbine_id not found or attribute missing from dataset
    """
    import json

    # Check turbine dimension exists
    if "turbine" not in self.data.dims:
        raise ValueError(
            f"No turbines in dataset; cannot retrieve attribute '{attribute_name}' for turbine_id={turbine_id!r}."
        )

    # Read turbine metadata from cleo_turbines_json attr
    if "cleo_turbines_json" not in self.data.attrs:
        raise ValueError(
            f"Dataset missing cleo_turbines_json attr; re-run materialize_canonical()."
        )
    turbines_meta = json.loads(self.data.attrs["cleo_turbines_json"])
    turbine_id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}

    if turbine_id not in turbine_id_to_idx:
        raise ValueError(
            f"Turbine ID {turbine_id!r} not found in dataset. Available: {list(turbine_id_to_idx.keys())}"
        )

    turbine_idx = turbine_id_to_idx[turbine_id]
    turbine_meta = turbines_meta[turbine_idx]

    # String attributes from JSON attr
    json_attrs = {"manufacturer", "model", "model_key"}
    if attribute_name in json_attrs:
        if attribute_name not in turbine_meta:
            raise ValueError(
                f"Turbine attribute '{attribute_name}' not found in cleo_turbines_json for turbine_id={turbine_id!r}."
            )
        return turbine_meta[attribute_name]

    # Numeric attributes from dataset variables
    var_name = f"turbine_{attribute_name}"
    if var_name not in self.data.data_vars:
        raise ValueError(
            f"Turbine attribute '{attribute_name}' not found in dataset for turbine_id={turbine_id!r}. "
            f"Dataset may have been created before turbine metadata storage was implemented. "
            f"Re-add turbines to store metadata."
        )

    value = self.data[var_name].isel(turbine=turbine_idx).values
    # Convert numpy scalar to Python scalar
    if hasattr(value, "item"):
        value = value.item()
    return value


def load_weibull_parameters(self, height):
    """
    Load Weibull parameters for a specific height.

    Contract:
    - Missing required rasters must raise immediately (no silent (None, None) fallback).
    - Returned rasters are in `self.parent.crs` (Atlas CRS), unless already matching.

    :param self: an instance of the Atlas class
    :param height: Height for which to load Weibull parameters (e.g. 50, 100, 150, 200)
    :type height: int
    :return: Tuple containing Weibull parameter rasters (a, k)
    :rtype: tuple[xarray.DataArray, xarray.DataArray]
    :raises FileNotFoundError: If required raster files are missing
    :raises RuntimeError: If reprojection/clipping fails
    """
    from pathlib import Path

    path_raw_country = Path(self.parent.path) / "data" / "raw" / f"{self.parent.country}"
    a_path = path_raw_country / f"{self.parent.country}_combined-Weibull-A_{height}.tif"
    k_path = path_raw_country / f"{self.parent.country}_combined-Weibull-k_{height}.tif"

    missing = [str(p) for p in [a_path, k_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Weibull parameter raster(s) for height={height}: {missing}"
        )

    try:
        a = _maybe_chunk_auto(rxr.open_rasterio(a_path))
        k = _maybe_chunk_auto(rxr.open_rasterio(k_path))
        a.name = "weibull_a"
        k.name = "weibull_k"

        # Ensure CRS is set before reprojection
        a = ensure_crs_from_gwa(a, self.parent.country)
        k = ensure_crs_from_gwa(k, self.parent.country)

        # Reproject to Atlas CRS if needed (semantic comparison)
        a = reproject_raster_if_needed(a, self.parent.crs, nodata=np.nan)
        k = reproject_raster_if_needed(k, self.parent.crs, nodata=np.nan)

        if self.parent.region is not None:
            clip_shape = self.parent.get_nuts_region(self.parent.region)
            clip_shape = to_crs_if_needed(clip_shape, self.parent.crs)
            a = a.rio.clip(clip_shape.geometry)
            k = k.rio.clip(clip_shape.geometry)

        return a, k
    except Exception as e:
        raise RuntimeError(f"Failed to load Weibull parameters for height {height}: {e}") from e


def load_air_density(self, height):
    """
    Load air density raster for a specific height.

    Contract:
    - Missing raster file must raise immediately (no silent fallback).
    - Returned raster is in `self.parent.crs` (Atlas CRS).

    :param self: wind atlas instance with data property
    :param height: Height for which to load air density (e.g. 50, 100, 150, 200)
    :type height: int
    :return: DataArray containing air density raster
    :rtype: xarray.DataArray
    :raises FileNotFoundError: If raster file is missing
    :raises RuntimeError: If reprojection/clipping fails
    """
    from pathlib import Path

    path_raw_country = Path(self.parent.path) / "data" / "raw" / f"{self.parent.country}"
    rho_path = path_raw_country / f"{self.parent.country}_air-density_{height}.tif"

    if not rho_path.is_file():
        raise FileNotFoundError(
            f"Missing air-density raster for height={height}: {rho_path}. "
            f"Run atlas.materialize() or atlas.wind._load_gwa() to download GWA data."
        )

    try:
        rho = _maybe_chunk_auto(rxr.open_rasterio(rho_path))
        rho.name = "air_density"

        # Ensure CRS is set before reprojection
        rho = ensure_crs_from_gwa(rho, self.parent.country)

        # Reproject to Atlas CRS if needed
        rho = reproject_raster_if_needed(rho, self.parent.crs, nodata=np.nan)

        if self.parent.region is not None:
            clip_shape = self.parent.get_nuts_region(self.parent.region)
            clip_shape = to_crs_if_needed(clip_shape, self.parent.crs)
            rho = rho.rio.clip(clip_shape.geometry)

        return rho
    except Exception as e:
        raise RuntimeError(f"Failed to load air density for height {height}: {e}") from e


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
    Download wind resource data for the specified country from GWA API.

    Downloads:
    - air density
    - combined Weibull parameters (A, k)
    for multiple heights.

    Note: elevation is NOT downloaded from GWA (handled via local GeoTIFF or Copernicus DEM).
    """
    url = "https://globalwindatlas.info/api/gis/country"
    layers = ["air-density", "combined-Weibull-A", "combined-Weibull-k"]
    heights = ["10", "50", "100", "150", "200"]

    c = self.parent.country
    path_raw = self.parent.path / "data" / "raw" / c
    path_raw.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing WindScape with Global Wind Atlas data")

    for ly in layers:
        for h in heights:
            fname = f"{c}_{ly}_{h}.tif"
            fpath = path_raw / fname
            if fpath.is_file():
                continue

            durl = f"{url}/{c}/{ly}/{h}"
            try:
                success = download_file(durl, fpath)
            except requests.RequestException as e:
                logger.error(f"Error downloading {fname} from {durl}: {e}")
                continue

            if success:
                logger.info(f"Download of {fname} from {durl} complete")
            else:
                logger.info(f"Download of {fname} from {durl} failed")

    logger.info("Skipping GWA elevation download; handled via local GeoTIFF or Copernicus DEM")
    logger.info(f"Global Wind Atlas data for {c} initialized.")


def load_nuts(self, resolution="03M", year=2021, crs=4326):
    RESOLUTION = ["01M", "03M", "10M", "20M", "60M"]
    YEAR = [2021, 2016, 2013, 2010, 2006, 2003]
    CRS = [3035, 4326, 3857]

    if resolution not in RESOLUTION:
        raise ValueError(f"Invalid resolution: {resolution}")

    if year not in YEAR:
        raise ValueError(f"Invalid year: {year}")

    if crs not in CRS:
        raise ValueError(f"Invalid crs: {crs}")

    nuts_path = Path(self.parent.path) / "data" / "nuts"
    nuts_path.mkdir(parents=True, exist_ok=True)

    url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/"
    file_collection = f"ref-nuts-{year}-{resolution}.shp.zip"
    file_name = f"NUTS_RG_{resolution}_{year}_{crs}.shp.zip"

    target_inner_zip = nuts_path / file_name
    if target_inner_zip.is_file():
        logger.info("NUTS borders initialised.")
        return

    # Download outer zip (collection) if needed
    outer_zip_path = nuts_path / file_collection
    if not outer_zip_path.is_file():
        download_file(url + file_collection, outer_zip_path)
        logger.info(f"Downloaded {file_collection}")

    # Extract the requested inner zip safely (no trust in zip paths)
    with zipfile.ZipFile(str(outer_zip_path), "r") as zip_ref:
        try:
            info = zip_ref.getinfo(file_name)
        except KeyError:
            raise FileNotFoundError(f"File {file_name} not found inside {file_collection}")

        # Write the inner zip to a known-safe location
        with zip_ref.open(info, "r") as src, open(target_inner_zip, "wb") as dst:
            dst.write(src.read())

    # Safely extract inner zip (zip-slip protection)
    with zipfile.ZipFile(str(target_inner_zip), "r") as zip_inner:
        _safe_extractall(zip_inner, nuts_path)

    logger.info(f"Extracted {file_name}")


def get_clc_codes(self, reverse=False):
    path = Path(self.parent.path) / "resources" / "clc_codes.yml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing resource file: {path}. "
            "Run `atlas.deploy_resources()` to populate `<atlas.path>/resources/` "
            "or ensure this YAML is present in your workdir resources override."
        )

    data = _load_yaml_file(path, context="CLC codes")
    return data["clc_reverse" if reverse else "clc_codes"]


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
    clc = to_crs_if_needed(clc, self.parent.crs)

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
                logger.info(f"Corine Land Cover class {clc_codes[clc_code].lower()} added.")
            else:
                cat_raster = cat_raster.expand_dims(dim="clc_class", axis=0)
                cat_raster.coords["clc_class"] = [clc_codes[clc_code]]
                clc_array.append(cat_raster)

    if len(classes_to_process) > 1:
        clc_3d = xr.concat(clc_array, dim="clc_class", join="exact")
        clc_3d = clc_3d.rio.write_crs(self.parent.crs)
        self.add(clc_3d, name="corine_land_cover")
        logger.info(f"Corine Land Cover added.")
