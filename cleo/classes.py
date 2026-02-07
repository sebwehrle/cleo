# %% imports
import os
import re
import json
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
from rasterio.enums import MergeAlg
from rasterio.features import rasterize as rio_rasterize

from cleo.class_helpers import (
    build_netcdf,
    deploy_resources,
    set_attributes,
    setup_logging,
    _sanitize_netcdf_attrs,
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
    clip_to_geometry, reproject, to_crs_if_needed, crs_equal,
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

logger = logging.getLogger(__name__)


def _parse_index_line(line):
    """
    Parse a single index line into a 6-tuple:
    (subclass, country, region, scenario, path, timestamp)

    Supports three formats:
    1. JSONL: line starts with '{'
    2. Tab-separated: line contains '\t' (strict 6 fields)
    3. Legacy colon-separated: parse from RIGHT to handle ':' in Windows paths
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty index line")

    # JSONL format
    if line.startswith("{"):
        try:
            obj = json.loads(line)
            return (
                obj["subclass"],
                obj["country"],
                obj["region"],
                obj["scenario"],
                obj["path"],
                obj["timestamp"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Malformed JSONL index entry: {line!r}") from e

    # Tab-separated format (strict 6 fields)
    if "\t" in line:
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"Malformed tab-separated index entry (expected 6 fields): {line!r}")
        return tuple(parts)

    # Legacy colon-separated format: rsplit from right for timestamp, then split first 5
    # This handles ":" inside Windows paths like "C:\path\to\file"
    try:
        prefix, timestamp = line.rsplit(":", 1)
        parts = prefix.split(":", 4)
        if len(parts) != 5:
            raise ValueError(f"Expected 5 fields before timestamp, got {len(parts)}")
        subclass, country, region, scenario, path = parts
        return (subclass, country, region, scenario, path, timestamp)
    except ValueError as e:
        raise ValueError(f"Malformed index entry: {line!r}") from e


def _timestamp_key(ts):
    """
    Convert timestamp string to a sortable key.
    "legacy" is treated as older than any real timestamp.
    Returns tuple (priority, datetime) where priority 0=legacy, 1=real timestamp.
    """
    if ts == "legacy":
        return (0, datetime.datetime.min)
    try:
        return (1, datetime.datetime.strptime(ts, "%Y%m%dT%H%M%S"))
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {ts!r}") from e


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
        self.index_file = self.path / "data" / "index.jsonl"
        # Defer instantiation until materialize() is called
        self._wind = None
        self._landscape = None
        self._materialized = False

    def materialize(self):
        """
        Build/download data and instantiate WindAtlas and LandscapeAtlas.
        Must be called before using wind/landscape data or calling save/load/clip.
        """
        if self._materialized:
            return
        self._wind = _WindAtlas(self)
        self._landscape = _LandscapeAtlas(self)
        self._materialized = True

    def _require_materialized(self):
        """Raise RuntimeError if atlas is not materialized."""
        # Use getattr for defensive access (tests may bypass __init__)
        wind = getattr(self, "_wind", None)
        landscape = getattr(self, "_landscape", None)
        if wind is None or landscape is None:
            raise RuntimeError(
                "Atlas not materialized. Call atlas.materialize() first."
            )

    @property
    def wind(self):
        """Access WindAtlas data (requires prior materialize() call)."""
        if self._wind is None:
            raise RuntimeError(
                "Atlas not materialized. Call atlas.materialize() first."
            )
        return self._wind

    @wind.setter
    def wind(self, value):
        self._wind = value

    @property
    def landscape(self):
        """Access LandscapeAtlas data (requires prior materialize() call)."""
        if self._landscape is None:
            raise RuntimeError(
                "Atlas not materialized. Call atlas.materialize() first."
            )
        return self._landscape

    @landscape.setter
    def landscape(self, value):
        self._landscape = value

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
        """
        Set CRS as a validated string.

        Contract:
        - Invalid CRS raises ValueError with a clear message (no pyproj CRSError leak).
        - Stored value is normalized when possible (EPSG codes become "epsg:<int>").
        """
        try:
            crs_obj = pyproj.CRS(value)
        except pyproj.exceptions.CRSError as e:
            raise ValueError(f"Invalid CRS: {value!r}") from e

        epsg = crs_obj.to_epsg()
        if epsg is not None:
            self._crs = f"epsg:{epsg}"
        else:
            # Fallback: stable string representation
            self._crs = crs_obj.to_string()

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
        index_file_path = self.path / "data" / "index.jsonl"
        if not index_file_path.exists():
            index_file_path.touch()  # Create an empty file
            logger.info(f"Created new index file: {index_file_path}")

    def load(self, user_string = None, *, region='None', scenario='default', timestamp='latest'):
        """
        Load the datasets for the specified country, region, and scenario. If no region is specified, the datasets for
        the entire country are loaded. If no scenario is specified, the default scenario is loaded.
        """
        self._require_materialized()
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

        # Fallback to legacy filenames if no timestamped files found
        if not matching_files:
            # Try legacy pattern: {Type}_{Country}.nc or {Type}_{Country}_{Region}.nc
            region_str = region if region != 'None' else None
            for atlas_type in ["WindAtlas", "LandscapeAtlas"]:
                if region_str:
                    legacy_name = f"{atlas_type}_{self.country}_{region_str}.nc"
                else:
                    legacy_name = f"{atlas_type}_{self.country}.nc"
                legacy_file = directory / legacy_name
                if legacy_file.exists():
                    matching_files.append((legacy_file, {
                        "type": atlas_type,
                        "country": self.country,
                        "region": region,
                        "scenario": scenario,
                        "timestamp": "legacy",
                    }))

        if not matching_files:
            raise FileNotFoundError(f"No matching files found in {directory}.")

        if timestamp == 'latest':
            matching_files.sort(key=lambda x: x[1]["timestamp"], reverse=True)

        for file, metadata in matching_files:
            subclass = metadata["type"]
            if subclass == "WindAtlas":
                wind_data = xr.open_dataset(file, engine="netcdf4")
                self.wind.data = wind_data
                if 'turbine' in self.wind.data.coords:
                    self._wind_turbines = self.wind.data.coords['turbine'].values.tolist()
                if self.wind.data.rio.crs is not None:
                    self._crs = f"epsg:{self.wind.data.rio.crs.to_epsg()}"
                if region != 'None':
                    self._region = region
                logger.info(f"Wind dataset loaded successfully: {file}")
            elif subclass == "LandscapeAtlas":
                landscape_data = xr.open_dataset(file, engine="netcdf4")
                self.landscape.data = landscape_data
                logger.info(f"Landscape dataset loaded successfully: {file}")
            else:
                logger.warning(f"Unknown subclass: {subclass}")

        return self

    def add_turbine(self, turbine_name):
        self._require_materialized()
        # Check if the YAML file exists
        yaml_file = self.path / "resources" / f"{turbine_name}.yml"
        if not yaml_file.is_file():
            raise FileNotFoundError(f"The YAML file for {turbine_name} does not exist.")
        if turbine_name not in self._wind_turbines:
            # Add the turbine data to the wind atlas
            self.wind.add_turbine_data(yaml_file)
            # Derive from dataset as source of truth (prevents list/dataset divergence)
            ds_turbs = self.wind.data.coords["turbine"].values.tolist()
            if turbine_name not in ds_turbs:
                raise RuntimeError(f"Failed to add turbine {turbine_name!r} to dataset; dataset turbines={ds_turbs}")
            self._wind_turbines = ds_turbs
        else:
            logger.warning(f"Turbine {turbine_name} already added.")

    def get_nuts_region(self, region, merged_name=None, to_atlascrs=True):
        nuts_dir = self.path / "data" / "nuts"
        shp_files = sorted(nuts_dir.rglob("*.shp")) if nuts_dir.exists() else []
        if not shp_files:
            raise FileNotFoundError(
                f"NUTS shapefile not found under {nuts_dir}. "
                f"Run NUTS download/extract first (e.g. atlas.landscape._load_nuts() / cleo.loaders.load_nuts)."
            )
        nuts_shape = shp_files[0]
        nuts = gpd.read_file(nuts_shape)

        # Convert three-digit country code to two-digit country code
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2

        # Filter regions by country code
        feasible_regions = nuts[nuts["CNTR_CODE"] == alpha_2]

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
            merged_shape = to_crs_if_needed(merged_shape, self.crs)

        return merged_shape

    def get_nuts_country(self):
        nuts_dir = self.path / "data" / "nuts"
        shp_files = sorted(nuts_dir.rglob("*.shp")) if nuts_dir.exists() else []
        if not shp_files:
            raise FileNotFoundError(
                f"NUTS shapefile not found under {nuts_dir}. "
                f"Run NUTS download/extract first (e.g. atlas.landscape._load_nuts() / cleo.loaders.load_nuts)."
            )
        nuts_shape = shp_files[0]
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
        self._require_materialized()
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
            logger.info(f"Atlas clipped to {region}")
        else:
            return wind_dataset, landscape_dataset

    def save(self, scenario=None):
        """
        Save NetCDF files for WindAtlas and LandscapeAtlas, adding a timestamp for versioning.
        """
        self._require_materialized()
        # Always ensure processed directory exists
        processed_dir = self.path / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        if scenario:
            savepath = self.path / 'data' / 'processed' / scenario
            savepath.mkdir(parents=True, exist_ok=True)
        else:
            savepath = self.path / 'data' / 'processed'

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        # Define file paths with scenario and timestamp
        scenario_name = scenario if scenario else "default"
        wind_file = savepath / f"WindAtlas_{self.country}_{self.region or 'None'}_{scenario_name}_{timestamp}.nc"
        landscape_file = savepath / f"LandscapeAtlas_{self.country}_{self.region or 'None'}_{scenario_name}_{timestamp}.nc"

        # Sanitize attrs to remove None values (NetCDF cannot serialize None)
        _sanitize_netcdf_attrs(self.wind.data)
        _sanitize_netcdf_attrs(self.landscape.data)

        # Save datasets - only update index on success
        try:
            self.wind.data.to_netcdf(wind_file, format="NETCDF4", engine="netcdf4")
            self.landscape.data.to_netcdf(landscape_file, format="NETCDF4", engine="netcdf4")
            logger.info(f"Datasets saved successfully: {wind_file}, {landscape_file}")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return  # Do not update index if save failed

        # Log to index (JSONL format)
        index_entries = [
            {"subclass": "WindAtlas", "country": self.country, "region": self.region or "None",
             "scenario": scenario or "default", "path": str(wind_file), "timestamp": timestamp},
            {"subclass": "LandscapeAtlas", "country": self.country, "region": self.region or "None",
             "scenario": scenario or "default", "path": str(landscape_file), "timestamp": timestamp},
        ]
        # Ensure index file parent directory exists
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        # write to index file
        try:
            with open(self.index_file, "a") as f:
                for entry in index_entries:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"index updated with entries: {index_entries}")
        except Exception as e:
            logger.error(f"Failed to update index file: {e}")
            raise

    def _read_index(self):
        """
        Read the index file into a list of 6-tuples:
        (subclass, country, region, scenario, path, timestamp)

        Supports JSONL, tab-separated, and legacy colon-separated formats.
        Malformed lines raise ValueError with context.
        """
        if not isinstance(self.index_file, Path):
            self.index_file = Path(self.index_file)

        # Also check for legacy index.txt if index.jsonl doesn't exist
        if not self.index_file.is_file():
            legacy_index = self.index_file.parent / "index.txt"
            if legacy_index.is_file():
                self.index_file = legacy_index
            else:
                logger.warning("index file not found.")
                return []

        entries = []
        with open(self.index_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                entries.append(_parse_index_line(line))
        return entries

    def _write_index(self, entries):
        """Write entries to the index file in JSONL format."""
        # Ensure we're writing to the jsonl file, not legacy txt
        if self.index_file.suffix == ".txt":
            self.index_file = self.index_file.parent / "index.jsonl"
        with open(self.index_file, "w") as f:
            for entry in entries:
                if len(entry) != 6:
                    raise ValueError(f"Index entry must have 6 fields, got {entry!r}")
                subclass, country, region, scenario, path, ts = entry
                obj = {
                    "subclass": subclass,
                    "country": country,
                    "region": region,
                    "scenario": scenario,
                    "path": str(path),
                    "timestamp": ts,
                }
                f.write(json.dumps(obj) + "\n")

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
            if (subclass not in latest_entries) or (_timestamp_key(entry[5]) > _timestamp_key(latest_entries[subclass][5])):
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

        logger.info(
            f"Cleanup complete for {self.country}, region {self.region or 'None'}, scenario {scenario or 'all'}.")


class _AtlasDataVarSetterMixin:
    """
    Mixin providing exact-grid enforcement for raster data variable assignments.

    Contract:
    - Template is stored as self.data["template"] with x/y dims.
    - Any DataArray with both x and y dims must match template coords exactly.
    - Non-raster DataArrays (without x/y dims) pass through unchanged.
    """

    @property
    def crs(self) -> str:
        """
        Backward-compatible alias for Atlas CRS.

        Canonical CRS is owned by parent Atlas (self.parent.crs). Sub-objects delegate.
        """
        if self.parent is None or getattr(self.parent, "crs", None) is None:
            raise ValueError("Atlas parent CRS is missing (self.parent.crs is None).")
        return self.parent.crs

    def _set_var(self, name: str, da) -> None:
        """
        Set a data variable with exact-grid enforcement for rasters.

        :param name: Variable name to set
        :param da: DataArray to assign
        :raises RuntimeError: If self.data is None
        :raises ValueError: If template missing (except when setting template itself)
        :raises ValueError: If raster x/y coords don't match template
        """
        if self.data is None:
            raise RuntimeError(f"Cannot set '{name}': self.data is None")

        # Special case: setting template itself
        if name == "template":
            if "x" not in da.dims or "y" not in da.dims:
                raise ValueError("template must have both x and y dims")
            self.data["template"] = da
            return

        # For all other vars, require template to exist
        if "template" not in self.data.data_vars:
            raise ValueError(
                f"Cannot set '{name}': template not in self.data.data_vars. "
                "Set template first via _set_var('template', ...)."
            )

        # Enforce exact grid for raster-like DataArrays
        from cleo.spatial import enforce_exact_grid

        da = enforce_exact_grid(da, self.data["template"], var_name=name)
        self.data[name] = da


class _WindAtlas(_AtlasDataVarSetterMixin):
    def __init__(self, parent):
        self.parent = parent
        self.data = None
        self._load_gwa()
        self._build_netcdf("WindAtlas")
        self._ensure_windatlas_schema()
        self._set_attributes()

    def _ensure_windatlas_schema(self):
        """
        Validate and migrate WindAtlas dataset to canonical schema (windatlas_v2).

        Ensures:
        - No stray 'band' coord/var/dim
        - 'template' data_var exists with dims ('y', 'x')
        - 'wind_speed' coord exists with canonical grid 0..40 step 0.5

        If migration occurs and _netcdf_path is set, persists changes to disk.
        """
        changed = False

        # Drop stray band coord/var (observed in real file: coords include 'band' but dims do not)
        if "band" in self.data.coords or "band" in self.data.variables:
            self.data = self.data.drop_vars("band", errors="ignore")
            changed = True
        if "band" in self.data.dims:
            if self.data.sizes["band"] == 1:
                self.data = self.data.isel(band=0, drop=True)
                changed = True
            else:
                raise ValueError("Unexpected multi-band WindAtlas dataset.")

        # Template presence and dims
        if "template" not in self.data.data_vars:
            raise ValueError("WindAtlas invalid: missing template.")
        if tuple(self.data["template"].dims) != ("y", "x"):
            raise ValueError("WindAtlas invalid: template must be dims ('y', 'x').")

        # wind_speed coord
        u = np.arange(0.0, 40.0 + 0.5, 0.5)
        if "wind_speed" not in self.data.coords:
            self.data = self.data.assign_coords(wind_speed=u)
            changed = True
        else:
            ws = np.asarray(self.data.coords["wind_speed"].values)
            if ws.shape != u.shape or not np.all(ws == u):
                raise ValueError("WindAtlas invalid: wind_speed grid mismatch vs canonical 0..40 step 0.5.")

        # Schema marker
        if self.data.attrs.get("cleo_schema_version") != "windatlas_v2":
            self.data.attrs["cleo_schema_version"] = "windatlas_v2"
            changed = True

        # Persist migration
        if changed and getattr(self, "_netcdf_path", None) is not None:
            logger.info(f"Migrated WindAtlas schema; writing back to {self._netcdf_path}")
            # Load all data into memory and close file handle before overwriting
            self.data = self.data.load()
            self.data.close()
            # Write to temp file then rename (avoids file handle conflicts)
            import tempfile
            import shutil
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=self._netcdf_path.parent)
            os.close(tmp_fd)
            self.data.to_netcdf(tmp_path)
            shutil.move(tmp_path, self._netcdf_path)

    def add_turbine_data(self, yaml_file):
        """
        Add wind turbine data to the xarray Dataset wrapped by the _WindAtlas class.
        Parameters:
        yaml_content (str): The YAML content containing the wind turbine data.
        Returns:
        None
        """
        # Strict preconditions (no self-heal - _ensure_windatlas_schema should have run)
        if "wind_speed" not in self.data.coords:
            raise ValueError("WindAtlas invalid: missing wind_speed coord.")
        if "band" in self.data.coords or "band" in self.data.dims:
            raise ValueError("WindAtlas invalid: unexpected band.")
        if "template" not in self.data.data_vars:
            raise ValueError("WindAtlas invalid: missing template.")

        # # Load the YAML file
        with yaml_file.open('r') as f:
            turbine_data = yaml.safe_load(f)
        # extract wind turbine data
        turbine_name = f"{turbine_data['manufacturer']}.{turbine_data['model']}.{turbine_data['capacity']}"
        old_u = np.array(list(map(float, turbine_data['V'])))
        old_p = np.array(list(map(float, turbine_data['cf'])))

        # Resample power curve to atlas wind_speed grid
        u = self.data.coords["wind_speed"].values
        new_p = np.interp(u, old_u, old_p, left=0.0, right=0.0)

        # initialize wind turbine power curves on atlas grid
        power_curve = xr.DataArray(data=new_p, coords={'wind_speed': u}, dims=['wind_speed'])
        power_curve = power_curve.assign_coords(turbine=turbine_name).expand_dims('turbine')

        # Direct assignment (power_curve is non-raster: dims are wind_speed/turbine, not x/y)
        power_curve.name = "power_curve"

        if "power_curve" not in self.data.data_vars:
            # First turbine: establish Dataset coord turbine=[turbine_name], then assign
            self.data = self.data.assign_coords(turbine=[turbine_name])
            self.data["power_curve"] = power_curve
            return

        # Append: concat, then replace entire power_curve in dataset
        # Must drop old power_curve first to avoid xarray dimension conflict during coord update
        combined = xr.concat([self.data["power_curve"], power_curve], dim="turbine")
        self.data = self.data.drop_vars("power_curve")
        self.data = self.data.assign_coords(turbine=combined.coords["turbine"].values)
        self.data["power_curve"] = combined

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


class _LandscapeAtlas(_AtlasDataVarSetterMixin):
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
        """
        Download files described by source_dict and extract archives safely.

        source_dict format:
            { filename: (directory, url), ... }

        Security/UX:
        - Never changes process CWD.
        - Safe ZIP extraction (rejects absolute paths / '..' traversal).
        """

        def _safe_extract_zip(zip_ref, dest_dir: Path):
            dest_dir = dest_dir.resolve()
            for info in zip_ref.infolist():
                name = info.filename
                # Reject absolute paths and path traversal
                target = (dest_dir / name).resolve()
                try:
                    target.relative_to(dest_dir)
                except Exception:
                    raise ValueError(f"Unsafe zip member path: {name!r}")
                zip_ref.extract(info, path=dest_dir)

        for file, (directory, url) in source_dict.items():
            directory_path = Path(directory)
            directory_path.mkdir(parents=True, exist_ok=True)

            download_path = directory_path / file
            dnld = download_file(
                url,
                download_path,
                proxy=proxy,
                proxy_user=proxy_user,
                proxy_pass=proxy_pass,
            )
            logger.info(f"Download of {file} complete")

            if dnld and file.endswith((".zip", ".kmz")):
                # Validate archive before opening
                if not zipfile.is_zipfile(download_path):
                    size = download_path.stat().st_size
                    head = download_path.read_bytes()[:200].decode("utf-8", "replace")
                    raise ValueError(
                        f"Invalid ZIP archive: {file!r}\n"
                        f"  download_path: {download_path}\n"
                        f"  url: {url}\n"
                        f"  size: {size} bytes\n"
                        f"  head: {head!r}"
                    )

                try:
                    with zipfile.ZipFile(download_path) as zip_ref:
                        zip_file_info = zip_ref.infolist()
                        zip_extensions = [info.filename[-3:] for info in zip_file_info]

                        if "shp" in zip_extensions:
                            # Preserve prior behavior (renaming members), but extract safely into directory_path
                            for info in zip_file_info:
                                info.filename = f"{file[:-3]}{info.filename[-3:]}"
                                # Validate renamed path before extraction
                                dest_dir = directory_path.resolve()
                                target = (dest_dir / info.filename).resolve()
                                try:
                                    target.relative_to(dest_dir)
                                except Exception:
                                    raise ValueError(f"Unsafe zip member path: {info.filename!r}")
                                zip_ref.extract(info, path=directory_path)
                        else:
                            _safe_extract_zip(zip_ref, directory_path)
                except zipfile.BadZipFile as e:
                    size = download_path.stat().st_size
                    head = download_path.read_bytes()[:200].decode("utf-8", "replace")
                    raise ValueError(
                        f"Corrupt ZIP archive: {file!r}\n"
                        f"  download_path: {download_path}\n"
                        f"  url: {url}\n"
                        f"  size: {size} bytes\n"
                        f"  head: {head!r}"
                    ) from e

                # Handle nested zips (as before), but safely and without chdir
                for index, nested_zip_path in enumerate(directory_path.glob("*.zip")):
                    # Validate nested archive before opening
                    if not zipfile.is_zipfile(nested_zip_path):
                        size = nested_zip_path.stat().st_size
                        head = nested_zip_path.read_bytes()[:200].decode("utf-8", "replace")
                        raise ValueError(
                            f"Invalid nested ZIP archive: {nested_zip_path.name!r}\n"
                            f"  nested_path: {nested_zip_path}\n"
                            f"  parent_archive: {file!r}\n"
                            f"  size: {size} bytes\n"
                            f"  head: {head!r}"
                        )

                    try:
                        with zipfile.ZipFile(nested_zip_path) as nested_zip:
                            nested_zip_info = nested_zip.infolist()
                            nested_extensions = [info.filename[-3:] for info in nested_zip_info]

                            if "shp" in nested_extensions:
                                for info in nested_zip_info:
                                    info.filename = f"{info.filename[:-4]}_{index}.{info.filename[-3:]}"
                                    dest_dir = directory_path.resolve()
                                    target = (dest_dir / info.filename).resolve()
                                    try:
                                        target.relative_to(dest_dir)
                                    except Exception:
                                        raise ValueError(f"Unsafe zip member path: {info.filename!r}")
                                    nested_zip.extract(info, path=directory_path)
                            else:
                                _safe_extract_zip(nested_zip, directory_path)
                    except zipfile.BadZipFile as e:
                        size = nested_zip_path.stat().st_size
                        head = nested_zip_path.read_bytes()[:200].decode("utf-8", "replace")
                        raise ValueError(
                            f"Corrupt nested ZIP archive: {nested_zip_path.name!r}\n"
                            f"  nested_path: {nested_zip_path}\n"
                            f"  parent_archive: {file!r}\n"
                            f"  size: {size} bytes\n"
                            f"  head: {head!r}"
                        ) from e

    def rasterize(self, *args, column=None, name=None, all_touched=False, inplace=True):
        """
        Rasterize vector geometries onto a template grid.

        Supported call styles (backward compatible):
          A) rasterize(template_da, shape, column=None, all_touched=False)
          B) rasterize(shape, column=None, name=None, all_touched=False)  # uses self.data["template"]

        shape can be a GeoDataFrame or a path (str/Path) to a vector file.
        If column is None, burns 1.0 inside geometries and leaves template values elsewhere.
        If column is provided, burns that numeric value per-geometry (last wins on overlap) and leaves template elsewhere.
        """
        # Backward compatibility: older call sites pass inplace=False to return the raster
        # without assigning it into self.data.
        # New behavior: if name is provided and inplace=True -> assign to self.data[name_to_assign].
        # If inplace=False -> return the raster and do not assign, regardless of name.
        if not inplace:
            name_to_assign = None
        else:
            name_to_assign = name

        # Parse args to support both conventions
        template = None
        shape = None

        if len(args) == 0:
            # rasterize(shape=..., ...) not provided positionally; require shape via keyword? (we don't support)
            raise TypeError("rasterize() missing required positional argument: 'shape'")

        if len(args) == 1:
            # New style: rasterize(shape, ...)
            shape = args[0]
            try:
                template = self.data["template"]
            except Exception as e:
                raise RuntimeError(
                    "No template available. Provide template as first argument or set self.data['template'].") from e

        elif len(args) == 2:
            # Old style: rasterize(template, shape, ...)
            template = args[0]
            shape = args[1]
        else:
            raise TypeError(f"rasterize() takes 2 positional arguments at most ({len(args)} given)")

        # Validate template
        if not isinstance(template, xr.DataArray):
            raise TypeError("template must be an xarray.DataArray")

        # Load shape
        if isinstance(shape, (str, Path)):
            shape = gpd.read_file(shape)
        elif not hasattr(shape, "geometry"):
            raise TypeError("shape must be a GeoDataFrame or a path to a vector file")

        if shape.empty:
            # Nothing to rasterize: return template unchanged
            out = template.copy()
            if name_to_assign is not None:
                self._set_var(name_to_assign, out)
            return out

        # Ensure CRS: shapes must match template CRS
        tmpl_crs = template.rio.crs
        if tmpl_crs is None:
            # fall back to parent.crs contract
            tmpl_crs = getattr(self.parent, "crs", None)
            if tmpl_crs is None:
                raise ValueError("Template CRS missing and parent.crs is not set.")
            template = template.rio.write_crs(tmpl_crs)

        if shape.crs is None:
            raise ValueError("Input shape has no CRS; cannot rasterize safely.")
        # Reproject to template CRS if needed (semantic comparison)
        shape = to_crs_if_needed(shape, tmpl_crs)

        # Raster grid spec
        transform = template.rio.transform(recalc=True)
        out_shape = (template.sizes["y"], template.sizes["x"])

        # Prepare (geometry, value) tuples
        if column is None:
            shapes = [(geom, 1.0) for geom in shape.geometry]
            burned = rio_rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0.0,
                all_touched=all_touched,
                merge_alg=MergeAlg.replace,
                dtype="float32",
            )
            burned_da = xr.DataArray(
                burned,
                dims=("y", "x"),
                coords={"y": template["y"].values, "x": template["x"].values},
            ).rio.write_transform(transform).rio.write_crs(tmpl_crs)

            out = xr.where(burned_da != 0.0, 1.0, template)

        else:
            if column not in shape.columns:
                available = [c for c in shape.columns if c != "geometry"]
                raise ValueError(
                    f"Column {column!r} not found in shape. Available columns: {available!r}"
                )
            # Require numeric (close to old behavior; avoids burning strings)
            vals = shape[column].to_numpy()
            if not np.issubdtype(vals.dtype, np.number):
                raise TypeError(f"Column {column!r} must be numeric, got dtype={vals.dtype!r}.")

            shapes = [(geom, float(val)) for geom, val in zip(shape.geometry, vals, strict=True)]
            burned = rio_rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=transform,
                fill=np.nan,
                all_touched=all_touched,
                merge_alg=MergeAlg.replace,
                dtype="float32",
            )
            burned_da = xr.DataArray(
                burned,
                dims=("y", "x"),
                coords={"y": template["y"].values, "x": template["x"].values},
            ).rio.write_transform(transform).rio.write_crs(tmpl_crs)

            out = xr.where(~np.isnan(burned_da), burned_da, template)

        if name_to_assign is not None:
            self._set_var(name_to_assign, out)
        return out

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
                if len(non_spatial_dim) != 1:
                    raise ValueError(f"Expected exactly one non-spatial dim, got {non_spatial_dim!r}")
                dim = non_spatial_dim[0]

                slices = []
                for coord in xrraster[dim].values:
                    raster_slice = xrraster.sel({dim: coord}).squeeze(drop=True)
                    distance_slice = proximity(xr.where(raster_slice > 0, 1, 0), x="x", y="y")
                    distance_slice = xr.where(self.data["template"].isnull(), np.nan, distance_slice)
                    distance_slice = distance_slice.rio.write_crs(self.data.rio.crs)
                    slices.append(distance_slice.expand_dims({dim: [coord]}))

                distance = xr.concat(slices, dim=dim, join="exact")
                distance = distance.assign_coords({dim: xrraster[dim].values})
            else:
                raise ValueError('More than 3 dimensions are not supported.')

            distance.name = f"distance_{xrraster.name}"
            distance.attrs["unit"] = distance.rio.crs.linear_units

            distances[distance.name] = distance

        if inplace:
            self.data.update({var: data for var, data in distances.items()})
        else:
            return distances
