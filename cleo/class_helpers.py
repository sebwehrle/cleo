# %% imports
import logging
import logging.config
import shutil
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio.crs
from pathlib import Path

from cleo.spatial import crs_equal, reproject_raster_if_needed

logger = logging.getLogger(__name__)


# %% helpers
def _sanitize_netcdf_attrs(ds):
    """
    Remove None values from Dataset attrs to ensure NetCDF serialization succeeds.

    :param ds: xarray Dataset with potentially None-valued attrs
    :return: Dataset with None attrs removed
    """
    sanitized_attrs = {k: v for k, v in ds.attrs.items() if v is not None}
    ds.attrs = sanitized_attrs
    return ds


# %% methods
def build_netcdf(self, atlas_type):
    """
    Build a NetCDF file from the downloaded data or open an existing one.
    The NetCDF file stores the wind resource data in a structured format.

    Architecture principles upheld:
    - Canonical grid is defined in the Atlas CRS (parent.crs), not the source CRS.
    - CRS comparisons are semantic (cleo.spatial.crs_equal), and reprojection is explicit.
    - Template is the single source of truth for exact-grid enforcement (_set_var).
    """
    path_raw = self.parent.path / "data" / "raw" / self.parent.country
    path_netcdf = self.parent.path / "data" / "processed"

    if self.parent.region is not None:
        fname_netcdf = path_netcdf / f"{atlas_type}_{self.parent.country}_{self.parent.region}.nc"
    else:
        fname_netcdf = path_netcdf / f"{atlas_type}_{self.parent.country}.nc"

    # ------------------------------------------------------------------
    # Create new NetCDF (first materialization)
    # ------------------------------------------------------------------
    if not fname_netcdf.is_file():
        logger.info(f"Building new {atlas_type} object at {str(path_netcdf)}")

        weibull_a_path = path_raw / f"{self.parent.country}_combined-Weibull-A_100.tif"
        if not weibull_a_path.is_file():
            raise FileNotFoundError(f"Missing reference raster for template grid: {weibull_a_path}")

        with rxr.open_rasterio(weibull_a_path, parse_coordinates=True).squeeze() as weibull_a_100:
            from cleo.loaders import ensure_crs_from_gwa
            from cleo.spatial import reproject_raster_if_needed

            # Guard against stray 'band' coord/dim from rasterio single-band rasters
            if "band" in weibull_a_100.dims:
                if weibull_a_100.sizes["band"] == 1:
                    weibull_a_100 = weibull_a_100.isel(band=0, drop=True)
                else:
                    raise ValueError("Unexpected multi-band raster in build_netcdf; expected single-band.")
            if "band" in weibull_a_100.coords:
                weibull_a_100 = weibull_a_100.drop_vars("band")

            # Ensure source CRS exists (GWA v4 rasters may lack CRS tags)
            weibull_a_100 = ensure_crs_from_gwa(weibull_a_100, self.parent.country)

            # IMPORTANT: define the canonical grid in Atlas CRS (parent.crs)
            weibull_a_100 = reproject_raster_if_needed(
                weibull_a_100, self.parent.crs, nodata=np.nan
            )

            # Build dataset coords from the reprojected reference raster
            self.data = xr.Dataset(coords=weibull_a_100.coords)
            self.data = self.data.rio.write_crs(self.parent.crs)

            # Create template (same canonical grid & CRS)
            nan_mask = np.isnan(weibull_a_100)
            template = xr.where(nan_mask, np.nan, 0).rename("template")
            try:
                template = template.rio.write_crs(self.parent.crs, inplace=False)
                template = template.rio.write_transform(weibull_a_100.rio.transform(), inplace=False)
            except Exception:
                # If rio metadata attachment fails, grid enforcement still holds via coords.
                pass

            self._set_var("template", template)

            # Validate template is 2D (y, x)
            if tuple(self.data["template"].dims) != ("y", "x"):
                raise ValueError(f"template must have dims ('y', 'x'), got {self.data['template'].dims}")

            fname_netcdf.parent.mkdir(parents=True, exist_ok=True)
            self.data.to_netcdf(fname_netcdf)

        # Store netcdf path for later migration writes
        self._netcdf_path = fname_netcdf

    # ------------------------------------------------------------------
    # Open existing NetCDF and enforce CRS invariants / migrate legacy files
    # ------------------------------------------------------------------
    else:
        self.data = xr.open_dataset(fname_netcdf)

        # Store netcdf path for later migration writes
        self._netcdf_path = fname_netcdf

        # Drop stray 'band' coord/var (observed in real files: coords include 'band' but dims do not)
        if "band" in self.data.coords or "band" in self.data.variables:
            self.data = self.data.drop_vars("band", errors="ignore")
        if "band" in self.data.dims:
            if self.data.sizes["band"] == 1:
                self.data = self.data.isel(band=0, drop=True)
            else:
                raise ValueError("Unexpected multi-band WindAtlas dataset.")

        # Ensure rioxarray knows which dims are spatial so CRS metadata can be read
        if "x" in self.data.dims and "y" in self.data.dims:
            try:
                self.data = self.data.rio.set_spatial_dims(x_dim="x", y_dim="y")
            except Exception:
                # If this fails, we'll still try to proceed; CRS detection may be unavailable
                pass

        # --- CRS detection & enforcement for existing NetCDFs -----------------
        existing_crs = self.data.rio.crs

        # rioxarray may fail to infer CRS if CF grid-mapping is not attached to vars.
        # Try to recover CRS from a CF-style spatial_ref variable.
        if existing_crs is None:
            try:
                import pyproj

                if "spatial_ref" in self.data.variables:
                    attrs = self.data["spatial_ref"].attrs
                    wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
                    if wkt:
                        existing_crs = pyproj.CRS.from_wkt(wkt)
                    else:
                        epsg = attrs.get("epsg_code")
                        if epsg is not None:
                            existing_crs = pyproj.CRS.from_user_input(epsg)
            except Exception:
                existing_crs = None

        def _attach_crs_to_raster_vars(ds: xr.Dataset, crs_value):
            """
            Ensure each raster-like data_var has CRS metadata so rio.reproject works.
            Only touches vars that look like rasters (have x/y dims).
            """
            if crs_value is None:
                return ds
            out = ds
            for name, da in ds.data_vars.items():
                if "x" in da.dims and "y" in da.dims:
                    try:
                        out[name] = da.rio.write_crs(crs_value, inplace=False)
                    except Exception:
                        # If this fails, keep var as-is; reprojection will raise
                        pass
            # Also ensure dataset-level CRS is present
            try:
                out = out.rio.write_crs(crs_value)
            except Exception:
                pass
            return out

        if existing_crs is None:
            logger.warning(
                f"{fname_netcdf} has no CRS metadata; writing parent.crs={self.parent.crs!r}."
            )
            self.data = _attach_crs_to_raster_vars(self.data, self.parent.crs)
        else:
            # Attach the detected existing CRS to raster vars so reproject can run
            self.data = _attach_crs_to_raster_vars(self.data, existing_crs)

            # Reproject if CRS differs (using centralized semantic comparison)
            if not crs_equal(existing_crs, self.parent.crs):
                from cleo.spatial import canonical_crs_str

                dst_crs = canonical_crs_str(self.parent.crs)
                logger.info(f"Reprojecting dataset from {existing_crs} to {dst_crs}")

                # Reproject entire dataset (updates coordinates and all variables)
                self.data = self.data.rio.reproject(dst_crs, nodata=np.nan)
                self.data = _attach_crs_to_raster_vars(self.data, self.parent.crs)

        logger.info(f"Existing {atlas_type} at {str(path_netcdf)} opened.")

        # Reconstruct template if missing (for legacy NetCDFs)
        if "template" not in self.data.data_vars:
            weibull_a_path = path_raw / f"{self.parent.country}_combined-Weibull-A_100.tif"
            if weibull_a_path.is_file():
                with rxr.open_rasterio(weibull_a_path, parse_coordinates=True).squeeze() as weibull_a_100:
                    from cleo.loaders import ensure_crs_from_gwa

                    weibull_a_100 = ensure_crs_from_gwa(weibull_a_100, self.parent.country)

                    # Build template from source mask, then align it to the existing dataset grid.
                    nan_mask = np.isnan(weibull_a_100)
                    template = xr.where(nan_mask, np.nan, 0).rename("template")

                    # Write CRS and transform from source for reprojection/match
                    try:
                        template = template.rio.write_crs(weibull_a_100.rio.crs, inplace=False)
                        template = template.rio.write_transform(weibull_a_100.rio.transform(), inplace=False)
                    except Exception:
                        pass

                    # Align to existing data coords (legacy migration)
                    template = template.rio.reproject_match(self.data, nodata=np.nan)
                    self._set_var("template", template)
                    logger.info("Reconstructed template from GWA weibull_a_100")


def deploy_resources(self):
    """
    Ensure YAML resource files are present in the workdir at `<atlas.path>/resources/`.

    Contract (A3, always-deploy, idempotent):
    - Packaged defaults live under `cleo/resources/*.yml`.
    - On every Atlas init, ensure workdir has a copy of each packaged YAML.
    - Do NOT overwrite existing workdir YAMLs (workdir is the override surface).
    - Fail loudly if packaged resources cannot be found (broken install).
    """
    import logging
    import shutil
    from pathlib import Path
    from importlib import resources as importlib_resources

    dest_dir = Path(self.path) / "resources"
    dest_dir.mkdir(parents=True, exist_ok=True)

    pkg_root = importlib_resources.files("cleo").joinpath("resources")
    if not pkg_root.is_dir():
        raise FileNotFoundError(
            "Cleo packaged resources are missing (expected package dir `cleo/resources`). "
            "This indicates a broken installation/build. "
            "Reinstall from a proper wheel/sdist, or use the conda environment.yaml install."
        )

    packaged = [
        p for p in pkg_root.iterdir()
        if p.is_file() and p.name.lower().endswith(".yml")
    ]
    if not packaged:
        raise FileNotFoundError(
            "Cleo packaged resources directory exists but contains no *.yml files. "
            "This indicates a broken installation/build."
        )

    copied = 0
    skipped = 0
    for p in packaged:
        dest = dest_dir / p.name
        if dest.exists():
            skipped += 1
            continue

        # `as_file` materializes the resource to a real filesystem path (works for wheels too)
        with importlib_resources.as_file(p) as src_path:
            shutil.copy(src_path, dest)
        copied += 1

    logger.info(
        f"Resource files ensured in {dest_dir} (copied={copied}, skipped_existing={skipped})."
    )


def set_attributes(self):
    self.data.attrs['country'] = self.parent.country
    self.data.attrs['region'] = self.parent.region
    if self.data.rio.crs is None:
        raise AttributeError(f"{self.data} does not have a coordinate reference system.")
    # Semantic CRS comparison using centralized helper
    if not crs_equal(self.data.rio.crs, self.parent.crs):
        raise ValueError(f"Coordinate reference system mismatch: expected={self.parent.crs} got={self.data.rio.crs}")


def setup_logging(self, console_level="INFO", file_level="DEBUG"):
    """
    Configure cleo logger namespace without touching the root logger.
    All cleo modules should log via logging.getLogger(__name__) so messages
    propagate to the 'cleo' logger.
    """
    log_dir = self.path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cleo_{self.country}.log"

    cleo_logger = logging.getLogger("cleo")
    cleo_logger.setLevel(logging.DEBUG)  # allow handlers to filter
    cleo_logger.propagate = False

    # Avoid duplicate handlers on repeated Atlas creation
    for h in list(cleo_logger.handlers):
        cleo_logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, str(console_level).upper(), logger.info))
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, str(file_level).upper(), logging.DEBUG))
    fh.setFormatter(fmt)

    cleo_logger.addHandler(ch)
    cleo_logger.addHandler(fh)
