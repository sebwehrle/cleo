# resource assessment methods of the Atlas class
import hashlib
import json
import numpy as np
import xarray as xr
import rioxarray as rxr
import logging

from scipy.special import gamma
from cleo.utils import _match_to_template
from cleo.chunk import compute_chunked
from cleo.spatial import crs_equal, reproject_raster_if_needed, to_crs_if_needed, _rio_clip_robust

logger = logging.getLogger(__name__)


# %% semantic cache helpers
def _canon(obj) -> str:
    """Canonical JSON encoding for signature computation."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sig(algo: str, ver: str, params: dict) -> str:
    """Compute SHA-256 hex signature for algo/version/params."""
    payload = {"algo": algo, "algo_version": ver, "params": params}
    return hashlib.sha256(_canon(payload).encode()).hexdigest()


def _set_prov(da: xr.DataArray, algo: str, ver: str, params: dict) -> xr.DataArray:
    """Set provenance attrs on DataArray, dropping any existing cleo:* attrs."""
    # Remove existing cleo:* attrs
    new_attrs = {k: v for k, v in da.attrs.items() if not k.startswith("cleo:")}
    new_attrs["cleo:algo"] = algo
    new_attrs["cleo:algo_version"] = ver
    new_attrs["cleo:params"] = _canon(params)
    new_attrs["cleo:sig"] = _sig(algo, ver, params)
    da = da.copy()
    da.attrs = new_attrs
    return da


def _validate_sig(sig) -> str | None:
    """Validate signature format (64 hex chars), return sig if valid else None."""
    if sig is None or not isinstance(sig, str) or len(sig) != 64:
        return None
    try:
        int(sig, 16)  # validate hex
        return sig
    except ValueError:
        return None


def _get_sig(da, height=None) -> str | None:
    """Return cleo:sig if valid (64 hex chars), else None.

    For multi-height DataArrays, looks up per-height sig from cleo:sigs dict.
    If height is None, tries to extract it from the da's height coord (for slices).
    """
    if da is None or not hasattr(da, "attrs"):
        return None

    # First try simple cleo:sig attr (single-value or non-height-indexed)
    sig = da.attrs.get("cleo:sig")
    if sig is not None:
        return _validate_sig(sig)

    # Try per-height lookup from cleo:sigs
    sigs_json = da.attrs.get("cleo:sigs")
    if sigs_json is None:
        return None

    # Parse sigs dict
    try:
        sigs = json.loads(sigs_json) if isinstance(sigs_json, str) else sigs_json
    except (json.JSONDecodeError, TypeError):
        return None

    # Determine height to look up
    if height is None:
        # Try to extract height from da's coords (for slices)
        if hasattr(da, "coords") and "height" in da.coords:
            h_coord = da.coords["height"]
            # Scalar coord from .sel(height=X)
            if h_coord.ndim == 0:
                height = float(h_coord.values)
            elif h_coord.size == 1:
                height = float(h_coord.values[0])

    if height is None:
        return None

    # Look up sig for this height (keys are strings in JSON)
    height_key = str(int(height)) if float(height) == int(height) else str(height)
    sig = sigs.get(height_key)
    return _validate_sig(sig)


def _matches(ds: xr.Dataset, var: str, expected_sig: str) -> bool:
    """Check if var exists in ds with matching signature."""
    if var not in ds.data_vars:
        return False
    return _get_sig(ds[var]) == expected_sig


def _set_height_prov(da: xr.DataArray, height, algo: str, ver: str, params: dict) -> xr.DataArray:
    """Set per-height provenance in DataArray attrs.

    Stores algo/version once and accumulates per-height sigs in cleo:sigs dict.
    """
    da = da.copy()
    new_attrs = dict(da.attrs)

    # Set algo metadata (shared across heights)
    new_attrs["cleo:algo"] = algo
    new_attrs["cleo:algo_version"] = ver

    # Get or create sigs dict
    sigs_json = new_attrs.get("cleo:sigs")
    try:
        sigs = json.loads(sigs_json) if isinstance(sigs_json, str) else {}
    except (json.JSONDecodeError, TypeError):
        sigs = {}

    # Add sig for this height
    height_key = str(int(height)) if float(height) == int(height) else str(height)
    sigs[height_key] = _sig(algo, ver, params)
    new_attrs["cleo:sigs"] = _canon(sigs)

    # Remove simple cleo:sig if present (we use per-height sigs now)
    new_attrs.pop("cleo:sig", None)
    new_attrs.pop("cleo:params", None)

    da.attrs = new_attrs
    return da


def _base_params(self) -> dict:
    """Extract base params from self for signature computation (defensive)."""
    parent = getattr(self, "parent", None)
    return {
        "country": getattr(parent, "country", None) if parent else None,
        "crs": str(getattr(parent, "crs", None)) if parent and hasattr(parent, "crs") else None,
    }


def _is_dask_backed(data):
    """
    Check if data is backed by a dask array.

    :param data: xarray DataArray, Dataset, or numpy-like array
    :return: True if dask-backed, False otherwise
    """
    # Handle xarray objects
    if hasattr(data, "data"):
        arr = data.data
    else:
        arr = data

    # Check for dask array - look for compute method and dask module
    if hasattr(arr, "compute") and hasattr(arr, "__module__"):
        module = getattr(arr, "__module__", "") or ""
        if module.startswith("dask"):
            return True

    # Alternative check using xarray's built-in if available
    try:
        from xarray.core.utils import is_duck_dask_array
        if hasattr(data, "data"):
            return is_duck_dask_array(data.data)
    except ImportError:
        pass

    return False


def _require_not_dask(*arrays):
    """
    Raise TypeError if any input is dask-backed.

    :param arrays: Variable number of arrays to check
    :raises TypeError: If any array is dask-backed
    """
    for arr in arrays:
        if _is_dask_backed(arr):
            raise TypeError(
                "Dask arrays are not supported; call .compute() first."
            )


# %% compute methods
def compute_air_density_correction(self, chunk_size=None, force: bool = False):
    """
    Compute air density correction factor rho based on elevation data.

    Contract:
    - Output is guaranteed to be on the same x/y grid as the Atlas template (if present),
      otherwise on the reference GWA raster grid.
    - CRS is enforced to self.parent.crs (Atlas CRS discipline).
    - Final assignment to self.data is exact-grid safe (prevents NaN stripes from implicit alignment).
    - Semantic caching: skips if existing result has matching signature.

    :param self: an instance of the _WindAtlas class
    :param chunk_size: size of chunks in pixels for chunked computation. If None, computation is not chunked
    :param force: if True, always recompute even if cached result exists
    """
    from cleo.loaders import load_elevation
    from pathlib import Path
    from rasterio.enums import Resampling
    from rasterio.crs import CRS

    # Semantic cache check
    algo = "compute_air_density_correction"
    ver = "1"
    params = _base_params(self)
    params["region"] = getattr(self.parent, "region", None) if self.parent else None
    expected_sig = _sig(algo, ver, params)

    if not force and _matches(self.data, "air_density_correction", expected_sig):
        logger.info(f"Air density correction for {self.parent.country} already computed (sig match), skipping.")
        return

    def rho_correction(elevation):
        """
        Compute air density correction factor based on elevation (meters a.s.l.)
        """
        rho_correction_factor = 1.247015 * np.exp(-0.000104 * elevation) / 1.225
        return rho_correction_factor.squeeze().rename("air_density_correction_factor")

    # Normalize path to Path object (handles str input)
    parent_path = Path(self.parent.path) if not isinstance(self.parent.path, Path) else self.parent.path

    # Use A_100 raster as reference for grid alignment (raw GWA raster)
    path_raw_country = parent_path / "data" / "raw" / f"{self.parent.country}"
    reference_file = path_raw_country / f"{self.parent.country}_combined-Weibull-A_100.tif"

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Raw GWA files missing: {reference_file}. "
            f"Run Atlas init/download first to fetch data for {self.parent.country}."
        )

    reference_da = rxr.open_rasterio(reference_file).squeeze()

    # Determine template grid to enforce exact alignment
    if "template" in self.data.data_vars:
        template = self.data["template"]
        # Hard fail if template CRS contradicts Atlas CRS discipline
        tpl_crs = template.rio.crs
        if tpl_crs is None:
            raise ValueError("template DataArray missing CRS (.rio.crs is None)")
        if not crs_equal(tpl_crs, self.parent.crs):
            raise ValueError(
                f"template CRS ({tpl_crs}) does not match Atlas CRS ({self.parent.crs}). "
                "Refuse to compute air_density_correction on inconsistent grids."
            )
    else:
        # Fallback: enforce Atlas CRS on the reference grid for a consistent output CRS
        ref_crs = reference_da.rio.crs
        if ref_crs is None:
            raise ValueError("reference GWA raster missing CRS (.rio.crs is None)")
        # Reproject to Atlas CRS if needed (semantic comparison)
        reference_da = reproject_raster_if_needed(reference_da, self.parent.crs, nodata=np.nan).squeeze()
        template = reference_da

    # Load elevation (legacy preferred; CopDEM fallback). Contract: must be made to match template.
    elevation = load_elevation(self.parent.path, self.parent.country, reference_da)

    src_crs = elevation.rio.crs
    if src_crs is None:
        raise ValueError("elevation DataArray missing CRS (.rio.crs is None)")

    # Enforce exact template grid + Atlas CRS with bilinear resampling (continuous elevation)
    elevation = elevation.rio.reproject_match(template, resampling=Resampling.bilinear, nodata=np.nan).squeeze()

    # Optional region clip AFTER reproject_match (avoids CRS mismatch NoDataInBounds)
    # Use drop=False to preserve exact x/y grid invariants
    if self.parent.region is not None:
        clip_shape = self.parent.get_nuts_region(self.parent.region)
        if clip_shape is None:
            raise ValueError(f"region={self.parent.region!r} produced no geometry from get_nuts_region()")
        clip_shape = to_crs_if_needed(clip_shape, elevation.rio.crs)
        geoms = list(clip_shape.geometry)

        # Import lazily for minimal envs (needed for error wrapping)
        try:
            from rioxarray.exceptions import NoDataInBounds
        except Exception:
            NoDataInBounds = ()  # fallback type, will never match

        try:
            elevation = _rio_clip_robust(elevation, geoms, drop=False, all_touched_primary=False)
        except NoDataInBounds as e:
            raise ValueError(
                f"Region clip failed: geometry does not overlap elevation bounds. "
                f"elevation.rio.crs={elevation.rio.crs}, clip_shape.total_bounds={clip_shape.total_bounds}"
            ) from e

    if chunk_size is None:
        rho = rho_correction(elevation)
    else:
        rho = compute_chunked(self, rho_correction, chunk_size, elevation=elevation)

    # Final exact-grid enforcement before storing (prevents silent alignment/NaN stripes)
    rho = rho.rio.reproject_match(template, resampling=Resampling.bilinear, nodata=np.nan).squeeze()

    # Set provenance and store
    rho = _set_prov(rho.rename("air_density_correction"), algo, ver, params)
    self._set_var("air_density_correction", rho)
    logger.info(f"Air density correction for {self.parent.country} computed.")


def compute_mean_wind_speed(self, height, chunk_size=None, inplace=True, force: bool = False):
    """
    Compute mean wind speed for a given height.

    Contract:
    - Semantic caching: skips if height slice already has matching signature.
    - Preserves dataset-level coords (never rebuild self.data from data_vars).
    - If template exists, all height slices are exact-grid matched to template and concat uses join="exact".
      Otherwise concat uses join="outer" (legacy/no-template mode).

    :param self: An instance of the Atlas-class
    :param height: Height for which to compute mean wind speed
    :type height: int
    :param chunk_size: number of chunks for chunked computation. If None, computation is not chunked.
    :type chunk_size: int
    :param inplace: If True, add mean wind speed to Dataset (kept for backward compatibility; function always updates self.data)
    :param force: if True, always recompute even if cached result exists
    """
    # Semantic cache check (per-height signature)
    algo = "compute_mean_wind_speed"
    ver = "1"
    params = _base_params(self)
    # Normalize height to int if whole number for consistent signatures (100.0 -> 100)
    params["height"] = int(height) if float(height) == int(height) else height
    expected_sig = _sig(algo, ver, params)

    # Check if this specific height slice already has matching signature
    if not force and "mean_wind_speed" in self.data:
        existing = self.data["mean_wind_speed"]
        if height in existing.coords["height"].values:
            slice_da = existing.sel(height=height)
            if _get_sig(slice_da) == expected_sig:
                logger.info(f"Mean wind speed at height '{height} m' already computed (sig match), skipping.")
                return

    def calculate_mean_wind_speed(weibull_a, weibull_k):
        mean_wind_speed = weibull_a * gamma(1 / weibull_k + 1)
        return mean_wind_speed.rename("mean_wind_speed").assign_attrs(units="m/s").squeeze()

    weibull_a, weibull_k = self.load_weibull_parameters(height)
    mean_wind_speed_data = (
        calculate_mean_wind_speed(weibull_a=weibull_a, weibull_k=weibull_k)
        if chunk_size is None
        else compute_chunked(self, calculate_mean_wind_speed, chunk_size, weibull_a=weibull_a, weibull_k=weibull_k)
    )

    # Align new slice to template grid if available
    has_template = "template" in self.data.data_vars
    if has_template:
        template = self.data["template"]
        mean_wind_speed_data = _match_to_template(mean_wind_speed_data, template)

    # Expand to 3D (no per-slice prov yet, will set per-height sig on combined array)
    mean_wind_speed_data = mean_wind_speed_data.expand_dims(height=[height])

    if "mean_wind_speed" not in self.data:
        # First height: create new array and set per-height provenance
        result = _set_height_prov(mean_wind_speed_data, height, algo, ver, params)
        self._set_var("mean_wind_speed", result)
        return

    existing = self.data["mean_wind_speed"]
    # Preserve existing sigs dict for later merge
    existing_sigs_json = existing.attrs.get("cleo:sigs")

    # If this height already exists, drop it first (we're replacing)
    if height in existing.coords["height"].values:
        existing = existing.sel(height=[h for h in existing.coords["height"].values if h != height])
        if existing.sizes["height"] == 0:
            # All heights were removed, just set the new one
            result = _set_height_prov(mean_wind_speed_data, height, algo, ver, params)
            self._set_var("mean_wind_speed", result)
            return

    # If template exists, align all existing slices to template before concatenation
    if has_template:
        template = self.data["template"]
        aligned_slices = []
        for h in existing.coords["height"].values:
            slice_2d = existing.sel(height=h).drop_vars("height")
            aligned_slice = _match_to_template(slice_2d, template)
            aligned_slices.append(aligned_slice.expand_dims(height=[h]))
        existing_aligned = xr.concat(aligned_slices, dim="height", join="exact")
        join_mode = "exact"
    else:
        existing_aligned = existing
        join_mode = "outer"

    combined = xr.concat([existing_aligned, mean_wind_speed_data], dim="height", join=join_mode)

    # Deterministic ordering and uniqueness
    combined = combined.sortby("height")

    # Enforce exact-grid before storing (if template exists)
    if has_template:
        from cleo.spatial import enforce_exact_grid
        combined = enforce_exact_grid(combined, self.data["template"], var_name="mean_wind_speed")

    # Restore existing sigs and add the new height's sig
    if existing_sigs_json:
        combined.attrs["cleo:sigs"] = existing_sigs_json
    combined = _set_height_prov(combined, height, algo, ver, params)

    # Update Dataset height coordinate safely:
    # drop the old variable first (it carries the old height dimension size), then set coord and reattach.
    ds = self.data.drop_vars("mean_wind_speed")
    ds = ds.assign_coords(height=combined.coords["height"])
    ds["mean_wind_speed"] = combined
    self.data = ds


def compute_wind_shear_coefficient(self, chunk_size=None, force: bool = False):
    """
    Compute wind shear coefficient

    :param self: An instance of the Atlas-class
    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :type chunk_size: int
    :param force: if True, always recompute even if cached result exists
    """
    # Semantic cache check
    algo = "compute_wind_shear_coefficient"
    ver = "1"
    params = _base_params(self)
    expected_sig = _sig(algo, ver, params)

    if not force and _matches(self.data, "wind_shear", expected_sig):
        logger.info(f"Wind shear coefficient already computed (sig match), skipping.")
        return

    # Ensure dependencies are computed (only call if method exists and data missing/invalid)
    if hasattr(self, "compute_mean_wind_speed"):
        if "mean_wind_speed" not in self.data.data_vars or 50 not in self.data["mean_wind_speed"].coords.get("height", []):
            self.compute_mean_wind_speed(50, chunk_size, force=force)
        if "mean_wind_speed" not in self.data.data_vars or 100 not in self.data["mean_wind_speed"].coords.get("height", []):
            self.compute_mean_wind_speed(100, chunk_size, force=force)

    u_mean_50 = self.data.mean_wind_speed.sel(height=50)
    u_mean_100 = self.data.mean_wind_speed.sel(height=100)

    # Dask arrays not supported - require eager computation
    _require_not_dask(u_mean_50, u_mean_100)

    # Mask invalid cells to avoid log(<=0) which produces inf/NaN
    valid = (u_mean_50 > 0) & (u_mean_100 > 0) & np.isfinite(u_mean_50) & np.isfinite(u_mean_100)

    # Suppress RuntimeWarning from log(0) - we handle invalid values via xr.where
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = xr.where(
            valid,
            (np.log(u_mean_100) - np.log(u_mean_50)) / (np.log(100) - np.log(50)),
            np.nan,
        )

    # Log warning if invalid cells exist (dask-safe scalar reduction)
    invalid_sum = (~valid).sum()
    if hasattr(invalid_sum.data, "compute"):
        invalid_sum = invalid_sum.compute()
    invalid_count = int(invalid_sum.values.item())
    if invalid_count > 0:
        total_count = int(valid.size)
        logger.warning(
            f"wind_shear: {invalid_count}/{total_count} cells masked (non-positive or non-finite wind speed)"
        )

    # Set provenance and store
    alpha = _set_prov(alpha.squeeze().rename("wind_shear"), algo, ver, params)
    self._set_var("wind_shear", alpha)
    logger.info(f'Wind shear coefficient for {self.parent.country} computed.')


def compute_weibull_pdf(self, chunk_size=None, force: bool = False):
    """
    Compute weibull probability density function for reference wind speeds

    :param self: an instance of the _WindAtlas class
    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :type chunk_size: int
    :param force: if True, always recompute even if cached result exists
    """
    # Semantic cache check
    algo = "compute_weibull_pdf"
    ver = "1"
    u = self.data.coords["wind_speed"].values
    params = _base_params(self)
    params["wind_speed_grid"] = list(float(v) for v in u)
    expected_sig = _sig(algo, ver, params)

    if not force and _matches(self.data, "weibull_pdf", expected_sig):
        logger.info(f"Weibull PDF already computed (sig match), skipping.")
        return

    a100, k100 = self.load_weibull_parameters(100)

    # Align to template grid if available
    if "template" in self.data.data_vars:
        tpl = self.data["template"]
        a100 = _match_to_template(a100, tpl)
        k100 = _match_to_template(k100, tpl)

    inputs = {
        "weibull_a": a100,
        "weibull_k": k100,
        "u_power_curve": u
    }

    if chunk_size is None:
        p = weibull_probability_density(**inputs)
    else:
        p = compute_chunked(self, weibull_probability_density, chunk_size, **inputs)

    # Align result to template if available
    if "template" in self.data.data_vars:
        tpl = self.data["template"]
        p = _match_to_template(p, tpl)

    # Set provenance and store
    p = _set_prov(p, algo, ver, params)
    self._set_var("weibull_pdf", p)
    logger.info(f'Weibull probability density function of wind speeds in {self.data.attrs["country"]} computed.')


# %% dependent methods
def simulate_capacity_factors(self, chunk_size=None, bias_correction=1, force: bool = False):
    """
    Simulate capacity factors for specified wind turbine models

    :param self: An instance of the Atlas-class
    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :type chunk_size: int
    :param bias_correction: Bias correction factor for simulated capacity factors.
    :type bias_correction: float
    :param force: if True, always recompute even if cached result exists
    """
    # Semantic cache check
    algo = "simulate_capacity_factors"
    ver = "1"
    turbines = sorted(str(t) for t in self.data.coords["turbine"].values)
    params = _base_params(self)
    params["bias_correction"] = bias_correction
    params["turbines"] = turbines
    expected_sig = _sig(algo, ver, params)

    if not force and _matches(self.data, "capacity_factors", expected_sig):
        logger.info(f"Capacity factors already computed (sig match), skipping.")
        return

    # Ensure dependencies are computed (only call if method exists and data missing)
    if hasattr(self, "compute_wind_shear_coefficient") and "wind_shear" not in self.data.data_vars:
        self.compute_wind_shear_coefficient(chunk_size, force=force)
    if hasattr(self, "compute_weibull_pdf") and "weibull_pdf" not in self.data.data_vars:
        self.compute_weibull_pdf(chunk_size, force=force)

    cap_factor = []

    for turbine_model in self.data.coords["turbine"].values:
        inputs = {
            "weibull_pdf": self.data["weibull_pdf"],
            "wind_shear": self.data["wind_shear"],
            "u_power_curve": self.data.coords["wind_speed"].values,
            "p_power_curve": self.data.power_curve.sel(turbine=turbine_model).values,
            "h_turbine": self.get_turbine_attribute(turbine_model, "hub_height"),
            "correction_factor": bias_correction
        }

        if chunk_size is None:
            cf = capacity_factor(**inputs)
        else:
            cf = compute_chunked(self, capacity_factor, chunk_size, **inputs)

        # cf = cf * self.data["air_density_correction"]
        cf = cf.expand_dims(turbine=[turbine_model])
        logger.info(f"Capacity factors for {turbine_model} in {self.data.attrs['country']} computed.")
        # cap_factor = xr.merge([cap_factor, cf.rename("capacity_factor")])
        cap_factor.append(cf)

    cap_factor = xr.concat(cap_factor, dim="turbine")
    # cap_factor = cap_factor.assign_coords(turbine=self.wind_turbines)
    cap_factor = cap_factor.rename("capacity_factor")

    # Set provenance and store (overwrite existing)
    cap_factor = _set_prov(cap_factor, algo, ver, params)
    self._set_var("capacity_factors", cap_factor)


def compute_lcoe(self, chunk_size=None, turbine_cost_share=1, force: bool = False):
    """
    Compute levelized cost of electricity (LCOE) for specified wind turbine models

    :param self: an instance of the Atlas-class
    :param chunk_size: Size of chunk in pixels. If None, computation is not chunked.
    :type chunk_size: int
    :param turbine_cost_share: Share of location-independent investment cost to compute pseudo-LCOE (see
    Wehrle et al. 2023, Inferring Local Social Cost of Wind Power. Evidence from Lower Austria)
    :type turbine_cost_share: float
    :param force: if True, always recompute even if cached result exists
    """
    # Semantic cache check
    algo = "compute_lcoe"
    ver = "1"
    turbines = sorted(str(t) for t in self.data.coords["turbine"].values)
    params = _base_params(self)
    params["turbine_cost_share"] = turbine_cost_share
    params["turbines"] = turbines
    expected_sig = _sig(algo, ver, params)

    if not force and _matches(self.data, "lcoe", expected_sig):
        logger.info(f"LCOE already computed (sig match), skipping.")
        return

    # Ensure dependencies are computed (only call if method exists and data missing)
    if hasattr(self, "simulate_capacity_factors") and "capacity_factors" not in self.data.data_vars:
        self.simulate_capacity_factors(chunk_size, force=force)

    lcoe_list = []
    for turbine_model in self.data.coords["turbine"].values:

        inputs = {
            "power": self.get_turbine_attribute(turbine_model, "capacity"),
            "capacity_factors": self.data["capacity_factors"].sel(turbine=turbine_model),
            "overnight_cost": self.get_overnight_cost(turbine_model) *
                              self.get_turbine_attribute(turbine_model, "capacity") * turbine_cost_share,
            "grid_cost": grid_connect_cost(self.get_turbine_attribute(turbine_model, "capacity")),
            "om_fixed": self.get_cost_assumptions("fixed_om_cost"),
            "om_variable": self.get_cost_assumptions("variable_om_cost"),
            "discount_rate": self.get_cost_assumptions("discount_rate"),
            "lifetime": self.get_cost_assumptions("turbine_lifetime")
        }

        if chunk_size is None:
            lcoe = levelized_cost(**inputs)
        else:
            lcoe = compute_chunked(self, levelized_cost, chunk_size, **inputs)

        lcoe = lcoe.expand_dims(turbine=[turbine_model])
        lcoe_list.append(lcoe)

    lcoe_combined = xr.concat(lcoe_list, dim='turbine').rename("lcoe").assign_attrs(units="EUR/MWh")

    # Set provenance and store
    lcoe_combined = _set_prov(lcoe_combined, algo, ver, params)
    self._set_var("lcoe", lcoe_combined)
    logger.info(f"Levelized Cost of Electricity in {self.data.attrs['country']} computed.")


def compute_optimal_power_energy(self, force: bool = False):
    """
    Compute optimal power and energy using least-cost turbine.

    :param force: if True, always recompute even if cached result exists
    """
    # Semantic cache check
    algo = "compute_optimal_power_energy"
    ver = "1"
    turbines = sorted(str(t) for t in self.data.coords["turbine"].values)
    params = _base_params(self)
    params["turbines"] = turbines
    expected_sig = _sig(algo, ver, params)

    if not force and _matches(self.data, "optimal_power", expected_sig):
        logger.info(f"Optimal power/energy already computed (sig match), skipping.")
        return

    # Ensure dependencies are computed (only call if method exists and data missing)
    if hasattr(self, "compute_lcoe") and "lcoe" not in self.data.data_vars:
        self.compute_lcoe(force=force)

    # Compute optimal power using turbine metadata (not string parsing)
    least_cost_turbine = self.data["lcoe"].idxmin(dim='turbine').compute()

    # Build lookup dict from turbine labels to capacity using metadata
    turbine_labels = self.data.coords["turbine"].values
    capacity_lookup = {
        label: self.get_turbine_attribute(label, "capacity")
        for label in turbine_labels
    }

    def get_capacity(turbine_label):
        if isinstance(turbine_label, str) and turbine_label in capacity_lookup:
            return float(capacity_lookup[turbine_label])
        else:
            return np.nan

    power = xr.apply_ufunc(get_capacity, least_cost_turbine, vectorize=True)

    # Set provenance and store
    power = _set_prov(power, algo, ver, params)
    self._set_var("optimal_power", power)

    # compute optimal energy
    least_cost_index = self.data["lcoe"].fillna(9999).argmin(dim='turbine').compute()
    energy = self.data["capacity_factors"].isel(turbine=least_cost_index).drop_vars("turbine")
    energy = energy.assign_coords({'turbine': "min_lcoe"})
    # power is in kW; CF is dimensionless; 8766 h/year -> kWh/year; divide by 1e6 -> GWh/year
    energy = (energy * power * 8766 / 10 ** 6).rename("optimal_energy").assign_attrs(units="GWh/a")
    energy = _set_prov(energy, algo, ver, params)
    self._set_var("optimal_energy", energy)


def minimum_lcoe(self, force: bool = False):
    """
    Calculate the minimum lcoe for each location among all turbines

    :param force: if True, always recompute even if cached result exists
    """
    # Semantic cache check
    algo = "minimum_lcoe"
    ver = "1"
    turbines = sorted(str(t) for t in self.data.coords["turbine"].values)
    params = _base_params(self)
    params["turbines"] = turbines
    expected_sig = _sig(algo, ver, params)

    if not force and _matches(self.data, "min_lcoe", expected_sig):
        logger.info(f"Minimum LCOE already computed (sig match), skipping.")
        return

    # Ensure dependencies are computed (only call if method exists and data missing)
    if hasattr(self, "compute_lcoe") and "lcoe" not in self.data.data_vars:
        self.compute_lcoe(force=force)

    lcoe = self.data["lcoe"]
    lc_min = lcoe.min(dim='turbine', keep_attrs=True)
    lc_min = lc_min.assign_coords({'turbine': 'min_lcoe'})

    # Set provenance and store
    lc_min = _set_prov(lc_min.rename("minimal_lcoe"), algo, ver, params)
    self._set_var("min_lcoe", lc_min)


# %% functions
def weibull_probability_density(u_power_curve, weibull_k, weibull_a):
    """
    Calculate Weibull probability density at wind-speed grid u_power_curve.

    Performance contract:
    - Vectorized / broadcasted computation (dask-friendly).
    - Produces dims ('wind_speed', 'y', 'x') (plus any extra non-spatial dims from inputs),
      with wind_speed coordinate exactly equal to u_power_curve.

    Mathematical definition (for u > 0):
        f(u) = (k/a) * (u/a)^(k-1) * exp(-(u/a)^k)

    At u = 0:
        f(0) := 0 (continuous Weibull; avoids inf when k < 1)

    :param u_power_curve: 1D array-like wind speed grid (exact labels contract)
    :param weibull_k: raster k parameter
    :param weibull_a: raster a parameter
    :return: DataArray with wind_speed dimension
    """
    u = np.asarray(u_power_curve)
    u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})

    # Broadcasting: u_da (wind_speed) over raster (y,x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = u_da / weibull_a
        pdf = (weibull_k / weibull_a) * (z ** (weibull_k - 1)) * np.exp(-(z ** weibull_k))

    # Exact u=0 handling (avoids inf for k<1); keep dtype float
    pdf = xr.where(u_da == 0, 0.0, pdf)

    pdf = pdf.transpose("wind_speed", ...)  # stable dim order
    return pdf.squeeze().rename("weibull_probability_density")


def capacity_factor(weibull_pdf, wind_shear, u_power_curve, p_power_curve, h_turbine, h_reference=100,
                    correction_factor=1):
    """
    calculates wind turbine capacity factors given Weibull probability density pdf, roughness factor alpha, wind turbine
    power curve data in u_power_curve and p_power_curve, turbine height h_turbine and reference height of wind speed
    modelling h_reference
    :param correction_factor:
    :param weibull_pdf: probability density function from weibull_probability_density()
    :param wind_shear: wind shear coefficient
    :param u_power_curve: power curve wind speed
    :param p_power_curve: power curve output
    :param h_turbine: hub height of wind turbine in m
    :param h_reference: reference height at which weibull pdf is computed
    :return:
    """
    # Ensure u and p are numpy arrays
    u = np.asarray(u_power_curve)
    p = np.asarray(p_power_curve)

    # Validate input length
    if len(u) < 2:
        raise ValueError(
            f"u_power_curve must have at least 2 points for integration, got {len(u)}"
        )
    if len(p) != len(u):
        raise ValueError(
            f"p_power_curve length ({len(p)}) must match u_power_curve length ({len(u)})"
        )

    # Compute shear scaling factor s(y,x) = (h_turbine / h_reference) ** wind_shear
    s = (h_turbine / h_reference) ** wind_shear.values

    # Get spatial shape from wind_shear
    spatial_shape = wind_shear.shape

    # Helper to evaluate power curve at hub-height wind speed (2D)
    def eval_power_curve(u_ref_scalar, s_2d):
        u_hub_2d = u_ref_scalar * s_2d
        flat = u_hub_2d.ravel()
        p_flat = np.interp(flat, u, p, left=0.0, right=0.0)
        return p_flat.reshape(spatial_shape)

    # Streaming trapezoidal integration over u_ref
    acc = np.zeros(spatial_shape, dtype=np.float64)

    # Exact wind_speed grid contract (no tolerant float matching).
    # Order may differ; alignment MUST be by label (prevents positional bugs).
    if "wind_speed" in weibull_pdf.dims:
        ws = np.asarray(weibull_pdf.coords["wind_speed"].values)

        # Require exact same labels (no tolerance), but allow different order.
        if ws.shape != u.shape or not np.array_equal(np.sort(ws), np.sort(u)):
            raise ValueError(
                "wind_speed grid mismatch: weibull_pdf.coords['wind_speed'] must contain exactly the same "
                "labels as u_power_curve (order can differ). "
                f"Got coords={ws!r} vs u_power_curve={u!r}."
            )

        # Reorder by label to match u_power_curve order (contract: label-based)
        weibull_pdf_aligned = weibull_pdf.sel(wind_speed=u)
        pdf_vals = weibull_pdf_aligned.transpose("wind_speed", ...).values
    else:
        pdf_vals = weibull_pdf.values

    n = len(u)
    for i in range(n - 1):
        du = u[i + 1] - u[i]
        pdf_i = pdf_vals[i]
        pdf_ip1 = pdf_vals[i + 1]
        pc_i = eval_power_curve(u[i], s)
        pc_ip1 = eval_power_curve(u[i + 1], s)
        term_i = pdf_i * pc_i
        term_ip1 = pdf_ip1 * pc_ip1
        acc += 0.5 * (term_i + term_ip1) * du

    # Apply correction factor and build result
    cap_factor = wind_shear.copy()
    cap_factor.values = acc * correction_factor
    cap_factor.name = "capacity_factor"
    return cap_factor


def turbine_overnight_cost(power, hub_height, rotor_diameter, year):
    """
    calculates wind turbine investment cost in EUR per MW based on >>Rinne et al. (2018): Effects of turbine technology
    and land use on wind power resource potential, Nature Energy<<
    :param power: rated power in MW
    :param hub_height: hub height in meters
    :param rotor_diameter: rotor diameter in meters
    :param year: year of first commercial deployment
    :return: overnight investment cost in EUR per kW
    """
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    spec_power = power * 10 ** 6 / rotor_area
    cost = ((620 * np.log(hub_height)) - (1.68 * spec_power) + (182 * (2016 - year) ** 0.5) - 1005)
    return cost.astype('float')


def grid_connect_cost(power):
    """
    Calculates grid connection cost according to §54 (3,4) ElWOG
    https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=20007045
    :param power: power in kW
    :return:
    """
    cost = 50 * power
    return cost


def levelized_cost(power, capacity_factors, overnight_cost, grid_cost, om_fixed, om_variable, discount_rate, lifetime,
                   hours_per_year=8766, per_mwh=True):
    """
    Calculates wind turbines' levelized cost of electricity in EUR per MWh
    :param per_mwh: Returns LCOE in currency per megawatt hour if true (default). Else returns LCOE in currency per kwh.
    :type per_mwh: bool
    :param hours_per_year: Number of hours per year. Default is 8766 to account for leap years.
    :param power: rated power in kW
    :type power: float
    :param capacity_factors: wind turbine capacity factor (share of year)
    :type capacity_factors: xarray.DataArray
    :param overnight_cost: absolute overnight cost in EUR (lump sum for this turbine at this location)
    :type overnight_cost: float
    :param grid_cost: cost for connecting to the electricity grid
    :type grid_cost: xarray.DataArray
    :param om_fixed: EUR/kW
    :type om_fixed: float
    :param om_variable: EUR/kWh
    :type om_variable: float
    :param discount_rate: percent
    :type discount_rate: float
    :param lifetime: years
    :type lifetime: int
    :return: lcoe in EUR/kWh
    """

    def discount_factor(discount_rate, period):
        """
        Calculate the discount factor for a given discount rate and period.
        :param discount_rate: discount rate (fraction of 1)
        :type discount_rate: float
        :param period: Number of years
        :type period: int
        :return: Discount factor
        :rtype: float
        """
        if discount_rate == 0:
            return float(period)
        return (1 - (1 + discount_rate) ** (-period)) / discount_rate

    npv_factor = discount_factor(discount_rate, lifetime)

    # calculate net present amount of electricity generated over lifetime
    npv_electricity = capacity_factors * hours_per_year * power * npv_factor

    # calculate net present value of cost
    npv_cost = (om_variable * capacity_factors * hours_per_year + om_fixed) * power * npv_factor
    npv_cost = npv_cost + overnight_cost + grid_cost

    lcoe = npv_cost / npv_electricity

    if per_mwh:
        return lcoe * 1000
    else:
        return lcoe
