# resource assessment methods of the Atlas class
import hashlib
import json
import numpy as np
import xarray as xr
# rioxarray not imported here - assess.py is pure compute (no raw I/O)
import logging

from scipy.special import gamma
from cleo.utils import _match_to_template
from cleo.chunk import compute_chunked
from cleo.spatial import _validate_values

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


# %% compute methods


def compute_air_density_correction_core(
    *,
    elevation: xr.DataArray,
    template: xr.DataArray,
) -> xr.DataArray:
    """
    Pure compute function: air density correction factor based on elevation.

    This is a stateless, I/O-free computation primitive. The caller is responsible
    for providing elevation and template DataArrays that are already aligned to the
    canonical grid.

    Formula: rho_correction = 1.247015 * exp(-0.000104 * elevation) / 1.225

    Contract:
    - elevation and template MUST have identical y/x coords (caller must verify alignment).
    - Output is NaN where elevation is NaN (propagates nodata).
    - Remains lazy: never forces eager evaluation (compute/load/values).
    - Returns a DataArray named "air_density_correction" on the same coords as elevation.

    :param elevation: Elevation DataArray (meters a.s.l.) on canonical grid.
    :param template: Template DataArray for output coords/dims alignment.
    :return: DataArray "air_density_correction" factor, lazy if input is lazy.
    """
    # Core formula: barometric-like correction
    rho_correction_factor = 1.247015 * np.exp(-0.000104 * elevation) / 1.225

    # Ensure output has correct name and preserves coords from elevation
    result = rho_correction_factor.rename("air_density_correction")

    # Squeeze any singleton dimensions (e.g., band) but keep y, x
    if "band" in result.dims:
        result = result.squeeze("band", drop=True)

    return result


def mean_wind_speed_from_weibull(
    *,
    A: xr.DataArray,
    k: xr.DataArray,
) -> xr.DataArray:
    """
    Compute mean wind speed from Weibull A and k parameters.

    Pure compute primitive: no I/O, dask-friendly, returns same shape as inputs.

    Mathematical definition:
        mean_wind_speed = A * gamma(1 + 1/k)

    :param A: Weibull A (scale) parameter, DataArray with dims (y, x) or (height, y, x)
    :param k: Weibull k (shape) parameter, DataArray with same dims as A
    :return: DataArray with mean wind speed (m/s), same dims as inputs
    """
    mean_ws = A * gamma(1 / k + 1)
    return mean_ws.rename("mean_wind_speed").assign_attrs(units="m/s")


def compute_mean_wind_speed(self, height, chunk_size=None, inplace=True, force: bool = False):
    """
    Compute mean wind speed for a given height.

    Contract:
    - Semantic caching: skips if height slice already has matching signature.
    - Preserves dataset-level coords (never rebuild self.data from data_vars).
    - If template exists, all height slices are exact-grid matched to template and concat uses join="exact".
      Otherwise concat uses join="outer" (no-template mode).

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
    if chunk_size is None or _is_dask_backed(weibull_a):
        # Dask-backed: execute directly (dask handles chunking internally)
        mean_wind_speed_data = calculate_mean_wind_speed(weibull_a=weibull_a, weibull_k=weibull_k)
    else:
        mean_wind_speed_data = compute_chunked(self, calculate_mean_wind_speed, chunk_size, weibull_a=weibull_a, weibull_k=weibull_k)

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

    # Note: operations below are dask-compatible (comparisons, np.log, xr.where all work lazily)

    # Mask invalid cells to avoid log(<=0) which produces inf/NaN
    valid = (u_mean_50 > 0) & (u_mean_100 > 0) & np.isfinite(u_mean_50) & np.isfinite(u_mean_100)

    # Suppress RuntimeWarning from log(0) - we handle invalid values via xr.where
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = xr.where(
            valid,
            (np.log(u_mean_100) - np.log(u_mean_50)) / (np.log(100) - np.log(50)),
            np.nan,
        )

    # [VALIDATION BRANCH] Log warning if invalid cells exist (eager mode only).
    # For dask-backed arrays, skip count to avoid triggering full compute.
    if not _is_dask_backed(valid):
        invalid_sum = (~valid).sum()
        if bool(invalid_sum > 0):
            invalid_count = int(invalid_sum)
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

    :param self: wind atlas instance with data property
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

    if chunk_size is None or _is_dask_backed(a100):
        # Dask-backed: execute directly (dask handles chunking internally)
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
# TODO(rotor-eq-cf, REWS: Rotor Equivalent Wind Speed):
# Add an OPTIONAL rotor-equivalent mode to capacity factor simulation that stays consistent with
# the existing hub-height Weibull workflow and avoids introducing a separate shear exponent alpha.
#
# Idea (minimal & pythonic):
# - We already have GWA Weibull parameters A(z), k(z) across multiple heights (via existing hub-height
#   interpolation utilities). Use those directly to compute a rotor-equivalent cubic-moment factor f.
#
# Definitions:
#   U(z) ~ Weibull(A(z), k(z))
#   m3(z) = E[U(z)^3] = A(z)^3 * Gamma(1 + 3/k(z))
#   weights over rotor disk (vertical slice): w(y) = 2 * sqrt(R^2 - y^2),  y in [-R, R]
#   z(y) = z_hub + y
#   m3_rotor = (∫_{-R}^{R} m3(z(y)) * w(y) dy) / (∫_{-R}^{R} w(y) dy)
#   ueq_rotor = m3_rotor^(1/3)
#   ueq_hub   = m3(z_hub)^(1/3)
#   f = ueq_rotor / ueq_hub
#
# Implementation sketch:
# - New simulate_capacity_factors(...) parameter, e.g. rotor_mode="hub"|"rotor_eq_moment".
# - When rotor_mode=="rotor_eq_moment":
#     1) Ensure we can evaluate A(z), k(z) at arbitrary z near hub height (use existing A/k interpolation).
#     2) Numerically integrate m3(z(y)) across y in [-R, R] using quadrature points y_i and weights w_i.
#        Choose a small fixed N (e.g. 200) for determinism.
#     3) Compute f = ueq_rotor/ueq_hub and apply a SIMPLE scale adjustment:
#           A_hub_eff = A_hub * f
#        Keep k_hub unchanged.
#     4) Continue with the existing Weibull PDF + turbine power-curve integration exactly as in hub mode.
# - Important constraints:
#     * No network IO / downloads in compute paths.
#     * Do not introduce global DAG/pipeline; keep prerequisite logic local and explicit.
#     * If turbine metadata lacks rotor diameter / hub height, raise a clear ValueError.
#
# Tests (oracle-first):
# - alpha-free: purely based on A(z), k(z).
# - Property checks:
#     * If A(z),k(z) are constant in z => f == 1 (rotor == hub).
#     * Monotonic sanity: if A(z) increases with z (k constant), then f > 1.
# - Integration test: rotor_mode changes CF relative to hub_mode in expected direction on a toy case.

def simulate_capacity_factors(self, chunk_size=None, loss_factor=1, force: bool = False,
                             weibull_height_mode: str = "hub", air_density_mode: str = "gwa"):
    """
    Simulate wind turbine capacity factors for all configured turbines.

    Modes:
      - weibull_height_mode="hub" (default):
          Interpolate Weibull A/k to hub height and integrate the power curve against the hub-height PDF.

      - weibull_height_mode="hub_rews" (opt-in):
          Computes the standard hub-height capacity factors (stored as self.data["capacity_factors"]) AND
          additionally computes rotor-equivalent (REWS) capacity factors (stored as self.data["capacity_factors_rews"]).
          REWS is computed via a cubic-moment rotor-area average using multi-height Weibull A/k.

      - weibull_height_mode="100m_shear" (legacy):
          Uses the 100m Weibull PDF plus a wind-shear exponent alpha to scale wind speeds to hub height.

    Air density:
      - air_density_mode="none": no density correction
      - air_density_mode="gwa": apply density correction inside the power curve via u_eq = u * (rho/rho0)^(1/3)
        (only supported for hub / hub_rews modes)

    Contract:
      - Function signature and return (None) are stable.
      - Existing variable self.data["capacity_factors"] retains its meaning (hub-height CF or legacy 100m_shear CF).
      - In hub_rews mode, an additional variable self.data["capacity_factors_rews"] is created/updated.

    :param chunk_size: Size of chunk in pixels. If None, computation is not chunked.
    :type chunk_size: int
    :param loss_factor: multiplicative loss / correction factor applied to CF (use 1.0 for gross CF)
    :type loss_factor: float
    :param force: if True, always recompute even if cached result exists
    :param weibull_height_mode: "hub", "hub_rews", or "100m_shear"
    :param air_density_mode: "none" or "gwa"
    """
    if weibull_height_mode not in ("hub", "hub_rews", "100m_shear"):
        raise ValueError(
            "weibull_height_mode must be 'hub', 'hub_rews' or '100m_shear', "
            f"got {weibull_height_mode!r}"
        )
    if air_density_mode not in ("none", "gwa"):
        raise ValueError(f"air_density_mode must be 'none' or 'gwa', got {air_density_mode!r}")

    # Density correction is only supported for hub-based modes
    hub_like_mode = weibull_height_mode in ("hub", "hub_rews")
    if air_density_mode == "gwa" and not hub_like_mode:
        raise ValueError("air_density_mode='gwa' is only supported with weibull_height_mode='hub' or 'hub_rews'")

    turbines = sorted(str(t) for t in self.data.coords["turbine"].values)

    # --- semantic cache checks (hub CF) ---
    algo_hub = "simulate_capacity_factors"
    ver_hub = "4"  # bump: hub mode now reuses stacked A/k and supports hub_rews without changing hub-CF meaning

    params_hub = _base_params(self)
    params_hub["loss_factor"] = float(loss_factor)
    params_hub["turbines"] = turbines
    params_hub["weibull_height_mode"] = "hub" if hub_like_mode else "100m_shear"
    params_hub["air_density_mode"] = air_density_mode
    expected_sig_hub = _sig(algo_hub, ver_hub, params_hub)

    hub_cached = (not force) and _matches(self.data, "capacity_factors", expected_sig_hub)

    # --- semantic cache checks (REWS CF) ---
    do_rews = (weibull_height_mode == "hub_rews")
    algo_rews = "simulate_capacity_factors_rews"
    ver_rews = "1"
    rews_n = 9  # Chebyshev-2 quadrature nodes (performance/accuracy sweet spot)

    if do_rews:
        params_rews = _base_params(self)
        params_rews["loss_factor"] = float(loss_factor)
        params_rews["turbines"] = turbines
        params_rews["weibull_height_mode"] = "hub_rews"
        params_rews["air_density_mode"] = air_density_mode
        params_rews["rews_quadrature_n"] = int(rews_n)
        expected_sig_rews = _sig(algo_rews, ver_rews, params_rews)
        rews_cached = (not force) and _matches(self.data, "capacity_factors_rews", expected_sig_rews)
    else:
        rews_cached = True

    if hub_cached and rews_cached:
        logger.info("Capacity factors already computed (sig match), skipping.")
        return

    # --- dependencies for legacy 100m_shear ---
    if not hub_like_mode and not hub_cached:
        if hasattr(self, "compute_wind_shear_coefficient") and "wind_shear" not in self.data.data_vars:
            self.compute_wind_shear_coefficient(chunk_size, force=force)
        if hasattr(self, "compute_weibull_pdf") and "weibull_pdf" not in self.data.data_vars:
            self.compute_weibull_pdf(chunk_size, force=force)

    # --- hub-like modes: load Weibull stacks once ---
    if hub_like_mode and (not hub_cached or (do_rews and not rews_cached)):
        A_stack, k_stack, _available_heights = _load_weibull_param_stacks(self)

    # Lazy caches within this call
    ak_cache: dict[float, tuple[xr.DataArray, xr.DataArray]] = {}
    pdf_cache: dict[float, xr.DataArray] = {}
    rho_cache: dict[float, xr.DataArray] = {}
    rews_cache: dict[tuple[float, float, int], xr.DataArray] = {}

    dummy_shear = None

    cap_factor_hub: list[xr.DataArray] = []
    cap_factor_rews: list[xr.DataArray] = []

    for turbine_id in turbines:
        u_power_curve = self.data.coords["wind_speed"].data
        p_power_curve = self.data["power_curve"].sel(turbine=turbine_id).data

        h_turbine = float(self.get_turbine_attribute(turbine_id, "hub_height"))

        # --- HUB CF (standard) ---
        if not hub_cached:
            if hub_like_mode:
                h_key = float(h_turbine)

                # Interpolate A/k at hub and build hub PDF (cached per height)
                if h_key not in ak_cache:
                    A_hub, k_hub = interpolate_weibull_params_to_height(A_stack, k_stack, h_key)
                    # Align to template if available
                    if "template" in self.data.data_vars:
                        tpl = self.data["template"]
                        A_hub = _match_to_template(A_hub, tpl)
                        k_hub = _match_to_template(k_hub, tpl)
                    ak_cache[h_key] = (A_hub, k_hub)
                else:
                    A_hub, k_hub = ak_cache[h_key]

                if h_key not in pdf_cache:
                    u = self.data.coords["wind_speed"].values
                    pdf_hub = weibull_probability_density(u_power_curve=u, weibull_k=k_hub, weibull_a=A_hub)
                    if "template" in self.data.data_vars:
                        pdf_hub = _match_to_template(pdf_hub, self.data["template"])
                    pdf_cache[h_key] = pdf_hub
                else:
                    pdf_hub = pdf_cache[h_key]

                if air_density_mode == "gwa":
                    if h_key not in rho_cache:
                        rho_hub = compute_air_density_at_height(self, h_key)
                        rho_cache[h_key] = rho_hub
                    else:
                        rho_hub = rho_cache[h_key]

                    c = (rho_hub / RHO_0) ** (1 / 3)
                    cf = _integrate_cf_with_density_correction(
                        pdf=pdf_hub,
                        u_grid=u_power_curve,
                        p_curve=p_power_curve,
                        c=c,
                        loss_factor=loss_factor,
                    )
                else:
                    # No density correction: reuse capacity_factor with a dummy shear array (s == 1)
                    if dummy_shear is None:
                        template = self.data["template"] if "template" in self.data.data_vars else pdf_hub.isel(wind_speed=0)
                        dummy_shear = xr.zeros_like(template).rename("wind_shear")

                    inputs = {
                        "weibull_pdf": pdf_hub,
                        "wind_shear": dummy_shear,
                        "u_power_curve": u_power_curve,
                        "p_power_curve": p_power_curve,
                        "h_turbine": h_turbine,
                        "h_reference": h_turbine,
                        "correction_factor": loss_factor,
                    }
                    if chunk_size is None or _is_dask_backed(pdf_hub):
                        cf = capacity_factor(**inputs)
                    else:
                        cf = compute_chunked(self, capacity_factor, chunk_size, **inputs)

            else:
                # Legacy 100m_shear mode
                inputs = {
                    "weibull_pdf": self.data["weibull_pdf"],
                    "wind_shear": self.data["wind_shear"],
                    "u_power_curve": u_power_curve,
                    "p_power_curve": p_power_curve,
                    "h_turbine": h_turbine,
                    "correction_factor": loss_factor,
                }
                if chunk_size is None or _is_dask_backed(self.data["weibull_pdf"]):
                    cf = capacity_factor(**inputs)
                else:
                    cf = compute_chunked(self, capacity_factor, chunk_size, **inputs)

            cf = cf.expand_dims(turbine=[turbine_id])
            cap_factor_hub.append(cf)

        # --- REWS CF (opt-in) ---
        if do_rews and not rews_cached:
            rotor_diameter = self.get_turbine_attribute(turbine_id, "rotor_diameter")
            if rotor_diameter is None:
                raise ValueError(
                    f"hub_rews requested but turbine {turbine_id!r} has no 'rotor_diameter' attribute."
                )
            rotor_diameter = float(rotor_diameter)

            h_key = float(h_turbine)
            if h_key not in ak_cache:
                A_hub, k_hub = interpolate_weibull_params_to_height(A_stack, k_stack, h_key)
                if "template" in self.data.data_vars:
                    tpl = self.data["template"]
                    A_hub = _match_to_template(A_hub, tpl)
                    k_hub = _match_to_template(k_hub, tpl)
                ak_cache[h_key] = (A_hub, k_hub)
            else:
                A_hub, k_hub = ak_cache[h_key]

            rews_key = (h_key, float(rotor_diameter), int(rews_n))
            if rews_key not in rews_cache:
                f_rews = _rews_moment_factor(
                    A_stack=A_stack,
                    k_stack=k_stack,
                    hub_height=h_key,
                    rotor_diameter=rotor_diameter,
                    n=rews_n,
                )
                if "template" in self.data.data_vars:
                    f_rews = _match_to_template(f_rews, self.data["template"])
                rews_cache[rews_key] = f_rews
            else:
                f_rews = rews_cache[rews_key]

            A_eff = (A_hub * f_rews).rename("weibull_A_rews")

            u = self.data.coords["wind_speed"].values
            pdf_rews = weibull_probability_density(u_power_curve=u, weibull_k=k_hub, weibull_a=A_eff)
            if "template" in self.data.data_vars:
                pdf_rews = _match_to_template(pdf_rews, self.data["template"])

            if air_density_mode == "gwa":
                if h_key not in rho_cache:
                    rho_hub = compute_air_density_at_height(self, h_key)
                    rho_cache[h_key] = rho_hub
                else:
                    rho_hub = rho_cache[h_key]
                c = (rho_hub / RHO_0) ** (1 / 3)
                cf_rews = _integrate_cf_with_density_correction(
                    pdf=pdf_rews,
                    u_grid=u_power_curve,
                    p_curve=p_power_curve,
                    c=c,
                    loss_factor=loss_factor,
                )
            else:
                if dummy_shear is None:
                    template = self.data["template"] if "template" in self.data.data_vars else pdf_rews.isel(wind_speed=0)
                    dummy_shear = xr.zeros_like(template).rename("wind_shear")

                inputs_rews = {
                    "weibull_pdf": pdf_rews,
                    "wind_shear": dummy_shear,
                    "u_power_curve": u_power_curve,
                    "p_power_curve": p_power_curve,
                    "h_turbine": h_turbine,
                    "h_reference": h_turbine,
                    "correction_factor": loss_factor,
                }
                if chunk_size is None or _is_dask_backed(pdf_rews):
                    cf_rews = capacity_factor(**inputs_rews)
                else:
                    cf_rews = compute_chunked(self, capacity_factor, chunk_size, **inputs_rews)

            cf_rews = cf_rews.expand_dims(turbine=[turbine_id])
            cap_factor_rews.append(cf_rews)

    # --- store hub CF if computed ---
    if not hub_cached:
        cap_factor = xr.concat(cap_factor_hub, dim="turbine").rename("capacity_factor")
        cap_factor = _set_prov(cap_factor, algo_hub, ver_hub, params_hub)
        self._set_var("capacity_factors", cap_factor)
        logger.info(f"Capacity factors in {self.data.attrs.get('country', '?')} computed.")

    # --- store REWS CF if computed ---
    if do_rews and not rews_cached:
        cap_factor_r = xr.concat(cap_factor_rews, dim="turbine").rename("capacity_factor_rews")
        cap_factor_r = _set_prov(cap_factor_r, algo_rews, ver_rews, params_rews)
        self._set_var("capacity_factors_rews", cap_factor_r)
        logger.info(f"REWS capacity factors in {self.data.attrs.get('country', '?')} computed.")

def _interp_power_curve(u_eq: xr.DataArray, u: np.ndarray, p: np.ndarray) -> xr.DataArray:
    """
    Interpolate power curve at equivalent wind speeds (dask-friendly).

    Uses xr.apply_ufunc to wrap np.interp with dask="parallelized".

    :param u_eq: Equivalent wind speeds, any shape (dask-backed OK)
    :param u: 1D wind speed grid (numpy)
    :param p: 1D power curve values (numpy)
    :return: Power curve values at u_eq, same shape as u_eq
    """
    return xr.apply_ufunc(
        lambda x: np.interp(x, u, p, left=0.0, right=0.0),
        u_eq,
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _trapz_over_wind_speed(y: xr.DataArray, x: xr.DataArray) -> xr.DataArray:
    """
    Trapezoidal integration over the wind_speed dimension (dask-friendly).

    Uses np.trapezoid via xr.apply_ufunc with dask="parallelized".

    :param y: Integrand with dims ("wind_speed", ...spatial...), dask-backed OK
    :param x: 1D wind_speed coordinate DataArray (same for all pixels)
    :return: Integrated values with wind_speed dim removed
    """
    return xr.apply_ufunc(
        np.trapezoid,
        y, x,
        input_core_dims=[["wind_speed"], ["wind_speed"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float64],
    )


def _integrate_cf_with_density_correction(
    pdf: xr.DataArray,
    u_grid: np.ndarray,
    p_curve: np.ndarray,
    c: xr.DataArray,
    loss_factor: float = 1.0,
) -> xr.DataArray:
    """
    Integrate capacity factor with air density correction.

    CF = loss_factor * ∫ PC(u * c) * pdf(u) du

    The correction factor c = (rho/rho0)^(1/3) scales wind speed to equivalent
    wind speed at reference density, applying the correction INSIDE the power
    curve evaluation, NOT as a post-hoc CF multiplier.

    This implementation is fully vectorized and dask-friendly (no Python loops,
    no eager array evaluation). Uses np.trapezoid via xr.apply_ufunc.

    :param pdf: Weibull PDF at hub height, dims (wind_speed, y, x)
    :param u_grid: 1D wind speed grid
    :param p_curve: 1D power curve values (same length as u_grid)
    :param c: Density correction factor (rho/rho0)^(1/3), dims (y, x)
    :param loss_factor: Additional correction factor
    :return: Capacity factor DataArray with dims (y, x)
    """
    u = np.asarray(u_grid, dtype=np.float64)
    p = np.asarray(p_curve, dtype=np.float64)

    # Align PDF to wind_speed order
    if "wind_speed" in pdf.dims:
        pdf_aligned = pdf.sel(wind_speed=u).transpose("wind_speed", ...)
    else:
        pdf_aligned = pdf

    # Build wind_speed as 1D DataArray for vectorized operations
    u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})

    # Compute equivalent wind speed: u_eq = u * c
    # Broadcasting: u_da (wind_speed,) * c (y, x) => (wind_speed, y, x)
    u_eq = u_da * c

    # Evaluate power curve at all equivalent wind speeds (vectorized, dask-friendly)
    p_eq = _interp_power_curve(u_eq, u, p)

    # Build integrand: pdf(u) * PC(u*c)
    # Both have dims (wind_speed, y, x) after alignment
    integrand = pdf_aligned * p_eq

    # Vectorized trapezoidal integration over wind_speed (dask-friendly)
    cf = _trapz_over_wind_speed(integrand, u_da)

    # Apply loss factor and set name
    result = cf * loss_factor
    result.name = "capacity_factor"
    return result


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
    for turbine_id in self.data.coords["turbine"].values:

        inputs = {
            "power": self.get_turbine_attribute(turbine_id, "capacity"),
            "capacity_factors": self.data["capacity_factors"].sel(turbine=turbine_id),
            "overnight_cost": self.get_overnight_cost(turbine_id) *
                              self.get_turbine_attribute(turbine_id, "capacity") * turbine_cost_share,
            "grid_cost": grid_connect_cost(self.get_turbine_attribute(turbine_id, "capacity")),
            "om_fixed": self.get_cost_assumptions("fixed_om_cost"),
            "om_variable": self.get_cost_assumptions("variable_om_cost"),
            "discount_rate": self.get_cost_assumptions("discount_rate"),
            "lifetime": self.get_cost_assumptions("turbine_lifetime")
        }

        if chunk_size is None or _is_dask_backed(self.data["capacity_factors"]):
            # Dask-backed: execute directly (dask handles chunking internally)
            lcoe = levelized_cost(**inputs)
        else:
            lcoe = compute_chunked(self, levelized_cost, chunk_size, **inputs)

        lcoe = lcoe.expand_dims(turbine=[turbine_id])
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
    # Keep lazy - apply_ufunc with dask="parallelized" handles dask arrays
    least_cost_turbine = self.data["lcoe"].idxmin(dim='turbine')

    # Build lookup dict from turbine labels to capacity using metadata
    turbine_labels = self.data.coords["turbine"].values  # coords-only .values OK
    capacity_lookup = {
        label: self.get_turbine_attribute(label, "capacity")
        for label in turbine_labels
    }

    def get_capacity(turbine_label):
        if isinstance(turbine_label, str) and turbine_label in capacity_lookup:
            return float(capacity_lookup[turbine_label])
        else:
            return np.nan

    power = xr.apply_ufunc(
        get_capacity, least_cost_turbine,
        vectorize=True, dask="parallelized", output_dtypes=[np.float64]
    )

    # Set provenance and store
    power = _set_prov(power, algo, ver, params)
    self._set_var("optimal_power", power)

    # Compute optimal energy using xr.where() to stay lazy
    # Select capacity_factors where lcoe == min(lcoe) for each (y, x)
    lcoe_filled = self.data["lcoe"].fillna(9999)
    min_lcoe = lcoe_filled.min(dim='turbine')
    is_min_lcoe = lcoe_filled == min_lcoe
    # Use where + max to select CF at min-LCOE turbine (max collapses dim, only one True per location)
    energy = self.data["capacity_factors"].where(is_min_lcoe).max(dim='turbine')
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
def _interp_da_to_height_log(
    da: xr.DataArray,
    target_height: float | int,
) -> xr.DataArray:
    """
    Interpolate a DataArray with "height" dimension to a target height using log-height-linear interpolation.

    Internal helper for height interpolation. Uses x = ln(height) space.
    No extrapolation: target_height must be within [min(height), max(height)].

    This implementation is xarray-native and dask-friendly (no Python loops, no eager evaluation).

    :param da: DataArray with dim "height" (e.g., Weibull A/k at multiple heights)
    :param target_height: Target height for interpolation
    :return: DataArray without "height" dim, with coord hub_height=target_height
    :raises ValueError: If target_height is outside available height range or insufficient heights
    """
    # Validate height dimension exists
    if "height" not in da.dims:
        raise ValueError("DataArray must have a 'height' dimension")

    heights = da.coords["height"].values
    if len(heights) < 2:
        raise ValueError(f"At least 2 heights required for interpolation, got {len(heights)}")

    # No extrapolation: check bounds
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    if target_height < h_min or target_height > h_max:
        raise ValueError(
            f"target_height={target_height} is outside available height range [{h_min}, {h_max}]. "
            "Extrapolation is not supported."
        )

    # Create ln_height coordinate and transform to log-space
    ln_heights = np.log(np.asarray(heights, dtype=np.float64))
    da_log = da.assign_coords(ln_height=("height", ln_heights))

    # Swap dims to use ln_height, sort, and interpolate
    da_log = da_log.swap_dims({"height": "ln_height"})
    da_log = da_log.sortby("ln_height")

    # Interpolate in log-height space (xarray-native, dask-friendly)
    ln_target = np.log(float(target_height))
    result = da_log.interp(ln_height=ln_target, method="linear")

    # Drop the ln_height coordinate (now scalar after interp)
    if "ln_height" in result.coords:
        result = result.drop_vars("ln_height")
    if "height" in result.coords:
        result = result.drop_vars("height")

    # Assign hub_height coordinate
    result = result.assign_coords(hub_height=target_height)
    result.name = da.name

    return result


def interpolate_weibull_params_to_height(
    weibull_A: xr.DataArray,
    weibull_k: xr.DataArray,
    target_height: float | int,
    *,
    method: str = "log_height_linear",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Interpolate Weibull A and k parameters to an arbitrary target height.

    Uses log-height-linear interpolation: x = ln(height).
    For each cell: A* = interp(x*, x_i, A_i), k* = interp(x*, x_i, k_i).

    :param weibull_A: DataArray with dim "height" containing A parameter at multiple heights
    :param weibull_k: DataArray with dim "height" containing k parameter at multiple heights
    :param target_height: Target height for interpolation (e.g., turbine hub height)
    :param method: Interpolation method. Only "log_height_linear" supported in v1.
    :return: Tuple (A_hub, k_hub) as DataArrays without "height" dim, with coord hub_height=target_height
    :raises ValueError: If method is not supported, or if target_height is outside available height range
    """
    if method != "log_height_linear":
        raise ValueError(f"Unsupported interpolation method: {method!r}. Only 'log_height_linear' is supported.")

    # Validate height dimension exists in both
    if "height" not in weibull_A.dims or "height" not in weibull_k.dims:
        raise ValueError("weibull_A and weibull_k must have a 'height' dimension")

    # Use shared interpolation helper
    A_hub = _interp_da_to_height_log(weibull_A, target_height)
    A_hub.name = "weibull_A"

    k_hub = _interp_da_to_height_log(weibull_k, target_height)
    k_hub.name = "weibull_k"

    return A_hub, k_hub


# Standard GWA heights for multi-height Weibull data
GWA_HEIGHTS = (10, 50, 100, 150, 200)

# Reference air density at sea level (kg/m³)
RHO_0 = 1.225


def _load_weibull_param_stacks(
    self,
    *,
    heights: tuple[int, ...] = GWA_HEIGHTS,
) -> tuple[xr.DataArray, xr.DataArray, list[float]]:
    """
    Load Weibull A/k rasters at multiple heights and stack them into (height, y, x) arrays.

    - Skips heights whose rasters are missing (FileNotFoundError).
    - Aligns each slice to the dataset template if present.
    - Requires at least 2 heights for interpolation.

    Returns:
      (A_stack, k_stack, available_heights)
    """
    A_list: list[xr.DataArray] = []
    k_list: list[xr.DataArray] = []
    available: list[float] = []

    tpl = self.data["template"] if "template" in self.data.data_vars else None

    for h in heights:
        try:
            a_h, k_h = self.load_weibull_parameters(h)
        except FileNotFoundError:
            continue

        # Align to template grid if available
        if tpl is not None:
            a_h = _match_to_template(a_h, tpl)
            k_h = _match_to_template(k_h, tpl)

        A_list.append(a_h)
        k_list.append(k_h)
        available.append(float(h))

    if len(available) < 2:
        raise ValueError(
            f"At least 2 Weibull height levels required, but only found: {available}"
        )

    # Stack into (height, ...) arrays
    A_stack = xr.concat(A_list, dim="height").assign_coords(height=available)
    k_stack = xr.concat(k_list, dim="height").assign_coords(height=available)
    A_stack.name = "weibull_A"
    k_stack.name = "weibull_k"
    return A_stack, k_stack, available


def _interp_da_to_heights_log(
    da: xr.DataArray,
    target_heights: np.ndarray,
    *,
    dim_name: str = "sample",
) -> xr.DataArray:
    """
    Log-height-linear interpolation for multiple target heights in one xarray-native call.

    Returns a DataArray with a new dimension `dim_name` (length = len(target_heights)).
    No extrapolation: all target heights must lie within the available height range.
    """
    if "height" not in da.dims:
        raise ValueError("DataArray must have a 'height' dimension")

    heights = np.asarray(da.coords["height"].values, dtype=np.float64)
    if heights.size < 2:
        raise ValueError(f"At least 2 heights required for interpolation, got {heights.size}")

    th = np.asarray(target_heights, dtype=np.float64)
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    if float(np.min(th)) < h_min or float(np.max(th)) > h_max:
        raise ValueError(
            f"target_heights outside available height range [{h_min}, {h_max}]. "
            "Extrapolation is not supported."
        )

    ln_heights = np.log(heights)
    da_log = da.assign_coords(ln_height=("height", ln_heights)).swap_dims({"height": "ln_height"}).sortby("ln_height")

    ln_targets = np.log(th)
    out = da_log.interp(ln_height=ln_targets, method="linear")

    # Rename the interpolation dimension and attach the physical heights as a coordinate.
    out = out.rename({"ln_height": dim_name})
    out = out.assign_coords({dim_name: np.arange(th.size, dtype=np.int64), f"{dim_name}_height": (dim_name, th)})
    return out


def _weibull_moment(
    *,
    A: xr.DataArray,
    k: xr.DataArray,
    p: int,
) -> xr.DataArray:
    """
    Compute the p-th raw moment of a Weibull(A, k) distribution:

        E[U^p] = A^p * Gamma(1 + p/k)

    This is vectorized and dask-friendly.
    """
    return (A ** p) * gamma(1 + (p / k))


def _rews_moment_factor(
    *,
    A_stack: xr.DataArray,
    k_stack: xr.DataArray,
    hub_height: float,
    rotor_diameter: float,
    method: str = "log_height_linear",
    n: int = 9,
) -> xr.DataArray:
    """
    Rotor-Equivalent Wind Speed (REWS) factor based on the cubic moment.

    Let U(z) ~ Weibull(A(z), k(z)).
    Define the rotor-area-averaged cubic moment:

        m3_rotor = (2/pi) * ∫_{-1}^{1} m3(H + R*t) * sqrt(1 - t^2) dt,
        where m3(z) = E[U(z)^3] = A(z)^3 * Gamma(1 + 3/k(z)),
              H is hub height, R = rotor_diameter/2.

    The REWS factor is:
        f = (m3_rotor / m3_hub)^(1/3)

    Numerical integration:
    Uses Gauss-Chebyshev quadrature of the second kind with `n` nodes, which is efficient
    for the weight sqrt(1 - t^2) and needs only `n` interpolations of A/k.

    Contract:
    - No extrapolation: rotor top/bottom must lie within available Weibull height range.
    - Returns a dask-friendly DataArray on the spatial grid (y, x).
    """
    if method != "log_height_linear":
        raise ValueError(f"Unsupported interpolation method: {method!r}. Only 'log_height_linear' is supported.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    R = float(rotor_diameter) / 2.0
    if R <= 0:
        raise ValueError(f"rotor_diameter must be > 0, got {rotor_diameter!r}")

    # Quadrature nodes for Chebyshev (2nd kind): t_i = cos(i*pi/(n+1)), weights ∝ sin^2(...)
    i = np.arange(1, n + 1, dtype=np.float64)
    theta = i * np.pi / (n + 1.0)
    t = np.cos(theta)  # in [-1, 1]
    w = np.sin(theta) ** 2  # positive weights

    # Physical sample heights along the rotor
    z = hub_height + R * t  # meters

    # Interpolate A(z), k(z) for all samples in one go
    A_s = _interp_da_to_heights_log(A_stack, z, dim_name="sample")
    k_s = _interp_da_to_heights_log(k_stack, z, dim_name="sample")

    # Cubic moment at samples and hub
    m3_s = _weibull_moment(A=A_s, k=k_s, p=3)
    A_hub = _interp_da_to_height_log(A_stack, hub_height)
    k_hub = _interp_da_to_height_log(k_stack, hub_height)
    m3_hub = _weibull_moment(A=A_hub, k=k_hub, p=3)

    # Weighted quadrature for (2/pi)∫ f(t) sqrt(1-t^2) dt
    # Using the identity: (2/pi)∫ ≈ 2/(n+1) * Σ sin^2(theta_i) * f(cos(theta_i))
    w_da = xr.DataArray(w, dims=("sample",), coords={"sample": np.arange(n, dtype=np.int64)})
    m3_rotor = (2.0 / (n + 1.0)) * (m3_s * w_da).sum(dim="sample")

    # REWS factor
    f = (m3_rotor / m3_hub) ** (1.0 / 3.0)
    f = f.rename("rews_factor")
    return f




def _validate_air_density(arr: np.ndarray, context: str) -> None:
    """
    Validate air density values are physically plausible (kg/m³).

    Checks:
    - All finite values must be strictly > 0
    - Median must be in [0.5, 2.0] kg/m³ (plausible atmospheric range)

    :param arr: numpy array of air density values
    :param context: description of where validation is happening (for error messages)
    :raises ValueError: If validation fails
    """
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]

    if finite_vals.size == 0:
        raise ValueError(
            f"Invalid air_density {context}: no finite values found."
        )

    min_finite = float(np.min(finite_vals))
    max_finite = float(np.max(finite_vals))
    median_finite = float(np.nanmedian(finite_vals))

    # All finite values must be > 0
    if min_finite <= 0:
        raise ValueError(
            f"Invalid air_density {context}: values must be > 0. "
            f"Found min={min_finite:.6g}, max={max_finite:.6g}, median={median_finite:.6g}"
        )

    # Median must be in plausible atmospheric range [0.5, 2.0] kg/m³
    if not (0.5 <= median_finite <= 2.0):
        raise ValueError(
            f"Unexpected air_density units/values {context}: median={median_finite:.6g} kg/m³ "
            f"outside plausible range [0.5, 2.0]. "
            f"min={min_finite:.6g}, max={max_finite:.6g}"
        )


def compute_air_density_at_height(
    self,
    height: float | int,
    *,
    chunk_size: int | None = None,
) -> xr.DataArray:
    """
    Compute air density at a specific hub height using GWA air_density data.

    Uses LINEAR interpolation in height on rho (physically correct for air density).
    This is different from Weibull parameter interpolation which uses log-height space.

    The correction factor c=(rho/rho0)^(1/3) is computed AFTER interpolating rho,
    never interpolated directly.

    If "air_density" exists in dataset with "height" dim, uses that.
    Otherwise, lazily loads GWA air-density rasters from disk via load_air_density().

    :param self: wind atlas instance with data property
    :param height: Target height (e.g., turbine hub height)
    :param chunk_size: Unused, kept for API consistency
    :return: DataArray with spatial dims (y, x) containing air density at target height
    :raises ValueError: If height out of range or values are invalid
    :raises FileNotFoundError: If air_density not in dataset and raw GWA rasters missing
    """
    air_density = None

    # Option 1: Use pre-loaded air_density from dataset if available
    if "air_density" in self.data.data_vars:
        air_density = self.data["air_density"]
        if "height" not in air_density.dims:
            raise ValueError(
                "air_density in dataset must have 'height' dimension for interpolation."
            )

    # Option 2: Lazy-load from raw GWA rasters
    if air_density is None:
        loaded = []
        loaded_heights = []
        for h in GWA_HEIGHTS:
            try:
                rho_h = self.load_air_density(h)
                # Drop trivial band dim if present
                if "band" in rho_h.dims and rho_h.sizes["band"] == 1:
                    rho_h = rho_h.isel(band=0, drop=True)
                loaded.append(rho_h)
                loaded_heights.append(float(h))
            except FileNotFoundError:
                continue  # Skip missing heights

        if len(loaded) < 2:
            raise FileNotFoundError(
                f"Insufficient air-density rasters found (need >=2, got {len(loaded)}). "
                f"Run atlas.materialize() or atlas.wind._load_gwa() to download GWA data."
            )

        # Stack into single DataArray with height dimension
        air_density = xr.concat(loaded, dim="height").assign_coords(height=loaded_heights)
        air_density.name = "air_density"

    # Interpolate rho linearly in height (xarray-native, dask-friendly)
    # Sort and use xarray's interp (no log transform for air density)
    heights = air_density.coords["height"].values
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    if height < h_min or height > h_max:
        raise ValueError(
            f"target_height={height} is outside available height range [{h_min}, {h_max}]. "
            "Extrapolation is not supported."
        )

    rho_sorted = air_density.sortby("height")
    rho_hub = rho_sorted.interp(height=float(height), method="linear")
    rho_hub.name = "air_density"

    # Drop height coordinate (now scalar after interp)
    if "height" in rho_hub.coords:
        rho_hub = rho_hub.drop_vars("height")

    # Drop trivial band dim if present (from lazy-loaded rasters)
    if "band" in rho_hub.dims and rho_hub.sizes["band"] == 1:
        rho_hub = rho_hub.isel(band=0, drop=True)

    # Assign hub_height coordinate
    rho_hub = rho_hub.assign_coords(hub_height=height)

    # Align to template if available
    if "template" in self.data.data_vars:
        tpl = self.data["template"]
        rho_hub = _match_to_template(rho_hub, tpl)

    # [VALIDATION BRANCH] Validate interpolated values: must be > 0 and in plausible kg/m³ range.
    # Uses policy-driven validation: probe-based for dask, full for eager.
    _validate_values(
        rho_hub, self.data,
        validation="auto",
        check_nan=False,  # NaNs allowed at edges
        check_positive=True,
        check_range=(0.5, 2.0),
        context=f"air_density after interpolation to height={height}m",
    )

    return rho_hub


def compute_weibull_pdf_at_height(
    self,
    height: float | int,
    *,
    chunk_size: int | None = None,
    method: str = "log_height_linear",
) -> xr.DataArray:
    """
    Compute Weibull PDF at a specific hub height using interpolated A and k parameters.

    Loads Weibull parameters for all available heights (e.g., 50/100/150/200m from GWA),
    interpolates to target height, and computes the PDF.

    :param self: wind atlas instance with data property containing weibull parameters
    :param height: Target height (e.g., turbine hub height)
    :param chunk_size: Size of chunks in pixels. If None, computation is not chunked.
    :param method: Interpolation method for Weibull parameters. Only "log_height_linear" supported.
    :return: DataArray with dims (wind_speed, y, x) containing PDF at target height
    :raises ValueError: If height is outside available range or method not supported
    """
    # Load Weibull params at all standard heights
    A_list = []
    k_list = []
    available_heights = []

    for h in GWA_HEIGHTS:
        try:
            a_h, k_h = self.load_weibull_parameters(h)
            A_list.append(a_h)
            k_list.append(k_h)
            available_heights.append(h)
        except FileNotFoundError:
            # Skip heights that don't have data files
            continue

    if len(available_heights) < 2:
        raise ValueError(
            f"At least 2 height levels required for interpolation, but only found: {available_heights}"
        )

    # Stack into DataArrays with height dimension
    A_stack = xr.concat(A_list, dim="height").assign_coords(height=available_heights)
    k_stack = xr.concat(k_list, dim="height").assign_coords(height=available_heights)

    # Interpolate to target height
    A_hub, k_hub = interpolate_weibull_params_to_height(
        A_stack, k_stack, height, method=method
    )

    # Align to template if available
    if "template" in self.data.data_vars:
        tpl = self.data["template"]
        A_hub = _match_to_template(A_hub, tpl)
        k_hub = _match_to_template(k_hub, tpl)

    # Get wind speed grid
    u = self.data.coords["wind_speed"].values

    inputs = {
        "weibull_a": A_hub,
        "weibull_k": k_hub,
        "u_power_curve": u,
    }

    if chunk_size is None or _is_dask_backed(A_hub):
        # Dask-backed: execute directly (dask handles chunking internally)
        pdf = weibull_probability_density(**inputs)
    else:
        pdf = compute_chunked(self, weibull_probability_density, chunk_size, **inputs)

    # Align result to template if available
    if "template" in self.data.data_vars:
        pdf = _match_to_template(pdf, self.data["template"])

    return pdf


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

    This implementation is fully vectorized and dask-friendly (no Python loops,
    no eager array evaluation). Uses np.trapezoid via xr.apply_ufunc.

    :param correction_factor:
    :param weibull_pdf: probability density function from weibull_probability_density()
    :param wind_shear: wind shear coefficient
    :param u_power_curve: power curve wind speed
    :param p_power_curve: power curve output
    :param h_turbine: hub height of wind turbine in m
    :param h_reference: reference height at which weibull pdf is computed
    :return:
    """
    # Ensure u and p are numpy arrays (1D power curve, small)
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
    # Stay as xarray DataArray (dask-friendly)
    s = (h_turbine / h_reference) ** wind_shear

    # Exact wind_speed grid contract (no tolerant float matching).
    # Order may differ; alignment MUST be by label (prevents positional bugs).
    if "wind_speed" in weibull_pdf.dims:
        ws = np.asarray(weibull_pdf.coords["wind_speed"].values)  # coords-only .values OK

        # Require exact same labels (no tolerance), but allow different order.
        if ws.shape != u.shape or not np.array_equal(np.sort(ws), np.sort(u)):
            raise ValueError(
                "wind_speed grid mismatch: weibull_pdf.coords['wind_speed'] must contain exactly the same "
                "labels as u_power_curve (order can differ). "
                f"Got coords={ws!r} vs u_power_curve={u!r}."
            )

        # Reorder by label to match u_power_curve order (contract: label-based)
        pdf_aligned = weibull_pdf.sel(wind_speed=u).transpose("wind_speed", ...)
    else:
        pdf_aligned = weibull_pdf

    # Build wind_speed as 1D DataArray for vectorized operations
    u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})

    # Compute hub-height wind speed: u_hub = u * s
    # Broadcasting: u_da (wind_speed,) * s (y, x) => (wind_speed, y, x)
    u_hub = u_da * s

    # Evaluate power curve at all hub-height wind speeds (vectorized, dask-friendly)
    pc = _interp_power_curve(u_hub, u, p)

    # Build integrand: pdf(u) * PC(u*s)
    # Both have dims (wind_speed, y, x) after alignment
    integrand = pdf_aligned * pc

    # Vectorized trapezoidal integration over wind_speed (dask-friendly)
    cf = _trapz_over_wind_speed(integrand, u_da)

    # Apply correction factor
    result = cf * correction_factor
    result.name = "capacity_factor"
    return result


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