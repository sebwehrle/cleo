# CLEO

CLEO is an `xarray`-based Python package for wind resource assessment with Global Wind Atlas (GWA) inputs.
It materializes canonical Zarr stores, computes wind metrics, and persists/export results.

## Installation

```bash
python -m pip install -e .
```

## Quick Start

```python
from cleo import Atlas

atlas = Atlas(
    "/path/to/workdir",
    country="AUT",      # ISO-3166 alpha-3
    crs="EPSG:3035",
)

# Optional: restrict materialized turbines to an explicit subset
atlas.configure_turbines(["Enercon.E40.500", "Vestas.V100.2000"])

# Build/update canonical stores (wind.zarr + landscape.zarr)
atlas.materialize()

# Select turbines for compute calls that require turbines
atlas.wind.select(turbines=["Enercon.E40.500"])

# Compute + cache into active wind store
atlas.wind.capacity_factors(mode="hub", air_density=True).cache()

# Compute another metric
mean_ws = atlas.wind.mean_wind_speed(height=100).data
```

## Workspace Layout

After materialization, CLEO uses:

- `<workdir>/wind.zarr`
- `<workdir>/landscape.zarr`
- `<workdir>/regions/<region_id>/wind.zarr` (only after region selection/materialization)
- `<workdir>/regions/<region_id>/landscape.zarr`
- `<workdir>/results/<run_id>/<metric_name>.zarr` (only after `persist`)
- `<workdir>/resources/*.yml`
- `<workdir>/data/raw/<ISO3>/*.tif`
- `<workdir>/data/nuts/*`

## Public API

### Atlas Construction

```python
Atlas(
    path,
    country,
    crs,
    *,
    chunk_policy: dict[str, int] | None = None,
    compute_backend: str = "serial",
    compute_workers: int | None = None,
    region: str | None = None,
    results_root: Path | None = None,
    fingerprint_method: str = "path_mtime_size",
)
```

Options:

- `path`: workspace root for stores/resources/results.
- `country`: ISO3 country code (for GWA/raw data resolution).
- `crs`: canonical projected CRS for the atlas stores.
- `chunk_policy`: xarray/dask chunk sizes for `y/x`.
- `compute_backend`: compute execution backend for internal eager materialization (`"serial"|"threads"|"processes"|"distributed"`).
- `compute_workers`: optional worker count for local dask backends (`threads`/`processes`).
  Use `None` for dask defaults. Must be `None` or `1` for `serial`. Must be `None` for `distributed` (configure workers on the active client).
- `region`: optional initial region selection.
- `results_root`: optional custom results directory.
- `fingerprint_method`: hashing/fingerprinting policy used by unification internals.

Good for:

- Creating a reproducible atlas workspace with explicit compute/storage policy.

### Lifecycle and Region Selection

`atlas.materialize()`
- Builds/updates canonical stores and selected region stores.
- Good for normal day-to-day workflow.

`atlas.materialize_canonical()`
- Builds/updates only base country stores (`wind.zarr`, `landscape.zarr`).
- Good for preparing a base before region-specific work.

`atlas.select(region=..., region_level=None, inplace=False)`
- `region`: region name or `None` (clear selection).
- `region_level`: optional NUTS level (1/2/3) disambiguation.
- `inplace`: if `True`, mutates atlas; otherwise returns a selected copy.
- Good for switching scope between full-country and region-scoped analysis.

`atlas.region`
- Current selected region name (`None` means full-country).

`atlas.nuts_regions` / `atlas.nuts_regions_level(level)`
- Discover available region names (all levels or one specific level).

### Domains and Data Access

`atlas.wind` / `atlas.landscape`
- Domain facades for compute/load operations.

`atlas.wind_data` / `atlas.landscape_data`
- Direct dataset access shortcuts.

`atlas.flatten(domain="wind"|"landscape"|"both", digits=5, exclude_template=True, include_domain_prefix=True, cast_binary_to_int=False, include_only=None)`
- `domain`: select source dataset(s).
- `digits`: coordinate rounding precision for `(y, x)` index.
- `exclude_template`: omit template variable from output columns.
- `include_domain_prefix`: for `domain="both"`, prefix columns with `wind__` / `landscape__`.
- `cast_binary_to_int`: if `True`, cast binary columns (`bool` or numeric `{0,1}`) to nullable `Int8`.
- `include_only`: optional list of output columns to keep; raises if any requested column is missing.
- Good for exporting tidy tabular data (e.g. econometric pipelines).

`Atlas.validate_flatten_schema(df, required_columns)`
- Validates that flattened output contains all required columns.
- Raises with missing column names; does not mutate/filter data.

### WindDomain APIs

`atlas.wind.turbines`
- Tuple of available turbine IDs in the active store.

`atlas.wind.selected_turbines`
- Persistent turbine selection or `None` (all turbines).

`atlas.wind.select(turbines=[...])`
- Set persistent turbine selection for future turbine-dependent metrics.

`atlas.wind.clear_selection()`
- Clear persistent selection.

`atlas.wind.compute(metric, **kwargs)`
- Generic metric entrypoint.
- Good for dynamic metric dispatch.

`atlas.wind.mean_wind_speed(height, **kwargs)`
- Convenience wrapper for `compute("mean_wind_speed", ...)`.

`atlas.wind.capacity_factors(turbines=None, height=100, air_density=False, loss_factor=1.0, **kwargs)`
- Convenience wrapper for `compute("capacity_factors", ...)`.

`compute(...)` returns `DomainResult`:

- `.data`
  - Lazy/eager xarray `DataArray` depending on backing.
- `.cache(overwrite=True, allow_mode_change=False)`
  - `overwrite`: replace existing variable in active wind store.
  - `allow_mode_change`: required when replacing cached `capacity_factors` with different mode (`hub` vs `rews`).
  - Good for making results part of active domain state.
- `.persist(run_id=None, params=None, metric_name=None)`
  - `run_id`: explicit run id (or auto-generate).
  - `params`: metadata dict to store in attrs.
  - `metric_name`: override default metric variable name.
  - Good for run-tracked artifacts under `results_root`.

### Supported Wind Metrics

- `mean_wind_speed`
  - Required: `height` (int)
- `capacity_factors`
  - Requires turbines via selection or `turbines=[...]`
  - Optional: `mode="hub"|"rews"`, `rews_n`, `air_density`, `loss_factor`
- `lcoe`
  - Requires turbines and:
  - `om_fixed_eur_per_kw_a`, `om_variable_eur_per_kwh`, `discount_rate`, `lifetime_a`
  - Optional: `turbine_cost_share`, `hours_per_year`, plus capacity-factor options
- `min_lcoe_turbine`
- `optimal_power`
- `optimal_energy`

### LandscapeDomain APIs

`atlas.landscape.data`
- Active landscape dataset.

`atlas.landscape.add(name, source_path, *, kind="raster", params=None, materialize=True, if_exists="error")`
- `name`: output variable name in landscape store.
- `source_path`: input source path.
- `kind`: currently only `"raster"`.
- `params`: source-specific params (e.g. `{"categorical": True}`).
- `materialize`: if `True`, immediately writes variable into landscape store.
- `if_exists`: `"error"|"replace"|"noop"` conflict policy.
- Good for incrementally enriching landscape layers.

`atlas.materialize_clc(source="clc2018", url=None, force_download=False, force_prepare=False)`
- Downloads CLC source raster (if missing) and prepares a country-cropped cache aligned to wind/GWA grid.
- For first-time download, provide a direct raster URL via `url=...` (if cache is already present, `url` is optional).
- Returns path to prepared CLC raster cache.

`atlas.landscape.add_clc_category(categories, *, name=None, source="clc2018", if_exists="error", materialize=True)`
- Adds CLC data into the active landscape store via the same clipping/masking pipeline as other landscape rasters.
- `categories="all"`: adds categorical `land_cover` layer (or `name` override).
- `categories=int`: adds one binary CLC-code mask variable (`0/1`, NaN outside valid mask).
- `categories=list[int]`: adds combined binary mask for those codes; requires `name`.

### Results API

`atlas.new_run_id(prefix=None)`
- Creates sortable run ID string; optional safe prefix.

`atlas.persist(metric_name, obj, run_id=None, params=None)` (transitional low-level API)
- Persists arbitrary `Dataset`/`DataArray` into results store.
- Prefer fluent `DomainResult.persist(...)` when available.
- `run_id` and `metric_name` must be simple path tokens (no `/`, `\\`, `.` or `..`).

`atlas.open_result(run_id, metric_name)`
- Opens a persisted result store lazily.
- `run_id` and `metric_name` use the same path-token validation as `persist`.

`atlas.export_result_netcdf(run_id, metric_name, out_path, encoding=None)`
- Exports a persisted result store to `.nc`.
- `encoding`: optional xarray encoding overrides.
- `run_id` and `metric_name` use the same path-token validation as `persist`.

```python
run = atlas.wind.compute("capacity_factors", mode="hub", air_density=True)
store_path = run.persist(run_id="baseline")
opened = atlas.open_result(store_path.parent.name, "capacity_factors")
atlas.export_result_netcdf(store_path.parent.name, "capacity_factors", "cf.nc")
```

### Cleanup APIs

`atlas.clean_results(run_id=None, older_than=None, metric_name=None)`
- Remove persisted result stores by run/age/metric filters.
- When provided, `run_id`/`metric_name` use the same path-token validation as `persist`.

`atlas.clean_regions(region=None, older_than=None, include_incomplete=True)`
- Remove materialized region stores under `<workdir>/regions`.
- `include_incomplete=False` keeps partial/incomplete region stores untouched.

## Dask and Chunking

CLEO does not expose a `DaskConfig` object in the public API.
Chunking behavior is controlled by `chunk_policy` and by how xarray opens data.

- If `dask` is installed and arrays are chunked, computations can remain lazy.
- If data is unchunked/eager, computations execute eagerly.
- For local schedulers, set `compute_workers` on `Atlas(...)` to cap worker count.
- `compute_backend="distributed"` requires an active `dask.distributed.Client`.
  In distributed mode, worker count is managed by the client/cluster (not `Atlas`).
  If no active client exists, CLEO raises a clear `RuntimeError`.
  Local worker cap example:

```python
atlas = Atlas(..., compute_backend="processes", compute_workers=4)
```
  Typical setup:

```python
from dask.distributed import Client
Client()  # start and register active client
atlas = Atlas(..., compute_backend="distributed")
```

## Remaining Issues

These are known and intentionally left as follow-up work:

- Broad exception handling (`except Exception`) still exists in several internal modules (`atlas`, `unify`, `spatial`, `loaders`) and can hide root causes.
- Core orchestration modules are large (`unify.py`, `atlas.py`) and could be split to reduce maintenance risk.

## Testing

Test suite location:

- `tests/unit/`
- `tests/integration/`
- `tests/smoke/`

Run:

```bash
python -m pytest tests/ -v
```
