# CLEO

CLEO is an `xarray`-based package for wind resource assessment with Global Wind Atlas (GWA) inputs.
It materializes canonical Zarr stores, computes wind/energy metrics, and persists/exports results.

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

# Optional: restrict turbines materialized into wind.zarr
atlas.configure_turbines(["Enercon.E40.500", "Vestas.V100.2000"])

# Build/update canonical stores (wind.zarr + landscape.zarr)
atlas.build()

# Select turbines for turbine-dependent metrics
atlas.wind.select(turbines=["Enercon.E40.500"])

# Compute + materialize into active wind store
atlas.wind.capacity_factors(mode="direct_cf_quadrature", air_density=True).materialize()

# Compute another metric (lazy result)
mean_ws = atlas.wind.mean_wind_speed(height=100).data
```

## Workspace Layout

After materialization, CLEO uses:

- `<workdir>/wind.zarr`
- `<workdir>/landscape.zarr`
- `<workdir>/regions/<region_id>/wind.zarr` (after region selection/materialization)
- `<workdir>/regions/<region_id>/landscape.zarr`
- `<workdir>/results/<run_id>/<metric_name>.zarr` (after `persist`)
- `<workdir>/resources/*.yml`
- `<workdir>/data/raw/<ISO3>/*.tif`
- `<workdir>/data/nuts/*`

## Public API

Public API boundary:

- Supported entry point: `from cleo import Atlas`.
- Public behavior is defined by the `Atlas`, `WindDomain`, `LandscapeDomain`, and `DomainResult` contracts below.
- Internal modules (for example `cleo.assess`, `cleo.loaders`, `cleo.unification.*`, `cleo.clc`) are implementation details and not part of the stability contract.

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

- `path`: workspace root for stores/resources/results.
- `country`: ISO3 country code.
- `crs`: canonical projected CRS for atlas stores.
- `chunk_policy`: chunk sizes for `y/x` when opening zarr datasets.
- `compute_backend`: eager-materialization backend (`"serial"|"threads"|"processes"|"distributed"`).
- `compute_workers`: optional worker cap for local backends (`threads`/`processes`).
  Must be `None` or `1` for `serial`; must be `None` for `distributed`.
- `region`: optional initial region selection (applied on `build()`).
- `results_root`: optional custom results directory.
- `fingerprint_method`: internal fingerprinting policy used by unification internals.

### Lifecycle and Selection

- `atlas.build()`
  - Ensures base stores and, when a region is selected, region stores.
- `atlas.build_canonical()`
  - Ensures base stores only (`wind.zarr`, `landscape.zarr`).
- `atlas.select(region=..., region_level=None, inplace=False)`
  - `region`: region name or `None` to clear selection.
  - `region_level`: optional NUTS level disambiguation (`1|2|3`).
  - `inplace=False` returns a selected clone; `inplace=True` mutates current atlas.
- `atlas.region`
  - current selected region name, or `None`.
- `atlas.nuts_regions`, `atlas.nuts_regions_level(level)`
  - discover available region names.

### Domains and Data Access

- `atlas.wind`, `atlas.landscape`
  - domain facades.
- `atlas.wind_data`, `atlas.landscape_data`
  - direct shortcuts to `atlas.wind.data` / `atlas.landscape.data`.
- `atlas.configure_turbines(turbines)`
  - configure turbines used for wind materialization.
- `atlas.turbines_configured`
  - configured materialization turbine set or `None`.
- `atlas.configure_timebase(hours_per_year)`
  - configure timebase assumptions for annualized metrics (LCOE-family).
  - default is 8766.0 hours/year when not configured.
- `atlas.timebase_configured`
  - configured timebase dict (`{"hours_per_year": float}`) or `None`.
- `atlas.flatten(domain="wind"|"landscape"|"both", digits=5, exclude_template=True, include_domain_prefix=True, cast_binary_to_int=False, include_only=None)`
  - flattens domain data into a tabular frame.
- `Atlas.validate_flatten_schema(df, required_columns)`
  - raises if required flattened columns are missing.

### WindDomain

- `atlas.wind.turbines`
  - available turbine IDs in active wind store.
- `atlas.wind.selected_turbines`
  - persistent selection or `None` (all turbines).
- `atlas.wind.select(turbines=[...] | None, turbine_indices=[...] | None)`
  - set persistent turbine selection by IDs (`turbines`) or by positional indices into `atlas.wind.turbines` (`turbine_indices`).
  - exactly one of `turbines` or `turbine_indices` must be provided.
  - mutates selection in place and returns `None`.
  - example: `atlas.wind.select(turbine_indices=[1, 3, 4])`
- `atlas.wind.clear_selection()`
  - clear persistent selection; returns `None`.
- `atlas.wind.clear_computed()`
  - clear transient computed overlays from `atlas.wind.data`; returns `None`.
- `atlas.wind.compute(metric, **kwargs)`
  - generic metric entrypoint.
  - rejects materialize-only kwargs `overwrite` and `allow_mode_change`; pass those to `.materialize(...)`.
  - stages a lazy normalized overlay into `atlas.wind.data[metric]` before store writes.
  - staged wind overlays are cleared by `atlas.select(...)`, `atlas.build()`, `atlas.build_canonical()`, and `atlas.build_clc()`.
- `atlas.wind.mean_wind_speed(height, **kwargs)`
- `atlas.wind.capacity_factors(turbines=None, air_density=False, loss_factor=1.0, mode="direct_cf_quadrature", rews_n=12, **kwargs)`
  - does not accept `height`; hub height is derived from each turbine definition.
- `atlas.wind.rews_mps(turbines=None, air_density=False, rews_n=12, **kwargs)`

`compute(...)` returns `DomainResult`:

- `.data`
  - computed `xarray.DataArray` (lazy when backed by dask).
- `.materialize(overwrite=True, allow_mode_change=False)`
  - writes metric to active wind store and reloads surfaced domain state.
  - `allow_mode_change` is required to replace existing `capacity_factors` with a different `cleo:cf_mode`.
- `.persist(run_id=None, params=None, metric_name=None)`
  - writes standalone result artifact under `results_root`.

### Supported Wind Metrics

- `mean_wind_speed`
  - Required: `height` (int)
- `capacity_factors`
  - Requires turbines via selection or `turbines=[...]`
  - Options: `mode="direct_cf_quadrature"|"momentmatch_weibull"|"hub"|"rews"`, `rews_n`, `air_density`, `loss_factor`
- `rews_mps`
  - Requires turbines via selection or `turbines=[...]`
  - Options: `rews_n`, `air_density`
- `lcoe`
  - Requires turbines and: `om_fixed_eur_per_kw_a`, `om_variable_eur_per_kwh`, `discount_rate`, `lifetime_a`
  - Optional: `turbine_cost_share`, plus `capacity_factors` options (`mode`, `rews_n`, `air_density`, `loss_factor`)
  - Timebase (`hours_per_year`) is configured at Atlas level via `atlas.configure_timebase(...)`, not per-metric.
- `min_lcoe_turbine`
  - Same parameter requirements/options as `lcoe`
- `optimal_power`
  - Same parameter requirements/options as `lcoe`
- `optimal_energy`
  - Same parameter requirements/options as `lcoe`

### LandscapeDomain

- `atlas.landscape.data`
  - active landscape dataset.
- `atlas.landscape.compute(metric, **kwargs)`
  - compute entrypoint for landscape metrics.
  - currently supported metric:
    - `metric="distance"` with:
      - `source`: one source variable name or a list/tuple of source variable names.
      - `name`: optional output name or list/tuple of output names; default is `distance_<source>`.
      - `if_exists`: `"error"|"replace"|"noop"`.
  - distance sources must be store-backed landscape variables (not staged-only overlays).
  - returns a batch result object with:
    - `.data`: staged `xr.Dataset` containing computed distance variables.
    - `.materialize(if_exists=None)`: writes staged distance variables into the active landscape store and returns materialized `xr.Dataset`.
- `atlas.landscape.add(name, source_path, *, kind="raster", params=None, if_exists="error")`
  - stages a raster landscape candidate and returns `LandscapeAddResult`.
  - `add(...)` is raster-only (`kind` must be `"raster"`); use `rasterize(...)` for vector sources.
  - staged variables are visible in `atlas.landscape.data` before store writes.
  - `if_exists`: `"error"|"replace"|"noop"`.
- `atlas.landscape.rasterize(shape, *, name, column=None, all_touched=False, if_exists="error")`
  - stages a vector-rasterized landscape candidate and returns `LandscapeAddResult`.
  - `shape` accepts path-like vector sources or a `geopandas.GeoDataFrame`.
  - `column=None` burns binary coverage (`1.0`); otherwise burns numeric values from `column`.
  - staged variables are visible in `atlas.landscape.data` before store writes.
  - `if_exists`: `"error"|"replace"|"noop"`.
- `atlas.landscape.clear_staged()`
  - clears staged (not yet materialized) landscape overlays.
- `atlas.build_clc(source="clc2018", url=None, force_download=False, force_prepare=False)`
  - prepares CLC cache aligned to wind/GWA grid.
- `atlas.landscape.add_clc_category(categories, *, name=None, source="clc2018", if_exists="error")`
  - `categories="all"`: full categorical layer (`land_cover` default name).
  - `categories=int`: single binary CLC-code mask.
  - `categories=list[int]`: combined binary mask (requires `name`).
  - returns `LandscapeAddResult`.

`LandscapeAddResult`:

- `.data`
  - staged `xarray.DataArray` candidate.
- `.materialize(if_exists=None)`
  - commits the staged variable to the active landscape store.
  - staged landscape overlays are cleared by `atlas.select(...)`, `atlas.build()`, `atlas.build_canonical()`, and `atlas.build_clc()`.

Distance compute example:

```python
run = atlas.landscape.compute(
    "distance",
    source=["roads_mask", "water_mask"],
    name=["distance_roads", "distance_water"],
    if_exists="replace",
)
ds_stage = run.data
ds_mat = run.materialize()
```

Notes:
- Distance outputs are in meters (`units="m"`), based on projected metric CRS and exact atlas grid spacing.
- `if_exists="noop"` skips existing distance variables only when their stored distance spec matches the requested source/algorithm; otherwise it raises and requires `if_exists="replace"`.
- Distance variables materialized in region stores are region-local artifacts and may disappear when region stores are rebuilt from base stores.

Vector rasterization example:

```python
op = atlas.landscape.rasterize(
    "/path/to/stays.geojson",
    name="overnight_stays",
    column="overnight_stays",
    all_touched=False,
    if_exists="replace",
)
da = op.materialize()
```

### Results and Cleanup

- `atlas.new_run_id(prefix=None)`
  - creates sortable run ID with optional safe prefix.
- `atlas.persist(metric_name, obj, run_id=None, params=None)`
  - low-level persistence API; `DomainResult.persist(...)` is preferred.
- `atlas.open_result(run_id, metric_name)`
  - open persisted result store lazily.
- `atlas.export_result_netcdf(run_id, metric_name, out_path, encoding=None)`
  - export persisted result store to `.nc`.
- `atlas.clean_results(run_id=None, older_than=None, metric_name=None)`
  - cleanup persisted result stores.
- `atlas.clean_regions(region=None, older_than=None, include_incomplete=True)`
  - cleanup region materialization stores.

Example:

```python
run = atlas.wind.compute("capacity_factors", mode="hub", air_density=True)
store_path = run.persist(run_id="baseline")
opened = atlas.open_result(store_path.parent.name, "capacity_factors")
atlas.export_result_netcdf(store_path.parent.name, "capacity_factors", "cf.nc")
```

## Dask and Chunking

CLEO does not expose a `DaskConfig` object in the public API.
Chunking and execution behavior are controlled via `chunk_policy`, dataset chunking, and `compute_backend`.

- Dask-backed arrays can stay lazy until materialization paths (`materialize`, `persist`, export).
- Local backend worker cap is controlled by `compute_workers`.
- `compute_backend="distributed"` requires an active `dask.distributed.Client`.

```python
atlas = Atlas(..., compute_backend="processes", compute_workers=4)
```

```python
from dask.distributed import Client

Client()
atlas = Atlas(..., compute_backend="distributed")
```

## Benchmarking

Use `cleo.bench.benchmark_metric_variants(...)` for side-by-side metric variant comparisons.

```python
from cleo.bench import benchmark_metric_variants

df = benchmark_metric_variants(
    atlas,
    "capacity_factors",
    variants=[
        {"label": "baseline", "kwargs": {"mode": "direct_cf_quadrature"}},
        {"label": "candidate_rews7", "kwargs": {"mode": "rews", "rews_n": 7}},
    ],
    repeats=3,
    warmup=1,
    cache=True,
    baseline_label="baseline",
)

print(df)
```

## Testing

```bash
python -m pytest -q
```
