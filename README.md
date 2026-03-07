# CLEO

CLEO is an `xarray`-based package for wind resource assessment with Global Wind Atlas (GWA) inputs.
It materializes canonical Zarr stores, computes wind/energy metrics, and persists/exports results.

CLEO prepares analysis-ready wind/landscape rasters and tabular exports; econometric regression is out of scope.

## Supported Environment

- **Python**: 3.10+
- **OS**: Linux (primary), macOS (secondary)
- **System deps**: GDAL/PROJ (via rasterio/pyproj), HDF5/NetCDF

For detailed environment requirements and dependency policies, see `docs/CONTRACT_UNIFIED_ATLAS.md` Section C.

## Installation

### System Dependencies (GDAL/PROJ)

CLEO requires GDAL and PROJ system libraries for geospatial operations. Install these before installing CLEO:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libgdal-dev gdal-bin libproj-dev proj-bin
```

**macOS (Homebrew):**
```bash
brew install gdal proj
```

**Conda (any platform):**
```bash
conda install -c conda-forge gdal proj
```

**Verify installation:**
```bash
gdalinfo --version   # Should show GDAL version
proj                 # Should show PROJ usage
```

### Package Installation

```bash
python -m pip install -e .
```

Dependency management is defined in `pyproject.toml`. A `requirements-lock.txt` is available for reproducible installs.

## Development Commands (Canonical)

Use these commands directly (no Makefile required):

```bash
# install dev + docs dependencies
python -m pip install -e ".[dev,docs]"

# format/lint
python -m ruff format --check .
python -m ruff check .

# architecture/boundary guardrails
python -m pytest -q tests/unit/compat

# test suite
python -m pytest -q tests/unit
python -m pytest -q tests/integration
python -m pytest -q tests/smoke

# build + import verification
python -m pip install build
python -m build
python -m pip install dist/*.whl
python -c "from cleo import __version__; print(f'cleo {__version__}')"

# docs build
python -m mkdocs build --strict
python -m mkdocs serve
```

Optional local cleanup:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
rm -rf build/ dist/ 2>/dev/null || true
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
atlas.wind.compute(
    "capacity_factors",
    method="rotor_node_average",
    interpolation="auto",
    air_density=True,
).materialize()

# Compute another metric (lazy result)
mean_ws = atlas.wind.compute(
    "wind_speed",
    method="height_weibull_mean",
    height=100,
).data
# mean_ws name: "mean_wind_speed", dims: ("height","y","x")
```

## Workspace Layout

After materialization, CLEO uses:

- `<workdir>/wind.zarr`
- `<workdir>/landscape.zarr`
- `<workdir>/areas/<area_id>/wind.zarr` (after area selection/materialization)
- `<workdir>/areas/<area_id>/landscape.zarr`
- `<workdir>/results/<run_id>/<metric_name>.zarr` (after `persist`)
- `<workdir>/resources/*.yml`
- `<workdir>/data/raw/<ISO3>/*.tif`
- `<workdir>/data/nuts/*`

## Public API

Public API boundary:

- Supported entry point: `from cleo import Atlas, __version__`.
- `__version__` provides runtime version access.
- Public behavior is defined by the `Atlas`, `WindDomain`, `LandscapeDomain`, and `DomainResult` contracts below.
- Internal modules (for example `cleo.assess`, `cleo.loaders`, `cleo.unification.*`, `cleo.clc`) are implementation details and not part of the stability contract.

```python
from cleo import __version__
print(__version__)  # e.g., "0.0.1"
```

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
    area: str | None = None,
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
- `area`: optional initial area selection (applied on `build()`).
- `results_root`: optional custom results directory.
- `fingerprint_method`: internal fingerprinting policy used by unification internals.

### Lifecycle and Selection

- `atlas.build()`
  - Ensures base stores and, when an area is selected, area stores.
  - If required GWA wind rasters are missing under
    `<workdir>/data/raw/<ISO3>/`, attempts automatic download of the required
    GWA layers/heights before failing.
  - If area-aware paths require NUTS boundaries and no local NUTS shapefile
    exists under `<workdir>/data/nuts/`, attempts automatic NUTS
    download/extract before failing.
- `atlas.build_canonical()`
  - Ensures base stores only (`wind.zarr`, `landscape.zarr`).
- `atlas.select(area=..., nuts_level=None, inplace=False)`
  - `area`: area name or `None` to clear selection.
  - `nuts_level`: optional NUTS level disambiguation (`0|1|2|3`).
    - `0` selects the country-border area (NUTS level 0).
  - `inplace=False` returns a selected clone; `inplace=True` mutates current atlas.
- `atlas.area`
  - current selected area name, or `None`.
- `atlas.nuts_areas`, `atlas.nuts_areas_level(level)`
  - discover available area names.

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
- `atlas.configure_economics(discount_rate, lifetime_a, om_fixed_eur_per_kw_a, om_variable_eur_per_kwh, bos_cost_share)`
  - configure baseline economics assumptions for LCOE-family metrics.
  - all parameters are optional; multiple calls merge values.
  - per-call overrides via `economics={...}` take precedence over baseline.
- `atlas.economics_configured`
  - configured economics dict or `None`.
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
- `atlas.wind.convert_units(variable, to_unit, *, from_unit=None, inplace=False)`
  - convert a wind variable to a different unit.
  - `variable`: variable name in `atlas.wind.data`.
  - `to_unit`: target unit string (e.g., `"km/h"`, `"MW"`).
  - `from_unit`: source unit override; if `None`, reads from variable's `units` attr.
  - `inplace=True` stages the converted DataArray; `inplace=False` returns it.
  - conversion is dask-friendly (lazy arrays stay lazy).
- `atlas.wind.compute(metric, **kwargs)`
  - metric entrypoint for all wind metrics.
  - rejects materialize-only kwargs `overwrite` and `allow_method_change`; pass those to `.materialize(...)`.
  - stages a lazy normalized overlay into `atlas.wind.data[<resolved_variable_name>]` before store writes.
  - staged wind overlays are cleared by `atlas.select(...)`, `atlas.build()`, `atlas.build_canonical()`, and `atlas.build_clc()`.

`compute(...)` returns `DomainResult`:

- `.data`
  - computed `xarray.DataArray` (lazy when backed by dask).
- `.materialize(overwrite=True, allow_method_change=False)`
  - writes metric to active wind store and reloads surfaced domain state.
  - `allow_method_change` is required to replace existing `capacity_factors` with a different `cleo:cf_method`.
  - `mean_wind_speed` is materialized per requested height slice into one
    aggregated `mean_wind_speed(height,y,x)` variable.
  - `overwrite=False` for `mean_wind_speed` blocks only if that requested
    height slice already has data; other heights can still be added.
  - existing legacy `mean_wind_speed(y,x)` variables fail with an explicit
    migration error.
- `.persist(run_id=None, params=None, metric_name=None)`
  - writes standalone result artifact under `results_root`.

### Supported Wind Metrics

- `wind_speed`
  - Public selector with method-dependent output variable names.
  - `method="height_weibull_mean"`
    - Required: `height`
    - Output variable: `mean_wind_speed`
    - Output dims: `("height","y","x")`
    - Repeated `.materialize()` calls at different heights aggregate into one
      `mean_wind_speed(height,y,x)` variable in active wind store.
  - `method="rotor_equivalent"`
    - Requires turbines via selection or `turbines=[...]`
    - Output variable: `rotor_equivalent_wind_speed`
    - Output dims: `("turbine","y","x")`
    - Options: `rews_n`, `air_density`, `interpolation="auto"|"ak_logz"|"mu_cv_loglog"`
    - `auto` resolves to `mu_cv_loglog`; explicit `ak_logz` is allowed and
      preserves no-extrapolation semantics across the rotor heights.
- `capacity_factors`
  - Requires turbines via selection or `turbines=[...]`
  - Options:
    - `method="rotor_node_average"|"rotor_moment_matched_weibull"|"hub_height_weibull"|"hub_height_weibull_rews_scaled"`
    - `interpolation="auto"|"ak_logz"|"mu_cv_loglog"`
    - `auto` resolves by method family: hub-height methods -> `ak_logz`,
      rotor-aware methods -> `mu_cv_loglog`
    - explicit `ak_logz` keeps no-extrapolation semantics and may raise if a
      requested rotor node falls outside the available GWA height range
    - `rews_n`, `air_density`, `loss_factor`
  - Example:
    ```python
    atlas.wind.select(turbines=["Enercon.E40.500"])
    cf = atlas.wind.compute(
        "capacity_factors",
        method="rotor_node_average",
        interpolation="ak_logz",
        rews_n=12,
    ).data
    ```
- `lcoe`
  - Requires turbines via selection or `turbines=[...]`
  - Uses grouped spec API:
    - `cf={...}`: CF parameters. Keys: `method`, `interpolation`, `air_density`, `rews_n`, `loss_factor`.
      Defaults: `method="rotor_node_average"`, `interpolation="auto"`, `air_density=False`, `rews_n=12`, `loss_factor=1.0`.
    - `economics={...}`: Economics parameters. Required: `discount_rate`, `lifetime_a`,
      `om_fixed_eur_per_kw_a`, `om_variable_eur_per_kwh`. Optional: `bos_cost_share` (default 0.0).
    - Economics can be pre-configured at Atlas level via `atlas.configure_economics(...)`.
  - Timebase (`hours_per_year`) is configured at Atlas level via `atlas.configure_timebase(...)`.
  - Example:
    ```python
    atlas.configure_economics(discount_rate=0.05, lifetime_a=25)
    atlas.wind.compute(
        "lcoe",
        cf={"method": "hub_height_weibull"},
        economics={"om_fixed_eur_per_kw_a": 20, "om_variable_eur_per_kwh": 0.008},
    )
    ```
- `min_lcoe_turbine`
  - Same grouped spec API as `lcoe`
- `optimal_power`
  - Same grouped spec API as `lcoe`
- `optimal_energy`
  - Same grouped spec API as `lcoe`

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
- `atlas.landscape.convert_units(variable, to_unit, *, from_unit=None, inplace=False)`
  - convert a landscape variable to a different unit.
  - `variable`: variable name in `atlas.landscape.data`.
  - `to_unit`: target unit string (e.g., `"km"`, `"ft"`).
  - `from_unit`: source unit override; if `None`, reads from variable's `units` attr.
  - `inplace=True` stages the converted DataArray; `inplace=False` returns it.
  - conversion is dask-friendly (lazy arrays stay lazy).
- `atlas.build_clc(source="clc2018", url=None, force_download=False, force_prepare=False)`
  - prepares CLC cache aligned to wind/GWA grid.
  - with `url=None`, CLC2018 auto-download uses the CLMS API prepackaged-download workflow and expects CLMS auth via
    `CLEO_CLMS_ACCESS_TOKEN` or service key envs (`CLEO_CLMS_SERVICE_KEY_JSON`, `CLEO_CLMS_SERVICE_KEY_PATH`, `CLMS_API_SERVICE_KEY`).
  - multiband rendered inputs (for example RGB/RGBA imagery) are rejected; CLEO expects single-band categorical CLC class rasters.
- `atlas.landscape.add_clc_category(categories, *, name=None, source="clc2018", if_exists="error")`
  - `categories="all"`: full categorical layer (`land_cover` default name).
  - `categories=int`: single binary CLC-code mask.
  - `categories=list[int]`: combined binary mask (requires `name`).
  - returns `LandscapeAddResult`.

#### CLC Authentication Setup (CLMS)

When `atlas.build_clc(url=None)` needs to download CLC automatically, CLEO authenticates against the CLMS API.

Get credentials (service key) from CLMS:

1. Sign in at the Copernicus Land Monitoring Service portal (`https://land.copernicus.eu`).
2. Open your profile/settings page and create an API token/service key.
3. Save the downloaded JSON key file (or copy its JSON payload).

Inject credentials into CLEO (choose one method):

- Method 1: pass an existing bearer token (short-lived)

```bash
export CLEO_CLMS_ACCESS_TOKEN="<access-token>"
```

- Method 2: point CLEO to a service-key JSON file

```bash
export CLEO_CLMS_SERVICE_KEY_PATH="/absolute/path/to/clms_service_key.json"
# equivalent alias also supported:
export CLMS_API_SERVICE_KEY="/absolute/path/to/clms_service_key.json"
```

- Method 3: provide service-key JSON inline (for ephemeral environments/CI)

```bash
export CLEO_CLMS_SERVICE_KEY_JSON='{"service_name":"...","secret":"...","username":"..."}'
```

Then run:

```python
atlas.build_clc(source="clc2018")
```

Notes:

- If both are set, `CLEO_CLMS_ACCESS_TOKEN` is used first.
- Direct service-key login in CLEO expects JSON fields `service_name`, `secret`, and `username`.
- If your CLMS key uses another schema (for example `client_id`/`private_key`), mint a bearer token externally and set `CLEO_CLMS_ACCESS_TOKEN`.
- If no CLMS credential env var is set and `url=None`, `build_clc` raises an explicit authentication error.
- You can always bypass CLMS auth by passing an explicit `url=...` to `build_clc`.
- CLMS references: `https://land.copernicus.eu/en/how-to-guides/how-to-download-spatial-data/how-to-download-m2m` and `https://eea.github.io/clms-api-docs/authentication.html`.

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
- Distance variables materialized in area stores are area-local artifacts and may disappear when area stores are rebuilt from base stores.

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

### Unit Conversion

Wind and landscape domains support unit conversion via `convert_units()`:

```python
# Convert wind speed from m/s to km/h (in-place, visible in atlas.wind.data)
atlas.wind.convert_units("mean_wind_speed", "km/h", inplace=True)

# Get converted DataArray without modifying store
power_mw = atlas.wind.convert_units("optimal_power", "MW")

# Convert distance from m to km
atlas.landscape.convert_units("distance_roads", "km", inplace=True)

# Override source unit (useful for data with missing/wrong unit attr)
dist_ft = atlas.landscape.convert_units("elevation", "ft", from_unit="m")
```

Unit metadata is stored in the `units` attr. Conversion is dask-friendly (lazy arrays stay lazy).

**Note:** Currency units (e.g., `EUR/MWh`) are stored as string labels but cannot be converted by pint.
For LCOE unit changes, export and post-process manually.

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
- `atlas.clean_areas(area=None, older_than=None, include_incomplete=True)`
  - cleanup area materialization stores.

Example:

```python
run = atlas.wind.compute("capacity_factors", method="hub_height_weibull", air_density=True)
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
        {"label": "baseline", "kwargs": {"method": "rotor_node_average"}},
        {"label": "candidate_rews7", "kwargs": {"method": "hub_height_weibull_rews_scaled", "rews_n": 7}},
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

## Troubleshooting

### GDAL/PROJ Installation Issues

**Error: `rasterio` or `pyproj` fails to install**

Ensure GDAL/PROJ system libraries are installed before pip install:
```bash
# Ubuntu/Debian
sudo apt-get install -y libgdal-dev gdal-bin libproj-dev proj-bin

# macOS
brew install gdal proj

# Then retry
pip install rasterio pyproj
```

**Error: `GDAL not found` or `Could not find PROJ`**

Set environment variables to help pip find the libraries:
```bash
# Find GDAL config
export GDAL_CONFIG=$(which gdal-config)

# On macOS with Homebrew
export CFLAGS="-I$(brew --prefix)/include"
export LDFLAGS="-L$(brew --prefix)/lib"
```

### CLC Authentication Issues

**Error: `CLMS authentication cannot be resolved`**

CLEO needs CLMS credentials to download CLC data. See the [CLC Authentication Setup](#clc-authentication-setup-clms) section.

Quick fix: Set one of these environment variables:
```bash
export CLEO_CLMS_SERVICE_KEY_PATH="/path/to/clms_service_key.json"
# or
export CLEO_CLMS_ACCESS_TOKEN="<your-access-token>"
```

**Alternative: Manual download**

Download CLC data manually from [Copernicus Land Monitoring Service](https://land.copernicus.eu/en/products/corine-land-cover) and pass the URL or local path:
```python
atlas.build_clc(url="/path/to/clc_raster.tif")
```

### Memory Issues

**Error: `MemoryError` or system becomes unresponsive during `build()` or `compute()`**

CLEO processes large rasters. For memory-constrained systems:

1. **Use smaller chunk sizes:**
   ```python
   atlas = Atlas(..., chunk_policy={"y": 512, "x": 512})
   ```

2. **Process by area instead of full country:**
   ```python
   atlas.select(area="Wien", inplace=True)
   atlas.build()
   ```

3. **Use processes backend for better memory isolation:**
   ```python
   atlas = Atlas(..., compute_backend="processes", compute_workers=2)
   ```

### GWA Download Issues

**Error: `Required GWA file not found`**

CLEO auto-downloads missing GWA data on first `build()`. If download fails:

1. Check internet connectivity
2. Verify the country code is valid (ISO 3166-1 alpha-3)
3. Manual download: Get GWA data from [Global Wind Atlas](https://globalwindatlas.info/en/download/gis-files) and place in `<workdir>/data/raw/<ISO3>/`

### Chunk Policy Warnings

**Warning: `Stored chunk policy {...} differs from configured policy`**

This warning appears when opening a store with different chunk settings than it was created with. Options:

1. **Ignore** - CLEO uses stored chunks for optimal read performance
2. **Rebuild stores** - Delete `wind.zarr`/`landscape.zarr` and re-run `atlas.build()` with desired chunk policy
3. **Set matching policy:**
   ```python
   atlas = Atlas(..., chunk_policy={"y": 512, "x": 512})  # Match stored chunks
   ```

## Citation

If you use CLEO in your research, please cite:

```bibtex
@article{wehrle2024inferring,
  title={Inferring local social cost from renewable zoning decisions. Evidence from Lower Austria's wind power zoning},
  author={Wehrle, Sebastian and Regner, Peter and Morawetz, Ulrich B. and Schmidt, Johannes},
  journal={Energy Economics},
  volume={139},
  pages={107865},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.eneco.2024.107865}
}
```

See `CITATION.cff` for machine-readable citation metadata.
