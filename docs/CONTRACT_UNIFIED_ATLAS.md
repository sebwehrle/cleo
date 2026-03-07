# Unified Atlas Contract

Status: **Normative**
Scope: This document defines **both** (A) the stable *user-facing API* and (B) the *architectural / data invariants* of Cleo’s unified atlas.

**No backward compatibility is required.** The codebase is expected to change to satisfy this contract.

---

## 0. Core concepts (terminology)

- **Base stores**: country-wide canonical data, written by `Atlas.build()`.
- **Area selection**: an optional NUTS area identifier (e.g. `"AT13"`) that defines a *subsetting mask*.
- **Derived area stores**: area-scoped data products (metrics and user-added rasters) written on demand, keyed by the current area selection.
- **`atlas.<domain>.data` is the primary interface**: computed or loaded data must surface there as an `xarray.Dataset`.

---

## A. Public API

### A1. Public entry point

```python
from cleo import Atlas
```

Normative:
- Public API stability is defined by the `Atlas`/domain/result contracts in section A.
- Internal modules are not part of the public compatibility surface (for example `cleo.assess`, `cleo.loaders`, `cleo.unification.*`, `cleo.clc`).

---

### A2. `Atlas` construction

```python
atlas = Atlas(
    path,
    country="AUT",          # ISO3
    crs="epsg:3035",        # canonical projected CRS for stores
    chunk_policy={"y": 1024, "x": 1024},  # optional
    compute_backend="serial",             # optional: serial|threads|processes|distributed
    compute_workers=None,                 # optional local worker cap
    area=None,            # optional initial area selection (see A4)
    results_root=None,      # optional custom results directory
    fingerprint_method="path_mtime_size", # optional unification fingerprint policy
)
```

Normative:
- `area` is optional at construction time; it must also be settable later (A4).
- Construction performs **no heavy I/O**.

---

### A3. Turbines for wind materialization (optional)

```python
atlas.configure_turbines(["Enercon.E40.500", "Vestas.V112.3075"])
```

Normative:
- This config affects **wind base-store materialization only** (what ends up in `wind.zarr`).
- If not configured, **all available turbines** (all `*.yml` turbine definitions present in the turbine registry) are materialized.
- Config validation is syntactic only (non-empty; strings; stripped non-empty; no duplicates; order preserved).
- Changing configured turbines changes the **wind base-store inputs identity** so the next `build()` refreshes wind if needed.

---

### A3.1. Timebase configuration (optional)

```python
atlas.configure_timebase(hours_per_year=8760.0)
```

Normative:
- Default `hours_per_year` is **8766.0** when not configured (accounts for leap years).
- Configured timebase is preserved across `select(..., inplace=False)` clones.
- `compute("lcoe", hours_per_year=...)` must raise `ValueError` directing user to `configure_timebase()`.
- **Physics metrics** (`capacity_factors`, `wind_speed`) must NOT carry `cleo:hours_per_year` attr.
- **Economics metrics** (`lcoe`, `min_lcoe_turbine`, `optimal_power`, `optimal_energy`) must carry `cleo:hours_per_year` attr.
- Timebase affects LCOE-family metrics only; changing timebase does NOT invalidate cached `capacity_factors`.

---

### A3.2. Economics configuration (optional)

```python
atlas.configure_economics(
    discount_rate=0.05,
    lifetime_a=25,
    om_fixed_eur_per_kw_a=20.0,
    om_variable_eur_per_kwh=0.008,
    bos_cost_share=0.0,
    grid_connect_cost_eur_per_kw=50.0,  # or 0.0 for paper qLCOE
)
```

Normative:
- All parameters are optional; multiple calls merge values (later calls override earlier ones).
- `atlas.economics_configured` returns configured dict or `None` if not configured.
- Configured economics is preserved across `select(..., inplace=False)` clones.
- LCOE-family metrics resolve economics via: baseline config + per-call `economics={...}` overrides.
- Required economics fields (`discount_rate`, `lifetime_a`, `om_fixed_eur_per_kw_a`, `om_variable_eur_per_kwh`) must be present in effective resolution; missing fields raise explicit error.
- `bos_cost_share` defaults to `0.0` when not specified (all CAPEX is turbine/location-independent).
- `grid_connect_cost_eur_per_kw` defaults to `50.0` when not specified (Austrian regulation §54 ElWOG). Set to `0.0` to exclude grid connection costs (e.g., for paper qLCOE without grid costs).

---

### A4. Area selection (optional, may change over time)

Typical usage:

```python
atlas.select(area="Niederösterreich", inplace=True)   # sets active area selection (persistent)
atlas.select(area=None, inplace=True)       # clears area selection (full-country view)
```

Normative:
- Area selection must be changeable **at any time** after construction.
- Selection API supports optional disambiguation and copy-vs-inplace semantics:
  - `select(area=..., nuts_level=0|1|2|3|None, inplace=False|True)`
- Area selection affects:
  - *what computations operate on* (they apply a mask/subset), and
  - *where derived outputs are written on disk* (area-scoped stores).
- Area selection **must not rewrite** the base stores.

---

### A5. Materialization

```python
atlas.build()
```

Normative:
- Creates/updates **base stores**:
  - `wind.zarr` (country-wide wind inputs, incl. turbine metadata needed for hub-height computations),
  - `landscape.zarr` (country-wide landscape inputs, incl. `valid_mask`).
- Must be **idempotent**.
- When required GWA wind rasters are missing, build must attempt deterministic
  download of the required file set for the configured country before failing.
- When area-aware build paths require NUTS boundaries and local NUTS files
  are missing, build must attempt deterministic NUTS download/extract before
  failing.
- Must be offline-safe if local raw inputs are present (and may use CopDEM tile download if configured/required by contract B).

---

### A6. Domains and the primary data interface

Supported domains:

```python
atlas.wind.data         # xr.Dataset
atlas.landscape.data    # xr.Dataset
```

Normative:
- The standard user workflow is via `atlas.wind.data` and `atlas.landscape.data`.
- Assignment-style usage (e.g. `wind = atlas.wind`) may be supported, but is **not** the canonical usage described in this contract.

---

## A7. Wind: turbine selection for computations

```python
atlas.wind.turbines                 # tuple[str, ...]  (inventory from base store)
atlas.wind.select(turbines=[...])   # sets persistent selection by turbine IDs
atlas.wind.select(turbine_indices=[...])  # sets persistent selection by indices into atlas.wind.turbines
atlas.wind.clear_selection()
atlas.wind.clear_computed()
atlas.wind.selected_turbines        # tuple[str, ...] | None
```

Normative:
- Selection is **persistent** (applies to subsequent compute calls) and stored in the `Atlas`/domain state.
- Exactly one of `turbines` or `turbine_indices` must be provided to `select(...)`.
- `turbines` accepts non-empty `list|tuple[str]` only (a plain string is invalid).
- `turbine_indices` accepts non-empty `list|tuple[int]`, with bounds and duplicate checks.
- `select(...)` and `clear_selection()` mutate selection state in place and return `None`.
- `clear_computed()` clears transient computed overlays from `atlas.wind.data` and returns `None`.
- Any compute call may also accept an explicit `turbines=[...]` override; explicit arguments override the persistent selection for that call only.

---

## A8. Wind: single compute entry point + materialization into `.data`

Breaking-change notes:
- For `metric="capacity_factors"`, the flat kwarg `mode` is replaced by `method`.
- Capacity-factor method names are renamed from legacy strings (`"hub"`, `"rews"`, `"direct_cf_quadrature"`, `"momentmatch_weibull"`) to the explicit vocabulary defined in A9.
- The split wind-speed metrics `mean_wind_speed` and `rews_mps` are replaced by one `metric="wind_speed"` surface with a `method` selector.

### Compute (lazy)

```python
run = atlas.wind.compute(
    metric="capacity_factors",
    method="rotor_node_average",
    interpolation="auto",
    air_density=False,
    rews_n=12,
    loss_factor=1.0,
)
da = run.data    # xr.DataArray (dask-backed if dask is configured)
```

Composed metric example (grouped dependency specs):

```python
run = atlas.wind.compute(
    metric="lcoe",
    cf={
        "method": "rotor_node_average",
        "interpolation": "auto",
        "air_density": False,
        "rews_n": 12,
        "loss_factor": 1.0,
    },
    economics={
        "discount_rate": 0.06,
        "lifetime_a": 25,
        "om_fixed_eur_per_kw_a": 25.0,
        "om_variable_eur_per_kwh": 0.009,
        "bos_cost_share": 0.28,  # 28% of CAPEX is BOS (location-dependent)
    },
)
da = run.data
```

### Materialize (write result into `.data`)

```python
da_materialized = atlas.wind.compute(
    metric="capacity_factors",
    method="rotor_node_average",
    interpolation="auto",
    air_density=False,
    rews_n=12,
    loss_factor=1.0,
).materialize()
```

Normative:
- `WindDomain.compute(metric=..., **kwargs)` returns a **DomainResult** object with:
  - `.data -> xr.DataArray` (lazy by default),
  - `.materialize(overwrite: bool = True, allow_method_change: bool = False) -> xr.DataArray`,
  - `.persist(run_id: str | None = None, params: dict | None = None, metric_name: str | None = None) -> Path`.
- `atlas.wind.compute(...)` is the primary public entrypoint for wind metrics.
- `compute(...)` also stages a lazy normalized overlay immediately in `atlas.wind.data[metric_name]` without writing to the wind store.
- `compute(...)` must reject unknown metric-specific parameters; silent unknown-kwarg drops are prohibited.
- Parameter-shape convention for wind metrics:
  - Leaf metrics (`wind_speed`, `capacity_factors`) use flat kwargs.
  - Composed metrics (`lcoe`, `min_lcoe_turbine`, `optimal_power`, `optimal_energy`) use grouped dependency specs:
    - `cf={...}` for capacity-factor dependency knobs.
    - `economics={...}` for economic assumptions.
  - For composed metrics, top-level flat dependency knobs are not part of the public contract.
- `.materialize()` must:
  1) write the result into the active wind store (area store when a area is selected, base store otherwise), and
  2) surface it immediately as `atlas.wind.data[metric_name]`, and
  3) return the materialized `xr.DataArray`.
- When replacing materialized `capacity_factors`, a capacity-factor method change requires `allow_method_change=True`.
- Staged wind overlays are transient: `select(...)`, `build()`, `build_canonical()`, and `build_clc()` clear them.
- Successful `.materialize()` clears the staged overlay for that metric.

---

## A9. Wind metrics (v1.3)

### `metric="capacity_factors"`

Parameters:
- `turbines`: optional override list (see A7).
- `air_density`: `bool`, default `False`.
- `method`: `str`, one of:
  - `"rotor_node_average"` (default) - weighted average of node-wise capacity factors across rotor disk using Gauss-Legendre quadrature.
  - `"rotor_moment_matched_weibull"` - single effective Weibull obtained by moment-matching across rotor disk nodes.
  - `"hub_height_weibull"` - single Weibull distribution queried at turbine hub height. No rotor averaging.
  - `"hub_height_weibull_rews_scaled"` - hub-height Weibull with scale parameter adjusted by a REWS moment factor.
- `rews_n`: `int`, default `12`.
  Used by:
  - rotor-aware methods as rotor quadrature node count;
  - `hub_height_weibull_rews_scaled` as the REWS moment-factor integration resolution.
  Ignored by:
  - `hub_height_weibull`.
- `interpolation`: `str`, default `"auto"`, one of:
  - `"auto"` - behavior-preserving resolver by method family.
  - `"ak_logz"` - direct interpolation of `A` and `k` in `ln(z)` space.
  - `"mu_cv_loglog"` - interpolate `mu` and `CV` in log-log space, invert `CV -> k`, and derive `A`.
- `loss_factor`: `float`, default `1.0` (applied multiplicatively).

Normative physics/semantics:
- Capacity factors are produced **per turbine**, with the turbine definition supplying hub height and other turbine-specific parameters.
- Rotor-aware methods (`rotor_node_average`, `rotor_moment_matched_weibull`) use height-continuous evaluation over rotor disk heights with vertical policy.
- Hub-height methods (`hub_height_weibull`, `hub_height_weibull_rews_scaled`) use hub-height interpolation path with no extrapolation.
- `interpolation="auto"` preserves current behavior mapping:
  - hub-height methods -> `ak_logz`
  - rotor-aware methods -> `mu_cv_loglog`
- Explicit interpolation selection is supported for all method families:
  - `ak_logz` keeps no-extrapolation semantics and raises if any required query
    height lies outside the available GWA height range.
  - `mu_cv_loglog` uses the vertical-policy path, including configured
    extrapolation behavior where applicable.
- This metric must not accept a free `height=` parameter (hub height is implied by the turbine).

Intermediate requirement:
- When `air_density=True`, the wind dataset must be able to hold air density fields as:
  - `atlas.wind.data["rho"]` with dims `("height", "y", "x")` (height in meters; values correspond to the hub-height bins used).

#### Scientific limitations

CLEO computes rotor-level capacity factors using surrogate families because only
height-wise marginal Weibull distributions are available from the GWA data source.
The four capacity-factor methods represent different approaches to constructing a
rotor-level wind speed or capacity factor surrogate from these marginal distributions.

No method provides exact rotor-level capacity-factor derivation from a fully
specified joint stochastic vertical wind field, because such information is not
present in the input data.

### `metric="wind_speed"`

Parameters:
- `method`: `str`, one of:
  - `"height_weibull_mean"` - compute Weibull mean wind speed at a requested height.
  - `"rotor_equivalent"` - compute rotor-equivalent wind speed for selected turbines.
- `interpolation`: `str`, default `"auto"`.
  - `"auto"` - behavior-preserving resolver by method family.
  - `"ak_logz"` - explicit hub-height interpolation backend.
  - `"mu_cv_loglog"` - explicit policy-style interpolation backend.

Normative:
- `method="height_weibull_mean"`:
  - required kwargs: `height`
  - forbidden kwargs: `turbines`, `rews_n`, `air_density`
  - output variable name: `mean_wind_speed`
  - output dims: `("height", "y", "x")`
  - `compute(..., height=<h>)` returns a singleton height slice (`height=[h]`).
- `method="rotor_equivalent"`:
  - required kwargs: `turbines` via selection or explicit override
  - optional kwargs: `air_density`, `rews_n`, `interpolation`
  - forbidden kwargs: free `height`
  - output variable name: `rotor_equivalent_wind_speed`
  - output dims: `("turbine", "y", "x")`
- Must operate on the current area selection if set (A4).
- `method="height_weibull_mean"` materialization stores/updates height slices into one
  `mean_wind_speed(height,y,x)` variable.
- `method="rotor_equivalent"` materialization stores/updates
  `rotor_equivalent_wind_speed(turbine,y,x)`.
- If an existing store contains legacy 2D `mean_wind_speed(y,x)`,
  materialization of `method="height_weibull_mean"` must fail with an explicit migration error.
- For `method="rotor_equivalent"`, interpolation follows the same rotor-aware
  backend rules as `capacity_factors(method="rotor_node_average")`:
  - `interpolation="auto"` resolves to `mu_cv_loglog`
  - `interpolation="ak_logz"` is allowed and preserves no-extrapolation
    semantics across the rotor query heights
  - `interpolation="mu_cv_loglog"` is allowed and uses the vertical-policy path

### Additional wind metrics (implemented)

- `metric="lcoe"`
  - Requires turbines and grouped dependency specs:
    - `cf` (optional): `method`, `interpolation`, `rews_n`, `air_density`, `loss_factor`.
    - `economics` (required by effective resolution): `om_fixed_eur_per_kw_a`, `om_variable_eur_per_kwh`, `discount_rate`, `lifetime_a`.
  - Optional in `economics`: `bos_cost_share` (default 0.0, meaning all CAPEX is turbine/location-independent).
  - `hours_per_year` must not be passed to `compute("lcoe", ...)`; it belongs to Atlas-level timebase assumptions.
- `metric="min_lcoe_turbine"`, `metric="optimal_power"`, `metric="optimal_energy"`
  - Same grouped `cf` / `economics` contract as `lcoe`.

---

## A10. Landscape

Primary interface:

```python
atlas.landscape.data["valid_mask"]
```

Adding rasters (area-scoped derived data):

```python
op = atlas.landscape.add(name="my_raster", source_path="/path/to/raster.tif")
da_stage = op.data
da_mat = op.materialize()
atlas.landscape.clear_staged()
```

Rasterizing vector sources (area-scoped derived data):

```python
op = atlas.landscape.rasterize(
    "/path/to/stays.geojson",
    name="overnight_stays",
    column="overnight_stays",
    all_touched=False,
)
da_stage = op.data
da_mat = op.materialize()
```

Distance-to-feature compute (area-scoped derived data):

```python
run = atlas.landscape.compute(
    metric="distance",
    source=["roads_mask", "water_mask"],
    name=["distance_roads", "distance_water"],  # optional; default distance_<source>
    if_exists="error",
)
ds_stage = run.data
ds_mat = run.materialize()
```

Normative:
- `add(...)` stages a lazy raster candidate and returns an operation object with `.data` and `.materialize(...)`.
- `add(...)` is raster-only (`kind="raster"`); vector sources must use `rasterize(...)`.
- `rasterize(...)` stages a lazy vector-rasterized candidate and returns an operation object with `.data` and `.materialize(...)`.
- `rasterize(...)` accepts path-like vector sources or `geopandas.GeoDataFrame`.
- `rasterize(column=None)` burns coverage (`1.0`); `column="<numeric_column>"` burns numeric column values.
- Staged variables must surface in `atlas.landscape.data[name]` before materialization.
- If a area selection is active, added rasters must be written to the **derived area store** for that area.
- `clear_staged()` must remove staged landscape overlays from `atlas.landscape.data`.
- Staged landscape overlays are transient: `select(...)`, `build()`, `build_canonical()`, and `build_clc()` clear them.
- Successful landscape `.materialize()` clears the staged overlay for that variable.
- `LandscapeDomain.compute(metric="distance", ...)` must support one or many source variables in one call:
  - `source: str | list[str] | tuple[str, ...]`
  - `name: None | str | list[str] | tuple[str, ...]` (defaults to `distance_<source>`)
  - `if_exists: "error" | "replace" | "noop"`
- Distance compute returns a batch result object with:
  - `.data -> xr.Dataset` (staged distance variables),
  - `.materialize(...) -> xr.Dataset` (writes staged distance variables to active landscape store).
- Distance sources are store-backed landscape variables on the active store grid (not staged-only overlays).
- Distance semantics:
  - nearest finite positive cell (`isfinite(source) & source > 0`),
  - units in meters,
  - outside `valid_mask` is `NaN`,
  - no target cells in valid area -> all `NaN` in valid area,
  - all valid cells are targets -> all zeros in valid area.
- Distance computation is eager by design for correctness (global Euclidean distance transform).
- `if_exists="error"` is atomic preflight for batch conflicts (no writes if any conflict).
- `if_exists="noop"` for distance requires exact spec match via `cleo:distance_spec_json`; missing/invalid/different spec must raise and require `if_exists="replace"`.
- Distance variables materialized into area stores are area-local artifacts and are not guaranteed to persist when area stores are rebuilt from base stores.

---

## A11. Results: persist, open, export

Persist arbitrary results:

```python
store_path = atlas.persist(metric_name="capacity_factors", obj=da_or_ds, params={"air_density": False})
```

Open:

```python
obj = atlas.open_result(run_id, metric_name="capacity_factors")  # xr.Dataset
```

Export NetCDF:

```python
atlas.export_result_netcdf(run_id, metric_name="capacity_factors", out_path="out.nc")
```

Normative:
- Results are stored under a **results store** keyed by `run_id` (see B4).
- Export must be NetCDF4-compatible (including handling of non-numeric coordinate types).
- `run_id`/`metric_name` are validated as path tokens (no separators/traversal).

---

## A12. Consolidated Analysis Export

```python
atlas.export_analysis_dataset_zarr(
    path,
    domain="both",           # "wind" | "landscape" | "both"
    include_only=None,       # optional: list of variables to export
    prefix=True,             # prefix vars with "wind__" / "landscape__" for domain="both"
    exclude_template=True,   # exclude template variables
    compute=True,            # compute dask arrays before writing
)
```

Normative:
- Creates a schema-versioned Zarr store with provenance tracking at `path`.
- `path` must end with `.zarr`.
- Raises `FileExistsError` if the export store already exists.
- Raises `ValueError` if stores are not ready (call `atlas.build()` first).
- When `domain="both"`:
  - If `prefix=True`, variables are prefixed with `wind__` / `landscape__`.
  - If `prefix=False`, raises `ValueError` on variable name collisions.
  - `include_only` with prefixed names (e.g., `["wind__capacity_factors", "landscape__valid_mask"]`).
- When `domain="wind"` or `domain="landscape"`:
  - `prefix` parameter is ignored.
  - `include_only` uses raw variable names.
- `exclude_template=True` removes `template` variables from the export.
- `compute=True` (default) forces dask computation before write for reliability.
- Returns `Path` to the created Zarr store.

Store attributes:
- `store_state`: `"complete"` on successful write
- `schema_version`: integer from `cleo.contracts.ANALYSIS_EXPORT_SCHEMA_VERSION`
- `created_at`: ISO 8601 timestamp
- `cleo:package_version`: cleo version string
- `export_spec_json`: JSON encoding of export parameters
- `upstream_provenance_json`: JSON encoding of upstream store provenance (grid_id, inputs_id, etc.)

Validation:
- Pre-write: `validate_dataset(ds, kind="export")` before write
- Post-write: `validate_store(path, kind="export")` after atomic write

---

# B. Architecture and invariants

## B1. On-disk store layout (normative)

Given `atlas.path = <ROOT>`:

Base stores:
- `<ROOT>/wind.zarr`
- `<ROOT>/landscape.zarr`

Derived area stores:
- `<ROOT>/areas/<area_id>/wind.zarr`
- `<ROOT>/areas/<area_id>/landscape.zarr`

Results stores:
- `<ROOT>/results/<run_id>/<name>.zarr`
- metadata/provenance fields are stored in Zarr attrs (no required `meta.json` file).

Area ID rules:
- If `atlas.area` is `None`, materialized wind metrics are written to the base wind store (`<ROOT>/wind.zarr`).
- Otherwise `area_id` is the provided NUTS id (e.g. `"AT13"`).

---

## B2. Identity, determinism, and manifests

- Each store must have an **inputs identity** (`inputs_id`) computed from deterministic inputs:
  - country, CRS, resolution, upstream data fingerprints,
  - for wind base store: the configured turbine list (or `"default"`),
  - for area derived stores: area selection id + area-shape fingerprint.
- Identity computation must be stable across runs:
  - no timestamps,
  - stable ordering (e.g. turbine list order preserved),
  - JSON serialization must be stable.

Each store must record provenance (manifest) in a way that is:
- **deterministic**, and
- **Zarr v3 compatible** (see B3).

---

## B3. Zarr v3 compatibility requirements (normative)

The code must avoid writing data types that currently trigger Zarr v3 “unstable specification” warnings.

Therefore:

1) **No consolidated metadata dependency**
- Do not rely on `.zmetadata` or “consolidated metadata” for correctness.
- Opening stores must work with `consolidated=False`.

2) **No string-like array dtypes in stored arrays**
- Stored arrays must not have dtype kind `"U"`, `"O"`, or `"S"`.

3) **String metadata must be stored in attributes**
- Any string lists (e.g. turbine ids, manifest sources/variables) must be stored as JSON in root/group attributes, e.g.:
  - `cleo_turbines_json`
  - `cleo_manifest_sources_json`
  - `cleo_manifest_variables_json`

---

## B4. Region reuse (no duplication of base stores)

- Base stores are country-wide and reusable across any number of areas.
- Area selection only affects derived area stores and compute masking.
- It must be possible to:
  - materialize base stores once (e.g. Austria),
  - switch area selection (e.g. `AT12`, then `AT13`),
  - compute/materialize metrics for each area into separate derived area stores.

---

## B5. Elevation sourcing (normative)

Landscape materialization must source elevation deterministically using:

1) Prefer **local elevation GeoTIFF** if present:
- `<ROOT>/data/raw/<ISO3>/<ISO3>_elevation_w_bathymetry.tif`

2) Otherwise, build from **Copernicus DEM tiles (CopDEM)**:
- tile planning must be deterministic from bbox in EPSG:4326,
- tile ids must be lexicographically ordered,
- mosaic must be aligned/reprojected to the wind reference grid.

---

## B6. Dask / laziness invariants

- Wind-domain `compute(...).data` is lazy if Dask is configured (unless explicitly configured otherwise by the caller flow).
- Landscape `compute(metric="distance", ...)` is an explicit eager exception for correctness (global Euclidean distance transform).
- `.materialize()` is the canonical user-triggered materialization step that may compute.

---

## B7. I/O Layer Boundaries (normative)

- `cleo/unification/**` is the canonical location for raw geospatial/base-store/area-store I/O.
- `cleo/results.py` is the canonical location for results-store persistence/open/export internals.
- `cleo/exports.py` is the canonical location for consolidated analysis export I/O (schema-versioned Zarr exports).
- `cleo/atlas_policies/**` is policy-only and must not perform direct raw/store/filesystem I/O.
- `cleo/atlas.py` is orchestration/control-plane only and should delegate storage operations to dedicated I/O helpers.
- `cleo/domains.py` should access stores through storage helper functions (not direct raw-I/O call sites).
- `cleo/assess.py` remains pure compute only: no raw/store/network I/O and no eager evaluation triggers.

---

## B8. Contract checking

The repository must provide a contract check that:
- validates store schemas and required variables,
- validates identity/manifest invariants,
- exercises the v1 happy-path workflow end-to-end offline.

---

## B9. Unit Metadata Contract (normative)

### B9.1 Canonical Attr Key

- The canonical unit metadata attr key is `units` (plural).
- Legacy attr key `unit` (singular) is supported for reads during migration.
- Writing unit metadata must use `units` only.
- If both `unit` and `units` exist on the same DataArray and differ, operations must raise `ValueError`.

### B9.2 Canonical Units for Public Variables

| Variable | Canonical Unit | Notes |
|----------|----------------|-------|
| `wind_speed` | `m/s` | Method-dependent: height-specific or rotor-equivalent |
| `capacity_factors` | `1` | Dimensionless fraction |
| `lcoe` | `EUR/MWh` | Levelized cost of electricity |
| `min_lcoe_turbine` | *(none)* | Index, dimensionless |
| `optimal_power` | `kW` | Rated power of optimal turbine |
| `optimal_energy` | `GWh/a` | Annual energy output |
| `distance_*` | `m` | Euclidean distance |
| `elevation` | `m` | Above sea level |
| `turbine_capacity` | `kW` | Rated power |
| `turbine_hub_height` | `m` | Above ground |
| `turbine_rotor_diameter` | `m` | Rotor diameter |
| `weibull_A` / `weibull_a` | `m/s` | Scale parameter |
| `weibull_K` / `weibull_k` | `1` | Shape parameter (dimensionless) |
| `rho` | `kg/m³` | Air density |

### B9.3 Conversion Invariants

- Unit conversions must be multiplicative (no affine transforms like °C ↔ K).
- Conversion must preserve dask laziness (scalar factor multiplication only).
- Conversion must preserve all non-unit attrs.
- The canonical unit utilities are in `cleo/units.py`.

### B9.4 Storage Policy

- Variables with defined canonical units should be stored in those units.
- Users may convert after loading if different units are needed for analysis.

---

# C. QA and Operability Addendum (Normative)

This section codifies quality assurance expectations, scope boundaries, and export contracts.

---

## C1. Scope Boundary

CLEO prepares analysis-ready wind/landscape rasters and tabular exports for subsequent analysis workflows.

**In scope:**
- Wind resource metrics: mean wind speed, capacity factors, REWS, and derived energy/power metrics
- Economics metrics: LCOE, optimal turbine selection, optimal power/energy
- Landscape processing: rasterization, distance transforms, CLC category extraction
- Data export: `flatten()` for tabular handoff, `persist()` for result storage, NetCDF export

**Out of scope:**
- Econometric regression/estimation (logit, probit, GLM, spatial regression, etc.)
- Coefficient-derived outputs or statistical inference
- Model fitting or parameter estimation beyond what is required for wind resource assessment

The package outputs (capacity factors, LCOE, flattened rasters) are designed as inputs to downstream econometric or optimization workflows, not as substitutes for them.

---

## C2. Supported Environment

### C2.1 Python Version

- **Minimum**: Python 3.11
- **Tested**: Python 3.11, 3.12 (CI matrix when available)

### C2.2 Operating System

- **Primary**: Linux (Ubuntu LTS)
- **Secondary**: macOS (development/testing)
- **Windows**: Not actively tested; may work but not guaranteed

### C2.3 System Dependencies

CLEO relies on geospatial libraries that require system-level dependencies:

- **GDAL/PROJ**: Required by `rasterio`, `pyproj`, `geopandas`
- **HDF5/NetCDF**: Required by `netcdf4`, `h5py`

These are typically installed via system package managers or a user-managed conda/mamba environment. Python dependency resolution is defined in `pyproject.toml`.

---

## C3. Dependency Constraint Policy

### C3.1 Version Bounds

Dependencies in `pyproject.toml` must specify:

- **Lower bounds**: Minimum versions known to work (based on CI/testing)
- **Upper bounds**: Protective caps for ecosystem components with known breaking changes (xarray, dask, zarr, geopandas)

### C3.2 Rationale Documentation

Significant version constraints should include inline comments explaining the rationale (e.g., API changes, dtype handling, deprecation cycles).

### C3.3 Lock Files

- `constraints-ci.txt` (optional): Fully pinned versions for CI reproducibility
- Production installs should use version ranges from `pyproject.toml`

---

## C4. CI Gates (Normative)

Continuous integration must enforce:

1. **Linting**: `ruff check` and `ruff format --check`
2. **Unit tests**: `pytest tests/unit`
3. **Integration tests**: `pytest tests/integration`
4. **Smoke tests**: `pytest tests/smoke`
5. **Build verification**: Package builds and imports successfully
6. **Documentation build**: `mkdocs build --strict`

CI failures block merge.

---

## C5. Export Contracts

### C5.1 Tabular Export: `flatten()`

`Atlas.flatten(domain, ...)` is the primary tabular handoff API for downstream analysis.

**Contract:**
- Returns a `pandas.DataFrame` with one row per valid grid cell
- Column naming follows `{domain}__{variable}` convention when `include_domain_prefix=True`
- Schema is deterministic given the same inputs and parameters
- `Atlas.validate_flatten_schema(df, required_columns)` validates presence of required columns

**Schema stability:**
- Column names for a given variable/domain remain stable across patch versions
- New columns may be added in minor versions
- Column removal or renaming requires major version bump

### C5.2 Result Persistence: `persist()` and `export_result_netcdf()`

`DomainResult.persist(...)` and `Atlas.export_result_netcdf(...)` handle result storage and export.

**Contract:**
- Persisted results are Zarr stores with `store_state="complete"` marker
- Results include provenance attrs: `created_at`, `cleo:package_version`, metric-specific parameters
- NetCDF export produces CF-compliant files where feasible

### C5.3 Consolidated Analysis Export: `export_analysis_dataset_zarr()`

`Atlas.export_analysis_dataset_zarr(...)` provides schema-versioned xarray exports with provenance tracking.

See **A12** for full API documentation.

**Contract:**
- Schema-versioned exports with `schema_version` attr (see `cleo.contracts.ANALYSIS_EXPORT_SCHEMA_VERSION`)
- `store_state="complete"` marker on successful write
- Provenance attrs: `created_at`, `cleo:package_version`, `export_spec_json`, `upstream_provenance_json`
- Domain prefixing for combined wind/landscape exports (`wind__`, `landscape__`)
- Collision detection when prefixing is disabled
- Pre-write validation via `validate_dataset(kind="export")`
- Post-write validation via `validate_store(kind="export")`

---

## C6. Determinism and Reproducibility

### C6.1 Deterministic Outputs

The following are **bitwise deterministic** given identical inputs:
- Store skeleton attrs and metadata
- Content hashes (`inputs_id`, `grid_id`)
- Manifest JSON serialization

### C6.2 Tolerance-Deterministic Outputs

Floating-point reductions (capacity factors, LCOE, etc.) are deterministic within tolerance:
- Results may vary at machine epsilon level across runs
- Results may vary with dask chunking at floating-point tolerance level
- Tests assert `allclose` with explicit tolerances (typically `rtol=1e-10`, `atol=1e-10`)

### C6.3 Chunk Sensitivity

Dask chunking may affect floating-point reduction order. Tests verify:
- Results are within tolerance across different chunk configurations
- No silent precision degradation from chunking choices

---

## C7. Schema Versioning (Normative)

### C7.1 Scope

Schema versioning applies ONLY to external-facing exports with promised schema stability:
- Consolidated analysis xarray exports (future `export_analysis_dataset_zarr` or equivalent)
- Flatten tabular exports with sidecar manifests (future `export_flatten` with parquet/csv)

Schema versioning does NOT apply to internal canonical stores:
- `wind.zarr` and `landscape.zarr` use `inputs_id` / `grid_id` / `unify_version` for content+code identity
- Internal stores do not promise read-compatibility across versions

### C7.2 Version Constants

Schema version constants are defined in `cleo/contracts.py`:

```python
ANALYSIS_EXPORT_SCHEMA_VERSION = 1  # xarray exports (zarr/netcdf)
FLATTEN_EXPORT_SCHEMA_VERSION = 1   # tabular exports (parquet/csv)
```

### C7.3 Semantic Definition

A schema version represents a **layout/interpretation contract** for exported data. The version guarantees that downstream parsers can rely on:
- Variable/column presence and naming
- Coordinate semantics (dimensions, ordering)
- Data types and units
- Required attribute presence and semantics

### C7.4 Schema-Breaking Changes (Require Version Bump)

The schema version MUST be incremented when any of the following changes occur:

1. **Removal or renaming** of an existing variable or column
2. **Dtype changes** that affect interpretation (e.g., `float64` -> `float32`, `int64` -> `int32`)
3. **Coordinate semantic changes** (e.g., cell-center vs. cell-edge interpretation)
4. **Unit changes** without backward-compatible conversion path
5. **Attribute semantic changes** that downstream parsers rely on
6. **Structural layout changes** (e.g., new required dimensions, changed dimension order)

### C7.5 Non-Breaking Changes (No Version Bump Required)

The schema version should NOT be bumped for:

1. **Addition of new optional** variables, columns, or attributes
2. **Internal refactoring** that preserves output byte-for-byte
3. **Performance optimizations** with identical output
4. **Documentation or metadata clarifications** that don't affect parsing
5. **New optional parameters** to export APIs

### C7.6 Export Attribute Requirements

Exports with schema versioning must include these root attributes:
- `schema_version`: integer matching the relevant constant from `cleo/contracts.py`
- `store_state`: `"complete"` on successful write (existing contract)
- `created_at`: ISO 8601 timestamp
- `cleo:package_version`: version string of cleo that produced the export

---

## C8. Store and Dataset Validation (Normative)

### C8.1 Validation Module

Centralized validation is provided by `cleo/validation.py`:

```python
from cleo.validation import validate_dataset, validate_store, ValidationError
```

### C8.2 `validate_dataset(ds, *, kind, deep=False)`

Validates an xarray Dataset against store schema expectations.

**Parameters:**
- `ds`: The xarray Dataset to validate
- `kind`: `"wind"` | `"landscape"` | `"export"` | `"result"` | `"generic"`
- `deep`: If `True`, perform additional coordinate/data checks (default `False`)

**Contract:**
- `deep=False` (default) checks only attrs, dims, coords, and variable presence
- `deep=False` MUST NOT trigger compute on dask-backed arrays
- `deep=True` may check coordinate monotonicity and sample data validity
- Raises `ValidationError` on validation failure

### C8.3 `validate_store(path, *, kind, allow_incomplete=False)`

Validates a Zarr store at the given path.

**Parameters:**
- `path`: Path to the Zarr store directory
- `kind`: `"wind"` | `"landscape"` | `"export"` | `"result"` | `"generic"`
- `allow_incomplete`: If `True`, allow stores with `store_state != "complete"`

**Contract:**
- Opens only store metadata via zarr (lightweight)
- Raises `FileNotFoundError` if store does not exist
- Raises `ValidationError` on validation failure

### C8.4 Required Schema by Store Kind

**Wind stores** require:
- Attrs: `store_state`, `grid_id`, `inputs_id`, `cleo_turbines_json`
- Variables: `weibull_A`, `weibull_k`, `power_curve`
- Dimensions: `y`, `x`, `turbine`, `height`

**Landscape stores** require:
- Attrs: `store_state`, `grid_id`, `inputs_id`
- Variables: `valid_mask`
- Dimensions: `y`, `x`

**Export stores** require:
- Attrs: `store_state`, `schema_version`, `created_at`

**Result stores** require:
- Attrs: `store_state`, `run_id`, `metric_name`, `created_at`

### C8.5 Integration Points

Domain store access (`WindDomain._store_data()`, `LandscapeDomain._store_data()`) uses `validate_dataset()` to provide centralized validation on store load.
