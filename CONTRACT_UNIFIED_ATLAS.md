# Unified Atlas Contract

Status: **Normative**  
Scope: This document defines **both** (A) the stable *user-facing API* and (B) the *architectural / data invariants* of Cleo’s unified atlas.

**No backward compatibility is required.** The codebase is expected to change to satisfy this contract.

---

## 0. Core concepts (terminology)

- **Base stores**: country-wide canonical data, written by `Atlas.build()`.
- **Region selection**: an optional NUTS region identifier (e.g. `"AT13"`) that defines a *subsetting mask*.
- **Derived region stores**: region-scoped data products (metrics and user-added rasters) written on demand, keyed by the current region selection.
- **`atlas.<domain>.data` is the primary interface**: computed or loaded data must surface there as an `xarray.Dataset`.

---

## A. Public API

### A1. Public entry point

```python
from cleo import Atlas
```

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
    region=None,            # optional initial region selection (see A4)
    results_root=None,      # optional custom results directory
    fingerprint_method="path_mtime_size", # optional unification fingerprint policy
)
```

Normative:
- `region` is optional at construction time; it must also be settable later (A4).
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

### A4. Region selection (optional, may change over time)

Typical usage:

```python
atlas.select(region="Niederösterreich", inplace=True)   # sets active region selection (persistent)
atlas.select(region=None, inplace=True)       # clears region selection (full-country view)
```

Normative:
- Region selection must be changeable **at any time** after construction.
- Selection API supports optional disambiguation and copy-vs-inplace semantics:
  - `select(region=..., region_level=1|2|3|None, inplace=False|True)`
- Region selection affects:
  - *what computations operate on* (they apply a mask/subset), and
  - *where derived outputs are written on disk* (region-scoped stores).
- Region selection **must not rewrite** the base stores.

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

### Compute (lazy)

```python
run = atlas.wind.compute(
    metric="capacity_factors",
    mode="direct_cf_quadrature",
    air_density=False,
    rews_n=12,
    loss_factor=1.0,
)
da = run.data    # xr.DataArray (dask-backed if dask is configured)
```

### Materialize (write result into `.data`)

```python
da_materialized = atlas.wind.compute(
    metric="capacity_factors",
    mode="direct_cf_quadrature",
    air_density=False,
    rews_n=12,
    loss_factor=1.0,
).materialize()
```

Normative:
- `WindDomain.compute(metric=..., **kwargs)` returns a **DomainResult** object with:
  - `.data -> xr.DataArray` (lazy by default),
  - `.materialize(overwrite: bool = True, allow_mode_change: bool = False) -> xr.DataArray`,
  - `.persist(run_id: str | None = None, params: dict | None = None, metric_name: str | None = None) -> Path`.
- `compute(...)` also stages a lazy normalized overlay immediately in `atlas.wind.data[metric_name]` without writing to the wind store.
- `.materialize()` must:
  1) write the result into the active wind store (region store when a region is selected, base store otherwise), and
  2) surface it immediately as `atlas.wind.data[metric_name]`, and
  3) return the materialized `xr.DataArray`.
- When replacing materialized `capacity_factors`, a mode change requires `allow_mode_change=True`.
- Staged wind overlays are transient: `select(...)`, `build()`, `build_canonical()`, and `build_clc()` clear them.
- Successful `.materialize()` clears the staged overlay for that metric.

Convenience wrappers:
- `atlas.wind.capacity_factors(...)` is a convenience wrapper equivalent to `compute(metric="capacity_factors", ...)` and returns the same kind of object supporting `.materialize()`.

---

## A9. Wind metrics (v1.3)

### `metric="capacity_factors"`

Parameters:
- `turbines`: optional override list (see A7).
- `air_density`: `bool`, default `False`.
- `mode`: `"direct_cf_quadrature"` (default), `"momentmatch_weibull"`, `"hub"`, or `"rews"`.
- `rews_n`: `int`, default `12` (rotor quadrature points for rotor-aware modes).
- `loss_factor`: `float`, default `1.0` (applied multiplicatively).

Normative physics/semantics:
- Capacity factors are computed **per turbine at its hub height**, derived from the turbine definition.
- Rotor-aware modes use height-continuous evaluation over rotor disk heights.
- This metric must not accept a free `height=` parameter (hub height is implied by the turbine).

Intermediate requirement:
- When `air_density=True`, the wind dataset must be able to hold air density fields as:
  - `atlas.wind.data["rho"]` with dims `("height", "y", "x")` (height in meters; values correspond to the hub-height bins used).

### `metric="mean_wind_speed"`

Parameters:
- `height: int` (required; meters above ground).

Normative:
- Must not require turbine selection.
- Must operate on the current region selection if set (A4).

### Additional wind metrics (implemented)

- `metric="rews_mps"`
  - Requires turbines via selection or `turbines=[...]`.
  - Optional: `air_density: bool = False`, `rews_n: int = 12`.
- `metric="lcoe"`
  - Requires turbines and cost/economic params:
    - `om_fixed_eur_per_kw_a`, `om_variable_eur_per_kwh`, `discount_rate`, `lifetime_a`.
  - Optional: `turbine_cost_share`, `hours_per_year`, plus capacity-factor options (`mode`, `rews_n`, `air_density`, `loss_factor`).
- `metric="min_lcoe_turbine"`, `metric="optimal_power"`, `metric="optimal_energy"`
  - Same turbine/economic parameter contract as `lcoe`.

---

## A10. Landscape

Primary interface:

```python
atlas.landscape.data["valid_mask"]
```

Adding rasters (region-scoped derived data):

```python
op = atlas.landscape.add(name="my_raster", source_path="/path/to/raster.tif")
da_stage = op.data
da_mat = op.materialize()
atlas.landscape.clear_staged()
```

Normative:
- `add(...)` stages a lazy candidate and returns an operation object with `.data` and `.materialize(...)`.
- Staged variables must surface in `atlas.landscape.data[name]` before materialization.
- If a region selection is active, added rasters must be written to the **derived region store** for that region.
- `clear_staged()` must remove staged landscape overlays from `atlas.landscape.data`.
- Staged landscape overlays are transient: `select(...)`, `build()`, `build_canonical()`, and `build_clc()` clear them.
- Successful landscape `.materialize()` clears the staged overlay for that variable.

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

# B. Architecture and invariants

## B1. On-disk store layout (normative)

Given `atlas.path = <ROOT>`:

Base stores:
- `<ROOT>/wind.zarr`
- `<ROOT>/landscape.zarr`

Derived region stores:
- `<ROOT>/regions/<region_id>/wind.zarr`
- `<ROOT>/regions/<region_id>/landscape.zarr`

Results stores:
- `<ROOT>/results/<run_id>/<name>.zarr`
- metadata/provenance fields are stored in Zarr attrs (no required `meta.json` file).

Region ID rules:
- If `atlas.region` is `None`, materialized wind metrics are written to the base wind store (`<ROOT>/wind.zarr`).
- Otherwise `region_id` is the provided NUTS id (e.g. `"AT13"`).

---

## B2. Identity, determinism, and manifests

- Each store must have an **inputs identity** (`inputs_id`) computed from deterministic inputs:
  - country, CRS, resolution, upstream data fingerprints,
  - for wind base store: the configured turbine list (or `"default"`),
  - for region derived stores: region selection id + region-shape fingerprint.
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

- Base stores are country-wide and reusable across any number of regions.
- Region selection only affects derived region stores and compute masking.
- It must be possible to:
  - materialize base stores once (e.g. Austria),
  - switch region selection (e.g. `AT12`, then `AT13`),
  - compute/materialize metrics for each region into separate derived region stores.

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

- `compute(...).data` must be lazy if Dask is configured (unless the user explicitly requests eager execution).
- `.materialize()` is the canonical user-triggered materialization step that may compute.

---

## B7. I/O Layer Boundaries (normative)

- `cleo/unification/**` is the canonical location for raw geospatial/base-store/region-store I/O.
- `cleo/results.py` is the canonical location for results-store persistence/open/export internals.
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
