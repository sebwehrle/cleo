# Knobs and Methods Employed

## Atlas-first principle (canonical usage)

CLEO is designed around a single mutable `Atlas` workspace object.

- Canonical flow: configure atlas -> compute on atlas domains -> materialize back into atlas stores -> read from `atlas.wind.data` / `atlas.landscape.data`.
- `compute(...)` stages overlays in atlas domain state; `materialize(...)` writes to active stores and updates atlas-facing data.
- `DomainResult` and `Landscape*Result` wrappers are operation handles, not the primary long-lived data model.
- `select(..., inplace=False)` clone-style usage is supported, but the intended default is one working atlas with `inplace=True` updates.

## Atlas-level knobs

### Constructor knobs

| Knob | Type | Default | Meaning |
|---|---|---|---|
| `path` | path-like | required | Atlas workspace root. |
| `country` | `str` (ISO3) | required | Country identity for data/materialization scope. |
| `crs` | CRS string | required | Canonical projected CRS for stores. |
| `chunk_policy` | `dict[str, int] \| None` | `{"y": 1024, "x": 1024}` | Open/chunk policy for xarray/zarr reads. |
| `compute_backend` | `"serial"\|"threads"\|"processes"\|"distributed"` | `"serial"` | Backend used when eager compute is needed for I/O. |
| `compute_workers` | `int \| None` | `None` | Local worker cap (backend-dependent). |
| `region` | `str \| None` | `None` | Optional initial region selection applied after build. |
| `results_root` | `Path \| None` | `<path>/results` | Persisted run artifact root. |
| `fingerprint_method` | `str` | `"path_mtime_size"` | Unification fingerprint strategy. |

### Configuration knobs

| Method | Knobs | Meaning |
|---|---|---|
| `configure_turbines(...)` | `turbines: Sequence[str]` | Restricts turbines materialized into `wind.zarr`; does not change compute API. |
| `configure_timebase(...)` | `hours_per_year: float` | Timebase for LCOE-family metrics; default when unset is `8766.0`. |
| `configure_economics(...)` | `discount_rate`, `lifetime_a`, `om_fixed_eur_per_kw_a`, `om_variable_eur_per_kwh`, `bos_cost_share`, `grid_connect_cost_eur_per_kw` | Baseline economics for LCOE-family metrics; per-call overrides still possible. |

### Selection/workflow knobs

| Method | Knobs | Meaning |
|---|---|---|
| `select(...)` | `region`, `region_level`, `inplace` | Region routing selection; `inplace=True` is intended atlas-first workflow. |
| `build_clc(...)` | `source`, `url`, `force_download`, `force_prepare` | CLC source preparation and cache behavior. |
| `clean_results(...)` | `run_id`, `older_than`, `metric_name` | Cleanup filters for persisted result stores. |
| `clean_regions(...)` | `region`, `older_than`, `include_incomplete` | Cleanup filters for region stores. |

## Wind domain knobs

### Selection knobs

| Method | Knobs | Notes |
|---|---|---|
| `atlas.wind.select(...)` | exactly one of `turbines` or `turbine_indices` | Sets persistent turbine selection for subsequent wind computes. |
| `atlas.wind.clear_selection()` | none | Clears persistent turbine selection. |
| `atlas.wind.clear_computed()` | none | Clears staged wind overlays. |

### `compute(metric, **kwargs)` knobs by metric

| Metric | Required knobs | Optional knobs | Output location semantics |
|---|---|---|---|
| `mean_wind_speed` | `height` | none | Staged in `atlas.wind.data["mean_wind_speed"]`; `materialize()` writes to active wind store. |
| `capacity_factors` | turbines (via persistent selection or `turbines=`) | `mode`, `air_density`, `loss_factor`, `rews_n` | Same staging/materialization semantics. |
| `rews_mps` | turbines (via persistent selection or `turbines=`) | `air_density`, `rews_n` | Same staging/materialization semantics. |
| `lcoe` | turbines + grouped specs | `cf={...}`, `economics={...}` | Same staging/materialization semantics; economics/timebase attrs added. |
| `min_lcoe_turbine` | turbines + grouped specs | `cf={...}`, `economics={...}` | Same staging/materialization semantics. |
| `optimal_power` | turbines + grouped specs | `cf={...}`, `economics={...}` | Same staging/materialization semantics. |
| `optimal_energy` | turbines + grouped specs | `cf={...}`, `economics={...}` | Same staging/materialization semantics. |

### Grouped spec knobs for composed metrics

#### `cf={...}` keys

| Key | Default | Meaning |
|---|---|---|
| `mode` | `"direct_cf_quadrature"` | Capacity-factor method variant. |
| `air_density` | `False` | Apply density correction via `rho` if available. |
| `rews_n` | `12` | Quadrature/sample nodes for REWS-related integrations. |
| `loss_factor` | `1.0` | Multiplicative loss factor on CF output. |

#### `economics={...}` keys

Required fields:

- `discount_rate`
- `lifetime_a`
- `om_fixed_eur_per_kw_a`
- `om_variable_eur_per_kwh`

Optional fields:

- `bos_cost_share` (default `0.0`)
- `grid_connect_cost_eur_per_kw` (default `50.0`)

Notes:

- Flat CF/economics kwargs are rejected for composed metrics; use grouped `cf={...}` and `economics={...}`.
- `hours_per_year` is atlas-level (`configure_timebase`) and intentionally rejected in `compute(...)` kwargs.

## Wind methods by metric (explicit)

### `mean_wind_speed`

Algorithm:

1. Select `A` and `k` Weibull parameters at requested height.
2. Compute expectation using Weibull mean formula:
   `E[U] = A * Gamma(1 + 1/k)`.
3. Apply `valid_mask` when available.

Output:

- Units: `m/s`
- Dims: spatial only (`y`, `x`)

### `capacity_factors`

Common steps:

1. Resolve turbines and read per-turbine metadata (hub height, power curve, rotor diameter).
2. Read `A(height,y,x)` and `k(height,y,x)` plus wind-speed grid.
3. Execute selected CF mode numerics turbine-by-turbine.
4. Apply `loss_factor` in integration and `valid_mask` on output.

Mode details:

- `direct_cf_quadrature`:
  1. Build rotor node heights from Gauss-Legendre nodes.
  2. Weight nodes with rotor-chord area weighting.
  3. Evaluate vertical Weibull at each node via vertical policy.
  4. Integrate expected normalized power at each node over wind-speed PDF.
  5. Area-weight sum node CF values.

- `momentmatch_weibull`:
  1. Sample rotor nodes as above.
  2. Compute rotor-averaged first and third moments (`m1`, `m3`).
  3. Solve equivalent Weibull `k_rot` from `m3 / m1^3`.
  4. Derive `A_rot` from `m1` and `k_rot`.
  5. Integrate CF once using equivalent Weibull.

- `hub`:
  1. Interpolate Weibull parameters to hub height.
  2. Integrate expected normalized power once at hub-height distribution.

- `rews` (legacy):
  1. Compute REWS moment factor from rotor/hub cubic moments.
  2. Scale hub Weibull `A` by factor.
  3. Integrate expected normalized power at adjusted distribution.

Air density (`air_density=True`):

- Uses local `rho`; speed-equivalent scaling enters via `(rho / rho0)^(1/3)`.

Output:

- Units: `1`
- Dims: `turbine`, `y`, `x`

### `rews_mps`

Algorithm:

1. Resolve turbines and rotor geometry.
2. Run direct rotor quadrature core in REWS-only mode.
3. Compute `REWS = (E_rotor[U^3])^(1/3)`.
4. Apply `valid_mask`.

Output:

- Units: `m/s`
- Dims: `turbine`, `y`, `x`

### `lcoe`

Algorithm:

1. Resolve `cf={...}` and `economics={...}` grouped specs.
2. Reuse matching materialized CF if available; else compute CF.
3. Build turbine-specific CAPEX inputs (rated power and overnight cost).
4. Compute discounted energy and discounted cost streams over lifetime.
5. Return `LCOE = NPV(costs) / NPV(energy)` in `EUR/MWh`.

Cost/electricity structure:

- Energy stream: `CF * power_kw * hours_per_year`, discounted across lifetime.
- Cost stream: fixed O&M + variable O&M + one-time overnight/grid components.
- BOS and grid-connect knobs shape CAPEX structure.

Output:

- Units: `EUR/MWh`
- Dims: `turbine`, `y`, `x`

### `min_lcoe_turbine`

Algorithm:

1. Compute (or reuse) LCOE cube.
2. Take argmin across turbine dimension per cell.
3. Encode all-NaN cells as nodata index `-1`.

Output:

- Units: dimensionless index
- Dims: `y`, `x`

### `optimal_power`

Algorithm:

1. Compute (or reuse) LCOE cube.
2. Find minimum-LCOE turbine index per cell.
3. Select rated turbine power for that index.

Output:

- Units: `kW`
- Dims: `y`, `x`

### `optimal_energy`

Algorithm:

1. Compute/reuse CF and compute LCOE cube.
2. Find minimum-LCOE turbine index per cell.
3. Select CF and power for that selected turbine.
4. Compute annual energy:
   `optimal_energy = CF_selected * power_selected_kw * hours_per_year / 1e6`.

Output:

- Units: `GWh/a`
- Dims: `y`, `x`

## Landscape domain knobs

### `compute(metric="distance", ...)`

| Knob | Required | Default | Meaning |
|---|---|---|---|
| `source` | yes | - | Source variable name or list/tuple of names in active landscape store. |
| `name` | no | `distance_<source>` | Output variable name(s). |
| `if_exists` | no | `"error"` | Conflict handling: `error`, `replace`, or `noop`. |

### `add(...)` raster staging

| Knob | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | - | Target landscape variable name. |
| `source_path` | yes | - | Raster source path. |
| `kind` | no | `"raster"` | Must be `"raster"` (vector path handled via `rasterize`). |
| `params` | no | `{}` | Registration parameters passed to unification pipeline. |
| `if_exists` | no | `"error"` | Conflict handling: `error`, `replace`, or `noop`. |

### `rasterize(...)` vector staging

| Knob | Required | Default | Meaning |
|---|---|---|---|
| `shape` | yes | - | Path-like vector source or GeoDataFrame. |
| `name` | yes | - | Target landscape variable name. |
| `column` | no | `None` | Optional value column to rasterize; if `None`, burns binary mask. |
| `all_touched` | no | `False` | Rasterization inclusion rule for pixel touch behavior. |
| `if_exists` | no | `"error"` | Conflict handling: `error`, `replace`, or `noop`. |

### `add_clc_category(...)`

| Knob | Required | Default | Meaning |
|---|---|---|---|
| `categories` | yes | - | `"all"`, one integer code, or non-empty list of integer codes. |
| `name` | conditional | `None` | Required for multi-code requests; inferred for known single-code defaults. |
| `source` | no | `"clc2018"` | CLC source selection. |
| `if_exists` | no | `"error"` | Conflict handling: `error`, `replace`, or `noop`. |

## Landscape methods (explicit)

### `compute(metric="distance")`

Algorithm:

1. Validate source variable and exact y/x alignment with `valid_mask`.
2. Require projected CRS with meter units.
3. Define targets as finite and strictly positive source cells.
4. Run Euclidean distance transform using actual grid spacing.
5. Emit meter-valued raster with deterministic method attrs.

### `add(...)`

Algorithm:

1. Register raster source and params in unification source registry.
2. Normalize/reproject/align to atlas canonical grid.
3. Stage overlay and write on `.materialize()`.

### `rasterize(...)`

Algorithm:

1. Register vector source and rasterization settings.
2. Rasterize onto atlas canonical grid and CRS.
3. Stage overlay and write on `.materialize()`.

### `add_clc_category(...)`

Algorithm:

1. Prepare aligned CLC source cache.
2. Resolve category selection and output naming.
3. Route through `add(...)` pipeline with CLC params.

## Persist vs materialize (important distinction)

- `materialize(...)` writes into active domain stores and surfaces immediately in `atlas.<domain>.data`.
- `persist(...)` writes run artifacts under `results_root` for external/result-management workflows.
- Atlas-first workflow typically uses `materialize(...)` and continues through `atlas.wind.data` / `atlas.landscape.data`.
