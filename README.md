# CLEO

CLEO is an **xarray-based** Python package for wind resource assessment using the **Global Wind Atlas (GWA)**.
It downloads (once) country rasters into a local work directory, builds a canonical grid, and computes wind-energy metrics
such as Weibull PDFs, capacity factors, and LCOE.

Design principles:

- **Explicit IO, deterministic compute:** network downloads happen during `materialize()` (or explicit loader calls), not inside compute methods.
- **Local, explicit prerequisites:** higher-level computations trigger required *derived* prerequisites when needed.
- **Reproducible & cacheable:** results live in `xarray.Dataset`s with stable coordinates; repeated calls can skip recomputation when inputs/parameters match.
- **Workdir-first resources:** packaged YAML resources (turbines, costs, land cover codes) are deployed into your workdir for inspection/override.
- **Optional Dask:** CLEO works without Dask; when enabled explicitly, it supports chunked, parallel execution for large rasters.

---

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

Optional (recommended for large rasters): install Dask support (chunking + parallel execution).  
If your project defines extras, prefer the package extra; otherwise install Dask directly:

```bash
python -m pip install "dask[array]"
```

(For distributed clusters, install `dask[distributed]`.)

---

## Quick start

```python
from cleo import Atlas

atlas = Atlas("/path/to/cleo-workdir", country="AUT", crs="EPSG:3035")

# First-time setup: downloads required raw inputs (GWA rasters, NUTS) and builds base NetCDF scaffolds
atlas.materialize()

# Add one or more turbine models (YAML expected at <workdir>/resources/<turbine>.yml)
atlas.add_turbine("Vestas.V150.4200")

# Derived metrics
atlas.wind.compute_wind_shear_coefficient()
atlas.wind.compute_mean_wind_speed(100)

# Capacity factors (hub-height Weibull + optional air-density correction)
atlas.wind.simulate_capacity_factors(
    weibull_height_mode="hub",
    air_density_mode="gwa",   # loads local GWA air-density GeoTIFFs; no network during compute
)

# LCOE (uses cost assumptions YAML in <workdir>/resources/cost_assumptions.yml)
atlas.wind.compute_lcoe()
```

Notes:

- `country` is an **ISO-3166 alpha-3** country code used by the GWA endpoints (e.g. `AUT`).
- `crs` is the target CRS for the processed atlas grid (e.g. `EPSG:3035` or `EPSG:4326`).

---

## Atlas model

The public API is centered around the `Atlas` class:

- `atlas.wind` is a WindAtlas holding wind resource data and derived products.
- `atlas.landscape` is a LandscapeAtlas holding supporting spatial characteristics (e.g., NUTS shapes).
- Both are backed by `xarray.Dataset`s:
  - `atlas.wind.data`
  - `atlas.landscape.data`

### Workdir layout

CLEO uses the following structure:

- `<workdir>/resources/`
  - YAML resources deployed from the package (not overwritten if you edit them).
- `<workdir>/data/raw/<ISO3>/`
  - Downloaded GWA rasters (GeoTIFF).
- `<workdir>/data/nuts/`
  - Downloaded/extracted NUTS shapefiles for region clipping.
- `<workdir>/data/processed/`
  - NetCDF outputs for WindAtlas and LandscapeAtlas (optionally versioned by scenario/timestamp).
- `<workdir>/data/index.jsonl`
  - Append-only index used by `save()` / `load()` for version selection.
- `<workdir>/logs/`
  - Log output.

---

## Dask support (chunking + parallel execution)

CLEO supports **optional** Dask usage for:
- **Chunking** large rasters (GeoTIFF and NetCDF-backed datasets).
- **Parallel execution** when you explicitly request execution of lazy results.

Two separate ideas matter:

1) **Chunking** (how data is represented)
- If data is opened with `chunks=...`, `xarray` will use Dask-backed arrays internally.
- If `chunks=None`, arrays are typically NumPy-backed (eager). Dask schedulers have nothing to schedule in that case.

2) **Execution mode** (when work actually runs)
- Many operations build a lazy task graph when arrays are Dask-backed.
- Work runs when you explicitly execute (or when an output operation triggers execution).

### Unified Dask configuration

CLEO uses a unified configuration object (stored on the Atlas) and allows per-call overrides:

- **Atlas default:** configured at `Atlas(...)` construction, stored as `atlas.dask_cfg`.
- **Method override:** heavy compute methods accept `dask=...` to override the Atlas default for that call.

Configuration fields:

- `use_dask`: `False | True | "auto"`
  - `False`: never use Dask.
  - `True`: require Dask to be installed.
  - `"auto"`: enable Dask only if installed.
- `chunks`: `None | "auto" | dict[str,int]`
  - `None`: do not chunk (typically eager arrays).
  - `"auto"`: let xarray choose chunking (only when Dask is enabled).
  - `{"y": 1024, "x": 1024}`: explicit chunk sizes.
- `scheduler`: `None | "threads" | "processes" | "single-threaded" | "distributed"`
  - Used when executing graphs (see `execute` below).
  - `"distributed"` requires an active `dask.distributed.Client` (see below).
- `execute`: `"lazy" | "eager" | "cached"`
  - `"lazy"`: store results without executing (keeps laziness if Dask-backed).
  - `"eager"`: execute immediately and store NumPy-backed results.
  - `"cached"`: execute immediately but keep Dask-backed cached results (`persist`).

### Example: enable chunking at Atlas creation

```python
from cleo import Atlas

atlas = Atlas(
    "/path/to/cleo-workdir",
    country="AUT",
    crs="EPSG:3035",
    use_dask="auto",
    chunks={"y": 1024, "x": 1024},
    scheduler="threads",
    execute="lazy",
)
atlas.materialize()
```

### Example: execute a computation immediately (parallel) and store eager results

```python
from cleo.dask_utils import DaskConfig

atlas.wind.simulate_capacity_factors(
    weibull_height_mode="hub",
    air_density_mode="gwa",
    dask=DaskConfig(execute="eager", scheduler="threads"),
)
```

### Example: keep results lazy but cache chunks for reuse

```python
from cleo.dask_utils import DaskConfig

atlas.wind.simulate_capacity_factors(
    weibull_height_mode="hub",
    air_density_mode="gwa",
    dask=DaskConfig(execute="cached", scheduler="threads"),
)
```

### Scheduler warning (guidance)

If you request a parallel scheduler (`"threads"` / `"processes"`) while **not chunking** (`chunks=None`), results are usually **not**
Dask-backed and the scheduler cannot be applied. In this case CLEO emits a warning like:

- "scheduler='threads' requested but result is not dask-backed (chunks=None); scheduler will be ignored. Set chunks …"

To benefit from Dask parallelism on large rasters, enable chunking (`chunks="auto"` or a dict).

### Distributed scheduler

CLEO supports the `dask.distributed` scheduler for cluster-based parallelism. Key requirements:

1. **Install `dask[distributed]`:**
   ```bash
   pip install "dask[distributed]"
   ```

2. **Start a Client outside CLEO:**
   CLEO does **not** create or manage distributed clusters/clients. You must start a client before calling CLEO methods:
   ```python
   from dask.distributed import Client
   client = Client()  # local cluster, or connect to existing cluster
   ```

3. **Use `scheduler="distributed"`:**
   ```python
   from cleo.dask_utils import DaskConfig

   atlas.wind.simulate_capacity_factors(
       weibull_height_mode="hub",
       air_density_mode="gwa",
       dask=DaskConfig(
           chunks={"y": 1024, "x": 1024},
           execute="eager",
           scheduler="distributed",
       ),
   )
   ```

4. **Dashboard link logging:**
   When `execute` is `"eager"` or `"cached"` and `scheduler="distributed"`, CLEO logs:
   ```
   INFO:cleo.assess:Dask dashboard: http://127.0.0.1:8787/status
   ```
   The link can change if the cluster restarts; CLEO does **not** persist or store it.

If no active client exists when `scheduler="distributed"` is used, CLEO raises a clear `RuntimeError` with guidance.

---

## Turbines and YAML resources

Packaged turbine definitions live in `cleo/resources/*.yml` and are deployed to:

- `<workdir>/resources/<turbine_name>.yml`

Add a turbine by name:

```python
atlas.add_turbine("Enercon.E115.3000")
```

To add your own turbine:

1. Create `<workdir>/resources/MyMaker.MyModel.yml` (follow the existing turbine YAML structure).
2. Call `atlas.add_turbine("MyMaker.MyModel")`.

---

## Wind resource assessment

All wind assessment methods are called on `atlas.wind`.

### Wind shear coefficient

```python
atlas.wind.compute_wind_shear_coefficient()
```

### Mean wind speed at height

```python
atlas.wind.compute_mean_wind_speed(100)
```

### Weibull PDF

```python
atlas.wind.compute_weibull_pdf()
```

### Capacity factors

```python
atlas.wind.simulate_capacity_factors(
    weibull_height_mode="hub",
    air_density_mode="none",  # or "gwa"
)
```

Key modes:

- `weibull_height_mode="hub"`:
  - Interpolates GWA Weibull parameters to hub height and computes the PDF at hub height.
- `weibull_height_mode="100m_shear"` (legacy):
  - Uses the 100 m Weibull PDF and shear scaling for hub-height adjustment.

Air density correction (`air_density_mode="gwa"`) uses local GWA air-density rasters and the equivalent wind-speed mapping

\[
u_{eq} = u \left(\frac{\rho}{\rho_0}\right)^{1/3},
\qquad \rho_0 = 1.225\ \mathrm{kg/m^3}.
\]

Air density contract (important):

- **No network downloads during compute.**
- `air_density_mode="gwa"` loads local GeoTIFFs from `<workdir>/data/raw/<ISO3>/` via the package loaders.
- Interpolation requires **at least two** available GWA air-density height levels (from the package’s `GWA_HEIGHTS`, e.g. 50/100/150/200 m).
- If required raw files are missing, CLEO raises an actionable `FileNotFoundError` (run `atlas.materialize()` first).

### LCOE

```python
lcoe = atlas.wind.compute_lcoe()
```

LCOE uses cost assumptions from:

- `<workdir>/resources/cost_assumptions.yml`

---

## Spatial operations

### Clip to NUTS region

```python
atlas.clip_to_nuts("Wien")
atlas.clip_to_nuts(["Wien", "Niederösterreich"], merged_name="W + NÖ")
```

This clips both `atlas.wind.data` and `atlas.landscape.data` to the chosen shape and sets `atlas.region`.

---

## Persistence and versioning

### Save

```python
atlas.save()
atlas.save(scenario="my_scenario")
```

### Load

```python
atlas.load()  # load latest
atlas.load(scenario="my_scenario")
atlas.load(scenario="my_scenario", timestamp="2026-02-08T12-34-56")
```

(Timestamp formats follow what was written by `save()`.)

---

## Testing

From repository root:

```bash
pytest -q
```

---

## License

MIT License. See `LICENSE.md`.
