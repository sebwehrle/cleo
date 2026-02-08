# CLEO

CLEO is an **xarray-based** Python package for wind resource assessment using the **Global Wind Atlas (GWA)**.
It downloads (once) country rasters into a local work directory, builds a canonical grid, and computes wind-energy metrics
such as Weibull PDFs, capacity factors, and LCOE.

Design principles:

- **Explicit IO, deterministic compute:** network downloads happen during `materialize()` (or explicit loader calls), not inside compute methods.
- **Local, explicit prerequisites:** higher-level computations trigger required *derived* prerequisites when needed.
- **Reproducible & cacheable:** results live in `xarray.Dataset`s with stable coordinates; repeated calls can skip recomputation when inputs/parameters match.
- **Workdir-first resources:** packaged YAML resources (turbines, costs, land cover codes) are deployed into your workdir for inspection/override.

---

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

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
cf = atlas.wind.simulate_capacity_factors(
    weibull_height_mode="hub",
    air_density_mode="gwa",   # lazily loads local GWA air-density GeoTIFFs; no network during compute
)

# LCOE (uses cost assumptions YAML in <workdir>/resources/cost_assumptions.yml)
lcoe = atlas.wind.compute_lcoe()
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
cf = atlas.wind.simulate_capacity_factors(
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
- `air_density_mode="gwa"` lazily loads local GeoTIFFs from `<workdir>/data/raw/<ISO3>/` via `load_air_density(height)`.
- Interpolation requires **at least two** available GWA air-density height levels (from the package’s `GWA_HEIGHTS`, e.g. 50/100/150/200 m).
- If required raw files are missing, CLEO raises an actionable `FileNotFoundError` (run `atlas.materialize()` / `atlas.wind._load_gwa()` first).

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
