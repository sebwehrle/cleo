# CLEO

CLEO is a [xarray](https://docs.xarray.dev)-based Python library that downloads and converts 
[Global Wind Atlas](https://globalwindatlas.info) rasters into a structured, analysis-ready form for wind resource 
assessment at the GWA's native resolution.

CLEO can produce data such as

* wind shear coefficient
* air density correction
* mean wind speed at various heights
* capacity factor of wind turbines at their hub height (given power curves)
* levelized cost of electricity (LCOE)

CLEO also provides common geospatial operations (CRS handling, reprojection, clipping) tailored to the atlas workflow.

## Status and scope

- The public API is centered around the `Atlas` class.
- YAML resources (turbine power curves, cost assumptions, etc.) are **packaged** with the wheel and are **deployed to the atlas workdir** automatically on `Atlas(...)` construction.
- Python requirement: **Python >= 3.10**.

## Installation
### From source (local checkout)

From the repository root:

```bash
python -m pip install -e .
```

### Directly from GitHub

```bash
python -m pip install "cleo @ git+https://github.com/sebwehrle/cleo.git"
```

If you use conda/mamba for scientific stacks, you may prefer creating an environment first and then installing CLEO into it.

## Quick start

```python
from cleo import Atlas

atlas = Atlas("/path/to/cleo-workdir", "AUT", "EPSG:4326")
atlas.materialize()

# Compute wind resource metrics
atlas.wind.compute_wind_shear_coefficient()
atlas.wind.compute_air_density_correction()
atlas.wind.compute_mean_wind_speed(100)
```

Notes:

- `AUT` is a 3-letter ISO country code as used by the GWA API.
- `EPSG:4326` is the target coordinate reference system for the atlas.

On first use, run `.materialize()` to download required GWA rasters for the selected country into the workdir and build 
processed NetCDF outputs under `data/processed`.
The class wraps two `xarray.Dataset`s, one for wind resources and one for further spatial characteristics.
The corresponding data is accessible through `atlas.wind.data` and `atlas.landscape.data`, respectively

## GWA4 CRS and elevation behavior

- **GWA4 CRS metadata:** GWA4 rasters lack CRS metadata. CLEO infers CRS from GWA4's country GeoJSON metadata and 
enforces consistent CRS throughout the atlas pipeline.
- **Elevation:** if a legacy elevation file (`elevation_w_bathymetry`) is available, it is used. Otherwise, CLEO falls 
back to Copernicus DEM.

#### Properties
An `Atlas`-object has the following properties:
* `path`: the atlas base-path
* `country`: 3-digit ISO code of the country to assess
* `region`: (optional) the Latin name of a EU NUTS-region within a `country`. 
* `crs`: coordinate reference system of the `WindResourceAtlas`' spatial data.
* `wind_turbines`: list of wind turbine models to process. **Must** be set by the user to allow wind resource assessment. 
Further turbines can be added as a list. Additional turbines require a data file in the `/resources`-directory.

#### Methods
The `WindAtlas`-subclass provides several methods, including:
* `compute_wind_shear_coefficient()`: computes wind shear coefficient
* `compute_air_density_correction()`: computes air density correction factor alpha
* `compute_weibull_pdf()`: computes a Weibull pdf of wind speeds
* `simulate_capacity_factors()`: simulate capacity factors of the wind turbines in `atlas.wind_turbine`. 
Before simulating capacity factors, properly named (`manufacturer.model.power.yml`) `yaml`-file in `/resources` for all
wind turbines in `atlas.wind_turbine` must be loaded with `atlas.load_powercurves()`. 
* `compute_lcoe()`: calculates LCOE for each pixel and each wind turbine in `atlas.wind_turbines` based on the 
cost-assumptions in `/resources/cost_assumptions.yml`. By default, overnight cost of wind turbines are estimated with 
the [Rinne et al. cost model](https://doi.org/10.1038/s41560-018-0137-9), which is implemented in `turbine_overnight_cost()` in `cleo.assess`.

## Testing

From the repository root:
```bash
pytest -q
```

### Author and Copyright
Copyright (c) 2024 Sebastian Wehrle

### License
MIT License

