# CLEO

CLEO is a [xarray](https://docs.xarray.dev)-based Python library for converting data from 
the [Global Wind Atlas](https://globalwindatlas.info) to wind resource assessment.
CLEO can produce data such as

* wind shear
* air density correction
* mean wind speed at various heights
* capacity factor of wind turbines at their hub height
* levelized cost of electricity (LCOE) from wind turbines

in the high, native spatial resolution of the Global Wind Atlas.
CLEO also supports some geospatial operations such as reprojecting to EPSG-coordinate reference systems
and clipping to geopandas geometries.

## Installation
At the moment, CLEO is usable only from its GitHub repo. 

`pip install git+https://github.com/sebwehrle/cleo.git`

In a later release we aim to improve installation.

## Documentation
CLEO's functions are documented in-line. 
We are working towards an improved documentation.

### How to get started
To get started, initialize an `Atlas`-class object via
```Python
from cleo.wind_atlas import Atlas

atlas = Atlas("path/to/base/dir", "XYZ")
```
where `XYZ` is a 3-digit ISO country code as used by the GWA API.
Upon initialization, the class will download data for `XYZ`.
The class wraps an `xarray.Dataset` and data is accessible through `atlas.data`.

#### Properties
An `Atlas`-object has the following properties:
* `atlas.path`: the atlas base-path
* `country`: 3-digit ISO code of the country to assess
* `wind_turbine`: list of wind turbine models to process. Defaults to `Vestas.V112.3075`. Further turbines can be added 
as a list. Additional turbines require a data file in the `/resources`-directory.
* `crs`: coordinate reference system of the `Atlas`' spatial data.
* `power_curves`: loaded power curves.

#### Methods
The `Atlas`-class provides several methods, including:
* `load_powercurves()`: loads powercurves from `
* `compute_wind_shear()`: computes wind shear factor alpha
* `compute_air_density_correction()`: computes air density correction factor alpha
* `compute_weibull_pdf()`: computes a Weibull pdf of wind speeds
* `simulate_capacity_factors()`: simulate capacity factors of the wind turbines in `atlas.wind_turbine`. 
Before simulating capacity factors, properly named (`manufacturer.model.power.yml`) `yaml`-file in `/resources` for all
wind turbines in `atlas.wind_turbine` must be loaded with `atlas.load_powercurves()`. 
* `compute_lcoe()`: calculates LCOE for each pixel and each wind turbine in `atlas.wind_turbines` based on the 
cost-assumptions in `/resources/cost_assumptions.yml`. By default, overnight cost of wind turbines are estimated with 
the [Rinne et al. cost model](https://doi.org/10.1038/s41560-018-0137-9), which is implemented in 
`turbine_overnight_cost()` in `cleo.utils`.
* `process()`: performs all required computations to generate LCOE estimates.

### Author and Copyright
Copyright (c) 2024 Sebastian Wehrle

### License
MIT License

