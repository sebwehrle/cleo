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

In a later release we aim to provide it as a package.

## Documentation
CLEO's functions are documented in-line. An online documentation is on our to-do list.

