# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Chunk policy auto-detection: `open_zarr_dataset` now reads stored chunk policy from Zarr attrs and warns if it differs from configured policy, using stored chunks for optimal read performance
- `DEFAULT_CHUNK_POLICY` constant exported from `cleo.unification.store_io`
- `__version__` attribute accessible via `cleo.__version__`
- `CITATION.cff` for academic citation support
- Smoke tests in `tests/smoke/` for import verification
- GDAL/PROJ installation instructions in README (Ubuntu, macOS, Conda)
- Troubleshooting section in README covering common issues (GDAL setup, CLC auth, memory, GWA downloads, chunk warnings)
- This changelog

### Changed
- Minimum supported Python version is now 3.11 because the required stable Zarr v3 releases are not available for Python 3.10
- Wind public API now uses explicit wind-assessment method vocabulary:
  `capacity_factors` accepts `method`/`interpolation` instead of legacy `mode`,
  `.materialize()` uses `allow_method_change`, and CF provenance is stored via
  `cleo:cf_method` / `cleo:interpolation`
- `atlas.wind.compute("wind_speed", ...)` is now the public wind-speed selector:
  `method="height_weibull_mean"` yields `mean_wind_speed(height,y,x)` and
  `method="rotor_equivalent"` yields `rotor_equivalent_wind_speed(turbine,y,x)`
- Rotor-aware wind assessment methods now accept explicit interpolation
  selection (`ak_logz` or `mu_cv_loglog`) while preserving `auto` as the
  behavior-preserving default mapping
- Added type hints to key public methods: `Atlas.__init__`, `Atlas.build`, `Atlas.flatten`, `Atlas.export_analysis_dataset_zarr`
- Extracted parameter validation helpers in `configure_economics()` for more consistent error messages
- Improved wind-capacity-factor validation coverage with a new internal rotor-inflow seam test suite (`tests/unit/assess/test_rotor_inflow.py`) while preserving public API and metric semantics
- Refactored wind capacity-factor internals to use a unified inflow-based integration pipeline across the four public method families with parity-preserving outputs
- Consolidated duplicate direct/momentmatch turbine helper paths onto the same inflow-seam implementation, improving maintainability with unchanged public metric semantics
- Refactored wind capacity-factor internals to prepare explicit vertical profiles before rotor approximation, preserving current public method behavior while clarifying the computation pipeline
- Unified hub-height and rotor-aware capacity-factor families onto one policy-backed vertical evaluation framework, preserving current outputs while making the interpolation backend explicit internally
- Unified height-based wind-speed evaluation onto the same internal vertical evaluator used by wind assessment methods, preserving public behavior while removing duplicated interpolation logic
- Simplified internal wind-metric orchestration by reusing shared wind-store input resolution helpers and removing redundant internal compute parameters, with unchanged public behavior

### Fixed
- Normalize test-suite formatting so Ruff format checks pass in CI
- Pin Ruff to `0.15.2` in dev dependencies and CI so local and GitHub formatting checks use the same version

### Security
- Added secret scanning to CI workflow via `detect-secrets`
- Added Dependabot for automated dependency updates (conservative settings)
- Hardened `.gitignore` with secret file patterns
- Added security documentation for CLMS authentication in `cleo/clc.py`

## [0.0.1] - 2024-12-01

### Added
- **Core Atlas API**
  - `Atlas` class as single public entry point
  - Area selection with NUTS region support (`atlas.select(area=...)`)
  - Turbine configuration (`atlas.configure_turbines(...)`)
  - Timebase configuration (`atlas.configure_timebase(...)`)
  - Economics configuration (`atlas.configure_economics(...)`)

- **Wind Domain**
  - `atlas.wind.data` primary data interface
  - `atlas.wind.compute(metric=...)` unified compute API
  - Metrics: `capacity_factors`, `mean_wind_speed`, `rews_mps`, `lcoe`, `min_lcoe_turbine`, `optimal_power`, `optimal_energy`
  - Grouped dependency specs for composed metrics (`cf={...}`, `economics={...}`)
  - Turbine selection via `atlas.wind.select(turbines=...)` or `atlas.wind.select(turbine_indices=...)`
  - Multiple CF computation modes: `direct_cf_quadrature`, `momentmatch_weibull`, `hub`, `rews`

- **Landscape Domain**
  - `atlas.landscape.data` primary data interface
  - `atlas.landscape.add(...)` for raster ingestion
  - `atlas.landscape.rasterize(...)` for vector-to-raster conversion
  - `atlas.landscape.compute(metric="distance", ...)` for distance transforms
  - CLC (CORINE Land Cover) integration via `atlas.build_clc(...)`

- **Materialization System**
  - Idempotent `atlas.build()` for base store creation
  - `wind.zarr` and `landscape.zarr` canonical stores
  - Area-scoped derived stores under `areas/<area_id>/`
  - Manifest-based provenance tracking
  - Single-writer lock enforcement for concurrent safety

- **Results API**
  - `DomainResult` wrapper with `.data`, `.materialize()`, `.persist()`
  - `atlas.persist(...)` for result storage
  - `atlas.open_result(...)` for result retrieval
  - `atlas.export_result_netcdf(...)` for NetCDF export
  - `atlas.export_analysis_dataset_zarr(...)` for schema-versioned exports

- **Data Export**
  - `atlas.flatten(...)` for tabular DataFrame export
  - Schema-versioned Zarr exports with provenance tracking
  - NetCDF export with CF-compliant handling

- **Cleanup APIs**
  - `atlas.clean_results(...)` for result store cleanup
  - `atlas.clean_areas(...)` for area store cleanup

- **Unit System**
  - Canonical unit registry in `cleo/units.py`
  - Unit conversion utilities (`convert_dataarray`, `conversion_factor`)
  - Domain-level `convert_units()` method

- **Infrastructure**
  - Dask-native lazy computation with multi-backend support (`serial`, `threads`, `processes`, `distributed`)
  - Automatic GWA data download for missing wind rasters
  - Automatic NUTS shapefile download
  - CopDEM tile download and mosaic for elevation data
  - CLMS API integration for CLC data download

- **Quality Assurance**
  - Comprehensive test suite (877+ unit tests)
  - CI pipeline with lint, test, and build jobs
  - Cyclomatic complexity enforcement (no Grade E functions)
  - Architecture boundary enforcement via static analysis
  - Schema versioning for external exports

- **Documentation**
  - Contract-driven design (`docs/CONTRACT_UNIFIED_ATLAS.md`)
  - MkDocs-based documentation site
  - Domain guides for wind and landscape workflows
  - Canonical workflow guide

### Technical Details
- Python 3.11+ required
- Built on xarray, dask, rasterio, geopandas ecosystem
- Zarr v3 compatible storage (no string arrays, no consolidated metadata dependency)
- 5-layer architecture: orchestration, materialization, computation, boundaries, policies

[Unreleased]: https://github.com/sebwehrle/cleo/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/sebwehrle/cleo/releases/tag/v0.0.1
