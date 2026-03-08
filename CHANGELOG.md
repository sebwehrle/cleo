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
- Additional packaged turbine models: `Enercon.E126.3500`, `Vestas.V136.4200`, `Vestas.V162.5600`, and `Vestas.V162.7200`
- Smoke tests in `tests/smoke/` for import verification
- GDAL/PROJ installation instructions in README (Ubuntu, macOS, Conda)
- Troubleshooting section in README covering common issues (GDAL setup, CLC auth, memory, GWA downloads, chunk warnings)
- This changelog

### Changed
- Minimum supported Python version is now 3.11 because the required stable Zarr v3 releases are not available for Python 3.10
- README now gives a clearer first-time-user path: source-checkout installation, first-run download expectations, defined CLC/NUTS terminology, optional CLC setup, Europe-focused scope notes, remote-only `build_clc(url=...)`, and a more common-path-first API section
- `atlas.wind.compute("min_lcoe_turbine", ...)` now masks invalid pixels at the public API boundary instead of surfacing the internal `-1` nodata sentinel
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
- Unit conversion now requires the canonical `units` attr, and current wind stores are treated as canonical `weibull_A` / `weibull_k` datasets without legacy alias tolerance
- Simplified internal unification orchestration so incremental landscape registration/materialization routes directly to its owning materializer helpers while preserving Atlas workflows and canonical unification behavior
- Simplified internal wind metric dispatch plumbing so `WindDomain.compute(...)` now consumes the canonical wind-metric registry directly, removing boundary wrapper helpers while preserving grouped spec behavior, CF reuse semantics, and metric outputs
- Narrowed the remaining internal wind-compute helper chain by folding tiny single-use validation wrappers into one compute-kwarg validator, preserving grouped spec behavior, CF reuse, timebase injection, and metric outputs
- Simplified internal result persistence and materialization plumbing by sharing atlas I/O evaluation policy and validation requirement helpers while preserving result-store semantics, export behavior, and active-store alignment behavior

### Removed
- Developer benchmark helpers are no longer shipped in the runtime package as `cleo.bench`; maintainer benchmarking now lives under `tools/bench.py`

### Fixed
- Fixed CI test execution to stop referencing the removed internal compat test directory.
- Normalize test-suite formatting so Ruff format checks pass in CI
- Pin Ruff to `0.15.2` in dev dependencies and CI so local and GitHub formatting checks use the same version
- Stabilized the phase-3 materialization golden test by pinning its turbine inventory so packaged turbine additions do not spuriously change the expected wind `inputs_id`
- Removed transient rollout wording from internal prose, renamed integration test modules to stable names, and replaced legacy runtime/provenance labels with canonical identifiers
- Clarified and regression-tested chunked dask selection for `optimal_power` and `optimal_energy`, confirming the current economics path avoids unsupported fancy indexing

### Security
- Added a pinned `detect-secrets` workflow with a shared `tools/secret_scan.sh` entrypoint for CI and optional local maintainer checks, replacing the invalid CI `audit --baseline` flow and removing dependence on an activated shell for local pre-commit runs
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
