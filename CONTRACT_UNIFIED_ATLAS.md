CONTRACT_UNIFIED_ATLAS.md

Purpose
This document defines the non-negotiable architectural contract for Cleo’s “unified atlas” design.
It is intended to be enforced by automated checks and code review.

Definitions
- Raw source: Any non-Zarr file or external resource (e.g., GeoTIFF, Shapefile/GeoJSON, YAML/CSV, legacy NetCDF).
- Canonical store: A Zarr store produced by the Unifier. Canonical stores are the only allowed input to analysis code.
- Wind store: The canonical Zarr store containing wind-resource layers and technology/cost inputs.
- Landscape store: The canonical Zarr store containing landscape layers, masks, and rasterized zones.
- Results store: A disposable Zarr store holding derived outputs for a given run/metric.
- Grid identity (grid_id): A deterministic identifier derived from CRS + transform + shape + coordinate arrays + nodata/mask policy.
- Inputs identity (inputs_id): A deterministic identifier derived from raw source fingerprints + unification parameters + technology/cost inputs.
- Unify version (unify_version): A code identifier stored as the git commit hash; additionally store whether the code was “dirty”.

Top-level invariants (MUST hold)
1) Canonical-only analysis
   All analysis computations must operate exclusively on data originating from canonical Zarr stores.
   Analysis code must not open/read any raw source.

2) Single unification boundary
   Reprojection, clipping/masking to AOI, alignment/resampling to the canonical grid, and nodata normalization
   must occur only during unification, never during analysis.

3) Unified grid across the atlas
   Wind and landscape stores MUST share the same grid identity (same CRS, transform, shape, coordinates).

4) Dask-first execution
   Canonical data is dask-backed; chunking is applied only on spatial dimensions (x,y) unless explicitly justified.
   Analysis code must not trigger implicit computation (no hidden .compute()).

Roles and allowed I/O (STRICT)
A) Unifier
   - The Unifier is the ONLY component allowed to read raw sources.
   - The Unifier is allowed to write/update canonical Zarr stores (single-writer policy).
   - The Unifier is responsible for maintaining the “registered sources manifest” inside each canonical store.

B) Atlas / public classes
   - Allowed I/O: open/read canonical Zarr stores; write results Zarr stores; export final deliverables.
   - Forbidden I/O: reading raw sources of any kind.

C) Analysis modules (e.g., wind assessment computations)
   - Forbidden I/O: any file open/read, including Zarr.
   - Inputs MUST be xr.Dataset/xr.DataArray originating from canonical stores.

Canonical stores: required contents
1) Wind store MUST contain:
   - wind resource layers on canonical grid (e.g., Weibull parameters; optional air density)
   - technology inputs:
       * power_curve(turbine, wind_speed)
       * turbine_* metadata variables on turbine dimension
   - cost assumptions as scalar variables (not read from YAML at runtime)
   - required attrs: grid_id, inputs_id, unify_version, code_dirty, chunk_policy

2) Landscape store MUST contain:
   - landscape layers / masks on canonical grid
   - rasterized zones if needed for downstream zonal computations
   - required attrs: grid_id, inputs_id, unify_version, code_dirty, chunk_policy

Registered sources manifest (MUST live inside the Zarr store)
- Each canonical store MUST include a manifest describing:
  - which raw sources are registered (name, kind, path, params, fingerprint)
  - which variables were materialized from which registered source
- The manifest must be self-contained within the Zarr store (no required external sidecar files).

Incremental updates (SUPPORTED, still clean)
1) Landscape incremental adds:
   - It MUST be possible to register a new landscape source and materialize it into the landscape store
     without rebuilding unrelated existing variables.
   - Materialization may be performed on demand ONLY for sources already registered in the manifest.

2) Turbine selection for computations:
   - It MUST be possible to compute outputs for any subset of turbines present in the wind store,
     without additional raw I/O.
   - Adding new turbines requires unification ingestion (raw YAML read by Unifier) and updates wind store.

Results stores and export
- Results stores are disposable and user-cleanable by design.
- There MUST be an export method that writes a single deliverable artifact to disk
  (e.g., one Zarr root with groups wind/landscape/results and/or a NetCDF export for final sharing).

Concurrency
- Single-writer policy for canonical stores: at most one writer process at a time.
- Readers may be concurrent.

Compliance checks
The repository MUST include automated checks that fail if:
- Any module outside Unifier reads raw sources (e.g., rxr.open_rasterio, rasterio.open, gpd.read_file, yaml.safe_load, xr.open_dataset).
- Analysis modules perform file I/O or open Zarr directly.
- Canonical stores lack required attrs or manifest structure.

Local contract enforcement
Run the contract check locally before committing:

```bash
make contract-check
```

This executes `tools/check_contract.sh` and verifies:
1. **Raw I/O boundary**: `rxr.open_rasterio`, `rasterio.open`, `gpd.read_file`, `yaml.safe_load`, `xr.open_dataset`, etc. only appear in `cleo/unify.py` and `cleo/loaders.py`.
2. **Zarr I/O boundary**: `xr.open_zarr`, `.to_zarr()` only appear in `cleo/classes.py` and `cleo/unify.py`.
3. **NetCDF I/O boundary**: `to_netcdf()` only appears in `cleo/classes.py`.
4. **assess.py purity**: `cleo/assess.py` contains no I/O calls (pure compute).
5. **loaders import restriction**: `cleo.loaders` may only be imported in `cleo/unify.py`.

Violations are reported with file:line matches and a hint for resolution.
