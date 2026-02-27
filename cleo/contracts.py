"""Schema versioning constants for CLEO exports.

These constants define schema versions for external-facing exports that promise
schema stability to downstream users. Internal stores (wind.zarr, landscape.zarr)
use `inputs_id` / `grid_id` / `unify_version` for content+code identity and do
NOT use these schema versions.

Schema Version Semantics
------------------------
A schema version represents a *layout/interpretation contract* for exported data.
The version must be bumped when any of the following changes occur:

- Removal or renaming of an existing variable/column
- Change in variable dtype that affects interpretation (e.g., float64 -> float32)
- Change in coordinate semantics (e.g., cell-center vs. cell-edge)
- Change in units without backward-compatible conversion
- Change in attribute semantics that downstream parsers rely on
- Structural changes to export layout (e.g., new required dimensions)

The version should NOT be bumped for:

- Addition of new optional variables/columns (additive changes)
- Internal refactoring that preserves output byte-for-byte
- Performance optimizations with identical output
- Documentation or metadata clarifications that don't affect parsing
"""

# Schema version for consolidated analysis xarray exports (zarr/netcdf).
# Used by future `Atlas.export_analysis_dataset_zarr()` or equivalent API.
ANALYSIS_EXPORT_SCHEMA_VERSION: int = 1

# Schema version for flatten() tabular exports and sidecar manifests.
# Used when writing parquet/csv exports with schema metadata.
FLATTEN_EXPORT_SCHEMA_VERSION: int = 1
