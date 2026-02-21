"""Backward-compatible facade for the Phase 1 unification split.

This module re-exports selected symbols from ``cleo.unification`` and
``cleo.unification.unifier`` to keep legacy import paths stable.
"""

from cleo.unification.fingerprint import (
    fingerprint_file,
    fingerprint_path_mtime_size,
    get_git_info,
    hash_grid_id,
    hash_inputs_id,
)
from cleo.unification.gwa_io import (
    GWA_HEIGHTS,
    _assert_all_required_gwa_present,
    _crs_cache_path,
    _load_or_fetch_gwa_crs,
    _open_gwa_raster,
    _required_gwa_files,
)
from cleo.unification.manifest import (
    _read_manifest,
    _write_manifest_atomic,
    init_manifest,
    write_manifest_sources,
    write_manifest_variables,
)
from cleo.unification.nuts_io import _read_nuts_region_catalog, _read_vector_file
from cleo.unification.raster_io import (
    _atomic_replace_variable_dir,
    _build_copdem_elevation,
    _build_copdem_mosaic,
    _open_dataset,
    _open_local_elevation,
    _open_raster,
)
from cleo.unification.turbines import (
    _default_turbines_from_resources,
    _ingest_turbines_and_costs,
    _load_turbine_yaml,
)
from cleo.unification.unifier import (
    Unifier,
    _aoi_geom_or_none,
    _ensure_crs_from_gwa,
    _get_clip_geometry,
    _now_iso,
    _nuts_region_geom,
    _stable_json,
)

PUBLIC_EXPORTS = [
    "Unifier",
    "GWA_HEIGHTS",
    "get_git_info",
    "hash_grid_id",
    "hash_inputs_id",
    "fingerprint_file",
    "fingerprint_path_mtime_size",
    "init_manifest",
    "write_manifest_sources",
    "write_manifest_variables",
]

# Transitional compatibility exports. Keep until long-term refactor completion,
# then remove in the final consolidation cleanup.
COMPAT_EXPORTS = [
    # Used by existing compat and unit tests
    "_read_vector_file",
    "_read_nuts_region_catalog",
    "_default_turbines_from_resources",
    "_crs_cache_path",
    "_load_or_fetch_gwa_crs",
    "_open_gwa_raster",
    "_stable_json",
    "_now_iso",
    # Additional private compatibility paths kept during transition
    "_read_manifest",
    "_write_manifest_atomic",
    "_required_gwa_files",
    "_assert_all_required_gwa_present",
    "_open_raster",
    "_open_dataset",
    "_build_copdem_mosaic",
    "_atomic_replace_variable_dir",
    "_open_local_elevation",
    "_build_copdem_elevation",
    "_default_turbines_from_resources",
    "_ingest_turbines_and_costs",
    "_load_turbine_yaml",
    "_nuts_region_geom",
    "_aoi_geom_or_none",
    "_get_clip_geometry",
    "_ensure_crs_from_gwa",
]

__all__ = [*PUBLIC_EXPORTS, *COMPAT_EXPORTS]
