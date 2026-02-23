"""
Unified Atlas foundations: Unifier and __manifest__ schema helpers.

Phase 3 status:
- Unifier is a thin coordination facade.
- Materialization/store I/O semantics are owned by cleo.unification.materializers.*.
- Compatibility helper paths are preserved via forwarders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from cleo.unification.gwa_io import ensure_crs_from_gwa as ensure_crs_from_gwa_gwaio
from cleo.unification.materializers.shared import (
    _aoi_geom_or_none as _aoi_geom_or_none_shared,
    _get_clip_geometry as _get_clip_geometry_shared,
    _now_iso as _now_iso_shared,
    _nuts_region_geom as _nuts_region_geom_shared,
    _stable_json as _stable_json_shared,
    ensure_store_skeleton as ensure_store_skeleton_shared,
)
from cleo.unification.materializers import landscape as landscape_materializer
from cleo.unification.materializers import region as region_materializer
from cleo.unification.materializers import wind as wind_materializer


# =============================================================================
# NUTS Region Geometry
# =============================================================================


def _get_clip_geometry(atlas):
    """Compatibility forwarder to shared helper ownership."""
    return _get_clip_geometry_shared(atlas)


# =============================================================================
# Unifier Class
# =============================================================================


class Unifier:
    """Unified atlas store management.

    The Unifier handles creation and validation of unified atlas stores
    with proper versioning, identity tracking, and metadata.
    """

    def __init__(
        self,
        *,
        chunk_policy: dict[str, int] | None = None,
        fingerprint_method: str = "path_mtime_size",
    ) -> None:
        """Initialize the Unifier.

        Args:
            chunk_policy: Default chunking policy (e.g., {"x": 512, "y": 512}).
            fingerprint_method: Method for computing source fingerprints.
                Default is "path_mtime_size".
        """
        self.chunk_policy = chunk_policy or {}
        self.fingerprint_method = fingerprint_method

    def ensure_store_skeleton(
        self,
        store_path: Path,
        *,
        chunk_policy: dict[str, int],
    ) -> None:
        """Ensure a store skeleton exists at the given path.

        If the store exists, ensures __manifest__ group is present.
        If missing, creates a new skeleton store atomically.

        The skeleton store has:
        - store_state="skeleton" attribute (marker for incomplete store)
        - Placeholder grid_id and inputs_id
        - Version and code state information
        - __manifest__ group with source/variable tables

        :param store_path: Path to the Zarr store.
        :param chunk_policy: Chunking policy for the store.
        """
        ensure_store_skeleton_shared(self, store_path, chunk_policy=chunk_policy)

    def materialize_wind(self, atlas) -> None:
        """Materialize wind.zarr as a complete canonical store.

        Creates a unified wind dataset from GWA GeoTIFFs, turbine configs,
        and cost assumptions. The store is marked as store_state="complete"
        only after successful atomic write.

        :param atlas: Atlas instance with path, country, CRS, and region attributes.
        :returns: ``None``
        :raises FileNotFoundError: If required GWA files are missing.
        :raises RuntimeError: If CRS fetch/assignment fails.
        """
        wind_materializer.materialize_wind(self, atlas)

    def _build_region_name_index(self, atlas) -> dict[str, str]:
        """Build region-name to region-id mapping from NUTS metadata."""
        return landscape_materializer._build_region_name_index(self, atlas)

    def materialize_landscape(self, atlas) -> None:
        """Materialize landscape.zarr as a complete canonical store."""
        return landscape_materializer.materialize_landscape(self, atlas)

    def register_landscape_source(
        self,
        atlas,
        *,
        name: str,
        source_path: Path,
        kind: str = "raster",
        params: dict | None = None,
        if_exists: str = "error",
    ) -> bool:
        """Register a new landscape source in __manifest__/sources."""
        return landscape_materializer.register_landscape_source(
            self,
            atlas,
            name=name,
            source_path=source_path,
            kind=kind,
            params=params,
            if_exists=if_exists,
        )

    def materialize_landscape_variable(
        self,
        atlas,
        variable_name: str,
        *,
        if_exists: str = "error",
    ) -> bool:
        """Materialize a single landscape variable from a registered source."""
        return landscape_materializer.materialize_landscape_variable(
            self,
            atlas,
            variable_name,
            if_exists=if_exists,
        )

    def compute_air_density_correction(
        self,
        atlas,
        *,
        chunk_size=None,
        force: bool = False,
    ) -> xr.DataArray:
        """Compute air density correction from canonical stores."""
        return landscape_materializer.compute_air_density_correction(
            self,
            atlas,
            chunk_size=chunk_size,
            force=force,
        )

    def materialize_region(self, atlas, region_id: str) -> None:
        """Materialize region stores by subsetting from country stores."""
        return region_materializer.materialize_region(self, atlas, region_id)

    # =========================================================================
# =============================================================================
# Helper Functions for Landscape Materialization
# =============================================================================


def _stable_json(obj: Any) -> str:
    """Compatibility forwarder to shared helper ownership."""
    return _stable_json_shared(obj)


def _now_iso() -> str:
    """Compatibility forwarder to shared helper ownership."""
    return _now_iso_shared()


def _nuts_region_geom(atlas):
    """Compatibility forwarder to shared helper ownership."""
    return _nuts_region_geom_shared(atlas)


def _aoi_geom_or_none(atlas):
    """Compatibility forwarder to shared helper ownership."""
    return _aoi_geom_or_none_shared(atlas)


def _ensure_crs_from_gwa(da: xr.DataArray, iso3: str) -> xr.DataArray:
    """
    Ensure a raster DataArray has a CRS set, fetching from GWA if needed.

    Delegates to cleo.unification.gwa_io.ensure_crs_from_gwa.

    Args:
        da: Raster DataArray.
        iso3: ISO 3166-1 alpha-3 country code.

    Returns:
        DataArray with CRS set.
    """
    return ensure_crs_from_gwa_gwaio(da, iso3)
