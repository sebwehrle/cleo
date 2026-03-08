"""Unified Atlas foundations: core unification/materialization coordination."""

from __future__ import annotations

from pathlib import Path

from cleo.unification.materializers._landscape_core import materialize_landscape
from cleo.unification.materializers.shared import (
    ensure_store_skeleton as ensure_store_skeleton_shared,
)
from cleo.unification.materializers import region as area_materializer
from cleo.unification.materializers import wind as wind_materializer

# =============================================================================
# Unifier Class
# =============================================================================


class Unifier:
    """Unified atlas store management.

    The Unifier coordinates canonical store and area-store materialization with
    proper versioning, identity tracking, and metadata.
    """

    def __init__(
        self,
        *,
        chunk_policy: dict[str, int] | None = None,
        fingerprint_method: str = "path_mtime_size",
    ) -> None:
        """
        Initialize the Unifier.

        :param chunk_policy: Default chunking policy.
        :param fingerprint_method: Method for computing source fingerprints.
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

        :param atlas: Atlas instance with path, country, CRS, and area attributes.
        :returns: ``None``
        :raises FileNotFoundError: If required GWA files are missing.
        :raises RuntimeError: If CRS fetch/assignment fails.
        """
        wind_materializer.materialize_wind(self, atlas)

    def materialize_landscape(self, atlas) -> None:
        """Materialize landscape.zarr as a complete canonical store."""
        return materialize_landscape(self, atlas)

    def materialize_area(self, atlas, area_id: str) -> None:
        """Materialize area stores by subsetting from country stores."""
        return area_materializer.materialize_area(self, atlas, area_id)

    def ensure_area_stores(self, atlas, area_id: str, *, logger) -> None:
        """Ensure area stores exist for a given area selection.

        This method handles area store lifecycle including:
        - Removing stale/partial area directories
        - Delegating to ``materialize_area`` for freshness/completeness checks
        - Verifying area stores exist after materialization

        :param atlas: Atlas instance with path and area attributes.
        :param area_id: The area identifier.
        :param logger: Logger instance for diagnostics.
        :raises RuntimeError: If area stores are missing after materialization.
        """
        return area_materializer._ensure_area_stores_ready(
            atlas=atlas,
            unifier=self,
            area_id=area_id,
            logger=logger,
        )
