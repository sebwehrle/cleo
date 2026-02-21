"""Manifest helpers extracted from ``cleo.unify``."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import zarr


# Manifest is stored in zarr root attrs:
#   cleo_manifest_sources_json: JSON array of source entries
#   cleo_manifest_variables_json: JSON array of variable entries


def _read_manifest(store_path: Path) -> dict:
    """Read manifest from zarr root attrs.

    Returns:
        Manifest dict with keys: version, sources, variables.
        Returns empty manifest if attrs not present.
    """
    try:
        root = zarr.open_group(store_path, mode="r")
        sources_json = root.attrs.get("cleo_manifest_sources_json", "[]")
        variables_json = root.attrs.get("cleo_manifest_variables_json", "[]")
        return {
            "version": 1,
            "sources": json.loads(sources_json),
            "variables": json.loads(variables_json),
        }
    except Exception:
        return {"version": 1, "sources": [], "variables": []}


def _write_manifest_atomic(store_path: Path, manifest: dict) -> None:
    """Write manifest to zarr root attrs."""
    root = zarr.open_group(store_path, mode="a")
    root.attrs["cleo_manifest_sources_json"] = json.dumps(
        manifest.get("sources", []), separators=(",", ":"), ensure_ascii=False
    )
    root.attrs["cleo_manifest_variables_json"] = json.dumps(
        manifest.get("variables", []), separators=(",", ":"), ensure_ascii=False
    )


def init_manifest(store_path: Path) -> None:
    """Initialize empty manifest in zarr root attrs if not present."""
    try:
        root = zarr.open_group(store_path, mode="a")
        if "cleo_manifest_sources_json" not in root.attrs:
            root.attrs["cleo_manifest_sources_json"] = "[]"
        if "cleo_manifest_variables_json" not in root.attrs:
            root.attrs["cleo_manifest_variables_json"] = "[]"
    except Exception:
        pass  # Store may not exist yet; will be initialized on first write


def write_manifest_sources(store_path: Path, sources: list[dict]) -> None:
    """Write/replace sources in the manifest.

    :param store_path: Path to the Zarr store root.
    :param sources: Source dictionaries with keys including ``source_id``,
        ``name``, ``kind``, ``path``, ``params_json``, and ``fingerprint``.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = _read_manifest(store_path)

    # Add created_at timestamp to each source
    sources_with_ts = []
    for s in sources:
        s_copy = dict(s)
        s_copy["created_at"] = now
        sources_with_ts.append(s_copy)

    manifest["sources"] = sources_with_ts
    _write_manifest_atomic(store_path, manifest)


def write_manifest_variables(store_path: Path, variables: list[dict]) -> None:
    """Write/replace variables in the manifest.

    :param store_path: Path to the Zarr store root.
    :param variables: Variable dictionaries with keys including
        ``variable_name``, ``source_id``, ``resampling_method``,
        ``nodata_policy``, and ``dtype``.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = _read_manifest(store_path)

    # Add materialized_at timestamp to each variable
    vars_with_ts = []
    for v in variables:
        v_copy = dict(v)
        v_copy["materialized_at"] = now
        vars_with_ts.append(v_copy)

    manifest["variables"] = vars_with_ts
    _write_manifest_atomic(store_path, manifest)
