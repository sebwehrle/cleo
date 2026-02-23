"""Shared materializer helpers.

Phase 3 PR2 moves cross-cutting helper ownership here while preserving
legacy compatibility import paths through thin forwarders in unifier.py.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import zarr

from cleo.store import atomic_dir
from cleo.unification.fingerprint import get_git_info
from cleo.unification.manifest import init_manifest


def ensure_store_skeleton(
    unifier,
    store_path: Path,
    *,
    chunk_policy: dict[str, int],
) -> None:
    """Ensure a store skeleton exists at the given path."""
    if store_path.exists():
        init_manifest(store_path)
        return

    git_info = get_git_info(Path.cwd())

    with atomic_dir(store_path) as tmp_path:
        root = zarr.open_group(tmp_path, mode="w")
        root.attrs["store_state"] = "skeleton"
        root.attrs["grid_id"] = ""
        root.attrs["inputs_id"] = ""
        root.attrs["unify_version"] = git_info["unify_version"]
        root.attrs["code_dirty"] = git_info["code_dirty"]
        if "git_diff_hash" in git_info:
            root.attrs["git_diff_hash"] = git_info["git_diff_hash"]
        root.attrs["chunk_policy"] = json.dumps(chunk_policy)
        root.attrs["fingerprint_method"] = unifier.fingerprint_method
        init_manifest(tmp_path)


def _get_clip_geometry(atlas):
    """Get clipping geometry from NUTS region if specified."""
    if atlas.region is None:
        return None
    return atlas.get_nuts_region(atlas.region)


def _stable_json(obj: Any) -> str:
    """Convert object to stable JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _nuts_region_geom(atlas):
    """Get NUTS region geometry if atlas.region is set."""
    if atlas.region is None:
        return None
    return atlas.get_nuts_region(atlas.region)


def _aoi_geom_or_none(atlas):
    """Get AOI geometry from NUTS region if specified."""
    return _nuts_region_geom(atlas)
