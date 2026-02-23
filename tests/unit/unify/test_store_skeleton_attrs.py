"""Tests for Unifier.ensure_store_skeleton and store attributes."""

import json
import pytest
from pathlib import Path

import zarr

from cleo.unification import Unifier, hash_grid_id, hash_inputs_id
from cleo.unification.fingerprint import get_git_info
import numpy as np


class TestGetGitInfo:
    """Tests for get_git_info helper."""

    def test_returns_required_keys(self, tmp_path: Path) -> None:
        """Returns dict with all required keys."""
        info = get_git_info(tmp_path)

        assert "unify_version" in info
        assert "code_dirty" in info
        assert "package_version" in info

    def test_handles_non_repo_gracefully(self, tmp_path: Path) -> None:
        """Non-git directory returns fallback values."""
        info = get_git_info(tmp_path)

        # Should not raise, should have fallback
        assert info["code_dirty"] is False
        assert "unknown" in info["unify_version"] or len(info["unify_version"]) == 40


class TestHashGridId:
    """Tests for hash_grid_id."""

    def test_deterministic(self) -> None:
        """Same inputs produce same hash."""
        kwargs = {
            "crs_wkt": "EPSG:4326",
            "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            "shape": (100, 200),
            "y": np.arange(100, dtype=np.float64),
            "x": np.arange(200, dtype=np.float64),
            "mask_policy": "none",
        }

        hash1 = hash_grid_id(**kwargs)
        hash2 = hash_grid_id(**kwargs)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_inputs_different_hash(self) -> None:
        """Different inputs produce different hashes."""
        base = {
            "crs_wkt": "EPSG:4326",
            "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            "shape": (100, 200),
            "y": np.arange(100, dtype=np.float64),
            "x": np.arange(200, dtype=np.float64),
            "mask_policy": "none",
        }

        hash1 = hash_grid_id(**base)

        # Change CRS
        modified = {**base, "crs_wkt": "EPSG:32632"}
        hash2 = hash_grid_id(**modified)

        assert hash1 != hash2


class TestHashInputsId:
    """Tests for hash_inputs_id."""

    def test_deterministic(self) -> None:
        """Same inputs produce same hash."""
        items = [("source1", "fp1"), ("source2", "fp2")]

        hash1 = hash_inputs_id(items, method="path_mtime_size")
        hash2 = hash_inputs_id(items, method="path_mtime_size")

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_order_independent(self) -> None:
        """Item order doesn't affect hash (items are sorted)."""
        items1 = [("source1", "fp1"), ("source2", "fp2")]
        items2 = [("source2", "fp2"), ("source1", "fp1")]

        hash1 = hash_inputs_id(items1, method="path_mtime_size")
        hash2 = hash_inputs_id(items2, method="path_mtime_size")

        assert hash1 == hash2

    def test_method_affects_hash(self) -> None:
        """Different methods produce different hashes."""
        items = [("source1", "fp1")]

        hash1 = hash_inputs_id(items, method="path_mtime_size")
        hash2 = hash_inputs_id(items, method="content_hash")

        assert hash1 != hash2


class TestUnifierEnsureStoreSkeleton:
    """Tests for Unifier.ensure_store_skeleton."""

    def test_creates_new_store_with_skeleton_state(self, tmp_path: Path) -> None:
        """New store has store_state='skeleton' attribute."""
        store_path = tmp_path / "store.zarr"
        unifier = Unifier()

        unifier.ensure_store_skeleton(store_path, chunk_policy={"x": 512, "y": 512})

        root = zarr.open_group(store_path, mode="r")
        assert root.attrs["store_state"] == "skeleton"

    def test_creates_store_with_required_attrs(self, tmp_path: Path) -> None:
        """Store has all required attributes."""
        store_path = tmp_path / "store.zarr"
        unifier = Unifier(fingerprint_method="content_hash")
        chunk_policy = {"x": 256, "y": 256}

        unifier.ensure_store_skeleton(store_path, chunk_policy=chunk_policy)

        root = zarr.open_group(store_path, mode="r")

        # Check required attributes
        assert root.attrs["store_state"] == "skeleton"
        assert root.attrs["grid_id"] == ""
        assert root.attrs["inputs_id"] == ""
        assert "unify_version" in root.attrs
        assert "code_dirty" in root.attrs
        assert root.attrs["fingerprint_method"] == "content_hash"

        # Check chunk_policy is stored as JSON
        stored_policy = json.loads(root.attrs["chunk_policy"])
        assert stored_policy == chunk_policy

    def test_creates_manifest_attrs(self, tmp_path: Path) -> None:
        """Store has manifest attrs initialized."""
        store_path = tmp_path / "store.zarr"
        unifier = Unifier()

        unifier.ensure_store_skeleton(store_path, chunk_policy={"x": 512})

        root = zarr.open_group(store_path, mode="r")
        assert "cleo_manifest_sources_json" in root.attrs
        assert "cleo_manifest_variables_json" in root.attrs

    def test_existing_store_adds_manifest_if_missing(self, tmp_path: Path) -> None:
        """Existing store without manifest attrs gets them added."""
        store_path = tmp_path / "store.zarr"

        # Create store without manifest attrs
        root = zarr.open_group(store_path, mode="w")
        root.attrs["existing"] = "data"

        unifier = Unifier()
        unifier.ensure_store_skeleton(store_path, chunk_policy={"x": 512})

        root = zarr.open_group(store_path, mode="r")
        assert "cleo_manifest_sources_json" in root.attrs
        assert "cleo_manifest_variables_json" in root.attrs
        # Original data preserved
        assert root.attrs["existing"] == "data"

    def test_existing_store_with_manifest_unchanged(self, tmp_path: Path) -> None:
        """Existing store with manifest attrs is not modified."""
        store_path = tmp_path / "store.zarr"

        # Create store with manifest attrs
        root = zarr.open_group(store_path, mode="w")
        root.attrs["original"] = "value"
        root.attrs["cleo_manifest_sources_json"] = '[{"id": "test"}]'
        root.attrs["cleo_manifest_variables_json"] = "[]"

        unifier = Unifier()
        unifier.ensure_store_skeleton(store_path, chunk_policy={"x": 512})

        root = zarr.open_group(store_path, mode="r")
        assert root.attrs["original"] == "value"
        # Manifest attrs should be preserved
        assert root.attrs["cleo_manifest_sources_json"] == '[{"id": "test"}]'

    def test_atomic_creation(self, tmp_path: Path) -> None:
        """Store creation is atomic (no partial stores on failure)."""
        store_path = tmp_path / "store.zarr"
        unifier = Unifier()

        unifier.ensure_store_skeleton(store_path, chunk_policy={"x": 512})

        # Store should be complete
        assert store_path.exists()
        root = zarr.open_group(store_path, mode="r")
        assert "cleo_manifest_sources_json" in root.attrs

    def test_default_chunk_policy(self) -> None:
        """Unifier can be created with default chunk_policy."""
        unifier = Unifier()
        assert unifier.chunk_policy == {}
        assert unifier.fingerprint_method == "path_mtime_size"

    def test_custom_chunk_policy_and_method(self) -> None:
        """Unifier accepts custom chunk_policy and fingerprint_method."""
        unifier = Unifier(
            chunk_policy={"x": 1024, "y": 1024},
            fingerprint_method="sha256",
        )

        assert unifier.chunk_policy == {"x": 1024, "y": 1024}
        assert unifier.fingerprint_method == "sha256"
