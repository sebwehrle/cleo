"""Tests for Atlas construction contract compliance (CONTRACT_UNIFIED_ATLAS.md A2).

Verifies:
- Atlas accepts all constructor parameters per contract A2
- region parameter is optional and stored correctly
- Construction performs no heavy I/O
"""

import pytest
import zarr
import json
from pathlib import Path
from cleo.atlas import Atlas

def _mock_region_index(tmp_path: Path) -> None:
    # Create minimal landscape.zarr root group with attrs only (no arrays needed)
    g = zarr.open_group(str(tmp_path / "landscape.zarr"), mode="w")
    name_to_id = {
        # keys must match whatever _normalize_region_name() produces
        "niederösterreich": "AT13",
    }
    g.attrs["cleo_region_name_to_id_json"] = json.dumps(
        name_to_id, sort_keys=True, separators=(",", ":")
    )

class TestAtlasConstructorContractA2:
    """Tests for Atlas.__init__() per contract section A2."""

    def test_atlas_accepts_region_parameter(self, tmp_path: Path) -> None:
        """Atlas constructor accepts region parameter per contract A2.

        Contract A2 states:
        - region=None is optional at construction time
        - region must also be settable later (A4)
        """
        # Should not raise TypeError for 'region' parameter
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
            region="Niederösterreich",
        )
        assert atlas.region == "Niederösterreich"

    def test_atlas_region_defaults_to_none(self, tmp_path: Path) -> None:
        """Atlas region defaults to None when not specified."""
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
        )
        assert atlas.region is None

    def test_atlas_region_can_be_set_after_construction(self, tmp_path: Path) -> None:
        """Atlas region can be changed after construction per contract A4."""
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
        )
        assert atlas.region is None

        _mock_region_index(tmp_path)

        # Contract A4: region must be changeable at any time
        atlas.region = "Niederösterreich"
        assert atlas.region == "Niederösterreich"

        # Clear region selection
        atlas.region = None
        assert atlas.region is None

    def test_atlas_accepts_all_optional_parameters(self, tmp_path: Path) -> None:
        """Atlas constructor accepts all optional parameters per contract A2."""
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
            chunk_policy={"y": 512, "x": 512},
            region="Niederösterreich",
            results_root=tmp_path / "custom_results",
        )
        assert atlas.region == "Niederösterreich"
        assert atlas.chunk_policy == {"y": 512, "x": 512}
        assert atlas.results_root == tmp_path / "custom_results"

    def test_atlas_construction_performs_no_heavy_io(self, tmp_path: Path) -> None:
        """Atlas construction performs no heavy I/O per contract A2.

        Contract A2 states: 'Construction performs no heavy I/O.'
        This is verified by checking that no zarr stores are created.
        """
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
            region="Niederösterreich",
        )

        # No zarr stores should exist after construction
        assert not (tmp_path / "wind.zarr").exists()
        assert not (tmp_path / "landscape.zarr").exists()

        # _canonical_ready should be False
        assert atlas._canonical_ready is False
