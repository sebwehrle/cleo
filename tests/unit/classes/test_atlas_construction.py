"""Tests for Atlas construction contract compliance (docs/CONTRACT_UNIFIED_ATLAS.md A2).

Verifies:
- Atlas accepts all constructor parameters per contract A2
- region parameter is optional and stored correctly
- Construction performs no heavy I/O
"""

import pytest
import zarr
import json
from pathlib import Path
from types import SimpleNamespace
from cleo.atlas import Atlas


def _mock_region_catalog(tmp_path: Path) -> None:
    # Create minimal landscape.zarr root group with attrs only (no arrays needed).
    g = zarr.open_group(str(tmp_path / "landscape.zarr"), mode="w")
    catalog = [
        {
            "name": "Niederösterreich",
            "name_norm": "niederösterreich",
            "nuts_id": "AT13",
            "level": 2,
        }
    ]
    g.attrs["cleo_region_catalog_json"] = json.dumps(catalog, sort_keys=True, separators=(",", ":"))


class TestAtlasConstructorContractA2:
    """Tests for Atlas.__init__() per contract section A2."""

    def test_atlas_exposes_build_lifecycle_and_not_legacy_materialize_names(self, tmp_path: Path) -> None:
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
        )

        assert hasattr(atlas, "build")
        assert hasattr(atlas, "build_canonical")
        assert hasattr(atlas, "build_clc")

        assert not hasattr(atlas, "materialize")
        assert not hasattr(atlas, "materialize_canonical")
        assert not hasattr(atlas, "materialize_clc")

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

        _mock_region_catalog(tmp_path)

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
            compute_backend="threads",
            compute_workers=4,
            region="Niederösterreich",
            results_root=tmp_path / "custom_results",
        )
        assert atlas.region == "Niederösterreich"
        assert atlas.chunk_policy == {"y": 512, "x": 512}
        assert atlas.compute_backend == "threads"
        assert atlas.compute_workers == 4
        assert atlas.results_root == tmp_path / "custom_results"

    def test_atlas_compute_backend_defaults_to_serial(self, tmp_path: Path) -> None:
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
        )
        assert atlas.compute_backend == "serial"
        assert atlas.compute_workers is None

    def test_atlas_compute_backend_rejects_invalid_value(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown compute_backend"):
            Atlas(
                tmp_path,
                country="AUT",
                crs="epsg:3035",
                compute_backend="invalid",
            )

    def test_atlas_compute_workers_rejects_non_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="compute_workers must be >= 1"):
            Atlas(
                tmp_path,
                country="AUT",
                crs="epsg:3035",
                compute_backend="threads",
                compute_workers=0,
            )

    def test_atlas_compute_workers_rejects_serial_gt_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="compute_workers must be None or 1"):
            Atlas(
                tmp_path,
                country="AUT",
                crs="epsg:3035",
                compute_backend="serial",
                compute_workers=2,
            )

    def test_atlas_compute_workers_rejects_distributed_override(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not supported for compute_backend='distributed'"):
            Atlas(
                tmp_path,
                country="AUT",
                crs="epsg:3035",
                compute_backend="distributed",
                compute_workers=2,
            )

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

    def test_clone_for_selection_preserves_compute_policy(self, tmp_path: Path) -> None:
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
            compute_backend="threads",
            compute_workers=3,
        )

        clone = atlas._clone_for_selection()

        assert clone.compute_backend == "threads"
        assert clone.compute_workers == 3

    def test_select_copy_preserves_compute_policy(self, tmp_path: Path) -> None:
        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
            compute_backend="threads",
            compute_workers=3,
        )

        clone = atlas.select(region=None, inplace=False)
        assert clone is not None
        assert clone.compute_backend == "threads"
        assert clone.compute_workers == 3

    def test_build_invalidates_wind_and_landscape_transients(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas._canonical_ready = True

        wind = SimpleNamespace(
            _data=object(),
            _computed_overlays={"mean_wind_speed": object()},
            clear_computed=lambda: wind._computed_overlays.clear(),
        )
        land = SimpleNamespace(
            _data=object(),
            _staged_overlays={"foo": object()},
            clear_staged=lambda: land._staged_overlays.clear(),
        )
        atlas._wind_domain = wind
        atlas._landscape_domain = land

        atlas.build()

        assert wind._data is None
        assert wind._computed_overlays == {}
        assert land._data is None
        assert land._staged_overlays == {}

    def test_select_inplace_invalidates_wind_and_landscape_transients(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        wind = SimpleNamespace(
            _data=object(),
            _computed_overlays={"mean_wind_speed": object()},
            clear_computed=lambda: wind._computed_overlays.clear(),
        )
        land = SimpleNamespace(
            _data=object(),
            _staged_overlays={"foo": object()},
            clear_staged=lambda: land._staged_overlays.clear(),
        )
        atlas._wind_domain = wind
        atlas._landscape_domain = land

        atlas.select(region=None, inplace=True)

        assert wind._data is None
        assert wind._computed_overlays == {}
        assert land._data is None
        assert land._staged_overlays == {}

    def test_build_canonical_invalidates_wind_and_landscape_transients(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        class _DummyUnifier:
            def __init__(self, **kwargs):  # noqa: ANN003
                pass

            def materialize_wind(self, atlas):  # noqa: ANN001
                return None

            def materialize_landscape(self, atlas):  # noqa: ANN001
                return None

        monkeypatch.setattr("cleo.unification.Unifier", _DummyUnifier)

        wind = SimpleNamespace(
            _data=object(),
            _computed_overlays={"mean_wind_speed": object()},
            clear_computed=lambda: wind._computed_overlays.clear(),
        )
        land = SimpleNamespace(
            _data=object(),
            _staged_overlays={"foo": object()},
            clear_staged=lambda: land._staged_overlays.clear(),
        )
        atlas._wind_domain = wind
        atlas._landscape_domain = land

        atlas.build_canonical()

        assert wind._data is None
        assert wind._computed_overlays == {}
        assert land._data is None
        assert land._staged_overlays == {}

    def test_build_clc_invalidates_wind_and_landscape_transients(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        prepared = tmp_path / "prepared_clc.tif"

        monkeypatch.setattr("cleo.clc.materialize_clc", lambda *a, **k: prepared)

        wind = SimpleNamespace(
            _data=object(),
            _computed_overlays={"mean_wind_speed": object()},
            clear_computed=lambda: wind._computed_overlays.clear(),
        )
        land = SimpleNamespace(
            _data=object(),
            _staged_overlays={"foo": object()},
            clear_staged=lambda: land._staged_overlays.clear(),
        )
        atlas._wind_domain = wind
        atlas._landscape_domain = land

        out = atlas.build_clc()

        assert out == prepared
        assert wind._data is None
        assert wind._computed_overlays == {}
        assert land._data is None
        assert land._staged_overlays == {}
