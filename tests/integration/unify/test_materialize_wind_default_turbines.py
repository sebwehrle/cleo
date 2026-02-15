"""Integration tests for wind materialization with default turbines.

Tests are offline-only: no network calls, uses local test fixtures.

These tests verify that:
- When no configure_turbines() is called, default turbines are discovered
  from <atlas>/resources/*.yml
- Wind materialization succeeds with discovered turbines
- The wind store contains the discovered turbines
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS

import cleo
from cleo.unify import Unifier, GWA_HEIGHTS, _default_turbines_from_resources


class MockAtlas:
    """Minimal Atlas-like object for testing."""

    def __init__(
        self,
        path: Path,
        country: str = "AUT",
        crs: str = "epsg:3035",
        region: str | None = None,
    ):
        self.path = path
        self.country = country
        self.crs = crs
        self.region = region
        self.turbines_configured = None  # NOT configured - will use defaults
        self.chunk_policy = {"y": 1024, "x": 1024}
        self.fingerprint_method = "path_mtime_size"

        # Create required directories
        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)

    def get_nuts_region(self, region: str):
        """Return None - no NUTS region for basic tests."""
        return None


def _create_gwa_raster(
    path: Path,
    shape: tuple[int, int] = (20, 20),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    fill_value: float | None = None,
    add_nodata_region: bool = True,
) -> None:
    """Create a minimal GWA-style GeoTIFF with CRS."""
    if fill_value is not None:
        data = np.full(shape, fill_value, dtype=np.float32)
    else:
        data = np.random.rand(*shape).astype(np.float32) * 10 + 1

    if add_nodata_region:
        data[:3, :3] = np.nan

    transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], shape[1], shape[0]
    )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
        "nodata": np.nan,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT") -> None:
    """Create all required GWA rasters for wind materialization."""
    raw_dir = atlas_path / "data" / "raw" / country

    for h in GWA_HEIGHTS:
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            fill_value=8.0 + h * 0.01,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            fill_value=2.0 + h * 0.001,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            fill_value=1.2 - h * 0.0001,
        )


def _copy_turbine_yaml(atlas_path: Path, turbine_name: str) -> None:
    """Copy turbine YAML from package resources to test resources."""
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)

    src = resources_src / f"{turbine_name}.yml"
    assert src.exists(), f"Turbine YAML not found: {src}"
    shutil.copy(src, resources_dest / f"{turbine_name}.yml")


def _copy_all_turbine_yamls(atlas_path: Path) -> list[str]:
    """Copy all turbine YAMLs from package resources to test resources.

    Returns:
        List of turbine IDs that were copied.
    """
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)

    # Get all YAMLs from package resources, excluding non-turbine files
    non_turbine_stems = {"clc_codes", "cost_assumptions"}
    turbine_ids = []

    for pattern in ("*.yml", "*.yaml"):
        for src_path in resources_src.glob(pattern):
            stem = src_path.stem
            if stem not in non_turbine_stems:
                shutil.copy(src_path, resources_dest / src_path.name)
                turbine_ids.append(stem)

    return sorted(set(turbine_ids))


class TestDefaultTurbinesDiscovery:
    """Tests for _default_turbines_from_resources helper."""

    def test_discovers_turbines_from_yamls(self, tmp_path: Path) -> None:
        """Discovers turbine IDs from *.yml files."""
        resources_dir = tmp_path / "resources"
        resources_dir.mkdir()

        # Create some fake turbine YAMLs
        (resources_dir / "Turbine.A.100.yml").touch()
        (resources_dir / "Turbine.B.200.yml").touch()
        (resources_dir / "Turbine.C.300.yaml").touch()

        result = _default_turbines_from_resources(resources_dir)

        assert result == ["Turbine.A.100", "Turbine.B.200", "Turbine.C.300"]

    def test_excludes_non_turbine_resources(self, tmp_path: Path) -> None:
        """Excludes known non-turbine resource files."""
        resources_dir = tmp_path / "resources"
        resources_dir.mkdir()

        # Create turbine and non-turbine YAMLs
        (resources_dir / "Turbine.A.100.yml").touch()
        (resources_dir / "clc_codes.yml").touch()
        (resources_dir / "cost_assumptions.yml").touch()

        result = _default_turbines_from_resources(resources_dir)

        assert result == ["Turbine.A.100"]
        assert "clc_codes" not in result
        assert "cost_assumptions" not in result

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        """Returns turbine IDs in sorted order."""
        resources_dir = tmp_path / "resources"
        resources_dir.mkdir()

        # Create YAMLs in non-sorted order
        (resources_dir / "Zebra.Z.999.yml").touch()
        (resources_dir / "Alpha.A.100.yml").touch()
        (resources_dir / "Beta.B.200.yml").touch()

        result = _default_turbines_from_resources(resources_dir)

        assert result == ["Alpha.A.100", "Beta.B.200", "Zebra.Z.999"]

    def test_returns_empty_if_no_yamls(self, tmp_path: Path) -> None:
        """Returns empty list if no YAML files exist."""
        resources_dir = tmp_path / "resources"
        resources_dir.mkdir()

        result = _default_turbines_from_resources(resources_dir)

        assert result == []

    def test_returns_empty_if_dir_missing(self, tmp_path: Path) -> None:
        """Returns empty list if resources directory doesn't exist."""
        resources_dir = tmp_path / "nonexistent"

        result = _default_turbines_from_resources(resources_dir)

        assert result == []


class TestMaterializeWindDefaultTurbines:
    """Integration tests for wind materialization with default turbines."""

    def test_materialize_wind_uses_default_turbines(self, tmp_path: Path) -> None:
        """materialize_wind uses turbines from resources when not configured."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # Copy turbine YAMLs to resources (do NOT configure turbines)
        copied_turbines = _copy_all_turbine_yamls(tmp_path)
        assert len(copied_turbines) >= 1, "Need at least one turbine YAML"

        # Verify turbines_configured is None
        assert atlas.turbines_configured is None

        # Materialize wind
        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)

        # Verify wind store was created
        wind_path = tmp_path / "wind.zarr"
        assert wind_path.exists()

        # Verify turbines are in the wind store
        ds = xr.open_zarr(wind_path, consolidated=False)
        assert "turbine" in ds.dims, "Wind store should have turbine dimension"
        assert ds.sizes["turbine"] >= 1, "Should have at least one turbine"

        # Verify turbine IDs match discovered turbines
        import json
        root = zarr.open_group(wind_path, mode="r")
        turbines_json = root.attrs.get("cleo_turbines_json", "[]")
        turbines_meta = json.loads(turbines_json)
        turbine_ids = [t["id"] for t in turbines_meta]

        assert len(turbine_ids) >= 1
        assert set(turbine_ids) == set(copied_turbines)

    def test_materialize_wind_with_single_turbine(self, tmp_path: Path) -> None:
        """materialize_wind works with a single default turbine."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # Copy just one turbine
        turbine_name = "Enercon.E40.500"
        _copy_turbine_yaml(tmp_path, turbine_name)

        # Verify turbines_configured is None
        assert atlas.turbines_configured is None

        # Materialize wind
        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)

        # Verify the specific turbine is present
        import json
        root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        turbines_json = root.attrs.get("cleo_turbines_json", "[]")
        turbines_meta = json.loads(turbines_json)
        turbine_ids = [t["id"] for t in turbines_meta]

        assert turbine_ids == [turbine_name]

    def test_materialize_wind_raises_if_no_turbines(self, tmp_path: Path) -> None:
        """materialize_wind raises RuntimeError if no turbines available."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # Do NOT copy any turbine YAMLs
        # resources/ exists but is empty (created by MockAtlas)

        # Verify turbines_configured is None
        assert atlas.turbines_configured is None

        # materialize_wind should raise RuntimeError
        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        with pytest.raises(RuntimeError, match="No turbines configured"):
            unifier.materialize_wind(atlas)

    def test_configured_turbines_take_precedence(self, tmp_path: Path) -> None:
        """Configured turbines take precedence over discovered defaults."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # Copy multiple turbine YAMLs
        _copy_all_turbine_yamls(tmp_path)

        # Configure a specific subset
        configured_turbine = "Enercon.E40.500"
        atlas.turbines_configured = [configured_turbine]

        # Materialize wind
        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)

        # Verify only configured turbine is present
        import json
        root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        turbines_json = root.attrs.get("cleo_turbines_json", "[]")
        turbines_meta = json.loads(turbines_json)
        turbine_ids = [t["id"] for t in turbines_meta]

        assert turbine_ids == [configured_turbine]

    def test_inputs_id_differs_for_different_defaults(self, tmp_path: Path) -> None:
        """inputs_id differs when default turbines change."""
        # Create two atlas instances with different default turbines
        atlas1_path = tmp_path / "atlas1"
        atlas2_path = tmp_path / "atlas2"

        atlas1 = MockAtlas(atlas1_path)
        atlas2 = MockAtlas(atlas2_path)

        # Create GWA rasters for both
        _create_all_gwa_rasters(atlas1_path)
        _create_all_gwa_rasters(atlas2_path)

        # Copy different turbines to each
        _copy_turbine_yaml(atlas1_path, "Enercon.E40.500")
        _copy_turbine_yaml(atlas2_path, "Enercon.E40.500")
        # Add a second turbine only to atlas2
        # First check if there's another turbine available
        resources_src = Path(cleo.__file__).resolve().parent / "resources"
        all_yamls = list(resources_src.glob("*.yml"))
        non_turbine_stems = {"clc_codes", "cost_assumptions"}
        other_turbines = [
            p.stem for p in all_yamls
            if p.stem not in non_turbine_stems and p.stem != "Enercon.E40.500"
        ]

        if other_turbines:
            second_turbine = other_turbines[0]
            _copy_turbine_yaml(atlas2_path, second_turbine)

            # Materialize both
            unifier1 = Unifier(chunk_policy={"y": 1024, "x": 1024})
            unifier1.materialize_wind(atlas1)

            unifier2 = Unifier(chunk_policy={"y": 1024, "x": 1024})
            unifier2.materialize_wind(atlas2)

            # Get inputs_ids
            root1 = zarr.open_group(atlas1_path / "wind.zarr", mode="r")
            root2 = zarr.open_group(atlas2_path / "wind.zarr", mode="r")

            inputs_id1 = root1.attrs["inputs_id"]
            inputs_id2 = root2.attrs["inputs_id"]

            assert inputs_id1 != inputs_id2, (
                "inputs_id should differ when default turbines differ"
            )
