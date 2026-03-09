"""Integration tests for consolidated analysis xarray export.

Tests the full export workflow including:
- Zarr store creation with schema versioning
- Provenance tracking and attrs
- Validation before and after write
- Domain prefixing for combined exports
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS

import cleo
from cleo.contracts import ANALYSIS_EXPORT_SCHEMA_VERSION
from cleo.unification.gwa_io import GWA_HEIGHTS
from cleo.unification.unifier import Unifier
from cleo.validation import validate_store


def _copy_default_turbine(atlas_path: Path) -> None:
    """Copy a default turbine YAML to resources for testing."""
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)

    turbine_name = "Enercon.E40.500"
    src = resources_src / f"{turbine_name}.yml"
    if src.exists():
        shutil.copy(src, resources_dest / f"{turbine_name}.yml")


class MockAtlasForExport:
    """Minimal Atlas-like object for testing export functionality."""

    def __init__(
        self,
        path: Path,
        country: str = "AUT",
        crs: str = "epsg:3035",
    ):
        self.path = path
        self.country = country
        self._crs = crs
        self.area = None
        self.turbines_configured = None
        self.chunk_policy = {"y": 1024, "x": 1024}
        self.fingerprint_method = "path_mtime_size"
        self._canonical_ready = False
        self.compute_backend = "serial"
        self.compute_workers = None

        # Create required directories
        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)

        # Copy default turbine for wind materialization
        _copy_default_turbine(path)

        # Create results_root
        self.results_root = path / "results"
        self.results_root.mkdir(parents=True, exist_ok=True)

        # Domain data caches
        self._wind_data = None
        self._landscape_data = None

    @property
    def crs(self) -> str:
        return self._crs

    @property
    def wind_store_path(self) -> Path:
        return self.path / "wind.zarr"

    @property
    def landscape_store_path(self) -> Path:
        return self.path / "landscape.zarr"

    @property
    def wind_data(self) -> xr.Dataset:
        if self._wind_data is not None:
            return self._wind_data
        if not self._canonical_ready:
            raise RuntimeError("Canonical stores not ready")
        self._wind_data = xr.open_zarr(self.wind_store_path, consolidated=False)
        return self._wind_data

    @property
    def landscape_data(self) -> xr.Dataset:
        if self._landscape_data is not None:
            return self._landscape_data
        if not self._canonical_ready:
            raise RuntimeError("Canonical stores not ready")
        self._landscape_data = xr.open_zarr(self.landscape_store_path, consolidated=False)
        return self._landscape_data

    @property
    def wind_zarr(self) -> xr.Dataset:
        return self.wind_data

    @property
    def landscape_zarr(self) -> xr.Dataset:
        return self.landscape_data

    def get_nuts_area(self, area: str):
        return None

    def build_canonical(self) -> None:
        """Materialize both wind.zarr and landscape.zarr."""
        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True


def _create_gwa_fixture(atlas_path: Path, country: str = "AUT", crs: str = "epsg:3035") -> None:
    """Create minimal GWA fixture data for testing."""
    raw_dir = atlas_path / "data" / "raw" / country
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Minimal raster parameters
    n = 5
    x0, y0 = 4600000.0, 2700000.0
    dx = dy = 100.0

    transform = rasterio.transform.from_origin(x0, y0 + n * dy, dx, dy)
    raster_crs = CRS.from_string(crs)

    # Create Weibull A and k for each height
    for h in GWA_HEIGHTS:
        a_path = raw_dir / f"{country}_combined-Weibull-A_{h}.tif"
        k_path = raw_dir / f"{country}_combined-Weibull-k_{h}.tif"

        # A: scale parameter, varies with height
        a_data = np.full((n, n), 6.0 + h / 50.0, dtype=np.float32)
        with rasterio.open(
            a_path,
            "w",
            driver="GTiff",
            height=n,
            width=n,
            count=1,
            dtype="float32",
            crs=raster_crs,
            transform=transform,
        ) as dst:
            dst.write(a_data, 1)

        # k: shape parameter, constant
        k_data = np.full((n, n), 2.0, dtype=np.float32)
        with rasterio.open(
            k_path,
            "w",
            driver="GTiff",
            height=n,
            width=n,
            count=1,
            dtype="float32",
            crs=raster_crs,
            transform=transform,
        ) as dst:
            dst.write(k_data, 1)

        # air-density: varies slightly with height
        rho_path = raw_dir / f"{country}_air-density_{h}.tif"
        rho_data = np.full((n, n), 1.2 - h * 0.0001, dtype=np.float32)
        with rasterio.open(
            rho_path,
            "w",
            driver="GTiff",
            height=n,
            width=n,
            count=1,
            dtype="float32",
            crs=raster_crs,
            transform=transform,
        ) as dst:
            dst.write(rho_data, 1)

    # Create elevation raster
    elev_path = raw_dir / f"{country}_elevation_w_bathymetry.tif"
    elev_data = np.full((n, n), 500.0, dtype=np.float32)
    with rasterio.open(
        elev_path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs=raster_crs,
        transform=transform,
    ) as dst:
        dst.write(elev_data, 1)


@pytest.fixture
def atlas_with_stores(tmp_path: Path) -> MockAtlasForExport:
    """Create a mock atlas with materialized stores."""
    atlas_path = tmp_path / "atlas"
    atlas_path.mkdir()

    _create_gwa_fixture(atlas_path)
    atlas = MockAtlasForExport(atlas_path)
    atlas.build_canonical()

    return atlas


class TestExportAnalysisDatasetZarr:
    """Integration tests for export_analysis_dataset_zarr."""

    def test_export_wind_domain(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export wind domain creates valid Zarr store."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "wind_export.zarr"

        result_path = export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="wind",
            exclude_template=True,
        )

        assert result_path == export_path
        assert export_path.exists()

        # Validate store
        validate_store(export_path, kind="export")

        # Check attrs
        g = zarr.open_group(export_path, mode="r")
        assert g.attrs["store_state"] == "complete"
        assert g.attrs["schema_version"] == ANALYSIS_EXPORT_SCHEMA_VERSION
        assert "created_at" in g.attrs
        assert "cleo:package_version" in g.attrs
        assert "export_spec_json" in g.attrs

        # Verify export spec
        spec = json.loads(g.attrs["export_spec_json"])
        assert spec["domain"] == "wind"
        assert spec["exclude_template"] is True

    def test_export_landscape_domain(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export landscape domain creates valid Zarr store."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "landscape_export.zarr"

        result_path = export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="landscape",
            exclude_template=True,
        )

        assert result_path == export_path
        assert export_path.exists()

        # Open and verify data
        ds = xr.open_zarr(export_path, consolidated=False)
        assert "valid_mask" in ds.data_vars
        assert "elevation" in ds.data_vars

    def test_export_both_domains_with_prefix(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export both domains with prefixing."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "combined_export.zarr"

        result_path = export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="both",
            prefix=True,
            exclude_template=True,
        )

        assert export_path.exists()

        # Open and verify prefixed variables
        ds = xr.open_zarr(export_path, consolidated=False)

        # Wind variables should have wind__ prefix
        wind_vars = [v for v in ds.data_vars if v.startswith("wind__")]
        assert len(wind_vars) > 0

        # Landscape variables should have landscape__ prefix
        land_vars = [v for v in ds.data_vars if v.startswith("landscape__")]
        assert len(land_vars) > 0

        # No unprefixed variables (except template if not excluded)
        assert "valid_mask" not in ds.data_vars
        assert "elevation" not in ds.data_vars

    def test_export_includes_upstream_provenance(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export includes upstream provenance attrs."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "provenance_test.zarr"

        export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="both",
        )

        g = zarr.open_group(export_path, mode="r")

        # Should have upstream provenance JSON
        assert "upstream_provenance_json" in g.attrs
        provenance = json.loads(g.attrs["upstream_provenance_json"])

        # Should reference both stores
        assert "wind_store" in provenance or "landscape_store" in provenance

    def test_export_with_include_only(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export with include_only filters variables."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "filtered_export.zarr"

        export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="both",
            prefix=True,
            include_only=["landscape__valid_mask", "landscape__elevation"],
        )

        ds = xr.open_zarr(export_path, consolidated=False)

        # Only requested variables present
        assert set(ds.data_vars) == {"landscape__valid_mask", "landscape__elevation"}

    def test_export_existing_store_raises(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export to existing path raises FileExistsError."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "existing.zarr"
        export_path.mkdir(parents=True)

        with pytest.raises(FileExistsError, match="already exists"):
            export_analysis_dataset_zarr(
                atlas_with_stores,
                export_path,
                domain="wind",
            )

    def test_export_invalid_path_extension_raises(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export to path without .zarr extension raises."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "bad_extension.nc"

        with pytest.raises(ValueError, match="must end with '.zarr'"):
            export_analysis_dataset_zarr(
                atlas_with_stores,
                export_path,
                domain="wind",
            )

    def test_export_stores_not_ready_raises(self, tmp_path: Path) -> None:
        """Export before build raises ValueError."""
        from cleo.exports import export_analysis_dataset_zarr

        atlas_path = tmp_path / "atlas_not_ready"
        atlas_path.mkdir()
        atlas = MockAtlasForExport(atlas_path)
        # Don't call build_canonical

        export_path = tmp_path / "export" / "fail.zarr"

        with pytest.raises(ValueError, match="stores not ready"):
            export_analysis_dataset_zarr(
                atlas,
                export_path,
                domain="wind",
            )

    def test_export_schema_version_matches_constant(
        self, atlas_with_stores: MockAtlasForExport, tmp_path: Path
    ) -> None:
        """Exported schema_version matches ANALYSIS_EXPORT_SCHEMA_VERSION."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "schema_test.zarr"

        export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="wind",
        )

        g = zarr.open_group(export_path, mode="r")
        assert g.attrs["schema_version"] == ANALYSIS_EXPORT_SCHEMA_VERSION
        assert isinstance(g.attrs["schema_version"], int)

    def test_export_created_at_is_valid_iso(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export created_at is valid ISO 8601 timestamp."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "timestamp_test.zarr"

        export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="wind",
        )

        g = zarr.open_group(export_path, mode="r")
        created_at = g.attrs["created_at"]

        # Should parse without error
        dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        assert dt.tzinfo is not None  # Should be timezone-aware

    def test_export_preserves_coordinates(self, atlas_with_stores: MockAtlasForExport, tmp_path: Path) -> None:
        """Export preserves coordinate values from source."""
        from cleo.exports import export_analysis_dataset_zarr

        export_path = tmp_path / "export" / "coords_test.zarr"

        # Get source coordinates
        source_y = atlas_with_stores.wind_data.coords["y"].values
        source_x = atlas_with_stores.wind_data.coords["x"].values

        export_analysis_dataset_zarr(
            atlas_with_stores,
            export_path,
            domain="wind",
        )

        ds = xr.open_zarr(export_path, consolidated=False)

        np.testing.assert_array_equal(ds.coords["y"].values, source_y)
        np.testing.assert_array_equal(ds.coords["x"].values, source_x)

    def test_compute_false_matches_compute_true_with_chunked_sources(
        self, atlas_with_stores: MockAtlasForExport, tmp_path: Path
    ) -> None:
        """compute=False skips precompute but preserves synchronous export output."""
        from cleo.exports import export_analysis_dataset_zarr

        atlas_with_stores.compute_backend = "threads"

        export_true = tmp_path / "export" / "compute_true.zarr"
        export_false = tmp_path / "export" / "compute_false.zarr"

        export_analysis_dataset_zarr(
            atlas_with_stores,
            export_true,
            domain="both",
            prefix=True,
            exclude_template=True,
            compute=True,
        )
        export_analysis_dataset_zarr(
            atlas_with_stores,
            export_false,
            domain="both",
            prefix=True,
            exclude_template=True,
            compute=False,
        )

        ds_true = xr.open_zarr(export_true, consolidated=False)
        ds_false = xr.open_zarr(export_false, consolidated=False)

        assert set(ds_true.data_vars) == set(ds_false.data_vars)
        assert set(ds_true.coords) == set(ds_false.coords)

        for coord_name in ds_true.coords:
            np.testing.assert_array_equal(ds_true.coords[coord_name].values, ds_false.coords[coord_name].values)

        for var_name in ds_true.data_vars:
            xr.testing.assert_equal(ds_true[var_name], ds_false[var_name])

        attrs_true = dict(zarr.open_group(export_true, mode="r").attrs)
        attrs_false = dict(zarr.open_group(export_false, mode="r").attrs)

        for key in ("store_state", "schema_version", "cleo:package_version", "export_spec_json"):
            assert attrs_true[key] == attrs_false[key]

        if "upstream_provenance_json" in attrs_true or "upstream_provenance_json" in attrs_false:
            assert attrs_true["upstream_provenance_json"] == attrs_false["upstream_provenance_json"]

        for attrs in (attrs_true, attrs_false):
            created_at = attrs["created_at"]
            parsed = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            assert parsed.tzinfo is not None


class TestAtlasExportMethod:
    """Integration tests for Atlas.export_analysis_dataset_zarr method."""

    def test_atlas_export_method_works(self, tmp_path: Path) -> None:
        """Atlas.export_analysis_dataset_zarr method works end-to-end."""
        from cleo import Atlas

        atlas_path = tmp_path / "atlas"
        atlas_path.mkdir()

        _create_gwa_fixture(atlas_path)

        atlas = Atlas(atlas_path, country="AUT", crs="epsg:3035")
        atlas.build()

        export_path = tmp_path / "export" / "atlas_export.zarr"

        result = atlas.export_analysis_dataset_zarr(
            export_path,
            domain="both",
            prefix=True,
            exclude_template=True,
        )

        assert result == export_path
        assert export_path.exists()

        # Validate the store
        validate_store(export_path, kind="export")
