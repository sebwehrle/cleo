"""Tests for cleo.unification.materializers._landscape_core helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
import zarr

from cleo.unification.materializers._landscape_core import (
    _build_landscape_dataset,
    _ElevationResult,
    _write_area_catalog_attr,
    _write_wind_propagated_attrs,
)


class TestElevationResult:
    """Tests for _ElevationResult dataclass."""

    def test_local_elevation(self, tmp_path: Path) -> None:
        """Represents local elevation file."""
        data = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
        result = _ElevationResult(
            data=data,
            kind="local",
            meta=None,
            local_path=tmp_path / "elevation.tif",
        )

        assert result.kind == "local"
        assert result.meta is None
        assert result.local_path is not None

    def test_copdem_elevation(self) -> None:
        """Represents CopDEM elevation."""
        data = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
        meta = {"provider": "copdem", "version": "30m", "tile_ids": ["N45E010"]}
        result = _ElevationResult(
            data=data,
            kind="copdem",
            meta=meta,
            local_path=None,
        )

        assert result.kind == "copdem"
        assert result.meta["provider"] == "copdem"
        assert result.local_path is None


class TestBuildLandscapeDataset:
    """Tests for _build_landscape_dataset function."""

    @pytest.fixture
    def mock_wind_ref(self) -> MagicMock:
        """Create mock wind reference."""
        import rioxarray  # noqa: F401

        ref_da = xr.DataArray(
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            dims=("y", "x"),
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        ref_da = ref_da.rio.write_crs("EPSG:32632")

        dataset = xr.Dataset(
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        )

        mock = MagicMock()
        mock.ref_da = ref_da
        mock.dataset = dataset
        mock.crs = "EPSG:32632"
        mock.transform = ref_da.rio.transform()
        return mock

    @pytest.fixture
    def mock_elev_result(self) -> _ElevationResult:
        """Create mock elevation result."""
        data = xr.DataArray(
            np.array([[100.0, 150.0], [200.0, 250.0]]),
            dims=("y", "x"),
            name="elevation",
        )
        return _ElevationResult(data=data, kind="local", meta=None, local_path=Path("/test/elev.tif"))

    def test_creates_valid_mask_from_wind(self, mock_wind_ref: MagicMock, mock_elev_result: _ElevationResult) -> None:
        """Creates valid_mask from wind ref_da notnull."""
        chunk_policy = {"y": 1, "x": 1}
        ds = _build_landscape_dataset(mock_wind_ref, mock_elev_result, chunk_policy)

        assert "valid_mask" in ds.data_vars
        # Check valid_mask matches where ref_da is not NaN
        expected_mask = np.array([[True, False], [True, True]])
        np.testing.assert_array_equal(ds["valid_mask"].values, expected_mask)

    def test_applies_chunk_policy(self, mock_wind_ref: MagicMock, mock_elev_result: _ElevationResult) -> None:
        """Applies chunking from chunk policy."""
        chunk_policy = {"y": 1, "x": 2}
        ds = _build_landscape_dataset(mock_wind_ref, mock_elev_result, chunk_policy)

        # Check that chunking is applied
        assert ds.chunks is not None

    def test_masks_elevation_with_valid_mask(
        self, mock_wind_ref: MagicMock, mock_elev_result: _ElevationResult
    ) -> None:
        """Elevation is NaN where valid_mask is False."""
        chunk_policy = {"y": 2, "x": 2}
        ds = _build_landscape_dataset(mock_wind_ref, mock_elev_result, chunk_policy)

        # Where ref_da is NaN (y=0, x=1), elevation should be NaN
        assert np.isnan(ds["elevation"].values[0, 1])
        # Where ref_da is valid, elevation should have original value
        assert ds["elevation"].values[0, 0] == 100.0


class TestWriteWindPropagatedAttrs:
    """Tests for _write_wind_propagated_attrs function."""

    def test_propagates_valid_attrs(self, tmp_path: Path) -> None:
        """Propagates valid wind attrs to group."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        wind_ds = xr.Dataset(
            attrs={
                "cleo_vertical_policy_checksum": "abc123",
                "wind_speed_grid_len": 100,
                "wind_speed_grid_checksum": "def456",
            }
        )

        _write_wind_propagated_attrs(g, wind_ds)

        assert g.attrs["cleo_vertical_policy_checksum"] == "abc123"
        assert g.attrs["wind_speed_grid_len"] == 100
        assert g.attrs["wind_speed_grid_checksum"] == "def456"

    def test_skips_none_values(self, tmp_path: Path) -> None:
        """Skips attrs with None values."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        wind_ds = xr.Dataset(
            attrs={
                "cleo_vertical_policy_checksum": None,
                "wind_speed_grid_len": 100,
            }
        )

        _write_wind_propagated_attrs(g, wind_ds)

        assert "cleo_vertical_policy_checksum" not in g.attrs
        assert g.attrs["wind_speed_grid_len"] == 100

    def test_skips_empty_strings(self, tmp_path: Path) -> None:
        """Skips attrs with empty string values."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        wind_ds = xr.Dataset(
            attrs={
                "cleo_vertical_policy_checksum": "",
                "wind_speed_grid_len": 100,
            }
        )

        _write_wind_propagated_attrs(g, wind_ds)

        assert "cleo_vertical_policy_checksum" not in g.attrs

    def test_handles_missing_attrs(self, tmp_path: Path) -> None:
        """Handles wind dataset without propagated attrs."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        wind_ds = xr.Dataset(attrs={"other_attr": "value"})

        _write_wind_propagated_attrs(g, wind_ds)

        # Should not raise, and group should be empty of propagated attrs
        assert "cleo_vertical_policy_checksum" not in g.attrs


class TestWriteRegionCatalogAttr:
    """Tests for _write_area_catalog_attr function."""

    def test_writes_region_catalog(self, tmp_path: Path) -> None:
        """Writes area catalog to attrs when available."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        mock_atlas = MagicMock()
        catalog = [{"nuts_id": "AT", "name": "Austria"}]

        with patch(
            "cleo.unification.materializers._landscape_core._read_nuts_area_catalog",
            return_value=catalog,
        ):
            _write_area_catalog_attr(g, mock_atlas)

        assert "cleo_area_catalog_json" in g.attrs
        assert "AT" in g.attrs["cleo_area_catalog_json"]

    def test_handles_missing_catalog(self, tmp_path: Path) -> None:
        """Handles FileNotFoundError gracefully."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        mock_atlas = MagicMock()

        with patch(
            "cleo.unification.materializers._landscape_core._read_nuts_area_catalog",
            side_effect=FileNotFoundError("catalog not found"),
        ):
            _write_area_catalog_attr(g, mock_atlas)

        assert "cleo_area_catalog_json" not in g.attrs

    def test_handles_value_error(self, tmp_path: Path) -> None:
        """Handles ValueError from catalog reading."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        mock_atlas = MagicMock()

        with patch(
            "cleo.unification.materializers._landscape_core._read_nuts_area_catalog",
            side_effect=ValueError("invalid catalog"),
        ):
            _write_area_catalog_attr(g, mock_atlas)

        assert "cleo_area_catalog_json" not in g.attrs

    def test_skips_empty_catalog(self, tmp_path: Path) -> None:
        """Skips writing when catalog is empty."""
        store_path = tmp_path / "test.zarr"
        store_path.mkdir()
        g = zarr.open_group(store_path, mode="w")

        mock_atlas = MagicMock()

        with patch(
            "cleo.unification.materializers._landscape_core._read_nuts_area_catalog",
            return_value=[],
        ):
            _write_area_catalog_attr(g, mock_atlas)

        assert "cleo_area_catalog_json" not in g.attrs
