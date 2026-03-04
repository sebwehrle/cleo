"""Tests for cleo.unification.materializers.wind helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from cleo.unification.materializers.wind import (
    _compute_encoding,
    _GWALoadResult,
    _stack_by_height,
    _turbine_config_items,
)


class TestStackByHeight:
    """Tests for _stack_by_height function."""

    def test_stacks_arrays_by_height(self) -> None:
        """Stacks multiple DataArrays by height coordinate."""
        da_50 = xr.DataArray(np.ones((3, 3)) * 1.0, dims=("y", "x"))
        da_100 = xr.DataArray(np.ones((3, 3)) * 2.0, dims=("y", "x"))
        da_150 = xr.DataArray(np.ones((3, 3)) * 3.0, dims=("y", "x"))

        arrays_list = [(100, da_100), (50, da_50), (150, da_150)]  # Unsorted
        result = _stack_by_height(arrays_list, "weibull_A")

        assert result.name == "weibull_A"
        assert "height" in result.dims
        assert list(result.coords["height"].values) == [50, 100, 150]  # Sorted
        assert result.sel(height=50).values.flatten()[0] == 1.0
        assert result.sel(height=100).values.flatten()[0] == 2.0
        assert result.sel(height=150).values.flatten()[0] == 3.0

    def test_single_height(self) -> None:
        """Handles single height array."""
        da = xr.DataArray(np.ones((2, 2)) * 5.0, dims=("y", "x"))
        result = _stack_by_height([(100, da)], "test_var")

        assert result.name == "test_var"
        assert list(result.coords["height"].values) == [100]


class TestComputeEncoding:
    """Tests for _compute_encoding function."""

    def test_encoding_for_spatial_vars(self) -> None:
        """Creates encoding dict with chunk sizes for spatial variables."""
        ds = xr.Dataset(
            {
                "var_yx": xr.DataArray(np.ones((100, 100)), dims=("y", "x")),
                "var_hyx": xr.DataArray(np.ones((3, 100, 100)), dims=("height", "y", "x")),
            }
        )
        chunk_policy = {"y": 32, "x": 64}

        encoding = _compute_encoding(ds, chunk_policy)

        assert "var_yx" in encoding
        assert encoding["var_yx"]["chunks"] == (32, 64)
        assert "var_hyx" in encoding
        assert encoding["var_hyx"]["chunks"] == (3, 32, 64)  # height dim uses full size

    def test_skips_non_spatial_vars(self) -> None:
        """Skips variables without y or x dimensions."""
        ds = xr.Dataset(
            {
                "spatial_var": xr.DataArray(np.ones((10, 10)), dims=("y", "x")),
                "non_spatial": xr.DataArray(np.ones((5,)), dims=("time",)),
            }
        )
        chunk_policy = {"y": 5, "x": 5}

        encoding = _compute_encoding(ds, chunk_policy)

        assert "spatial_var" in encoding
        assert "non_spatial" not in encoding

    def test_uses_var_size_for_missing_dims(self) -> None:
        """Uses variable size when dim not in chunk policy."""
        ds = xr.Dataset(
            {
                "var": xr.DataArray(np.ones((7, 50, 50)), dims=("extra", "y", "x")),
            }
        )
        chunk_policy = {"y": 25, "x": 25}

        encoding = _compute_encoding(ds, chunk_policy)

        # extra dim should use its full size (7)
        assert encoding["var"]["chunks"] == (7, 25, 25)


class TestGWALoadResult:
    """Tests for _GWALoadResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Dataclass has expected fields."""
        ref_da = xr.DataArray(np.ones((2, 2)))
        weibull_A = xr.DataArray(np.ones((3, 2, 2)))
        weibull_k = xr.DataArray(np.ones((3, 2, 2)))
        rho = xr.DataArray(np.ones((3, 2, 2)))

        result = _GWALoadResult(
            ref_da=ref_da,
            weibull_A=weibull_A,
            weibull_k=weibull_k,
            rho=rho,
            sources=[{"source_id": "test"}],
            weibull_A_source_ids=["gwa:file:weibull_A:100"],
            weibull_k_source_ids=["gwa:file:weibull_k:100"],
            rho_source_ids=["gwa:file:rho:100"],
        )

        assert result.ref_da is ref_da
        assert result.weibull_A is weibull_A
        assert len(result.sources) == 1

    def test_default_source_ids_empty(self) -> None:
        """Source ID lists default to empty."""
        result = _GWALoadResult(
            ref_da=xr.DataArray(np.ones((2, 2))),
            weibull_A=xr.DataArray(np.ones((2, 2))),
            weibull_k=xr.DataArray(np.ones((2, 2))),
            rho=xr.DataArray(np.ones((2, 2))),
            sources=[],
        )

        assert result.weibull_A_source_ids == []
        assert result.weibull_k_source_ids == []
        assert result.rho_source_ids == []


class TestTurbineConfigItems:
    """Tests for _turbine_config_items function."""

    def test_with_configured_turbines(self, tmp_path: Path) -> None:
        """Returns items for configured turbines."""
        mock_atlas = MagicMock()
        mock_atlas.turbines_configured = ["Turbine.A", "Turbine.B"]
        mock_atlas.path = tmp_path
        (tmp_path / "resources").mkdir()

        items = _turbine_config_items(mock_atlas)

        item_dict = dict(items)
        assert "turbines_configured" in item_dict
        assert "Turbine.A" in item_dict["turbines_configured"]
        assert "turbines_effective" in item_dict

    def test_with_default_turbines(self, tmp_path: Path) -> None:
        """Uses default turbines when not configured."""
        mock_atlas = MagicMock()
        mock_atlas.turbines_configured = None
        mock_atlas.path = tmp_path
        (tmp_path / "resources").mkdir()

        with patch(
            "cleo.unification.materializers.wind._default_turbines_from_resources",
            return_value=["Default.Turbine"],
        ):
            items = _turbine_config_items(mock_atlas)

        item_dict = dict(items)
        assert item_dict["turbines_configured"] == "default"
        assert "Default.Turbine" in item_dict["turbines_effective"]

    def test_hashes_turbine_yaml_files(self, tmp_path: Path) -> None:
        """Includes SHA256 of turbine YAML files."""
        mock_atlas = MagicMock()
        mock_atlas.turbines_configured = ["Test.Turbine"]
        mock_atlas.path = tmp_path
        resources = tmp_path / "resources"
        resources.mkdir()
        (resources / "Test.Turbine.yml").write_text("turbine: data")

        items = _turbine_config_items(mock_atlas)

        item_dict = dict(items)
        assert "turbines_sha256" in item_dict
        assert "Test.Turbine" in item_dict["turbines_sha256"]
