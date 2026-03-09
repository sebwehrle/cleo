"""Unit tests for analysis export dataset construction.

Tests the `build_analysis_export_dataset` function and related helpers
in `cleo.exports`.
"""

import numpy as np
import pytest
import xarray as xr

from cleo.exports import (
    build_analysis_export_dataset,
    _build_export_spec,
    _extract_upstream_provenance,
    _filter_variables,
    _prefix_variables,
)


class TestBuildExportSpec:
    """Tests for _build_export_spec helper."""

    def test_basic_spec(self) -> None:
        """Basic spec construction."""
        spec = _build_export_spec(
            domain="both",
            include_only=None,
            prefix=True,
            exclude_template=True,
        )
        assert spec["domain"] == "both"
        assert spec["include_only"] is None
        assert spec["prefix"] is True
        assert spec["exclude_template"] is True

    def test_include_only_converted_to_list(self) -> None:
        """include_only sequence is converted to list."""
        spec = _build_export_spec(
            domain="wind",
            include_only=("var1", "var2"),
            prefix=False,
            exclude_template=False,
        )
        assert spec["include_only"] == ["var1", "var2"]
        assert isinstance(spec["include_only"], list)


class TestExtractUpstreamProvenance:
    """Tests for _extract_upstream_provenance helper."""

    def test_extracts_wind_attrs(self) -> None:
        """Extracts provenance attrs from wind dataset."""
        wind_ds = xr.Dataset(
            attrs={
                "grid_id": "wind_grid_123",
                "inputs_id": "wind_inputs_456",
                "unify_version": "1.0",
                "created_at": "2026-01-01T00:00:00Z",
                "other_attr": "ignored",
            }
        )
        provenance = _extract_upstream_provenance(wind_ds, None)

        assert "wind_store" in provenance
        assert provenance["wind_store"]["grid_id"] == "wind_grid_123"
        assert provenance["wind_store"]["inputs_id"] == "wind_inputs_456"
        assert "other_attr" not in provenance["wind_store"]

    def test_extracts_landscape_attrs(self) -> None:
        """Extracts provenance attrs from landscape dataset."""
        land_ds = xr.Dataset(
            attrs={
                "grid_id": "land_grid_789",
                "inputs_id": "land_inputs_012",
            }
        )
        provenance = _extract_upstream_provenance(None, land_ds)

        assert "landscape_store" in provenance
        assert provenance["landscape_store"]["grid_id"] == "land_grid_789"

    def test_extracts_both(self) -> None:
        """Extracts provenance from both datasets."""
        wind_ds = xr.Dataset(attrs={"grid_id": "wind_grid"})
        land_ds = xr.Dataset(attrs={"grid_id": "land_grid"})
        provenance = _extract_upstream_provenance(wind_ds, land_ds)

        assert "wind_store" in provenance
        assert "landscape_store" in provenance

    def test_empty_when_no_attrs(self) -> None:
        """Returns empty dict when no relevant attrs."""
        wind_ds = xr.Dataset(attrs={"irrelevant": "value"})
        provenance = _extract_upstream_provenance(wind_ds, None)

        assert provenance == {}


class TestFilterVariables:
    """Tests for _filter_variables helper."""

    def _make_ds(self) -> xr.Dataset:
        """Create a test dataset with multiple variables."""
        y = np.arange(3)
        x = np.arange(4)
        return xr.Dataset(
            {
                "var1": (["y", "x"], np.ones((3, 4))),
                "var2": (["y", "x"], np.ones((3, 4)) * 2),
                "template": (["y", "x"], np.ones((3, 4)) * 3),
            },
            coords={"y": y, "x": x},
        )

    def test_exclude_template_default(self) -> None:
        """Template is excluded by default."""
        ds = self._make_ds()
        filtered = _filter_variables(ds, include_only=None, exclude_template=True)

        assert "var1" in filtered.data_vars
        assert "var2" in filtered.data_vars
        assert "template" not in filtered.data_vars

    def test_include_template(self) -> None:
        """Template is included when exclude_template=False."""
        ds = self._make_ds()
        filtered = _filter_variables(ds, include_only=None, exclude_template=False)

        assert "template" in filtered.data_vars

    def test_include_only_filters(self) -> None:
        """include_only filters to specified variables."""
        ds = self._make_ds()
        filtered = _filter_variables(ds, include_only=["var1"], exclude_template=True)

        assert list(filtered.data_vars) == ["var1"]

    def test_include_only_unknown_variable_raises(self) -> None:
        """Unknown variable in include_only raises ValueError."""
        ds = self._make_ds()

        with pytest.raises(ValueError, match="include_only contains unknown variables.*unknown_var"):
            _filter_variables(ds, include_only=["var1", "unknown_var"], exclude_template=True)

    def test_empty_result_raises(self) -> None:
        """Raises when no variables remain after filtering."""
        ds = xr.Dataset(
            {"template": (["x"], np.arange(3))},
            coords={"x": np.arange(3)},
        )

        with pytest.raises(ValueError, match="No variables to export"):
            _filter_variables(ds, include_only=None, exclude_template=True)


class TestPrefixVariables:
    """Tests for _prefix_variables helper."""

    def test_prefixes_all_variables(self) -> None:
        """All variables are prefixed."""
        ds = xr.Dataset(
            {
                "var1": (["x"], np.arange(3)),
                "var2": (["x"], np.arange(3)),
            }
        )
        prefixed = _prefix_variables(ds, "wind__")

        assert "wind__var1" in prefixed.data_vars
        assert "wind__var2" in prefixed.data_vars
        assert "var1" not in prefixed.data_vars


class TestBuildAnalysisExportDataset:
    """Tests for build_analysis_export_dataset function."""

    def _make_wind_ds(self) -> xr.Dataset:
        """Create a minimal wind-like dataset."""
        y = np.arange(3)
        x = np.arange(4)
        return xr.Dataset(
            {
                "capacity_factors": (["y", "x"], np.ones((3, 4)) * 0.3),
                "mean_wind_speed": (["y", "x"], np.ones((3, 4)) * 7.0),
                "template": (["y", "x"], np.ones((3, 4))),
            },
            coords={"y": y, "x": x},
            attrs={"store_state": "complete"},
        )

    def _make_landscape_ds(self) -> xr.Dataset:
        """Create a minimal landscape-like dataset."""
        y = np.arange(3)
        x = np.arange(4)
        return xr.Dataset(
            {
                "valid_mask": (["y", "x"], np.ones((3, 4), dtype=bool)),
                "elevation": (["y", "x"], np.ones((3, 4)) * 500),
                "template": (["y", "x"], np.ones((3, 4))),
            },
            coords={"y": y, "x": x},
            attrs={"store_state": "complete"},
        )

    def test_domain_wind(self) -> None:
        """domain='wind' exports only wind variables."""
        wind_ds = self._make_wind_ds()

        result = build_analysis_export_dataset(
            wind_ds,
            None,
            domain="wind",
            include_only=None,
            prefix=True,
            exclude_template=True,
        )

        assert "capacity_factors" in result.data_vars
        assert "mean_wind_speed" in result.data_vars
        assert "template" not in result.data_vars
        assert "valid_mask" not in result.data_vars

    def test_domain_landscape(self) -> None:
        """domain='landscape' exports only landscape variables."""
        land_ds = self._make_landscape_ds()

        result = build_analysis_export_dataset(
            None,
            land_ds,
            domain="landscape",
            include_only=None,
            prefix=True,
            exclude_template=True,
        )

        assert "valid_mask" in result.data_vars
        assert "elevation" in result.data_vars
        assert "capacity_factors" not in result.data_vars

    def test_domain_both_with_prefix(self) -> None:
        """domain='both' with prefix=True adds domain prefixes."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds()

        result = build_analysis_export_dataset(
            wind_ds,
            land_ds,
            domain="both",
            include_only=None,
            prefix=True,
            exclude_template=True,
        )

        assert "wind__capacity_factors" in result.data_vars
        assert "wind__mean_wind_speed" in result.data_vars
        assert "landscape__valid_mask" in result.data_vars
        assert "landscape__elevation" in result.data_vars
        # No unprefixed names
        assert "capacity_factors" not in result.data_vars
        assert "valid_mask" not in result.data_vars

    def test_domain_both_without_prefix_no_collision(self) -> None:
        """domain='both' with prefix=False works when no collisions."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds()

        result = build_analysis_export_dataset(
            wind_ds,
            land_ds,
            domain="both",
            include_only=None,
            prefix=False,
            exclude_template=True,
        )

        assert "capacity_factors" in result.data_vars
        assert "valid_mask" in result.data_vars

    def test_domain_both_without_prefix_collision_raises(self) -> None:
        """domain='both' with prefix=False raises on collision."""
        wind_ds = xr.Dataset(
            {"shared_var": (["x"], np.arange(3))},
            coords={"x": np.arange(3)},
        )
        land_ds = xr.Dataset(
            {"shared_var": (["x"], np.arange(3))},
            coords={"x": np.arange(3)},
        )

        with pytest.raises(ValueError, match="Variable name collision.*shared_var"):
            build_analysis_export_dataset(
                wind_ds,
                land_ds,
                domain="both",
                include_only=None,
                prefix=False,
                exclude_template=True,
            )

    def test_include_only_with_prefixed_names(self) -> None:
        """include_only with prefixed names for domain='both'."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds()

        result = build_analysis_export_dataset(
            wind_ds,
            land_ds,
            domain="both",
            include_only=["wind__capacity_factors", "landscape__valid_mask"],
            prefix=True,
            exclude_template=True,
        )

        assert set(result.data_vars) == {"wind__capacity_factors", "landscape__valid_mask"}

    def test_include_only_with_single_domain_prefixed_name_keeps_other_domain_empty(self) -> None:
        """One-sided prefixed include_only keeps the non-selected domain empty."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds()

        result = build_analysis_export_dataset(
            wind_ds,
            land_ds,
            domain="both",
            include_only=["wind__capacity_factors"],
            prefix=True,
            exclude_template=True,
        )

        assert set(result.data_vars) == {"wind__capacity_factors"}
        np.testing.assert_array_equal(result.coords["x"].values, wind_ds.coords["x"].values)
        np.testing.assert_array_equal(result.coords["y"].values, wind_ds.coords["y"].values)

    def test_single_domain_prefixed_selection_skips_non_exported_grid_mismatch(self) -> None:
        """One-sided prefixed exports do not require the other domain grid to match."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds().assign_coords(x=np.array([10, 11, 12, 13]))

        result = build_analysis_export_dataset(
            wind_ds,
            land_ds,
            domain="both",
            include_only=["wind__capacity_factors"],
            prefix=True,
            exclude_template=True,
        )

        assert set(result.data_vars) == {"wind__capacity_factors"}

    def test_include_only_empty_for_both_raises(self) -> None:
        """An empty combined include_only selection must fail."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds()

        with pytest.raises(ValueError, match="No variables to export"):
            build_analysis_export_dataset(
                wind_ds,
                land_ds,
                domain="both",
                include_only=[],
                prefix=True,
                exclude_template=True,
            )

    def test_include_only_unprefixed_name_raises_for_both(self) -> None:
        """include_only with unprefixed name raises for domain='both' with prefix=True."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds()

        with pytest.raises(ValueError, match="does not have expected prefix"):
            build_analysis_export_dataset(
                wind_ds,
                land_ds,
                domain="both",
                include_only=["capacity_factors"],  # Missing wind__ prefix
                prefix=True,
                exclude_template=True,
            )

    def test_missing_wind_ds_raises(self) -> None:
        """Missing wind_ds for domain='wind' raises."""
        with pytest.raises(ValueError, match="wind_ds required"):
            build_analysis_export_dataset(
                None,
                None,
                domain="wind",
            )

    def test_missing_landscape_ds_raises(self) -> None:
        """Missing landscape_ds for domain='landscape' raises."""
        with pytest.raises(ValueError, match="landscape_ds required"):
            build_analysis_export_dataset(
                None,
                None,
                domain="landscape",
            )

    def test_missing_both_ds_raises(self) -> None:
        """Missing both datasets for domain='both' raises."""
        wind_ds = self._make_wind_ds()

        with pytest.raises(ValueError, match="landscape_ds required"):
            build_analysis_export_dataset(
                wind_ds,
                None,
                domain="both",
            )

    def test_invalid_domain_raises(self) -> None:
        """Invalid domain value raises."""
        with pytest.raises(ValueError, match="Invalid domain"):
            build_analysis_export_dataset(
                None,
                None,
                domain="invalid",
            )

    def test_preserves_coords(self) -> None:
        """Export preserves coordinate information."""
        wind_ds = self._make_wind_ds()

        result = build_analysis_export_dataset(
            wind_ds,
            None,
            domain="wind",
        )

        assert "y" in result.coords
        assert "x" in result.coords

    def test_domain_both_rejects_mismatched_x_coords(self) -> None:
        """Combined export requires exact x-coordinate equality."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds().assign_coords(x=np.array([10, 11, 12, 13]))

        with pytest.raises(ValueError, match="identical 'x' coordinates"):
            build_analysis_export_dataset(
                wind_ds,
                land_ds,
                domain="both",
                prefix=True,
                exclude_template=True,
            )

    def test_domain_both_rejects_mismatched_y_coords(self) -> None:
        """Combined export requires exact y-coordinate equality."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds().assign_coords(y=np.array([10, 11, 12]))

        with pytest.raises(ValueError, match="identical 'y' coordinates"):
            build_analysis_export_dataset(
                wind_ds,
                land_ds,
                domain="both",
                prefix=True,
                exclude_template=True,
            )

    def test_domain_both_rejects_reversed_coordinate_order(self) -> None:
        """Combined export rejects reversed coordinate order even with same values."""
        wind_ds = self._make_wind_ds()
        land_ds = self._make_landscape_ds().assign_coords(x=np.array([3, 2, 1, 0]))

        with pytest.raises(ValueError, match="identical 'x' coordinates"):
            build_analysis_export_dataset(
                wind_ds,
                land_ds,
                domain="both",
                prefix=True,
                exclude_template=True,
            )


class TestExportDatasetValidation:
    """Tests for export dataset validation compatibility."""

    def test_export_dataset_has_required_dims(self) -> None:
        """Export dataset has required y/x dimensions for validation."""
        y = np.arange(3)
        x = np.arange(4)
        wind_ds = xr.Dataset(
            {"var1": (["y", "x"], np.ones((3, 4)))},
            coords={"y": y, "x": x},
        )

        result = build_analysis_export_dataset(
            wind_ds,
            None,
            domain="wind",
        )

        assert "y" in result.sizes
        assert "x" in result.sizes
