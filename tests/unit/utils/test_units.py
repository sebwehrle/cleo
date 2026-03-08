"""Tests for cleo/units.py - centralized unit utilities."""

import numpy as np
import pytest
import xarray as xr

from cleo.units import (
    CANONICAL_UNITS,
    assert_convertible,
    conversion_factor,
    convert_dataarray,
    convert_dataset_variable,
    get_canonical_unit,
    get_unit_attr,
    is_known_variable,
    list_canonical_units,
    set_unit_attr,
    validate_unit_attr,
)


class TestGetUnitAttr:
    """Tests for get_unit_attr function."""

    def test_reads_canonical_units_attr(self):
        """Reads 'units' when present."""
        da = xr.DataArray([1, 2, 3])
        da.attrs["units"] = "m/s"

        assert get_unit_attr(da) == "m/s"

    def test_returns_none_when_no_unit_attr(self):
        """Returns None when neither attr present."""
        da = xr.DataArray([1, 2, 3])

        assert get_unit_attr(da) is None

    def test_ignores_legacy_unit_attr_when_units_absent(self):
        """Ignores legacy 'unit' attr now that canonical key is required."""
        da = xr.DataArray([1, 2, 3])
        da.attrs["unit"] = "km"

        assert get_unit_attr(da) is None


class TestSetUnitAttr:
    """Tests for set_unit_attr function."""

    def test_sets_canonical_units_attr(self):
        """Sets 'units' attr."""
        da = xr.DataArray([1, 2, 3])

        result = set_unit_attr(da, "m/s")

        assert result.attrs["units"] == "m/s"

    def test_strips_legacy_unit_attr_if_present(self):
        """Strips stale legacy 'unit' attr if present."""
        da = xr.DataArray([1, 2, 3])
        da.attrs["unit"] = "old_unit"

        result = set_unit_attr(da, "new_unit")

        assert result.attrs["units"] == "new_unit"
        assert "unit" not in result.attrs

    def test_preserves_other_attrs(self):
        """Preserves non-unit attrs."""
        da = xr.DataArray([1, 2, 3])
        da.attrs["foo"] = "bar"
        da.attrs["cleo:algo"] = "test"

        result = set_unit_attr(da, "m")

        assert result.attrs["foo"] == "bar"
        assert result.attrs["cleo:algo"] == "test"
        assert result.attrs["units"] == "m"

    def test_does_not_mutate_input(self):
        """Returns new DataArray, does not mutate input."""
        da = xr.DataArray([1, 2, 3])
        da.attrs["foo"] = "bar"

        result = set_unit_attr(da, "m")

        assert "units" not in da.attrs
        assert result.attrs["units"] == "m"


class TestAssertConvertible:
    """Tests for assert_convertible function."""

    def test_passes_for_compatible_units(self):
        """Passes for dimensionally compatible units."""
        # Length
        assert_convertible("m", "km")
        assert_convertible("m", "cm")
        assert_convertible("m", "ft")

        # Speed
        assert_convertible("m/s", "km/h")
        assert_convertible("m/s", "mph")

        # Energy
        assert_convertible("kWh", "MWh")
        assert_convertible("J", "kWh")

    def test_raises_for_incompatible_units(self):
        """Raises ValueError for incompatible units."""
        with pytest.raises(ValueError, match="Cannot convert"):
            assert_convertible("m", "kg")

        with pytest.raises(ValueError, match="Cannot convert"):
            assert_convertible("m/s", "EUR")

    def test_raises_for_invalid_unit_string(self):
        """Raises ValueError for invalid unit strings."""
        with pytest.raises(ValueError, match="Cannot convert"):
            assert_convertible("not_a_unit", "m")


class TestConversionFactor:
    """Tests for conversion_factor function."""

    def test_correct_factor_length(self):
        """Returns correct factor for length conversion."""
        assert conversion_factor("m", "cm") == 100.0
        assert conversion_factor("km", "m") == 1000.0
        assert conversion_factor("m", "km") == 0.001

    def test_correct_factor_speed(self):
        """Returns correct factor for speed conversion."""
        factor = conversion_factor("m/s", "km/h")
        assert abs(factor - 3.6) < 1e-10

    def test_correct_factor_energy(self):
        """Returns correct factor for energy conversion."""
        factor = conversion_factor("kWh", "MWh")
        assert factor == 0.001

        factor = conversion_factor("MWh", "kWh")
        assert factor == 1000.0

    def test_identity_conversion(self):
        """Returns 1.0 for same unit."""
        assert conversion_factor("m", "m") == 1.0
        assert conversion_factor("kW", "kW") == 1.0

    def test_raises_for_incompatible(self):
        """Raises ValueError for incompatible units."""
        with pytest.raises(ValueError, match="Cannot convert"):
            conversion_factor("m", "s")


class TestConvertDataarray:
    """Tests for convert_dataarray function."""

    def test_basic_conversion(self):
        """Converts values correctly."""
        da = xr.DataArray([1.0, 2.0, 3.0])
        da.attrs["units"] = "m"

        result = convert_dataarray(da, "cm")

        np.testing.assert_array_equal(result.values, [100.0, 200.0, 300.0])
        assert result.attrs["units"] == "cm"

    def test_uses_from_unit_override(self):
        """Uses explicit from_unit when provided."""
        da = xr.DataArray([1.0, 2.0, 3.0])
        da.attrs["units"] = "m"  # Will be ignored

        result = convert_dataarray(da, "m", from_unit="km")

        np.testing.assert_array_equal(result.values, [1000.0, 2000.0, 3000.0])
        assert result.attrs["units"] == "m"

    def test_preserves_other_attrs(self):
        """Preserves non-unit attrs."""
        da = xr.DataArray([1.0, 2.0, 3.0])
        da.attrs["units"] = "m"
        da.attrs["cleo:algo"] = "test"
        da.attrs["foo"] = "bar"

        result = convert_dataarray(da, "cm")

        assert result.attrs["cleo:algo"] == "test"
        assert result.attrs["foo"] == "bar"

    def test_preserves_coords_and_dims(self):
        """Preserves coordinates and dimensions."""
        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=["y", "x"],
            coords={"y": [0, 1], "x": [10, 20]},
        )
        da.attrs["units"] = "m"

        result = convert_dataarray(da, "km")

        assert result.dims == ("y", "x")
        np.testing.assert_array_equal(result.coords["y"].values, [0, 1])
        np.testing.assert_array_equal(result.coords["x"].values, [10, 20])

    def test_preserves_name(self):
        """Preserves DataArray name."""
        da = xr.DataArray([1.0, 2.0], name="my_var")
        da.attrs["units"] = "m"

        result = convert_dataarray(da, "cm")

        assert result.name == "my_var"

    def test_dask_friendly(self):
        """Conversion preserves dask laziness."""
        pytest.importorskip("dask")
        import dask.array as dask_array

        data = dask_array.from_array([1.0, 2.0, 3.0], chunks=2)
        da = xr.DataArray(data)
        da.attrs["units"] = "m"

        result = convert_dataarray(da, "cm")

        # Result should still be dask-backed
        assert hasattr(result.data, "dask")

        # Values correct when computed
        np.testing.assert_array_equal(result.compute().values, [100.0, 200.0, 300.0])

    def test_raises_when_no_unit_source(self):
        """Raises ValueError when no unit source available."""
        da = xr.DataArray([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="No unit attr found"):
            convert_dataarray(da, "cm")

    def test_raises_when_only_legacy_unit_attr_exists(self):
        """Raises ValueError when only legacy 'unit' attr exists."""
        da = xr.DataArray([1.0, 2.0, 3.0])
        da.attrs["unit"] = "m"

        with pytest.raises(ValueError, match="No unit attr found"):
            convert_dataarray(da, "cm")

    def test_raises_for_incompatible_units(self):
        """Raises ValueError for incompatible units."""
        da = xr.DataArray([1.0, 2.0, 3.0])
        da.attrs["units"] = "m"

        with pytest.raises(ValueError, match="Cannot convert"):
            convert_dataarray(da, "kg")


class TestConvertDatasetVariable:
    """Tests for convert_dataset_variable function."""

    def test_converts_single_variable(self):
        """Converts a single variable in dataset."""
        ds = xr.Dataset(
            {
                "distance": xr.DataArray([100.0, 200.0, 300.0]),
                "count": xr.DataArray([1, 2, 3]),
            }
        )
        ds["distance"].attrs["units"] = "m"

        result = convert_dataset_variable(ds, "distance", "km")

        np.testing.assert_array_equal(result["distance"].values, [0.1, 0.2, 0.3])
        assert result["distance"].attrs["units"] == "km"
        # Other variable unchanged
        np.testing.assert_array_equal(result["count"].values, [1, 2, 3])

    def test_does_not_mutate_input(self):
        """Returns new Dataset, does not mutate input."""
        ds = xr.Dataset(
            {
                "distance": xr.DataArray([100.0, 200.0, 300.0]),
            }
        )
        ds["distance"].attrs["units"] = "m"

        result = convert_dataset_variable(ds, "distance", "km")

        # Original unchanged
        np.testing.assert_array_equal(ds["distance"].values, [100.0, 200.0, 300.0])
        assert ds["distance"].attrs["units"] == "m"
        # Result converted
        np.testing.assert_array_equal(result["distance"].values, [0.1, 0.2, 0.3])

    def test_raises_for_missing_variable(self):
        """Raises ValueError for missing variable."""
        ds = xr.Dataset(
            {
                "distance": xr.DataArray([100.0]),
            }
        )

        with pytest.raises(ValueError, match="not found in Dataset"):
            convert_dataset_variable(ds, "nonexistent", "km")


# =============================================================================
# Canonical Unit Registry Tests
# =============================================================================


class TestCanonicalUnitsRegistry:
    """Tests for CANONICAL_UNITS registry."""

    def test_registry_contains_wind_metrics(self):
        """Registry has canonical units for wind metrics."""
        assert CANONICAL_UNITS["mean_wind_speed"] == "m/s"
        assert CANONICAL_UNITS["rotor_equivalent_wind_speed"] == "m/s"
        assert CANONICAL_UNITS["rews_mps"] == "m/s"
        assert CANONICAL_UNITS["capacity_factors"] == "1"

    def test_registry_contains_economics_metrics(self):
        """Registry has canonical units for economics metrics."""
        assert CANONICAL_UNITS["lcoe"] == "EUR/MWh"
        assert CANONICAL_UNITS["min_lcoe_turbine"] is None  # index
        assert CANONICAL_UNITS["optimal_power"] == "kW"
        assert CANONICAL_UNITS["optimal_energy"] == "GWh/a"

    def test_registry_contains_turbine_metadata(self):
        """Registry has canonical units for turbine metadata."""
        assert CANONICAL_UNITS["turbine_capacity"] == "kW"
        assert CANONICAL_UNITS["turbine_hub_height"] == "m"
        assert CANONICAL_UNITS["turbine_rotor_diameter"] == "m"

    def test_registry_contains_weibull_params(self):
        """Registry has canonical units for Weibull parameters."""
        assert CANONICAL_UNITS["weibull_A"] == "m/s"
        assert CANONICAL_UNITS["weibull_k"] == "1"
        assert CANONICAL_UNITS["rho"] == "kg/m**3"


class TestGetCanonicalUnit:
    """Tests for get_canonical_unit function."""

    def test_returns_unit_for_known_variable(self):
        """Returns canonical unit for known variable."""
        assert get_canonical_unit("lcoe") == "EUR/MWh"
        assert get_canonical_unit("mean_wind_speed") == "m/s"
        assert get_canonical_unit("rotor_equivalent_wind_speed") == "m/s"

    def test_returns_none_for_dimensionless(self):
        """Returns None for dimensionless variables."""
        assert get_canonical_unit("min_lcoe_turbine") is None

    def test_returns_none_for_unknown_variable(self):
        """Returns None for unknown variable."""
        assert get_canonical_unit("unknown_variable") is None

    def test_pattern_matches_distance_prefix(self):
        """Matches distance_* pattern."""
        assert get_canonical_unit("distance_roads") == "m"
        assert get_canonical_unit("distance_water") == "m"
        assert get_canonical_unit("distance_settlements") == "m"


class TestIsKnownVariable:
    """Tests for is_known_variable function."""

    def test_returns_true_for_known(self):
        """Returns True for known variables."""
        assert is_known_variable("lcoe") is True
        assert is_known_variable("capacity_factors") is True
        assert is_known_variable("min_lcoe_turbine") is True  # Even if None unit

    def test_returns_true_for_distance_pattern(self):
        """Returns True for distance_* pattern."""
        assert is_known_variable("distance_roads") is True
        assert is_known_variable("distance_anything") is True

    def test_returns_false_for_unknown(self):
        """Returns False for unknown variables."""
        assert is_known_variable("unknown_variable") is False
        assert is_known_variable("my_custom_metric") is False


class TestValidateUnitAttr:
    """Tests for validate_unit_attr function."""

    def test_passes_for_correct_unit(self):
        """Passes when unit matches canonical."""
        da = xr.DataArray([1.0, 2.0])
        da.attrs["units"] = "EUR/MWh"

        # Should not raise
        validate_unit_attr(da, "lcoe")

    def test_passes_for_dimensionless(self):
        """Passes for dimensionless variables regardless of unit attr."""
        da = xr.DataArray([0, 1, 2])
        # No unit attr, but min_lcoe_turbine has None canonical unit

        # Should not raise
        validate_unit_attr(da, "min_lcoe_turbine")

    def test_passes_for_unknown_variable_non_strict(self):
        """Passes for unknown variable when strict=False (default)."""
        da = xr.DataArray([1.0, 2.0])
        # No unit attr, unknown variable

        # Should not raise in non-strict mode
        validate_unit_attr(da, "unknown_variable")

    def test_raises_for_missing_unit(self):
        """Raises when canonical unit exists but attr missing."""
        da = xr.DataArray([1.0, 2.0])
        # No unit attr, but lcoe requires EUR/MWh

        with pytest.raises(ValueError, match="missing required 'units' attr"):
            validate_unit_attr(da, "lcoe")

    def test_raises_for_wrong_unit(self):
        """Raises when unit doesn't match canonical."""
        da = xr.DataArray([1.0, 2.0])
        da.attrs["units"] = "EUR/kWh"  # Wrong unit

        with pytest.raises(ValueError, match="non-canonical unit"):
            validate_unit_attr(da, "lcoe")

    def test_raises_for_unknown_variable_strict(self):
        """Raises for unknown variable when strict=True."""
        da = xr.DataArray([1.0, 2.0])

        with pytest.raises(ValueError, match="no canonical unit definition"):
            validate_unit_attr(da, "unknown_variable", strict=True)

    def test_validates_distance_pattern(self):
        """Validates distance_* pattern variables."""
        da = xr.DataArray([100.0, 200.0])
        da.attrs["units"] = "m"

        # Should not raise
        validate_unit_attr(da, "distance_roads")

        # Wrong unit should raise
        da2 = xr.DataArray([100.0, 200.0])
        da2.attrs["units"] = "km"

        with pytest.raises(ValueError, match="non-canonical unit"):
            validate_unit_attr(da2, "distance_roads")


class TestListCanonicalUnits:
    """Tests for list_canonical_units function."""

    def test_returns_copy(self):
        """Returns a copy, not the original dict."""
        result = list_canonical_units()

        # Modifying result should not affect original
        result["test_var"] = "test_unit"
        assert "test_var" not in CANONICAL_UNITS

    def test_contains_all_entries(self):
        """Contains all entries from CANONICAL_UNITS."""
        result = list_canonical_units()

        assert len(result) == len(CANONICAL_UNITS)
        for key, value in CANONICAL_UNITS.items():
            assert result[key] == value
