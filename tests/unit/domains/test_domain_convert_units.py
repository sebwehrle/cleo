"""Tests for domain-level convert_units API.

Tests WindDomain.convert_units() and LandscapeDomain.convert_units() methods.

Note: These tests use physical units that pint can handle. Currency units
(EUR/MWh, etc.) are stored as string labels but cannot be converted by pint.
"""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock

from cleo.domains import WindDomain, LandscapeDomain


class TestWindDomainConvertUnits:
    """Tests for WindDomain.convert_units()."""

    def _make_mock_wind_domain(self, data_vars: dict[str, xr.DataArray]) -> WindDomain:
        """Create a WindDomain with mock atlas and data."""
        mock_atlas = MagicMock()
        mock_atlas._wind_selected_turbines = None

        domain = WindDomain(mock_atlas)
        # Mock the _store_data method to return our test dataset
        ds = xr.Dataset(data_vars)
        ds.attrs["store_state"] = "complete"
        domain._data = ds
        return domain

    def test_convert_units_returns_converted_dataarray(self):
        """convert_units returns converted DataArray when inplace=False."""
        # Use physical units that pint can handle
        wind_speed = xr.DataArray(
            np.array([[10.0, 20.0], [15.0, 25.0]]),
            dims=["y", "x"],
            name="mean_wind_speed",
        )
        wind_speed.attrs["units"] = "m/s"

        domain = self._make_mock_wind_domain({"mean_wind_speed": wind_speed})

        result = domain.convert_units("mean_wind_speed", "km/h")

        assert result is not None
        assert result.attrs["units"] == "km/h"
        # 10 m/s = 36 km/h
        np.testing.assert_allclose(result.values[0, 0], 36.0)

    def test_convert_units_inplace_stages_overlay(self):
        """convert_units with inplace=True stages the result."""
        wind_speed = xr.DataArray(
            np.array([[10.0, 20.0]]),
            dims=["y", "x"],
            name="mean_wind_speed",
        )
        wind_speed.attrs["units"] = "m/s"

        domain = self._make_mock_wind_domain({"mean_wind_speed": wind_speed})

        result = domain.convert_units("mean_wind_speed", "km/h", inplace=True)

        assert result is None
        assert "mean_wind_speed" in domain._computed_overlays
        assert domain._computed_overlays["mean_wind_speed"].attrs["units"] == "km/h"

    def test_convert_units_inplace_visible_in_data(self):
        """convert_units with inplace=True makes result visible in .data."""
        wind_speed = xr.DataArray(
            np.array([[10.0]]),
            dims=["y", "x"],
            name="mean_wind_speed",
        )
        wind_speed.attrs["units"] = "m/s"

        domain = self._make_mock_wind_domain({"mean_wind_speed": wind_speed})

        domain.convert_units("mean_wind_speed", "km/h", inplace=True)

        # The converted value should be visible through .data
        assert domain.data["mean_wind_speed"].attrs["units"] == "km/h"
        np.testing.assert_allclose(domain.data["mean_wind_speed"].values[0, 0], 36.0)

    def test_convert_units_with_from_unit_override(self):
        """convert_units uses explicit from_unit when provided."""
        # Variable has units in m, but we override to treat it as km
        distance = xr.DataArray(
            np.array([[1.0, 2.0]]),
            dims=["y", "x"],
            name="distance",
        )
        distance.attrs["units"] = "m"  # Will be ignored

        domain = self._make_mock_wind_domain({"distance": distance})

        result = domain.convert_units("distance", "m", from_unit="km")

        # 1 km = 1000 m
        np.testing.assert_allclose(result.values[0, 0], 1000.0)
        assert result.attrs["units"] == "m"

    def test_convert_units_raises_for_unknown_variable(self):
        """convert_units raises ValueError for unknown variable."""
        domain = self._make_mock_wind_domain({})

        with pytest.raises(ValueError, match="not found in wind data"):
            domain.convert_units("nonexistent", "m")

    def test_convert_units_raises_for_incompatible_units(self):
        """convert_units raises ValueError for incompatible units."""
        wind_speed = xr.DataArray(np.array([[10.0]]), dims=["y", "x"], name="wind_speed")
        wind_speed.attrs["units"] = "m/s"

        domain = self._make_mock_wind_domain({"wind_speed": wind_speed})

        with pytest.raises(ValueError, match="Cannot convert"):
            domain.convert_units("wind_speed", "kg")

    def test_convert_units_raises_when_no_unit_source(self):
        """convert_units raises when variable has no units attr and from_unit not given."""
        var = xr.DataArray(np.array([[1.0]]), dims=["y", "x"], name="var")
        # No units attr

        domain = self._make_mock_wind_domain({"var": var})

        with pytest.raises(ValueError, match="No unit attr found"):
            domain.convert_units("var", "m")

    def test_convert_units_preserves_dask_laziness(self):
        """convert_units preserves dask laziness."""
        pytest.importorskip("dask")
        import dask.array as dask_array

        data = dask_array.from_array(np.array([[10.0, 20.0]]), chunks=1)
        wind_speed = xr.DataArray(data, dims=["y", "x"], name="mean_wind_speed")
        wind_speed.attrs["units"] = "m/s"

        domain = self._make_mock_wind_domain({"mean_wind_speed": wind_speed})

        result = domain.convert_units("mean_wind_speed", "km/h")

        # Result should still be dask-backed
        assert hasattr(result.data, "dask")

    def test_convert_power_kw_to_mw(self):
        """convert_units handles power unit conversion."""
        power = xr.DataArray(
            np.array([[2000.0, 3000.0]]),
            dims=["y", "x"],
            name="optimal_power",
        )
        power.attrs["units"] = "kW"

        domain = self._make_mock_wind_domain({"optimal_power": power})

        result = domain.convert_units("optimal_power", "MW")

        assert result.attrs["units"] == "MW"
        # 2000 kW = 2 MW
        np.testing.assert_allclose(result.values[0, 0], 2.0)


class TestLandscapeDomainConvertUnits:
    """Tests for LandscapeDomain.convert_units()."""

    def _make_mock_landscape_domain(self, data_vars: dict[str, xr.DataArray]) -> LandscapeDomain:
        """Create a LandscapeDomain with mock atlas and data."""
        mock_atlas = MagicMock()

        domain = LandscapeDomain(mock_atlas)
        # Create dataset with valid_mask (required by LandscapeDomain)
        valid_mask = xr.DataArray(
            np.array([[True, True], [True, True]]),
            dims=["y", "x"],
            name="valid_mask",
        )
        all_vars = {"valid_mask": valid_mask, **data_vars}
        ds = xr.Dataset(all_vars)
        ds.attrs["store_state"] = "complete"
        domain._data = ds
        return domain

    def test_convert_units_returns_converted_dataarray(self):
        """convert_units returns converted DataArray when inplace=False."""
        distance = xr.DataArray(
            np.array([[1000.0, 2000.0], [1500.0, 2500.0]]),
            dims=["y", "x"],
            name="distance_roads",
        )
        distance.attrs["units"] = "m"

        domain = self._make_mock_landscape_domain({"distance_roads": distance})

        result = domain.convert_units("distance_roads", "km")

        assert result is not None
        assert result.attrs["units"] == "km"
        # 1000 m = 1 km
        np.testing.assert_allclose(result.values[0, 0], 1.0)

    def test_convert_units_inplace_stages_overlay(self):
        """convert_units with inplace=True stages the result."""
        distance = xr.DataArray(
            np.array([[1000.0, 2000.0], [1500.0, 2500.0]]),
            dims=["y", "x"],
            name="distance_roads",
        )
        distance.attrs["units"] = "m"

        domain = self._make_mock_landscape_domain({"distance_roads": distance})

        result = domain.convert_units("distance_roads", "km", inplace=True)

        assert result is None
        assert "distance_roads" in domain._staged_overlays
        assert domain._staged_overlays["distance_roads"].attrs["units"] == "km"

    def test_convert_units_inplace_visible_in_data(self):
        """convert_units with inplace=True makes result visible in .data."""
        distance = xr.DataArray(
            np.array([[1000.0, 2000.0], [1500.0, 2500.0]]),
            dims=["y", "x"],
            name="distance_roads",
        )
        distance.attrs["units"] = "m"

        domain = self._make_mock_landscape_domain({"distance_roads": distance})

        domain.convert_units("distance_roads", "km", inplace=True)

        # The converted value should be visible through .data
        assert domain.data["distance_roads"].attrs["units"] == "km"
        np.testing.assert_allclose(domain.data["distance_roads"].values[0, 0], 1.0)

    def test_convert_units_raises_for_unknown_variable(self):
        """convert_units raises ValueError for unknown variable."""
        domain = self._make_mock_landscape_domain({})

        with pytest.raises(ValueError, match="not found in landscape data"):
            domain.convert_units("nonexistent", "m")

    def test_convert_units_raises_for_incompatible_units(self):
        """convert_units raises ValueError for incompatible units."""
        distance = xr.DataArray(np.array([[1000.0, 2000.0], [1500.0, 2500.0]]), dims=["y", "x"])
        distance.attrs["units"] = "m"

        domain = self._make_mock_landscape_domain({"distance": distance})

        with pytest.raises(ValueError, match="Cannot convert"):
            domain.convert_units("distance", "kg")

    def test_convert_elevation_m_to_ft(self):
        """convert_units handles elevation conversion to feet."""
        elevation = xr.DataArray(
            np.array([[100.0, 200.0], [150.0, 250.0]]),
            dims=["y", "x"],
            name="elevation",
        )
        elevation.attrs["units"] = "m"

        domain = self._make_mock_landscape_domain({"elevation": elevation})

        result = domain.convert_units("elevation", "ft")

        assert result.attrs["units"] == "ft"
        # 100 m ~ 328.084 ft
        np.testing.assert_allclose(result.values[0, 0], 328.084, rtol=1e-3)


class TestConvertUnitsErrorMessages:
    """Tests for helpful error messages in convert_units."""

    def _make_mock_wind_domain(self, data_vars: dict[str, xr.DataArray]) -> WindDomain:
        mock_atlas = MagicMock()
        mock_atlas._wind_selected_turbines = None
        domain = WindDomain(mock_atlas)
        ds = xr.Dataset(data_vars)
        ds.attrs["store_state"] = "complete"
        domain._data = ds
        return domain

    def test_error_message_shows_available_variables(self):
        """Error message includes available variables when variable not found."""
        wind_speed = xr.DataArray(np.array([[10.0]]), dims=["y", "x"], name="mean_wind_speed")
        wind_speed.attrs["units"] = "m/s"

        domain = self._make_mock_wind_domain({"mean_wind_speed": wind_speed})

        with pytest.raises(ValueError) as exc_info:
            domain.convert_units("wrong_name", "km/h")

        error_msg = str(exc_info.value)
        assert "mean_wind_speed" in error_msg
        assert "Available" in error_msg
