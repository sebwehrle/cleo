"""Tests for cleo/utils.py convert function.

Tests the convert() function's integration with canonical unit utilities.
"""

import numpy as np
import pytest
import xarray as xr
from types import SimpleNamespace

from cleo.utils import convert


class TestConvertCanonicalUnits:
    """Tests for convert() using canonical 'units' attr."""

    def test_convert_reads_canonical_units_attr(self):
        """Convert reads from canonical 'units' attr."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["units"] = "m"
        self = SimpleNamespace(data=ds)

        convert(self, "v", "cm", inplace=True)

        assert self.data["v"].attrs["units"] == "cm"
        assert float(self.data["v"].values[0, 0]) == 100.0

    def test_convert_writes_canonical_units_attr(self):
        """Convert writes to canonical 'units' attr."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["units"] = "m"
        self = SimpleNamespace(data=ds)

        convert(self, "v", "cm", inplace=True)

        assert "units" in self.data["v"].attrs
        assert self.data["v"].attrs["units"] == "cm"

    def test_convert_preserves_other_attrs(self):
        """Convert preserves non-unit attrs."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["units"] = "m"
        ds["v"].attrs["foo"] = "bar"
        ds["v"].attrs["cleo:algo"] = "test"
        self = SimpleNamespace(data=ds)

        convert(self, "v", "cm", inplace=True)

        assert self.data["v"].attrs["foo"] == "bar"
        assert self.data["v"].attrs["cleo:algo"] == "test"
        assert self.data["v"].attrs["units"] == "cm"


class TestConvertLegacyFallback:
    """Tests for convert() legacy 'unit' attr fallback."""

    def test_convert_reads_legacy_unit_attr(self):
        """Convert falls back to legacy 'unit' attr when 'units' absent."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["unit"] = "m"  # Legacy key
        self = SimpleNamespace(data=ds)

        convert(self, "v", "cm", inplace=True)

        # Should convert correctly
        assert float(self.data["v"].values[0, 0]) == 100.0
        # Result should have canonical 'units' attr
        assert self.data["v"].attrs["units"] == "cm"

    def test_convert_removes_legacy_unit_attr(self):
        """Convert removes legacy 'unit' attr when converting."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["unit"] = "m"  # Legacy key
        self = SimpleNamespace(data=ds)

        convert(self, "v", "cm", inplace=True)

        assert "unit" not in self.data["v"].attrs
        assert "units" in self.data["v"].attrs


class TestConvertConflictDetection:
    """Tests for convert() conflict detection between 'units' and 'unit'."""

    def test_convert_raises_on_conflicting_attrs(self):
        """Convert raises when 'units' and 'unit' exist with different values."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["units"] = "m"
        ds["v"].attrs["unit"] = "km"  # Conflicting value
        self = SimpleNamespace(data=ds)

        with pytest.raises(ValueError, match="Conflicting unit attrs"):
            convert(self, "v", "cm", inplace=True)

    def test_convert_allows_matching_attrs(self):
        """Convert succeeds when 'units' and 'unit' match."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["units"] = "m"
        ds["v"].attrs["unit"] = "m"  # Same value
        self = SimpleNamespace(data=ds)

        convert(self, "v", "cm", inplace=True)

        assert float(self.data["v"].values[0, 0]) == 100.0
        assert self.data["v"].attrs["units"] == "cm"


class TestConvertMultipleVariables:
    """Tests for convert() with multiple variables."""

    def test_convert_multiple_variables_inplace(self):
        """Convert handles multiple variables inplace."""
        ds = xr.Dataset(
            {
                "distance": (("y", "x"), np.ones((2, 2)) * 1000),
                "elevation": (("y", "x"), np.ones((2, 2)) * 500),
            }
        )
        ds["distance"].attrs["units"] = "m"
        ds["elevation"].attrs["units"] = "m"
        self = SimpleNamespace(data=ds)

        convert(self, ["distance", "elevation"], "km", inplace=True)

        assert self.data["distance"].attrs["units"] == "km"
        assert self.data["elevation"].attrs["units"] == "km"
        assert float(self.data["distance"].values[0, 0]) == 1.0
        assert float(self.data["elevation"].values[0, 0]) == 0.5

    def test_convert_multiple_variables_returns_dict(self):
        """Convert returns dict when inplace=False."""
        ds = xr.Dataset(
            {
                "distance": (("y", "x"), np.ones((2, 2)) * 1000),
            }
        )
        ds["distance"].attrs["units"] = "m"
        self = SimpleNamespace(data=ds)

        result = convert(self, ["distance"], "km", inplace=False)

        assert isinstance(result, dict)
        assert "distance" in result
        assert result["distance"].attrs["units"] == "km"
        # Original unchanged
        assert self.data["distance"].attrs["units"] == "m"


class TestConvertFromUnitOverride:
    """Tests for convert() with explicit from_unit override."""

    def test_convert_uses_from_unit_override(self):
        """Convert uses explicit from_unit when provided."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["units"] = "m"  # Will be ignored
        self = SimpleNamespace(data=ds)

        # Convert as if it were in km, override the attr
        convert(self, "v", "m", from_unit="km", inplace=True)

        # 1 km = 1000 m
        assert float(self.data["v"].values[0, 0]) == 1000.0
        assert self.data["v"].attrs["units"] == "m"


class TestConvertErrorHandling:
    """Tests for convert() error handling."""

    def test_convert_raises_when_no_unit_source(self):
        """Convert raises when no unit attr and no from_unit."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        # No unit attr
        self = SimpleNamespace(data=ds)

        with pytest.raises(ValueError, match="No from-unit given"):
            convert(self, "v", "cm", inplace=True)

    def test_convert_raises_for_invalid_data_variable_type(self):
        """Convert raises for invalid data_variable type."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        self = SimpleNamespace(data=ds)

        with pytest.raises(ValueError, match="data_variable must be a string or a list"):
            convert(self, 123, "cm", inplace=True)

    def test_convert_raises_for_incompatible_units(self):
        """Convert raises for incompatible unit conversion."""
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        ds["v"].attrs["units"] = "m"
        self = SimpleNamespace(data=ds)

        with pytest.raises(ValueError, match="Cannot convert"):
            convert(self, "v", "kg", inplace=True)


class TestConvertDaskFriendly:
    """Tests for convert() dask compatibility."""

    def test_convert_preserves_dask_laziness(self):
        """Convert preserves dask laziness."""
        pytest.importorskip("dask")
        import dask.array as dask_array

        data = dask_array.from_array(np.ones((2, 2)), chunks=1)
        ds = xr.Dataset({"v": (("y", "x"), data)})
        ds["v"].attrs["units"] = "m"
        self = SimpleNamespace(data=ds)

        convert(self, "v", "cm", inplace=True)

        # Result should still be dask-backed
        assert hasattr(self.data["v"].data, "dask")

        # Values correct when computed
        assert float(self.data["v"].compute().values[0, 0]) == 100.0
