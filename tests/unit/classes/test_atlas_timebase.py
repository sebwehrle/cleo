"""Tests for Atlas timebase configuration (CONTRACT A9 compliance).

Verifies:
- configure_timebase() accepts valid values
- configure_timebase() rejects invalid values
- timebase_configured property returns configured state
- _effective_hours_per_year() resolves correctly
- Timebase config is preserved across clone/select operations
"""

import math
import pytest
from pathlib import Path

from cleo.atlas import Atlas


class TestConfigureTimebase:
    """Tests for Atlas.configure_timebase() and related properties."""

    def test_configure_timebase_valid_float(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_timebase(hours_per_year=8760.0)
        assert atlas.timebase_configured == {"hours_per_year": 8760.0}
        assert atlas._effective_hours_per_year() == 8760.0

    def test_configure_timebase_valid_int(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_timebase(hours_per_year=8760)
        assert atlas.timebase_configured == {"hours_per_year": 8760.0}
        assert atlas._effective_hours_per_year() == 8760.0

    def test_configure_timebase_default_unconfigured(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        assert atlas.timebase_configured is None
        assert atlas._effective_hours_per_year() == Atlas.DEFAULT_HOURS_PER_YEAR
        assert atlas._effective_hours_per_year() == 8766.0

    def test_configure_timebase_overwrites_previous(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_timebase(hours_per_year=8760.0)
        assert atlas._effective_hours_per_year() == 8760.0
        atlas.configure_timebase(hours_per_year=8000.0)
        assert atlas._effective_hours_per_year() == 8000.0

    def test_configure_timebase_rejects_zero(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(ValueError, match="must be finite and > 0"):
            atlas.configure_timebase(hours_per_year=0)

    def test_configure_timebase_rejects_negative(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(ValueError, match="must be finite and > 0"):
            atlas.configure_timebase(hours_per_year=-100)

    def test_configure_timebase_rejects_inf(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(ValueError, match="must be finite and > 0"):
            atlas.configure_timebase(hours_per_year=float("inf"))

    def test_configure_timebase_rejects_negative_inf(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(ValueError, match="must be finite and > 0"):
            atlas.configure_timebase(hours_per_year=float("-inf"))

    def test_configure_timebase_rejects_nan(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(ValueError, match="must be finite and > 0"):
            atlas.configure_timebase(hours_per_year=float("nan"))

    def test_configure_timebase_rejects_string(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(TypeError, match="must be numeric"):
            atlas.configure_timebase(hours_per_year="8766")

    def test_configure_timebase_rejects_none(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(TypeError, match="must be numeric"):
            atlas.configure_timebase(hours_per_year=None)

    def test_configure_timebase_rejects_list(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        with pytest.raises(TypeError, match="must be numeric"):
            atlas.configure_timebase(hours_per_year=[8766])


class TestTimebaseClonePreservation:
    """Tests for timebase preservation across clone/select operations."""

    def test_clone_preserves_timebase_configured(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_timebase(hours_per_year=8760.0)
        clone = atlas._clone_for_selection()
        assert clone.timebase_configured == {"hours_per_year": 8760.0}
        assert clone._effective_hours_per_year() == 8760.0

    def test_clone_preserves_timebase_none(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        clone = atlas._clone_for_selection()
        assert clone.timebase_configured is None
        assert clone._effective_hours_per_year() == 8766.0

    def test_clone_timebase_is_independent_copy(self, tmp_path: Path) -> None:
        """Modifying clone timebase should not affect original."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_timebase(hours_per_year=8760.0)
        clone = atlas._clone_for_selection()
        clone.configure_timebase(hours_per_year=8000.0)
        assert atlas._effective_hours_per_year() == 8760.0
        assert clone._effective_hours_per_year() == 8000.0

    def test_select_inplace_false_preserves_timebase(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_timebase(hours_per_year=8760.0)
        clone = atlas.select(region=None, inplace=False)
        assert clone is not None
        assert clone.timebase_configured == {"hours_per_year": 8760.0}
        assert clone._effective_hours_per_year() == 8760.0

    def test_select_inplace_false_preserves_timebase_none(self, tmp_path: Path) -> None:
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        clone = atlas.select(region=None, inplace=False)
        assert clone is not None
        assert clone.timebase_configured is None


class TestDefaultHoursPerYearConstant:
    """Tests for the DEFAULT_HOURS_PER_YEAR class constant."""

    def test_default_value_is_8766(self) -> None:
        assert Atlas.DEFAULT_HOURS_PER_YEAR == 8766.0

    def test_default_is_finite(self) -> None:
        assert math.isfinite(Atlas.DEFAULT_HOURS_PER_YEAR)

    def test_default_is_positive(self) -> None:
        assert Atlas.DEFAULT_HOURS_PER_YEAR > 0
