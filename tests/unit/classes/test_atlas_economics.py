"""Unit tests for Atlas.configure_economics() and related methods."""

import math
import pytest
from pathlib import Path

from cleo import Atlas


class TestConfigureEconomics:
    """Tests for Atlas.configure_economics()."""

    def test_configure_economics_single_field(self, tmp_path: Path) -> None:
        """Can configure a single economics field."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        atlas.configure_economics(discount_rate=0.05)

        assert atlas.economics_configured is not None
        assert atlas.economics_configured["discount_rate"] == 0.05

    def test_configure_economics_all_fields(self, tmp_path: Path) -> None:
        """Can configure all economics fields at once."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        atlas.configure_economics(
            discount_rate=0.05,
            lifetime_a=25,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            bos_cost_share=0.30,
        )

        config = atlas.economics_configured
        assert config is not None
        assert config["discount_rate"] == 0.05
        assert config["lifetime_a"] == 25
        assert config["om_fixed_eur_per_kw_a"] == 20.0
        assert config["om_variable_eur_per_kwh"] == 0.008
        assert config["bos_cost_share"] == 0.30

    def test_configure_economics_incremental(self, tmp_path: Path) -> None:
        """Multiple configure_economics calls merge values."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        atlas.configure_economics(discount_rate=0.05)
        atlas.configure_economics(lifetime_a=25)

        config = atlas.economics_configured
        assert config is not None
        assert config["discount_rate"] == 0.05
        assert config["lifetime_a"] == 25

    def test_configure_economics_override_previous(self, tmp_path: Path) -> None:
        """Later configure_economics calls override previous values."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        atlas.configure_economics(discount_rate=0.05)
        atlas.configure_economics(discount_rate=0.06)

        assert atlas.economics_configured["discount_rate"] == 0.06

    def test_economics_configured_none_initially(self, tmp_path: Path) -> None:
        """economics_configured is None before any configuration."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        assert atlas.economics_configured is None


class TestConfigureEconomicsValidation:
    """Tests for configure_economics() parameter validation."""

    def test_discount_rate_type_error(self, tmp_path: Path) -> None:
        """discount_rate must be numeric."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(TypeError, match="discount_rate must be numeric"):
            atlas.configure_economics(discount_rate="0.05")

    def test_discount_rate_range_negative(self, tmp_path: Path) -> None:
        """discount_rate must be >= 0."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="discount_rate must be finite and in range"):
            atlas.configure_economics(discount_rate=-0.01)

    def test_discount_rate_range_too_high(self, tmp_path: Path) -> None:
        """discount_rate must be < 1."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="discount_rate must be finite and in range"):
            atlas.configure_economics(discount_rate=1.0)

    def test_discount_rate_nan(self, tmp_path: Path) -> None:
        """discount_rate must be finite."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="discount_rate must be finite"):
            atlas.configure_economics(discount_rate=float("nan"))

    def test_lifetime_a_type_error(self, tmp_path: Path) -> None:
        """lifetime_a must be int."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(TypeError, match="lifetime_a must be int"):
            atlas.configure_economics(lifetime_a=25.0)

    def test_lifetime_a_range_zero(self, tmp_path: Path) -> None:
        """lifetime_a must be positive."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="lifetime_a must be positive"):
            atlas.configure_economics(lifetime_a=0)

    def test_lifetime_a_range_negative(self, tmp_path: Path) -> None:
        """lifetime_a must be positive."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="lifetime_a must be positive"):
            atlas.configure_economics(lifetime_a=-5)

    def test_om_fixed_type_error(self, tmp_path: Path) -> None:
        """om_fixed_eur_per_kw_a must be numeric."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(TypeError, match="om_fixed_eur_per_kw_a must be numeric"):
            atlas.configure_economics(om_fixed_eur_per_kw_a="20.0")

    def test_om_fixed_range_negative(self, tmp_path: Path) -> None:
        """om_fixed_eur_per_kw_a must be >= 0."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="om_fixed_eur_per_kw_a must be finite and >= 0"):
            atlas.configure_economics(om_fixed_eur_per_kw_a=-1.0)

    def test_om_variable_type_error(self, tmp_path: Path) -> None:
        """om_variable_eur_per_kwh must be numeric."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(TypeError, match="om_variable_eur_per_kwh must be numeric"):
            atlas.configure_economics(om_variable_eur_per_kwh="0.008")

    def test_om_variable_range_negative(self, tmp_path: Path) -> None:
        """om_variable_eur_per_kwh must be >= 0."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="om_variable_eur_per_kwh must be finite and >= 0"):
            atlas.configure_economics(om_variable_eur_per_kwh=-0.001)

    def test_bos_cost_share_type_error(self, tmp_path: Path) -> None:
        """bos_cost_share must be numeric."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(TypeError, match="bos_cost_share must be numeric"):
            atlas.configure_economics(bos_cost_share="0.3")

    def test_bos_cost_share_range_negative(self, tmp_path: Path) -> None:
        """bos_cost_share must be >= 0."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="bos_cost_share must be finite and in range"):
            atlas.configure_economics(bos_cost_share=-0.1)

    def test_bos_cost_share_range_too_high(self, tmp_path: Path) -> None:
        """bos_cost_share must be <= 1."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="bos_cost_share must be finite and in range"):
            atlas.configure_economics(bos_cost_share=1.1)


class TestEffectiveEconomics:
    """Tests for Atlas._effective_economics()."""

    def test_effective_economics_defaults_only(self, tmp_path: Path) -> None:
        """Without configuration, returns only defaults."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        effective = atlas._effective_economics()

        assert effective == {"bos_cost_share": 0.0}

    def test_effective_economics_with_baseline(self, tmp_path: Path) -> None:
        """Baseline configuration is included in effective."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(discount_rate=0.05, lifetime_a=25)

        effective = atlas._effective_economics()

        assert effective["discount_rate"] == 0.05
        assert effective["lifetime_a"] == 25
        assert effective["bos_cost_share"] == 0.0  # default

    def test_effective_economics_override_takes_precedence(self, tmp_path: Path) -> None:
        """Per-call overrides take precedence over baseline."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(discount_rate=0.05)

        effective = atlas._effective_economics(overrides={"discount_rate": 0.06})

        assert effective["discount_rate"] == 0.06

    def test_effective_economics_override_adds_new_fields(self, tmp_path: Path) -> None:
        """Override can add fields not in baseline."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(discount_rate=0.05)

        effective = atlas._effective_economics(overrides={"lifetime_a": 30})

        assert effective["discount_rate"] == 0.05
        assert effective["lifetime_a"] == 30

    def test_effective_economics_override_none(self, tmp_path: Path) -> None:
        """Override=None behaves same as no override."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(discount_rate=0.05)

        effective = atlas._effective_economics(overrides=None)

        assert effective["discount_rate"] == 0.05


class TestEconomicsClonePreservation:
    """Tests for economics configuration preservation across clones."""

    def test_clone_preserves_economics_configured(self, tmp_path: Path) -> None:
        """_clone_for_selection preserves economics_configured."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(
            discount_rate=0.05,
            lifetime_a=25,
            bos_cost_share=0.30,
        )

        clone = atlas._clone_for_selection()

        assert clone.economics_configured is not None
        assert clone.economics_configured["discount_rate"] == 0.05
        assert clone.economics_configured["lifetime_a"] == 25
        assert clone.economics_configured["bos_cost_share"] == 0.30

    def test_clone_economics_is_independent_copy(self, tmp_path: Path) -> None:
        """Clone's economics_configured is an independent copy."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(discount_rate=0.05)

        clone = atlas._clone_for_selection()
        clone.configure_economics(discount_rate=0.06)

        # Original should be unchanged
        assert atlas.economics_configured["discount_rate"] == 0.05
        assert clone.economics_configured["discount_rate"] == 0.06

    def test_clone_preserves_none_economics(self, tmp_path: Path) -> None:
        """Clone preserves None economics_configured."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")

        clone = atlas._clone_for_selection()

        assert clone.economics_configured is None

    def test_select_inplace_false_preserves_economics(self, tmp_path: Path) -> None:
        """select(..., inplace=False) preserves economics configuration."""
        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(discount_rate=0.05, bos_cost_share=0.25)

        clone = atlas.select(region=None, inplace=False)

        assert clone is not None
        assert clone.economics_configured is not None
        assert clone.economics_configured["discount_rate"] == 0.05
        assert clone.economics_configured["bos_cost_share"] == 0.25
