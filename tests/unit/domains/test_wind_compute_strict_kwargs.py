"""Tests for strict kwarg rejection in WindDomain.compute().

Verifies that unknown kwargs are rejected with helpful error messages.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from cleo.domains import WindDomain


class TestStrictKwargRejection:
    """Tests for rejecting unknown kwargs in WindDomain.compute()."""

    def _make_mock_domain(self, tmp_path: Path) -> WindDomain:
        """Create a WindDomain with minimal mock atlas."""
        mock_atlas = MagicMock()
        mock_atlas._wind_selected_turbines = ("FakeTurbine.T1.1000",)
        mock_atlas.wind_data = MagicMock()
        mock_atlas.landscape_data = MagicMock()
        return WindDomain(mock_atlas)

    def test_compute_rejects_unknown_kwarg_for_capacity_factors(self, tmp_path: Path) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="Unknown parameter"):
            domain.compute(
                "capacity_factors",
                turbines=["FakeTurbine.T1.1000"],
                bogus_param=42,
            )

    def test_compute_rejects_unknown_kwarg_for_mean_wind_speed(self, tmp_path: Path) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="Unknown parameter"):
            domain.compute(
                "mean_wind_speed",
                height=100,
                unknown_key="value",
            )

    def test_compute_rejects_unknown_kwarg_for_lcoe(self, tmp_path: Path) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="Unknown parameter"):
            domain.compute(
                "lcoe",
                turbines=["FakeTurbine.T1.1000"],
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
                invalid_param="test",
            )

    def test_error_message_shows_allowed_params(self, tmp_path: Path) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="Allowed:") as exc_info:
            domain.compute(
                "capacity_factors",
                turbines=["FakeTurbine.T1.1000"],
                bad_param=42,
            )

        error_msg = str(exc_info.value)
        # Should mention some allowed params
        assert "air_density" in error_msg or "mode" in error_msg

    def test_error_message_hides_internal_params(self, tmp_path: Path) -> None:
        """hours_per_year is internal and should not appear in allowed list."""
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="Allowed:") as exc_info:
            domain.compute(
                "lcoe",
                turbines=["FakeTurbine.T1.1000"],
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
                invalid_param="test",
            )

        error_msg = str(exc_info.value)
        # hours_per_year should not be in the allowed list shown to user
        assert "hours_per_year" not in error_msg

    def test_compute_accepts_valid_capacity_factors_params(self, tmp_path: Path) -> None:
        """Valid params should not raise unknown param error."""
        domain = self._make_mock_domain(tmp_path)

        # This should fail later (when trying to access wind data),
        # but not with "Unknown parameter" error
        try:
            domain.compute(
                "capacity_factors",
                turbines=["FakeTurbine.T1.1000"],
                air_density=True,
                mode="hub",
                loss_factor=0.95,
            )
        except ValueError as e:
            # Should not be about unknown parameters
            assert "Unknown parameter" not in str(e)
        except Exception:
            # Other errors are fine - we're just testing param validation
            pass
