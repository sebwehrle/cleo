"""Tests for hours_per_year rejection in WindDomain.compute().

Verifies CONTRACT A9 compliance:
- hours_per_year must not be passed to compute("lcoe", ...)
- hours_per_year must not be passed to any economics metric
- Error message directs user to configure_timebase()
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cleo.domains import WindDomain


class TestTimebaseRejection:
    """Tests for rejecting hours_per_year in WindDomain.compute()."""

    def _make_mock_domain(self, tmp_path: Path) -> WindDomain:
        """Create a WindDomain with minimal mock atlas."""
        mock_atlas = MagicMock()
        mock_atlas._wind_selected_turbines = ("FakeTurbine.T1.1000",)
        mock_atlas.wind_data = MagicMock()
        mock_atlas.landscape_data = MagicMock()
        return WindDomain(mock_atlas)

    def test_compute_rejects_hours_per_year_for_lcoe(self, tmp_path: Path) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="cannot be passed to compute"):
            domain.compute(
                "lcoe",
                turbines=["FakeTurbine.T1.1000"],
                hours_per_year=8760,
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
            )

    def test_compute_rejects_hours_per_year_for_min_lcoe_turbine(
        self, tmp_path: Path
    ) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="cannot be passed to compute"):
            domain.compute(
                "min_lcoe_turbine",
                turbines=["FakeTurbine.T1.1000"],
                hours_per_year=8760,
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
            )

    def test_compute_rejects_hours_per_year_for_optimal_power(
        self, tmp_path: Path
    ) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="cannot be passed to compute"):
            domain.compute(
                "optimal_power",
                turbines=["FakeTurbine.T1.1000"],
                hours_per_year=8760,
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
            )

    def test_compute_rejects_hours_per_year_for_optimal_energy(
        self, tmp_path: Path
    ) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="cannot be passed to compute"):
            domain.compute(
                "optimal_energy",
                turbines=["FakeTurbine.T1.1000"],
                hours_per_year=8760,
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
            )

    def test_error_message_mentions_configure_timebase(self, tmp_path: Path) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="configure_timebase"):
            domain.compute(
                "lcoe",
                turbines=["FakeTurbine.T1.1000"],
                hours_per_year=8760,
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
            )

    def test_compute_rejects_hours_per_year_for_physics_metric(
        self, tmp_path: Path
    ) -> None:
        """Even physics metrics should reject hours_per_year at the gate."""
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="cannot be passed to compute"):
            domain.compute(
                "capacity_factors",
                turbines=["FakeTurbine.T1.1000"],
                hours_per_year=8760,
            )

    def test_compute_rejects_hours_per_year_for_mean_wind_speed(
        self, tmp_path: Path
    ) -> None:
        domain = self._make_mock_domain(tmp_path)

        with pytest.raises(ValueError, match="cannot be passed to compute"):
            domain.compute(
                "mean_wind_speed",
                height=100,
                hours_per_year=8760,
            )
