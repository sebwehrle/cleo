"""Tests for kwarg validation in WindDomain.compute().

Consolidated tests for:
- Strict kwarg rejection (unknown parameters)
- Timebase rejection (hours_per_year) per CONTRACT A9
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cleo.domains import WindDomain


@pytest.fixture
def mock_domain(tmp_path: Path) -> WindDomain:
    """Create a WindDomain with minimal mock atlas."""
    mock_atlas = MagicMock()
    mock_atlas._wind_selected_turbines = ("FakeTurbine.T1.1000",)
    mock_atlas.wind_data = MagicMock()
    mock_atlas.landscape_data = MagicMock()
    return WindDomain(mock_atlas)


class TestUnknownKwargRejection:
    """Tests for rejecting unknown kwargs in WindDomain.compute()."""

    @pytest.mark.parametrize(
        "metric,base_kwargs",
        [
            ("capacity_factors", {"turbines": ["FakeTurbine.T1.1000"]}),
            ("wind_speed", {"height": 100}),
            (
                "lcoe",
                {
                    "turbines": ["FakeTurbine.T1.1000"],
                    "om_fixed_eur_per_kw_a": 20,
                    "om_variable_eur_per_kwh": 0.008,
                    "discount_rate": 0.05,
                    "lifetime_a": 25,
                },
            ),
        ],
    )
    def test_compute_rejects_unknown_kwarg(self, mock_domain: WindDomain, metric: str, base_kwargs: dict) -> None:
        """Unknown kwargs are rejected with helpful error for all metrics."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            mock_domain.compute(metric, **base_kwargs, bogus_param=42)

    def test_error_message_shows_allowed_params(self, mock_domain: WindDomain) -> None:
        with pytest.raises(ValueError, match="Allowed:") as exc_info:
            mock_domain.compute(
                "capacity_factors",
                turbines=["FakeTurbine.T1.1000"],
                bad_param=42,
            )
        error_msg = str(exc_info.value)
        assert "air_density" in error_msg or "method" in error_msg

    def test_error_message_hides_internal_params(self, mock_domain: WindDomain) -> None:
        """hours_per_year is internal and should not appear in allowed list."""
        with pytest.raises(ValueError, match="Allowed:") as exc_info:
            mock_domain.compute(
                "lcoe",
                turbines=["FakeTurbine.T1.1000"],
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
                invalid_param="test",
            )
        assert "hours_per_year" not in str(exc_info.value)

    def test_compute_accepts_valid_capacity_factors_params(self, mock_domain: WindDomain) -> None:
        """Valid params should not raise unknown param error."""
        try:
            mock_domain.compute(
                "capacity_factors",
                turbines=["FakeTurbine.T1.1000"],
                air_density=True,
                method="hub_height_weibull",
                loss_factor=0.95,
            )
        except ValueError as e:
            assert "Unknown parameter" not in str(e)
        except Exception:
            pass  # Other errors are fine - we're just testing param validation


class TestTimebaseRejection:
    """Tests for rejecting hours_per_year in WindDomain.compute() per CONTRACT A9."""

    @pytest.mark.parametrize(
        "metric,base_kwargs",
        [
            (
                "lcoe",
                {
                    "turbines": ["FakeTurbine.T1.1000"],
                    "om_fixed_eur_per_kw_a": 20,
                    "om_variable_eur_per_kwh": 0.008,
                    "discount_rate": 0.05,
                    "lifetime_a": 25,
                },
            ),
            (
                "min_lcoe_turbine",
                {
                    "turbines": ["FakeTurbine.T1.1000"],
                    "om_fixed_eur_per_kw_a": 20,
                    "om_variable_eur_per_kwh": 0.008,
                    "discount_rate": 0.05,
                    "lifetime_a": 25,
                },
            ),
            (
                "optimal_power",
                {
                    "turbines": ["FakeTurbine.T1.1000"],
                    "om_fixed_eur_per_kw_a": 20,
                    "om_variable_eur_per_kwh": 0.008,
                    "discount_rate": 0.05,
                    "lifetime_a": 25,
                },
            ),
            (
                "optimal_energy",
                {
                    "turbines": ["FakeTurbine.T1.1000"],
                    "om_fixed_eur_per_kw_a": 20,
                    "om_variable_eur_per_kwh": 0.008,
                    "discount_rate": 0.05,
                    "lifetime_a": 25,
                },
            ),
            ("capacity_factors", {"turbines": ["FakeTurbine.T1.1000"]}),
            ("wind_speed", {"height": 100}),
        ],
    )
    def test_compute_rejects_hours_per_year(self, mock_domain: WindDomain, metric: str, base_kwargs: dict) -> None:
        """hours_per_year is rejected for all metrics."""
        with pytest.raises(ValueError, match="cannot be passed to compute"):
            mock_domain.compute(metric, **base_kwargs, hours_per_year=8760)

    def test_error_message_mentions_configure_timebase(self, mock_domain: WindDomain) -> None:
        with pytest.raises(ValueError, match="configure_timebase"):
            mock_domain.compute(
                "lcoe",
                turbines=["FakeTurbine.T1.1000"],
                hours_per_year=8760,
                om_fixed_eur_per_kw_a=20,
                om_variable_eur_per_kwh=0.008,
                discount_rate=0.05,
                lifetime_a=25,
            )
