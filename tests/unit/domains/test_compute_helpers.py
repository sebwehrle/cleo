"""Tests for compute helper functions in cleo.domains module."""

from __future__ import annotations

import pytest
import numpy as np
import xarray as xr

from cleo.domains import (
    _reject_inplace_kwarg,
    _validate_wind_compute_kwargs,
    _validate_landscape_compute_kwargs,
    _validate_distance_sources,
    _check_distance_conflicts,
    _compute_single_distance,
)


# =============================================================================
# Tests for shared validation helpers
# =============================================================================


class TestRejectInplaceKwarg:
    """Tests for _reject_inplace_kwarg function."""

    def test_raises_when_inplace_present(self) -> None:
        """Raises ValueError when inplace kwarg is present."""
        with pytest.raises(ValueError, match="does not accept inplace"):
            _reject_inplace_kwarg({"inplace": True}, "compute")

    def test_passes_when_inplace_absent(self) -> None:
        """Does not raise when inplace kwarg is absent."""
        _reject_inplace_kwarg({"other": "value"}, "compute")
        _reject_inplace_kwarg({}, "compute")


class TestValidateWindComputeKwargs:
    """Tests for _validate_wind_compute_kwargs function."""

    @staticmethod
    def _spec(*, allowed: set[str], required: set[str] | None = None) -> dict:
        return {
            "allowed": frozenset(allowed),
            "required": frozenset() if required is None else frozenset(required),
        }

    def test_raises_when_overwrite_present(self) -> None:
        """Raises ValueError when overwrite kwarg is present."""
        with pytest.raises(ValueError, match="materialize-only parameter"):
            _validate_wind_compute_kwargs({"overwrite": True}, "test_metric", self._spec(allowed={"overwrite"}))

    def test_raises_when_allow_method_change_present(self) -> None:
        """Raises ValueError when allow_method_change kwarg is present."""
        with pytest.raises(ValueError, match="materialize-only parameter"):
            _validate_wind_compute_kwargs(
                {"allow_method_change": True},
                "test_metric",
                self._spec(allowed={"allow_method_change"}),
            )

    def test_raises_when_hours_per_year_present(self) -> None:
        """Raises ValueError when hours_per_year kwarg is present."""
        with pytest.raises(ValueError, match="Timebase parameter"):
            _validate_wind_compute_kwargs(
                {"hours_per_year": 8760}, "test_metric", self._spec(allowed={"hours_per_year"})
            )

    def test_raises_when_unknown_kwargs_present(self) -> None:
        """Raises ValueError when unknown kwargs are present."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            _validate_wind_compute_kwargs({"a": 1, "unknown": 2}, "test_metric", self._spec(allowed={"a", "b", "c"}))

    def test_filters_internal_params_from_error_message(self) -> None:
        """Internal params like hours_per_year are filtered from allowed list in error."""
        with pytest.raises(ValueError) as exc_info:
            _validate_wind_compute_kwargs({"unknown": 1}, "test_metric", self._spec(allowed={"a", "hours_per_year"}))
        assert "hours_per_year" not in str(exc_info.value).split("Allowed:")[1]

    def test_raises_when_required_kwargs_missing(self) -> None:
        """Raises ValueError when required kwargs are missing."""
        with pytest.raises(ValueError, match="Missing required parameters"):
            _validate_wind_compute_kwargs({}, "test_metric", self._spec(allowed={"a"}, required={"a"}))

    def test_passes_when_kwargs_are_valid(self) -> None:
        """Does not raise when required and allowed kwargs are satisfied."""
        _validate_wind_compute_kwargs({"a": 1}, "test_metric", self._spec(allowed={"a"}, required={"a"}))


# =============================================================================
# Tests for landscape compute helpers
# =============================================================================


class TestValidateLandscapeComputeKwargs:
    """Tests for _validate_landscape_compute_kwargs function."""

    def test_rejects_inplace(self) -> None:
        """Rejects inplace kwarg."""
        with pytest.raises(ValueError, match="does not accept inplace"):
            _validate_landscape_compute_kwargs({"inplace": True, "source": "roads"}, "distance")

    def test_rejects_unknown_kwargs(self) -> None:
        """Rejects unknown kwargs."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            _validate_landscape_compute_kwargs({"source": "roads", "bad_kwarg": 1}, "distance")

    def test_requires_source(self) -> None:
        """Requires source parameter."""
        with pytest.raises(ValueError, match="Missing required"):
            _validate_landscape_compute_kwargs({}, "distance")

    def test_accepts_valid_kwargs(self) -> None:
        """Accepts valid kwargs."""
        _validate_landscape_compute_kwargs({"source": "roads"}, "distance")
        _validate_landscape_compute_kwargs({"source": "roads", "name": "dist_roads"}, "distance")
        _validate_landscape_compute_kwargs(
            {"source": "roads", "name": "dist_roads", "if_exists": "replace"}, "distance"
        )


class TestValidateDistanceSources:
    """Tests for _validate_distance_sources function."""

    def test_raises_when_source_not_in_store(self) -> None:
        """Raises ValueError when source variable doesn't exist in store."""
        store_ds = xr.Dataset({"var1": (["y", "x"], np.ones((3, 3)))})
        with pytest.raises(ValueError, match="Unknown distance source variable"):
            _validate_distance_sources(["nonexistent"], store_ds)

    def test_passes_when_source_exists(self) -> None:
        """Does not raise when source variable exists."""
        store_ds = xr.Dataset({"roads": (["y", "x"], np.ones((3, 3)))})
        _validate_distance_sources(["roads"], store_ds)

    def test_validates_multiple_sources(self) -> None:
        """Validates all sources in list."""
        store_ds = xr.Dataset(
            {
                "roads": (["y", "x"], np.ones((3, 3))),
                "water": (["y", "x"], np.ones((3, 3))),
            }
        )
        _validate_distance_sources(["roads", "water"], store_ds)

        with pytest.raises(ValueError, match="Unknown distance source"):
            _validate_distance_sources(["roads", "missing"], store_ds)


class TestCheckDistanceConflicts:
    """Tests for _check_distance_conflicts function."""

    def test_raises_on_conflict_when_if_exists_error(self) -> None:
        """Raises ValueError when variable exists and if_exists='error'."""
        store_ds = xr.Dataset({"distance_roads": (["y", "x"], np.ones((3, 3)))})
        with pytest.raises(ValueError, match="would overwrite existing"):
            _check_distance_conflicts(["distance_roads"], "error", {}, store_ds)

    def test_raises_on_staged_conflict(self) -> None:
        """Raises ValueError when variable is staged and if_exists='error'."""
        store_ds = xr.Dataset({"other": (["y", "x"], np.ones((3, 3)))})
        staged = {"distance_roads": xr.DataArray(np.ones((3, 3)))}
        with pytest.raises(ValueError, match="would overwrite existing"):
            _check_distance_conflicts(["distance_roads"], "error", staged, store_ds)

    def test_passes_when_no_conflict(self) -> None:
        """Does not raise when no conflicts exist."""
        store_ds = xr.Dataset({"other": (["y", "x"], np.ones((3, 3)))})
        _check_distance_conflicts(["distance_roads"], "error", {}, store_ds)

    def test_passes_when_if_exists_not_error(self) -> None:
        """Does not raise when if_exists is not 'error'."""
        store_ds = xr.Dataset({"distance_roads": (["y", "x"], np.ones((3, 3)))})
        _check_distance_conflicts(["distance_roads"], "replace", {}, store_ds)
        _check_distance_conflicts(["distance_roads"], "noop", {}, store_ds)


class TestComputeSingleDistance:
    """Tests for _compute_single_distance function."""

    @pytest.fixture
    def sample_store_ds(self) -> xr.Dataset:
        """Create a sample store dataset with CRS."""
        import rioxarray  # noqa: F401 - needed for rio accessor

        ds = xr.Dataset(
            {
                "roads": (["y", "x"], np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)),
                "valid_mask": (["y", "x"], np.ones((3, 3), dtype=bool)),
            },
            coords={"y": [0.0, 100.0, 200.0], "x": [0.0, 100.0, 200.0]},
        )
        # Add CRS (projected CRS required for distance calculation)
        ds = ds.rio.write_crs("EPSG:32632")
        return ds

    def test_computes_new_distance(self, sample_store_ds: xr.Dataset) -> None:
        """Computes distance for new variable."""
        dist, should_stage = _compute_single_distance(
            src="roads",
            nm="distance_roads",
            if_exists="error",
            expected_spec='{"test": "spec"}',
            staged_overlays={},
            store_ds=sample_store_ds,
            valid_mask=sample_store_ds["valid_mask"],
        )
        assert should_stage is True
        assert dist.name == "distance_roads"
        assert dist.attrs["cleo:metric"] == "distance"
        assert dist.attrs["cleo:distance_source"] == "roads"

    def test_noop_returns_staged_when_matching(self, sample_store_ds: xr.Dataset) -> None:
        """Returns staged variable when noop and spec matches."""
        staged_da = xr.DataArray(
            np.zeros((3, 3)),
            attrs={"cleo:distance_spec_json": '{"test":"spec"}'},
            name="distance_roads",
        )
        dist, should_stage = _compute_single_distance(
            src="roads",
            nm="distance_roads",
            if_exists="noop",
            expected_spec='{"test":"spec"}',
            staged_overlays={"distance_roads": staged_da},
            store_ds=sample_store_ds,
            valid_mask=sample_store_ds["valid_mask"],
        )
        assert should_stage is False
        assert dist is staged_da

    def test_noop_raises_when_staged_spec_differs(self, sample_store_ds: xr.Dataset) -> None:
        """Raises ValueError when noop but staged spec differs."""
        staged_da = xr.DataArray(
            np.zeros((3, 3)),
            attrs={"cleo:distance_spec_json": '{"different":"spec"}'},
            name="distance_roads",
        )
        with pytest.raises(ValueError, match="different distance spec"):
            _compute_single_distance(
                src="roads",
                nm="distance_roads",
                if_exists="noop",
                expected_spec='{"test":"spec"}',
                staged_overlays={"distance_roads": staged_da},
                store_ds=sample_store_ds,
                valid_mask=sample_store_ds["valid_mask"],
            )

    def test_noop_returns_store_var_when_matching(self, sample_store_ds: xr.Dataset) -> None:
        """Returns store variable when noop and spec matches."""
        store_with_distance = sample_store_ds.assign(
            {
                "distance_roads": xr.DataArray(
                    np.zeros((3, 3)),
                    dims=["y", "x"],
                    attrs={"cleo:distance_spec_json": '{"test":"spec"}'},
                )
            }
        )
        dist, should_stage = _compute_single_distance(
            src="roads",
            nm="distance_roads",
            if_exists="noop",
            expected_spec='{"test":"spec"}',
            staged_overlays={},
            store_ds=store_with_distance,
            valid_mask=store_with_distance["valid_mask"],
        )
        assert should_stage is False

    def test_replace_computes_even_when_exists(self, sample_store_ds: xr.Dataset) -> None:
        """Computes new distance even when variable exists with replace."""
        store_with_distance = sample_store_ds.assign(
            {
                "distance_roads": xr.DataArray(
                    np.zeros((3, 3)),
                    dims=["y", "x"],
                    attrs={"cleo:distance_spec_json": '{"old":"spec"}'},
                )
            }
        )
        dist, should_stage = _compute_single_distance(
            src="roads",
            nm="distance_roads",
            if_exists="replace",
            expected_spec='{"new":"spec"}',
            staged_overlays={},
            store_ds=store_with_distance,
            valid_mask=store_with_distance["valid_mask"],
        )
        assert should_stage is True
        assert dist.attrs["cleo:distance_spec_json"] == '{"new":"spec"}'
