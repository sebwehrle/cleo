"""Tests for CF reuse in LCOE-family metrics (PR 6)."""

import json
import numpy as np
import pytest
import xarray as xr

from cleo.domains import (
    _cf_spec_is_compatible,
    _cf_spec_matches,
    _extract_requested_cf_subset,
    _inject_economics_params_and_cf_reuse,
)
from cleo.wind_metrics import resolve_cf_spec


def _cf_store_dataset(*, coord_labels: list[object] | None = None) -> xr.Dataset:
    """Build a minimal wind store with full-axis capacity factors."""
    if coord_labels is None:
        coord_labels = ["T1", "T2"]

    cf = xr.DataArray(
        np.arange(8, dtype=np.float64).reshape(2, 2, 2),
        dims=("turbine", "y", "x"),
        coords={
            "turbine": np.asarray(coord_labels, dtype=object),
            "y": np.array([0.0, 1.0], dtype=np.float64),
            "x": np.array([0.0, 1.0], dtype=np.float64),
        },
        name="capacity_factors",
        attrs={
            "cleo:cf_method": "rotor_node_average",
            "cleo:interpolation": "mu_cv_loglog",
            "cleo:air_density": 0,
            "cleo:rews_n": 12,
            "cleo:loss_factor": 1.0,
            "cleo:turbines_json": json.dumps(["T1", "T2"]),
            "cleo:computed_turbines_json": json.dumps(["T1", "T2"]),
        },
    )
    return xr.Dataset(
        {"capacity_factors": cf},
        coords=cf.coords,
        attrs={"cleo_turbines_json": '[{"id":"T1"},{"id":"T2"}]'},
    )


class _FakeAtlas:
    """Minimal atlas stub for economics reuse tests."""

    def _effective_hours_per_year(self) -> float:
        """Return the configured economics timebase."""
        return 8766.0


class _FakeDomain:
    """Minimal wind-domain stub exposing only store access used by reuse logic.

    Staged overlays are intentionally omitted because stored-CF reuse is the
    only path in scope for these tests.
    """

    def __init__(self, store_ds: xr.Dataset):
        """Store the active wind dataset used by the test."""
        self._atlas = _FakeAtlas()
        self._store_ds = store_ds

    def _store_data(self) -> xr.Dataset:
        """Return the active wind-store dataset."""
        return self._store_ds


class TestCfSpecMatches:
    """Tests for _cf_spec_matches helper function."""

    def test_returns_false_when_missing_metadata(self):
        """Returns False if existing CF lacks required metadata keys."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        # No cleo:* attrs
        spec = resolve_cf_spec(None)
        turbines = ("T1",)
        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_mode_mismatch(self):
        """Returns False if cf_mode doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"method": "rotor_node_average"})  # Different mode
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_air_density_mismatch(self):
        """Returns False if air_density doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:air_density"] = 0  # False
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"air_density": True})  # Different
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_rews_n_mismatch(self):
        """Returns False if rews_n doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"rews_n": 24})  # Different
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_loss_factor_mismatch(self):
        """Returns False if loss_factor doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"loss_factor": 0.95})  # Different
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_turbines_mismatch(self):
        """Returns False if turbines don't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec(None)
        turbines = ("T1", "T2")  # Different

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_turbine_order_differs(self):
        """Returns False if turbine order differs (order matters)."""
        cf = xr.DataArray(np.ones((2, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1", "T2"])

        spec = resolve_cf_spec(None)
        turbines = ("T2", "T1")  # Same turbines, different order

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_true_when_all_match(self):
        """Returns True when all CF parameters and turbines match exactly."""
        cf = xr.DataArray(np.ones((2, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1", "T2"])

        spec = resolve_cf_spec(
            None
        )  # Defaults: method=rotor_node_average, air_density=False, rews_n=12, loss_factor=1.0
        turbines = ("T1", "T2")

        assert _cf_spec_matches(cf, spec, turbines)

    def test_returns_true_with_custom_spec_match(self):
        """Returns True when custom spec matches existing CF exactly."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:air_density"] = 1  # True
        cf.attrs["cleo:rews_n"] = 24
        cf.attrs["cleo:loss_factor"] = 0.95
        cf.attrs["cleo:turbines_json"] = json.dumps(["TurbineA"])

        spec = resolve_cf_spec(
            {
                "method": "hub_height_weibull",
                "air_density": True,
                "rews_n": 24,
                "loss_factor": 0.95,
            }
        )
        turbines = ("TurbineA",)

        assert _cf_spec_matches(cf, spec, turbines)

    def test_air_density_int_bool_conversion(self):
        """Handles air_density stored as int (0/1) for netCDF compatibility."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 1  # Stored as int
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"air_density": True})  # Bool in spec
        turbines = ("T1",)

        assert _cf_spec_matches(cf, spec, turbines)


class TestResolveCfSpecForReuse:
    """Tests for resolve_cf_spec used in CF reuse context."""

    def test_defaults_match_expected_values(self):
        """Default CF spec values match what assess module produces."""
        spec = resolve_cf_spec(None)
        assert spec["method"] == "rotor_node_average"
        assert spec["air_density"] is False
        assert spec["rews_n"] == 12
        assert spec["loss_factor"] == 1.0

    def test_partial_override_preserves_other_defaults(self):
        """Partial spec override keeps non-specified defaults."""
        spec = resolve_cf_spec({"method": "hub_height_weibull"})
        assert spec["method"] == "hub_height_weibull"
        assert spec["air_density"] is False  # Default preserved
        assert spec["rews_n"] == 12  # Default preserved
        assert spec["loss_factor"] == 1.0  # Default preserved


class TestCfSpecCompatibility:
    """Tests for CF-settings compatibility independent of turbine breadth."""

    def test_returns_true_for_matching_full_axis_cf(self) -> None:
        """Matching physics settings stay reusable independent of turbine breadth."""
        cf = _cf_store_dataset()["capacity_factors"]

        assert _cf_spec_is_compatible(cf, resolve_cf_spec(None)) is True

    def test_returns_false_for_interpolation_mismatch(self) -> None:
        """Interpolation mismatch disables reuse."""
        cf = _cf_store_dataset()["capacity_factors"]

        assert _cf_spec_is_compatible(cf, resolve_cf_spec({"interpolation": "ak_logz"})) is False

    def test_returns_false_for_method_mismatch(self) -> None:
        """Method mismatch disables reuse."""
        cf = _cf_store_dataset()["capacity_factors"]

        assert _cf_spec_is_compatible(cf, resolve_cf_spec({"method": "hub_height_weibull"})) is False

    def test_returns_false_for_air_density_mismatch(self) -> None:
        """Air-density mismatch disables reuse."""
        cf = _cf_store_dataset()["capacity_factors"]

        assert _cf_spec_is_compatible(cf, resolve_cf_spec({"air_density": True})) is False

    def test_returns_false_for_rews_n_mismatch(self) -> None:
        """REWS quadrature mismatch disables reuse when method uses it."""
        cf = _cf_store_dataset()["capacity_factors"]

        assert _cf_spec_is_compatible(cf, resolve_cf_spec({"rews_n": 24})) is False

    def test_returns_false_for_loss_factor_mismatch(self) -> None:
        """Loss-factor mismatch disables reuse."""
        cf = _cf_store_dataset()["capacity_factors"]

        assert _cf_spec_is_compatible(cf, resolve_cf_spec({"loss_factor": 0.95})) is False


class TestExtractRequestedCfSubset:
    """Tests for extracting subset-scoped CF arrays from full-axis store data."""

    def test_extracts_subset_with_public_turbine_labels(self) -> None:
        """Public turbine labels support direct ordered subset selection."""
        store_ds = _cf_store_dataset()

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T2",))

        assert subset is not None
        assert subset.coords["turbine"].values.tolist() == ["T2"]
        np.testing.assert_allclose(subset.values, store_ds["capacity_factors"].sel(turbine="T2").values[None, ...])
        assert json.loads(subset.attrs["cleo:turbines_json"]) == ["T2"]

    def test_extracts_subset_with_numeric_store_labels(self) -> None:
        """Fallback mapping supports stores whose turbine coord labels are not IDs."""
        store_ds = _cf_store_dataset(coord_labels=[10, 20])

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T2",))

        assert subset is not None
        assert subset.coords["turbine"].values.tolist() == ["T2"]
        np.testing.assert_allclose(subset.values, store_ds["capacity_factors"].sel(turbine=20).values[None, ...])
        assert json.loads(subset.attrs["cleo:turbines_json"]) == ["T2"]

    def test_preserves_requested_turbine_order(self) -> None:
        """Requested turbine order remains authoritative in the extracted subset."""
        store_ds = _cf_store_dataset()

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T2", "T1"))

        assert subset is not None
        assert subset.coords["turbine"].values.tolist() == ["T2", "T1"]
        np.testing.assert_allclose(subset.sel(turbine="T2").values, store_ds["capacity_factors"].sel(turbine="T2"))
        np.testing.assert_allclose(subset.sel(turbine="T1").values, store_ds["capacity_factors"].sel(turbine="T1"))

    def test_returns_none_when_requested_turbine_is_missing(self) -> None:
        """Missing turbines disable reuse conservatively."""
        store_ds = _cf_store_dataset()

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T9",))

        assert subset is None

    def test_returns_none_when_turbine_dim_is_absent(self) -> None:
        """Unsupported CF shapes do not participate in reuse."""
        cf = xr.DataArray(np.ones((2, 2), dtype=np.float64), dims=("y", "x"))

        subset = _extract_requested_cf_subset(cf, _cf_store_dataset(), ("T1",))

        assert subset is None

    def test_preserves_non_turbine_cf_attrs(self) -> None:
        """Subset extraction preserves CF settings attrs while rewriting turbines."""
        store_ds = _cf_store_dataset()

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T2",))

        assert subset is not None
        assert subset.attrs["cleo:cf_method"] == "rotor_node_average"
        assert subset.attrs["cleo:air_density"] == 0
        assert subset.attrs["cleo:rews_n"] == 12
        assert subset.attrs["cleo:loss_factor"] == 1.0
        assert json.loads(subset.attrs["cleo:turbines_json"]) == ["T2"]

    def test_reuses_exact_match_for_older_store_without_computed_provenance(self) -> None:
        """Older stores still reuse when the requested turbine list matches exactly."""
        store_ds = _cf_store_dataset()
        del store_ds["capacity_factors"].attrs["cleo:computed_turbines_json"]

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T1", "T2"))

        assert subset is not None
        assert subset.coords["turbine"].values.tolist() == ["T1", "T2"]
        np.testing.assert_allclose(subset.values, store_ds["capacity_factors"].values)

    def test_rejects_subset_reuse_for_older_store_without_computed_provenance(self) -> None:
        """Older normalized full-axis stores do not guess whether subsets were computed."""
        store_ds = _cf_store_dataset()
        del store_ds["capacity_factors"].attrs["cleo:computed_turbines_json"]

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T2",))

        assert subset is None

    def test_rejects_reuse_when_computed_provenance_is_malformed(self) -> None:
        """Malformed computed-slice provenance disables reuse instead of guessing."""
        store_ds = _cf_store_dataset()
        store_ds["capacity_factors"].attrs["cleo:computed_turbines_json"] = "{not-json"

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T1",))

        assert subset is None

    def test_rejects_reuse_when_store_axis_metadata_cannot_be_interpreted(self) -> None:
        """Subset extraction stays conservative when store-axis metadata is missing."""
        store_ds = _cf_store_dataset(coord_labels=["store-T1", "store-T2"])
        del store_ds.attrs["cleo_turbines_json"]

        subset = _extract_requested_cf_subset(store_ds["capacity_factors"], store_ds, ("T2",))

        assert subset is None


class TestInjectEconomicsParamsAndCfReuse:
    """Tests for economics reuse injection against stored full-axis CF data."""

    def test_injects_subset_scoped_precomputed_cf(self) -> None:
        """Compatible full-axis stored CF is reused as a requested subset."""

        store_ds = _cf_store_dataset()
        kwargs = {"turbines": ("T2",)}

        _inject_economics_params_and_cf_reuse(
            kwargs,
            metric="lcoe",
            spec={"composed": True},
            resolved_cf=resolve_cf_spec(None),
            domain=_FakeDomain(store_ds),
        )

        assert kwargs["hours_per_year"] == pytest.approx(8766.0)
        assert "_precomputed_cf" in kwargs
        subset = kwargs["_precomputed_cf"]
        assert subset.coords["turbine"].values.tolist() == ["T2"]
        assert json.loads(subset.attrs["cleo:turbines_json"]) == ["T2"]

    def test_does_not_inject_precomputed_cf_for_uncomputed_turbine_slice(self) -> None:
        """NaN-padded full-axis slices do not count as reusable computed turbines."""

        store_ds = _cf_store_dataset()
        store_ds["capacity_factors"].attrs["cleo:computed_turbines_json"] = json.dumps(["T2"])
        kwargs = {"turbines": ("T1",)}

        _inject_economics_params_and_cf_reuse(
            kwargs,
            metric="lcoe",
            spec={"composed": True},
            resolved_cf=resolve_cf_spec(None),
            domain=_FakeDomain(store_ds),
        )

        assert kwargs["hours_per_year"] == pytest.approx(8766.0)
        assert "_precomputed_cf" not in kwargs

    def test_injects_precomputed_cf_for_older_exact_match_store(self) -> None:
        """Older stores still reuse exactly matching turbine requests."""
        store_ds = _cf_store_dataset()
        del store_ds["capacity_factors"].attrs["cleo:computed_turbines_json"]
        kwargs = {"turbines": ("T1", "T2")}

        _inject_economics_params_and_cf_reuse(
            kwargs,
            metric="lcoe",
            spec={"composed": True},
            resolved_cf=resolve_cf_spec(None),
            domain=_FakeDomain(store_ds),
        )

        assert "_precomputed_cf" in kwargs
        subset = kwargs["_precomputed_cf"]
        assert subset.coords["turbine"].values.tolist() == ["T1", "T2"]

    def test_does_not_inject_precomputed_cf_when_store_axis_metadata_is_missing(self) -> None:
        """Malformed store metadata falls back to recomputation instead of guessing."""
        store_ds = _cf_store_dataset(coord_labels=["store-T1", "store-T2"])
        del store_ds.attrs["cleo_turbines_json"]
        kwargs = {"turbines": ("T2",)}

        _inject_economics_params_and_cf_reuse(
            kwargs,
            metric="lcoe",
            spec={"composed": True},
            resolved_cf=resolve_cf_spec(None),
            domain=_FakeDomain(store_ds),
        )

        assert kwargs["hours_per_year"] == pytest.approx(8766.0)
        assert "_precomputed_cf" not in kwargs
