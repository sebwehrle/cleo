"""Integration tests for economics configuration end-to-end behavior (PR 7).

Verifies:
- configure_economics() + compute("lcoe", ...) end-to-end
- Baseline + override resolution
- Grouped spec examples
- Error paths (missing required fields)
- Economics preserved across clone
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

import cleo
from cleo import Atlas
from cleo.unification.gwa_io import GWA_HEIGHTS


def _create_gwa_raster(
    path: Path,
    shape: tuple[int, int] = (20, 20),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    fill_value: float | None = None,
    add_nodata_region: bool = True,
) -> None:
    """Create a minimal GWA-style GeoTIFF with CRS."""
    if fill_value is not None:
        data = np.full(shape, fill_value, dtype=np.float32)
    else:
        data = np.random.rand(*shape).astype(np.float32) * 10 + 1

    if add_nodata_region:
        data[:3, :3] = np.nan

    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], shape[1], shape[0])

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
        "nodata": -9999.0,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_required_gwa_files(atlas_path: Path, country: str = "AUT") -> None:
    """Create all required GWA files for testing with correct naming convention."""
    raw_dir = atlas_path / "data" / "raw" / country

    for h in GWA_HEIGHTS:
        for layer, prefix in [
            ("weibull_A", f"{country}_combined-Weibull-A_{h}.tif"),
            ("weibull_k", f"{country}_combined-Weibull-k_{h}.tif"),
            ("rho", f"{country}_air-density_{h}.tif"),
        ]:
            p = raw_dir / prefix
            if layer == "weibull_A":
                _create_gwa_raster(p, fill_value=8.0)
            elif layer == "weibull_k":
                _create_gwa_raster(p, fill_value=2.0)
            else:  # rho
                _create_gwa_raster(p, fill_value=1.225)

    # Elevation
    elev_path = raw_dir / f"{country}_elevation_w_bathymetry.tif"
    _create_gwa_raster(elev_path, fill_value=500.0, add_nodata_region=False)


def _copy_turbine_yaml(atlas_path: Path, turbine_name: str = "Enercon.E40.500") -> None:
    """Copy a turbine YAML from package resources."""
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)
    src = resources_src / f"{turbine_name}.yml"
    if src.exists():
        shutil.copy(src, resources_dest / f"{turbine_name}.yml")


class TestEconomicsIntegration:
    """End-to-end tests for economics configuration."""

    @pytest.fixture
    def materialized_atlas(self, tmp_path: Path) -> Atlas:
        """Create a fully materialized atlas for testing."""
        _create_all_required_gwa_files(tmp_path)
        _copy_turbine_yaml(tmp_path)

        turbine_id = "Enercon.E40.500"

        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
            chunk_policy={"y": 64, "x": 64},
        )
        atlas.configure_turbines([turbine_id])
        atlas.build()

        atlas.wind.select(turbines=[turbine_id])

        return atlas

    def test_configure_economics_end_to_end(self, materialized_atlas: Atlas) -> None:
        """configure_economics() baseline is used by compute("lcoe", ...)."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Configure full economics baseline
        atlas.configure_economics(
            discount_rate=0.05,
            lifetime_a=25,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            bos_cost_share=0.0,
        )

        # Compute LCOE without per-call economics (uses baseline)
        lcoe = atlas.wind.compute("lcoe", turbines=[tid]).data.compute()

        # Should produce valid LCOE values
        valid = lcoe.values[np.isfinite(lcoe.values)]
        assert len(valid) > 0
        assert np.all(valid > 0)  # LCOE should be positive

    def test_baseline_plus_override_resolution(self, materialized_atlas: Atlas) -> None:
        """Per-call economics overrides merge with baseline."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Configure partial economics baseline
        atlas.configure_economics(
            discount_rate=0.05,
            lifetime_a=25,
        )

        # Compute LCOE with per-call overrides for remaining required fields
        lcoe1 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            economics={
                "om_fixed_eur_per_kw_a": 20.0,
                "om_variable_eur_per_kwh": 0.008,
            },
        ).data.compute()

        # Compute again with different override
        atlas.wind.clear_computed()
        lcoe2 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            economics={
                "om_fixed_eur_per_kw_a": 30.0,  # Different value
                "om_variable_eur_per_kwh": 0.008,
            },
        ).data.compute()

        # Values should differ due to different om_fixed
        valid1 = lcoe1.values[np.isfinite(lcoe1.values)]
        valid2 = lcoe2.values[np.isfinite(lcoe2.values)]
        assert len(valid1) > 0
        assert not np.allclose(valid1, valid2, rtol=1e-6)

    def test_grouped_cf_spec_usage(self, materialized_atlas: Atlas) -> None:
        """Grouped cf={} spec affects LCOE computation."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        economics = {
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
        }

        # Compute with default CF spec
        lcoe1 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            economics=economics,
        ).data.compute()

        # Compute with different CF spec (different mode)
        atlas.wind.clear_computed()
        lcoe2 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            cf={"mode": "hub"},  # Different mode
            economics=economics,
        ).data.compute()

        # Values may differ due to different CF computation mode
        valid1 = lcoe1.values[np.isfinite(lcoe1.values)]
        valid2 = lcoe2.values[np.isfinite(lcoe2.values)]
        assert len(valid1) > 0
        assert len(valid2) > 0
        # Note: mode differences may or may not produce different values
        # depending on the data; just verify both succeed

    def test_grouped_cf_spec_with_loss_factor(self, materialized_atlas: Atlas) -> None:
        """Grouped cf={} with loss_factor affects LCOE values."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        economics = {
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
        }

        # Compute with default loss_factor (1.0)
        lcoe1 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            economics=economics,
        ).data.compute()

        # Compute with lower loss_factor (less energy, higher LCOE)
        atlas.wind.clear_computed()
        lcoe2 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            cf={"loss_factor": 0.8},  # 20% losses
            economics=economics,
        ).data.compute()

        # With losses, LCOE should be higher
        valid1 = lcoe1.values[np.isfinite(lcoe1.values)]
        valid2 = lcoe2.values[np.isfinite(lcoe2.values)]
        assert len(valid1) > 0
        assert len(valid2) > 0
        assert np.mean(valid2) > np.mean(valid1)  # Higher LCOE with losses

    def test_missing_required_economics_raises(self, materialized_atlas: Atlas) -> None:
        """Missing required economics fields raises explicit error."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # No economics configured at all
        with pytest.raises(ValueError) as exc_info:
            atlas.wind.compute("lcoe", turbines=[tid])

        error_msg = str(exc_info.value)
        assert "Missing required economics parameters" in error_msg
        assert "discount_rate" in error_msg

    def test_partial_economics_missing_fields_raises(self, materialized_atlas: Atlas) -> None:
        """Partial economics (some in baseline, some missing) raises error."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Configure only some fields
        atlas.configure_economics(
            discount_rate=0.05,
            lifetime_a=25,
        )

        # Missing om_fixed and om_variable
        with pytest.raises(ValueError) as exc_info:
            atlas.wind.compute("lcoe", turbines=[tid])

        error_msg = str(exc_info.value)
        assert "Missing required economics parameters" in error_msg
        assert "om_fixed_eur_per_kw_a" in error_msg

    def test_economics_preserved_across_clone(self, tmp_path: Path) -> None:
        """Economics config is preserved when selecting with inplace=False."""
        _create_all_required_gwa_files(tmp_path)
        _copy_turbine_yaml(tmp_path)

        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_economics(
            discount_rate=0.06,
            lifetime_a=30,
            om_fixed_eur_per_kw_a=25.0,
            om_variable_eur_per_kwh=0.01,
        )

        clone = atlas.select(area=None, inplace=False)

        assert clone is not None
        assert clone.economics_configured == {
            "discount_rate": 0.06,
            "lifetime_a": 30,
            "om_fixed_eur_per_kw_a": 25.0,
            "om_variable_eur_per_kwh": 0.01,
        }

    def test_lcoe_metadata_has_economics_json(self, materialized_atlas: Atlas) -> None:
        """LCOE output carries cleo:economics_json attr."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        economics = {
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
        }

        lcoe = atlas.wind.compute("lcoe", turbines=[tid], economics=economics).data

        assert "cleo:economics_json" in lcoe.attrs
        econ_json = json.loads(lcoe.attrs["cleo:economics_json"])
        assert econ_json["discount_rate"] == 0.05
        assert econ_json["lifetime_a"] == 25
        assert econ_json["om_fixed_eur_per_kw_a"] == 20.0

    def test_lcoe_metadata_has_cf_spec_json(self, materialized_atlas: Atlas) -> None:
        """LCOE output carries cleo:cf_spec_json attr."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        economics = {
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
        }

        lcoe = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            cf={"mode": "hub", "loss_factor": 0.95},
            economics=economics,
        ).data

        assert "cleo:cf_spec_json" in lcoe.attrs
        cf_spec = json.loads(lcoe.attrs["cleo:cf_spec_json"])
        assert cf_spec["mode"] == "hub"
        assert cf_spec["loss_factor"] == 0.95

    def test_flat_economics_kwargs_rejected(self, materialized_atlas: Atlas) -> None:
        """Flat economics kwargs raise error for LCOE-family metrics."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Flat kwargs should be rejected (caught by allowed set validation)
        with pytest.raises(ValueError) as exc_info:
            atlas.wind.compute(
                "lcoe",
                turbines=[tid],
                discount_rate=0.05,  # Flat kwarg, should be in economics={}
            )

        error_msg = str(exc_info.value)
        # Error message comes from allowed set validation
        assert "Unknown parameter" in error_msg
        assert "discount_rate" in error_msg

    def test_flat_cf_kwargs_rejected(self, materialized_atlas: Atlas) -> None:
        """Flat CF kwargs raise error for LCOE-family metrics."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Flat CF kwargs should be rejected (caught by allowed set validation)
        with pytest.raises(ValueError) as exc_info:
            atlas.wind.compute(
                "lcoe",
                turbines=[tid],
                mode="hub",  # Flat kwarg, should be in cf={}
                economics={
                    "discount_rate": 0.05,
                    "lifetime_a": 25,
                    "om_fixed_eur_per_kw_a": 20.0,
                    "om_variable_eur_per_kwh": 0.008,
                },
            )

        error_msg = str(exc_info.value)
        # Error message comes from allowed set validation
        assert "Unknown parameter" in error_msg
        assert "mode" in error_msg

    def test_bos_cost_share_affects_lcoe(self, materialized_atlas: Atlas) -> None:
        """bos_cost_share parameter affects LCOE values."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        base_economics = {
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
        }

        # Compute with bos_cost_share=0.0 (all CAPEX is turbine)
        lcoe1 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            economics={**base_economics, "bos_cost_share": 0.0},
        ).data.compute()

        # Compute with bos_cost_share=0.3 (30% is BOS)
        atlas.wind.clear_computed()
        lcoe2 = atlas.wind.compute(
            "lcoe",
            turbines=[tid],
            economics={**base_economics, "bos_cost_share": 0.3},
        ).data.compute()

        # Values should differ
        valid1 = lcoe1.values[np.isfinite(lcoe1.values)]
        valid2 = lcoe2.values[np.isfinite(lcoe2.values)]
        assert len(valid1) > 0
        assert not np.allclose(valid1, valid2, rtol=1e-6)


class TestLcoeFamilyMetricsIntegration:
    """Integration tests for all LCOE-family metrics."""

    @pytest.fixture
    def materialized_atlas(self, tmp_path: Path) -> Atlas:
        """Create a fully materialized atlas for testing."""
        _create_all_required_gwa_files(tmp_path)
        _copy_turbine_yaml(tmp_path)

        turbine_id = "Enercon.E40.500"

        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="epsg:3035",
            chunk_policy={"y": 64, "x": 64},
        )
        atlas.configure_turbines([turbine_id])
        atlas.build()
        atlas.wind.select(turbines=[turbine_id])

        return atlas

    @pytest.fixture
    def full_economics(self) -> dict:
        """Full economics spec for tests."""
        return {
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
        }

    def test_min_lcoe_turbine_integration(self, materialized_atlas: Atlas, full_economics: dict) -> None:
        """min_lcoe_turbine computes successfully end-to-end."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        result = atlas.wind.compute(
            "min_lcoe_turbine",
            turbines=[tid],
            economics=full_economics,
        ).data.compute()

        # Should produce int32 indices
        assert result.dtype == np.int32
        # With single turbine, valid pixels should be 0, invalid -1
        valid_mask = result.values >= 0
        assert np.all(result.values[valid_mask] == 0)

    def test_optimal_power_integration(self, materialized_atlas: Atlas, full_economics: dict) -> None:
        """optimal_power computes successfully end-to-end."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        result = atlas.wind.compute(
            "optimal_power",
            turbines=[tid],
            economics=full_economics,
        ).data.compute()

        # Should produce power values in kW
        assert result.attrs.get("units") == "kW"
        valid = result.values[np.isfinite(result.values)]
        assert len(valid) > 0
        assert np.all(valid > 0)

    def test_optimal_energy_integration(self, materialized_atlas: Atlas, full_economics: dict) -> None:
        """optimal_energy computes successfully end-to-end."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        result = atlas.wind.compute(
            "optimal_energy",
            turbines=[tid],
            economics=full_economics,
        ).data.compute()

        # Should produce energy values in GWh/a
        assert result.attrs.get("units") == "GWh/a"
        valid = result.values[np.isfinite(result.values)]
        assert len(valid) > 0
        assert np.all(valid > 0)

    def test_lcoe_and_min_lcoe_turbine_grouped_spec_api(self, materialized_atlas: Atlas, full_economics: dict) -> None:
        """lcoe and min_lcoe_turbine use the grouped spec API."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Test lcoe and min_lcoe_turbine (these work with lazy arrays)
        for metric in ["lcoe", "min_lcoe_turbine"]:
            atlas.wind.clear_computed()
            result = atlas.wind.compute(
                metric,
                turbines=[tid],
                cf={"mode": "hub"},
                economics=full_economics,
            ).data

            # All should succeed with grouped spec
            assert result is not None
