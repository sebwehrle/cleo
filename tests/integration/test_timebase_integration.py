"""Integration tests for timebase configuration end-to-end behavior.

Verifies:
- configure_timebase() affects LCOE-family metrics
- configure_timebase() does NOT affect physics metrics (CF)
- Metadata presence/absence contract is enforced
- Timebase is preserved across select(..., inplace=False)
"""

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


class TestTimebaseIntegration:
    """End-to-end tests for timebase configuration."""

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
        # Configure to limit turbines in wind store
        atlas.configure_turbines([turbine_id])
        atlas.build()

        # Select the turbine
        atlas.wind.select(turbines=[turbine_id])

        return atlas

    def test_configure_timebase_affects_lcoe(self, materialized_atlas: Atlas) -> None:
        """LCOE values change when timebase changes."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Compute LCOE with default timebase (using grouped economics spec)
        economics = {
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
            "discount_rate": 0.05,
            "lifetime_a": 25,
        }

        lcoe1 = atlas.wind.compute("lcoe", turbines=[tid], economics=economics).data.compute()

        # Configure different timebase
        atlas.configure_timebase(hours_per_year=8760.0)

        # Compute LCOE again
        atlas.wind.clear_computed()
        lcoe2 = atlas.wind.compute("lcoe", turbines=[tid], economics=economics).data.compute()

        # Values should differ
        valid1 = lcoe1.values[np.isfinite(lcoe1.values)]
        valid2 = lcoe2.values[np.isfinite(lcoe2.values)]
        assert len(valid1) > 0
        assert not np.allclose(valid1, valid2, rtol=1e-6)

    def test_timebase_does_not_affect_capacity_factors(self, materialized_atlas: Atlas) -> None:
        """CF values are identical regardless of timebase configuration."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Compute CF with default timebase
        cf1 = atlas.wind.compute("capacity_factors", turbines=[tid]).data.compute()

        # Configure different timebase
        atlas.configure_timebase(hours_per_year=8760.0)

        # Compute CF again
        atlas.wind.clear_computed()
        cf2 = atlas.wind.compute("capacity_factors", turbines=[tid]).data.compute()

        # Values should be identical
        np.testing.assert_allclose(
            cf1.values,
            cf2.values,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_lcoe_has_hours_per_year_metadata(self, materialized_atlas: Atlas) -> None:
        """LCOE outputs carry cleo:hours_per_year attr."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        economics = {
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
            "discount_rate": 0.05,
            "lifetime_a": 25,
        }

        lcoe = atlas.wind.compute("lcoe", turbines=[tid], economics=economics).data

        assert "cleo:hours_per_year" in lcoe.attrs
        assert lcoe.attrs["cleo:hours_per_year"] == Atlas.DEFAULT_HOURS_PER_YEAR

    def test_capacity_factors_has_no_hours_per_year_metadata(self, materialized_atlas: Atlas) -> None:
        """CF outputs do NOT carry cleo:hours_per_year attr."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        cf = atlas.wind.compute("capacity_factors", turbines=[tid]).data

        assert "cleo:hours_per_year" not in cf.attrs

    def test_timebase_preserved_across_clone(self, tmp_path: Path) -> None:
        """Timebase config is preserved when selecting with inplace=False."""
        _create_all_required_gwa_files(tmp_path)
        _copy_turbine_yaml(tmp_path)

        atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
        atlas.configure_timebase(hours_per_year=8760.0)

        clone = atlas.select(region=None, inplace=False)

        assert clone is not None
        assert clone.timebase_configured == {"hours_per_year": 8760.0}
        assert clone._effective_hours_per_year() == 8760.0

    def test_lcoe_metadata_reflects_configured_timebase(self, materialized_atlas: Atlas) -> None:
        """LCOE metadata reflects the atlas-configured timebase value."""
        atlas = materialized_atlas
        tid = atlas.wind.turbines[0]

        # Configure custom timebase
        atlas.configure_timebase(hours_per_year=8750.0)

        economics = {
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
            "discount_rate": 0.05,
            "lifetime_a": 25,
        }

        lcoe = atlas.wind.compute("lcoe", turbines=[tid], economics=economics).data

        assert lcoe.attrs["cleo:hours_per_year"] == 8750.0
