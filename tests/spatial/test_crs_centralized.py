"""Tests for centralized CRS handling in cleo.spatial.

These tests ensure:
1. CRS utilities work correctly for various input formats
2. Semantic CRS equality is properly detected
3. No unnecessary reprojections occur when CRS is already matching
"""

import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

# Import the centralized CRS utilities
from cleo.spatial import canonical_crs_str, crs_equal, to_crs_if_needed, reproject_raster_if_needed


class TestCanonicalCrsStr:
    """Tests for canonical_crs_str function."""

    def test_epsg_int(self):
        """Integer EPSG code should return canonical string."""
        assert canonical_crs_str(4326) == "epsg:4326"

    def test_epsg_string_uppercase(self):
        """EPSG:4326 string should return lowercase canonical."""
        assert canonical_crs_str("EPSG:4326") == "epsg:4326"

    def test_epsg_string_lowercase(self):
        """epsg:4326 string should pass through."""
        assert canonical_crs_str("epsg:4326") == "epsg:4326"

    def test_proj4_string(self):
        """Proj4 string for EPSG:4326 should return canonical."""
        proj4 = "+proj=longlat +datum=WGS84 +no_defs"
        result = canonical_crs_str(proj4)
        # Should resolve to epsg:4326 or similar
        assert "4326" in result or "longlat" in result.lower()

    def test_invalid_crs_raises(self):
        """Invalid CRS should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse CRS"):
            canonical_crs_str("not_a_valid_crs_string")


class TestCrsEqual:
    """Tests for crs_equal function (semantic comparison)."""

    def test_same_epsg_int(self):
        """Same EPSG as integers should be equal."""
        assert crs_equal(4326, 4326) is True

    def test_same_epsg_different_formats(self):
        """EPSG:4326 in different formats should be equal."""
        assert crs_equal("EPSG:4326", "epsg:4326") is True
        assert crs_equal(4326, "EPSG:4326") is True
        assert crs_equal("epsg:4326", 4326) is True

    def test_different_crs(self):
        """Different CRS should not be equal."""
        assert crs_equal(4326, 3857) is False
        assert crs_equal("EPSG:4326", "EPSG:3857") is False

    def test_proj4_self_equivalent(self):
        """Proj4 string should match itself."""
        proj4 = "+proj=longlat +datum=WGS84 +no_defs"
        # Note: proj4 strings may not be equal to EPSG codes due to internal
        # differences (axis order, etc.) - this is expected pyproj behavior
        assert crs_equal(proj4, proj4) is True

    def test_invalid_crs_returns_false(self):
        """Invalid CRS should return False (not raise)."""
        assert crs_equal("invalid_crs", 4326) is False
        assert crs_equal(4326, "invalid_crs") is False
        assert crs_equal("invalid1", "invalid2") is False

    def test_none_handling(self):
        """None CRS should return False."""
        assert crs_equal(None, 4326) is False
        assert crs_equal(4326, None) is False
        assert crs_equal(None, None) is False


class TestToCrsIfNeeded:
    """Tests for to_crs_if_needed function (GeoDataFrame reprojection)."""

    @pytest.fixture
    def gdf_4326(self):
        """Create a simple GeoDataFrame in EPSG:4326."""
        geom = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")
        return gdf

    def test_no_reproject_when_same_crs(self, gdf_4326):
        """Should return same object when CRS matches."""
        result = to_crs_if_needed(gdf_4326, "EPSG:4326")
        # Should be same object (no reprojection)
        assert result is gdf_4326

    def test_no_reproject_when_equivalent_crs(self, gdf_4326):
        """Should return same object when CRS is semantically equivalent."""
        result = to_crs_if_needed(gdf_4326, 4326)
        assert result is gdf_4326

    def test_reproject_when_different_crs(self, gdf_4326):
        """Should reproject when CRS differs."""
        result = to_crs_if_needed(gdf_4326, "EPSG:3857")
        assert result is not gdf_4326
        assert crs_equal(result.crs, 3857)

    def test_raises_when_gdf_has_no_crs(self):
        """Should raise ValueError when input has no CRS."""
        geom = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=None)
        with pytest.raises(ValueError, match="GeoDataFrame has no CRS set"):
            to_crs_if_needed(gdf, 4326)


class TestReprojectRasterIfNeeded:
    """Tests for reproject_raster_if_needed function (DataArray reprojection)."""

    @pytest.fixture
    def raster_4326(self):
        """Create a simple raster DataArray in EPSG:4326."""
        import rioxarray  # noqa: F401
        data = np.ones((3, 4), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
        )
        da = da.rio.write_crs("EPSG:4326")
        return da

    def test_no_reproject_when_same_crs(self, raster_4326):
        """Should return same object when CRS matches."""
        result = reproject_raster_if_needed(raster_4326, "EPSG:4326")
        # Should be same object (no reprojection)
        assert result is raster_4326

    def test_no_reproject_when_equivalent_crs(self, raster_4326):
        """Should return same object when CRS is semantically equivalent."""
        result = reproject_raster_if_needed(raster_4326, 4326)
        assert result is raster_4326

    def test_reproject_when_different_crs(self, raster_4326):
        """Should reproject when CRS differs."""
        result = reproject_raster_if_needed(raster_4326, "EPSG:3857")
        assert result is not raster_4326
        assert crs_equal(result.rio.crs, 3857)

    def test_raises_when_raster_has_no_crs(self):
        """Should raise ValueError when input has no CRS."""
        data = np.ones((3, 4), dtype=np.float32)
        da = xr.DataArray(data, dims=["y", "x"])
        with pytest.raises(ValueError, match="Raster DataArray has no CRS set"):
            reproject_raster_if_needed(da, 4326)


class TestCrsEquivalenceMatrix:
    """Test CRS equivalence across various format representations."""

    @pytest.mark.parametrize(
        "crs_a,crs_b,expected",
        [
            # Same CRS, different formats
            ("EPSG:4326", "epsg:4326", True),
            ("EPSG:4326", 4326, True),
            (4326, "epsg:4326", True),
            # Different CRS
            ("EPSG:4326", "EPSG:3857", False),
            (4326, 3857, False),
            # UTM zones
            ("EPSG:32633", 32633, True),
            ("EPSG:32633", "EPSG:32634", False),
            # Note: proj4 strings may not match EPSG codes due to internal
            # differences (axis order, etc.) - this is expected pyproj behavior
            # so we only test self-equivalence for proj4
        ],
    )
    def test_crs_equivalence(self, crs_a, crs_b, expected):
        """Verify CRS equivalence matrix."""
        assert crs_equal(crs_a, crs_b) == expected


class TestNoUnnecessaryReprojection:
    """Tests ensuring no unnecessary reprojections occur."""

    def test_gdf_same_crs_no_copy(self):
        """GeoDataFrame with same CRS should not be copied."""
        geom = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")

        # Call with various equivalent CRS representations
        for crs in ["EPSG:4326", "epsg:4326", 4326]:
            result = to_crs_if_needed(gdf, crs)
            assert result is gdf, f"Unnecessary copy for CRS: {crs}"

    def test_raster_same_crs_no_copy(self):
        """Raster with same CRS should not be reprojected."""
        import rioxarray  # noqa: F401
        data = np.ones((3, 4), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
        )
        da = da.rio.write_crs("EPSG:4326")

        # Call with various equivalent CRS representations
        for crs in ["EPSG:4326", "epsg:4326", 4326]:
            result = reproject_raster_if_needed(da, crs)
            assert result is da, f"Unnecessary reprojection for CRS: {crs}"
