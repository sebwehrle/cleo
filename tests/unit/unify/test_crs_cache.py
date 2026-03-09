"""Tests for CRS cache functionality in unification GWA helpers.

Tests are offline-only: network calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from cleo.unification.gwa_io import (
    _crs_cache_path,
    _ensure_proj_crs_preflight,
    _load_or_fetch_gwa_crs,
    _open_gwa_raster,
)


class MockAtlas:
    """Minimal Atlas-like object for testing."""

    def __init__(self, path: Path, country: str = "AUT"):
        self.path = path
        self.country = country
        self.crs = "epsg:3035"
        self.area = None


class TestCrsCachePath:
    """Tests for _crs_cache_path."""

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        """Returns path under intermediates/crs_cache/<iso3>.wkt."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "AUT")

        expected = tmp_path / "intermediates" / "crs_cache" / "AUT.wkt"
        assert cache_path == expected

    def test_different_iso3_different_path(self, tmp_path: Path) -> None:
        """Different ISO3 codes produce different paths."""
        atlas = MockAtlas(tmp_path)

        path_aut = _crs_cache_path(atlas, "AUT")
        path_deu = _crs_cache_path(atlas, "DEU")

        assert path_aut != path_deu
        assert path_aut.name == "AUT.wkt"
        assert path_deu.name == "DEU.wkt"


def test_proj_preflight_raises_clear_error_on_proj_db_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight reports environment-level PROJ mismatch with actionable hints."""

    def _fail_resolve(_code: str) -> CRS:
        raise RuntimeError(
            "The EPSG code is unknown. PROJ: proj_create_from_database: "
            "proj.db contains DATABASE.LAYOUT.VERSION.MINOR = 3 whereas a number >= 6 is expected."
        )

    monkeypatch.setattr("cleo.unification.gwa_io._resolve_epsg_crs", _fail_resolve)
    monkeypatch.setenv("PROJ_DATA", "/env/share/proj")
    monkeypatch.setenv("GDAL_DATA", "/env/share/gdal")

    with pytest.raises(RuntimeError, match="outside cleo") as exc:
        _ensure_proj_crs_preflight("epsg:3035")

    msg = str(exc.value)
    assert "PROJ_DATA" in msg
    assert "GDAL_DATA" in msg
    assert "DATABASE.LAYOUT.VERSION" in msg


def test_open_gwa_raster_runs_preflight_before_raster_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight guard should fail before trying to open raster files."""
    atlas = MockAtlas(tmp_path)
    calls = {"count": 0}

    def _fail_preflight(_target_crs) -> None:  # noqa: ANN001
        calls["count"] += 1
        raise RuntimeError("preflight failed")

    monkeypatch.setattr("cleo.unification.gwa_io._ensure_proj_crs_preflight", _fail_preflight)

    with pytest.raises(RuntimeError, match="preflight failed"):
        _open_gwa_raster(
            atlas,
            tmp_path / "missing.tif",
            iso3="AUT",
            target_crs="epsg:3035",
        )

    assert calls["count"] == 1


class TestLoadOrFetchGwaCrs:
    """Tests for _load_or_fetch_gwa_crs."""

    def test_reads_from_cache_if_exists(self, tmp_path: Path) -> None:
        """Returns CRS from cache file without calling fetch."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "AUT")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a valid WKT to cache
        test_crs = CRS.from_epsg(3035)
        cache_path.write_text(test_crs.to_wkt(), encoding="utf-8")

        with patch("cleo.unification.gwa_io.fetch_gwa_crs") as mock_fetch:
            result = _load_or_fetch_gwa_crs(atlas, "AUT")

            # Should NOT call fetch
            mock_fetch.assert_not_called()

            # Should return CRS from cache
            assert result.to_epsg() == 3035

    def test_fetches_and_caches_if_missing(self, tmp_path: Path) -> None:
        """Fetches CRS and writes to cache if cache doesn't exist."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "AUT")

        # Ensure cache doesn't exist
        assert not cache_path.exists()

        # Mock the fetch to return a CRS string
        test_crs_str = "urn:ogc:def:crs:EPSG::3035"

        with patch("cleo.unification.gwa_io.fetch_gwa_crs", return_value=test_crs_str) as mock_fetch:
            result = _load_or_fetch_gwa_crs(atlas, "AUT")

            # Should call fetch exactly once
            mock_fetch.assert_called_once_with("AUT")

            # Should return valid CRS
            assert result is not None
            assert result.to_epsg() == 3035

            # Cache file should exist now
            assert cache_path.exists()

    def test_fetch_called_only_once_across_multiple_calls(self, tmp_path: Path) -> None:
        """Second call reads from cache, doesn't fetch again."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "AUT")

        test_crs_str = "urn:ogc:def:crs:EPSG::3035"

        with patch("cleo.unification.gwa_io.fetch_gwa_crs", return_value=test_crs_str) as mock_fetch:
            # First call - should fetch
            result1 = _load_or_fetch_gwa_crs(atlas, "AUT")
            assert mock_fetch.call_count == 1

            # Second call - should read from cache
            result2 = _load_or_fetch_gwa_crs(atlas, "AUT")
            assert mock_fetch.call_count == 1  # Still 1, not 2

            # Both should return same CRS
            assert result1.to_epsg() == result2.to_epsg() == 3035

    def test_parses_and_caches_crs84_name(self, tmp_path: Path) -> None:
        """Parses OGC CRS84 from fetched GeoJSON CRS name and caches it."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "ROU")
        assert not cache_path.exists()

        crs_name = "urn:ogc:def:crs:OGC:1.3:CRS84"
        with patch("cleo.unification.gwa_io.fetch_gwa_crs", return_value=crs_name) as mock_fetch:
            result = _load_or_fetch_gwa_crs(atlas, "ROU")

        mock_fetch.assert_called_once_with("ROU")
        assert result.to_epsg() == 4326
        assert cache_path.exists()

    def test_raises_on_fetch_failure(self, tmp_path: Path) -> None:
        """Raises RuntimeError if fetch fails and cache doesn't exist."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "AUT")

        # Ensure no cache
        assert not cache_path.exists()

        with patch("cleo.unification.gwa_io.fetch_gwa_crs", side_effect=RuntimeError("Network error")):
            with pytest.raises(RuntimeError, match="Failed to fetch GWA CRS for AUT"):
                _load_or_fetch_gwa_crs(atlas, "AUT")

            # Cache should NOT be created on failure
            assert not cache_path.exists()

    def test_raises_on_empty_cache(self, tmp_path: Path) -> None:
        """Raises RuntimeError if cache file is empty."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "AUT")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty cache file
        cache_path.write_text("", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Empty CRS cache file"):
            _load_or_fetch_gwa_crs(atlas, "AUT")

    def test_raises_on_whitespace_only_cache(self, tmp_path: Path) -> None:
        """Raises RuntimeError if cache file contains only whitespace."""
        atlas = MockAtlas(tmp_path)
        cache_path = _crs_cache_path(atlas, "AUT")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Create whitespace-only cache file
        cache_path.write_text("   \n\t  ", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Empty CRS cache file"):
            _load_or_fetch_gwa_crs(atlas, "AUT")


class TestOpenGwaRasterWithCrsCache:
    """Tests for _open_gwa_raster using CRS cache."""

    def _create_test_raster(
        self,
        path: Path,
        *,
        with_crs: bool = True,
        crs_epsg: int = 3035,
        shape: tuple[int, int] = (10, 10),
        bounds: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    ) -> None:
        """Create a minimal test GeoTIFF."""
        data = np.random.rand(*shape).astype(np.float32)

        transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], shape[1], shape[0])

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": shape[1],
            "height": shape[0],
            "count": 1,
            "transform": transform,
        }

        if with_crs:
            profile["crs"] = CRS.from_epsg(crs_epsg)

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

    def test_opens_raster_with_embedded_crs(self, tmp_path: Path) -> None:
        """Opens raster that has CRS embedded, no fetch needed."""
        atlas = MockAtlas(tmp_path)

        # Create raster WITH CRS
        raster_path = tmp_path / "test_with_crs.tif"
        self._create_test_raster(raster_path, with_crs=True, crs_epsg=3035)

        with patch("cleo.unification.gwa_io.fetch_gwa_crs") as mock_fetch:
            da = _open_gwa_raster(
                atlas,
                raster_path,
                iso3="AUT",
                target_crs="epsg:3035",
            )

            # Should NOT call fetch (CRS already in raster)
            mock_fetch.assert_not_called()

            # Should have valid data
            assert da is not None
            assert "x" in da.dims
            assert "y" in da.dims

    def test_uses_cache_for_raster_without_crs(self, tmp_path: Path) -> None:
        """Uses CRS cache when raster lacks embedded CRS."""
        atlas = MockAtlas(tmp_path)

        # Pre-populate CRS cache
        cache_path = _crs_cache_path(atlas, "AUT")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(CRS.from_epsg(3035).to_wkt(), encoding="utf-8")

        # Create raster WITHOUT CRS
        raster_path = tmp_path / "test_no_crs.tif"
        self._create_test_raster(raster_path, with_crs=False)

        with patch("cleo.unification.gwa_io.fetch_gwa_crs") as mock_fetch:
            da = _open_gwa_raster(
                atlas,
                raster_path,
                iso3="AUT",
                target_crs="epsg:3035",
            )

            # Should NOT call fetch (cache exists)
            mock_fetch.assert_not_called()

            # Should have valid data with CRS applied
            assert da is not None
            assert da.rio.crs is not None

    def test_fetches_crs_for_raster_without_crs_and_no_cache(self, tmp_path: Path) -> None:
        """Fetches CRS when raster lacks CRS and cache doesn't exist."""
        atlas = MockAtlas(tmp_path)

        # Create raster WITHOUT CRS
        raster_path = tmp_path / "test_no_crs.tif"
        self._create_test_raster(raster_path, with_crs=False)

        test_crs_str = "urn:ogc:def:crs:EPSG::3035"

        with patch("cleo.unification.gwa_io.fetch_gwa_crs", return_value=test_crs_str) as mock_fetch:
            da = _open_gwa_raster(
                atlas,
                raster_path,
                iso3="AUT",
                target_crs="epsg:3035",
            )

            # Should call fetch exactly once
            mock_fetch.assert_called_once_with("AUT")

            # Should have valid data with CRS applied
            assert da is not None
            assert da.rio.crs is not None

            # Cache should be created
            cache_path = _crs_cache_path(atlas, "AUT")
            assert cache_path.exists()

    def test_raises_when_fetch_fails_for_crsless_raster(self, tmp_path: Path) -> None:
        """Keeps fail-fast behavior when CRS-less raster lookup fails."""
        atlas = MockAtlas(tmp_path)

        raster_path = tmp_path / "test_no_crs_non_geo.tif"
        self._create_test_raster(
            raster_path,
            with_crs=False,
            bounds=(1000.0, 1000.0, 2000.0, 2000.0),
        )

        with patch(
            "cleo.unification.gwa_io._load_or_fetch_gwa_crs",
            side_effect=RuntimeError("CRS lookup failed"),
        ):
            with pytest.raises(RuntimeError, match="CRS lookup failed"):
                _open_gwa_raster(
                    atlas,
                    raster_path,
                    iso3="ROU",
                    target_crs="epsg:3035",
                )
