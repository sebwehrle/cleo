"""assess: test_air_density.

Contract tests for air density correction:
- output aligns to the dataset template grid
- CRS normalization comparisons behave as expected
- parent.path may be a str without breaking path handling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import rioxarray as rxr
import xarray as xr
from pyproj import CRS
from rasterio.transform import from_origin

import cleo.assess as assess
from tests.helpers.asserts import assert_same_coords
from tests.helpers.factories import da_xy_with_crs
from tests.helpers.raster import write_geotiff


@dataclass
class DummyParent:
    path: Path | str
    country: str
    crs: str
    region: str | None = None
    get_nuts_region: Callable[[str | None], object | None] = lambda _region: None  # noqa: E731


@dataclass
class DummySelf:
    parent: DummyParent
    data: xr.Dataset


def _make_self(*, base_dir: Path | str, country: str = "AUT", crs: str = "EPSG:4326") -> DummySelf:
    """Small, explicit test double for the 'self' expected by compute_air_density_correction."""
    parent = DummyParent(path=base_dir, country=country, crs=crs)
    return DummySelf(parent=parent, data=xr.Dataset())


def test_air_density_correction_matches_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Air-density correction must be aligned to the template grid (coords exactly match)."""
    country = "AUT"
    crs = "EPSG:4326"

    # Reference file at the exact path the function expects
    raw_dir = tmp_path / "data" / "raw" / country
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_file = raw_dir / f"{country}_combined-Weibull-A_100.tif"

    # Reference raster: 2x2
    ref_transform = from_origin(0, 2, 1, 1)
    write_geotiff(reference_file, np.ones((2, 2), dtype=np.float32), crs=crs, transform=ref_transform)

    # Template: 4x4 (different resolution/coords)
    tpl_transform = from_origin(0, 2, 0.5, 0.5)
    tpl_path = tmp_path / "tpl.tif"
    write_geotiff(tpl_path, np.zeros((4, 4), dtype=np.float32), crs=crs, transform=tpl_transform)
    template = rxr.open_rasterio(tpl_path).squeeze(drop=True)

    # Elevation returned by load_elevation: 3x3 on another grid
    elev_transform = from_origin(0, 3, 1, 1)
    elev_path = tmp_path / "elev.tif"
    write_geotiff(
        elev_path,
        (np.ones((3, 3), dtype=np.float32) * 100.0),
        crs=crs,
        transform=elev_transform,
    )
    elevation = rxr.open_rasterio(elev_path).squeeze(drop=True)

    monkeypatch.setattr("cleo.loaders.load_elevation", lambda base_dir, iso3, reference_da: elevation)

    self = _make_self(base_dir=tmp_path, country=country, crs=crs)
    self.data = xr.Dataset({"template": template})

    assess.compute_air_density_correction(self, chunk_size=None)

    out = self.data["air_density_correction"]
    assert_same_coords(out, template)


@pytest.mark.parametrize("user_input", ["EPSG:3035", 3035])
def test_crs_from_user_input_normalizes_to_expected(user_input) -> None:
    """We rely on CRS.from_user_input for robust comparison across string/int forms."""
    expected = CRS.from_epsg(3035)
    assert CRS.from_user_input(user_input) == expected


def test_crs_comparison_detects_actual_difference() -> None:
    """Different CRS should be detected as different."""
    elevation = da_xy_with_crs(values=np.ones((2, 2), dtype=float), n=2, name="elev", crs="EPSG:4326")
    assert elevation.rio.crs != CRS.from_user_input("EPSG:3035")


def test_compute_air_density_correction_with_str_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """compute_air_density_correction should work when parent.path is a str."""
    self = _make_self(base_dir=str(tmp_path), country="AUT", crs="EPSG:4326")

    # Reference file must exist (function validates presence)
    raw_dir = tmp_path / "data" / "raw" / "AUT"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "AUT_combined-Weibull-A_100.tif").touch()

    tiny = da_xy_with_crs(
        values=np.ones((3, 3), dtype=float) * 500.0,
        n=3,
        name="elevation",
        crs="EPSG:4326",
    )

    # Patch raster-open + load_elevation to avoid real IO while preserving behavior.
    monkeypatch.setattr("cleo.assess.rxr.open_rasterio", lambda *a, **k: tiny)
    monkeypatch.setattr("cleo.loaders.load_elevation", lambda base_dir, iso3, reference_da: tiny.copy())

    assess.compute_air_density_correction(self)

    assert "air_density_correction" in self.data.data_vars
    assert np.all(np.isfinite(self.data["air_density_correction"].values))


def test_compute_air_density_correction_missing_file_error(tmp_path: Path) -> None:
    """Missing raw GWA reference file must raise a clear, actionable FileNotFoundError."""
    self = _make_self(base_dir=str(tmp_path), country="AUT", crs="EPSG:4326")

    with pytest.raises(FileNotFoundError) as exc_info:
        assess.compute_air_density_correction(self)

    msg = str(exc_info.value)
    assert "Raw GWA files missing" in msg
    assert "AUT" in msg
