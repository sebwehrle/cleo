"""classes: test_loading_contracts.

Covers:
- load_and_extract_from_dict: must not change cwd; must reject zip-slip paths
- Atlas.load: must fall back to legacy filenames when no timestamped files exist
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import cleo.classes as C


def _mini_ds() -> xr.Dataset:
    return xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )


def _mk_landscape_atlas_stub(tmp_path: Path) -> C._LandscapeAtlas:
    """
    Create a minimal _LandscapeAtlas instance without heavy __init__.
    Ensure any optional proxy attributes exist for download/extract flows.
    """
    atlas = C._LandscapeAtlas.__new__(C._LandscapeAtlas)
    atlas.path = Path(tmp_path)
    atlas.parent = SimpleNamespace(proxy=None, proxy_user=None, proxy_pass=None)
    return atlas


def test_load_and_extract_from_dict_does_not_change_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare a zip file we can "download"
    src_zip = tmp_path / "src.zip"
    with zipfile.ZipFile(src_zip, "w") as z:
        z.writestr("ok.txt", "hello")

    # Monkeypatch download_file to copy local zip
    def fake_download_file(url, dest, proxy=None, proxy_user=None, proxy_pass=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(url).read_bytes())
        return True

    monkeypatch.setattr(C, "download_file", fake_download_file, raising=True)

    atlas = _mk_landscape_atlas_stub(tmp_path)

    cwd_before = os.getcwd()
    dest_dir = tmp_path / "dest"

    atlas.load_and_extract_from_dict({"downloaded.zip": (str(dest_dir), str(src_zip))})

    assert os.getcwd() == cwd_before
    assert (dest_dir / "ok.txt").is_file()


def test_load_and_extract_from_dict_rejects_zip_slip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("../evil.txt", "x")

    def fake_download_file(url, dest, proxy=None, proxy_user=None, proxy_pass=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(url).read_bytes())
        return True

    monkeypatch.setattr(C, "download_file", fake_download_file, raising=True)

    atlas = _mk_landscape_atlas_stub(tmp_path)

    with pytest.raises(ValueError, match="Unsafe zip member path"):
        atlas.load_and_extract_from_dict({"downloaded.zip": (str(tmp_path / "dest2"), str(bad_zip))})


class _DummySub:
    def __init__(self) -> None:
        self.data = None


def test_load_falls_back_to_legacy_files(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # legacy names expected by code when no timestamped matches exist
    ds = _mini_ds()
    ds.to_netcdf(processed / "WindAtlas_AUT.nc")
    ds.close()

    ds = _mini_ds()
    ds.to_netcdf(processed / "LandscapeAtlas_AUT.nc")
    ds.close()

    a = C.Atlas.__new__(C.Atlas)
    a._path = Path(tmp_path)
    a.country = "AUT"
    a.region = None

    # Satisfy materialization guards without invoking heavy materialize()
    a._materialized = True
    a._wind = _DummySub()
    a._landscape = _DummySub()

    a.load(region=None, scenario="default", timestamp="latest")

    assert a.wind.data is not None
    assert a.landscape.data is not None


def test_atlasdatavarsettermixin_crs_property_delegates() -> None:
    """crs property on mixin must delegate to parent.crs."""
    from cleo.classes import _AtlasDataVarSetterMixin

    class Dummy(_AtlasDataVarSetterMixin):
        pass

    d = Dummy()
    d.parent = SimpleNamespace(crs="EPSG:4326")
    d.data = None

    assert d.crs == "EPSG:4326"


def test_utils_add_clip_uses_to_crs_and_robust_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    utils.add must use to_crs_if_needed and _rio_clip_robust when clipping.
    Verify geometry is reprojected to raster CRS before clipping.
    """
    import geopandas as gpd
    from shapely.geometry import box
    import cleo.utils as utils
    import cleo.spatial

    # Create template and data var with CRS
    tpl = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
    ).rio.write_crs("EPSG:4326")

    a = xr.DataArray(
        np.ones((3, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
    ).rio.write_crs("EPSG:4326")

    ds = xr.Dataset({"template": tpl, "a": a})

    # Clip geometry in EPSG:3857 (intentionally mismatched CRS)
    # Build box in EPSG:4326 bounds then convert to 3857
    gdf_4326 = gpd.GeoDataFrame(
        geometry=[box(-0.5, -0.5, 2.5, 2.5)],
        crs="EPSG:4326",
    )
    gdf_3857 = gdf_4326.to_crs("EPSG:3857")

    # Create other DataArray with same coords as template (will be matched anyway)
    other = xr.DataArray(
        np.ones((3, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        name="b",
    ).rio.write_crs("EPSG:4326")

    # Build dummy self object
    dummy = SimpleNamespace(
        data=ds,
        crs="EPSG:4326",
        parent=SimpleNamespace(
            region="TestRegion",
            crs="EPSG:4326",
            get_nuts_region=lambda r: gdf_3857,
            get_nuts_country=lambda: gdf_3857,
        ),
    )

    # Spy on _rio_clip_robust
    clip_calls = []

    def spy_clip_robust(da, geoms, *, drop, all_touched_primary=False):
        clip_calls.append({"drop": drop, "geoms_is_list": isinstance(geoms, list)})
        # Return input unchanged (geometry covers all data)
        return da

    monkeypatch.setattr(utils, "_rio_clip_robust", spy_clip_robust)

    # Mock bbox to return different values to force clip path
    call_count = [0]

    def mock_bbox(obj):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: bbox(self) - return self bbox
            return (0.0, 0.0, 2.0, 2.0)
        else:
            # Second call: bbox(other) - return different bbox to trigger clip
            return (-1.0, -1.0, 3.0, 3.0)

    monkeypatch.setattr(utils, "bbox", mock_bbox)

    utils.add(dummy, other, name="b")

    # Assertions
    assert "b" in dummy.data.data_vars, "data var 'b' should be added"
    assert len(clip_calls) == 1, f"Expected 1 clip call, got {len(clip_calls)}"
    assert clip_calls[0]["drop"] is False, "drop should be False"
    assert clip_calls[0]["geoms_is_list"] is True, "geoms should be a list"


def test_load_and_extract_from_dict_badzip_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    When a downloaded file is not a valid ZIP, error message must include:
    - file name
    - download_path
    - url
    so user can immediately inspect the offending file.
    """
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Create a fake "zip" that is actually HTML (simulating server error page)
    src = tmp_path / "not_a_zip.zip"
    src.write_bytes(b"<html>no</html>")

    def fake_download_file(url, dest, proxy=None, proxy_user=None, proxy_pass=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(src.read_bytes())
        return True

    monkeypatch.setattr(C, "download_file", fake_download_file, raising=True)

    atlas = _mk_landscape_atlas_stub(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        atlas.load_and_extract_from_dict({
            "downloaded.zip": (str(dest_dir), "http://example.com/downloaded.zip")
        })

    msg = str(exc_info.value)
    # Error must contain actionable info
    assert "downloaded.zip" in msg, "Error should include file name"
    assert "http://example.com/downloaded.zip" in msg, "Error should include URL"
    assert str(dest_dir) in msg, "Error should include download path"
