"""tests/classes: test_windatlas_schema.

Contracts:
- _ensure_windatlas_schema adds wind_speed coord if missing (canonical grid 0..40 step 0.5).
- _ensure_windatlas_schema drops stray 'band' coord/var/dim.
- After schema migration, attrs["cleo_schema_version"] == "windatlas_v2".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("rioxarray")
import rioxarray  # noqa: F401


def test_windatlas_migrates_missing_wind_speed_and_drops_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Arrange:
    - Create a legacy-like WindAtlas_AUT.nc with coords x,y, plus a stray coord 'band' (not dim), and template var.
    - Monkeypatch load_gwa to no-op to avoid network.
    - Instantiate Atlas(tmp_path, "AUT", "epsg:31287") then atlas.materialize()

    Assert:
    - "wind_speed" in atlas.wind.data.coords and equals canonical u
    - "band" not in coords/dims
    - attrs["cleo_schema_version"] == "windatlas_v2"
    """
    from cleo.classes import Atlas

    # Create directory structure
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    raw_dir = tmp_path / "data" / "raw" / "AUT"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal legacy-like WindAtlas NetCDF with:
    # - x, y coords
    # - a stray 'band' coord (not a dim)
    # - template data var
    # - no wind_speed coord
    x_coords = np.linspace(100000.0, 200000.0, 4)
    y_coords = np.linspace(300000.0, 400000.0, 4)
    template_data = np.zeros((4, 4), dtype=np.float32)

    ds = xr.Dataset(
        {"template": (("y", "x"), template_data)},
        coords={
            "x": x_coords,
            "y": y_coords,
            "band": 1,  # stray coord (not a dim)
        },
    )
    ds = ds.rio.write_crs("EPSG:31287")

    legacy_nc_path = processed / "WindAtlas_AUT.nc"
    ds.to_netcdf(legacy_nc_path)
    ds.close()

    # Also create a matching LandscapeAtlas (required for full Atlas.materialize())
    landscape_ds = xr.Dataset(
        {"template": (("y", "x"), template_data.copy())},
        coords={"x": x_coords, "y": y_coords},
    )
    landscape_ds = landscape_ds.rio.write_crs("EPSG:31287")
    landscape_nc_path = processed / "LandscapeAtlas_AUT.nc"
    landscape_ds.to_netcdf(landscape_nc_path)
    landscape_ds.close()

    # Monkeypatch loaders to avoid network calls
    import cleo.loaders
    from cleo.classes import _WindAtlas, _LandscapeAtlas

    monkeypatch.setattr(cleo.loaders, "ensure_crs_from_gwa", lambda dset, iso3: dset)
    # Patch class methods directly (module-level patches don't affect already-bound methods)
    monkeypatch.setattr(_WindAtlas, "_load_gwa", lambda self: None)
    monkeypatch.setattr(_LandscapeAtlas, "_load_nuts", lambda self: None)

    # Instantiate and materialize Atlas
    atlas = Atlas(tmp_path, "AUT", "epsg:31287")
    atlas.materialize()

    # Assertions
    assert "wind_speed" in atlas.wind.data.coords, "wind_speed coord should be added by schema migration"

    expected_u = np.arange(0.0, 40.0 + 0.5, 0.5)
    actual_u = atlas.wind.data.coords["wind_speed"].values
    np.testing.assert_array_equal(
        actual_u, expected_u, err_msg="wind_speed grid does not match canonical 0..40 step 0.5"
    )

    assert "band" not in atlas.wind.data.coords, "'band' coord should be dropped"
    assert "band" not in atlas.wind.data.dims, "'band' dim should not exist"

    assert atlas.wind.data.attrs.get("cleo_schema_version") == "windatlas_v2", (
        f"schema version mismatch: {atlas.wind.data.attrs.get('cleo_schema_version')}"
    )
