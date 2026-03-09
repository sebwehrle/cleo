"""Phase 5 loaders branch/error-path tests for fast coverage gains."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.helpers.optional import requires_rasterio, requires_rioxarray

requires_rasterio()
requires_rioxarray()

import rasterio
import cleo.loaders as L
from cleo.net import RequestException


def _write_tif(path: Path, data: np.ndarray, *, crs: str = "EPSG:3035") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = rasterio.transform.from_bounds(0, 0, 2, 2, data.shape[1], data.shape[0])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)


def test_load_yaml_file_raises_for_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yml"
    p.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        L._load_yaml_file(p, context="unit-test")


def test_get_cost_assumptions_missing_key_raises_keyerror(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    (resources / "cost_assumptions.yml").write_text("foo: 1\n", encoding="utf-8")
    parent = SimpleNamespace(path=tmp_path)
    self = SimpleNamespace(parent=parent)
    with pytest.raises(KeyError):
        L.get_cost_assumptions(self, "missing_key")


def test_get_powercurves_invalid_schema_raises(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    # Missing required key 'cf'
    (resources / "T0.yml").write_text("manufacturer: M\nmodel: X\ncapacity: 1\nV: [0]\n", encoding="utf-8")
    self = SimpleNamespace(path=tmp_path, wind_turbines=["T0"])
    with pytest.raises(ValueError, match="Invalid power curve YAML schema"):
        L.get_powercurves(self)


def test_get_turbine_attribute_error_branches() -> None:
    # Missing turbine dim
    self_no_dim = SimpleNamespace(
        data=xr.Dataset(coords={"x": [0], "y": [0]}),
    )
    with pytest.raises(ValueError, match="No turbines in dataset"):
        L.get_turbine_attribute(self_no_dim, "T1", "hub_height")

    # Missing cleo_turbines_json
    ds = xr.Dataset(
        {"turbine_hub_height": ("turbine", np.array([100.0], dtype=np.float64))},
        coords={"turbine": [0]},
    )
    self_no_attr = SimpleNamespace(data=ds)
    with pytest.raises(ValueError, match="missing cleo_turbines_json"):
        L.get_turbine_attribute(self_no_attr, "T1", "hub_height")


def test_get_turbine_attribute_unknown_turbine_and_missing_attr() -> None:
    ds = xr.Dataset(
        {"turbine_hub_height": ("turbine", np.array([100.0], dtype=np.float64))},
        coords={"turbine": [0]},
    )
    ds.attrs["cleo_turbines_json"] = '[{"id":"T1","manufacturer":"M","model":"X","model_key":"M.X.1"}]'
    self_obj = SimpleNamespace(data=ds)

    with pytest.raises(ValueError, match="not found in dataset"):
        L.get_turbine_attribute(self_obj, "T2", "hub_height")

    with pytest.raises(ValueError, match="not found in dataset for turbine_id"):
        L.get_turbine_attribute(self_obj, "T1", "rotor_diameter")


def test_get_overnight_cost_computes_numeric_value(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {
            "turbine_capacity": ("turbine", np.array([2000.0], dtype=np.float64)),
            "turbine_hub_height": ("turbine", np.array([80.0], dtype=np.float64)),
            "turbine_rotor_diameter": ("turbine", np.array([90.0], dtype=np.float64)),
            "turbine_commissioning_year": ("turbine", np.array([2010], dtype=np.int64)),
        },
        coords={"turbine": [0]},
    )
    ds.attrs["cleo_turbines_json"] = '[{"id":"T1","manufacturer":"M","model":"X","model_key":"M.X.2"}]'
    self_obj = SimpleNamespace(data=ds)
    self_obj.get_turbine_attribute = lambda turbine_id, attr: L.get_turbine_attribute(self_obj, turbine_id, attr)
    out = L.get_overnight_cost(self_obj, "T1")
    assert isinstance(out, (float, np.floating))


def test_load_air_density_wraps_open_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="EPSG:3035",
        area=None,
        get_nuts_area=lambda _r: None,
    )
    self_obj = SimpleNamespace(parent=parent)

    rho_path = tmp_path / "data" / "raw" / "AUT" / "AUT_air-density_100.tif"
    _write_tif(rho_path, np.ones((2, 2), dtype=np.float32))

    monkeypatch.setattr("cleo.loaders.rxr.open_rasterio", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(RuntimeError, match="Failed to load air density"):
        L.load_air_density(self_obj, 100)


def test_load_weibull_parameters_wraps_reprojection_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="EPSG:3035",
        area=None,
        get_nuts_area=lambda _r: None,
    )
    self_obj = SimpleNamespace(parent=parent)

    a_path = tmp_path / "data" / "raw" / "AUT" / "AUT_combined-Weibull-A_100.tif"
    k_path = tmp_path / "data" / "raw" / "AUT" / "AUT_combined-Weibull-k_100.tif"
    _write_tif(a_path, np.ones((2, 2), dtype=np.float32))
    _write_tif(k_path, np.ones((2, 2), dtype=np.float32))

    monkeypatch.setattr(
        "cleo.loaders.reproject_raster_if_needed", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(RuntimeError, match="Failed to load Weibull parameters"):
        L.load_weibull_parameters(self_obj, 100)


def test_load_gwa_skips_existing_files_and_handles_download_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = SimpleNamespace(path=tmp_path, country="AUT")
    self_obj = SimpleNamespace(parent=parent)
    raw_dir = tmp_path / "data" / "raw" / "AUT"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create one file so skip path is exercised.
    existing = raw_dir / "AUT_air-density_10.tif"
    existing.write_text("x", encoding="utf-8")

    calls: list[Path] = []

    def _fake_download(url, out_path, **kwargs):  # noqa: ANN001, ARG001
        out = Path(out_path)
        calls.append(out)
        if out.name.endswith("combined-Weibull-k_50.tif"):
            raise RequestException("network")
        out.touch()
        return out

    monkeypatch.setattr("cleo.loaders.download_to_path", _fake_download)
    L.load_gwa(self_obj)
    assert existing.exists()
    assert existing not in calls
    assert any(p.name.endswith("combined-Weibull-k_50.tif") for p in calls)
    assert any(p.exists() for p in calls if not p.name.endswith("combined-Weibull-k_50.tif"))
