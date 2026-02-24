from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cleo.domains import LandscapeComputeBatchResult, LandscapeDomain
from tests.helpers.domains import make_landscape_domain_atlas_stub


def _with_crs(da: xr.DataArray, crs: str = "epsg:3035") -> xr.DataArray:
    return da.rio.write_crs(crs)


def _base_store() -> xr.Dataset:
    y = [0.0, 1.0, 2.0]
    x = [0.0, 1.0, 2.0]
    valid_mask = _with_crs(
        xr.DataArray(np.ones((3, 3), dtype=bool), dims=("y", "x"), coords={"y": y, "x": x}, name="valid_mask")
    )
    roads = _with_crs(
        xr.DataArray(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="roads",
        )
    )
    water = _with_crs(
        xr.DataArray(
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="water",
        )
    )
    return xr.Dataset(data_vars={"valid_mask": valid_mask, "roads": roads, "water": water}, coords={"y": y, "x": x})


def test_compute_distance_batch_stages_and_returns_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    store = _base_store()
    monkeypatch.setattr(domain, "_store_data", lambda: store)

    result = domain.compute(
        "distance",
        source=["roads", "water"],
        name=["distance_roads", "distance_water"],
    )

    assert isinstance(result, LandscapeComputeBatchResult)
    assert list(result.data.data_vars) == ["distance_roads", "distance_water"]
    assert "distance_roads" in domain._staged_overlays  # noqa: SLF001
    assert "distance_water" in domain._staged_overlays  # noqa: SLF001
    assert result.data["distance_roads"].attrs["cleo:distance_source"] == "roads"
    assert result.data["distance_water"].attrs["cleo:distance_source"] == "water"


def test_compute_distance_rejects_source_not_in_store_even_if_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    store = _base_store()[["valid_mask"]]
    monkeypatch.setattr(domain, "_store_data", lambda: store)
    domain._staged_overlays["roads"] = _base_store()["roads"]  # noqa: SLF001

    with pytest.raises(ValueError, match="Unknown distance source variable"):
        domain.compute("distance", source="roads")


def test_compute_distance_if_exists_error_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    store = _base_store()
    existing = store["roads"].copy().rename("distance_roads")
    existing.attrs["cleo:distance_spec_json"] = LandscapeDomain._distance_spec_json("roads")
    store = store.assign({"distance_roads": existing})
    monkeypatch.setattr(domain, "_store_data", lambda: store)

    with pytest.raises(ValueError, match="would overwrite existing variable"):
        domain.compute(
            "distance",
            source=["roads", "water"],
            name=["distance_roads", "distance_water"],
            if_exists="error",
        )

    assert "distance_water" not in domain._staged_overlays  # noqa: SLF001


def test_compute_distance_noop_requires_matching_spec_for_existing_store_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    store = _base_store()
    existing = store["roads"].copy().rename("distance_roads")
    existing.attrs["cleo:distance_spec_json"] = LandscapeDomain._distance_spec_json("water")
    store = store.assign({"distance_roads": existing})
    monkeypatch.setattr(domain, "_store_data", lambda: store)

    with pytest.raises(ValueError, match="different distance spec"):
        domain.compute("distance", source="roads", if_exists="noop")


def test_compute_distance_noop_uses_store_var_without_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    store = _base_store()
    existing = store["roads"].copy().rename("distance_roads")
    existing.attrs["cleo:distance_spec_json"] = LandscapeDomain._distance_spec_json("roads")
    store = store.assign({"distance_roads": existing})
    monkeypatch.setattr(domain, "_store_data", lambda: store)

    result = domain.compute("distance", source="roads", if_exists="noop")
    assert "distance_roads" in result.data.data_vars
    assert "distance_roads" not in domain._staged_overlays  # noqa: SLF001


def test_compute_distance_result_materialize_routes_to_batch_materializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    store = _base_store()
    monkeypatch.setattr(domain, "_store_data", lambda: store)

    calls: dict[str, object] = {}

    def _fake_materialize(*, names: tuple[str, ...], if_exists: str) -> xr.Dataset:
        calls["names"] = names
        calls["if_exists"] = if_exists
        return xr.Dataset(
            {"distance_roads": store["roads"]},
            coords={"y": store.coords["y"], "x": store.coords["x"]},
        )

    monkeypatch.setattr(domain, "_materialize_staged_batch", _fake_materialize)

    result = domain.compute("distance", source="roads")
    out = result.materialize(if_exists="replace")

    assert list(out.data_vars) == ["distance_roads"]
    assert calls["names"] == ("distance_roads",)
    assert calls["if_exists"] == "replace"
