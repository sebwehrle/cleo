"""Unit tests for LandscapeDomain rasterize API routing and staging paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cleo.domains import LandscapeAddResult, LandscapeDomain
from tests.helpers.domains import make_landscape_domain_atlas_stub


def test_rasterize_routes_to_vector_registration_and_stages_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)

    base = xr.Dataset(
        data_vars={
            "valid_mask": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
            )
        },
        coords={"y": [0, 1], "x": [0, 1]},
    )
    monkeypatch.setattr(domain, "_store_data", lambda: base)

    staged_da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        name="overnight_stays",
    )

    calls: dict[str, object] = {}

    class _StubUnifier:
        def register_landscape_vector_source(self, atlas_obj, **kwargs):  # noqa: ANN001
            calls["atlas"] = atlas_obj
            calls["register_kwargs"] = kwargs
            return True

        def prepare_landscape_variable_data(self, atlas_obj, variable_name):  # noqa: ANN001
            calls["prepare_atlas"] = atlas_obj
            calls["prepare_variable_name"] = variable_name
            return staged_da

    monkeypatch.setattr(domain, "_build_unifier", lambda: _StubUnifier())

    op = domain.rasterize(
        "dummy.geojson",
        name="overnight_stays",
        column="overnight_stays",
        all_touched=True,
        if_exists="replace",
    )

    assert isinstance(op, LandscapeAddResult)
    assert op.data.name == "overnight_stays"
    assert calls["atlas"] is atlas
    assert calls["prepare_atlas"] is atlas
    assert calls["prepare_variable_name"] == "overnight_stays"
    assert calls["register_kwargs"] == {
        "name": "overnight_stays",
        "shape": "dummy.geojson",
        "column": "overnight_stays",
        "all_touched": True,
        "if_exists": "replace",
    }
    assert "overnight_stays" in domain.data.data_vars


def test_rasterize_noop_with_existing_store_var_returns_existing_without_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)

    existing = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        name="overnight_stays",
    )
    base = xr.Dataset(
        data_vars={
            "valid_mask": xr.DataArray(np.ones((2, 2), dtype=bool), dims=("y", "x")),
            "overnight_stays": existing,
        },
        coords={"y": [0, 1], "x": [0, 1]},
    )
    monkeypatch.setattr(domain, "_store_data", lambda: base)

    calls = {"prepare": 0}

    class _StubUnifier:
        def register_landscape_vector_source(self, atlas_obj, **kwargs):  # noqa: ANN001
            del atlas_obj, kwargs
            return False

        def prepare_landscape_variable_data(self, atlas_obj, variable_name):  # noqa: ANN001
            del atlas_obj, variable_name
            calls["prepare"] += 1
            raise AssertionError("prepare_landscape_variable_data should not be called")

    monkeypatch.setattr(domain, "_build_unifier", lambda: _StubUnifier())

    op = domain.rasterize(
        "dummy.geojson",
        name="overnight_stays",
        column="overnight_stays",
        if_exists="noop",
    )

    assert isinstance(op, LandscapeAddResult)
    assert op.data.identical(existing)
    assert calls["prepare"] == 0


def test_rasterize_error_when_variable_already_staged(tmp_path: Path) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    domain._store_data = lambda: xr.Dataset(  # noqa: SLF001
        data_vars={"valid_mask": xr.DataArray(np.ones((1, 1), dtype=bool), dims=("y", "x"))}
    )
    domain._staged_overlays["overnight_stays"] = xr.DataArray(  # noqa: SLF001
        np.ones((1, 1), dtype=np.float32),
        dims=("y", "x"),
    )

    with pytest.raises(ValueError, match="already staged"):
        domain.rasterize(
            "dummy.geojson",
            name="overnight_stays",
            column="overnight_stays",
            if_exists="error",
        )


def test_add_rejects_non_raster_kind_with_rasterize_hint(tmp_path: Path) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    with pytest.raises(ValueError, match="Use atlas\\.landscape\\.rasterize"):
        domain.add(
            "overnight_stays",
            tmp_path / "dummy.tif",
            kind="vector",
        )
