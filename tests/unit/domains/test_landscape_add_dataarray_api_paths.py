"""Unit tests for LandscapeDomain add_dataarray API routing and staging paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cleo.domains as domains_module
from cleo.domains import LandscapeAddResult, LandscapeDomain
from tests.helpers.domains import make_landscape_domain_atlas_stub


def test_add_dataarray_routes_to_registration_and_stages_overlay(
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

    source_da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        name="raw_slope",
    )
    staged_da = source_da.rename("slope")
    calls: dict[str, object] = {}

    def _register_landscape_dataarray_source(atlas_obj, **kwargs):  # noqa: ANN001
        calls["atlas"] = atlas_obj
        calls["register_kwargs"] = kwargs
        return True, staged_da

    monkeypatch.setattr(domains_module, "register_landscape_dataarray_source", _register_landscape_dataarray_source)

    op = domain.add_dataarray(
        "slope",
        source_da,
        categorical=True,
        if_exists="replace",
    )

    assert isinstance(op, LandscapeAddResult)
    assert op.data.name == "slope"
    assert calls["atlas"] is atlas
    assert calls["register_kwargs"] == {
        "name": "slope",
        "data": source_da,
        "categorical": True,
        "if_exists": "replace",
        "chunk_policy": atlas.chunk_policy,
    }
    assert "slope" in domain.data.data_vars


def test_add_dataarray_noop_with_existing_store_var_returns_existing_without_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)

    existing = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        name="slope",
    )
    base = xr.Dataset(
        data_vars={
            "valid_mask": xr.DataArray(np.ones((2, 2), dtype=bool), dims=("y", "x")),
            "slope": existing,
        },
        coords={"y": [0, 1], "x": [0, 1]},
    )
    monkeypatch.setattr(domain, "_store_data", lambda: base)
    monkeypatch.setattr(
        domains_module,
        "register_landscape_dataarray_source",
        lambda atlas_obj, **kwargs: (False, existing),  # noqa: ARG005
    )

    source_da = xr.DataArray(
        np.full((2, 2), 5.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        name="candidate",
    )

    op = domain.add_dataarray("slope", source_da, if_exists="noop")

    assert isinstance(op, LandscapeAddResult)
    assert op.data.identical(existing)
    assert domain._staged_overlays == {}  # noqa: SLF001


def test_add_dataarray_rejects_non_dataarray(tmp_path: Path) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)

    with pytest.raises(TypeError, match="xarray.DataArray"):
        domain.add_dataarray("slope", np.ones((2, 2), dtype=np.float32))  # type: ignore[arg-type]


def test_materialize_staged_reuses_prepared_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = make_landscape_domain_atlas_stub(tmp_path)
    domain = LandscapeDomain(atlas)
    staged = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
        name="slope",
    )
    domain._staged_overlays["slope"] = staged  # noqa: SLF001
    domain._staged_prepared_overlays["slope"] = staged  # noqa: SLF001

    calls: dict[str, object] = {}

    def _materialize_landscape_variable(atlas_obj, variable_name, **kwargs):  # noqa: ANN001
        calls["atlas"] = atlas_obj
        calls["variable_name"] = variable_name
        calls["prepared_da"] = kwargs.get("prepared_da")
        return True

    monkeypatch.setattr(domains_module, "materialize_landscape_variable", _materialize_landscape_variable)
    monkeypatch.setattr(
        domain,
        "_store_data",
        lambda: xr.Dataset(
            data_vars={
                "valid_mask": xr.DataArray(np.ones((2, 2), dtype=bool), dims=("y", "x")),
                "slope": staged,
            },
            coords={"y": [0, 1], "x": [0, 1]},
        ),
    )

    domain._materialize_staged(name="slope", if_exists="replace")  # noqa: SLF001

    assert calls["atlas"] is atlas
    assert calls["variable_name"] == "slope"
    assert calls["prepared_da"] is staged
