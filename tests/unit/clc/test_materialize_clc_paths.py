"""Unit tests for CLC materialize fast and error paths."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import xarray as xr

from tests.helpers.clc_fixtures import (
    JsonResponseStub,
    legacy_service_key,
    oauth_service_key,
    write_nested_zip_with_member,
    write_zip_with_member,
)
from tests.helpers.optional import requires_rasterio, requires_rioxarray
from cleo.clc import (
    CLC_SOURCES,
    _resolve_default_clc_download,
    materialize_clc,
    prepared_country_path,
)

rasterio = requires_rasterio()
rxr = requires_rioxarray()


def _write_raster_integer(path: Path, data: np.ndarray) -> None:
    """Write a small integer GeoTIFF test raster in EPSG:3035."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = rasterio.transform.from_bounds(0.0, 0.0, 2.0, 2.0, data.shape[1], data.shape[0])
    profile = {
        "driver": "GTiff",
        "dtype": str(data.dtype),
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": rasterio.crs.CRS.from_epsg(3035),
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_raster_non_georef(path: Path, data: np.ndarray) -> None:
    """Write integer GeoTIFF without CRS/transform metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": str(data.dtype),
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)


def _write_raster_multiband(path: Path, data: np.ndarray) -> None:
    """Write a multiband GeoTIFF for rendered-imagery guard tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = rasterio.transform.from_bounds(0.0, 0.0, 2.0, 2.0, data.shape[2], data.shape[1])
    profile = {
        "driver": "GTiff",
        "dtype": str(data.dtype),
        "width": data.shape[2],
        "height": data.shape[1],
        "count": data.shape[0],
        "crs": rasterio.crs.CRS.from_epsg(3035),
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_canonical_landscape_store(tmp_path: Path, ref: xr.DataArray) -> None:
    """Write a minimal canonical landscape store aligned to a wind reference."""
    y = np.asarray(ref.coords["y"].values, dtype=np.float64)
    x = np.asarray(ref.coords["x"].values, dtype=np.float64)
    valid_mask = np.asarray(np.isfinite(ref.values), dtype=bool)
    elevation = np.where(valid_mask, 1.0, np.nan).astype(np.float32)
    ds = xr.Dataset(
        data_vars={
            "valid_mask": (("y", "x"), valid_mask),
            "elevation": (("y", "x"), elevation),
        },
        coords={"y": y, "x": x},
        attrs={"store_state": "complete", "grid_id": "test-grid", "inputs_id": "test-land"},
    ).chunk({"y": min(2, valid_mask.shape[0]), "x": min(2, valid_mask.shape[1])})
    ds.to_zarr(tmp_path / "landscape.zarr", mode="w", consolidated=False)


def _atlas_stub(
    tmp_path: Path,
    *,
    canonical_ready: bool = True,
    build_canonical=None,  # noqa: ANN001
):
    """Create a minimal atlas-like object for CLC unit tests."""
    callback = build_canonical if build_canonical is not None else (lambda: None)
    return SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=canonical_ready,
        build_canonical=callback,
    )


def test_materialize_clc_returns_cached_prepared_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepared_country_path(tmp_path, "AUT", "clc2018")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text("cached", encoding="utf-8")

    called = {"canonical": 0}
    atlas = _atlas_stub(
        tmp_path,
        canonical_ready=False,
        build_canonical=lambda: called.__setitem__("canonical", called["canonical"] + 1),
    )

    out = materialize_clc(atlas, source="clc2018", force_prepare=False)
    assert out == prepared
    assert called["canonical"] == 0


def test_materialize_clc_raises_when_no_valid_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    _write_raster_integer(
        source_path,
        np.array(
            [
                [231, 231],
                [311, 231],
            ],
            dtype=np.uint16,
        ),
    )

    atlas = _atlas_stub(tmp_path)

    # Wind template with all-NaN => no valid cells.
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([10.0, 11.0], dtype=np.float64)
    ref = xr.DataArray(
        np.full((2, 2), np.nan, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")

    clc = xr.DataArray(
        np.full((2, 2), 231.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="clc",
    ).rio.write_crs("EPSG:3035")

    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)

    monkeypatch.setattr(
        "cleo.unification.clc_io.rxr.open_rasterio",
        lambda *a, **k: clc,
    )

    with pytest.raises(RuntimeError, match="No valid wind cells found"):
        materialize_clc(atlas, source="clc2018")


def test_materialize_clc_raises_for_unknown_source(tmp_path: Path) -> None:
    atlas = _atlas_stub(tmp_path)
    with pytest.raises(ValueError, match="Unsupported CLC source"):
        materialize_clc(atlas, source="clc2099")


def test_materialize_clc_accepts_integer_source_raster_with_nan_nodata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    _write_raster_integer(
        source_path,
        np.array(
            [
                [231, 231],
                [311, 231],
            ],
            dtype=np.uint16,
        ),
    )

    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.array([[1.0, np.nan], [1.0, 1.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([1.5, 0.5], dtype=np.float64),
            "x": np.array([0.5, 1.5], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)

    out = materialize_clc(atlas, source="clc2018")
    assert out.exists()

    prepared = rxr.open_rasterio(out).squeeze(drop=True)
    assert prepared.dtype == np.float32
    assert np.isnan(prepared.values[0, 1])


def test_materialize_clc_expands_compact_clc_ids_to_canonical_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    _write_raster_integer(
        source_path,
        np.array(
            [
                [23, 24],
                [18, 12],
            ],
            dtype=np.uint8,
        ),
    )
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    (resources / "clc_codes.yml").write_text(
        Path("cleo/resources/clc_codes.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([1.5, 0.5], dtype=np.float64),
            "x": np.array([0.5, 1.5], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)

    out = materialize_clc(atlas, source="clc2018")
    prepared = rxr.open_rasterio(out).squeeze(drop=True)
    np.testing.assert_allclose(prepared.values, np.array([[311.0, 312.0], [231.0, 211.0]], dtype=np.float32))


def test_resolve_default_clc_download_uses_clms_api_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use an allowlisted CLMS fixture while exercising the API download flow."""
    monkeypatch.setenv(
        "CLEO_CLMS_SERVICE_KEY_JSON",
        json.dumps(
            legacy_service_key(
                service_name="example-service",
                secret="not-a-real-secret",  # pragma: allowlist secret
                username="example-user",
            )
        ),  # pragma: allowlist secret
    )

    def _fake_post(url: str, **kwargs):
        if url.endswith("/@login"):
            assert kwargs["json_body"]["service_name"] == "example-service"
            return JsonResponseStub(payload={"access_token": "token-123"})
        if url.endswith("/@datarequest_post"):
            assert kwargs["headers"]["Authorization"] == "Bearer token-123"
            assert kwargs["json_body"]["Datasets"] == [{"DatasetID": "dataset-4368", "FileID": "file-31902"}]
            return JsonResponseStub(payload={"TaskIds": [{"TaskID": "request-token"}]})
        raise AssertionError(f"unexpected POST url: {url}")

    def _fake_get(url: str, **kwargs):
        if "/@search?" in url:
            return JsonResponseStub(
                payload={
                    "items": [
                        {
                            "UID": "dataset-4368",
                            "downloadable_files": {
                                "items": [
                                    {
                                        "@id": "file-31902",
                                        "file": "u2018_clc2018_v2020_20u1_raster100m",
                                        "format": "Geotiff",
                                        "path": "H:\\CLC\\u2018_clc2018_v2020_20u1_raster100m.zip",
                                        "type": "Raster",
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        if url.endswith("/@datarequest_status_get?TaskID=request-token"):
            return JsonResponseStub(
                payload={
                    "Status": "finished_ok",
                    "DownloadURL": ["https://copernicus-fme.eea.europa.eu/fmedatadownload/results/123.zip"],
                }
            )
        raise AssertionError(f"unexpected GET url: {url}")

    monkeypatch.setattr("cleo.clc.http_post", _fake_post)
    monkeypatch.setattr("cleo.clc.http_get", _fake_get)

    url, headers = _resolve_default_clc_download("clc2018")
    assert url == "https://copernicus-fme.eea.europa.eu/fmedatadownload/results/123.zip"
    assert headers == {"Authorization": "Bearer token-123"}


def test_resolve_default_clc_download_accepts_current_copernicus_service_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the current Copernicus key schema to resolve a CLMS download."""
    token_uri = "https://land.copernicus.eu/@@oauth2-token"
    monkeypatch.setenv(
        "CLEO_CLMS_SERVICE_KEY_JSON",
        json.dumps(
            oauth_service_key(
                client_id="example-client-id",
                key_id="example-key-id",
                private_key="-----BEGIN PRIVATE KEY-----\nexample\n-----END PRIVATE KEY-----",  # pragma: allowlist secret
                token_uri=token_uri,
                user_id="example-user-id",
                ip_range="*",
                issued="2026-03-08T00:00:00Z",
                title="example",
            )
        ),
    )

    def _fake_post(url: str, **kwargs):
        if url == token_uri:
            assert kwargs["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
            assert kwargs["data_body"]["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
            assert kwargs["data_body"]["assertion"] == "signed-jwt"
            assert kwargs.get("json_body") is None
            return JsonResponseStub(payload={"access_token": "oauth-token-123"})
        if url.endswith("/@datarequest_post"):
            assert kwargs["headers"]["Authorization"] == "Bearer oauth-token-123"
            assert kwargs["json_body"]["Datasets"] == [{"DatasetID": "dataset-4368", "FileID": "file-31902"}]
            return JsonResponseStub(payload={"TaskIds": [{"TaskID": "request-token"}]})
        raise AssertionError(f"unexpected POST url: {url}")

    def _fake_get(url: str, **kwargs):
        if "/@search?" in url:
            return JsonResponseStub(
                payload={
                    "items": [
                        {
                            "UID": "dataset-4368",
                            "downloadable_files": {
                                "items": [
                                    {
                                        "@id": "file-31902",
                                        "file": "u2018_clc2018_v2020_20u1_raster100m",
                                        "format": "Geotiff",
                                        "path": "H:\\CLC\\u2018_clc2018_v2020_20u1_raster100m.zip",
                                        "type": "Raster",
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        if url.endswith("/@datarequest_status_get?TaskID=request-token"):
            return JsonResponseStub(
                payload={
                    "Status": "finished_ok",
                    "DownloadURL": ["https://copernicus-fme.eea.europa.eu/fmedatadownload/results/456.zip"],
                }
            )
        raise AssertionError(f"unexpected GET url: {url}")

    monkeypatch.setattr("cleo.clc._sign_clms_oauth_assertion", lambda **kwargs: "signed-jwt")
    monkeypatch.setattr("cleo.clc.http_post", _fake_post)
    monkeypatch.setattr("cleo.clc.http_get", _fake_get)

    url, headers = _resolve_default_clc_download("clc2018")
    assert url == "https://copernicus-fme.eea.europa.eu/fmedatadownload/results/456.zip"
    assert headers == {"Authorization": "Bearer oauth-token-123"}


def test_resolve_default_clc_download_falls_back_to_finished_request_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CLEO_CLMS_SERVICE_KEY_JSON",
        json.dumps(
            legacy_service_key(
                service_name="example-service",
                secret="not-a-real-secret",  # pragma: allowlist secret
                username="example-user",
            )
        ),  # pragma: allowlist secret
    )

    def _fake_post(url: str, **kwargs):
        if url.endswith("/@login"):
            return JsonResponseStub(payload={"access_token": "token-123"})
        if url.endswith("/@datarequest_post"):
            return JsonResponseStub(payload={"TaskIds": [{"TaskID": "request-token"}]})
        raise AssertionError(f"unexpected POST url: {url}")

    def _fake_get(url: str, **kwargs):
        if "/@search?" in url:
            return JsonResponseStub(
                payload={
                    "items": [
                        {
                            "UID": "dataset-4368",
                            "downloadable_files": {
                                "items": [
                                    {
                                        "@id": "file-31902",
                                        "file": "u2018_clc2018_v2020_20u1_raster100m",
                                        "format": "Geotiff",
                                        "path": "H:\\CLC\\u2018_clc2018_v2020_20u1_raster100m.zip",
                                        "type": "Raster",
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        if url.endswith("/@datarequest_status_get?TaskID=request-token"):
            return JsonResponseStub(payload={"Status": "finished_ok"})
        if "/@datarequest_search?" in url:
            return JsonResponseStub(
                payload={
                    "items": [
                        {
                            "TaskID": "request-token",
                            "DownloadURL": ["https://copernicus-fme.eea.europa.eu/fmedatadownload/results/789.zip"],
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected GET url: {url}")

    monkeypatch.setattr("cleo.clc.http_post", _fake_post)
    monkeypatch.setattr("cleo.clc.http_get", _fake_get)

    url, headers = _resolve_default_clc_download("clc2018")
    assert url == "https://copernicus-fme.eea.europa.eu/fmedatadownload/results/789.zip"
    assert headers == {"Authorization": "Bearer token-123"}


def test_materialize_clc_uses_clms_default_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")

    captured: dict[str, str | Path | dict[str, str] | None] = {}
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.tif", {"Authorization": "Bearer test-token"}),
    )

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        captured["url"] = url
        captured["source_path"] = out_path
        captured["headers"] = kwargs.get("headers")
        out_path.write_text("source", encoding="utf-8")

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr(
        "cleo.clc.prepare_clc_to_wind_grid",
        lambda **kwargs: kwargs["prepared_path"],
    )

    out = materialize_clc(atlas, source="clc2018")
    assert out == prepared_country_path(tmp_path, "AUT", "clc2018")
    assert "url" in captured
    assert captured["url"] == "https://download.example.test/clc2018.tif"
    assert captured["headers"] == {"Authorization": "Bearer test-token"}


def test_materialize_clc_default_download_handles_non_georeferenced_integer_raster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.tif", {"Authorization": "Bearer test-token"}),
    )

    captured: dict[str, str] = {}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del kwargs
        captured["url"] = url
        _write_raster_non_georef(
            out_path,
            np.array(
                [
                    [231, 232],
                    [311, 312],
                ],
                dtype=np.uint16,
            ),
        )

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)

    out = materialize_clc(atlas, source="clc2018")
    assert out.exists()
    assert captured["url"] == "https://download.example.test/clc2018.tif"

    prepared = rxr.open_rasterio(out).squeeze(drop=True)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared.coords["y"].values, ref.coords["y"].values)
    np.testing.assert_allclose(prepared.coords["x"].values, ref.coords["x"].values)
    np.testing.assert_allclose(prepared.values, np.array([[231.0, 232.0], [311.0, 312.0]], dtype=np.float32))


def test_materialize_clc_extracts_tif_from_zip_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.zip", {"Authorization": "Bearer test-token"}),
    )

    captured: dict[str, Path] = {}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        captured["artifact_path"] = out_path
        write_zip_with_member(out_path, CLC_SOURCES["clc2018"]["filename"], b"fake-tif-bytes")

    def _fake_prepare(**kwargs):
        source_path = kwargs["source_path"]
        assert source_path.read_bytes() == b"fake-tif-bytes"
        return kwargs["prepared_path"]

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr("cleo.clc.prepare_clc_to_wind_grid", _fake_prepare)

    out = materialize_clc(atlas, source="clc2018")
    assert out == prepared_country_path(tmp_path, "AUT", "clc2018")
    assert captured["artifact_path"].suffix == ".zip"


def test_materialize_clc_extracts_tif_from_nested_zip_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.zip", {"Authorization": "Bearer test-token"}),
    )

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        write_nested_zip_with_member(
            out_path,
            "Results/u2018_clc2018_v2020_20u1_raster100m.zip",
            "u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif",
            b"nested-fake-tif-bytes",
        )

    def _fake_prepare(**kwargs):
        source_path = kwargs["source_path"]
        assert source_path.read_bytes() == b"nested-fake-tif-bytes"
        return kwargs["prepared_path"]

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr("cleo.clc.prepare_clc_to_wind_grid", _fake_prepare)

    out = materialize_clc(atlas, source="clc2018")
    assert out == prepared_country_path(tmp_path, "AUT", "clc2018")


def test_materialize_clc_retries_zip_artifact_until_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.zip", {"Authorization": "Bearer test-token"}),
    )
    monkeypatch.setattr("cleo.clc.time.sleep", lambda _seconds: None)

    calls = {"downloaded": 0}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        calls["downloaded"] += 1
        if calls["downloaded"] == 1:
            err = requests.exceptions.HTTPError("403 Client Error")
            err.response = SimpleNamespace(status_code=403)
            raise err
        write_zip_with_member(out_path, CLC_SOURCES["clc2018"]["filename"], b"fake-tif-bytes")

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr("cleo.clc.prepare_clc_to_wind_grid", lambda **kwargs: kwargs["prepared_path"])

    out = materialize_clc(atlas, source="clc2018")
    assert out == prepared_country_path(tmp_path, "AUT", "clc2018")
    assert calls["downloaded"] == 2


def test_materialize_clc_uses_browser_headers_for_fme_zip_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: (
            "https://copernicus-fme.eea.europa.eu/fmedatadownload/results/4050.zip",
            {"Authorization": "Bearer test-token"},
        ),
    )

    captured: dict[str, object] = {}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        write_zip_with_member(out_path, CLC_SOURCES["clc2018"]["filename"], b"fake-tif-bytes")

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr("cleo.clc.prepare_clc_to_wind_grid", lambda **kwargs: kwargs["prepared_path"])

    out = materialize_clc(atlas, source="clc2018")
    assert out == prepared_country_path(tmp_path, "AUT", "clc2018")
    assert captured["url"] == "https://copernicus-fme.eea.europa.eu/fmedatadownload/results/4050.zip"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["User-Agent"].startswith("Mozilla/5.0")
    assert "Authorization" not in headers


def test_materialize_clc_refreshes_unreadable_cached_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("not-a-raster", encoding="utf-8")

    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.tif", {"Authorization": "Bearer test-token"}),
    )

    calls = {"downloaded": 0}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        calls["downloaded"] += 1
        _write_raster_integer(
            out_path,
            np.array(
                [
                    [231, 231],
                    [311, 231],
                ],
                dtype=np.uint16,
            ),
        )

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)

    out = materialize_clc(atlas, source="clc2018")
    assert out.exists()
    assert calls["downloaded"] == 1


def test_materialize_clc_cached_non_georef_source_is_inferred_as_ref_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    _write_raster_non_georef(
        source_path,
        np.array(
            [
                [231, 232],
                [311, 312],
            ],
            dtype=np.uint16,
        ),
    )

    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)

    out = materialize_clc(atlas, source="clc2018")
    assert out.exists()

    prepared = rxr.open_rasterio(out).squeeze(drop=True)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared.coords["y"].values, ref.coords["y"].values)
    np.testing.assert_allclose(prepared.coords["x"].values, ref.coords["x"].values)
    np.testing.assert_allclose(prepared.values, np.array([[231.0, 232.0], [311.0, 312.0]], dtype=np.float32))


def test_materialize_clc_force_prepare_rebuilds_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = prepared_country_path(tmp_path, "AUT", "clc2018")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text("cached", encoding="utf-8")

    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    calls = {"downloaded": 0, "prepared": 0}
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.tif", {"Authorization": "Bearer test-token"}),
    )

    def _fake_download(_url: str, out_path: Path, **kwargs) -> None:
        del kwargs
        calls["downloaded"] += 1
        out_path.write_text("source", encoding="utf-8")

    def _fake_prepare(**kwargs):
        calls["prepared"] += 1
        return kwargs["prepared_path"]

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr("cleo.clc.prepare_clc_to_wind_grid", _fake_prepare)

    out = materialize_clc(atlas, source="clc2018", force_prepare=True)
    assert out == prepared
    assert calls["downloaded"] == 1
    assert calls["prepared"] == 1


def test_materialize_clc_refreshes_cached_multiband_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    _write_raster_multiband(
        source_path,
        np.array(
            [
                [[1, 2], [3, 4]],
                [[1, 2], [3, 4]],
                [[1, 2], [3, 4]],
                [[1, 2], [3, 4]],
            ],
            dtype=np.uint8,
        ),
    )

    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.tif", {"Authorization": "Bearer test-token"}),
    )

    calls = {"downloaded": 0}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        calls["downloaded"] += 1
        _write_raster_integer(
            out_path,
            np.array(
                [
                    [231, 231],
                    [311, 231],
                ],
                dtype=np.uint16,
            ),
        )

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)

    out = materialize_clc(atlas, source="clc2018")
    assert out.exists()
    assert calls["downloaded"] == 1


def test_materialize_clc_uses_shared_raster_band_count_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("placeholder", encoding="utf-8")

    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.tif", {"Authorization": "Bearer test-token"}),
    )
    monkeypatch.setattr("cleo.clc.prepare_clc_to_wind_grid", lambda **kwargs: kwargs["prepared_path"])

    helper_calls: list[Path] = []

    def _fake_band_count(path: Path) -> int | None:
        helper_calls.append(path)
        if path == source_path:
            return 4
        return None

    download_calls = {"count": 0}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        download_calls["count"] += 1
        out_path.write_text("source", encoding="utf-8")

    monkeypatch.setattr("cleo.clc.raster_band_count", _fake_band_count)
    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)

    materialize_clc(atlas, source="clc2018")

    assert helper_calls == [source_path]
    assert download_calls["count"] == 1


def test_materialize_clc_raises_for_multiband_rendered_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = _atlas_stub(tmp_path)

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)
    _write_canonical_landscape_store(tmp_path, ref)

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        _write_raster_multiband(
            out_path,
            np.array(
                [
                    [[1, 2], [3, 4]],
                    [[1, 2], [3, 4]],
                    [[1, 2], [3, 4]],
                    [[1, 2], [3, 4]],
                ],
                dtype=np.uint8,
            ),
        )

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr(
        "cleo.clc._resolve_default_clc_download",
        lambda _source: ("https://download.example.test/clc2018.tif", {"Authorization": "Bearer test-token"}),
    )

    with pytest.raises(RuntimeError, match="single-band categorical class codes"):
        materialize_clc(atlas, source="clc2018")
