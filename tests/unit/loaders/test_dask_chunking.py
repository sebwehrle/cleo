"""Tests for dask chunking of GWA rasters when Atlas uses use_dask=True."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from cleo import dask_utils


class TestMaybeChunkBandDimension:
    """Tests for maybe_chunk handling of band dimension."""

    def test_adds_band_dim_when_missing_from_chunks(self):
        """maybe_chunk adds band=1 to chunks when obj has band dim with size 1."""
        # Create DataArray with dims ("band", "y", "x") like GWA rasters
        da = xr.DataArray(
            np.random.rand(1, 64, 64),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": np.arange(64), "x": np.arange(64)},
        )

        # Chunk with only y and x specified
        result = dask_utils.maybe_chunk(da, enabled=True, chunks={"y": 32, "x": 32})

        # Verify it's dask-backed
        assert dask_utils.is_dask_backed(result)
        assert result.data.chunks is not None

        # Verify band dimension is chunked as 1 (access via .sizes and .chunks tuple)
        # result.chunks is ((1,), (32, 32), (32, 32)) for dims (band, y, x)
        band_idx = result.dims.index("band")
        y_idx = result.dims.index("y")
        x_idx = result.dims.index("x")
        assert result.chunks[band_idx] == (1,)
        assert result.chunks[y_idx] == (32, 32)
        assert result.chunks[x_idx] == (32, 32)

    def test_does_not_override_explicit_band_chunk(self):
        """maybe_chunk preserves explicitly specified band chunk size."""
        da = xr.DataArray(
            np.random.rand(1, 64, 64),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": np.arange(64), "x": np.arange(64)},
        )

        # Chunk with band explicitly specified
        result = dask_utils.maybe_chunk(da, enabled=True, chunks={"band": 1, "y": 32, "x": 32})

        assert dask_utils.is_dask_backed(result)
        band_idx = result.dims.index("band")
        assert result.chunks[band_idx] == (1,)

    def test_no_band_dim_works_normally(self):
        """maybe_chunk works normally for arrays without band dimension."""
        da = xr.DataArray(
            np.random.rand(64, 64),
            dims=("y", "x"),
            coords={"y": np.arange(64), "x": np.arange(64)},
        )

        result = dask_utils.maybe_chunk(da, enabled=True, chunks={"y": 32, "x": 32})

        assert dask_utils.is_dask_backed(result)
        y_idx = result.dims.index("y")
        x_idx = result.dims.index("x")
        assert result.chunks[y_idx] == (32, 32)
        assert result.chunks[x_idx] == (32, 32)

    def test_band_dim_larger_than_1_not_auto_added(self):
        """maybe_chunk does not auto-add band chunk if band size > 1."""
        da = xr.DataArray(
            np.random.rand(3, 64, 64),  # 3 bands
            dims=("band", "y", "x"),
            coords={"band": [1, 2, 3], "y": np.arange(64), "x": np.arange(64)},
        )

        result = dask_utils.maybe_chunk(da, enabled=True, chunks={"y": 32, "x": 32})

        assert dask_utils.is_dask_backed(result)
        # Band should be left unchunked (full size as single chunk)
        band_idx = result.dims.index("band")
        assert result.chunks[band_idx] == (3,)

    def test_disabled_returns_unchanged(self):
        """maybe_chunk returns unchanged object when enabled=False."""
        da = xr.DataArray(
            np.random.rand(1, 64, 64),
            dims=("band", "y", "x"),
        )

        result = dask_utils.maybe_chunk(da, enabled=False, chunks={"y": 32, "x": 32})

        assert not dask_utils.is_dask_backed(result)
        assert result is da

    def test_none_chunks_returns_unchanged(self):
        """maybe_chunk returns unchanged object when chunks=None."""
        da = xr.DataArray(
            np.random.rand(1, 64, 64),
            dims=("band", "y", "x"),
        )

        result = dask_utils.maybe_chunk(da, enabled=True, chunks=None)

        assert not dask_utils.is_dask_backed(result)
        assert result is da

    def test_auto_chunks_works(self):
        """maybe_chunk handles chunks='auto' correctly."""
        da = xr.DataArray(
            np.random.rand(1, 64, 64),
            dims=("band", "y", "x"),
        )

        result = dask_utils.maybe_chunk(da, enabled=True, chunks="auto")

        assert dask_utils.is_dask_backed(result)


class TestChunksForRasterio:
    """Tests for chunks_for_rasterio helper function."""

    def test_disabled_returns_none(self):
        """chunks_for_rasterio returns None when enabled=False."""
        result = dask_utils.chunks_for_rasterio(enabled=False, chunks={"y": 32, "x": 32})
        assert result is None

    def test_none_chunks_returns_none(self):
        """chunks_for_rasterio returns None when chunks=None."""
        result = dask_utils.chunks_for_rasterio(enabled=True, chunks=None)
        assert result is None

    def test_auto_chunks_returns_auto(self):
        """chunks_for_rasterio returns 'auto' when chunks='auto'."""
        result = dask_utils.chunks_for_rasterio(enabled=True, chunks="auto")
        assert result == "auto"

    def test_dict_chunks_adds_band_dim(self):
        """chunks_for_rasterio adds band=1 to dict chunks."""
        result = dask_utils.chunks_for_rasterio(enabled=True, chunks={"y": 32, "x": 32})
        assert result == {"band": 1, "y": 32, "x": 32}

    def test_dict_chunks_preserves_explicit_band(self):
        """chunks_for_rasterio preserves explicit band chunk."""
        result = dask_utils.chunks_for_rasterio(enabled=True, chunks={"band": 1, "y": 32, "x": 32})
        assert result == {"band": 1, "y": 32, "x": 32}


class TestAtlasDaskIntegration:
    """Tests for Atlas dask configuration affecting loaders."""

    def test_mock_loader_receives_dask_config(self):
        """Verify that loaders would receive correct dask_enabled and chunks from Atlas."""
        # Create a mock parent (Atlas) with dask config
        mock_parent = MagicMock()
        mock_parent.dask_enabled = True
        mock_parent.chunks = {"y": 64, "x": 64}

        # Simulate what loaders do
        dask_enabled = getattr(mock_parent, "dask_enabled", False)
        chunks = getattr(mock_parent, "chunks", None)

        assert dask_enabled is True
        assert chunks == {"y": 64, "x": 64}

        # Simulate chunking a GWA raster
        raster = xr.DataArray(
            np.random.rand(1, 128, 128),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": np.arange(128), "x": np.arange(128)},
        )

        result = dask_utils.maybe_chunk(raster, enabled=dask_enabled, chunks=chunks)

        assert dask_utils.is_dask_backed(result)
        band_idx = result.dims.index("band")
        y_idx = result.dims.index("y")
        x_idx = result.dims.index("x")
        assert result.chunks[band_idx] == (1,)
        assert result.chunks[y_idx] == (64, 64)
        assert result.chunks[x_idx] == (64, 64)


class TestRasterioLazyLoading:
    """Regression tests for lazy loading via rasterio with chunks."""

    def test_open_rasterio_with_chunks_is_dask_backed(self):
        """
        Regression test: rxr.open_rasterio with chunks= produces dask-backed array.

        The key fix is that loaders must pass chunks directly to open_rasterio(),
        not call open_rasterio() then maybe_chunk() afterwards.
        """
        import rioxarray as rxr
        import tempfile
        import rasterio
        from rasterio.transform import from_bounds

        # Create a small test GeoTIFF
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            data = np.random.rand(1, 64, 64).astype(np.float32)
            transform = from_bounds(0, 0, 64, 64, 64, 64)
            with rasterio.open(
                f.name,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=1,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)

            # Test 1: Open WITHOUT chunks - should be NumPy-backed
            da_eager = rxr.open_rasterio(f.name)
            assert not dask_utils.is_dask_backed(
                da_eager
            ), "Expected NumPy-backed when no chunks passed to open_rasterio"

            # Test 2: Open WITH chunks - should be dask-backed (THIS IS THE FIX)
            chunks = dask_utils.chunks_for_rasterio(enabled=True, chunks={"y": 32, "x": 32})
            da_lazy = rxr.open_rasterio(f.name, chunks=chunks)
            assert dask_utils.is_dask_backed(da_lazy), "Expected dask-backed when chunks passed to open_rasterio"

            # Verify chunk sizes
            band_idx = da_lazy.dims.index("band")
            y_idx = da_lazy.dims.index("y")
            x_idx = da_lazy.dims.index("x")
            assert da_lazy.chunks[band_idx] == (1,)
            assert da_lazy.chunks[y_idx] == (32, 32)
            assert da_lazy.chunks[x_idx] == (32, 32)


class TestComputeBackends:
    """Tests for compute/persist backend dispatch helpers."""

    def test_compute_backends_serial_threads_processes(self):
        da = xr.DataArray(
            np.random.rand(32, 32),
            dims=("y", "x"),
            coords={"y": np.arange(32), "x": np.arange(32)},
        ).chunk({"y": 16, "x": 16})

        out_serial = dask_utils.compute(da, backend="serial")
        out_threads = dask_utils.compute(da, backend="threads")
        try:
            out_processes = dask_utils.compute(da, backend="processes")
        except Exception as exc:
            msg = str(exc)
            if "Operation not permitted" in msg or "PermissionError" in msg:
                pytest.skip("Process backend unavailable in this execution environment.")
            raise

        assert isinstance(out_serial, xr.DataArray)
        assert isinstance(out_threads, xr.DataArray)
        assert isinstance(out_processes, xr.DataArray)
        assert out_serial.shape == da.shape
        assert out_threads.shape == da.shape
        assert out_processes.shape == da.shape

    def test_compute_backend_distributed_requires_active_client(self, monkeypatch: pytest.MonkeyPatch):
        da = xr.DataArray(
            np.random.rand(8, 8),
            dims=("y", "x"),
            coords={"y": np.arange(8), "x": np.arange(8)},
        ).chunk({"y": 4, "x": 4})

        monkeypatch.setattr(
            dask_utils,
            "get_distributed_client_and_dashboard",
            lambda: (_ for _ in ()).throw(
                RuntimeError(
                    "compute_backend='distributed' requires an active dask.distributed Client. "
                    "Start one first, e.g. `from dask.distributed import Client; Client()`"
                )
            ),
        )

        with pytest.raises(RuntimeError, match="compute_backend='distributed'.*active dask\\.distributed Client"):
            dask_utils.compute(da, backend="distributed")

    def test_compute_threads_propagates_num_workers_to_scheduler(self, monkeypatch: pytest.MonkeyPatch):
        da = xr.DataArray(
            np.random.rand(8, 8),
            dims=("y", "x"),
            coords={"y": np.arange(8), "x": np.arange(8)},
        ).chunk({"y": 4, "x": 4})

        import dask

        captured = []
        original_set = dask.config.set

        @contextmanager
        def _capturing_set(**kwargs):
            captured.append(kwargs)
            with original_set(**kwargs):
                yield

        monkeypatch.setattr(dask.config, "set", _capturing_set)

        out = dask_utils.compute(da, backend="threads", num_workers=3)
        assert isinstance(out, xr.DataArray)
        assert captured, "Expected dask.config.set to be called"
        assert captured[-1]["scheduler"] == "threads"
        assert captured[-1]["num_workers"] == 3

    def test_compute_num_workers_validation_by_backend(self):
        da = xr.DataArray(
            np.random.rand(4, 4),
            dims=("y", "x"),
            coords={"y": np.arange(4), "x": np.arange(4)},
        ).chunk({"y": 2, "x": 2})

        with pytest.raises(ValueError, match="compute_workers must be >= 1"):
            dask_utils.compute(da, backend="threads", num_workers=0)
        with pytest.raises(ValueError, match="compute_workers must be None or 1"):
            dask_utils.compute(da, backend="serial", num_workers=2)
        with pytest.raises(ValueError, match="not supported for compute_backend='distributed'"):
            dask_utils.compute(da, backend="distributed", num_workers=2)
