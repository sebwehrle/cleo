"""loaders: test_maybe_chunk_auto.

Tests for the _maybe_chunk_auto helper that makes dask optional.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.loaders import _maybe_chunk_auto


class TestMaybeChunkAuto:
    """Tests for _maybe_chunk_auto helper."""

    def test_returns_input_when_dask_unavailable(self, monkeypatch):
        """When dask import fails, returns input unchanged without chunking."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("dask"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        # Create a DataArray that would fail if .chunk() was called
        da = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))

        # Monkeypatch import to simulate dask not installed
        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Should return input unchanged, not attempt chunking
        result = _maybe_chunk_auto(da)

        assert result is da

    def test_chunks_when_dask_available(self):
        """When dask is available, returns chunked array."""
        dask = pytest.importorskip("dask.array")

        da = xr.DataArray(np.ones((10, 10)), dims=("y", "x"))

        result = _maybe_chunk_auto(da)

        # Result should be dask-backed
        assert hasattr(result.data, "__dask_graph__")

    def test_returns_input_when_chunk_fails(self):
        """When .chunk() fails for any reason, returns input unchanged."""
        dask = pytest.importorskip("dask.array")

        # Create an object that has .chunk but it raises
        class MockDataArray:
            def chunk(self, *args, **kwargs):
                raise RuntimeError("Chunking not supported")

        obj = MockDataArray()
        result = _maybe_chunk_auto(obj)

        assert result is obj

    def test_preserves_data_integrity(self):
        """Chunking preserves data values."""
        dask = pytest.importorskip("dask.array")

        data = np.arange(25.0).reshape((5, 5))
        da = xr.DataArray(data, dims=("y", "x"))

        result = _maybe_chunk_auto(da)

        # Values should be identical
        np.testing.assert_array_equal(result.values, data)
