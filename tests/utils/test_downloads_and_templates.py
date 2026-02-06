"""utils: test_downloads_and_templates.
Merged test file (imports preserved per chunk).
"""

import pytest
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from unittest.mock import patch, MagicMock
from cleo.utils import download_file, _match_to_template

# --- merged from tests/_staging/test_download_file_atomic.py ---

class _Resp:
    def __init__(self):
        self.status_code = 200
    def raise_for_status(self):
        return None
    def iter_content(self, chunk_size=1024):
        yield b"partial"
        raise RuntimeError("boom")

def test_download_file_atomic_no_partial_left(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _Resp()
    import cleo.utils as ut
    monkeypatch.setattr(ut.requests, "get", fake_get)

    target = tmp_path / "x.bin"
    ok = download_file("https://example.com/x.bin", save_to=target, overwrite=True)
    assert ok is False
    assert not target.exists()
    assert not (tmp_path / "x.bin.tmp").exists()


# --- merged from tests/_staging/test_download_file_no_write_on_error.py ---

class MockResponse:
    def __init__(self, status_code=200, content_chunks=None):
        self.status_code = status_code
        self._chunks = content_chunks or []

    def iter_content(self, chunk_size=1024):
        yield from self._chunks

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")

    # If production uses context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_file_http_403_no_file_written(tmp_path, monkeypatch):
    """HTTP 403 should return False and not create any file."""
    save_path = tmp_path / "bad.tif"

    def mock_get(url, **kwargs):
        return MockResponse(status_code=403, content_chunks=[b"error body"])

    monkeypatch.setattr("cleo.utils.requests.get", mock_get)

    result = download_file("http://example.com/file.tif", save_to=save_path, overwrite=True)

    assert result is False
    assert not save_path.exists()


def test_download_file_http_200_writes_file(tmp_path, monkeypatch):
    """HTTP 200 should return True and write file contents."""
    save_path = tmp_path / "good.tif"

    def mock_get(url, **kwargs):
        return MockResponse(status_code=200, content_chunks=[b"OK"])

    monkeypatch.setattr("cleo.utils.requests.get", mock_get)

    result = download_file("http://example.com/file.tif", save_to=save_path, overwrite=True)

    assert result is True
    assert save_path.exists()
    assert save_path.read_bytes() == b"OK"


# --- merged from tests/_staging/test_download_file_timeout_passed.py ---

def test_download_file_passes_default_timeout(tmp_path):
    """download_file should pass default timeout (10, 60) to requests.get."""
    captured_kwargs = {}

    def mock_get(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"test"])
        return mock_response

    with patch("cleo.utils.requests.get", side_effect=mock_get):
        download_file("http://example.com/file.txt", save_to=tmp_path / "x", overwrite=True)

    assert "timeout" in captured_kwargs, "timeout should be passed to requests.get"
    assert captured_kwargs["timeout"] == (10, 60), f"Expected (10, 60), got {captured_kwargs['timeout']}"


def test_download_file_passes_custom_timeout(tmp_path):
    """download_file should pass custom timeout to requests.get."""
    captured_kwargs = {}

    def mock_get(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"test"])
        return mock_response

    with patch("cleo.utils.requests.get", side_effect=mock_get):
        download_file(
            "http://example.com/file.txt",
            save_to=tmp_path / "x",
            overwrite=True,
            timeout=(5, 30),
        )

    assert "timeout" in captured_kwargs
    assert captured_kwargs["timeout"] == (5, 30), f"Expected (5, 30), got {captured_kwargs['timeout']}"


# --- merged from tests/_staging/test_grid_template_alignment.py ---

def make_template_raster(x_coords, y_coords, crs="EPSG:4326"):
    """Create a template raster with rioxarray CRS and transform."""
    data = np.ones((len(y_coords), len(x_coords)), dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        name="template",
    )
    da = da.rio.write_crs(crs)
    # Set transform based on coords
    x_res = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    y_res = y_coords[0] - y_coords[1] if len(y_coords) > 1 else -1.0
    from rasterio.transform import from_bounds
    transform = from_bounds(
        x_coords[0] - x_res / 2,
        y_coords[-1] - abs(y_res) / 2,
        x_coords[-1] + x_res / 2,
        y_coords[0] + abs(y_res) / 2,
        len(x_coords),
        len(y_coords),
    )
    da = da.rio.write_transform(transform)
    return da


def test_match_to_template_already_aligned():
    """If coords already match, return unchanged."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0])

    template = make_template_raster(x, y)
    other = make_template_raster(x, y)
    other.values[:] = 42.0

    result = _match_to_template(other, template)

    assert np.array_equal(result.coords["x"].values, template.coords["x"].values)
    assert np.array_equal(result.coords["y"].values, template.coords["y"].values)
    assert np.all(result.values == 42.0)


def test_match_to_template_misaligned_coords():
    """Misaligned coords get reprojected to match template."""
    x_template = np.array([0.0, 1.0, 2.0])
    y_template = np.array([1.0, 0.0])

    # Shifted coords (simulate grid mismatch)
    x_other = np.array([0.1, 1.1, 2.1])
    y_other = np.array([1.1, 0.1])

    template = make_template_raster(x_template, y_template)
    other = make_template_raster(x_other, y_other)
    other.values[:] = 99.0

    result = _match_to_template(other, template)

    # Result coords must exactly match template
    assert np.array_equal(result.coords["x"].values, template.coords["x"].values)
    assert np.array_equal(result.coords["y"].values, template.coords["y"].values)


def test_concat_with_join_exact_after_alignment():
    """After alignment, concat with join='exact' succeeds without NaN stripes."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0])

    template = make_template_raster(x, y)

    # Existing mean_wind_speed at height=50 on template grid
    mws_50 = template.copy()
    mws_50.values[:] = 5.0
    mws_50 = mws_50.expand_dims(height=[50])

    # New mean_wind_speed at height=100 on slightly misaligned grid
    x_shifted = np.array([0.05, 1.05, 2.05])
    y_shifted = np.array([1.05, 0.05])
    mws_100_misaligned = make_template_raster(x_shifted, y_shifted)
    mws_100_misaligned.values[:] = 10.0

    # Align to template
    mws_100_aligned = _match_to_template(mws_100_misaligned, template)
    mws_100_aligned = mws_100_aligned.expand_dims(height=[100])

    # Concat with join="exact" - should not raise
    result = xr.concat([mws_50, mws_100_aligned], dim="height", join="exact")

    assert result.dims == ("height", "y", "x")
    assert list(result.coords["height"].values) == [50, 100]
    assert np.array_equal(result.coords["x"].values, x)
    assert np.array_equal(result.coords["y"].values, y)


def test_no_nan_stripes_after_alignment():
    """Verify aligned output has no NaN stripes within valid mask."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 1.0, 0.0])

    template = make_template_raster(x, y)
    # Mark all cells as valid (no NaN in template)
    template.values[:] = 0.0

    # Create misaligned raster with actual data
    x_shifted = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
    y_shifted = np.array([2.1, 1.1, 0.1])
    other = make_template_raster(x_shifted, y_shifted)
    other.values[:] = 1.0

    aligned = _match_to_template(other, template)

    # Stripe metric: share of columns where NaN fraction >= 0.95
    valid_mask = ~np.isnan(template.values)
    aligned_vals = aligned.values

    stripe_cols = 0
    for col_idx in range(aligned_vals.shape[1]):
        col_data = aligned_vals[:, col_idx]
        col_valid = valid_mask[:, col_idx]
        if col_valid.sum() > 0:
            nan_frac = np.isnan(col_data[col_valid]).sum() / col_valid.sum()
            if nan_frac >= 0.95:
                stripe_cols += 1

    stripe_score = stripe_cols / aligned_vals.shape[1] if aligned_vals.shape[1] > 0 else 0

    assert stripe_score == 0, f"Stripe score {stripe_score} > 0 indicates NaN stripes"


def test_merge_with_join_exact_requires_alignment():
    """Merge with join='exact' raises if coords don't match - alignment is required."""
    x1 = np.array([0.0, 1.0, 2.0])
    y1 = np.array([1.0, 0.0])
    x2 = np.array([0.1, 1.1, 2.1])  # Misaligned
    y2 = np.array([1.0, 0.0])

    da1 = xr.DataArray(
        np.ones((2, 3)),
        dims=["y", "x"],
        coords={"y": y1, "x": x1},
        name="var1",
    )
    da2 = xr.DataArray(
        np.ones((2, 3)),
        dims=["y", "x"],
        coords={"y": y2, "x": x2},
        name="var2",
    )

    # Without alignment, join="exact" should raise
    with pytest.raises(ValueError, match="cannot align.*join='exact'"):
        xr.merge([da1, da2], join="exact")


def test_match_to_template_raises_without_xy_coords():
    """Calling _match_to_template without x/y coords raises ValueError."""
    # DataArray without x/y coords
    da_no_xy = xr.DataArray(
        np.ones((3, 4)),
        dims=["a", "b"],
        coords={"a": [0, 1, 2], "b": [0, 1, 2, 3]},
    )

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0])
    template = make_template_raster(x, y)

    with pytest.raises(ValueError, match="x/y coords"):
        _match_to_template(da_no_xy, template)
