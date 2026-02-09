"""Tests for _set_var exact-grid enforcement.

Oracle test: _set_var must raise ValueError when raster x/y coords
don't match template exactly.
"""

from tests.helpers.optional import requires_rioxarray

requires_rioxarray()

import pytest
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401

from cleo.classes import _WindAtlas


def _make_template(x_coords, y_coords, crs="EPSG:4326"):
    """Create a template DataArray with CRS."""
    data = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        name="template",
    )
    da = da.rio.write_crs(crs)
    return da


def _make_raster(x_coords, y_coords, value=1.0, name="test_var"):
    """Create a raster DataArray."""
    data = np.full((len(y_coords), len(x_coords)), value, dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        name=name,
    )
    return da


class TestSetVarExactGrid:
    """Tests for _set_var exact-grid enforcement."""

    @pytest.fixture
    def atlas(self):
        """Create a dummy _WindAtlas without running __init__."""
        # Use object.__new__ to bypass __init__
        atlas = object.__new__(_WindAtlas)
        atlas.parent = None
        atlas.data = xr.Dataset()
        return atlas

    def test_set_template_success(self, atlas):
        """Setting template with x/y dims should succeed."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0])
        template = _make_template(x, y)

        atlas._set_var("template", template)

        assert "template" in atlas.data.data_vars
        assert np.array_equal(atlas.data["template"].coords["x"].values, x)
        assert np.array_equal(atlas.data["template"].coords["y"].values, y)

    def test_set_template_no_xy_raises(self, atlas):
        """Setting template without x/y dims should raise ValueError."""
        da_no_xy = xr.DataArray(
            np.ones((3,)),
            dims=["time"],
            coords={"time": [0, 1, 2]},
            name="template",
        )

        with pytest.raises(ValueError, match="must have both x and y dims"):
            atlas._set_var("template", da_no_xy)

    def test_set_var_matching_grid_success(self, atlas):
        """Setting var with matching x/y coords should succeed."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0])

        # Set template first
        template = _make_template(x, y)
        atlas._set_var("template", template)

        # Set var with matching coords
        da_ok = _make_raster(x, y, value=42.0, name="test_ok")
        atlas._set_var("test_ok", da_ok)

        assert "test_ok" in atlas.data.data_vars
        assert np.all(atlas.data["test_ok"].values == 42.0)

    def test_set_var_x_mismatch_raises(self, atlas):
        """Setting var with mismatched x coords should raise ValueError."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0])

        # Set template
        template = _make_template(x, y)
        atlas._set_var("template", template)

        # Try to set var with shifted x coords
        x_shifted = np.array([0.1, 1.1, 2.1])
        da_bad = _make_raster(x_shifted, y, name="bad_x")

        with pytest.raises(ValueError, match="x coords differ from template"):
            atlas._set_var("bad_x", da_bad)

    def test_set_var_y_mismatch_raises(self, atlas):
        """Setting var with mismatched y coords should raise ValueError."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0])

        # Set template
        template = _make_template(x, y)
        atlas._set_var("template", template)

        # Try to set var with shifted y coords
        y_shifted = np.array([1.1, 0.1])
        da_bad = _make_raster(x, y_shifted, name="bad_y")

        with pytest.raises(ValueError, match="y coords differ from template"):
            atlas._set_var("bad_y", da_bad)

    def test_set_var_without_template_raises(self, atlas):
        """Setting non-template var without template should raise ValueError."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0])
        da = _make_raster(x, y, name="orphan")

        with pytest.raises(ValueError, match="template not in self.data"):
            atlas._set_var("orphan", da)

    def test_set_var_non_raster_passthrough(self, atlas):
        """Setting non-raster var (no x/y dims) should pass through."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0])

        # Set template
        template = _make_template(x, y)
        atlas._set_var("template", template)

        # Set non-raster var (e.g., power_curve with wind_speed dim only)
        da_1d = xr.DataArray(
            np.array([0.0, 0.5, 1.0]),
            dims=["wind_speed"],
            coords={"wind_speed": [0.0, 5.0, 10.0]},
            name="power_curve",
        )

        # Should not raise (non-raster passes through)
        atlas._set_var("power_curve", da_1d)
        assert "power_curve" in atlas.data.data_vars

    def test_set_var_data_none_raises(self, atlas):
        """Setting var when self.data is None should raise RuntimeError."""
        atlas.data = None
        da = xr.DataArray(np.ones((2, 3)), dims=["y", "x"])

        with pytest.raises(RuntimeError, match="self.data is None"):
            atlas._set_var("test", da)
