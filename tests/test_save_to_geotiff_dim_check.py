import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from cleo.spatial import save_to_geotiff

def test_save_to_geotiff_rejects_non_2d_xy(tmp_path: Path):
    da = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("band", "y", "x"),
        coords={"band": [0], "x": [0, 1], "y": [0, 1]},
    )
    with pytest.raises(TypeError, match="expects exactly dims"):
        save_to_geotiff(da, "EPSG:4326", tmp_path, "tmp.tif")
