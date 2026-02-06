import numpy as np
import xarray as xr
from pathlib import Path
from cleo.spatial import save_to_geotiff

def test_save_to_geotiff_all_finite_succeeds(tmp_path: Path):
    da = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"x": [0, 1], "y": [0, 1]})
    da = da.rio.write_crs("EPSG:4326")
    save_to_geotiff(da, "EPSG:4326", tmp_path, "tmp.tif")
    assert (tmp_path / "tmp.tif").is_file()
