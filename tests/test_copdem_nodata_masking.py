"""Oracle test: build_copdem_elevation_like must mask tile nodata to NaN."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
import rioxarray as rxr

from cleo.copdem import build_copdem_elevation_like


def test_build_copdem_elevation_like_masks_nodata(tmp_path):
    tile_path = tmp_path / "tile.tif"
    arr = np.array([[-9999, 10], [20, 30]], dtype="int16")
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999,
    }
    with rasterio.open(tile_path, "w", **profile) as dst:
        dst.write(arr, 1)

    ref_path = tmp_path / "ref.tif"
    ref_data = np.zeros((2, 2), dtype=np.float32)
    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=ref_data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(ref_data, 1)

    reference_da = rxr.open_rasterio(ref_path).squeeze()

    out = build_copdem_elevation_like(reference_da, [tile_path])
    assert np.isnan(out.values).any()
