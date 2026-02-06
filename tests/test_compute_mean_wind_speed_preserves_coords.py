from __future__ import annotations

import numpy as np
import xarray as xr

from cleo.assess import compute_mean_wind_speed


class DummyAtlas:
    def __init__(self):
        self.data = xr.Dataset(
            {
                "template": xr.DataArray(
                    np.zeros((2, 2), dtype=np.float32),
                    dims=("y", "x"),
                    coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
                )
            },
            coords={"wind_speed": [0, 1, 2]},
        )

    def load_weibull_parameters(self, height):
        a = xr.DataArray(
            np.ones((2, 2), dtype=np.float32) * 7.0,
            dims=("y", "x"),
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        k = xr.DataArray(
            np.ones((2, 2), dtype=np.float32) * 2.0,
            dims=("y", "x"),
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        return a, k


def test_compute_mean_wind_speed_preserves_dataset_coords():
    self = DummyAtlas()

    coords_before = set(self.data.coords.keys())
    compute_mean_wind_speed(self, height=50, chunk_size=None, inplace=True)
    compute_mean_wind_speed(self, height=100, chunk_size=None, inplace=True)

    coords_after = set(self.data.coords.keys())

    # Crucial: wind_speed must not disappear
    assert "wind_speed" in coords_after

    # Height is allowed to appear as a coord after adding a height-dimension variable
    assert coords_after.issuperset(coords_before.union({"height"}))
