"""utils: test_convert.
Merged test file (imports preserved per chunk).
"""

import numpy as np
import xarray as xr
from types import SimpleNamespace
from cleo.utils import convert


def test_convert_preserves_attrs_and_updates_unit():
    ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))}, coords={"x": [0, 1], "y": [0, 1]})
    ds["v"].attrs = {"unit": "m", "foo": "bar"}
    self = SimpleNamespace(data=ds)

    convert(self, "v", "cm", inplace=True)

    assert self.data["v"].attrs["unit"] == "cm"
    assert self.data["v"].attrs["foo"] == "bar"
    # 1 m = 100 cm
    assert float(self.data["v"].values[0, 0]) == 100.0
