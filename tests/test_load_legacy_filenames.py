# tests/test_load_legacy_filenames.py
from pathlib import Path
import numpy as np
import xarray as xr
import cleo.classes as C

class DummySub:
    def __init__(self):
        self.data = None

def _mini_ds():
    return xr.Dataset({"a": (("y","x"), np.ones((2,2), dtype=np.float32))}, coords={"x":[0,1],"y":[0,1]})

def test_load_falls_back_to_legacy_files(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    # legacy names expected by code when no timestamped matches exist
    (_mini_ds()).to_netcdf(processed / "WindAtlas_AUT.nc")
    (_mini_ds()).to_netcdf(processed / "LandscapeAtlas_AUT.nc")

    a = C.Atlas.__new__(C.Atlas)
    a._path = tmp_path
    a.country = "AUT"
    a.region = None
    a._wind = DummySub()
    a._landscape = DummySub()

    a.load(region="None", scenario="default", timestamp="latest")

    assert a.wind.data is not None
    assert a.landscape.data is not None
