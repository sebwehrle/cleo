from dataclasses import dataclass
from pathlib import Path
import xarray as xr

@dataclass
class DummyParent:
    path: Path | str
    country: str
    crs: str
    region: str | None = None

    def get_nuts_region(self, region):
        return None

@dataclass
class DummyWindAtlas:
    parent: DummyParent
    data: xr.Dataset
