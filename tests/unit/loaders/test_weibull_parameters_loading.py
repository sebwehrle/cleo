"""loaders: test_weibull_parameters_loading.
Merged test file (imports preserved per chunk).
"""

import pytest
from types import SimpleNamespace
import cleo.loaders as L

# --- merged from tests/_staging/test_load_weibull_parameters_missing_files_raises.py ---

def test_load_weibull_parameters_missing_files_raises(tmp_path):
    parent = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="EPSG:4326",
        region=None,
        get_nuts_region=lambda r: None,
    )
    self = SimpleNamespace(parent=parent)

    with pytest.raises(FileNotFoundError, match="Missing Weibull"):
        L.load_weibull_parameters(self, 100)
