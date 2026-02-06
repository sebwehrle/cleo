import pytest
import re
from pathlib import Path

from cleo.classes import Atlas

MSG = "Atlas not materialized. Call atlas.materialize() first."


def _mk_unmaterialized_atlas(tmp_path):
    a = Atlas.__new__(Atlas)
    a._path = Path(tmp_path)
    a.country = "AUT"
    a.region = None
    a._crs = "epsg:3035"
    return a


@pytest.mark.parametrize(
    "method_name,args,kwargs",
    [
        ("load", (), {}),
        ("save", (), {}),
        ("clip_to_nuts", ("Vienna",), {}),
        ("add_turbine", ("dummy",), {}),
    ],
)
def test_methods_require_materialize(tmp_path, method_name, args, kwargs):
    a = _mk_unmaterialized_atlas(tmp_path)
    fn = getattr(a, method_name)

    with pytest.raises(RuntimeError, match=re.escape(MSG)):
        fn(*args, **kwargs)
