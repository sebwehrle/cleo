import re
import pytest
import cleo.classes as C

MSG = "Atlas not materialized. Call atlas.materialize() first."

def test_wind_landscape_properties_guard_and_success(tmp_path):
    a = C.Atlas.__new__(C.Atlas)
    a._path = tmp_path
    a.country = "AUT"
    a.region = None
    a._wind = None
    a._landscape = None

    with pytest.raises(RuntimeError, match=re.escape(MSG)):
        _ = a.wind
    with pytest.raises(RuntimeError, match=re.escape(MSG)):
        _ = a.landscape

    a.wind = object()
    a.landscape = object()

    assert a.wind is not None
    assert a.landscape is not None
