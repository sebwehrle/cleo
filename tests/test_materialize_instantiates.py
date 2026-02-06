# tests/test_materialize_instantiates.py
import cleo.classes as C

def test_materialize_instantiates_and_is_idempotent(tmp_path, monkeypatch):
    calls = {"wind": 0, "land": 0}

    class DummyWind:
        def __init__(self, parent):
            calls["wind"] += 1
            self.parent = parent
            self.data = None

    class DummyLand:
        def __init__(self, parent):
            calls["land"] += 1
            self.parent = parent
            self.data = None

    monkeypatch.setattr(C, "_WindAtlas", DummyWind, raising=True)
    monkeypatch.setattr(C, "_LandscapeAtlas", DummyLand, raising=True)

    a = C.Atlas.__new__(C.Atlas)
    a._path = tmp_path
    a.country = "AUT"
    a.region = None
    a._wind = None
    a._landscape = None
    a._materialized = False

    a.materialize()
    assert a._materialized is True
    assert a._wind is not None
    assert a._landscape is not None
    assert calls == {"wind": 1, "land": 1}

    # second call should early-return
    a.materialize()
    assert calls == {"wind": 1, "land": 1}
