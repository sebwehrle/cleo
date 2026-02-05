import pytest
from types import SimpleNamespace

import cleo.classes as C


def test_atlas_requires_materialize(monkeypatch, tmp_path):
    # Replace heavy sub-atlases with cheap dummies
    class DummyWind:
        def __init__(self, parent):  # pragma: no cover
            self.parent = parent

    class DummyLandscape:
        def __init__(self, parent):  # pragma: no cover
            self.parent = parent

    monkeypatch.setattr(C, "_WindAtlas", DummyWind, raising=True)
    monkeypatch.setattr(C, "_LandscapeAtlas", DummyLandscape, raising=True)

    a = C.Atlas(tmp_path, "AUT", "EPSG:3035")

    with pytest.raises(RuntimeError, match="materialize"):
        _ = a.wind
    with pytest.raises(RuntimeError, match="materialize"):
        _ = a.landscape

    a.materialize()
    assert isinstance(a.wind, DummyWind)
    assert isinstance(a.landscape, DummyLandscape)
