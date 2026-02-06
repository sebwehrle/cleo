"""classes: test_atlas_materialize.

Contracts:
- Atlas.crs setter rejects invalid CRS strings with ValueError("Invalid CRS ...").
- Access to wind/landscape (and selected methods) requires prior materialize().
- materialize() instantiates _WindAtlas/_LandscapeAtlas exactly once (idempotent).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import cleo.classes as C

MSG = "Atlas not materialized. Call atlas.materialize() first."


def _mk_unmaterialized_atlas(tmp_path: Path) -> C.Atlas:
    a = C.Atlas.__new__(C.Atlas)  # bypass heavy __init__
    a._path = Path(tmp_path)
    a.country = "AUT"
    a.region = None
    a._crs = "epsg:3035"
    a._wind = None
    a._landscape = None
    a._materialized = False
    return a


def test_atlas_crs_invalid_raises_valueerror() -> None:
    atlas = C.Atlas.__new__(C.Atlas)  # bypass heavy __init__
    with pytest.raises(ValueError, match="Invalid CRS"):
        atlas.crs = "not-a-crs"


def test_atlas_requires_materialize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Replace heavy sub-atlases with cheap dummies.
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


@pytest.mark.parametrize(
    "method_name,args,kwargs",
    [
        ("load", (), {}),
        ("save", (), {}),
        ("clip_to_nuts", ("Vienna",), {}),
        ("add_turbine", ("dummy",), {}),
    ],
)
def test_methods_require_materialize(
    tmp_path: Path, method_name: str, args: tuple, kwargs: dict
) -> None:
    a = _mk_unmaterialized_atlas(tmp_path)
    fn = getattr(a, method_name)

    with pytest.raises(RuntimeError, match=re.escape(MSG)):
        fn(*args, **kwargs)


def test_materialize_instantiates_and_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    a = _mk_unmaterialized_atlas(tmp_path)

    a.materialize()
    assert a._materialized is True
    assert a._wind is not None
    assert a._landscape is not None
    assert calls == {"wind": 1, "land": 1}

    # Second call should early-return.
    a.materialize()
    assert calls == {"wind": 1, "land": 1}


def test_wind_landscape_properties_guard_and_success(tmp_path: Path) -> None:
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
