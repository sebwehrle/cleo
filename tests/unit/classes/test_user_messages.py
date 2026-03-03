"""classes: test_user_messages.
Merged test file (imports preserved per chunk).
"""

import pytest
from pathlib import Path
from cleo.atlas import Atlas


def test_get_nuts_country_missing_shapefile_is_actionable(tmp_path, monkeypatch):
    atlas = Atlas.__new__(Atlas)
    atlas._path = Path(tmp_path)
    atlas.country = "AUT"
    atlas._crs = "epsg:3035"

    def _fail_download(*_args, **_kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("cleo.loaders.load_nuts", _fail_download)

    with pytest.raises(
        FileNotFoundError,
        match="NUTS shapefile not found.*Automatic NUTS download/extract was attempted",
    ):
        atlas.get_nuts_country()
