import pytest
from cleo.classes import Atlas


def test_atlas_crs_invalid_raises_valueerror():
    atlas = Atlas.__new__(Atlas)  # bypass heavy __init__
    with pytest.raises(ValueError, match="Invalid CRS"):
        atlas.crs = "not-a-crs"
