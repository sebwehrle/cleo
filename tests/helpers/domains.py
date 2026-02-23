from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def make_landscape_domain_atlas_stub(
    tmp_path: Path,
    *,
    include_build_clc: bool = False,
):
    """Create a minimal Atlas-like stub for LandscapeDomain unit tests."""
    stub = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        build_canonical=lambda: None,
        fingerprint_method="path_mtime_size",
    )
    if include_build_clc:
        stub.build_clc = (
            lambda source="clc2018": tmp_path / "data" / "raw" / "AUT" / "clc" / f"{source}.tif"
        )
    return stub

