# tests/test_index_jsonl_and_cleanup.py
from pathlib import Path
import json
import cleo.classes as C

def test_index_roundtrip_and_cleanup_keeps_latest(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # create dummy dataset files referenced by index
    f_old = tmp_path / "old.nc"
    f_new = tmp_path / "new.nc"
    f_leg = tmp_path / "legacy.nc"
    for p in (f_old, f_new, f_leg):
        p.write_bytes(b"x")

    a = C.Atlas.__new__(C.Atlas)
    a._path = tmp_path
    a.country = "AUT"
    a.region = None
    a.index_file = data_dir / "index.jsonl"

    # write JSONL lines directly (includes legacy + 2 timestamps)
    lines = [
        {"subclass":"WindAtlas","country":"AUT","region":"None","scenario":"default","path":str(f_old),"timestamp":"20200101T000000"},
        {"subclass":"WindAtlas","country":"AUT","region":"None","scenario":"default","path":str(f_new),"timestamp":"20210101T000000"},
        {"subclass":"LandscapeAtlas","country":"AUT","region":"None","scenario":"default","path":str(f_leg),"timestamp":"legacy"},
    ]
    a.index_file.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    entries = a._read_index()
    assert len(entries) == 3

    # cleanup should keep latest per subclass:
    # - WindAtlas: keep 20210101..., delete 20200101...
    # - LandscapeAtlas: only legacy exists -> keep it
    a.cleanup_datasets(scenario="default")

    assert not f_old.exists()
    assert f_new.exists()
    assert f_leg.exists()

    # ensure index rewritten to only 2 entries
    rewritten = a._read_index()
    assert len(rewritten) == 2
