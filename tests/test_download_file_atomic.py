from pathlib import Path
from types import SimpleNamespace
from cleo.utils import download_file

class _Resp:
    def __init__(self):
        self.status_code = 200
    def raise_for_status(self):
        return None
    def iter_content(self, chunk_size=1024):
        yield b"partial"
        raise RuntimeError("boom")

def test_download_file_atomic_no_partial_left(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _Resp()
    import cleo.utils as ut
    monkeypatch.setattr(ut.requests, "get", fake_get)

    target = tmp_path / "x.bin"
    ok = download_file("https://example.com/x.bin", save_to=target, overwrite=True)
    assert ok is False
    assert not target.exists()
    assert not (tmp_path / "x.bin.tmp").exists()
