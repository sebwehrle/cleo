def block_downloads(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Download attempted during offline test.")
    monkeypatch.setattr("cleo.net.download_to_path", _blocked)
    monkeypatch.setattr("cleo.net.http_get", _blocked)
