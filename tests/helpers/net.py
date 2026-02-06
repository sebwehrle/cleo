def block_downloads(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Download attempted during offline test.")
    monkeypatch.setattr("cleo.utils.download_file", _blocked)
    monkeypatch.setattr("cleo.loaders.requests.get", _blocked)
