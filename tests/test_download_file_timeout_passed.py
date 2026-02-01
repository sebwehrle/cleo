"""Test that download_file passes timeout to requests.get."""

from unittest.mock import MagicMock, patch

from cleo.utils import download_file


def test_download_file_passes_default_timeout(tmp_path):
    """download_file should pass default timeout (10, 60) to requests.get."""
    captured_kwargs = {}

    def mock_get(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"test"])
        return mock_response

    with patch("cleo.utils.requests.get", side_effect=mock_get):
        download_file("http://example.com/file.txt", save_to=tmp_path / "x", overwrite=True)

    assert "timeout" in captured_kwargs, "timeout should be passed to requests.get"
    assert captured_kwargs["timeout"] == (10, 60), f"Expected (10, 60), got {captured_kwargs['timeout']}"


def test_download_file_passes_custom_timeout(tmp_path):
    """download_file should pass custom timeout to requests.get."""
    captured_kwargs = {}

    def mock_get(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"test"])
        return mock_response

    with patch("cleo.utils.requests.get", side_effect=mock_get):
        download_file(
            "http://example.com/file.txt",
            save_to=tmp_path / "x",
            overwrite=True,
            timeout=(5, 30),
        )

    assert "timeout" in captured_kwargs
    assert captured_kwargs["timeout"] == (5, 30), f"Expected (5, 30), got {captured_kwargs['timeout']}"
