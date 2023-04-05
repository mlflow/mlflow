import pytest

from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.os import is_windows


@pytest.mark.skipif(is_windows(), reason="This test fails on Windows")
@pytest.mark.parametrize(
    ("file_path", "expected_mime_type"),
    [
        ("c.txt", "text/plain"),
        ("c.pkl", "application/octet-stream"),
        ("/a/b/c.pkl", "application/octet-stream"),
        ("/a/b/c.png", "image/png"),
        ("/a/b/c.pdf", "application/pdf"),
        ("/a/b/MLmodel", "text/plain"),
        ("/a/b/mlproject", "text/plain"),
    ],
)
def test_guess_mime_type(file_path, expected_mime_type):
    assert _guess_mime_type(file_path) == expected_mime_type


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
@pytest.mark.parametrize(
    ("file_path", "expected_mime_type"),
    [
        ("C:\\a\\b\\c.txt", "text/plain"),
        ("c.txt", "text/plain"),
        ("c.pkl", "application/octet-stream"),
        ("C:\\a\\b\\c.pkl", "application/octet-stream"),
        ("C:\\a\\b\\c.png", "image/png"),
        ("C:\\a\\b\\c.pdf", "application/pdf"),
        ("C:\\a\\b\\MLmodel", "text/plain"),
        ("C:\\a\\b\\mlproject", "text/plain"),
    ],
)
def test_guess_mime_type_on_windows(file_path, expected_mime_type):
    assert _guess_mime_type(file_path) == expected_mime_type
