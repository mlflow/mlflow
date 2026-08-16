import http.client
import io
import json
import urllib.error
from pathlib import Path
from unittest import mock

import pytest
from skills.github import uploads

URL = "https://github.com/user-attachments/assets/2f1c0a3e-0000-4000-8000-000000000001"


def test_upload_asset_builds_the_request_and_returns_the_url(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    response = io.BytesIO(json.dumps({"url": URL}).encode())
    with mock.patch("urllib.request.urlopen", return_value=response) as opener:
        assert uploads.upload_asset(path, "136202695", "tok") == URL

    request = opener.call_args.args[0]
    assert "name=shot.png" in request.full_url
    assert "content_type=image%2Fpng" in request.full_url
    assert "repository_id=136202695" in request.full_url
    assert request.get_header("Authorization") == "Bearer tok"
    assert request.data == b"\x89PNG"


@pytest.mark.parametrize(
    "outcome",
    [
        urllib.error.URLError("boom"),
        TimeoutError("slow"),
        # Escapes URLError: urlopen only wraps h.request(), not h.getresponse().
        http.client.RemoteDisconnected("closed"),
        http.client.IncompleteRead(b"partial"),
        json.JSONDecodeError("bad", "", 0),
    ],
)
def test_upload_asset_returns_none_when_the_request_fails(
    tmp_path: Path, outcome: Exception
) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with mock.patch("urllib.request.urlopen", side_effect=outcome) as opener:
        assert uploads.upload_asset(path, "1", "t") is None

    opener.assert_called_once()


def test_upload_asset_returns_none_when_the_response_carries_no_url(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"{}")) as opener:
        assert uploads.upload_asset(path, "1", "t") is None

    opener.assert_called_once()


def test_upload_asset_skips_an_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("x")
    assert uploads.upload_asset(path, "1", "t") is None


def test_upload_asset_skips_a_file_over_the_size_cap(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(b"12345")
    with mock.patch.object(uploads, "MAX_IMAGE_BYTES", 4):
        assert uploads.upload_asset(path, "1", "t") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shot.png", uploads.MAX_IMAGE_BYTES),
        ("shot.gif", uploads.MAX_IMAGE_BYTES),
        ("clip.mp4", uploads.MAX_VIDEO_BYTES),
        ("clip.MOV", uploads.MAX_VIDEO_BYTES),
        ("clip.webm", uploads.MAX_VIDEO_BYTES),
    ],
)
def test_max_bytes_is_larger_for_video(name: str, expected: int) -> None:
    assert uploads.max_bytes(name) == expected


def test_a_video_between_the_image_and_video_caps_is_not_skipped(tmp_path: Path) -> None:
    # The old single 10MB cap silently dropped exactly this: a screen recording.
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"0" * (uploads.MAX_IMAGE_BYTES + 1))

    response = io.BytesIO(json.dumps({"url": URL}).encode())
    with mock.patch("urllib.request.urlopen", return_value=response):
        assert uploads.upload_asset(path, "1", "t") == URL


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(uploads.UPLOAD_URL, code, "nope", {}, None)  # type: ignore[arg-type]


def test_upload_asset_raises_on_a_rejected_token(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(401)),
        pytest.raises(uploads.TokenRejected, match="rejected \\(401\\)"),
    ):
        uploads.upload_asset(path, "1", "t")


def test_upload_asset_returns_none_on_other_http_errors(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with mock.patch("urllib.request.urlopen", side_effect=http_error(500)):
        assert uploads.upload_asset(path, "1", "t") is None
