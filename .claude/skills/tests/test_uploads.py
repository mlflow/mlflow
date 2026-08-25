import http.client
import io
import json
import re
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
def test_upload_asset_raises_when_the_request_fails(tmp_path: Path, outcome: Exception) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=outcome) as opener,
        pytest.raises(uploads.UploadFailed, match="shot.png") as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    assert excinfo.value.status is None
    opener.assert_called_once()


def test_upload_asset_raises_when_the_response_carries_no_url(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"{}")) as opener,
        pytest.raises(uploads.UploadFailed, match="response carried no url"),
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()


def test_upload_asset_rejects_an_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("x")
    with pytest.raises(uploads.UploadFailed, match="unsupported extension"):
        uploads.upload_asset(path, "1", "t")


def test_upload_asset_rejects_an_empty_file(tmp_path: Path) -> None:
    # The endpoint blames the size cap for this, so catching it here is the only way the
    # message points at the real problem.
    path = tmp_path / "shot.png"
    path.write_bytes(b"")
    with pytest.raises(uploads.UploadFailed, match="the file is empty"):
        uploads.upload_asset(path, "1", "t")


def test_upload_asset_rejects_a_file_over_the_size_cap(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(b"12345")
    with (
        mock.patch.object(uploads, "MAX_IMAGE_BYTES", 4),
        pytest.raises(uploads.UploadFailed, match="5 bytes exceeds 4"),
    ):
        uploads.upload_asset(path, "1", "t")


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


def http_error(
    code: int, body: bytes = b"", headers: dict[str, str] | None = None
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        uploads.UPLOAD_URL,
        code,
        "nope",
        headers or {},  # type: ignore[arg-type]
        io.BytesIO(body),
    )


SIZE_REFUSAL = json.dumps({
    "message": "Validation Failed",
    "errors": [
        {
            "field": "size",
            "message": "size Yowza that's a big file. <span class='x'> </span>Try again.",
        }
    ],
}).encode()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (401, r"the credential was rejected \(401\)"),
        (403, r"403, most likely a credential that is not scoped"),
        # A 404 is ambiguous, so the message has to offer both readings.
        (404, r"repository_id=1 does not resolve or the endpoint refuses this credential"),
    ],
)
def test_upload_asset_explains_a_credential_failure(
    tmp_path: Path, code: int, expected: str
) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(code)) as opener,
        pytest.raises(uploads.UploadFailed, match=expected) as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()
    assert excinfo.value.status == code
    assert excinfo.value.fatal


def test_a_404_names_the_credential_kind_without_printing_it(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(404)) as opener,
        pytest.raises(uploads.UploadFailed, match="a GitHub App or Actions token") as excinfo,
    ):
        uploads.upload_asset(path, "1", "ghs_secret")

    opener.assert_called_once()
    assert "ghs_secret" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("gho_x", "an OAuth user token"),
        ("ghp_x", "a classic PAT"),
        ("github_pat_x", "a fine-grained PAT"),
        ("ghu_x", "a GitHub App user-to-server token"),
        ("ghs_x", "a GitHub App or Actions token"),
        ("ghr_x", "a refresh token"),
        ("x", "a credential of unrecognized kind"),
    ],
)
def test_describe_token_names_the_kind(token: str, expected: str) -> None:
    assert uploads.describe_token(token) == expected


def test_upload_asset_carries_the_status_of_other_http_errors(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(500)) as opener,
        pytest.raises(uploads.UploadFailed, match="shot.png") as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()
    assert excinfo.value.status == 500
    # A server fault is worth retrying with the next file; a credential fault is not.
    assert not excinfo.value.fatal


def test_a_422_carries_what_the_endpoint_objected_to(tmp_path: Path) -> None:
    # The reason phrase is "Unprocessable Entity" whatever went wrong, so the body is
    # the only thing that separates a bad content type from an oversized file.
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(422, SIZE_REFUSAL)) as opener,
        pytest.raises(uploads.UploadFailed, match="Validation Failed") as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()
    message = str(excinfo.value)
    assert "Yowza that's a big file. Try again." in message
    # The endpoint answers with markup meant for the web uploader.
    assert "<span" not in message
    assert not excinfo.value.fatal


def test_a_body_that_is_not_json_leaves_the_status_readable(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch(
            "urllib.request.urlopen", side_effect=http_error(500, b"<html>oops</html>")
        ) as opener,
        pytest.raises(uploads.UploadFailed, match="shot.png: 500 nope") as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()
    assert str(excinfo.value) == "shot.png: 500 nope"


RATE_LIMITED = json.dumps({"message": "You have exceeded a secondary rate limit"}).encode()
SECONDARY = "You have exceeded a secondary rate limit"


@pytest.mark.parametrize(
    ("body", "headers", "expected"),
    [
        (b"", {"Retry-After": "60"}, "rate limited (429); retry after 60s"),
        (b"", {}, "rate limited (429)"),
        # Both halves at once, so the formatting between them is pinned.
        (RATE_LIMITED, {"Retry-After": "60"}, f"rate limited (429): {SECONDARY}; retry after 60s"),
    ],
)
def test_rate_limiting_stops_the_run_and_reports_the_wait(
    tmp_path: Path, body: bytes, headers: dict[str, str], expected: str
) -> None:
    # Every remaining file would be refused too, so the run has to stop rather than
    # spend the rest of the batch on it.
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(429, body, headers)) as opener,
        pytest.raises(uploads.UploadFailed, match=r"rate limited \(429\)") as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()
    assert str(excinfo.value) == f"shot.png: {expected}"
    assert excinfo.value.fatal


def test_a_429_carries_the_body_when_retry_after_is_absent(tmp_path: Path) -> None:
    # Retry-After rides along only on a secondary limit, so a primary one would otherwise
    # report nothing but the status.
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(429, RATE_LIMITED)) as opener,
        pytest.raises(uploads.UploadFailed, match="secondary rate limit") as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()
    assert str(excinfo.value) == f"shot.png: rate limited (429): {SECONDARY}"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"field": "content_type", "code": "invalid"}, "content_type: invalid"),
        ({"code": "unprocessable"}, "unprocessable"),
    ],
)
def test_a_per_field_error_without_a_message_still_names_the_cause(
    tmp_path: Path, error: dict[str, str], expected: str
) -> None:
    # Only `code: custom` is documented to carry a message; without this the report
    # collapses to a bare "Validation Failed".
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")
    body = json.dumps({"message": "Validation Failed", "errors": [error]}).encode()

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(422, body)) as opener,
        pytest.raises(uploads.UploadFailed, match=re.escape(expected)) as excinfo,
    ):
        uploads.upload_asset(path, "1", "t")

    opener.assert_called_once()
    assert str(excinfo.value) == f"shot.png: 422 nope: Validation Failed; {expected}"
