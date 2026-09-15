import pytest

from mlflow.genai.skill_content.errors import (
    error_code_for_http_status,
    invalid_content,
    redact_credentials,
    source_unavailable,
)
from mlflow.protos.databricks_pb2 import UNAUTHENTICATED, ErrorCode


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://user:p4ss@example.com/repo.git", "https://***@example.com/repo.git"),
        (
            "fatal: could not read from https://tok@host/x",
            "fatal: could not read from https://***@host/x",
        ),
        ("user:tok@github.com:acme/skills.git", "***@github.com:acme/skills.git"),
        ("Authorization: Bearer abcdefghij0123456789", "Authorization: Bearer ***"),
        ("header Basic dXNlcjpwYXNzd29yZA==", "header Basic ***"),
        ("Bearer abc", "Bearer ***"),
        ("basic dTpw", "basic ***"),
        ("nothing secret here", "nothing secret here"),
        ("git@github.com:acme/skills.git", "git@github.com:acme/skills.git"),
    ],
)
def test_redact_credentials(text, expected):
    assert redact_credentials(text) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "UNAUTHENTICATED"),
        (403, "PERMISSION_DENIED"),
        (404, "RESOURCE_DOES_NOT_EXIST"),
        (410, "RESOURCE_DOES_NOT_EXIST"),
        (429, "TEMPORARILY_UNAVAILABLE"),
        (500, "TEMPORARILY_UNAVAILABLE"),
        (503, "TEMPORARILY_UNAVAILABLE"),
        (400, "RESOURCE_DOES_NOT_EXIST"),
    ],
)
def test_error_code_for_http_status(status, expected):
    assert ErrorCode.Name(error_code_for_http_status(status)) == expected


def test_source_unavailable_redacts_and_carries_code():
    exc = source_unavailable(
        "https://u:p@h/r.git", "Bearer secrettoken123", error_code=UNAUTHENTICATED
    )
    assert exc.error_code == "UNAUTHENTICATED"
    assert "u:p@" not in exc.message
    assert "secrettoken123" not in exc.message
    assert "https://***@h/r.git" in exc.message


def test_invalid_content_redacts():
    exc = invalid_content("bad entry from https://u:p@h/x")
    assert exc.error_code == "INVALID_PARAMETER_VALUE"
    assert "u:p@" not in exc.message


@pytest.mark.parametrize("error_path", ["source_failure", "size_failure"])
@pytest.mark.parametrize("scheme", ["https", "HTTPS", "Http"])
def test_query_credentials_are_absent_from_errors(error_path, scheme):
    # Presigned URLs carry their credential in the query string, which fetch errors echo;
    # URL schemes are case-insensitive so the spelling must not matter.
    secret = "invented-review-secret"
    url = f"{scheme}://example.invalid/skill.zip?X-Amz-Signature={secret}"
    if error_path == "source_failure":
        error = source_unavailable(url, "HTTP 403 Forbidden")
    else:
        error = invalid_content(f"Download from '{url}' exceeds the size limit")
    assert secret not in str(error)
    assert f"{scheme}://example.invalid/skill.zip?***" in str(error)
