from unittest import mock

import pytest
from fastapi import HTTPException

from mlflow.gateway.providers.utils import (
    SUPPORTED_ACCEPT_ENCODING,
    _aiohttp_post,
    proxy_root_url,
    rename_payload_keys,
    send_request,
)

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def test_rename_payload_keys():
    payload = {"old_key1": "value1", "old_key2": "value2", "old_key3": None, "old_key4": []}
    mapping = {"old_key1": "new_key1", "old_key2": "new_key2"}
    expected = {"new_key1": "value1", "new_key2": "value2", "old_key3": None, "old_key4": []}
    assert rename_payload_keys(payload, mapping) == expected


@pytest.mark.parametrize(
    ("payload", "mapping", "expected"),
    [
        (
            {"old_key1": "value1", "old_key2": None, "old_key3": "value3"},
            {"old_key1": "new_key1", "old_key3": "new_key3"},
            {"new_key1": "value1", "old_key2": None, "new_key3": "value3"},
        ),
        (
            {"old_key1": None, "old_key2": "value2", "old_key3": []},
            {"old_key1": "new_key1", "old_key3": "new_key3"},
            {"new_key1": None, "old_key2": "value2", "new_key3": []},
        ),
        (
            {"old_key1": "value1", "old_key2": "value2"},
            {"old_key1": "new_key1", "old_key3": "new_key3"},
            {"new_key1": "value1", "old_key2": "value2"},
        ),
    ],
)
def test_rename_payload_keys_parameterized(payload, mapping, expected):
    assert rename_payload_keys(payload, mapping) == expected


@pytest.mark.asyncio
async def test_aiohttp_post_includes_supported_accept_encoding():
    mock_client = mock_http_client(MockAsyncResponse({}))
    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session_cls:
        async with _aiohttp_post(
            headers={"Authorization": "Bearer key"},
            base_url="https://api.example.com",
            path="/v1/chat",
            payload={"model": "x"},
        ):
            pass
        mock_session_cls.assert_called_once()
        call_headers = mock_session_cls.call_args.kwargs["headers"]
        assert call_headers.get("Accept-Encoding") == SUPPORTED_ACCEPT_ENCODING


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.anthropic.com/v1", "https://api.anthropic.com"),
        ("https://api.openai.com/v1", "https://api.openai.com"),
        ("https://api.groq.com/openai/v1", "https://api.groq.com/openai"),
        ("https://api.example.com/v1/", "https://api.example.com"),
    ],
)
def test_proxy_root_url(base_url, expected):
    assert proxy_root_url(base_url) == expected


@pytest.mark.asyncio
async def test_send_request_surfaces_upstream_status_when_content_type_missing():
    # Regression for https://github.com/mlflow/mlflow/issues/23617: when an
    # upstream (e.g., Ollama replying 403 to a disallowed browser Origin)
    # returns an error status with no Content-Type header, mlflow should
    # forward that status instead of masking it as a generic 502.
    mock_response = MockAsyncResponse({"headers": {}, "detail": "Origin not allowed"}, status=403)
    mock_client = mock_http_client(mock_response)
    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session_cls:
        with pytest.raises(HTTPException, match="Origin not allowed") as exc_info:
            await send_request(
                headers={"Origin": "http://localhost:5000"},
                base_url="http://upstream",
                path="v1/responses",
                payload={"model": "x", "input": "hi"},
            )
    mock_session_cls.assert_called_once()
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"message": "invalid model"}}, "invalid model"),
        ({"error": "invalid model"}, "invalid model"),
    ],
)
async def test_send_request_extracts_error_message_from_json_body(body, expected):
    mock_response = MockAsyncResponse(body, status=400)
    mock_client = mock_http_client(mock_response)
    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session_cls:
        with pytest.raises(HTTPException, match=expected) as exc_info:
            await send_request(
                headers={},
                base_url="http://upstream",
                path="v1/chat/completions",
                payload={"model": "x"},
            )
    mock_session_cls.assert_called_once()
    assert exc_info.value.status_code == 400
