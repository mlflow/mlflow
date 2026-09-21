import contextlib
import re
from unittest import mock

import aiohttp
import pytest
from fastapi import HTTPException

from mlflow.gateway.providers.utils import (
    SUPPORTED_ACCEPT_ENCODING,
    _aiohttp_post,
    proxy_root_url,
    rename_payload_keys,
    send_proxy_request,
    send_stream_request,
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


@pytest.mark.asyncio
async def test_aiohttp_post_strips_client_content_encoding():
    # The client's Content-Encoding describes the body it sent, which is decompressed before
    # reaching here, so forwarding it would label the re-serialized JSON as still encoded.
    mock_client = mock_http_client(MockAsyncResponse({}))
    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session_cls:
        async with _aiohttp_post(
            headers={"Authorization": "Bearer key", "content-encoding": "zstd"},
            base_url="https://api.example.com",
            path="/v1/chat",
            payload={"model": "x"},
        ):
            pass
        call_headers = mock_session_cls.call_args.kwargs["headers"]
        assert not any(k.lower() == "content-encoding" for k in call_headers)


@pytest.mark.asyncio
async def test_aiohttp_post_uses_timeout_from_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS", "2")

    mock_client = mock_http_client(MockAsyncResponse({}))
    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        async with _aiohttp_post(
            headers={"Authorization": "Bearer key"},
            base_url="https://api.example.com",
            path="/v1/chat",
            payload={"model": "x"},
        ):
            pass

    mock_client.post.assert_called_once()
    assert mock_client.post.call_args.kwargs["timeout"].total == 2


@pytest.mark.asyncio
async def test_aiohttp_post_never_follows_redirects():
    # A redirect from the upstream could otherwise steer the request to an internal host
    # that the SSRF guard never saw.
    mock_client = mock_http_client(MockAsyncResponse({}))
    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        async with _aiohttp_post(
            headers={"Authorization": "Bearer key"},
            base_url="https://api.example.com",
            path="/v1/chat",
            payload={"model": "x"},
        ):
            pass

    assert mock_client.post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.asyncio
async def test_aiohttp_post_sets_read_bufsize_for_large_sse_lines():
    mock_client = mock_http_client(MockAsyncResponse({}))
    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session_cls:
        async with _aiohttp_post(
            headers={"Authorization": "Bearer key"},
            base_url="https://api.example.com",
            path="/v1/chat",
            payload={"model": "x"},
        ):
            pass

    assert mock_session_cls.call_args.kwargs["read_bufsize"] == 2**20


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


# Body that cannot be read at all, as opposed to one that is read as None or "".
_UNREADABLE = object()


class _FailingResponse:
    """Response double for an upstream error, with the body shapes set per test.

    ``json``/``text`` raise for ``_UNREADABLE``, standing in for a response that is not
    JSON, or whose body cannot be read at all. aiohttp's own ``json()`` returns None for
    an empty body and its ``text()`` returns "", which tests pass explicitly.
    """

    def __init__(
        self, json_body=_UNREADABLE, text_body=_UNREADABLE, status=400, message="Bad Request"
    ):
        self.status = status
        self.message = message
        self._json_body = json_body
        self._text_body = text_body

    def raise_for_status(self):
        raise aiohttp.ClientResponseError(None, None, status=self.status, message=self.message)

    async def json(self):
        if self._json_body is _UNREADABLE:
            raise ValueError("body is not JSON")
        return self._json_body

    async def text(self):
        if self._text_body is _UNREADABLE:
            raise ValueError("body could not be read")
        return self._text_body


@contextlib.asynccontextmanager
async def _upstream(response):
    yield response


@pytest.mark.parametrize(
    ("error_body", "expected_detail"),
    [
        # OpenAI-compatible providers nest the message under "error".
        ({"error": {"message": "max_tokens is too large"}}, "max_tokens is too large"),
        # Amazon Bedrock returns a top-level "message".
        (
            {"message": "thinking.type.enabled is not supported"},
            "thinking.type.enabled is not supported",
        ),
        # Neither shape: the body is still more useful than the reason phrase.
        ({"reason": "quota exhausted"}, "{'reason': 'quota exhausted'}"),
        ({"error": {"message": ""}}, "{'error': {'message': ''}}"),
    ],
)
@pytest.mark.asyncio
async def test_send_stream_request_preserves_upstream_error_message(error_body, expected_detail):
    response = _FailingResponse(json_body=error_body)
    with (
        mock.patch(
            "mlflow.gateway.providers.utils._aiohttp_post",
            side_effect=lambda *args, **kwargs: _upstream(response),
        ) as mock_post,
        pytest.raises(HTTPException, match=re.escape(expected_detail)) as exc_info,
    ):
        [chunk async for chunk in send_stream_request({}, "https://api.example.com", "chat", {})]

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == expected_detail
    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_send_stream_request_falls_back_to_response_text():
    response = _FailingResponse(text_body="<html>Bad Gateway</html>", status=502)
    with (
        mock.patch(
            "mlflow.gateway.providers.utils._aiohttp_post",
            side_effect=lambda *args, **kwargs: _upstream(response),
        ) as mock_post,
        pytest.raises(HTTPException, match="Bad Gateway") as exc_info,
    ):
        [chunk async for chunk in send_stream_request({}, "https://api.example.com", "chat", {})]

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "<html>Bad Gateway</html>"
    mock_post.assert_called_once()


@pytest.mark.parametrize(
    ("json_body", "text_body"),
    [
        # aiohttp reads an empty JSON body as None and an empty text body as "".
        (None, _UNREADABLE),
        ({}, _UNREADABLE),
        (_UNREADABLE, ""),
        (_UNREADABLE, _UNREADABLE),
    ],
)
@pytest.mark.asyncio
async def test_send_stream_request_falls_back_to_reason_phrase(json_body, text_body):
    response = _FailingResponse(json_body=json_body, text_body=text_body)
    with (
        mock.patch(
            "mlflow.gateway.providers.utils._aiohttp_post",
            side_effect=lambda *args, **kwargs: _upstream(response),
        ) as mock_post,
        pytest.raises(HTTPException, match="Bad Request") as exc_info,
    ):
        [chunk async for chunk in send_stream_request({}, "https://api.example.com", "chat", {})]

    assert exc_info.value.detail == "Bad Request"
    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_send_proxy_request_preserves_upstream_error_message():
    response = _FailingResponse(json_body={"message": "model is not enabled in this region"})
    with (
        mock.patch(
            "mlflow.gateway.providers.utils._aiohttp_post",
            side_effect=lambda *args, **kwargs: _upstream(response),
        ) as mock_post,
        pytest.raises(HTTPException, match="model is not enabled in this region") as exc_info,
    ):
        await send_proxy_request({}, "https://api.example.com", "v1/messages", {}).__anext__()

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "model is not enabled in this region"
    mock_post.assert_called_once()
