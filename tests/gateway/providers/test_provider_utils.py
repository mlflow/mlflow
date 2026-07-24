from unittest import mock

import pytest

from mlflow.gateway.providers.utils import (
    SUPPORTED_ACCEPT_ENCODING,
    _aiohttp_post,
    proxy_root_url,
    rename_payload_keys,
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
