import json
from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.environment_variables import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.config import EndpointConfig, VertexAIConfig
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.vertex_ai import VertexAIProvider, _get_vertex_ai_host
from mlflow.gateway.schemas import chat, completions

from tests.gateway.tools import MockAsyncResponse, MockAsyncStreamingResponse, mock_http_client


def _mock_credentials():
    creds = mock.Mock()
    creds.token = "mock-access-token"
    creds.valid = True
    return creds


def _make_provider() -> VertexAIProvider:
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-2.0-flash",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "us-central1",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    return provider


def _chat_response():
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello from Vertex AI!"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
        },
        "headers": {"Content-Type": "application/json"},
    }


def test_base_url():
    provider = _make_provider()
    assert provider.base_url == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/publishers/google/models"
    )


def test_headers_use_bearer_token():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer mock-access-token"}


def test_passthrough_headers_keep_provider_bearer_token():
    provider = _make_provider()
    # ASGI servers lower-case inbound header names, so a client Authorization arrives
    # as "authorization". Forwarding it alongside Vertex's own bearer token would leave
    # the upstream with two conflicting Authorization headers (Google returns 401).
    merged = provider._get_headers(
        headers={
            "authorization": "Bearer client-token",
            "user-agent": "python-httpx/0.27.0",
            "X-Custom": "value",
        }
    )
    assert merged["Authorization"] == "Bearer mock-access-token"
    assert "authorization" not in merged
    assert merged["user-agent"] == "python-httpx/0.27.0"
    assert merged["X-Custom"] == "value"


@pytest.mark.parametrize(
    ("betas", "expected"),
    [
        (None, "interleaved-thinking-2025-05-14, advisor-tool-2026-03-01"),
        ([], None),
        (["interleaved-thinking-2025-05-14"], "interleaved-thinking-2025-05-14"),
        (["web-search-2025-03-05"], None),
    ],
)
def test_claude_passthrough_headers_filter_anthropic_beta(betas, expected):
    # Vertex rejects the whole request on an anthropic-beta value it does not support, so
    # the endpoint config chooses which client betas are forwarded: None keeps the header
    # as is, [] drops it, and a list keeps only the listed values.
    provider = _make_claude_provider(vertex_anthropic_betas=betas)
    merged = provider._delegate._get_headers(
        headers={
            "anthropic-beta": "interleaved-thinking-2025-05-14, advisor-tool-2026-03-01",
            "x-request-id": "req-1",
        }
    )
    assert merged.get("anthropic-beta") == expected
    assert merged["x-request-id"] == "req-1"
    assert merged["Authorization"] == "Bearer mock-access-token"


def test_claude_passthrough_headers_without_anthropic_beta_are_unchanged():
    provider = _make_claude_provider(vertex_anthropic_betas=[])
    merged = provider._delegate._get_headers(headers={"x-request-id": "req-1"})
    assert merged == {"Authorization": "Bearer mock-access-token", "x-request-id": "req-1"}


@pytest.mark.asyncio
async def test_claude_passthrough_request_omits_filtered_anthropic_beta():
    provider = _make_claude_provider(vertex_anthropic_betas=["web-search-2025-03-05"])
    captured_session_headers = {}
    mock_client = mock_http_client(MockAsyncResponse(_claude_chat_response()))

    def mock_client_session(headers=None, **kwargs):
        captured_session_headers.update(headers or {})
        return mock_client

    with mock.patch("aiohttp.ClientSession", mock_client_session):
        await provider.passthrough(
            PassthroughAction.ANTHROPIC_MESSAGES,
            _claude_passthrough_payload(),
            headers={"anthropic-beta": "advisor-tool-2026-03-01", "x-request-id": "req-1"},
        )

    mock_client.post.assert_called_once()
    assert "anthropic-beta" not in captured_session_headers
    assert captured_session_headers["x-request-id"] == "req-1"
    assert captured_session_headers["Authorization"] == "Bearer mock-access-token"


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "Vertex AI"
    assert provider.get_provider_name() == "vertex_ai"


@pytest.mark.asyncio
async def test_chat():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["choices"][0]["message"]["content"] == "Hello from Vertex AI!"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20


@pytest.mark.asyncio
async def test_gemini_passthrough_uses_vertex_endpoint():
    provider = _make_provider()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(_chat_response())
    ) as mock_post:
        response = await provider.passthrough(
            PassthroughAction.GEMINI_GENERATE_CONTENT,
            {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

    assert response["candidates"][0]["content"]["parts"][0]["text"] == "Hello from Vertex AI!"
    assert mock_post.call_args[0][0] == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/publishers/google/models"
        "/gemini-2.0-flash:generateContent"
    )


def _tool_calling_second_turn_payload():
    return {
        "messages": [
            {"role": "user", "content": "What's the weather like in Singapore today?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "function": {
                            "arguments": '{"location": "Singapore"}',
                            "name": "get_weather",
                        },
                        "type": "function",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": '{"temperature": 31.2, "condition": "sunny"}',
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The name of a city"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    }


def _find_part(contents, key):
    return next(
        part[key] for content in contents for part in content.get("parts", []) if key in part
    )


@pytest.mark.asyncio
async def test_chat_tool_calling_omits_function_call_id():
    # Regression test for #24127: Vertex AI rejects `id` on functionCall/functionResponse
    # parts with 400 INVALID_ARGUMENT, unlike the Developer Gemini API which requires it.
    provider = _make_provider()
    resp = {
        "candidates": [
            {"content": {"parts": [{"text": "Sunny, 31.2 degrees."}]}, "finishReason": "stop"}
        ]
    }
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        await provider.chat(chat.RequestPayload(**_tool_calling_second_turn_payload()))

    contents = mock_post.call_args[1]["json"]["contents"]
    function_call = _find_part(contents, "functionCall")
    function_response = _find_part(contents, "functionResponse")

    assert "id" not in function_call
    assert "id" not in function_response
    # The function name must survive — Gemini requires it on both parts.
    assert function_call["name"] == "get_weather"
    assert function_response["name"] == "get_weather"
    assert function_call["args"] == {"location": "Singapore"}


@pytest.mark.asyncio
async def test_chat_tool_calling_preserves_thought_signature():
    provider = _make_provider()
    payload = _tool_calling_second_turn_payload()
    payload["messages"][1]["tool_calls"][0]["thought_signature"] = "opaque_thought_sig_token"
    resp = {
        "candidates": [
            {"content": {"parts": [{"text": "Sunny, 31.2 degrees."}]}, "finishReason": "stop"}
        ]
    }
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        await provider.chat(chat.RequestPayload(**payload))

    contents = mock_post.call_args[1]["json"]["contents"]
    function_call = _find_part(contents, "functionCall")
    part = next(
        part for content in contents for part in content.get("parts", []) if "functionCall" in part
    )
    assert "id" not in function_call
    assert "thoughtSignature" not in function_call
    assert part["thoughtSignature"] == "opaque_thought_sig_token"


@pytest.mark.asyncio
async def test_chat_stream_tool_calling_omits_function_call_id():
    provider = _make_provider()
    stream_resp = [
        b'data: {"candidates": [{"content": {"parts": [{"text": "Sunny."}]}, '
        b'"finishReason": "stop"}]}\n\n',
    ]
    mock_client = mock_http_client(MockAsyncStreamingResponse(stream_resp))
    payload = _tool_calling_second_turn_payload()
    payload["stream"] = True

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        stream = provider.chat_stream(chat.RequestPayload(**payload))
        _ = [chunk async for chunk in stream]

    contents = mock_client.post.call_args[1]["json"]["contents"]
    assert "id" not in _find_part(contents, "functionCall")
    assert "id" not in _find_part(contents, "functionResponse")


def test_adapter_class_matches_the_active_delegate():
    # adapter_class must follow the delegate: Claude/MaaS models are formatted by their
    # own adapters (get_endpoint_url points at the delegate's endpoint), not the Gemini one.
    from mlflow.gateway.providers.openai_compatible import OpenAICompatibleAdapter
    from mlflow.gateway.providers.vertex_ai import (
        _VertexAIClaudeAdapter,
        _VertexGeminiAdapter,
    )

    assert _make_provider().adapter_class is _VertexGeminiAdapter
    # Claude resolves to the Vertex-specific Anthropic adapter, not the base one: callers
    # that format through ``adapter_class`` alone must still get ``anthropic_version``.
    assert _make_claude_provider().adapter_class is _VertexAIClaudeAdapter
    assert (
        _make_maas_provider("meta/llama-3.1-405b-instruct-maas").adapter_class
        is OpenAICompatibleAdapter
    )


@pytest.mark.parametrize(
    "model_name", ["claude-sonnet-4-5@20251101", "meta/llama-3.1-405b-instruct-maas"]
)
def test_delegate_inherits_enable_tracing(model_name):
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": model_name,
            "config": {"vertex_project": "my-gcp-project", "vertex_location": "us-east5"},
        },
    )
    provider = VertexAIProvider(endpoint_config, enable_tracing=True)
    # Streaming passthrough accumulates token usage on the delegate, and the delegate only
    # records it on the span when its own tracing flag is set.
    assert provider._delegate._enable_tracing is True


@pytest.mark.asyncio
async def test_chat_parallel_tool_calls_omit_all_function_call_ids():
    # Two parallel tool calls in one turn — the case `id` exists for on the Dev API.
    # Proves the strip loop covers every functionCall part, not just the first.
    provider = _make_provider()
    payload = _tool_calling_second_turn_payload()
    payload["messages"][1]["tool_calls"].append({
        "id": "call_002",
        "function": {"arguments": '{"location": "Tokyo"}', "name": "get_weather"},
        "type": "function",
    })
    payload["messages"].append({
        "role": "tool",
        "tool_call_id": "call_002",
        "content": '{"temperature": 18.0, "condition": "cloudy"}',
    })
    resp = {
        "candidates": [
            {"content": {"parts": [{"text": "Sunny and cloudy."}]}, "finishReason": "stop"}
        ]
    }
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        await provider.chat(chat.RequestPayload(**payload))

    contents = mock_post.call_args[1]["json"]["contents"]
    function_calls = [
        part["functionCall"]
        for c in contents
        for part in c.get("parts", [])
        if "functionCall" in part
    ]
    function_responses = [
        part["functionResponse"]
        for c in contents
        for part in c.get("parts", [])
        if "functionResponse" in part
    ]
    assert len(function_calls) == 2
    assert len(function_responses) == 2
    assert all("id" not in fc for fc in function_calls)
    assert all("id" not in fr for fr in function_responses)


def test_basic_config():
    config = VertexAIConfig(vertex_project="my-project")
    assert config.vertex_project == "my-project"
    assert config.vertex_location is None
    assert config.vertex_credentials is None
    assert config.vertex_anthropic_betas is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([], []),
        (["web-search-2025-03-05"], ["web-search-2025-03-05"]),
        ("", []),
        (
            "web-search-2025-03-05, interleaved-thinking-2025-05-14",
            ["web-search-2025-03-05", "interleaved-thinking-2025-05-14"],
        ),
    ],
)
def test_anthropic_betas_config(value, expected):
    # auth_config is a string map on the server API, so the list also comes in as a
    # comma-separated string.
    config = VertexAIConfig(vertex_project="my-project", vertex_anthropic_betas=value)
    assert config.vertex_anthropic_betas == expected


def test_custom_location():
    config = VertexAIConfig(vertex_project="my-project", vertex_location="europe-west4")
    assert config.vertex_location == "europe-west4"


def test_global_endpoint_when_no_location():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-2.0-flash",
            "config": {"vertex_project": "my-project"},
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://aiplatform.googleapis.com"
        "/v1/projects/my-project/locations/global/publishers/google/models"
    )


def test_global_location_uses_global_endpoint():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-3-pro-preview",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "global",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/global/publishers/google/models"
    )


def test_anthropic_model_uses_anthropic_publisher():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "claude-sonnet-4-5@20251101",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "us-east5",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://us-east5-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-east5/publishers/anthropic/models"
    )


def test_anthropic_model_global_location():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "claude-opus-4-5@20251101",
            "config": {
                "vertex_project": "my-project",
                "vertex_location": "global",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://aiplatform.googleapis.com"
        "/v1/projects/my-project/locations/global/publishers/anthropic/models"
    )


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("global", "https://aiplatform.googleapis.com"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com"),
        ("us", "https://aiplatform.us.rep.googleapis.com"),
        ("europe-west1", "https://europe-west1-aiplatform.googleapis.com"),
        ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
        ("us-east5", "https://us-east5-aiplatform.googleapis.com"),
    ],
)
def test_get_vertex_ai_host(location, expected):
    assert _get_vertex_ai_host(location) == expected


@pytest.mark.parametrize("location", ["eu", "us"])
def test_gemini_model_multi_region_location(location):
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-3-pro-preview",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": location,
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        f"https://aiplatform.{location}.rep.googleapis.com"
        f"/v1/projects/my-gcp-project/locations/{location}/publishers/google/models"
    )


@pytest.mark.parametrize("location", ["eu", "us"])
def test_anthropic_model_multi_region_location(location):
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "claude-sonnet-4-5@20251101",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": location,
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        f"https://aiplatform.{location}.rep.googleapis.com"
        f"/v1/projects/my-gcp-project/locations/{location}/publishers/anthropic/models"
    )
    assert provider.get_endpoint_url("llm/v1/chat") == (
        f"https://aiplatform.{location}.rep.googleapis.com"
        f"/v1/projects/my-gcp-project/locations/{location}/publishers/anthropic/models"
        "/claude-sonnet-4-5@20251101:rawPredict"
    )


def _make_claude_provider(**config) -> VertexAIProvider:
    endpoint_config = EndpointConfig(
        name="vertex-claude-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "claude-sonnet-4-5@20251101",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "us-east5",
                **config,
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    return provider


def _claude_chat_response():
    return {
        "content": [{"text": "Hello from Claude on Vertex AI!", "type": "text"}],
        "id": "msg-vertex-test",
        "model": "claude-sonnet-4-5@20251101",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }


@pytest.mark.asyncio
async def test_claude_chat_uses_raw_predict_endpoint():
    provider = _make_claude_provider()
    resp = _claude_chat_response()

    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        payload = chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}])
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["choices"][0]["message"]["content"] == "Hello from Claude on Vertex AI!"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 15

    mock_post.assert_called_once_with(
        "https://us-east5-aiplatform.googleapis.com/v1/projects/my-gcp-project"
        "/locations/us-east5/publishers/anthropic/models"
        "/claude-sonnet-4-5@20251101:rawPredict",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
            "anthropic_version": "vertex-2023-10-16",
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get()),
        allow_redirects=False,
    )


def _claude_stream_data():
    return [
        b"event: message_start\n",
        b'data: {"type": "message_start", "message": {"id": "msg-1", "type": "message", '
        b'"role": "assistant", "content": [], "model": "claude-sonnet-4-5@20251101", '
        b'"stop_reason": null, "stop_sequence": null, '
        b'"usage": {"input_tokens": 5, "output_tokens": 1}}}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type": "content_block_delta", "index": 0, '
        b'"delta": {"type": "text_delta", "text": "Hello"}}\n',
        b"\n",
        b"event: message_delta\n",
        b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
        b'"usage": {"output_tokens": 1}}\n',
        b"\n",
    ]


@pytest.mark.asyncio
async def test_claude_chat_stream_uses_stream_raw_predict_endpoint():
    provider = _make_claude_provider()
    mock_client = mock_http_client(MockAsyncStreamingResponse(_claude_stream_data()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}], stream=True)
        chunks = [chunk async for chunk in provider.chat_stream(payload)]

    assert len(chunks) > 0
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert ":streamRawPredict" in call_kwargs[0][0]
    assert call_kwargs[1]["json"]["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in call_kwargs[1]["json"]


def test_claude_adapter_applies_vertex_fields_without_provider_hooks():
    """Formatting through ``adapter_class`` alone must produce a Vertex-valid payload.

    ``_prepare_payload`` is a provider hook reached only from ``AnthropicProvider._chat``
    and ``._chat_stream``. Callers that build a request from the adapter and post it
    themselves -- the judge path in ``mlflow.genai.judges.adapters.gateway_adapter``
    does exactly this -- never run that hook, so Vertex rejected the request with
    "anthropic_version: Field required". The transformation therefore has to live on the
    adapter, as it already does for Bedrock.
    """
    from mlflow.gateway.providers.vertex_ai import _VERTEX_ANTHROPIC_VERSION

    provider = _make_claude_provider()
    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}

    formatted = provider.adapter_class.chat_to_model(dict(payload), provider.config)
    assert formatted["anthropic_version"] == _VERTEX_ANTHROPIC_VERSION
    # Vertex carries the model in the URL and rejects it in the body.
    assert "model" not in formatted

    streamed = provider.adapter_class.chat_streaming_to_model(dict(payload), provider.config)
    assert streamed["anthropic_version"] == _VERTEX_ANTHROPIC_VERSION
    assert "model" not in streamed


def test_claude_adapter_applies_vertex_fields_via_judge_provider_resolution(monkeypatch):
    """Same guarantee as above, but resolved the way the judge path actually does it.

    ``mlflow.genai.judges.adapters.gateway_adapter`` resolves its provider via
    ``_get_provider_instance("vertex_ai", model_name)``, not by constructing
    ``VertexAIProvider`` directly. Going through that resolution function here pins
    the regression at the exact seam that broke: if it ever stops routing Claude
    models to ``_VertexAIClaudeAdapter``, this test -- not just the adapter-level one
    above -- would catch it.
    """
    from mlflow.gateway.providers.vertex_ai import _VERTEX_ANTHROPIC_VERSION, _VertexAIClaudeAdapter
    from mlflow.metrics.genai.model_utils import _get_provider_instance

    monkeypatch.setenv("VERTEX_PROJECT", "my-gcp-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")

    provider = _get_provider_instance("vertex_ai", "claude-sonnet-4-5@20251101")
    assert provider.adapter_class is _VertexAIClaudeAdapter

    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}
    formatted = provider.adapter_class.chat_to_model(dict(payload), provider.config)
    assert formatted["anthropic_version"] == _VERTEX_ANTHROPIC_VERSION
    assert "model" not in formatted


def _claude_passthrough_payload(**overrides):
    return {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, **overrides}


@pytest.mark.asyncio
async def test_claude_passthrough_uses_raw_predict_endpoint():
    provider = _make_claude_provider()
    captured_session_headers = {}
    mock_client = mock_http_client(MockAsyncResponse(_claude_chat_response()))

    def mock_client_session(headers=None, **kwargs):
        captured_session_headers.update(headers or {})
        return mock_client

    with mock.patch("aiohttp.ClientSession", mock_client_session):
        response = await provider.passthrough(
            PassthroughAction.ANTHROPIC_MESSAGES,
            _claude_passthrough_payload(),
            headers={
                "authorization": "Bearer client-token",
                "x-request-id": "req-001",
                "host": "gateway.example.com",
            },
        )

    assert response["content"][0]["text"] == "Hello from Claude on Vertex AI!"
    mock_client.post.assert_called_once_with(
        "https://us-east5-aiplatform.googleapis.com/v1/projects/my-gcp-project"
        "/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20251101:rawPredict",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64,
            "anthropic_version": "vertex-2023-10-16",
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get()),
        allow_redirects=False,
    )
    assert captured_session_headers["Authorization"] == "Bearer mock-access-token"
    assert "authorization" not in captured_session_headers
    assert captured_session_headers["x-request-id"] == "req-001"
    assert "host" not in captured_session_headers


@pytest.mark.parametrize("auth_header", ["authorization", "x-api-key"])
def test_claude_passthrough_headers_drop_credential_agent_auth(auth_header):
    # AnthropicProvider keeps a Claude Code / Codex / Gemini CLI client's own credential in
    # place of the server key. A client's Anthropic credential is never valid on Vertex, and
    # Google rejects a request carrying two Authorization headers, so it must be dropped.
    provider = _make_claude_provider()
    merged = provider._delegate._get_headers(
        headers={
            "user-agent": "claude-cli/2.0.37 (external, cli)",
            auth_header: "client-credential",
        }
    )
    assert merged == {
        "user-agent": "claude-cli/2.0.37 (external, cli)",
        "Authorization": "Bearer mock-access-token",
    }


@pytest.mark.asyncio
async def test_claude_passthrough_stream_uses_stream_raw_predict_endpoint():
    provider = _make_claude_provider()
    stream_data = _claude_stream_data()
    mock_client = mock_http_client(MockAsyncStreamingResponse(stream_data))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        stream = await provider.passthrough(
            PassthroughAction.ANTHROPIC_MESSAGES, _claude_passthrough_payload(stream=True)
        )
        chunks = [chunk async for chunk in stream]

    assert chunks == stream_data
    mock_client.post.assert_called_once()
    url = mock_client.post.call_args[0][0]
    body = mock_client.post.call_args[1]["json"]
    assert url.endswith("/claude-sonnet-4-5@20251101:streamRawPredict")
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in body


@pytest.mark.asyncio
async def test_claude_passthrough_rejects_unsupported_action():
    provider = _make_claude_provider()
    with pytest.raises(AIGatewayException, match="Unsupported passthrough endpoint"):
        await provider.passthrough(PassthroughAction.OPENAI_CHAT, _claude_passthrough_payload())


def test_claude_passthrough_token_usage_is_read_from_anthropic_response():
    provider = _make_claude_provider()
    usage = provider._extract_passthrough_token_usage(
        PassthroughAction.ANTHROPIC_MESSAGES, _claude_chat_response()
    )
    assert usage == {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}


@pytest.mark.parametrize("path", ["v1/messages", "/v1/messages", "v1/messages?beta=true"])
@pytest.mark.asyncio
async def test_claude_proxy_maps_messages_path_to_raw_predict(path):
    provider = _make_claude_provider()
    mock_client = mock_http_client(MockAsyncResponse(_claude_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        response = await provider.proxy(
            path, _claude_passthrough_payload(model="claude-sonnet-4-5@20251101")
        )

    assert response["content"][0]["text"] == "Hello from Claude on Vertex AI!"
    mock_client.post.assert_called_once_with(
        "https://us-east5-aiplatform.googleapis.com/v1/projects/my-gcp-project"
        "/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20251101:rawPredict",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64,
            "anthropic_version": "vertex-2023-10-16",
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get()),
        allow_redirects=False,
    )


@pytest.mark.asyncio
async def test_claude_proxy_streams_from_stream_raw_predict_endpoint():
    provider = _make_claude_provider()
    stream_data = _claude_stream_data()
    mock_client = mock_http_client(
        MockAsyncStreamingResponse(stream_data, headers={"Content-Type": "text/event-stream"})
    )

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        result = await provider.proxy("v1/messages", _claude_passthrough_payload(stream=True))
        chunks = [chunk async for chunk in result]

    assert chunks == stream_data
    mock_client.post.assert_called_once()
    assert mock_client.post.call_args[0][0].endswith("/claude-sonnet-4-5@20251101:streamRawPredict")


@pytest.mark.asyncio
async def test_claude_proxy_rejects_paths_other_than_messages():
    provider = _make_claude_provider()
    with pytest.raises(AIGatewayException, match="only exposes the Messages API"):
        await provider.proxy("v1/complete", {"prompt": "Hello"})


def _make_maas_provider(model_name: str, location: str = "us-central1") -> VertexAIProvider:
    endpoint_config = EndpointConfig(
        name="vertex-maas-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": model_name,
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": location,
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    return provider


@pytest.mark.parametrize(
    "model_name",
    [
        "meta/llama-3.1-405b-instruct-maas",
        "mistral-large-2411",
        "codestral-2501",
        "jamba-1.5",
        "deepseek-ai/deepseek-r1-0528-maas",
        "xai/grok-4.1-fast-reasoning",
        "qwen/qwen3-235b-a22b-instruct-2507-maas",
    ],
)
def test_maas_model_uses_openapi_endpoint(model_name):
    provider = _make_maas_provider(model_name)
    assert provider._model_type == "maas"
    assert provider._delegate is not None
    assert provider._delegate._api_base == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi"
    )


@pytest.mark.parametrize("location", ["eu", "us"])
def test_maas_model_multi_region_location(location):
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas", location=location)
    assert provider._delegate._api_base == (
        f"https://aiplatform.{location}.rep.googleapis.com"
        f"/v1/projects/my-gcp-project/locations/{location}/endpoints/openapi"
    )


def _maas_chat_response():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "meta/llama-3.1-405b-instruct-maas",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Llama on Vertex AI!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


@pytest.mark.asyncio
async def test_maas_chat_uses_openai_format():
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")

    with (
        mock.patch(
            "aiohttp.ClientSession.post", return_value=MockAsyncResponse(_maas_chat_response())
        ) as mock_post,
    ):
        payload = chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}])
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["choices"][0]["message"]["content"] == "Hello from Llama on Vertex AI!"
    assert result["usage"]["prompt_tokens"] == 10

    mock_post.assert_called_once_with(
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi/chat/completions",
        json={
            "model": "meta/llama-3.1-405b-instruct-maas",
            "n": 1,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get()),
        allow_redirects=False,
    )


@pytest.mark.asyncio
async def test_maas_passthrough_uses_openapi_endpoint():
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")

    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(_maas_chat_response())
    ) as mock_post:
        response = await provider.passthrough(
            PassthroughAction.OPENAI_CHAT, {"messages": [{"role": "user", "content": "Hello"}]}
        )

    assert response["choices"][0]["message"]["content"] == "Hello from Llama on Vertex AI!"
    mock_post.assert_called_once_with(
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi/chat/completions",
        json={
            "model": "meta/llama-3.1-405b-instruct-maas",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get()),
        allow_redirects=False,
    )


def test_maas_passthrough_headers_drop_credential_agent_auth():
    # OpenAICompatibleProvider swaps the provider Authorization for a credential agent's
    # own. On Vertex that would replace the OAuth token with an unusable client token.
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")
    merged = provider._delegate._get_headers(
        headers={"user-agent": "codex_cli_rs/0.50.0", "authorization": "Bearer client-token"}
    )
    assert merged == {
        "user-agent": "codex_cli_rs/0.50.0",
        "Authorization": "Bearer mock-access-token",
    }


def test_claude_get_endpoint_url():
    provider = _make_claude_provider()
    assert provider.get_endpoint_url("llm/v1/chat") == (
        "https://us-east5-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-east5/publishers/anthropic/models"
        "/claude-sonnet-4-5@20251101:rawPredict"
    )


def test_maas_get_endpoint_url():
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")
    assert provider.get_endpoint_url("llm/v1/chat") == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi/chat/completions"
    )


@pytest.mark.parametrize(
    "path", ["chat/completions", "v1/chat/completions", "/v1/chat/completions"]
)
@pytest.mark.asyncio
async def test_maas_proxy_posts_to_openapi_endpoint(path):
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")
    resp = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "meta/llama-3.1-405b-instruct-maas",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
        ],
    }
    payload = {
        "model": "meta/llama-3.1-405b-instruct-maas",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    captured_session_headers = {}
    mock_client = mock_http_client(MockAsyncResponse(resp))

    def mock_client_session(headers=None, **kwargs):
        captured_session_headers.update(headers or {})
        return mock_client

    with mock.patch("aiohttp.ClientSession", mock_client_session):
        response = await provider.proxy(
            path, payload, headers={"authorization": "Bearer client-token", "x-request-id": "req-1"}
        )

    assert response == resp
    # The "/openapi" segment is part of the API root and must survive, unlike the "/v1"
    # suffix that OpenAICompatibleProvider._proxy is written to strip.
    mock_client.post.assert_called_once_with(
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi/chat/completions",
        json=payload,
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get()),
        allow_redirects=False,
    )
    assert captured_session_headers["Authorization"] == "Bearer mock-access-token"
    assert "authorization" not in captured_session_headers
    assert captured_session_headers["x-request-id"] == "req-1"


@pytest.mark.asyncio
async def test_maas_proxy_streams_when_upstream_sends_event_stream():
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")
    stream_data = [
        b'data: {"id": "chatcmpl-123", "choices": [{"index": 0, "delta": {"content": "Hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    mock_client = mock_http_client(
        MockAsyncStreamingResponse(stream_data, headers={"Content-Type": "text/event-stream"})
    )

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        result = await provider.proxy(
            "v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )
        chunks = [chunk async for chunk in result]

    assert chunks == stream_data
    mock_client.post.assert_called_once()
    assert mock_client.post.call_args[0][0].endswith("/endpoints/openapi/chat/completions")


@pytest.mark.asyncio
async def test_claude_completions_raises_gateway_exception():
    provider = _make_claude_provider()
    with pytest.raises(AIGatewayException, match="completions endpoint is not supported"):
        await provider.completions(completions.RequestPayload(prompt="hello", max_tokens=10))


def test_with_credentials():
    creds_json = json.dumps({"type": "service_account", "project_id": "test"})
    config = VertexAIConfig(vertex_project="my-project", vertex_credentials=creds_json)
    assert config.vertex_credentials == creds_json


def test_project_required():
    with pytest.raises(ValidationError, match="vertex_project"):
        VertexAIConfig()


def test_credentials_with_adc():
    config = VertexAIConfig(vertex_project="my-project")
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-2.0-flash",
            "config": config.model_dump(),
        },
    )
    provider = VertexAIProvider(endpoint_config)

    mock_creds = _mock_credentials()
    with mock.patch("google.auth.default", return_value=(mock_creds, "my-project")) as mock_default:
        headers = provider.headers
        mock_default.assert_called_once()
        assert headers == {"Authorization": "Bearer mock-access-token"}
