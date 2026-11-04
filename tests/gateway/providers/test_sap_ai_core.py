from typing import Any
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.sap_ai_core import (
    SapAiCoreAdapter,
    SapAiCoreConfig,
    SapAiCoreProvider,
)


def _make_endpoint_config(model_name: str = "gpt-4o-mini") -> EndpointConfig:
    return EndpointConfig(
        name="sap-ai-core-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "sap-ai-core",
            "name": model_name,
            "config": {},
        },
    )


def _make_provider(model_name: str = "gpt-4o-mini") -> SapAiCoreProvider:
    return SapAiCoreProvider(_make_endpoint_config(model_name))


def _make_orch_response(content: str = "Hello!") -> dict[str, Any]:
    inner = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1783313910,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return {
        "request_id": "ff3cb38e-215f-9c58-bb7f-45ee667d2d35",
        "intermediate_results": {"templating": [], "llm": inner},
        "final_result": inner,
    }


# ---------------------------------------------------------------------------
# SapAiCoreAdapter — request transformation
# ---------------------------------------------------------------------------


def test_adapter_basic_user_message():
    config = _make_endpoint_config()
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert result["config"]["modules"]["prompt_templating"]["prompt"]["template"] == [
        {"role": "user", "content": "Hello"}
    ]
    assert result["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o-mini"
    assert result["placeholder_values"] == {}


def test_adapter_system_and_user_messages():
    config = _make_endpoint_config()
    payload = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is MLflow?"},
        ]
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assert len(template) == 2
    assert template[0] == {"role": "system", "content": "You are helpful."}
    assert template[1] == {"role": "user", "content": "What is MLflow?"}


def test_adapter_model_name_from_config():
    config = _make_endpoint_config(model_name="gpt-4o")
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert result["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o"


def test_adapter_inference_params_forwarded():
    config = _make_endpoint_config()
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 300,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

    assert params["max_tokens"] == 300
    assert params["temperature"] == 0.5
    assert params["top_p"] == 0.9


def test_adapter_response_format_forwarded():
    config = _make_endpoint_config()
    response_format = {"type": "json_schema", "json_schema": {"name": "Out", "strict": True}}
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": response_format,
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

    assert params["response_format"] == response_format


def test_adapter_tools_forwarded():
    config = _make_endpoint_config()
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": "auto",
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

    assert params["tools"] == tools
    assert params["tool_choice"] == "auto"


def test_adapter_no_params_block_when_no_inference_params():
    config = _make_endpoint_config()
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    model_block = result["config"]["modules"]["prompt_templating"]["model"]

    assert "params" not in model_block


def test_adapter_unknown_keys_not_forwarded():
    config = _make_endpoint_config()
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert "stream" not in result
    assert "params" not in result["config"]["modules"]["prompt_templating"]["model"]


def test_adapter_placeholder_values_forwarded():
    config = _make_endpoint_config()
    placeholder = {"user_name": "Alice"}
    payload = {"messages": [{"role": "user", "content": "hi"}], "placeholder_values": placeholder}
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert result["placeholder_values"] == placeholder


def test_adapter_none_content_becomes_empty_string_for_user():
    config = _make_endpoint_config()
    payload = {"messages": [{"role": "user", "content": None}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assert template[0]["content"] == ""


def test_adapter_assistant_none_content_omitted():
    config = _make_endpoint_config()
    payload = {"messages": [{"role": "assistant", "content": None}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assert "content" not in template[0]


def test_adapter_preserves_assistant_tool_calls():
    config = _make_endpoint_config()
    payload = {
        "messages": [
            {"role": "user", "content": "Judge this trace."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_trace_info", "arguments": "{}"},
                    }
                ],
            },
        ]
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assert len(template) == 2
    assistant_msg = template[1]
    assert assistant_msg["role"] == "assistant"
    assert "content" not in assistant_msg
    assert "tool_calls" in assistant_msg
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_trace_info"


def test_adapter_preserves_tool_message_tool_call_id():
    config = _make_endpoint_config()
    payload = {
        "messages": [
            {
                "role": "tool",
                "content": '{"spans": 3}',
                "tool_call_id": "call_1",
                "name": "get_trace_info",
            }
        ]
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    tool_msg = template[0]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == '{"spans": 3}'
    assert tool_msg["tool_call_id"] == "call_1"
    assert "name" not in tool_msg


def test_adapter_tool_calls_full_round_trip():
    config = _make_endpoint_config()
    tools = [{"type": "function", "function": {"name": "get_trace_info", "parameters": {}}}]
    payload = {
        "messages": [
            {"role": "user", "content": "Judge this trace."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_trace_info", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"spans": 3}',
                "tool_call_id": "call_1",
                "name": "get_trace_info",
            },
        ],
        "tools": tools,
        "tool_choice": "auto",
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]
    params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

    assert len(template) == 3

    user_msg = template[0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "Judge this trace."

    assistant_msg = template[1]
    assert assistant_msg["role"] == "assistant"
    assert "content" not in assistant_msg
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"

    tool_msg = template[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert "name" not in tool_msg

    assert params["tools"] == tools
    assert params["tool_choice"] == "auto"


def test_adapter_multiple_tool_calls():
    config = _make_endpoint_config()
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_trace_info", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_span_info", "arguments": '{"span_id": 1}'},
                    },
                ],
            },
            {
                "role": "tool",
                "content": '{"spans": 3}',
                "tool_call_id": "call_1",
                "name": "get_trace_info",
            },
            {
                "role": "tool",
                "content": '{"name": "main"}',
                "tool_call_id": "call_2",
                "name": "get_span_info",
            },
        ]
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assistant_msg = template[0]
    assert len(assistant_msg["tool_calls"]) == 2
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][1]["id"] == "call_2"

    tool1 = template[1]
    assert tool1["tool_call_id"] == "call_1"
    tool2 = template[2]
    assert tool2["tool_call_id"] == "call_2"


def test_adapter_does_not_forward_unknown_fields():
    config = _make_endpoint_config()
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": "thinking...",
                "tool_calls": None,
                "unknown_field": "should_be_dropped",
            },
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call_1",
                "name": "func",
                "extra_key": "dropped",
            },
        ]
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assistant_msg = template[0]
    assert "unknown_field" not in assistant_msg
    assert "tool_calls" not in assistant_msg

    tool_msg = template[1]
    assert "name" not in tool_msg
    assert "extra_key" not in tool_msg


def test_adapter_forwards_assistant_refusal():
    config = _make_endpoint_config()
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "refusal": "I cannot help with that.",
            }
        ]
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assert template[0]["refusal"] == "I cannot help with that."
    assert "content" not in template[0]


# ---------------------------------------------------------------------------
# SapAiCoreAdapter — response decoding
# ---------------------------------------------------------------------------


def test_adapter_decodes_final_result():
    config = _make_endpoint_config()
    result = SapAiCoreAdapter.model_to_chat(_make_orch_response("Hello from AI Core!"), config)

    assert result.choices[0].message.content == "Hello from AI Core!"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5


def test_adapter_falls_back_to_root_when_no_final_result():
    config = _make_endpoint_config()
    bare = {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1783313910,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Bare"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    result = SapAiCoreAdapter.model_to_chat(bare, config)

    assert result.choices[0].message.content == "Bare"


def test_adapter_request_id_preserved():
    config = _make_endpoint_config()
    result = SapAiCoreAdapter.model_to_chat(_make_orch_response(), config)

    assert result.id == "chatcmpl-abc123"


def test_adapter_decodes_response_tool_calls():
    config = _make_endpoint_config()
    inner = {
        "id": "chatcmpl-toolcall",
        "object": "chat.completion",
        "created": 1783313910,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_trace_info",
                                "arguments": '{"trace_id": "tr-123"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    response = {
        "request_id": "req-toolcall",
        "intermediate_results": {},
        "final_result": inner,
    }
    result = SapAiCoreAdapter.model_to_chat(response, config)

    assert result.choices[0].message.content is None
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 1
    tc = result.choices[0].message.tool_calls[0]
    assert tc.id == "call_abc123"
    assert tc.function.name == "get_trace_info"
    assert tc.function.arguments == '{"trace_id": "tr-123"}'
    assert result.choices[0].finish_reason == "tool_calls"


class _MockAsyncStreamingResponse:
    def __init__(self, data: list[bytes], status: int = 200):
        self.status = status
        self.headers = {"Content-Type": "text/event-stream"}
        self._content = data

    def raise_for_status(self) -> None:
        pass

    async def _async_content(self):
        for line in self._content:
            yield line

    @property
    def content(self):
        return self._async_content()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass


class _MockHttpClient(mock.Mock):
    def __init__(self, mock_response=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_response = mock_response
        self.post = mock.Mock(return_value=mock_response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return


@pytest.mark.asyncio
async def test_adapter_streaming_tool_calls():
    from mlflow.gateway.schemas import chat

    provider = _make_provider()

    chunk1 = (
        b'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1783313910,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":null,'
        b'"tool_calls":[{"index":0,"id":"call_abc","type":"function",'
        b'"function":{"name":"get_trace_info","arguments":""}}]},"finish_reason":null}]}\n\n'
    )
    chunk2 = (
        b'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1783313910,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{'
        b'"tool_calls":[{"index":0,"function":{"arguments":"{\\"trace"}}]},"finish_reason":null}]}\n\n'
    )
    chunk3 = (
        b'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1783313910,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{'
        b'"tool_calls":[{"index":0,"function":{"arguments":"_id\\": \\"tr-1\\"}"}}]},'
        b'"finish_reason":null}]}\n\n'
    )
    chunk4 = (
        b'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1783313910,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":15,"total_tokens":25}}\n\n'
    )
    chunks = [chunk1, chunk2, chunk3, chunk4, b"data: [DONE]\n\n"]

    mock_client = _MockHttpClient(mock_response=_MockAsyncStreamingResponse(chunks))

    with (
        mock.patch("aiohttp.ClientSession", return_value=mock_client),
        mock.patch.dict(
            "os.environ", {"MLFLOW_GENAI_JUDGE_BASE_URL": "http://sap-orch.local/v2/chat"}
        ),
    ):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Get trace info"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_trace_info", "parameters": {}},
                }
            ],
        )
        responses = [chunk async for chunk in provider.chat_stream(payload)]

    assert len(responses) == 4

    first_chunk = responses[0]
    assert first_chunk.choices[0].delta.role == "assistant"
    assert first_chunk.choices[0].delta.tool_calls is not None
    assert len(first_chunk.choices[0].delta.tool_calls) == 1
    tc = first_chunk.choices[0].delta.tool_calls[0]
    assert tc.index == 0
    assert tc.id == "call_abc"
    assert tc.function.name == "get_trace_info"

    last_chunk = responses[3]
    assert last_chunk.choices[0].finish_reason == "tool_calls"
    assert last_chunk.usage is not None
    assert last_chunk.usage.prompt_tokens == 10
    assert last_chunk.usage.completion_tokens == 15


@pytest.mark.asyncio
async def test_adapter_streaming_multiple_tool_calls():
    from mlflow.gateway.schemas import chat

    provider = _make_provider()

    chunk1 = (
        b'data: {"id":"chatcmpl-multi","object":"chat.completion.chunk","created":1783313910,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"tool_calls":[{"index":0,"id":"call_1","type":"function",'
        b'"function":{"name":"get_trace_info","arguments":"{}"}},{"index":1,"id":"call_2",'
        b'"type":"function","function":{"name":"get_span_info","arguments":"{}"}}]},'
        b'"finish_reason":null}]}\n\n'
    )
    chunk2 = (
        b'data: {"id":"chatcmpl-multi","object":"chat.completion.chunk","created":1783313910,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    )
    chunks = [chunk1, chunk2, b"data: [DONE]\n\n"]

    mock_client = _MockHttpClient(mock_response=_MockAsyncStreamingResponse(chunks))

    with (
        mock.patch("aiohttp.ClientSession", return_value=mock_client),
        mock.patch.dict(
            "os.environ", {"MLFLOW_GENAI_JUDGE_BASE_URL": "http://sap-orch.local/v2/chat"}
        ),
    ):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Get trace and span info"}],
        )
        responses = [chunk async for chunk in provider.chat_stream(payload)]

    assert len(responses) == 2

    first_chunk = responses[0]
    assert first_chunk.choices[0].delta.tool_calls is not None
    assert len(first_chunk.choices[0].delta.tool_calls) == 2

    tc1 = first_chunk.choices[0].delta.tool_calls[0]
    assert tc1.index == 0
    assert tc1.id == "call_1"
    assert tc1.function.name == "get_trace_info"

    tc2 = first_chunk.choices[0].delta.tool_calls[1]
    assert tc2.index == 1
    assert tc2.id == "call_2"
    assert tc2.function.name == "get_span_info"

    assert responses[1].choices[0].finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# SapAiCoreProvider
# ---------------------------------------------------------------------------


def test_provider_adapter_class():
    assert _make_provider().adapter_class is SapAiCoreAdapter


def test_provider_headers_are_empty():
    assert _make_provider().headers == {}


def test_provider_display_name():
    assert SapAiCoreProvider.DISPLAY_NAME == "SAP AI Core"


def test_provider_config_type():
    assert SapAiCoreProvider.CONFIG_TYPE is SapAiCoreConfig


def test_provider_config_requires_no_api_key():
    assert SapAiCoreConfig() is not None


def test_provider_get_endpoint_url_reads_env(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_GENAI_JUDGE_BASE_URL",
        "http://egress-gw.cluster.local/v2/inference/deployments/abc/chat/completions",
    )
    url = _make_provider().get_endpoint_url("llm/v1/chat")

    assert url == "http://egress-gw.cluster.local/v2/inference/deployments/abc/chat/completions"


def test_provider_get_endpoint_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "http://egress-gw.cluster.local/v2/chat/")
    assert not _make_provider().get_endpoint_url("llm/v1/chat").endswith("/")


def test_provider_get_endpoint_url_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_GENAI_JUDGE_BASE_URL", raising=False)
    with pytest.raises(MlflowException, match="MLFLOW_GENAI_JUDGE_BASE_URL"):
        _make_provider().get_endpoint_url("llm/v1/chat")


def test_provider_get_endpoint_url_raises_on_invalid_scheme(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "ftp://bad-scheme/completions")
    with pytest.raises(MlflowException, match="http://"):
        _make_provider().get_endpoint_url("llm/v1/chat")


def test_provider_http_scheme_accepted(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "http://internal-gw/completions")
    assert _make_provider().get_endpoint_url("llm/v1/chat").startswith("http://")


def test_provider_https_scheme_accepted(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "https://public-api.example.com/completions")
    assert _make_provider().get_endpoint_url("llm/v1/chat").startswith("https://")


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


def test_sap_ai_core_is_supported_provider():
    from mlflow.gateway.provider_registry import is_supported_provider

    assert is_supported_provider("sap-ai-core")


def test_sap_ai_core_in_provider_enum():
    from mlflow.gateway.config import Provider

    assert Provider.SAP_AI_CORE == "sap-ai-core"


# ---------------------------------------------------------------------------
# GatewayAdapter integration
# ---------------------------------------------------------------------------


def test_gateway_adapter_is_applicable_for_sap_ai_core():
    from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
    from mlflow.types.llm import ChatMessage

    assert GatewayAdapter.is_applicable(
        "sap-ai-core:/gpt-4o-mini", [ChatMessage(role="user", content="hello")]
    )


def test_gateway_adapter_sends_v2_body_and_extra_headers(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_GENAI_JUDGE_BASE_URL",
        "http://egress-gw.cluster.local/v2/chat/completions",
    )

    final_result = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1783313910,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"result": "yes", "rationale": "looks good"}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response = {
        "request_id": "req-123",
        "intermediate_results": {},
        "final_result": final_result,
    }

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
        return_value=fake_response,
    ) as mock_send:
        from mlflow.genai.judges.adapters.base_adapter import AdapterInvocationInput
        from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
        from mlflow.types.llm import ChatMessage

        adapter = GatewayAdapter()
        params = AdapterInvocationInput(
            model_uri="sap-ai-core:/gpt-4o-mini",
            prompt=[ChatMessage(role="user", content="Is this safe?")],
            assessment_name="safety",
            extra_headers={"AI-Resource-Group": "default", "x-tenant-id": "tenant-1"},
        )
        result = adapter._invoke(params)

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs

    assert call_kwargs["endpoint"] == "http://egress-gw.cluster.local/v2/chat/completions"
    assert call_kwargs["headers"].get("AI-Resource-Group") == "default"
    assert call_kwargs["headers"].get("x-tenant-id") == "tenant-1"
    assert "Authorization" not in call_kwargs["headers"]
    assert "config" in call_kwargs["payload"]
    assert (
        call_kwargs["payload"]["config"]["modules"]["prompt_templating"]["model"]["name"]
        == "gpt-4o-mini"
    )
    assert result.feedback.value in ("yes", "no")
