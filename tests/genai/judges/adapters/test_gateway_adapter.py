import json
from unittest import mock

import pydantic
import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.base_adapter import AdapterInvocationInput
from mlflow.genai.judges.adapters.gateway_adapter import (
    GatewayAdapter,
    InvokeOutput,
    _build_request,
    _get_max_context_tokens,
    _get_mlflow_gateway_provider,
    _get_provider,
    _parse_response_message,
    _should_proactively_prune,
)
from mlflow.genai.judges.adapters.utils import ChatCompletionError, is_context_window_error
from mlflow.genai.judges.utils.tool_calling_utils import _remove_oldest_tool_call_pair
from mlflow.types.llm import ChatMessage


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


def _chat_response(content, tool_calls=None, usage=None):
    """Build a mock OpenAI-format chat completions response dict."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _mock_provider(endpoint_url="https://api.openai.com/v1/chat/completions", headers=None):
    from mlflow.gateway.providers.openai_compatible import OpenAICompatibleAdapter

    provider = mock.MagicMock()
    provider.get_endpoint_url.return_value = endpoint_url
    provider.headers = headers or {}
    provider.adapter_class = OpenAICompatibleAdapter
    # config.model.name is used by chat_to_model to add "model" to the payload
    provider.config.model.name = "gpt-4"
    return provider


# --- is_applicable tests ---


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "mistral", "gateway"])
def test_native_providers_applicable(provider):
    assert GatewayAdapter.is_applicable(model_uri=f"{provider}:/test-model", prompt="test")


def test_unsupported_provider_not_applicable():
    assert not GatewayAdapter.is_applicable(model_uri="unknown_provider:/test-model", prompt="test")


def test_endpoints_with_string_prompt_applicable():
    assert GatewayAdapter.is_applicable(model_uri="endpoints:/my-endpoint", prompt="test prompt")


def test_endpoints_with_list_prompt_not_applicable():
    assert not GatewayAdapter.is_applicable(
        model_uri="endpoints:/my-endpoint",
        prompt=[ChatMessage(role="user", content="test")],
    )


# --- invoke with trace tests ---


def test_invoke_with_trace_calls_tool_loop(mock_trace):
    mock_output = InvokeOutput(
        response=json.dumps({"result": "yes", "rationale": "Looks good"}),
        request_id="req-123",
        num_prompt_tokens=10,
        num_completion_tokens=5,
    )

    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="openai:/gpt-4",
        prompt=[ChatMessage(role="user", content="evaluate this")],
        assessment_name="test_metric",
        trace=mock_trace,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._invoke_and_handle_tools",
        return_value=mock_output,
    ) as mock_invoke:
        result = adapter.invoke(input_params)

    mock_invoke.assert_called_once()
    call_kwargs = mock_invoke.call_args[1]
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["model_name"] == "gpt-4"
    assert call_kwargs["trace"] is mock_trace

    assert result.feedback.name == "test_metric"
    assert result.feedback.value == "yes"
    assert result.feedback.rationale == "Looks good"
    assert result.feedback.trace_id == "test-trace"


def test_invoke_with_trace_string_prompt(mock_trace):
    mock_output = InvokeOutput(
        response=json.dumps({"result": "no", "rationale": "Bad"}),
        request_id=None,
        num_prompt_tokens=None,
        num_completion_tokens=None,
    )

    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="openai:/gpt-4",
        prompt="evaluate this string",
        assessment_name="test_metric",
        trace=mock_trace,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._invoke_and_handle_tools",
        return_value=mock_output,
    ) as mock_invoke:
        result = adapter.invoke(input_params)

    mock_invoke.assert_called_once()
    assert result.feedback.value == "no"


# --- invoke without trace tests ---


def test_invoke_without_trace_uses_gateway():
    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="openai:/gpt-4",
        prompt=[ChatMessage(role="user", content="test")],
        assessment_name="test_metric",
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._invoke_via_gateway",
        return_value=json.dumps({"result": "yes", "rationale": "ok"}),
    ) as mock_invoke:
        result = adapter.invoke(input_params)

    mock_invoke.assert_called_once()
    assert result.feedback.value == "yes"


# --- invoke_with_structured_output tests ---


def test_invoke_with_structured_output_success():
    class TestSchema(pydantic.BaseModel):
        result: str
        confidence: float

    adapter = GatewayAdapter()
    mock_output = InvokeOutput(
        response=json.dumps({"result": "yes", "confidence": 0.95}),
        request_id="req-1",
        num_prompt_tokens=10,
        num_completion_tokens=5,
    )

    with mock.patch.object(adapter, "_invoke_and_handle_tools", return_value=mock_output):
        result = adapter.invoke_with_structured_output(
            model_uri="openai:/gpt-4",
            messages=[ChatMessage(role="user", content="test")],
            output_schema=TestSchema,
        )

    assert isinstance(result, TestSchema)
    assert result.result == "yes"
    assert result.confidence == 0.95


def test_invoke_with_structured_output_invalid_json():
    class TestSchema(pydantic.BaseModel):
        result: str

    adapter = GatewayAdapter()
    mock_output = InvokeOutput(
        response="not valid json",
        request_id=None,
        num_prompt_tokens=None,
        num_completion_tokens=None,
    )

    with mock.patch.object(adapter, "_invoke_and_handle_tools", return_value=mock_output):
        with pytest.raises(MlflowException, match="Failed to parse response"):
            adapter.invoke_with_structured_output(
                model_uri="openai:/gpt-4",
                messages=[ChatMessage(role="user", content="test")],
                output_schema=TestSchema,
            )


# --- endpoint restriction tests ---


def test_endpoints_rejects_base_url():
    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="endpoints:/my-ep",
        prompt="test",
        assessment_name="test_metric",
        base_url="http://proxy:8080",
    )

    with pytest.raises(MlflowException, match="base_url and extra_headers are not supported"):
        adapter.invoke(input_params)


def test_endpoints_rejects_trace(mock_trace):
    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="endpoints:/my-ep",
        prompt="test",
        assessment_name="test_metric",
        trace=mock_trace,
    )

    with pytest.raises(MlflowException, match="Trace-based tool calling is not supported"):
        adapter.invoke(input_params)


# --- _invoke_via_gateway with gateway:/ URI tests ---


def _mock_gateway_config(api_base="http://localhost:5000/gateway/mlflow/v1/", extra_headers=None):
    from mlflow.genai.utils.gateway_utils import GatewayConfig

    return GatewayConfig(
        api_base=api_base,
        endpoint_name="my-endpoint",
        extra_headers=extra_headers,
    )


def test_invoke_via_gateway_string_prompt():
    from mlflow.genai.judges.adapters.gateway_adapter import _invoke_via_gateway

    response_json = {"choices": [{"message": {"content": '{"result": "yes", "rationale": "ok"}'}}]}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.get_gateway_config",
            return_value=_mock_gateway_config(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        result = _invoke_via_gateway(
            model_uri="gateway:/my-endpoint",
            provider="gateway",
            prompt="Is this helpful?",
        )

    assert '{"result": "yes"' in result
    call_kwargs = mock_send.call_args[1]
    # String prompt should be wrapped in a message
    assert call_kwargs["payload"]["messages"] == [{"role": "user", "content": "Is this helpful?"}]
    # Model should be endpoint name, not litellm format
    assert call_kwargs["payload"]["model"] == "my-endpoint"


def test_invoke_via_gateway_list_prompt():
    from mlflow.genai.judges.adapters.gateway_adapter import _invoke_via_gateway

    response_json = {"choices": [{"message": {"content": '{"result": "no", "rationale": "bad"}'}}]}
    messages = [
        {"role": "system", "content": "You are a judge"},
        {"role": "user", "content": "test"},
    ]

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.get_gateway_config",
            return_value=_mock_gateway_config(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        _invoke_via_gateway(
            model_uri="gateway:/my-endpoint",
            provider="gateway",
            prompt=messages,
        )

    call_kwargs = mock_send.call_args[1]
    assert call_kwargs["payload"]["messages"] == messages
    assert call_kwargs["payload"]["model"] == "my-endpoint"


def test_invoke_via_gateway_sets_caller_header():
    from mlflow.gateway.constants import MLFLOW_GATEWAY_CALLER_HEADER, GatewayCaller
    from mlflow.genai.judges.adapters.gateway_adapter import _invoke_via_gateway

    response_json = {"choices": [{"message": {"content": "ok"}}]}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.get_gateway_config",
            return_value=_mock_gateway_config(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        _invoke_via_gateway(model_uri="gateway:/my-endpoint", provider="gateway", prompt="test")

    call_kwargs = mock_send.call_args[1]
    assert MLFLOW_GATEWAY_CALLER_HEADER in call_kwargs["headers"]
    assert call_kwargs["headers"][MLFLOW_GATEWAY_CALLER_HEADER] == GatewayCaller.JUDGE.value


def test_invoke_via_gateway_with_inference_params():
    from mlflow.genai.judges.adapters.gateway_adapter import _invoke_via_gateway

    response_json = {"choices": [{"message": {"content": "ok"}}]}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.get_gateway_config",
            return_value=_mock_gateway_config(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        _invoke_via_gateway(
            model_uri="gateway:/my-endpoint",
            provider="gateway",
            prompt="test",
            inference_params={"temperature": 0.5},
        )

    call_kwargs = mock_send.call_args[1]
    assert call_kwargs["payload"]["temperature"] == 0.5


def test_invoke_via_gateway_endpoint_url():
    from mlflow.genai.judges.adapters.gateway_adapter import _invoke_via_gateway

    response_json = {"choices": [{"message": {"content": "ok"}}]}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.get_gateway_config",
            return_value=_mock_gateway_config(api_base="http://myserver:8080/gateway/v1/"),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        _invoke_via_gateway(model_uri="gateway:/my-endpoint", provider="gateway", prompt="test")

    call_kwargs = mock_send.call_args[1]
    assert call_kwargs["endpoint"] == "http://myserver:8080/gateway/v1/chat/completions"


# --- _build_request tests ---


def test_build_request_basic():
    messages = [ChatMessage(role="user", content="hello")]
    payload = _build_request(
        messages=messages,
        tools=None,
        response_format=None,
        include_response_format=False,
        inference_params=None,
    )
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in payload
    assert "response_format" not in payload


def test_build_request_with_tools():
    messages = [ChatMessage(role="user", content="test")]
    tools = [{"type": "function", "function": {"name": "get_info", "parameters": {}}}]
    payload = _build_request(
        messages=messages,
        tools=tools,
        response_format=None,
        include_response_format=False,
        inference_params=None,
    )
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


def test_build_request_with_inference_params():
    messages = [ChatMessage(role="user", content="test")]
    payload = _build_request(
        messages=messages,
        tools=None,
        response_format=None,
        include_response_format=False,
        inference_params={"temperature": 0.5, "max_tokens": 100},
    )
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100


# --- _parse_response_message tests ---


def test_parse_response_basic():
    resp = _chat_response("Hello!")
    msg, usage = _parse_response_message(resp, provider=_mock_provider())
    assert isinstance(msg, ChatMessage)
    assert msg.role == "assistant"
    assert msg.content == "Hello!"
    assert msg.tool_calls is None


def test_parse_response_with_tool_calls():
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
    ]
    resp = _chat_response(None, tool_calls=tool_calls)
    msg, usage = _parse_response_message(resp, provider=_mock_provider())
    assert msg.content is None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_1"


def test_parse_response_empty_choices_raises():
    resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    with pytest.raises(MlflowException, match="Empty choices"):
        _parse_response_message(resp, provider=_mock_provider())


def test_parse_anthropic_response_with_tool_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    provider = _get_provider("anthropic", "claude-3-5-sonnet")

    anthropic_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [
            {"type": "text", "text": "Let me check that."},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_root_span",
                "input": {"trace_id": "tr-123"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "output_tokens": 30},
    }

    msg, usage = _parse_response_message(anthropic_response, provider)
    assert msg.role == "assistant"
    assert msg.content == "Let me check that."
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "toolu_abc"
    assert msg.tool_calls[0].function.name == "get_root_span"


def test_parse_anthropic_response_text_only(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    provider = _get_provider("anthropic", "claude-3-5-sonnet")

    anthropic_response = {
        "id": "msg_456",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [{"type": "text", "text": '{"result": "yes", "rationale": "Looks good"}'}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    msg, usage = _parse_response_message(anthropic_response, provider)
    assert msg.role == "assistant"
    assert msg.tool_calls is None
    parsed = json.loads(msg.content)
    assert parsed["result"] == "yes"


# --- _remove_oldest_tool_call_pair tests ---


def test_remove_tool_call_pair():
    messages = [
        ChatMessage(role="user", content="test"),
        ChatMessage(
            role="assistant",
            tool_calls=[{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}],
        ),
        ChatMessage(role="tool", content="result", tool_call_id="call_1", name="f"),
        ChatMessage(role="assistant", content="final answer"),
    ]
    result = _remove_oldest_tool_call_pair(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[1].content == "final answer"


def test_remove_no_tool_calls_returns_none():
    messages = [
        ChatMessage(role="user", content="test"),
        ChatMessage(role="assistant", content="answer"),
    ]
    assert _remove_oldest_tool_call_pair(messages) is None


def test_remove_oldest_tool_call_pair_with_dict_tool_calls():
    msg_user = mock.MagicMock(role="user", tool_calls=None, tool_call_id=None)
    msg_assistant = mock.MagicMock(
        role="assistant",
        tool_calls=[{"id": "call_1", "function": {"name": "f"}}],
        tool_call_id=None,
    )
    msg_tool = mock.MagicMock(role="tool", tool_calls=None, tool_call_id="call_1")
    msg_final = mock.MagicMock(role="assistant", tool_calls=None, tool_call_id=None, content="done")

    result = _remove_oldest_tool_call_pair([msg_user, msg_assistant, msg_tool, msg_final])
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"


# --- is_context_window_error tests ---


@pytest.mark.parametrize(
    "message",
    [
        "maximum context length exceeded",
        "This model's maximum context length is 8192 tokens",
        "Request too large: too many tokens",
    ],
)
def test_context_window_errors(message):
    assert is_context_window_error(message)


def test_non_context_error():
    assert not is_context_window_error("invalid api key")


# --- _get_provider tests ---


def test_get_provider_delegates_to_get_provider_instance(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = _get_provider("openai", "gpt-4")
    assert provider.adapter_class is not None
    assert provider.config is not None
    assert provider.config.model.name == "gpt-4"
    assert "openai.com" in provider.get_endpoint_url("llm/v1/chat")
    header_keys_lower = {k.lower() for k in provider.headers}
    assert "authorization" in header_keys_lower or "api-key" in header_keys_lower


def test_get_provider_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    provider = _get_provider("anthropic", "claude-3-5-sonnet")
    assert provider.config.model.name == "claude-3-5-sonnet"
    assert "anthropic.com" in provider.get_endpoint_url("llm/v1/chat")


def test_get_mlflow_gateway_provider():
    mock_config = mock.MagicMock()
    mock_config.api_base = "http://localhost:5000/gateway/mlflow/v1/"
    mock_config.extra_headers = None

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.get_gateway_config",
        return_value=mock_config,
    ):
        provider = _get_mlflow_gateway_provider("my-endpoint")

    assert (
        provider.get_endpoint_url("llm/v1/chat")
        == "http://localhost:5000/gateway/mlflow/v1/chat/completions"
    )
    assert provider.config is not None
    assert provider.config.model.name == "my-endpoint"
    payload = provider.adapter_class.chat_to_model(
        {"messages": [{"role": "user", "content": "hi"}]}, provider.config
    )
    assert payload["model"] == "my-endpoint"


def test_get_provider_unsupported_raises():
    with pytest.raises(MlflowException, match="not supported"):
        _get_provider("cohere", "command-r")


# --- _invoke_and_handle_tools tests ---


def test_single_shot_no_tools():
    response_json = _chat_response(json.dumps({"result": "yes", "rationale": "Looks good"}))

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_provider",
            return_value=_mock_provider(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        output = GatewayAdapter()._invoke_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="test")],
            trace=None,
            num_retries=3,
        )

    assert isinstance(output, InvokeOutput)
    assert json.loads(output.response) == {"result": "yes", "rationale": "Looks good"}
    assert output.num_prompt_tokens == 10
    assert output.num_completion_tokens == 5
    mock_send.assert_called_once()


def test_tool_calling_loop(mock_trace):
    tool_call_response = _chat_response(
        None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_trace_info", "arguments": "{}"},
            }
        ],
    )
    final_response = _chat_response(json.dumps({"result": "yes", "rationale": "Trace looks good"}))

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_provider",
            return_value=_mock_provider(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            side_effect=[tool_call_response, final_response],
        ) as mock_send,
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._process_tool_calls",
        ) as mock_process,
    ):
        mock_process.return_value = [
            ChatMessage(
                role="tool",
                content='{"trace_id": "test-trace"}',
                tool_call_id="call_1",
                name="get_trace_info",
            )
        ]

        output = GatewayAdapter()._invoke_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="evaluate this")],
            trace=mock_trace,
            num_retries=3,
        )

    assert mock_send.call_count == 2
    mock_process.assert_called_once()
    assert json.loads(output.response) == {"result": "yes", "rationale": "Trace looks good"}


def test_context_window_error_triggers_pruning(mock_trace):
    final_response = _chat_response(json.dumps({"result": "yes", "rationale": "ok"}))

    tool_call_response = _chat_response(
        None,
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ],
    )

    call_count = 0

    def mock_send_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_call_response
        elif call_count == 2:
            raise ChatCompletionError(
                status_code=400,
                message="maximum context length exceeded",
                is_context_window_error=True,
            )
        else:
            return final_response

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_provider",
            return_value=_mock_provider(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            side_effect=mock_send_side_effect,
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._process_tool_calls",
            return_value=[ChatMessage(role="tool", content="{}", tool_call_id="c1", name="f")],
        ),
    ):
        output = GatewayAdapter()._invoke_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="test")],
            trace=mock_trace,
            num_retries=3,
        )

    assert call_count == 3
    assert json.loads(output.response) == {"result": "yes", "rationale": "ok"}


def test_max_iterations_exceeded(mock_trace, monkeypatch):
    tool_call_response = _chat_response(
        None,
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ],
    )

    monkeypatch.setenv("MLFLOW_JUDGE_MAX_ITERATIONS", "1")

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_provider",
            return_value=_mock_provider(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=tool_call_response,
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._process_tool_calls",
            return_value=[ChatMessage(role="tool", content="{}", tool_call_id="c1", name="f")],
        ),
    ):
        with pytest.raises(MlflowException, match="iteration limit of 1 exceeded"):
            GatewayAdapter()._invoke_and_handle_tools(
                provider="openai",
                model_name="gpt-4",
                messages=[ChatMessage(role="user", content="test")],
                trace=mock_trace,
                num_retries=3,
            )


def test_response_format_fallback():
    class TestSchema(pydantic.BaseModel):
        result: str = pydantic.Field(description="test")

    final_response = _chat_response(json.dumps({"result": "yes"}))

    call_count = 0

    def mock_send_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ChatCompletionError(
                status_code=400,
                message="response_format is not supported",
            )
        return final_response

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_provider",
            return_value=_mock_provider(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            side_effect=mock_send_side_effect,
        ),
    ):
        output = GatewayAdapter()._invoke_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="test")],
            trace=None,
            num_retries=3,
            response_format=TestSchema,
        )

    assert call_count == 2
    assert json.loads(output.response) == {"result": "yes"}


def test_custom_base_url_overrides_provider():
    response_json = _chat_response(json.dumps({"result": "yes"}))

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_provider",
            return_value=_mock_provider(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        GatewayAdapter()._invoke_and_handle_tools(
            provider="anthropic",
            model_name="claude-3",
            messages=[ChatMessage(role="user", content="test")],
            trace=None,
            num_retries=3,
            base_url="https://custom.proxy.com/v1",
        )

    call_kwargs = mock_send.call_args[1]
    assert "custom.proxy.com" in call_kwargs["endpoint"]


# --- _get_max_context_tokens tests ---


def test_lookup_with_provider_prefix():
    mock_cost = {"openai/gpt-4": {"max_input_tokens": 128000}}
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._get_model_cost", return_value=mock_cost
    ):
        assert _get_max_context_tokens("openai", "gpt-4") == 128000


def test_lookup_without_provider_prefix():
    mock_cost = {"gpt-4": {"max_input_tokens": 8192}}
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._get_model_cost", return_value=mock_cost
    ):
        assert _get_max_context_tokens("openai", "gpt-4") == 8192


def test_lookup_missing_model():
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._get_model_cost", return_value={}
    ):
        assert _get_max_context_tokens("openai", "unknown-model") is None


def test_provider_prefix_takes_priority():
    mock_cost = {
        "openai/gpt-4": {"max_input_tokens": 128000},
        "gpt-4": {"max_input_tokens": 8192},
    }
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._get_model_cost", return_value=mock_cost
    ):
        assert _get_max_context_tokens("openai", "gpt-4") == 128000


# --- _should_proactively_prune tests ---


def test_prune_above_threshold():
    assert _should_proactively_prune(
        usage={"prompt_tokens": 9000}, max_context_tokens=10000, threshold=0.85
    )


def test_prune_below_threshold():
    assert not _should_proactively_prune(
        usage={"prompt_tokens": 5000}, max_context_tokens=10000, threshold=0.85
    )


def test_prune_no_max_context():
    assert not _should_proactively_prune(usage={"prompt_tokens": 9000}, max_context_tokens=None)


def test_prune_no_prompt_tokens_in_usage():
    assert not _should_proactively_prune(usage={}, max_context_tokens=10000)


def test_prune_exact_threshold():
    assert _should_proactively_prune(
        usage={"prompt_tokens": 8501}, max_context_tokens=10000, threshold=0.85
    )
    assert not _should_proactively_prune(
        usage={"prompt_tokens": 8500}, max_context_tokens=10000, threshold=0.85
    )


# --- Proactive pruning integration test ---


def test_proactive_pruning_triggers_during_tool_loop(mock_trace):
    tool_call_response = _chat_response(
        None,
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ],
        usage={"prompt_tokens": 9000, "completion_tokens": 100, "total_tokens": 9100},
    )
    final_response = _chat_response(json.dumps({"result": "yes"}))

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_provider",
            return_value=_mock_provider(),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
            side_effect=[tool_call_response, final_response],
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._process_tool_calls",
            return_value=[ChatMessage(role="tool", content="{}", tool_call_id="c1", name="f")],
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._get_max_context_tokens",
            return_value=10000,
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter._remove_oldest_tool_call_pair",
            wraps=_remove_oldest_tool_call_pair,
        ) as mock_prune,
    ):
        output = GatewayAdapter()._invoke_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="test")],
            trace=mock_trace,
            num_retries=3,
        )

    mock_prune.assert_called_once()
    assert json.loads(output.response) == {"result": "yes"}
