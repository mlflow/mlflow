import json
from unittest import mock

import pydantic
import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.gateway_invocation import (
    ChatCompletionError,
    InvokeOutput,
    _build_request,
    _get_max_context_tokens,
    _is_context_window_error,
    _parse_response_message,
    _remove_oldest_tool_call_pair,
    _resolve_provider_config,
    _should_proactively_prune,
    invoke_via_gateway_and_handle_tools,
)
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


# --- _build_request tests ---


def test_build_request_basic():
    messages = [ChatMessage(role="user", content="hello")]
    payload = _build_request(
        model="gpt-4",
        messages=messages,
        tools=None,
        response_format=None,
        include_response_format=False,
        inference_params=None,
    )
    assert payload["model"] == "gpt-4"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in payload
    assert "response_format" not in payload


def test_build_request_with_tools():
    messages = [ChatMessage(role="user", content="test")]
    tools = [{"type": "function", "function": {"name": "get_info", "parameters": {}}}]
    payload = _build_request(
        model="gpt-4",
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
        model="gpt-4",
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
    msg = _parse_response_message(resp)
    assert isinstance(msg, ChatMessage)
    assert msg.role == "assistant"
    assert msg.content == "Hello!"
    assert msg.tool_calls is None


def test_parse_response_with_tool_calls():
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
    ]
    resp = _chat_response(None, tool_calls=tool_calls)
    msg = _parse_response_message(resp)
    assert msg.content is None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_1"


def test_parse_response_empty_choices_raises():
    with pytest.raises(MlflowException, match="Empty choices"):
        _parse_response_message({"choices": []})


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


# --- _is_context_window_error tests ---


@pytest.mark.parametrize(
    "message",
    [
        "maximum context length exceeded",
        "This model's maximum context length is 8192 tokens",
        "Request too large: too many tokens",
    ],
)
def test_context_window_errors(message):
    assert _is_context_window_error(message)


def test_non_context_error():
    assert not _is_context_window_error("invalid api key")


# --- _resolve_provider_config tests ---


def test_resolve_openai_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    api_base, model, headers = _resolve_provider_config("openai", "gpt-4", None, None)
    assert api_base == "https://api.openai.com/v1"
    assert model == "gpt-4"
    assert headers["Authorization"] == "Bearer sk-test"


def test_resolve_custom_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    api_base, model, headers = _resolve_provider_config(
        "openai", "gpt-4", "https://custom.proxy.com/v1", None
    )
    assert api_base == "https://custom.proxy.com/v1"


def test_resolve_extra_headers_preserved(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _, _, headers = _resolve_provider_config("openai", "gpt-4", None, {"X-Custom": "value"})
    assert headers["X-Custom"] == "value"
    assert headers["Authorization"] == "Bearer sk-test"


def test_resolve_gateway_provider():
    mock_config = mock.MagicMock()
    mock_config.api_base = "http://localhost:5000/gateway/mlflow/v1/"
    mock_config.extra_headers = None

    with mock.patch(
        "mlflow.genai.utils.gateway_utils.get_gateway_litellm_config",
        return_value=mock_config,
    ):
        api_base, model, headers = _resolve_provider_config("gateway", "my-endpoint", None, None)
    assert api_base == "http://localhost:5000/gateway/mlflow/v1/"
    assert model == "my-endpoint"


# --- invoke_via_gateway_and_handle_tools tests ---


def test_single_shot_no_tools():
    response_json = _chat_response(json.dumps({"result": "yes", "rationale": "Looks good"}))

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._resolve_provider_config",
            return_value=("https://api.openai.com/v1", "gpt-4", {"Authorization": "Bearer k"}),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._send_chat_request",
            return_value=response_json,
        ) as mock_send,
    ):
        output = invoke_via_gateway_and_handle_tools(
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
            "mlflow.genai.judges.adapters.gateway_invocation._resolve_provider_config",
            return_value=("https://api.openai.com/v1", "gpt-4", {}),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._send_chat_request",
            side_effect=[tool_call_response, final_response],
        ) as mock_send,
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._process_tool_calls",
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

        output = invoke_via_gateway_and_handle_tools(
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

    # First call: tool call. Second call: context window error. Third call: success.
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
            "mlflow.genai.judges.adapters.gateway_invocation._resolve_provider_config",
            return_value=("https://api.openai.com/v1", "gpt-4", {}),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._send_chat_request",
            side_effect=mock_send_side_effect,
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._process_tool_calls",
            return_value=[ChatMessage(role="tool", content="{}", tool_call_id="c1", name="f")],
        ),
    ):
        output = invoke_via_gateway_and_handle_tools(
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
            "mlflow.genai.judges.adapters.gateway_invocation._resolve_provider_config",
            return_value=("https://api.openai.com/v1", "gpt-4", {}),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._send_chat_request",
            return_value=tool_call_response,
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._process_tool_calls",
            return_value=[ChatMessage(role="tool", content="{}", tool_call_id="c1", name="f")],
        ),
    ):
        with pytest.raises(MlflowException, match="iteration limit of 1 exceeded"):
            invoke_via_gateway_and_handle_tools(
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
            "mlflow.genai.judges.adapters.gateway_invocation._resolve_provider_config",
            return_value=("https://api.openai.com/v1", "gpt-4", {}),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._send_chat_request",
            side_effect=mock_send_side_effect,
        ),
    ):
        output = invoke_via_gateway_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="test")],
            trace=None,
            num_retries=3,
            response_format=TestSchema,
        )

    assert call_count == 2
    assert json.loads(output.response) == {"result": "yes"}


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_non_openai_provider_without_base_url_raises(provider):
    with pytest.raises(MlflowException, match="does not support the OpenAI-compatible"):
        invoke_via_gateway_and_handle_tools(
            provider=provider,
            model_name="test-model",
            messages=[ChatMessage(role="user", content="test")],
            trace=None,
            num_retries=3,
        )


def test_non_openai_provider_with_base_url_allowed():
    response_json = _chat_response(json.dumps({"result": "yes"}))

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._resolve_provider_config",
            return_value=("https://custom.proxy.com/v1", "model", {}),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._send_chat_request",
            return_value=response_json,
        ),
    ):
        output = invoke_via_gateway_and_handle_tools(
            provider="anthropic",
            model_name="claude-3",
            messages=[ChatMessage(role="user", content="test")],
            trace=None,
            num_retries=3,
            base_url="https://custom.proxy.com/v1",
        )

    assert json.loads(output.response) == {"result": "yes"}


# --- _get_max_context_tokens tests ---


def test_lookup_with_provider_prefix():
    mock_cost = {"openai/gpt-4": {"max_input_tokens": 128000}}
    with mock.patch("mlflow.utils.providers._get_model_cost", return_value=mock_cost):
        assert _get_max_context_tokens("openai", "gpt-4") == 128000


def test_lookup_without_provider_prefix():
    mock_cost = {"gpt-4": {"max_input_tokens": 8192}}
    with mock.patch("mlflow.utils.providers._get_model_cost", return_value=mock_cost):
        assert _get_max_context_tokens("openai", "gpt-4") == 8192


def test_lookup_missing_model():
    with mock.patch("mlflow.utils.providers._get_model_cost", return_value={}):
        assert _get_max_context_tokens("openai", "unknown-model") is None


def test_provider_prefix_takes_priority():
    mock_cost = {
        "openai/gpt-4": {"max_input_tokens": 128000},
        "gpt-4": {"max_input_tokens": 8192},
    }
    with mock.patch("mlflow.utils.providers._get_model_cost", return_value=mock_cost):
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
    # First response: tool call with high token usage (triggers proactive pruning)
    tool_call_response = _chat_response(
        None,
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ],
        usage={"prompt_tokens": 9000, "completion_tokens": 100, "total_tokens": 9100},
    )
    # Second response: final answer
    final_response = _chat_response(json.dumps({"result": "yes"}))

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._resolve_provider_config",
            return_value=("https://api.openai.com/v1", "gpt-4", {}),
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._send_chat_request",
            side_effect=[tool_call_response, final_response],
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._process_tool_calls",
            return_value=[ChatMessage(role="tool", content="{}", tool_call_id="c1", name="f")],
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._get_max_context_tokens",
            return_value=10000,
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_invocation._remove_oldest_tool_call_pair",
            wraps=_remove_oldest_tool_call_pair,
        ) as mock_prune,
    ):
        output = invoke_via_gateway_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="test")],
            trace=mock_trace,
            num_retries=3,
        )

    # Pruning should have been called proactively (not from a 400 error)
    mock_prune.assert_called_once()
    assert json.loads(output.response) == {"result": "yes"}
