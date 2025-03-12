import pytest
from pydantic import ValidationError

import mlflow
from mlflow.entities import LiveSpan
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowTracingException
from mlflow.tracing import set_span_chat_messages, set_span_chat_tools
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import (
    construct_full_inputs,
    deduplicate_span_names_in_place,
    encode_span_id,
    maybe_get_request_id,
)

from tests.tracing.helper import create_mock_otel_span


def test_deduplicate_span_names():
    span_names = ["red", "red", "blue", "red", "green", "blue"]

    spans = [
        LiveSpan(create_mock_otel_span("trace_id", span_id=i, name=span_name), request_id="tr-123")
        for i, span_name in enumerate(span_names)
    ]
    deduplicate_span_names_in_place(spans)

    assert [span.name for span in spans] == [
        "red_1",
        "red_2",
        "blue_1",
        "red_3",
        "green",
        "blue_2",
    ]
    # Check if the span order is preserved
    assert [span.span_id for span in spans] == [encode_span_id(i) for i in [0, 1, 2, 3, 4, 5]]


def test_maybe_get_request_id():
    assert maybe_get_request_id(is_evaluate=True) is None

    try:
        from mlflow.pyfunc.context import Context, set_prediction_context
    except ImportError:
        pytest.skip("Skipping the rest of tests as mlflow.pyfunc module is not available.")

    with set_prediction_context(Context(request_id="eval", is_evaluate=True)):
        assert maybe_get_request_id(is_evaluate=True) == "eval"

    with set_prediction_context(Context(request_id="non_eval", is_evaluate=False)):
        assert maybe_get_request_id(is_evaluate=True) is None

    with pytest.raises(MlflowTracingException, match="Missing request_id for context"):
        with set_prediction_context(Context(request_id=None, is_evaluate=True)):
            maybe_get_request_id(is_evaluate=True)


def test_set_span_chat_messages_and_tools():
    messages = [
        {
            "role": "system",
            "content": "please use the provided tool to answer the user's questions",
        },
        {"role": "user", "content": "what is 1 + 1?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "123",
                    "function": {"arguments": '{"a": 1,"b": 2}', "name": "add"},
                    "type": "function",
                }
            ],
        },
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]

    @mlflow.trace(span_type=SpanType.CHAT_MODEL)
    def dummy_call(messages, tools):
        span = mlflow.get_current_active_span()
        set_span_chat_messages(span, messages)
        set_span_chat_tools(span, tools)
        return None

    dummy_call(messages, tools)

    trace = mlflow.get_last_active_trace()
    span = trace.data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == messages
    assert span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == tools


def test_set_span_chat_messages_append():
    messages = [
        {"role": "system", "content": "you are a confident bot"},
        {"role": "user", "content": "what is 1 + 1?"},
    ]
    additional_messages = [{"role": "assistant", "content": "it is definitely 5"}]

    # Append messages
    with mlflow.start_span(name="foo") as span:
        set_span_chat_messages(span, messages)
        set_span_chat_messages(span, additional_messages, append=True)

    trace = mlflow.get_last_active_trace()
    span = trace.data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == messages + additional_messages

    # Overwrite messages
    with mlflow.start_span(name="bar") as span:
        set_span_chat_messages(span, messages)
        set_span_chat_messages(span, additional_messages, append=False)

    trace = mlflow.get_last_active_trace()
    span = trace.data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == additional_messages


def test_set_chat_messages_validation():
    messages = [{"invalid_field": "user", "content": "hello"}]

    @mlflow.trace(span_type=SpanType.CHAT_MODEL)
    def dummy_call(messages):
        span = mlflow.get_current_active_span()
        set_span_chat_messages(span, messages)
        return None

    with pytest.raises(ValidationError, match="validation error for ChatMessage"):
        dummy_call(messages)


def test_set_chat_tools_validation():
    tools = [
        {
            "type": "unsupported_function",
            "unsupported_function": {
                "name": "test",
            },
        }
    ]

    @mlflow.trace(span_type=SpanType.CHAT_MODEL)
    def dummy_call(tools):
        span = mlflow.get_current_active_span()
        set_span_chat_tools(span, tools)
        return None

    with pytest.raises(ValidationError, match="validation error for ChatTool"):
        dummy_call(tools)


def test_construct_full_inputs_simple_function():
    def func(a, b, c=3, d=4, **kwargs):
        pass

    result = construct_full_inputs(func, 1, 2)
    assert result == {"a": 1, "b": 2}

    result = construct_full_inputs(func, 1, 2, c=30)
    assert result == {"a": 1, "b": 2, "c": 30}

    result = construct_full_inputs(func, 1, 2, c=30, d=40, e=50)
    assert result == {"a": 1, "b": 2, "c": 30, "d": 40, "kwargs": {"e": 50}}

    def no_args_func():
        pass

    result = construct_full_inputs(no_args_func)
    assert result == {}

    class TestClass:
        def func(self, a, b, c=3, d=4, **kwargs):
            pass

    result = construct_full_inputs(TestClass().func, 1, 2)
    assert result == {"a": 1, "b": 2}
