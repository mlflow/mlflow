from unittest.mock import patch

import pytest
from pydantic import ValidationError

import mlflow
from mlflow.entities import (
    LiveSpan,
    SpanType,
)
from mlflow.entities.span import SpanType
from mlflow.tracing import set_span_chat_tools
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TokenUsageKey,
)
from mlflow.tracing.utils import (
    _calculate_percentile,
    aggregate_usage_from_spans,
    capture_function_input_args,
    construct_full_inputs,
    deduplicate_span_names_in_place,
    encode_span_id,
    maybe_get_request_id,
)

from tests.tracing.helper import create_mock_otel_span


def test_capture_function_input_args_does_not_raise():
    # Exception during inspecting inputs: trace should be logged without inputs field
    with patch("inspect.signature", side_effect=ValueError("Some error")) as mock_input_args:
        args = capture_function_input_args(lambda: None, (), {})

    assert args is None
    assert mock_input_args.call_count > 0


def test_deduplicate_span_names():
    span_names = ["red", "red", "blue", "red", "green", "blue"]

    spans = [
        LiveSpan(create_mock_otel_span("trace_id", span_id=i, name=span_name), trace_id="tr-123")
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


def test_aggregate_usage_from_spans():
    spans = [
        LiveSpan(create_mock_otel_span("trace_id", span_id=i, name=f"span_{i}"), trace_id="tr-123")
        for i in range(3)
    ]
    spans[0].set_attribute(
        SpanAttributeKey.CHAT_USAGE,
        {
            TokenUsageKey.INPUT_TOKENS: 10,
            TokenUsageKey.OUTPUT_TOKENS: 20,
            TokenUsageKey.TOTAL_TOKENS: 30,
        },
    )
    spans[1].set_attribute(
        SpanAttributeKey.CHAT_USAGE,
        {TokenUsageKey.OUTPUT_TOKENS: 15, TokenUsageKey.TOTAL_TOKENS: 15},
    )
    spans[2].set_attribute(
        SpanAttributeKey.CHAT_USAGE,
        {
            TokenUsageKey.INPUT_TOKENS: 5,
            TokenUsageKey.OUTPUT_TOKENS: 10,
            TokenUsageKey.TOTAL_TOKENS: 15,
        },
    )

    usage = aggregate_usage_from_spans(spans)
    assert usage == {
        TokenUsageKey.INPUT_TOKENS: 15,
        TokenUsageKey.OUTPUT_TOKENS: 45,
        TokenUsageKey.TOTAL_TOKENS: 60,
    }


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


def test_calculate_percentile():
    # Test empty list
    assert _calculate_percentile([], 0.5) == 0.0

    # Test single element
    assert _calculate_percentile([100], 0.25) == 100
    assert _calculate_percentile([100], 0.5) == 100
    assert _calculate_percentile([100], 0.75) == 100

    # Test two elements
    assert _calculate_percentile([10, 20], 0.0) == 10
    assert _calculate_percentile([10, 20], 0.5) == 15  # Linear interpolation
    assert _calculate_percentile([10, 20], 1.0) == 20

    # Test odd number of elements
    data = [10, 20, 30, 40, 50]
    assert _calculate_percentile(data, 0.0) == 10
    assert _calculate_percentile(data, 0.25) == 20
    assert _calculate_percentile(data, 0.5) == 30  # Median
    assert _calculate_percentile(data, 0.75) == 40
    assert _calculate_percentile(data, 1.0) == 50

    # Test even number of elements
    data = [10, 20, 30, 40]
    assert _calculate_percentile(data, 0.0) == 10
    assert _calculate_percentile(data, 0.25) == 17.5  # Between 10 and 20
    assert _calculate_percentile(data, 0.5) == 25  # Between 20 and 30
    assert _calculate_percentile(data, 0.75) == 32.5  # Between 30 and 40
    assert _calculate_percentile(data, 1.0) == 40

    # Test with larger dataset
    data = list(range(1, 101))  # 1 to 100
    assert _calculate_percentile(data, 0.25) == 25.75
    assert _calculate_percentile(data, 0.5) == 50.5
    assert _calculate_percentile(data, 0.75) == 75.25
