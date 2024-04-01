import time
from unittest import mock

from mlflow.tracing.types.wrapper import MlflowSpanWrapper


def test_wrapper_property():
    start_time = time.time_ns()
    end_time = start_time + 1_000_000

    mock_otel_span = mock.MagicMock()
    mock_otel_span.get_span_context().trace_id = "trace_id"
    mock_otel_span.get_span_context().span_id = "span_id"
    mock_otel_span._start_time = start_time
    mock_otel_span._end_time = end_time
    mock_otel_span.parent.span_id = "parent_span_id"

    span = MlflowSpanWrapper(mock_otel_span)

    assert span.request_id == "trace_id"
    assert span.span_id == "span_id"
    assert span.start_time == start_time // 1_000
    assert span.end_time == end_time // 1_000
    assert span.parent_span_id == "parent_span_id"
