import time
from unittest import mock

import pytest

import mlflow
from mlflow.entities import SpanStatus, TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.tracing.types.wrapper import MlflowSpanWrapper


def test_wrapper_property():
    start_time = time.time_ns()
    end_time = start_time + 1_000_000

    mock_otel_span = mock.MagicMock()
    mock_otel_span.get_span_context().trace_id = "12345"
    mock_otel_span.get_span_context().span_id = "span_id"
    mock_otel_span._start_time = start_time
    mock_otel_span._end_time = end_time
    mock_otel_span.parent.span_id = "parent_span_id"

    span = MlflowSpanWrapper(mock_otel_span)

    assert span.request_id == "tr-12345"
    assert span.span_id == "span_id"
    assert span.start_time == start_time // 1_000
    assert span.end_time == end_time // 1_000
    assert span.parent_span_id == "parent_span_id"


@pytest.mark.parametrize(
    "status",
    [SpanStatus("OK"), SpanStatus(TraceStatus.ERROR, "Error!"), "OK", "ERROR"],
)
def test_set_status(status):
    with mlflow.start_span("test_span") as span:
        span.set_status(status)

    assert isinstance(span.status, SpanStatus)


def test_set_status_raise_for_invalid_value():
    with mlflow.start_span("test_span") as span:
        with pytest.raises(MlflowException, match=r"INVALID is not a valid TraceStatus value."):
            span.set_status("INVALID")
