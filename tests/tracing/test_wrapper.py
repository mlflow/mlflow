import time

import pytest

import mlflow
from mlflow.entities import SpanStatus, SpanType, TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.tracing.types.wrapper import MlflowSpanWrapper

from tests.tracing.helper import create_mock_otel_span


def test_wrapper_property():
    start_time = time.time_ns()
    end_time = start_time + 1_000_000
    request_id = "tr-12345"
    trace_id = "12345"
    span_id = "test_span_id"
    parent_id = "parent_id"

    mock_otel_span = create_mock_otel_span(
        trace_id, span_id, parent_id=parent_id, start_time=start_time, end_time=end_time
    )
    span = MlflowSpanWrapper(mock_otel_span, request_id=request_id, span_type=SpanType.LLM)

    assert span.request_id == request_id
    assert span.span_id == span_id
    assert span.start_time == start_time
    assert span.end_time == end_time
    assert span.parent_id == parent_id

    span.set_inputs({"input": 1})
    span.set_outputs(2)
    span.set_attribute("key", 3)

    assert span.inputs == {"input": 1}
    assert span.outputs == 2
    assert mock_otel_span._attributes == {
        "mlflow.traceRequestId": request_id,
        "mlflow.spanInputs": '{"input": 1}',
        "mlflow.spanOutputs": "2",
        "mlflow.spanType": "LLM",
        "key": "3",
    }


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
