import pytest
from opentelemetry import trace as trace_api

from mlflow.entities import SpanStatus, TraceStatus
from mlflow.exceptions import MlflowException


@pytest.mark.parametrize("status_code", [TraceStatus.OK, "OK"])
def test_span_status_init(status_code):
    span_status = SpanStatus(status_code)
    assert span_status.status_code == TraceStatus.OK


def test_span_status_raise_invalid_status_code():
    with pytest.raises(MlflowException, match=r"INVALID is not a valid TraceStatus value."):
        SpanStatus("INVALID", description="test")


@pytest.mark.parametrize(
    ("status_code", "otel_status_code"),
    [
        (TraceStatus.OK, trace_api.StatusCode.OK),
        (TraceStatus.ERROR, trace_api.StatusCode.ERROR),
        (TraceStatus.UNSPECIFIED, trace_api.StatusCode.UNSET),
    ],
)
def test_otel_status_conversion(status_code, otel_status_code):
    span_status = SpanStatus(status_code, description="test")
    otel_status = span_status.to_otel_status()

    # OpenTelemetry only allows specify description when status is ERROR
    # Otherwise it will be ignored with warning message.
    expected_description = "test" if status_code == TraceStatus.ERROR else None

    assert otel_status.status_code == otel_status_code
    assert otel_status.description == expected_description

    span_status = SpanStatus.from_otel_status(otel_status)
    assert span_status.status_code == status_code
    assert span_status.description == expected_description
