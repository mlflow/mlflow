import pytest
from opentelemetry import trace as trace_api

from mlflow.entities import SpanStatus, SpanStatusCode
from mlflow.exceptions import MlflowException


@pytest.mark.parametrize("status_code", [SpanStatusCode.OK, "OK"])
def test_span_status_init(status_code):
    span_status = SpanStatus(status_code)
    assert span_status.status_code == SpanStatusCode.OK


def test_span_status_raise_invalid_status_code():
    with pytest.raises(MlflowException, match=r"INVALID is not a valid SpanStatusCode value."):
        SpanStatus("INVALID", description="test")


@pytest.mark.parametrize(
    ("status_code", "otel_status_code"),
    [
        (SpanStatusCode.OK, trace_api.StatusCode.OK),
        (SpanStatusCode.ERROR, trace_api.StatusCode.ERROR),
        (SpanStatusCode.UNSET, trace_api.StatusCode.UNSET),
    ],
)
def test_otel_status_conversion(status_code, otel_status_code):
    span_status = SpanStatus(status_code, description="test")
    otel_status = span_status.to_otel_status()

    # OpenTelemetry only allows specify description when status is ERROR
    # Otherwise it will be ignored with warning message.
    expected_description = "test" if status_code == SpanStatusCode.ERROR else None

    assert otel_status.status_code == otel_status_code
    assert otel_status.description == expected_description

    span_status = SpanStatus.from_otel_status(otel_status)
    assert span_status.status_code == status_code
    assert span_status.description == (expected_description or "")
