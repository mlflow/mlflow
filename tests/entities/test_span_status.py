import pytest
from opentelemetry import trace as trace_api
from opentelemetry.proto.trace.v1.trace_pb2 import Status as OTelProtoStatus
from opentelemetry.trace import Status as OTelStatus
from opentelemetry.trace import StatusCode as OTelStatusCode

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


@pytest.mark.parametrize(
    ("status_code", "status_desc", "expected_code", "expected_desc"),
    [
        # OTel only keeps description for ERROR status
        (OTelStatusCode.OK, "Success", SpanStatusCode.OK, ""),
        (OTelStatusCode.ERROR, "Failed", SpanStatusCode.ERROR, "Failed"),
        (OTelStatusCode.UNSET, "", SpanStatusCode.UNSET, ""),
    ],
)
def test_otel_status_to_mlflow_status_conversion(
    status_code, status_desc, expected_code, expected_desc
):
    """Test conversion from OTel SDK status to MLflow SpanStatus."""
    # Create OTel status
    otel_status = OTelStatus(status_code, status_desc)

    # Convert to MLflow status
    mlflow_status = SpanStatus.from_otel_status(otel_status)
    assert mlflow_status.status_code == expected_code
    # OTel SDK clears description for non-ERROR statuses
    assert mlflow_status.description == (status_desc if status_code == OTelStatusCode.ERROR else "")

    # Convert back to OTel status
    converted_status = mlflow_status.to_otel_status()
    assert converted_status.status_code == status_code
    # Description is only preserved for ERROR status
    assert converted_status.description == expected_desc


@pytest.mark.parametrize(
    ("proto_code", "expected_mlflow_code"),
    [
        (OTelProtoStatus.STATUS_CODE_OK, SpanStatusCode.OK),
        (OTelProtoStatus.STATUS_CODE_ERROR, SpanStatusCode.ERROR),
        (OTelProtoStatus.STATUS_CODE_UNSET, SpanStatusCode.UNSET),
    ],
)
def test_otel_proto_status_to_mlflow_status_conversion(proto_code, expected_mlflow_code):
    """Test conversion from OTel protobuf status to MLflow SpanStatus."""
    # Create proto status
    proto_status = OTelProtoStatus()
    proto_status.code = proto_code
    proto_status.message = "test message"

    # Convert to MLflow status
    mlflow_status = SpanStatus.from_otel_proto_status(proto_status)
    assert mlflow_status.status_code == expected_mlflow_code
    assert mlflow_status.description == "test message"


@pytest.mark.parametrize(
    ("mlflow_code", "description", "expected_proto_code"),
    [
        (SpanStatusCode.OK, "Success", OTelProtoStatus.STATUS_CODE_OK),
        (SpanStatusCode.ERROR, "Failed", OTelProtoStatus.STATUS_CODE_ERROR),
        (SpanStatusCode.UNSET, "", OTelProtoStatus.STATUS_CODE_UNSET),
    ],
)
def test_mlflow_status_to_otel_proto_status_conversion(
    mlflow_code, description, expected_proto_code
):
    """Test conversion from MLflow SpanStatus to OTel protobuf status."""
    # Create MLflow status
    mlflow_status = SpanStatus(mlflow_code, description)

    # Convert to OTel proto status
    proto_status = mlflow_status.to_otel_proto_status()
    assert proto_status.code == expected_proto_code
    assert proto_status.message == description
