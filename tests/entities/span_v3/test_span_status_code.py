import pytest

from mlflow.entities.span_v3.span_status_code import SpanStatusCode
from mlflow.protos.databricks_trace_server_pb2 import Span


@pytest.mark.parametrize(
    ("status_code", "proto_status_code"), zip(SpanStatusCode, Span.Status.StatusCode.values())
)
def test_span_status_code(status_code, proto_status_code):
    proto_value = status_code.to_proto()
    assert proto_value == proto_status_code
    assert SpanStatusCode.from_proto(proto_value) == status_code
