import pytest

from mlflow.entities.span_v3.span_kind import SpanKind
from mlflow.protos.databricks_trace_server_pb2 import Span


@pytest.mark.parametrize(("span_kind", "proto_span_kind"), zip(SpanKind, Span.SpanKind.values()))
def test_span_kind(span_kind, proto_span_kind):
    proto_value = span_kind.to_proto()
    assert proto_value == proto_span_kind
    assert SpanKind.from_proto(proto_value) == span_kind
