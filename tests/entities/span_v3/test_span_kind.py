import pytest

from mlflow.entities.span_v3.span_kind import SpanKind
from mlflow.protos.databricks_trace_server_pb2 import Span


@pytest.mark.parametrize(
    ("span_kind", "proto_span_kind"),
    [
        (SpanKind.SPAN_KIND_UNSPECIFIED, Span.SpanKind.SPAN_KIND_UNSPECIFIED),
        (SpanKind.SPAN_KIND_INTERNAL, Span.SpanKind.SPAN_KIND_INTERNAL),
        (SpanKind.SPAN_KIND_SERVER, Span.SpanKind.SPAN_KIND_SERVER),
        (SpanKind.SPAN_KIND_CLIENT, Span.SpanKind.SPAN_KIND_CLIENT),
        (SpanKind.SPAN_KIND_PRODUCER, Span.SpanKind.SPAN_KIND_PRODUCER),
        (SpanKind.SPAN_KIND_CONSUMER, Span.SpanKind.SPAN_KIND_CONSUMER),
    ],
)
def test_span_kind(span_kind, proto_span_kind):
    proto_value = span_kind.to_proto()
    assert proto_value == proto_span_kind
    assert SpanKind.from_proto(proto_value) == span_kind
