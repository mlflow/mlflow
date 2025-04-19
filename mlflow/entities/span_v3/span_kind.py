from enum import Enum

from mlflow.protos.databricks_trace_server_pb2 import Span


class SpanKind(str, Enum):
    SPAN_KIND_UNSPECIFIED = "SPAN_KIND_UNSPECIFIED"
    SPAN_KIND_INTERNAL = "SPAN_KIND_INTERNAL"
    SPAN_KIND_SERVER = "SPAN_KIND_SERVER"
    SPAN_KIND_CLIENT = "SPAN_KIND_CLIENT"
    SPAN_KIND_PRODUCER = "SPAN_KIND_PRODUCER"
    SPAN_KIND_CONSUMER = "SPAN_KIND_CONSUMER"

    def to_proto(self):
        return Span.SpanKind.Value(self.value)

    @classmethod
    def from_proto(cls, proto: int) -> "SpanKind":
        return cls(Span.SpanKind.Name(proto))
