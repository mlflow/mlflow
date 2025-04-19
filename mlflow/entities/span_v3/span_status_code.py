from enum import Enum

from mlflow.protos.databricks_trace_server_pb2 import Span


class SpanStatusCode(str, Enum):
    STATUS_CODE_UNSET = "STATUS_CODE_UNSET"
    STATUS_CODE_OK = "STATUS_CODE_OK"
    STATUS_CODE_ERROR = "STATUS_CODE_ERROR"

    def to_proto(self):
        return Span.Status.StatusCode.Value(self.value)

    @classmethod
    def from_proto(cls, proto: int) -> "SpanStatusCode":
        return cls(Span.Status.StatusCode.Name(proto))
