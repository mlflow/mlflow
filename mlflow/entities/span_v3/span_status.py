from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.span_v3.span_status_code import SpanStatusCode
from mlflow.protos.databricks_trace_server_pb2 import Span


@dataclass
class SpanStatus(_MlflowObject):
    message: str
    code: SpanStatusCode

    def to_proto(self):
        return Span.Status(message=self.message, code=self.code.to_proto())

    @classmethod
    def from_proto(cls, proto) -> "SpanStatus":
        return cls(message=proto.message, code=SpanStatusCode.from_proto(proto.code))
