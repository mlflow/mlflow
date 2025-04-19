from dataclasses import dataclass, field
from typing import Any, Optional

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.databricks_trace_server_pb2 import Span


@dataclass
class SpanEvent(_MlflowObject):
    time_unix_nano: int
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    dropped_attributes_count: Optional[int] = None

    def to_proto(self):
        return Span.Event(
            time_unix_nano=self.time_unix_nano,
            name=self.name,
            attributes={k: ParseDict(v, Value()) for k, v in self.attributes.items()},
            dropped_attributes_count=self.dropped_attributes_count,
        )

    @classmethod
    def from_proto(cls, proto) -> "SpanEvent":
        return cls(
            time_unix_nano=proto.time_unix_nano,
            name=proto.name,
            attributes={k: MessageToDict(v) for k, v in proto.attributes.items()},
            dropped_attributes_count=proto.dropped_attributes_count,
        )
