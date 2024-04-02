from dataclasses import dataclass, field
from typing import Dict, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata


@dataclass
class TraceInfo(_MlflowObject):
    """Metadata about a trace.

    Args:
        request_id: id of the trace.
        experiment_id: id of the experiment.
        timestamp_ms: start time of the trace, in milliseconds.
        execution_time_ms: duration of the trace, in milliseconds.
        status: status of the trace.
        request_metadata: request metadata associated with the trace.
        tags: tags associated with the trace.
    """

    request_id: str
    experiment_id: str
    timestamp_ms: int
    execution_time_ms: Optional[int]
    status: TraceStatus
    request_metadata: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def to_proto(self):
        proto = ProtoTraceInfo()
        proto.request_id = self.request_id
        proto.experiment_id = self.experiment_id
        proto.timestamp_ms = self.timestamp_ms
        proto.execution_time_ms = self.execution_time_ms
        proto.status = self.status.to_proto()

        request_metadata = []
        for key, value in self.request_metadata.items():
            attr = ProtoTraceRequestMetadata()
            attr.key = key
            attr.value = value
            request_metadata.append(attr)
        proto.request_metadata.extend(request_metadata)

        tags = []
        for key, value in self.tags.items():
            tag = ProtoTraceRequestMetadata()
            tag.key = key
            tag.value = value
            tags.append(tag)

        proto.tags.extend(tags)
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            request_id=proto.request_id,
            experiment_id=proto.experiment_id,
            timestamp_ms=proto.timestamp_ms,
            execution_time_ms=proto.execution_time_ms,
            status=TraceStatus.from_proto(proto.status),
            request_metadata={attr.key: attr.value for attr in proto.request_metadata},
            tags={tag.key: tag.value for tag in proto.tags},
        )
