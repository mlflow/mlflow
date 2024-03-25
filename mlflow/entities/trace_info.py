from typing import List, Optional

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.trace_request_metadata import TraceRequestMetadata
from mlflow.entities.trace_status import TraceStatus
from mlflow.entities.trace_tag import TraceTag
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo


class TraceInfo(_MLflowObject):
    """Metadata about a trace.

    Args:
        request_id: id of the trace.
        experiment_id: id of the experiment.
        timestamp_ms: start time of the trace, in millisecond.
        execution_time_ms: duration of the trace, in millisecond.
        status: status of the trace.
        request_metadata: request_metadata associated with the trace.
        tags: tags associated with the trace.
    """

    def __init__(
        self,
        request_id: str,
        experiment_id: str,
        timestamp_ms: int,
        execution_time_ms: int,
        status: TraceStatus,
        request_metadata: Optional[List[TraceRequestMetadata]] = None,
        tags: Optional[List[TraceTag]] = None,
    ):
        if request_id is None:
            raise MlflowException("`request_id` cannot be None.")
        if experiment_id is None:
            raise MlflowException("`experiment_id` cannot be None.")
        if timestamp_ms is None:
            raise MlflowException("`timestamp_ms` cannot be None.")
        if execution_time_ms is None:
            raise MlflowException("`execution_time_ms` cannot be None.")
        if status is None:
            raise MlflowException("`status` cannot be None.")

        self.request_id = request_id
        self.experiment_id = experiment_id
        self.timestamp_ms = timestamp_ms
        self.execution_time_ms = execution_time_ms
        self.status = status
        self.request_metadata = request_metadata
        self.tags = tags

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
        proto.status = TraceStatus.from_string(self.status)
        proto.request_metadata.extend([attr.to_proto() for attr in self.request_metadata])
        proto.tags.extend([tag.to_proto() for tag in self.tags])
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            request_id=proto.request_id,
            experiment_id=proto.experiment_id,
            timestamp_ms=proto.timestamp_ms,
            execution_time_ms=proto.execution_time_ms,
            status=TraceStatus.to_string(proto.status),
            request_metadata=[
                TraceRequestMetadata.from_proto(attr) for attr in proto.request_metadata
            ],
            tags=[TraceTag.from_proto(tag) for tag in proto.tags],
        )
