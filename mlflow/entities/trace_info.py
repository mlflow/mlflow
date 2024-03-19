from typing import List, Optional

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.trace_attribute import TraceAttribute
from mlflow.entities.trace_status import TraceStatus
from mlflow.entities.trace_tag import TraceTag
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo


class TraceInfo(_MLflowObject):
    """Metadata about a trace.

    Args:
        trace_id: id of the trace.
        experiment_id: id of the experiment.
        start_time: start time of the trace.
        end_time: end time of the trace.
        status: status of the trace.
        attributes: attributes associated with the trace.
        tags: tags associated with the trace.
    """

    def __init__(
        self,
        trace_id: str,
        experiment_id: str,
        start_time: int,
        end_time: int,
        status: TraceStatus,
        attributes: Optional[List[TraceAttribute]] = None,
        tags: Optional[List[TraceTag]] = None,
    ):
        if trace_id is None:
            raise MlflowException("`trace_id` cannot be None.")
        if experiment_id is None:
            raise MlflowException("`experiment_id` cannot be None.")
        if start_time is None:
            raise MlflowException("`start_time` cannot be None.")
        if end_time is None:
            raise MlflowException("`end_time` cannot be None.")
        if status is None:
            raise MlflowException("`status` cannot be None.")

        self.trace_id = trace_id
        self.experiment_id = experiment_id
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self.attributes = attributes
        self.tags = tags

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def to_proto(self):
        proto = ProtoTraceInfo()
        proto.trace_id = self.trace_id
        proto.experiment_id = self.experiment_id
        proto.start_time = self.start_time
        proto.end_time = self.end_time
        proto.status = TraceStatus.from_string(self.status)
        proto.attributes.extend([attr.to_proto() for attr in self.attributes])
        proto.tags.extend([tag.to_proto() for tag in self.tags])
        return proto

    @classmethod
    def from_proto(cls, proto):
        return cls(
            trace_id=proto.trace_id,
            experiment_id=proto.experiment_id,
            start_time=proto.start_time,
            end_time=proto.end_time,
            status=TraceStatus.to_string(proto.status),
            attributes=[TraceAttribute.from_proto(attr) for attr in proto.attributes],
            tags=[TraceTag.from_proto(tag) for tag in proto.tags],
        )
