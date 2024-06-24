from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag


@dataclass
class TraceInfo(_MlflowObject):
    """Metadata about a trace.

    Args:
        request_id: id of the trace.
        experiment_id: id of the experiment.
        timestamp_ms: start time of the trace, in milliseconds.
        execution_time_ms: duration of the trace, in milliseconds.
        status: status of the trace.
        request_metadata: Key-value pairs associated with the trace. Request metadata are designed
            for immutable values like run ID associated with the trace.
        tags: Tags associated with the trace. Tags are designed for mutable values like trace name,
            that can be updated by the users after the trace is created, unlike request_metadata.
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
        from mlflow.tracing.constant import MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS

        proto = ProtoTraceInfo()
        proto.request_id = self.request_id
        proto.experiment_id = self.experiment_id
        proto.timestamp_ms = self.timestamp_ms
        # NB: Proto setter does not support nullable fields (even with 'optional' keyword),
        # so we substitute None with 0 for execution_time_ms. This should be not too confusing
        # as we only put None when starting a trace i.e. the execution time is actually 0.
        proto.execution_time_ms = self.execution_time_ms or 0
        proto.status = self.status.to_proto()

        request_metadata = []
        for key, value in self.request_metadata.items():
            attr = ProtoTraceRequestMetadata()
            attr.key = key[:MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS]
            attr.value = str(value)[:MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS]
            request_metadata.append(attr)
        proto.request_metadata.extend(request_metadata)

        tags = []
        for key, value in self.tags.items():
            tag = ProtoTraceTag()
            tag.key = key[:MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS]
            tag.value = str(value)[:MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS]
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

    def to_dict(self):
        """
        Convert trace info to a dictionary for persistence.
        Update status field to the string value for serialization.
        """
        trace_info_dict = asdict(self)
        trace_info_dict["status"] = self.status.value
        return trace_info_dict

    @classmethod
    def from_dict(cls, trace_info_dict):
        """
        Convert trace info dictionary to TraceInfo object.
        """
        if "status" not in trace_info_dict:
            raise ValueError("status is required in trace info dictionary.")
        trace_info_dict["status"] = TraceStatus(trace_info_dict["status"])
        return cls(**trace_info_dict)
