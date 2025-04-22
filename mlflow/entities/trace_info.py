from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.databricks_trace_server_pb2 import TraceInfo as ProtoTraceInfoV3
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo
from mlflow.protos.service_pb2 import TraceLocation as ProtoTraceLocation
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag


def _truncate_request_metadata(d: dict[str, Any]) -> dict[str, str]:
    from mlflow.tracing.constant import MAX_CHARS_IN_TRACE_INFO_METADATA

    return {
        k[:MAX_CHARS_IN_TRACE_INFO_METADATA]: str(v)[:MAX_CHARS_IN_TRACE_INFO_METADATA]
        for k, v in d.items()
    }


def _truncate_tags(d: dict[str, Any]) -> dict[str, str]:
    from mlflow.tracing.constant import (
        MAX_CHARS_IN_TRACE_INFO_TAGS_KEY,
        MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    )

    return {
        k[:MAX_CHARS_IN_TRACE_INFO_TAGS_KEY]: str(v)[:MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE]
        for k, v in d.items()
    }


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
    request_metadata: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    assessments: list[Assessment] = field(default_factory=list)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def to_proto(self):
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
        for key, value in _truncate_request_metadata(self.request_metadata).items():
            attr = ProtoTraceRequestMetadata()
            attr.key = key
            attr.value = value
            request_metadata.append(attr)
        proto.request_metadata.extend(request_metadata)

        tags = []
        for key, value in _truncate_tags(self.tags).items():
            tag = ProtoTraceTag()
            tag.key = key
            tag.value = str(value)
            tags.append(tag)

        proto.tags.extend(tags)
        return proto

    @classmethod
    def from_proto(cls, proto, assessments=None):
        return cls(
            request_id=proto.request_id,
            experiment_id=proto.experiment_id,
            timestamp_ms=proto.timestamp_ms,
            execution_time_ms=proto.execution_time_ms,
            status=TraceStatus.from_proto(proto.status),
            request_metadata={attr.key: attr.value for attr in proto.request_metadata},
            tags={tag.key: tag.value for tag in proto.tags},
            assessments=assessments or [],
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

    def to_v3_proto(self, request: Optional[str], response: Optional[str]):
        """Convert into the V3 TraceInfo proto object."""
        proto = ProtoTraceInfoV3()

        proto.trace_id = self.request_id
        proto.trace_location.type = ProtoTraceLocation.MLFLOW_EXPERIMENT
        proto.trace_location.mlflow_experiment.experiment_id = self.experiment_id

        proto.request = request or ""
        proto.response = response or ""
        proto.state = ProtoTraceInfoV3.State.Value(self.status.name)

        proto.request_time.FromMilliseconds(self.timestamp_ms)
        if self.execution_time_ms is not None:
            proto.execution_duration.FromMilliseconds(self.execution_time_ms)

        if self.request_metadata:
            proto.trace_metadata.update(_truncate_request_metadata(self.request_metadata))

        if self.tags:
            proto.tags.update(_truncate_tags(self.tags))

        return proto
