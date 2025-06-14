from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY


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
class TraceInfoV2(_MlflowObject):
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

    @property
    def trace_id(self) -> str:
        """Returns the trace ID of the trace info."""
        return self.request_id

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
        # Client request ID field is only added for internal use, and should not be
        # serialized for V2 TraceInfo.
        trace_info_dict.pop("client_request_id", None)
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

    def to_v3(self, request: Optional[str] = None, response: Optional[str] = None) -> TraceInfo:
        # NB: MLflow automatically converts trace metadata and spans to V3 format, even if the
        # trace was originally created in V2 format with an earlier version of MLflow. Accordingly,
        # we also update the `TRACE_SCHEMA_VERSION_KEY` in the trace metadata to V3 for consistency
        trace_metadata = self.request_metadata.copy()
        trace_metadata[TRACE_SCHEMA_VERSION_KEY] = str(TRACE_SCHEMA_VERSION)

        return TraceInfo(
            trace_id=self.request_id,
            trace_location=TraceLocation.from_experiment_id(self.experiment_id),
            request_preview=request,
            response_preview=response,
            request_time=self.timestamp_ms,
            execution_duration=self.execution_time_ms,
            state=self.status.to_state(),
            trace_metadata=trace_metadata,
            tags=self.tags,
            assessments=self.assessments,
        )
