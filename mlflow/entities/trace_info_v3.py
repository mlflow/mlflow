from dataclasses import dataclass, field
from enum import Enum

from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_location import TraceLocation
from mlflow.protos import databricks_trace_server_pb2 as pb


@dataclass
class TraceInfoStatus(Enum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    OK = "OK"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"

    def to_proto(self) -> pb.TraceInfo.State:
        if self == TraceInfoStatus.STATE_UNSPECIFIED:
            return pb.TraceInfo.State.STATE_UNSPECIFIED
        elif self == TraceInfoStatus.OK:
            return pb.TraceInfo.State.OK
        elif self == TraceInfoStatus.ERROR:
            return pb.TraceInfo.State.ERROR
        elif self == TraceInfoStatus.IN_PROGRESS:
            return pb.TraceInfo.State.IN_PROGRESS
        raise ValueError(f"Unknown TraceInfoStatus: {self}")

    @classmethod
    def from_proto(cls, proto: pb.TraceInfo.State) -> "TraceInfoStatus":
        if proto == pb.TraceInfo.State.STATE_UNSPECIFIED:
            return cls.STATE_UNSPECIFIED
        elif proto == pb.TraceInfo.State.OK:
            return cls.OK
        elif proto == pb.TraceInfo.State.ERROR:
            return cls.ERROR
        elif proto == pb.TraceInfo.State.IN_PROGRESS:
            return cls.IN_PROGRESS
        raise ValueError(f"Unknown TraceInfoStatus: {proto}")


@dataclass
class TraceInfoV3(_MlflowObject):
    trace_id: str
    client_request_id: str
    trace_location: TraceLocation
    request: str
    response: str
    request_time: int
    execution_duration: int
    state: TraceInfoStatus
    trace_metadata: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    assessments: list[Assessment] = field(default_factory=list)

    def to_proto(self):
        request_time = Timestamp()
        request_time.FromMilliseconds(self.request_time)
        execution_duration = Duration()
        execution_duration.FromMilliseconds(self.execution_duration)
        return pb.TraceInfo(
            trace_id=self.trace_id,
            client_request_id=self.client_request_id,
            trace_location=self.trace_location.to_proto(),
            request=self.request,
            response=self.response,
            request_time=request_time,
            execution_duration=execution_duration,
            state=self.state.to_proto(),
            trace_metadata=self.trace_metadata,
            tags=self.tags,
            assessments=[assessment.to_proto() for assessment in self.assessments],
        )

    @classmethod
    def from_proto(cls, proto: pb.TraceInfo) -> "TraceInfoV3":
        request_time = proto.request_time.ToMilliseconds()
        execution_duration = proto.execution_duration.ToMilliseconds()
        assessments = [Assessment.from_proto(assessment) for assessment in proto.assessments]
        return cls(
            trace_id=proto.trace_id,
            client_request_id=proto.client_request_id,
            trace_location=TraceLocation.from_proto(proto.trace_location),
            request=proto.request,
            response=proto.response,
            request_time=request_time,
            execution_duration=execution_duration,
            state=TraceInfoStatus.from_proto(proto.state),
            trace_metadata=proto.trace_metadata,
            tags=proto.tags,
            assessments=assessments,
        )
