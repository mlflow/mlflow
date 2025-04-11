from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_location import TraceLocation
from mlflow.protos import databricks_trace_server_pb2 as pb


@dataclass
class TraceInfoState(str, Enum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    OK = "OK"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"

    def to_proto(self) -> pb.TraceInfo.State:
        return pb.TraceInfo.State.Value(self)

    @classmethod
    def from_proto(cls, proto: int) -> "TraceInfoState":
        return TraceInfoState(pb.TraceInfo.State.Name(proto))


@dataclass
class TraceInfoV3(_MlflowObject):
    trace_id: str
    client_request_id: str
    trace_location: TraceLocation
    request: str
    response: str
    request_time: int
    state: TraceInfoState
    execution_duration: Optional[int] = None
    trace_metadata: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    assessments: list[Assessment] = field(default_factory=list)

    def to_proto(self) -> pb.TraceInfo:
        request_time = Timestamp()
        request_time.FromMilliseconds(self.request_time)
        execution_duration = None
        if self.execution_duration is not None:
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
            assessments=[a.to_proto() for a in self.assessments],
        )

    @classmethod
    def from_proto(cls, proto: pb.TraceInfo) -> "TraceInfoV3":
        request_time = proto.request_time.ToMilliseconds()
        execution_duration = (
            proto.execution_duration.ToMilliseconds()
            if proto.HasField("execution_duration")
            else None
        )
        return cls(
            trace_id=proto.trace_id,
            client_request_id=proto.client_request_id,
            trace_location=TraceLocation.from_proto(proto.trace_location),
            request=proto.request,
            response=proto.response,
            request_time=request_time,
            execution_duration=execution_duration,
            state=TraceInfoState.from_proto(proto.state),
            # ScalarMapContainer -> native dict
            trace_metadata=dict(proto.trace_metadata),
            tags=dict(proto.tags),
            assessments=[Assessment.from_proto(a) for a in proto.assessments],
        )

    # Aliases for backward compatibility with V2 format
    @property
    def request_id(self) -> str:
        return self.trace_id

    @property
    def experiment_id(self) -> Optional[str]:
        return (
            self.trace_location.mlflow_experiment
            and self.trace_location.mlflow_experiment.experiment_id
        )

    @property
    def request_metadata(self) -> dict[str, str]:
        return self.trace_metadata

    @property
    def timestamp_ms(self) -> int:
        return self.request_time

    @property
    def execution_time_ms(self) -> Optional[int]:
        return self.execution_duration

    @property
    def status(self) -> TraceInfoState:
        return self.state
