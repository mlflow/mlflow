from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_location import (
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
)
from mlflow.protos import databricks_trace_server_pb2 as pb
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfoV2


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
    trace_location: TraceLocation
    request_time: int
    state: TraceInfoState
    request: Optional[str] = None
    response: Optional[str] = None
    client_request_id: Optional[str] = None
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

    @classmethod
    def from_v2_proto(cls, proto: ProtoTraceInfoV2) -> "TraceInfoV3":
        trace_location = TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=proto.experiment_id),
        )
        return cls(
            trace_id=proto.request_id,
            trace_location=trace_location,
            request_time=proto.timestamp_ms,
            execution_duration=proto.execution_time_ms,
            state=TraceInfoState.from_proto(proto.status),
            trace_metadata={m.key: m.value for m in proto.request_metadata},
            tags={t.key: t.value for t in proto.tags},
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

    @timestamp_ms.setter
    def timestamp_ms(self, value: int) -> None:
        self.request_time = value

    @property
    def execution_time_ms(self) -> Optional[int]:
        return self.execution_duration

    @execution_time_ms.setter
    def execution_time_ms(self, value: Optional[int]) -> None:
        self.execution_duration = value

    @property
    def status(self) -> TraceInfoState:
        return self.state

    @status.setter
    def status(self, value: TraceInfoState) -> None:
        self.state = value
