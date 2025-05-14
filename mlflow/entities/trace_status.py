from enum import Enum

from opentelemetry import trace as trace_api

from mlflow.entities.trace_state import TraceState
from mlflow.protos.service_pb2 import TraceStatus as ProtoTraceStatus
from mlflow.utils.annotations import deprecated


@deprecated(alternative="mlflow.entities.trace_state.TraceState")
class TraceStatus(str, Enum):
    """Enum for status of an :py:class:`mlflow.entities.TraceInfo`."""

    UNSPECIFIED = "TRACE_STATUS_UNSPECIFIED"
    OK = "OK"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"

    def to_state(self) -> TraceState:
        if self == TraceStatus.UNSPECIFIED:
            return TraceState.STATE_UNSPECIFIED
        elif self == TraceStatus.OK:
            return TraceState.OK
        elif self == TraceStatus.ERROR:
            return TraceState.ERROR
        elif self == TraceStatus.IN_PROGRESS:
            return TraceState.IN_PROGRESS
        raise ValueError(f"Unknown TraceStatus: {self}")

    @classmethod
    def from_state(cls, state: TraceState) -> "TraceStatus":
        if state == TraceState.STATE_UNSPECIFIED:
            return cls.UNSPECIFIED
        elif state == TraceState.OK:
            return cls.OK
        elif state == TraceState.ERROR:
            return cls.ERROR
        elif state == TraceState.IN_PROGRESS:
            return cls.IN_PROGRESS
        raise ValueError(f"Unknown TraceState: {state}")

    def to_proto(self):
        return ProtoTraceStatus.Value(self)

    @staticmethod
    def from_proto(proto_status):
        return TraceStatus(ProtoTraceStatus.Name(proto_status))

    @staticmethod
    def from_otel_status(otel_status: trace_api.Status):
        return _OTEL_STATUS_CODE_TO_MLFLOW[otel_status.status_code]

    @classmethod
    def pending_statuses(cls):
        """Traces in pending statuses can be updated to any statuses."""
        return {cls.IN_PROGRESS}

    @classmethod
    def end_statuses(cls):
        """Traces in end statuses cannot be updated to any statuses."""
        return {cls.UNSPECIFIED, cls.OK, cls.ERROR}


_OTEL_STATUS_CODE_TO_MLFLOW = {
    trace_api.StatusCode.OK: TraceStatus.OK,
    trace_api.StatusCode.ERROR: TraceStatus.ERROR,
    trace_api.StatusCode.UNSET: TraceStatus.UNSPECIFIED,
}
