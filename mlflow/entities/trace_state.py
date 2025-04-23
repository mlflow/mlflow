from enum import Enum

from opentelemetry import trace as trace_api

from mlflow.protos import service_pb2 as pb


class TraceState(str, Enum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    OK = "OK"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"

    def to_proto(self):
        return pb.TraceInfoV3.State.Value(self)

    @classmethod
    def from_proto(cls, proto: int) -> "TraceState":
        return TraceState(pb.TraceInfoV3.State.Name(proto))

    @staticmethod
    def from_otel_status(otel_status: trace_api.Status):
        return _OTEL_STATUS_CODE_TO_MLFLOW[otel_status.status_code]

    @classmethod
    def pending_states(cls):
        """Traces in pending statuses can be updated to any statuses."""
        return {cls.IN_PROGRESS}

    @classmethod
    def end_states(cls):
        """Traces in end statuses cannot be updated to any statuses."""
        return {cls.UNSPECIFIED, cls.OK, cls.ERROR}


_OTEL_STATUS_CODE_TO_MLFLOW = {
    trace_api.StatusCode.OK: TraceState.OK,
    trace_api.StatusCode.ERROR: TraceState.ERROR,
    trace_api.StatusCode.UNSET: TraceState.STATE_UNSPECIFIED,
}
