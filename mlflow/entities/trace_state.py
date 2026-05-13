from enum import Enum

from opentelemetry import trace as trace_api

from mlflow.protos import service_pb2 as pb


class TraceState(str, Enum):
    """Enum representing the state of a trace.

    - ``STATE_UNSPECIFIED``: Unspecified trace state.
    - ``OK``: Trace successfully completed.
    - ``ERROR``: Trace encountered an error.
    - ``IN_PROGRESS``: Trace is currently in progress.
    """

    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    OK = "OK"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"

    def __str__(self):
        return self.value

    def to_proto(self):
        return pb.TraceInfoV3.State.Value(self)

    @classmethod
    def from_proto(cls, proto: int) -> "TraceState":
        return TraceState(pb.TraceInfoV3.State.Name(proto))

    @staticmethod
    def from_otel_status(otel_status: trace_api.Status):
        """Convert OpenTelemetry status code to MLflow TraceState."""
        return _OTEL_STATUS_CODE_TO_MLFLOW[otel_status.status_code]


_OTEL_STATUS_CODE_TO_MLFLOW = {
    trace_api.StatusCode.OK: TraceState.OK,
    trace_api.StatusCode.ERROR: TraceState.ERROR,
    trace_api.StatusCode.UNSET: TraceState.STATE_UNSPECIFIED,
}
