from mlflow.entities.span_status import SpanStatus
from mlflow.protos.service_pb2 import TraceStatus as ProtoTraceStatus


class TraceStatus:
    """Enum for status of an :py:class:`mlflow.entities.TraceInfo`."""

    UNSPECIFIED = ProtoTraceStatus.Value("TRACE_STATUS_UNSPECIFIED")
    OK = ProtoTraceStatus.Value("OK")
    ERROR = ProtoTraceStatus.Value("ERROR")

    _STRING_TO_STATUS = {k: ProtoTraceStatus.Value(k) for k in ProtoTraceStatus.keys()}
    _STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}

    @staticmethod
    def from_string(status_str):
        if status_str not in TraceStatus._STRING_TO_STATUS:
            raise Exception(
                f"Could not get trace status corresponding to string {status_str}. Valid trace "
                f"status strings: {list(TraceStatus._STRING_TO_STATUS.keys())}"
            )
        return TraceStatus._STRING_TO_STATUS[status_str]

    @staticmethod
    def from_span_status(status: SpanStatus):
        if status.status_code == SpanStatus.StatusCode.OK:
            return TraceStatus.OK
        elif status.status_code == SpanStatus.StatusCode.UNSET:
            return TraceStatus.UNSPECIFIED
        else:
            return TraceStatus.ERROR

    @staticmethod
    def to_string(status):
        if status not in TraceStatus._STATUS_TO_STRING:
            raise Exception(
                f"Could not get string corresponding to trace status {status}. Valid trace "
                f"statuses: {list(TraceStatus._STATUS_TO_STRING.keys())}"
            )
        return TraceStatus._STATUS_TO_STRING[status]

    @staticmethod
    def all_status():
        return list(TraceStatus._STATUS_TO_STRING.keys())
