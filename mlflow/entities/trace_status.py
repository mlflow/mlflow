from enum import Enum

from mlflow.protos.service_pb2 import TraceStatus as ProtoTraceStatus


class TraceStatus(str, Enum):
    """Enum for status of an :py:class:`mlflow.entities.TraceInfo`."""

    UNSPECIFIED = "TRACE_STATUS_UNSPECIFIED"
    OK = "OK"
    ERROR = "ERROR"

    @staticmethod
    def to_proto(status):
        return ProtoTraceStatus.Value(status)

    def from_proto(proto_status):
        return ProtoTraceStatus.Name(proto_status)
