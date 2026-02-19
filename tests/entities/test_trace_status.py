from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceStatus as ProtoTraceStatus


def test_trace_status_from_proto():
    assert TraceStatus.from_proto(ProtoTraceStatus.OK) == TraceStatus.OK
    assert isinstance(TraceStatus.from_proto(ProtoTraceStatus.OK), TraceStatus)
    assert (
        TraceStatus.from_proto(ProtoTraceStatus.TRACE_STATUS_UNSPECIFIED) == TraceStatus.UNSPECIFIED
    )
    assert TraceStatus.from_proto(ProtoTraceStatus.ERROR) == TraceStatus.ERROR
    assert TraceStatus.from_proto(ProtoTraceStatus.IN_PROGRESS) == TraceStatus.IN_PROGRESS


def test_trace_status_to_proto():
    assert TraceStatus.OK.to_proto() == ProtoTraceStatus.OK
    assert isinstance(TraceStatus.OK.to_proto(), int)
    assert TraceStatus.UNSPECIFIED.to_proto() == ProtoTraceStatus.TRACE_STATUS_UNSPECIFIED
    assert TraceStatus.ERROR.to_proto() == ProtoTraceStatus.ERROR
    assert TraceStatus.IN_PROGRESS.to_proto() == ProtoTraceStatus.IN_PROGRESS
