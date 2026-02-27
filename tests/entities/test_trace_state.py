import pytest

from mlflow.entities.trace_state import TraceState
from mlflow.protos.service_pb2 import TraceInfoV3 as ProtoTraceInfo


@pytest.mark.parametrize(
    ("state", "proto_state"),
    [
        (TraceState.STATE_UNSPECIFIED, ProtoTraceInfo.State.STATE_UNSPECIFIED),
        (TraceState.OK, ProtoTraceInfo.State.OK),
        (TraceState.ERROR, ProtoTraceInfo.State.ERROR),
        (TraceState.IN_PROGRESS, ProtoTraceInfo.State.IN_PROGRESS),
    ],
)
def test_trace_status_from_proto(state, proto_state):
    assert state.to_proto() == proto_state
    assert TraceState.from_proto(proto_state) == state
