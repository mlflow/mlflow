from mlflow.entities.trace_info_v3 import TraceInfoState, TraceInfoV3
from mlflow.entities.trace_location import (
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
)


def test_trace_info_v3():
    trace_info = TraceInfoV3(
        trace_id="trace_id",
        client_request_id="client_request_id",
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="123"),
        ),
        request="'request'",
        response="'response'",
        request_time=1234567890,
        state=TraceInfoState.OK,
        trace_metadata={"foo": "bar"},
        tags={"baz": "qux"},
    )

    from_proto = TraceInfoV3.from_proto(trace_info.to_proto())
    assert isinstance(from_proto, TraceInfoV3)
    assert from_proto == trace_info


def test_backwards_compatibility_with_v2():
    trace_info = TraceInfoV3(
        trace_id="trace_id",
        client_request_id="client_request_id",
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="123"),
        ),
        request="'request'",
        response="'response'",
        request_time=1234567890,
        state=TraceInfoState.OK,
        trace_metadata={"foo": "bar"},
        tags={"baz": "qux"},
    )

    assert trace_info.request_id == trace_info.trace_id
    assert trace_info.experiment_id == "123"
    assert trace_info.request_metadata == {"foo": "bar"}
    assert trace_info.timestamp_ms == 1234567890
    assert trace_info.execution_time_ms is None
