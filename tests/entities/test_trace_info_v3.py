from mlflow.entities.assessment import (
    Assessment,
    AssessmentError,
    AssessmentSource,
    Expectation,
    Feedback,
)
from mlflow.entities.trace_info_v3 import TraceInfoV3
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState


def test_trace_info_v3():
    assessments = [
        Assessment(
            trace_id="trace_id",
            name="relevance",
            source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
            create_time_ms=123456789,
            last_update_time_ms=123456789,
            expectation=expectation,
            feedback=feedback,
            rationale="Rationale text",
            metadata={"key1": "value1"},
            span_id="span_id",
        )
        for expectation, feedback in [
            (
                None,
                Feedback(
                    0.9,
                    error=AssessmentError(error_code="error_code", error_message="Error message"),
                ),
            ),
            (
                Expectation(0.8),
                None,
            ),
        ]
    ]
    trace_info = TraceInfoV3(
        trace_id="trace_id",
        client_request_id="client_request_id",
        trace_location=TraceLocation.from_experiment_id("123"),
        request_preview="request",
        response_preview="response",
        request_time=1234567890,
        execution_duration=100,
        state=TraceState.OK,
        trace_metadata={"foo": "bar"},
        tags={"baz": "qux"},
        assessments=assessments,
    )

    from_proto = TraceInfoV3.from_proto(trace_info.to_proto())
    assert isinstance(from_proto, TraceInfoV3)
    assert from_proto == trace_info

    trace_info_dict = trace_info.to_dict()
    assert trace_info_dict == {
        "trace_id": "trace_id",
        "client_request_id": "client_request_id",
        "trace_location": {
            "type": "MLFLOW_EXPERIMENT",
            "mlflow_experiment": {"experiment_id": "123"},
        },
        "request_preview": "request",
        "response_preview": "response",
        "request_time": "1970-01-15T06:56:07.890Z",
        "execution_duration_ms": 100,
        "state": "OK",
        "trace_metadata": {"foo": "bar"},
        "assessments": [
            {
                "assessment_name": "relevance",
                "trace_id": "trace_id",
                "span_id": "span_id",
                "source": {"source_type": "HUMAN", "source_id": "user_1"},
                "create_time": "1970-01-02T10:17:36.789Z",
                "last_update_time": "1970-01-02T10:17:36.789Z",
                "feedback": {
                    "value": 0.9,
                    "error": {"error_code": "error_code", "error_message": "Error message"},
                },
                "rationale": "Rationale text",
                "metadata": {"key1": "value1"},
            },
            {
                "assessment_name": "relevance",
                "trace_id": "trace_id",
                "span_id": "span_id",
                "source": {"source_type": "HUMAN", "source_id": "user_1"},
                "create_time": "1970-01-02T10:17:36.789Z",
                "last_update_time": "1970-01-02T10:17:36.789Z",
                "expectation": {"value": 0.8},
                "rationale": "Rationale text",
                "metadata": {"key1": "value1"},
            },
        ],
        "tags": {"baz": "qux"},
    }
    assert TraceInfoV3.from_dict(trace_info_dict) == trace_info


def test_backwards_compatibility_with_v2():
    trace_info = TraceInfoV3(
        trace_id="trace_id",
        client_request_id="client_request_id",
        trace_location=TraceLocation.from_experiment_id("123"),
        request_preview="'request'",
        response_preview="'response'",
        request_time=1234567890,
        state=TraceState.OK,
        trace_metadata={"foo": "bar"},
        tags={"baz": "qux"},
    )

    assert trace_info.request_id == trace_info.trace_id
    assert trace_info.experiment_id == "123"
    assert trace_info.request_metadata == {"foo": "bar"}
    assert trace_info.timestamp_ms == 1234567890
    assert trace_info.execution_time_ms is None
