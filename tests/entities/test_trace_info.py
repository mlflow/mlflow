import pytest

from mlflow.entities.assessment import (
    Assessment,
    AssessmentError,
    AssessmentSource,
    Expectation,
    Feedback,
)
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState


def test_trace_info():
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
    trace_info = TraceInfo(
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

    from_proto = TraceInfo.from_proto(trace_info.to_proto())
    assert isinstance(from_proto, TraceInfo)
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
    assert TraceInfo.from_dict(trace_info_dict) == trace_info


def test_backwards_compatibility_with_v2():
    trace_info = TraceInfo(
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


@pytest.mark.parametrize("client_request_id", [None, "client_request_id"])
@pytest.mark.parametrize(
    "assessments",
    [
        [],
        # Simple feedback
        [
            Assessment(
                trace_id="trace_id",
                name="relevance",
                source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
                feedback=Feedback("The answer is correct"),
                rationale="Rationale text",
                metadata={"key1": "value1"},
                span_id="span_id",
            )
        ],
        # Feedback with error
        [
            Assessment(
                trace_id="trace_id",
                name="relevance",
                source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
                feedback=Feedback(
                    None,
                    error=AssessmentError(error_code="error_code", error_message="Error message"),
                ),
            )
        ],
        # Simple expectation
        [
            Assessment(
                trace_id="trace_id",
                name="relevance",
                source=AssessmentSource(source_type="LLM_JUDGE", source_id="gpt"),
                expectation=Expectation(0.8),
            )
        ],
        # Complex expectation
        [
            Assessment(
                trace_id="trace_id",
                name="relevance",
                source=AssessmentSource(source_type="LLM_JUDGE", source_id="gpt"),
                expectation=Expectation({"complex": {"expectation": ["structure"]}}),
            )
        ],
    ],
)
def test_trace_info_proto(client_request_id, assessments):
    # TraceInfo -> proto
    trace_info = TraceInfo(
        trace_id="request_id",
        client_request_id=client_request_id,
        trace_location=TraceLocation.from_experiment_id("test_experiment"),
        request_preview="request",
        response_preview="response",
        request_time=0,
        execution_duration=1,
        state=TraceState.OK,
        trace_metadata={"foo": "bar"},
        tags={"baz": "qux"},
        assessments=assessments,
    )
    proto_trace_info = trace_info.to_proto()
    # proto -> TraceInfo
    assert TraceInfo.from_proto(proto_trace_info) == trace_info

    # TraceInfo -> dict
    dict_trace_info = trace_info.to_dict()
    assert TraceInfo.from_dict(dict_trace_info) == trace_info
