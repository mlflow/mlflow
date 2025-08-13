import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities import (
    AssessmentError,
    AssessmentSource,
    Expectation,
    Feedback,
)
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.protos.service_pb2 import TraceInfoV3 as ProtoTraceInfoV3
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY


def test_trace_info():
    assessments = [
        Feedback(
            trace_id="trace_id",
            name="feedback_test",
            value=0.9,
            source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
            create_time_ms=123456789,
            last_update_time_ms=123456789,
            error=AssessmentError(error_code="error_code", error_message="Error message"),
            rationale="Rationale text",
            metadata={"key1": "value1"},
            span_id="span_id",
        ),
        Expectation(
            trace_id="trace_id",
            name="expectation_test",
            value=0.8,
            source=AssessmentSource(source_type="HUMAN", source_id="user_1"),
            create_time_ms=123456789,
            last_update_time_ms=123456789,
            metadata={"key1": "value1"},
            span_id="span_id",
        ),
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
        trace_metadata={"foo": "bar", TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
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
        "trace_metadata": {"foo": "bar", TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        "assessments": [
            {
                "assessment_name": "feedback_test",
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
                "valid": True,
            },
            {
                "assessment_name": "expectation_test",
                "trace_id": "trace_id",
                "span_id": "span_id",
                "source": {"source_type": "HUMAN", "source_id": "user_1"},
                "create_time": "1970-01-02T10:17:36.789Z",
                "last_update_time": "1970-01-02T10:17:36.789Z",
                "expectation": {"value": 0.8},
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
    assert trace_info.request_metadata == {"foo": "bar", TRACE_SCHEMA_VERSION_KEY: "3"}
    assert trace_info.timestamp_ms == 1234567890
    assert trace_info.execution_time_ms is None


@pytest.mark.parametrize("client_request_id", [None, "client_request_id"])
@pytest.mark.parametrize(
    "assessments",
    [
        [],
        # Simple feedback
        [
            Feedback(
                trace_id="trace_id",
                name="relevance",
                value="The answer is correct",
                rationale="Rationale text",
                source=AssessmentSource(source_type="LLM_JUDGE", source_id="gpt"),
                metadata={"key1": "value1"},
                span_id="span_id",
            )
        ],
        # Feedback with error
        [
            Feedback(
                trace_id="trace_id",
                name="relevance",
                error=AssessmentError(error_code="error_code", error_message="Error message"),
            )
        ],
        # Simple expectation
        [Expectation(trace_id="trace_id", name="relevance", value=0.8)],
        # Complex expectation
        [
            Expectation(
                trace_id="trace_id",
                name="relevance",
                value={"complex": {"expectation": ["structure"]}},
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
        trace_metadata={"foo": "bar", TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={"baz": "qux"},
        assessments=assessments,
    )
    proto_trace_info = trace_info.to_proto()
    # proto -> TraceInfo
    assert TraceInfo.from_proto(proto_trace_info) == trace_info

    # TraceInfo -> dict
    dict_trace_info = trace_info.to_dict()
    assert TraceInfo.from_dict(dict_trace_info) == trace_info


def test_from_proto_excludes_undefined_fields():
    """
    Test that undefined fields (client_request_id, execution_duration) are excluded when
    constructing a TraceInfo from a protobuf message instance that does not define these fields.
    """
    # Manually create a protobuf without setting client_request_id or execution_duration fields
    request_time = Timestamp()
    request_time.FromMilliseconds(1234567890)

    proto = ProtoTraceInfoV3(
        trace_id="trace_id",
        # Intentionally NOT setting client_request_id
        # Intentionally NOT setting execution_duration
        trace_location=TraceLocation.from_experiment_id("123").to_proto(),
        request_preview="request",
        response_preview="response",
        request_time=request_time,
        state=TraceState.OK.to_proto(),
    )

    # Verify HasField returns false for undefined fields
    assert not proto.HasField("client_request_id")
    assert not proto.HasField("execution_duration")

    # Convert to TraceInfo
    trace_info = TraceInfo.from_proto(proto)

    # Verify undefined fields are None
    assert trace_info.client_request_id is None
    assert trace_info.execution_duration is None

    # Verify other fields are correctly populated
    assert trace_info.trace_id == "trace_id"
    assert trace_info.experiment_id == "123"
    assert trace_info.request_preview == "request"
    assert trace_info.response_preview == "response"
    assert trace_info.request_time == 1234567890
    assert trace_info.state == TraceState.OK


def test_trace_info_from_proto_updates_schema_version():
    """Test that TraceInfo.from_proto updates schema version when it exists and is outdated."""
    # Create a proto with old schema version in metadata
    request_time = Timestamp()
    request_time.FromMilliseconds(1234567890)

    proto = ProtoTraceInfoV3(
        trace_id="test_trace_id",
        trace_location=TraceLocation.from_experiment_id("123").to_proto(),
        request_preview="test request",
        response_preview="test response",
        request_time=request_time,
        state=TraceState.OK.to_proto(),
        trace_metadata={
            TRACE_SCHEMA_VERSION_KEY: "2",  # Old schema version
            "other_key": "other_value",
        },
        tags={"test_tag": "test_value"},
    )

    # Convert from proto
    trace_info = TraceInfo.from_proto(proto)

    # Verify the schema version was updated to current version
    assert trace_info.trace_metadata[TRACE_SCHEMA_VERSION_KEY] == str(TRACE_SCHEMA_VERSION)

    # Verify other metadata is preserved
    assert trace_info.trace_metadata["other_key"] == "other_value"

    # Verify other fields are correctly populated
    assert trace_info.trace_id == "test_trace_id"
    assert trace_info.experiment_id == "123"


def test_trace_info_from_proto_adds_missing_schema_version():
    """Test that TraceInfo.from_proto adds schema version when it doesn't exist."""
    # Create a proto without schema version in metadata
    request_time = Timestamp()
    request_time.FromMilliseconds(1234567890)

    proto = ProtoTraceInfoV3(
        trace_id="test_trace_id",
        trace_location=TraceLocation.from_experiment_id("123").to_proto(),
        request_preview="test request",
        response_preview="test response",
        request_time=request_time,
        state=TraceState.OK.to_proto(),
        trace_metadata={
            "other_key": "other_value",  # No schema version
        },
        tags={"test_tag": "test_value"},
    )

    # Convert from proto
    trace_info = TraceInfo.from_proto(proto)

    # Verify the schema version was added
    assert trace_info.trace_metadata[TRACE_SCHEMA_VERSION_KEY] == str(TRACE_SCHEMA_VERSION)

    # Verify other metadata is preserved
    assert trace_info.trace_metadata["other_key"] == "other_value"


def test_trace_info_from_proto_preserves_current_schema_version():
    """Test that TraceInfo.from_proto preserves current schema version."""
    # Create a proto with current schema version in metadata
    request_time = Timestamp()
    request_time.FromMilliseconds(1234567890)

    proto = ProtoTraceInfoV3(
        trace_id="test_trace_id",
        trace_location=TraceLocation.from_experiment_id("123").to_proto(),
        request_preview="test request",
        response_preview="test response",
        request_time=request_time,
        state=TraceState.OK.to_proto(),
        trace_metadata={
            TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),  # Current schema version
            "other_key": "other_value",
        },
        tags={"test_tag": "test_value"},
    )

    # Convert from proto
    trace_info = TraceInfo.from_proto(proto)

    # Verify the schema version is preserved as current version
    assert trace_info.trace_metadata[TRACE_SCHEMA_VERSION_KEY] == str(TRACE_SCHEMA_VERSION)

    # Verify other metadata is preserved
    assert trace_info.trace_metadata["other_key"] == "other_value"
