import pytest
from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp

from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    MAX_CHARS_IN_TRACE_INFO_TAGS_KEY,
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
)


@pytest.fixture
def trace_info():
    return TraceInfoV2(
        request_id="request_id",
        experiment_id="test_experiment",
        timestamp_ms=0,
        execution_time_ms=1,
        status=TraceStatus.OK,
        request_metadata={
            "foo": "bar",
            "k" * 1000: "v" * 1000,
        },
        tags={
            "baz": "qux",
            "k" * 2000: "v" * 8000,
        },
        assessments=[],
    )


@pytest.fixture
def trace_info_proto():
    ti_proto = ProtoTraceInfo()
    ti_proto.request_id = "request_id"
    ti_proto.experiment_id = "test_experiment"
    ti_proto.timestamp_ms = 0
    ti_proto.execution_time_ms = 1
    ti_proto.status = TraceStatus.OK.to_proto()
    request_metadata_1 = ti_proto.request_metadata.add()
    request_metadata_1.key = "foo"
    request_metadata_1.value = "bar"
    request_metadata_2 = ti_proto.request_metadata.add()
    request_metadata_2.key = "k" * 250
    request_metadata_2.value = "v" * 250
    request_metadata_3 = ti_proto.request_metadata.add()
    request_metadata_3.key = TRACE_SCHEMA_VERSION_KEY
    request_metadata_3.value = "2"
    tag_1 = ti_proto.tags.add()
    tag_1.key = "baz"
    tag_1.value = "qux"
    tag_2 = ti_proto.tags.add()
    tag_2.key = "k" * 250
    tag_2.value = "v" * 250
    return ti_proto


def test_to_proto(trace_info):
    proto = trace_info.to_proto()
    assert proto.request_id == "request_id"
    assert proto.experiment_id == "test_experiment"
    assert proto.timestamp_ms == 0
    assert proto.execution_time_ms == 1
    assert proto.status == 1
    request_metadata_1 = proto.request_metadata[0]
    assert isinstance(request_metadata_1, ProtoTraceRequestMetadata)
    assert request_metadata_1.key == "foo"
    assert request_metadata_1.value == "bar"
    request_metadata_2 = proto.request_metadata[1]
    assert isinstance(request_metadata_2, ProtoTraceRequestMetadata)
    assert request_metadata_2.key == "k" * MAX_CHARS_IN_TRACE_INFO_METADATA
    assert request_metadata_2.value == "v" * MAX_CHARS_IN_TRACE_INFO_METADATA
    tag_1 = proto.tags[0]
    assert isinstance(tag_1, ProtoTraceTag)
    assert tag_1.key == "baz"
    assert tag_1.value == "qux"
    tag_2 = proto.tags[1]
    assert isinstance(tag_2, ProtoTraceTag)
    assert tag_2.key == "k" * MAX_CHARS_IN_TRACE_INFO_TAGS_KEY
    assert tag_2.value == "v" * MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE


def test_to_dict(trace_info):
    trace_as_dict = trace_info.to_dict()
    assert trace_as_dict == {
        "request_id": "request_id",
        "experiment_id": "test_experiment",
        "timestamp_ms": 0,
        "execution_time_ms": 1,
        "status": "OK",
        "request_metadata": {
            "foo": "bar",
            "k" * 1000: "v" * 1000,
            TRACE_SCHEMA_VERSION_KEY: "2",
        },
        "tags": {
            "baz": "qux",
            "k" * 2000: "v" * 8000,
        },
        "assessments": [],
    }


def test_trace_info_serialization_deserialization(trace_info_proto):
    # trace info proto -> TraceInfo
    trace_info = TraceInfoV2.from_proto(trace_info_proto)
    assert trace_info.request_id == "request_id"
    assert trace_info.experiment_id == "test_experiment"
    assert trace_info.timestamp_ms == 0
    assert trace_info.execution_time_ms == 1
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata == {
        "foo": "bar",
        "k" * 250: "v" * 250,
        TRACE_SCHEMA_VERSION_KEY: "2",
    }
    assert trace_info.tags == {
        "baz": "qux",
        "k" * 250: "v" * 250,
    }
    # TraceInfo -> python native dictionary
    trace_info_as_dict = trace_info.to_dict()
    assert trace_info_as_dict == {
        "request_id": "request_id",
        "experiment_id": "test_experiment",
        "timestamp_ms": 0,
        "execution_time_ms": 1,
        "status": "OK",
        "request_metadata": {
            "foo": "bar",
            "k" * 250: "v" * 250,
            TRACE_SCHEMA_VERSION_KEY: "2",
        },
        "tags": {
            "baz": "qux",
            "k" * 250: "v" * 250,
        },
        "assessments": [],
    }
    # python native dictionary -> TraceInfo
    assert TraceInfoV2.from_dict(trace_info_as_dict) == trace_info
    # TraceInfo -> trace info proto
    assert trace_info.to_proto() == trace_info_proto


def test_trace_info_v3(trace_info):
    v3_proto = trace_info.to_v3("request", "response").to_proto()
    assert v3_proto.request_preview == "request"
    assert v3_proto.response_preview == "response"
    assert v3_proto.trace_id == "request_id"
    assert isinstance(v3_proto.request_time, Timestamp)
    assert v3_proto.request_time.ToSeconds() == 0
    assert isinstance(v3_proto.execution_duration, Duration)
    assert v3_proto.execution_duration.ToMilliseconds() == 1
    assert v3_proto.state == 1
    assert v3_proto.trace_metadata["foo"] == "bar"
    assert (
        v3_proto.trace_metadata["k" * MAX_CHARS_IN_TRACE_INFO_METADATA]
        == "v" * MAX_CHARS_IN_TRACE_INFO_METADATA
    )
    assert v3_proto.tags["baz"] == "qux"
    assert (
        v3_proto.tags["k" * MAX_CHARS_IN_TRACE_INFO_TAGS_KEY]
        == "v" * MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE
    )


def test_trace_info_v2_to_v3_updates_schema_version():
    """Test that converting from TraceInfoV2 to V3 updates the schema version correctly."""
    # Create a V2 trace with old schema version in metadata
    trace_info_v2 = TraceInfoV2(
        request_id="test_request_id",
        experiment_id="test_experiment",
        timestamp_ms=1234567890,
        execution_time_ms=500,
        status=TraceStatus.OK,
        request_metadata={
            TRACE_SCHEMA_VERSION_KEY: "2",  # Old schema version
            "other_key": "other_value",
        },
        tags={"test_tag": "test_value"},
        assessments=[],
    )

    # Convert to V3
    trace_info_v3 = trace_info_v2.to_v3(request="test request", response="test response")

    # Verify the schema version was updated to current version
    assert trace_info_v3.trace_metadata[TRACE_SCHEMA_VERSION_KEY] == str(TRACE_SCHEMA_VERSION)

    # Verify other metadata is preserved
    assert trace_info_v3.trace_metadata["other_key"] == "other_value"

    # Verify the trace_metadata is a copy and doesn't affect original
    assert trace_info_v2.request_metadata[TRACE_SCHEMA_VERSION_KEY] == "2"  # Original unchanged
