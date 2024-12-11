import pytest

from mlflow.entities import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceInfo as ProtoTraceInfo
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag


@pytest.fixture
def trace_info():
    return TraceInfo(
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
            "k" * 2000: "v" * 2000,
        },
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
    assert request_metadata_2.key == "k" * 250
    assert request_metadata_2.value == "v" * 250
    tag_1 = proto.tags[0]
    assert isinstance(tag_1, ProtoTraceTag)
    assert tag_1.key == "baz"
    assert tag_1.value == "qux"
    tag_2 = proto.tags[1]
    assert isinstance(tag_2, ProtoTraceTag)
    assert tag_2.key == "k" * 250
    assert tag_2.value == "v" * 250


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
        },
        "tags": {
            "baz": "qux",
            "k" * 2000: "v" * 2000,
        },
    }


def test_trace_info_serialization_deserialization(trace_info_proto):
    # trace info proto -> TraceInfo
    trace_info = TraceInfo.from_proto(trace_info_proto)
    assert trace_info.request_id == "request_id"
    assert trace_info.experiment_id == "test_experiment"
    assert trace_info.timestamp_ms == 0
    assert trace_info.execution_time_ms == 1
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata == {
        "foo": "bar",
        "k" * 250: "v" * 250,
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
        },
        "tags": {
            "baz": "qux",
            "k" * 250: "v" * 250,
        },
    }
    # python native dictionary -> TraceInfo
    assert TraceInfo.from_dict(trace_info_as_dict) == trace_info
    # TraceInfo -> trace info proto
    assert trace_info.to_proto() == trace_info_proto
