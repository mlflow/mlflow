from mlflow.entities import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag


def test_to_proto():
    trace_info = TraceInfo(
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
