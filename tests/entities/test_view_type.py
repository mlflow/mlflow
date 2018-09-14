from mlflow.protos import service_pb2
from mlflow.entities import ViewType


def test_to_proto():
    assert ViewType.to_proto(ViewType.ACTIVE_ONLY) == service_pb2.ACTIVE_ONLY
    assert ViewType.to_proto(ViewType.DELETED_ONLY) == service_pb2.DELETED_ONLY
    assert ViewType.to_proto(ViewType.ALL) == service_pb2.ALL


def test_from_proto():
    assert ViewType.from_proto(service_pb2.ACTIVE_ONLY) == ViewType.ACTIVE_ONLY
    assert ViewType.from_proto(service_pb2.DELETED_ONLY) == ViewType.DELETED_ONLY
    assert ViewType.from_proto(service_pb2.ALL) == ViewType.ALL
