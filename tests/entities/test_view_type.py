import pytest

from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos import service_pb2


def test_to_proto():
    assert ViewType.to_proto(ViewType.ACTIVE_ONLY) == service_pb2.ACTIVE_ONLY
    assert ViewType.to_proto(ViewType.DELETED_ONLY) == service_pb2.DELETED_ONLY
    assert ViewType.to_proto(ViewType.ALL) == service_pb2.ALL


def test_from_proto():
    assert ViewType.from_proto(service_pb2.ACTIVE_ONLY) == ViewType.ACTIVE_ONLY
    assert ViewType.from_proto(service_pb2.DELETED_ONLY) == ViewType.DELETED_ONLY
    assert ViewType.from_proto(service_pb2.ALL) == ViewType.ALL



def test_an_unknown_view_type_raises_the_named_mlflow_exception():
    with pytest.raises(MlflowException, match="Could not get valid view type"):
        ViewType.from_string("not_a_view")


def test_an_unknown_view_type_code_raises_the_named_mlflow_exception():
    with pytest.raises(MlflowException, match="Could not get valid view type"):
        ViewType.to_string(99)
