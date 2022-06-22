import mlflow
from mlflow import MlflowClient

from mlflow.utils.class_utils import _get_class_from_string


def test_get_class_from_string():
    assert _get_class_from_string("MlflowClient") == MlflowClient
