from unittest import mock

import pytest

import mlflow
import mlflow.spark
from mlflow.exceptions import MlflowException
from mlflow.spark.autologging import PythonSubscriber, _get_current_listener

from tests.spark.autologging.utils import _get_or_create_spark_session


@pytest.fixture(scope="module")
def spark_session():
    with _get_or_create_spark_session() as session:
        yield session


@pytest.fixture
def mock_get_current_listener():
    with mock.patch(
        "mlflow.spark.autologging._get_current_listener", return_value=None
    ) as get_listener_patch:
        yield get_listener_patch


@pytest.mark.usefixtures("spark_session")
def test_autolog_call_idempotent():
    mlflow.spark.autolog()
    listener = _get_current_listener()
    mlflow.spark.autolog()
    assert _get_current_listener() == listener


def test_subscriber_methods():
    # Test that PythonSubscriber satisfies the contract expected by the underlying Scala trait
    # it implements (MlflowAutologEventSubscriber)
    subscriber = PythonSubscriber()
    subscriber.ping()
    # Assert repl ID is stable & different between subscribers
    assert subscriber.replId() == subscriber.replId()
    assert PythonSubscriber().replId() != subscriber.replId()


def test_enabling_autologging_throws_for_wrong_spark_version(
    spark_session, mock_get_current_listener
):
    with mock.patch("mlflow.spark.autologging._get_spark_major_version", return_value=2):
        with pytest.raises(
            MlflowException, match="Spark autologging unsupported for Spark versions < 3"
        ):
            mlflow.spark.autolog()
