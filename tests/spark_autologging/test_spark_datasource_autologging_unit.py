import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
import mlflow.spark
from mlflow._spark_autologging import _get_current_listener, PythonSubscriber
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import
from tests.spark_autologging.utils import _get_or_create_spark_session


@pytest.fixture()
def spark_session():
    session = _get_or_create_spark_session()
    yield session
    session.stop()


@pytest.fixture()
def mock_get_current_listener():
    with mock.patch("mlflow._spark_autologging._get_current_listener") as get_listener_patch:
        get_listener_patch.return_value = None
        yield get_listener_patch


@pytest.mark.large
def test_autolog_call_idempotent(spark_session, tracking_uri_mock):
    # pylint: disable=unused-argument
    mlflow.spark.autolog()
    listener = _get_current_listener()
    mlflow.spark.autolog()
    assert _get_current_listener() == listener


@pytest.mark.large
def test_subscriber_methods():
    # Test that PythonSubscriber satisfies the contract expected by the underlying Scala trait
    # it implements (MlflowAutologEventSubscriber)
    subscriber = PythonSubscriber()
    subscriber.ping()
    # Assert repl ID is stable & different between subscribers
    assert subscriber.replId() == subscriber.replId()
    assert PythonSubscriber().replId() != subscriber.replId()


@pytest.mark.large
def test_enabling_autologging_throws_for_wrong_spark_version(
        spark_session, tracking_uri_mock, mock_get_current_listener):
    # pylint: disable=unused-argument
    with mock.patch("mlflow._spark_autologging._get_spark_major_version") as get_version_mock:
        get_version_mock.return_value = 2
        with pytest.raises(MlflowException) as exc:
            mlflow.spark.autolog()
        assert "Spark autologging unsupported for Spark versions < 3" in exc.value.message


@pytest.mark.large
def test_enabling_autologging_throws_when_spark_hasnt_been_started(
        spark_session, tracking_uri_mock, mock_get_current_listener):
    # pylint: disable=unused-argument
    spark_session.stop()
    with pytest.raises(MlflowException) as exc:
        mlflow.spark.autolog()
    assert "No active SparkContext found" in exc.value.message
