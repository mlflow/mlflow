import pytest
from unittest import mock

import mlflux
from mlflux.exceptions import MlflowException
import mlflux.spark
from mlflux._spark_autologging import _get_current_listener, PythonSubscriber
from tests.spark_autologging.utils import _get_or_create_spark_session


@pytest.fixture()
def spark_session():
    session = _get_or_create_spark_session()
    yield session
    session.stop()


@pytest.fixture()
def mock_get_current_listener():
    with mock.patch("mlflux._spark_autologging._get_current_listener") as get_listener_patch:
        get_listener_patch.return_value = None
        yield get_listener_patch


@pytest.mark.large
@pytest.mark.usefixtures("spark_session")
def test_autolog_call_idempotent():
    mlflux.spark.autolog()
    listener = _get_current_listener()
    mlflux.spark.autolog()
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
    spark_session, mock_get_current_listener
):
    # pylint: disable=unused-argument
    with mock.patch("mlflux._spark_autologging._get_spark_major_version") as get_version_mock:
        get_version_mock.return_value = 2

        with pytest.raises(MlflowException) as exc:
            mlflux.spark.autolog()
        assert "Spark autologging unsupported for Spark versions < 3" in exc.value.message
