from dataclasses import asdict
from unittest.mock import patch

import pytest
import sklearn

import mlflow
from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.telemetry.client import TelemetryClient, get_telemetry_client, set_telemetry_client
from mlflow.telemetry.schemas import APIStatus, AutologParams
from mlflow.telemetry.track import track_api_usage
from mlflow.telemetry.utils import is_telemetry_disabled


def full_func_name(func):
    return f"{func.__module__}.{func.__qualname__}"


def test_track_api_usage():
    telemetry_client = get_telemetry_client()
    assert telemetry_client.records == []

    @track_api_usage
    def succeed_func():
        return True

    @track_api_usage
    def fail_func():
        raise ValueError("test")

    succeed_func()
    with pytest.raises(ValueError, match="test"):
        fail_func()

    assert len(telemetry_client.records) == 2
    succeed_record = telemetry_client.records[0]
    assert asdict(succeed_record) == {
        "api_name": full_func_name(succeed_func),
        "params": None,
        "status": APIStatus.SUCCESS.value,
    }
    fail_record = telemetry_client.records[1]
    assert asdict(fail_record) == {
        "api_name": full_func_name(fail_func),
        "params": None,
        "status": APIStatus.FAILURE.value,
    }


@pytest.mark.parametrize(
    ("env_var", "value", "expected_result"),
    [
        (MLFLOW_DISABLE_TELEMETRY.name, "true", None),
        (MLFLOW_DISABLE_TELEMETRY.name, "false", TelemetryClient),
        ("DO_NOT_TRACK", "true", None),
        ("DO_NOT_TRACK", "false", TelemetryClient),
    ],
)
def test_track_api_usage_respect_env_var(monkeypatch, env_var, value, expected_result):
    monkeypatch.setenv(env_var, value)
    # mimic the behavior of `import mlflow`
    set_telemetry_client()
    telemetry_client = get_telemetry_client()
    if expected_result is None:
        assert is_telemetry_disabled() is True
        assert telemetry_client is None
    else:
        assert isinstance(telemetry_client, expected_result)


def test_track_api_usage_update_env_var_after_import(monkeypatch):
    telemetry_client = get_telemetry_client()
    assert isinstance(telemetry_client, TelemetryClient)

    @track_api_usage
    def test_func():
        pass

    test_func()
    assert len(telemetry_client.records) == 1
    assert telemetry_client.records[0].api_name == full_func_name(test_func)

    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    test_func()
    # no new record should be added
    assert len(telemetry_client.records) == 1


def test_track_api_usage_do_not_track_internal_api():
    def test_func():
        mlflow.sklearn.autolog()

    with patch("mlflow.telemetry.track.invoked_from_internal_api", return_value=True):
        test_func()
        assert len(get_telemetry_client().records) == 0

    # mlflow.sklearn.autolog internally calls mlflow.sklearn.log_model
    mlflow.sklearn.autolog()

    iris = sklearn.datasets.load_iris()
    sklearn.cluster.KMeans().fit(iris.data[:, :2], iris.target)

    assert mlflow.last_logged_model() is not None
    records = get_telemetry_client().records
    assert len(records) == 1
    assert asdict(records[0]) == {
        "api_name": full_func_name(mlflow.sklearn.autolog),
        "params": asdict(
            AutologParams(
                flavor="sklearn",
                disable=False,
                log_traces=False,
                log_models=True,
            )
        ),
        "status": APIStatus.SUCCESS.value,
    }


# TODO: apply track_api_usage to APIs and test the record params
