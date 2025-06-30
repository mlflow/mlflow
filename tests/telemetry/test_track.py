import json
import time
from dataclasses import asdict
from typing import Any
from unittest.mock import patch

import pytest
import sklearn

import mlflow
from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.telemetry.client import (
    TelemetryClient,
    get_telemetry_client,
    set_telemetry_client,
)
from mlflow.telemetry.schemas import APIStatus, AutologParams
from mlflow.telemetry.track import track_api_usage
from mlflow.telemetry.utils import is_telemetry_disabled

from tests.telemetry.helper_functions import wait_for_telemetry_threads


def full_func_name(func):
    return f"{func.__module__}.{func.__qualname__}"


def extract_record(data: str) -> dict[str, Any]:
    return json.loads(data["data"])


def test_track_api_usage(mock_requests):
    assert len(mock_requests) == 0

    @track_api_usage
    def succeed_func():
        time.sleep(0.1)
        return True

    @track_api_usage
    def fail_func():
        time.sleep(0.1)
        raise ValueError("test")

    succeed_func()
    with pytest.raises(ValueError, match="test"):
        fail_func()

    wait_for_telemetry_threads()

    assert len(mock_requests) == 2
    succeed_record = extract_record(mock_requests[0])
    assert succeed_record["api_name"] == full_func_name(succeed_func)
    assert succeed_record["status"] == APIStatus.SUCCESS.value
    assert succeed_record["params"] is None
    assert succeed_record["duration_ms"] > 0

    fail_record = extract_record(mock_requests[1])
    assert fail_record["api_name"] == full_func_name(fail_func)
    assert fail_record["status"] == APIStatus.FAILURE.value
    assert fail_record["params"] is None
    assert fail_record["duration_ms"] > 0

    telemetry_info = get_telemetry_client().info
    assert asdict(telemetry_info).items() <= succeed_record.items()
    assert asdict(telemetry_info).items() <= fail_record.items()


def test_backend_store_info(tmp_path):
    @track_api_usage
    def succeed_func():
        return True

    succeed_func()
    wait_for_telemetry_threads()

    telemetry_client = get_telemetry_client()
    assert telemetry_client.info.backend_store == "SqlAlchemyStore"

    mlflow.set_tracking_uri(tmp_path)
    succeed_func()
    wait_for_telemetry_threads()
    assert telemetry_client.info.backend_store == "FileStore"


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


def test_track_api_usage_update_env_var_after_import(monkeypatch, mock_requests):
    telemetry_client = get_telemetry_client()
    assert isinstance(telemetry_client, TelemetryClient)

    @track_api_usage
    def test_func():
        pass

    test_func()

    wait_for_telemetry_threads()
    assert len(mock_requests) == 1
    record = extract_record(mock_requests[0])
    assert record["api_name"] == full_func_name(test_func)

    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    test_func()
    # no new record should be added
    assert len(mock_requests) == 1


def test_track_api_usage_do_not_track_internal_api(mock_requests):
    def test_func():
        mlflow.sklearn.autolog()

    with patch("mlflow.telemetry.track.invoked_from_internal_api", return_value=True):
        test_func()
        assert len(mock_requests) == 0

    # mlflow.sklearn.autolog internally calls mlflow.sklearn.log_model
    mlflow.sklearn.autolog()

    iris = sklearn.datasets.load_iris()
    sklearn.cluster.KMeans().fit(iris.data[:, :2], iris.target)

    assert mlflow.last_logged_model() is not None

    wait_for_telemetry_threads()
    assert len(mock_requests) == 1
    record = extract_record(mock_requests[0])
    assert record["api_name"] == full_func_name(mlflow.sklearn.autolog)
    assert record["status"] == APIStatus.SUCCESS.value
    assert record["params"] == asdict(
        AutologParams(
            flavor="sklearn",
            disable=False,
            log_traces=False,
            log_models=True,
        )
    )
    assert record["duration_ms"] > 0


# TODO: apply track_api_usage to APIs and test the record params
