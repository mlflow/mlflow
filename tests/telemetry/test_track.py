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

from tests.helper_functions import validate_telemetry_record


def extract_record(data: str) -> dict[str, Any]:
    return json.loads(data["data"])


def test_track_api_usage(mock_requests):
    assert len(mock_requests) == 0

    @track_api_usage
    def succeed_func():
        # sleep to make sure duration_ms > 0
        time.sleep(0.01)
        return True

    @track_api_usage
    def fail_func():
        time.sleep(0.01)
        raise ValueError("test")

    succeed_func()
    with pytest.raises(ValueError, match="test"):
        fail_func()

    get_telemetry_client().flush()

    assert len(mock_requests) == 2
    succeed_record = extract_record(mock_requests[0])
    assert succeed_record["api_module"] == succeed_func.__module__
    assert succeed_record["api_name"] == succeed_func.__qualname__
    assert succeed_record["status"] == APIStatus.SUCCESS.value
    assert succeed_record["params"] is None
    assert succeed_record["duration_ms"] > 0

    fail_record = extract_record(mock_requests[1])
    assert fail_record["api_module"] == fail_func.__module__
    assert fail_record["api_name"] == fail_func.__qualname__
    assert fail_record["status"] == APIStatus.FAILURE.value
    assert fail_record["params"] is None
    assert fail_record["duration_ms"] > 0

    telemetry_info = get_telemetry_client().info
    assert telemetry_info.items() <= succeed_record.items()
    assert telemetry_info.items() <= fail_record.items()


def test_backend_store_info(tmp_path):
    @track_api_usage
    def succeed_func():
        return True

    succeed_func()
    get_telemetry_client().flush()

    telemetry_client = get_telemetry_client()
    assert telemetry_client.info["backend_store_scheme"] == "sqlite"

    mlflow.set_tracking_uri(tmp_path)
    succeed_func()
    get_telemetry_client().flush()
    assert telemetry_client.info["backend_store_scheme"] == "file"


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

    get_telemetry_client().flush()
    assert len(mock_requests) == 1
    record = extract_record(mock_requests[0])
    assert record["api_module"] == test_func.__module__
    assert record["api_name"] == test_func.__qualname__

    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    test_func()
    # no new record should be added
    assert len(mock_requests) == 1


def test_track_api_usage_do_not_track_internal_api(mock_requests):
    def test_func():
        mlflow.sklearn.autolog()

    with patch("mlflow.telemetry.track.should_skip_telemetry", return_value=True):
        test_func()
        assert len(mock_requests) == 0

    # mlflow.sklearn.autolog internally calls mlflow.sklearn.log_model
    mlflow.sklearn.autolog()

    iris = sklearn.datasets.load_iris()
    sklearn.cluster.KMeans().fit(iris.data[:, :2], iris.target)

    assert mlflow.last_logged_model() is not None

    get_telemetry_client().flush()
    assert len(mock_requests) == 1
    record = extract_record(mock_requests[0])
    assert record["api_module"] == "mlflow.sklearn"
    assert record["api_name"] == "autolog"
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


def test_track_api_usage_do_not_track_internal_api_complex(mock_requests):
    @track_api_usage
    def test_func_1():
        test_func_2()

    def test_func_2():
        test_func_3()

    @track_api_usage
    def test_func_3():
        pass

    test_func_1()
    validate_telemetry_record(mock_requests, test_func_1)


def test_trace_sends_telemetry_record(mock_requests):
    @mlflow.trace
    def test():
        pass

    validate_telemetry_record(mock_requests, mlflow.trace)

    def test_func():
        pass

    mlflow.trace(test_func)

    validate_telemetry_record(mock_requests, mlflow.trace, idx=1)


def test_spark_autolog_sends_telemetry_record(mock_requests):
    mlflow.spark.autolog(disable=True)

    validate_telemetry_record(
        mock_requests,
        mlflow.spark.autolog,
        params=AutologParams(
            flavor="spark",
            disable=True,
            log_traces=False,
            log_models=False,
        ),
    )
