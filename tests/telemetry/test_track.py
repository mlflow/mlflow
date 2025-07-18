import time

import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.telemetry.client import (
    TelemetryClient,
    get_telemetry_client,
    set_telemetry_client,
)
from mlflow.telemetry.schemas import APIStatus
from mlflow.telemetry.track import track_api_usage
from mlflow.telemetry.utils import is_telemetry_disabled

from tests.helper_functions import validate_telemetry_record


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
    succeed_record = mock_requests[0]["data"]
    assert succeed_record["schema_version"] == 1
    assert succeed_record["api_module"] == succeed_func.__module__
    assert succeed_record["api_name"] == succeed_func.__qualname__
    assert succeed_record["status"] == APIStatus.SUCCESS.value
    assert succeed_record["params"] is None
    assert succeed_record["duration_ms"] > 0

    fail_record = mock_requests[1]["data"]
    assert fail_record["schema_version"] == 1
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
    record = mock_requests[0]["data"]
    assert record["api_module"] == test_func.__module__
    assert record["api_name"] == test_func.__qualname__

    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    test_func()
    # no new record should be added
    assert len(mock_requests) == 1


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
