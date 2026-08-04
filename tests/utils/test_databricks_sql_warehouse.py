import time
from datetime import timedelta
from unittest import mock

import pytest
from databricks.sdk.service.sql import State

from mlflow.environment_variables import (
    MLFLOW_SQL_WAREHOUSE_AUTO_START,
    MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS,
)
from mlflow.exceptions import MlflowException
from mlflow.utils import databricks_sql_warehouse
from mlflow.utils.databricks_sql_warehouse import ensure_sql_warehouse_running


@pytest.fixture(autouse=True)
def _clear_cache():
    databricks_sql_warehouse._verified_running.clear()
    yield
    databricks_sql_warehouse._verified_running.clear()


def _make_mock_client(state_value):
    client = mock.MagicMock()
    warehouse_info = mock.MagicMock()
    warehouse_info.state = State[state_value]
    client.warehouses.get.return_value = warehouse_info
    return client


def _patch_client(client):
    return mock.patch.object(databricks_sql_warehouse, "_get_workspace_client", return_value=client)


def test_running_warehouse_skips_start():
    client = _make_mock_client("RUNNING")
    with _patch_client(client):
        ensure_sql_warehouse_running("wh-1")
    client.warehouses.get.assert_called_once_with("wh-1")
    client.warehouses.start_and_wait.assert_not_called()


def test_stopped_warehouse_starts_and_waits_with_default_timeout():
    client = _make_mock_client("STOPPED")
    with (
        _patch_client(client),
        mock.patch.object(databricks_sql_warehouse._logger, "info") as log_info,
    ):
        ensure_sql_warehouse_running("wh-1")
    client.warehouses.start_and_wait.assert_called_once_with(
        "wh-1", timeout=timedelta(seconds=1200)
    )
    log_info.assert_called_once()
    rendered = log_info.call_args.args[0]
    assert "wh-1" in rendered
    assert "STOPPED" in rendered


@pytest.mark.parametrize("state", ["STOPPING", "STARTING"])
def test_transitional_state_triggers_start_and_wait(state):
    client = _make_mock_client(state)
    with _patch_client(client):
        ensure_sql_warehouse_running("wh-1")
    client.warehouses.start_and_wait.assert_called_once()


def test_second_call_within_ttl_is_cached(monkeypatch):
    monkeypatch.setattr(databricks_sql_warehouse, "_CACHE_TTL_SECONDS", 60.0)
    client = _make_mock_client("RUNNING")
    with _patch_client(client):
        ensure_sql_warehouse_running("wh-1")
        ensure_sql_warehouse_running("wh-1")
    assert client.warehouses.get.call_count == 1


def test_second_call_after_ttl_rechecks(monkeypatch):
    monkeypatch.setattr(databricks_sql_warehouse, "_CACHE_TTL_SECONDS", 0.0)
    client = _make_mock_client("RUNNING")
    with _patch_client(client):
        ensure_sql_warehouse_running("wh-1")
        # TTL = 0, so the cache entry is expired immediately.
        time.sleep(0.001)
        ensure_sql_warehouse_running("wh-1")
    assert client.warehouses.get.call_count == 2


def test_different_warehouse_ids_have_independent_cache():
    client = _make_mock_client("RUNNING")
    with _patch_client(client):
        ensure_sql_warehouse_running("wh-1")
        ensure_sql_warehouse_running("wh-2")
    assert client.warehouses.get.call_count == 2
    client.warehouses.get.assert_any_call("wh-1")
    client.warehouses.get.assert_any_call("wh-2")


def test_env_var_timeout_is_honored(monkeypatch):
    monkeypatch.setenv(MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS.name, "30")
    client = _make_mock_client("STOPPED")
    with _patch_client(client):
        ensure_sql_warehouse_running("wh-1")
    client.warehouses.start_and_wait.assert_called_once_with("wh-1", timeout=timedelta(seconds=30))


def test_timeout_error_raises_mlflow_exception():
    client = _make_mock_client("STOPPED")
    client.warehouses.start_and_wait.side_effect = TimeoutError("deadline exceeded")
    with _patch_client(client), pytest.raises(MlflowException, match="wh-1") as exc_info:
        ensure_sql_warehouse_running("wh-1")
    assert MLFLOW_SQL_WAREHOUSE_AUTO_START_TIMEOUT_SECONDS.name in str(exc_info.value)


def test_generic_sdk_error_raises_mlflow_exception():
    client = _make_mock_client("STOPPED")
    client.warehouses.start_and_wait.side_effect = RuntimeError("warehouse not found")
    with _patch_client(client), pytest.raises(MlflowException, match="wh-1") as exc_info:
        ensure_sql_warehouse_running("wh-1")
    assert "warehouse not found" in str(exc_info.value)
    assert "MLFLOW_SQL_WAREHOUSE_AUTO_START" in str(exc_info.value)


def test_auto_start_disabled_is_noop(monkeypatch):
    monkeypatch.setenv(MLFLOW_SQL_WAREHOUSE_AUTO_START.name, "false")
    client = _make_mock_client("STOPPED")
    with _patch_client(client):
        ensure_sql_warehouse_running("wh-1")
    client.warehouses.get.assert_not_called()
    client.warehouses.start_and_wait.assert_not_called()
