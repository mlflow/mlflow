import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracking._tracking_service.utils import _get_databricks_rest_store
from mlflow.utils.databricks_utils import get_databricks_host_creds

pytest.importorskip("databricks.sdk")

_HOST = "https://test.cloud.databricks.com"


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "true")
    monkeypatch.setenv("DATABRICKS_HOST", _HOST)
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(tmp_path / ".databrickscfg"))
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    return _get_databricks_rest_store("databricks")


def test_repeated_requests_probe_once(store):
    client = SimpleNamespace(config=SimpleNamespace(host=_HOST, workspace_id="123"))
    with (
        mock.patch("databricks.sdk.WorkspaceClient", return_value=client) as probe,
        mock.patch("mlflow.store.tracking.rest_store.call_endpoint") as call_endpoint,
    ):
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="probe-test") as pool:
            list(pool.map(store.delete_experiment, range(10)))

    probe.assert_called_once_with()
    assert call_endpoint.call_count == 10
    for call in call_endpoint.call_args_list:
        creds = call.args[0]
        assert creds.use_databricks_sdk
        assert creds.host == _HOST
        assert creds.workspace_id == "123"


def test_retries_failed_probe(store):
    client = SimpleNamespace(config=SimpleNamespace(host=_HOST, workspace_id="123"))
    with mock.patch("databricks.sdk.WorkspaceClient", side_effect=[ValueError(), client]) as probe:
        fallback = store.get_host_creds()
        assert not fallback.use_databricks_sdk
        assert fallback.token == "test-token"
        assert store.get_host_creds().use_databricks_sdk
        assert store.get_host_creds().workspace_id == "123"
        assert probe.call_args_list == [mock.call(), mock.call()]


def test_sdk_only_host_and_workspace_id(store):
    client = SimpleNamespace(config=SimpleNamespace(host="https://sdk-host", workspace_id="321"))
    with (
        mock.patch("databricks.sdk.WorkspaceClient", return_value=client) as probe,
        mock.patch(
            "mlflow.utils.databricks_utils._get_databricks_creds_config",
            side_effect=MlflowException("No legacy credentials"),
        ) as legacy,
    ):
        for _ in range(2):
            creds = store.get_host_creds()
            assert creds.host == "https://sdk-host"
            assert creds.workspace_id == "321"
            assert creds.use_databricks_sdk
            assert creds.token is None

    probe.assert_called_once_with()
    assert legacy.call_args_list == [mock.call("databricks")] * 2


def test_store_reprobes_with_new_configuration(store, monkeypatch):
    new_host = "https://other.cloud.databricks.com"
    with mock.patch(
        "databricks.sdk.WorkspaceClient",
        side_effect=[
            SimpleNamespace(config=SimpleNamespace(host=_HOST, workspace_id="1")),
            SimpleNamespace(config=SimpleNamespace(host=new_host, workspace_id="2")),
        ],
    ) as probe:
        assert store.get_host_creds().workspace_id == "1"
        monkeypatch.setenv("DATABRICKS_HOST", new_host)
        monkeypatch.setenv("DATABRICKS_TOKEN", "rotated-token")
        creds = store.get_host_creds()
        assert creds.workspace_id == "2"
        assert creds.host == new_host
        assert creds.token == "rotated-token"
        assert probe.call_args_list == [mock.call(), mock.call()]


def test_profile_file_change_reprobes(store, monkeypatch):
    config_file = Path(os.environ["DATABRICKS_CONFIG_FILE"])
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "team")
    with mock.patch("databricks.sdk.WorkspaceClient") as probe:
        for timestamp, host in enumerate(("one", "two"), start=1):
            config_file.write_text(f"[team]\nhost = https://{host}\ntoken = test-token\n")
            os.utime(config_file, (timestamp, timestamp))
            for _ in range(2):
                assert store.get_host_creds().host == f"https://{host}"
    assert probe.call_args_list == [mock.call(profile="team")] * 2


def test_ambient_workspace_is_resolved_per_call(store, monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST")
    hosts = ["https://first", "https://second"]
    clients = [SimpleNamespace(config=SimpleNamespace(host=host)) for host in hosts]
    with mock.patch("databricks.sdk.WorkspaceClient", side_effect=clients) as probe:
        assert [store.get_host_creds().host for _ in hosts] == hosts
        assert probe.call_args_list == [mock.call(), mock.call()]


def test_disabled_sdk_skips_probe(store, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "false")
    with mock.patch("databricks.sdk.WorkspaceClient") as probe:
        creds = store.get_host_creds()
    assert not creds.use_databricks_sdk
    assert creds.token == "test-token"
    probe.assert_not_called()


def test_direct_calls_probe_each_time(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "true")
    with mock.patch("databricks.sdk.WorkspaceClient") as probe:
        for _ in range(2):
            assert get_databricks_host_creds("databricks://team").use_databricks_sdk
    assert probe.call_args_list == [mock.call(profile="team")] * 2


def test_unpickled_store_probes_again(store):
    with mock.patch("databricks.sdk.WorkspaceClient") as probe:
        assert store.get_host_creds().use_databricks_sdk
        restored = pickle.loads(pickle.dumps(store))
        assert restored.get_host_creds().use_databricks_sdk
        assert probe.call_args_list == [mock.call(), mock.call()]


def test_secret_scope_reads_current_token(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "true")
    store = _get_databricks_rest_store("databricks://scope:prefix")
    dbutils = mock.Mock()
    dbutils.secrets.get.side_effect = [_HOST, "first", _HOST, "second"]
    monkeypatch.setattr("mlflow.utils.databricks_utils._get_dbutils", lambda: dbutils)
    with mock.patch("databricks.sdk.WorkspaceClient") as probe:
        assert store.get_host_creds().token == "first"
        assert store.get_host_creds().token == "second"

    assert probe.call_args_list == [mock.call(host=_HOST, token=t) for t in ("first", "second")]
    assert (
        dbutils.secrets.get.call_args_list
        == [
            mock.call(scope="scope", key="prefix-host"),
            mock.call(scope="scope", key="prefix-token"),
        ]
        * 2
    )
