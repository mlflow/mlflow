from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import Experiment, GetExperiment
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.server_info import (
    SERVER_FEATURES_ENDPOINT,
    SERVER_INFO_ENDPOINT,
    _clear_server_info_cache,
)

ACTIVE_WORKSPACE = "team-a"


@pytest.fixture(autouse=True)
def clear_server_info_cache():
    _clear_server_info_cache()
    yield
    _clear_server_info_cache()


def test_supports_workspaces_queries_endpoint():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {"workspaces_enabled": True}

    with mock.patch("mlflow.utils.server_info.http_request", return_value=response) as mock_http:
        assert store.supports_workspaces is True
        # Cached result prevents additional requests
        assert store.supports_workspaces is True

    mock_http.assert_called_once()
    _, kwargs = mock_http.call_args
    assert kwargs["host_creds"] is creds
    assert kwargs["endpoint"] == SERVER_INFO_ENDPOINT
    assert kwargs["method"] == "GET"
    assert kwargs["timeout"] == 3
    assert kwargs["max_retries"] == 0
    assert kwargs["raise_on_status"] is False


def test_supports_workspaces_raises_when_tracking_uri_has_wrong_path():
    creds = MlflowHostCreds("https://example/not-mlflow")
    store = RestStore(lambda: creds)
    not_found_response = mock.MagicMock(status_code=404, text="not found")

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=[not_found_response, not_found_response],
    ) as mock_http:
        with pytest.raises(
            MlflowException, match="Verify that the configured MLflow URI"
        ) as exc_info:
            store.supports_workspaces

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "--enable-workspaces" not in str(exc_info.value)
    assert [call.kwargs["endpoint"] for call in mock_http.call_args_list] == [
        SERVER_INFO_ENDPOINT,
        SERVER_FEATURES_ENDPOINT,
    ]


def test_workspace_guard_includes_active_workspace_when_support_is_undetermined(monkeypatch):
    store = RestStore(lambda: MlflowHostCreds("https://example/not-mlflow"))
    not_found_response = mock.MagicMock(status_code=404, text="not found")
    monkeypatch.setattr(
        "mlflow.store.workspace_rest_store_mixin.get_request_workspace", lambda: ACTIVE_WORKSPACE
    )

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=[not_found_response, not_found_response],
    ) as mock_http:
        with pytest.raises(
            MlflowException,
            match="Active workspace 'team-a' cannot be used. MLflow could not determine",
        ) as exc_info:
            store._validate_workspace_support_if_specified()

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "--enable-workspaces" in str(exc_info.value)
    assert [call.kwargs["endpoint"] for call in mock_http.call_args_list] == [
        SERVER_INFO_ENDPOINT,
        SERVER_FEATURES_ENDPOINT,
    ]


def test_supports_workspaces_uses_legacy_server_features_endpoint():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    server_info_response = mock.MagicMock(status_code=404, text="not found")
    server_features_response = mock.MagicMock(status_code=200)
    server_features_response.json.return_value = {"workspaces_enabled": True}

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=[server_info_response, server_features_response],
    ) as mock_http:
        assert store.supports_workspaces is True

    assert [call.kwargs["endpoint"] for call in mock_http.call_args_list] == [
        SERVER_INFO_ENDPOINT,
        SERVER_FEATURES_ENDPOINT,
    ]


@pytest.mark.parametrize(
    "responses",
    [
        [mock.MagicMock(status_code=200)],
        [mock.MagicMock(status_code=404, text="not found"), mock.MagicMock(status_code=200)],
    ],
    ids=["current-server", "legacy-server"],
)
def test_workspace_guard_retains_workspace_disabled_error(responses, monkeypatch):
    for response in responses:
        response.json.return_value = {"workspaces_enabled": False}
    store = RestStore(lambda: MlflowHostCreds("https://example"))
    monkeypatch.setattr(
        "mlflow.store.workspace_rest_store_mixin.get_request_workspace", lambda: ACTIVE_WORKSPACE
    )

    with mock.patch("mlflow.utils.server_info.http_request", side_effect=responses) as mock_http:
        with pytest.raises(
            MlflowException, match="Restart the server with --enable-workspaces"
        ) as exc_info:
            store._validate_workspace_support_if_specified()

    assert exc_info.value.error_code == "FEATURE_DISABLED"
    assert [call.kwargs["endpoint"] for call in mock_http.call_args_list] == (
        [SERVER_INFO_ENDPOINT]
        if len(responses) == 1
        else [SERVER_INFO_ENDPOINT, SERVER_FEATURES_ENDPOINT]
    )


def test_supports_workspaces_handles_missing_json_keys():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {}

    with mock.patch("mlflow.utils.server_info.http_request", return_value=response):
        assert store.supports_workspaces is False


def test_supports_workspaces_returns_false_for_databricks_uri():
    creds = MlflowHostCreds("databricks")
    store = RestStore(lambda: creds)

    with mock.patch("mlflow.utils.server_info.http_request") as mock_http:
        assert store.supports_workspaces is False
        # Should not probe the server for Databricks URIs
        mock_http.assert_not_called()


def test_supports_workspaces_raises_on_server_error():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 500
    response.text = "Internal Server Error"

    with mock.patch("mlflow.utils.server_info.http_request", return_value=response):
        with pytest.raises(MlflowException, match="Failed to query.*500"):
            store.supports_workspaces


def test_supports_workspaces_raises_on_legacy_server_features_error():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    server_info_response = mock.MagicMock(status_code=404, text="not found")
    server_features_response = mock.MagicMock(status_code=500, text="Internal Server Error")

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=[server_info_response, server_features_response],
    ) as mock_http:
        with pytest.raises(
            MlflowException, match=f"Failed to query {SERVER_FEATURES_ENDPOINT}: 500"
        ) as exc_info:
            store.supports_workspaces

    assert exc_info.value.error_code == "TEMPORARILY_UNAVAILABLE"
    assert [call.kwargs["endpoint"] for call in mock_http.call_args_list] == [
        SERVER_INFO_ENDPOINT,
        SERVER_FEATURES_ENDPOINT,
    ]


def test_supports_workspaces_raises_on_legacy_server_features_request_error():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    server_info_response = mock.MagicMock(status_code=404, text="not found")

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=[server_info_response, ConnectionError("connection refused")],
    ) as mock_http:
        with pytest.raises(
            MlflowException,
            match=f"Failed to query {SERVER_FEATURES_ENDPOINT}: connection refused",
        ) as exc_info:
            store.supports_workspaces

    assert exc_info.value.error_code == "INTERNAL_ERROR"
    assert [call.kwargs["endpoint"] for call in mock_http.call_args_list] == [
        SERVER_INFO_ENDPOINT,
        SERVER_FEATURES_ENDPOINT,
    ]


def test_supports_workspaces_raises_internal_error_on_request_exception():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)

    with mock.patch(
        "mlflow.utils.server_info.http_request",
        side_effect=ConnectionError("connection refused"),
    ):
        with pytest.raises(MlflowException, match="Failed to query.*connection refused"):
            store.supports_workspaces


def test_supports_workspaces_raises_internal_error_on_malformed_json():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    response = mock.MagicMock(status_code=200)
    response.json.side_effect = ValueError("bad json")

    with mock.patch("mlflow.utils.server_info.http_request", return_value=response):
        with pytest.raises(MlflowException, match="Invalid JSON returned"):
            store.supports_workspaces


def test_supports_workspaces_shares_successful_result_across_store_instances():
    creds1 = MlflowHostCreds("https://example")
    creds2 = MlflowHostCreds("https://example")
    store1 = RestStore(lambda: creds1)
    store2 = RestStore(lambda: creds2)
    response = mock.MagicMock(status_code=200)
    response.json.return_value = {"workspaces_enabled": True}

    with mock.patch("mlflow.utils.server_info.http_request", return_value=response) as mock_http:
        assert store1.supports_workspaces is True
        assert store2.supports_workspaces is True

    mock_http.assert_called_once()


def test_supports_workspaces_does_not_share_different_path_prefixes():
    store1 = RestStore(lambda: MlflowHostCreds("https://example"))
    store2 = RestStore(lambda: MlflowHostCreds("https://example/mlflow"))
    response = mock.MagicMock(status_code=200)
    response.json.return_value = {"workspaces_enabled": True}

    with mock.patch("mlflow.utils.server_info.http_request", return_value=response) as mock_http:
        assert store1.supports_workspaces is True
        assert store2.supports_workspaces is True

    assert mock_http.call_count == 2


def test_rest_store_workspace_guard():
    creds = MlflowHostCreds("https://example")
    store = RestStore(lambda: creds)
    store._workspace_support = False

    with (
        mock.patch(
            "mlflow.store.workspace_rest_store_mixin.get_request_workspace",
            return_value=ACTIVE_WORKSPACE,
        ),
        mock.patch.object(RestStore, "supports_workspaces", property(lambda self: False)),
    ):
        with pytest.raises(
            MlflowException,
            match="Active workspace 'team-a' cannot be used because the remote server does not",
        ):
            store.search_experiments()


def test_workspace_guard_blocks_log_spans(monkeypatch):
    store = RestStore(lambda: MlflowHostCreds("https://workspace-host"))
    spans = [mock.MagicMock()]

    monkeypatch.setattr(
        "mlflow.store.workspace_rest_store_mixin.get_request_workspace",
        lambda: ACTIVE_WORKSPACE,
    )
    monkeypatch.setattr(RestStore, "supports_workspaces", property(lambda self: False))

    with pytest.raises(MlflowException, match="does not support workspaces"):
        store.log_spans("exp-1", spans)


def test_rest_store_get_experiment_has_workspace():
    proto = Experiment(
        experiment_id="1",
        name="test",
        artifact_location="/tmp",
        workspace="other_workspace",
    )
    response = GetExperiment.Response(experiment=proto)

    with mock.patch.object(RestStore, "_call_endpoint", return_value=response) as mock_call:
        store = RestStore(lambda: MlflowHostCreds("https://hello"))
        result = store.get_experiment("1")

    assert result.workspace == "other_workspace"
    mock_call.assert_called_once()
