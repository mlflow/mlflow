import json
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import requests
from databricks.sdk.oauth import ClientCredentials, Token
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from mlflow.entities.span import Span
from mlflow.tracing.export.databricks_otel_client import (
    _REQUEST_TIMEOUT_SECONDS,
    _SDK_HTTP_TIMEOUT_SECONDS,
    _TOKEN_REQUEST_TIMEOUT_SECONDS,
    DatabricksOTelClient,
    ZerobusOtelTokenError,
    ZerobusOtelTokenRefreshError,
    _resolve_collector_credentials,
    build_databricks_otel_collector_token_source,
    build_table_authorization_details,
    is_databricks_otel_collector_host,
    resolve_databricks_otel_collector_endpoint,
)
from mlflow.tracing.export.databricks_otel_collector import (
    DatabricksOtelCollectorSpanExporter,
    DatabricksOtelSerializationError,
)

from tests.tracing.helper import create_mock_otel_span

_MODULE = "mlflow.tracing.export.databricks_otel_client"

_TABLE_NAME = "cat.sch.tbl"
_ENDPOINT = "12345678.zerobus.us-west-2.cloud.databricks.com"
_HOST = "https://adb-12345678.azuredatabricks.net"
_WORKSPACE_ID = "12345678"


@pytest.fixture(autouse=True)
def _reset_collector_config_warning():
    import mlflow.tracing.export.databricks_otel_client as client_module

    client_module._collector_config_failure_warned = False
    client_module._resolved_endpoints.clear()
    yield
    client_module._collector_config_failure_warned = False
    client_module._resolved_endpoints.clear()


def _make_json_response(body, status_code=200):
    response = mock.MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.ok = status_code < 400
    response.content = json.dumps(body).encode()
    response.json.return_value = body
    response.raise_for_status.return_value = None
    return response


@pytest.mark.parametrize(
    ("endpoint", "workspace_id", "expected"),
    [
        ("12345678.zerobus.us-west-2.cloud.databricks.com", "12345678", True),
        ("12345678.zerobus.eastus.azuredatabricks.net", "12345678", True),
        ("12345678.zerobus.us-central1.gcp.databricks.com", "12345678", True),
        ("https://12345678.zerobus.us-west-2.cloud.databricks.com", "12345678", True),
        ("12345678.zerobus.us-west-2.staging.cloud.databricks.com", "12345678", True),
        ("99999999.zerobus.us-west-2.cloud.databricks.com", "12345678", False),
        ("12345678.zerobus.us-west-2.cloud.databricks.com:443", "12345678", False),
        ("12345678.zerobus.us-west-2.cloud.databricks.com/v1/traces", "12345678", False),
        ("user@12345678.zerobus.us-west-2.cloud.databricks.com", "12345678", False),
        ("12345678.zerobus.us-west-2.example.com", "12345678", False),
        ("12345678.zerobus.cloud.databricks.com", "12345678", False),
        ("12345678.zerobus..cloud.databricks.com", "12345678", False),
    ],
)
def test_is_databricks_otel_collector_host(endpoint: str, workspace_id: str, expected: bool):
    assert is_databricks_otel_collector_host(endpoint, workspace_id) == expected


def _make_summary(region="us-west-2", cloud="aws", global_metastore_id=None):
    return {
        "region": region,
        "cloud": cloud,
        "global_metastore_id": global_metastore_id,
    }


@pytest.mark.parametrize(
    ("region", "cloud", "host", "expected_endpoint"),
    [
        (
            "us-west-2",
            "aws",
            "https://adb-12345678.azuredatabricks.net",
            "12345678.zerobus.us-west-2.cloud.databricks.com",
        ),
        (
            "eastus",
            "azure",
            "https://adb-12345678.azuredatabricks.net",
            "12345678.zerobus.eastus.azuredatabricks.net",
        ),
        (
            "us-central1",
            "gcp",
            "https://12345678.gcp.databricks.com",
            "12345678.zerobus.us-central1.gcp.databricks.com",
        ),
        (
            "us-west-2",
            "aws",
            "https://adb-12345678.staging.cloud.databricks.com",
            "12345678.zerobus.us-west-2.staging.cloud.databricks.com",
        ),
    ],
)
def test_resolve_endpoint_assembles_host_from_metastore_summary(
    monkeypatch, region, cloud, host, expected_endpoint
):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    token_response = _make_json_response({
        "access_token": "metadata-token",
        "expires_in": 3600,
        "token_type": "Bearer",
    })
    summary_response = _make_json_response(_make_summary(region, cloud))
    with (
        mock.patch(f"{_MODULE}.requests.post", return_value=token_response) as mock_post,
        mock.patch(f"{_MODULE}.requests.get", return_value=summary_response) as mock_get,
    ):
        result = resolve_databricks_otel_collector_endpoint(
            host=host,
            workspace_id="12345678",
            client_id="sp-client-id",
            client_secret="sp-secret",
        )

    assert result == expected_endpoint
    mock_post.assert_called_once_with(
        f"{host}/oidc/v1/token",
        data={"grant_type": "client_credentials", "scope": "all-apis"},
        auth=mock.ANY,
        timeout=_TOKEN_REQUEST_TIMEOUT_SECONDS,
    )
    mock_get.assert_called_once_with(
        f"{host}/api/2.1/unity-catalog/metastore_summary",
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer metadata-token",
            "X-Databricks-Workspace-Id": "12345678",
        },
        timeout=_SDK_HTTP_TIMEOUT_SECONDS,
    )


def test_resolve_endpoint_override_takes_precedence(monkeypatch):
    valid_override = "12345678.zerobus.eu-west-1.cloud.databricks.com"
    monkeypatch.setenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", valid_override)
    with (
        mock.patch(f"{_MODULE}.requests.post") as mock_post,
        mock.patch(f"{_MODULE}.requests.get") as mock_get,
    ):
        result = resolve_databricks_otel_collector_endpoint(
            host=_HOST,
            workspace_id=_WORKSPACE_ID,
        )

    assert result == valid_override
    mock_post.assert_not_called()
    mock_get.assert_not_called()


def test_resolve_endpoint_override_normalizes_https_prefix(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT",
        "https://12345678.zerobus.eu-west-1.cloud.databricks.com",
    )

    with (
        mock.patch(f"{_MODULE}.requests.post") as mock_post,
        mock.patch(f"{_MODULE}.requests.get") as mock_get,
    ):
        result = resolve_databricks_otel_collector_endpoint(_HOST, _WORKSPACE_ID)

    assert result == "12345678.zerobus.eu-west-1.cloud.databricks.com"
    mock_post.assert_not_called()
    mock_get.assert_not_called()


def test_resolve_endpoint_invalid_override_returns_none(monkeypatch):
    monkeypatch.setenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", "badhost.example.com")
    with (
        mock.patch(f"{_MODULE}.requests.post") as mock_post,
        mock.patch(f"{_MODULE}.requests.get") as mock_get,
    ):
        result = resolve_databricks_otel_collector_endpoint(_HOST, _WORKSPACE_ID)

    assert result is None
    mock_post.assert_not_called()
    mock_get.assert_not_called()


def test_resolve_endpoint_invalid_override_warns_once(monkeypatch):
    monkeypatch.setenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", "badhost.example.com")
    with mock.patch(f"{_MODULE}._logger") as mock_log:
        result = resolve_databricks_otel_collector_endpoint(_HOST, _WORKSPACE_ID)

    assert result is None
    mock_log.warning.assert_called_once()


def test_resolve_endpoint_cloud_from_global_metastore_id(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    token_response = _make_json_response({
        "access_token": "metadata-token",
        "expires_in": 3600,
        "token_type": "Bearer",
    })
    summary_response = _make_json_response(
        _make_summary(region="us-west-2", cloud=None, global_metastore_id="aws:us-west-2:abc")
    )
    with (
        mock.patch(f"{_MODULE}.requests.post", return_value=token_response) as mock_post,
        mock.patch(f"{_MODULE}.requests.get", return_value=summary_response) as mock_get,
    ):
        result = resolve_databricks_otel_collector_endpoint(
            _HOST, _WORKSPACE_ID, "sp-client-id", "sp-secret"
        )

    assert result == "12345678.zerobus.us-west-2.cloud.databricks.com"
    mock_post.assert_called_once()
    mock_get.assert_called_once()


def test_resolve_endpoint_metastore_error_returns_none(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    with (
        mock.patch(
            f"{_MODULE}.requests.post", side_effect=RuntimeError("connection refused")
        ) as mock_post,
        mock.patch(f"{_MODULE}.requests.get") as mock_get,
    ):
        result = resolve_databricks_otel_collector_endpoint(
            _HOST, _WORKSPACE_ID, "sp-client-id", "sp-secret"
        )

    assert result is None
    mock_post.assert_called_once()
    mock_get.assert_not_called()


@pytest.mark.parametrize(
    "summary_setup",
    [
        pytest.param({"__raise__": True}, id="metastore-fetch-failure"),
        pytest.param({"region": "", "cloud": "aws"}, id="empty-region"),
        pytest.param({"region": "us-west-2", "cloud": "mars"}, id="unrecognised-cloud"),
        pytest.param({"region": "bad/region", "cloud": "aws"}, id="assembled-host-invalid"),
    ],
)
def test_resolve_endpoint_failure_warns_once(monkeypatch, summary_setup):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    summary_patch = (
        mock.patch(f"{_MODULE}._get_metastore_summary", side_effect=RuntimeError("boom"))
        if summary_setup.get("__raise__")
        else mock.patch(f"{_MODULE}._get_metastore_summary", return_value=summary_setup)
    )
    with summary_patch, mock.patch(f"{_MODULE}._logger") as mock_log:
        result = resolve_databricks_otel_collector_endpoint(
            _HOST, _WORKSPACE_ID, "sp-client-id", "sp-secret"
        )

    assert result is None
    mock_log.warning.assert_called_once()


def test_resolve_endpoint_failure_warns_once_per_process(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    with (
        mock.patch(f"{_MODULE}._get_metastore_summary", side_effect=RuntimeError("boom")),
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        first = resolve_databricks_otel_collector_endpoint(
            _HOST, _WORKSPACE_ID, "sp-client-id", "sp-secret"
        )
        second = resolve_databricks_otel_collector_endpoint(
            _HOST, _WORKSPACE_ID, "sp-client-id", "sp-secret"
        )

    assert first is None
    assert second is None
    mock_log.warning.assert_called_once()
    mock_log.debug.assert_called_once()


def test_build_table_authorization_details_structure():
    result = build_table_authorization_details([_TABLE_NAME])
    entries = json.loads(result)
    assert len(entries) == 3
    assert {entry["object_type"] for entry in entries} == {"CATALOG", "SCHEMA", "TABLE"}
    assert next(e for e in entries if e["object_type"] == "CATALOG") == {
        "type": "unity_catalog_privileges",
        "object_type": "CATALOG",
        "object_full_path": "cat",
        "privileges": ["USE CATALOG"],
    }
    assert next(e for e in entries if e["object_type"] == "SCHEMA")["object_full_path"] == "cat.sch"
    table_entry = next(e for e in entries if e["object_type"] == "TABLE")
    assert table_entry["object_full_path"] == _TABLE_NAME
    assert table_entry["privileges"] == ["SELECT", "MODIFY"]


def test_build_table_authorization_details_skips_malformed_tables():
    result = build_table_authorization_details(["badtable", "cat.sch.tbl"])
    entries = json.loads(result)
    assert len(entries) == 3
    assert all(entry["object_full_path"] != "badtable" for entry in entries)


def test_build_databricks_otel_collector_token_source_construction():
    with mock.patch("databricks.sdk.oauth.ClientCredentials", autospec=True) as mock_cc:
        token_source = build_databricks_otel_collector_token_source(
            host=_HOST,
            client_id="my-client-id",
            client_secret="my-client-secret",
            workspace_id=_WORKSPACE_ID,
            tables=[_TABLE_NAME],
        )

    mock_cc.assert_called_once()
    _, kwargs = mock_cc.call_args
    assert kwargs["client_id"] == "my-client-id"
    assert kwargs["client_secret"] == "my-client-secret"
    assert kwargs["token_url"].endswith("/oidc/v1/token")
    assert kwargs["use_header"] is True
    assert kwargs["endpoint_params"]["resource"].endswith(f"/{_WORKSPACE_ID}/zerobusDirectWriteApi")
    auth_details = json.loads(token_source._mlflow_authorization_details)
    assert any(entry["object_full_path"] == _TABLE_NAME for entry in auth_details)


def test_collector_token_mint_uses_bounded_http_timeout():
    response = mock.MagicMock()
    response.ok = True
    response.json.return_value = {
        "access_token": "token",
        "expires_in": 3600,
        "token_type": "Bearer",
    }
    with mock.patch(f"{_MODULE}.requests.post", return_value=response) as mock_post:
        token_source = build_databricks_otel_collector_token_source(
            host=_HOST,
            client_id="my-client-id",
            client_secret="my-client-secret",
            workspace_id=_WORKSPACE_ID,
            tables=[_TABLE_NAME],
        )
        token = token_source.token()
        token_source.refresh()

    assert token.access_token == "token"
    assert mock_post.call_count == 2
    assert all(
        call.kwargs["timeout"] == _TOKEN_REQUEST_TIMEOUT_SECONDS
        for call in mock_post.call_args_list
    )
    request_data = mock_post.call_args.kwargs["data"]
    assert request_data["grant_type"] == "client_credentials"
    assert request_data["scope"] == "all-apis"
    assert request_data["resource"].endswith(f"/{_WORKSPACE_ID}/zerobusDirectWriteApi")
    assert "authorization_details" in request_data
    assert "params" not in mock_post.call_args.kwargs


def _make_local_creds_config(
    *,
    host=_HOST,
    client_id="my-client-id",
    client_secret="my-client-secret",
    auth_type=None,
    token=None,
    username=None,
    password=None,
):
    return SimpleNamespace(
        host=host,
        client_id=client_id,
        client_secret=client_secret,
        auth_type=auth_type,
        token=token,
        username=username,
        password=password,
    )


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"token": "pat"},
        {"username": "user", "password": "password"},
        {"auth_type": "azure-cli"},
    ],
    ids=["pat", "basic", "explicit-alternate-auth"],
)
def test_resolve_collector_credentials_skips_alternate_auth_without_network(
    monkeypatch, config_kwargs
):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "true")
    config = _make_local_creds_config(**config_kwargs)
    with (
        mock.patch(f"{_MODULE}._get_databricks_creds_config", return_value=config),
        mock.patch(f"{_MODULE}.requests.get") as mock_get,
    ):
        result = _resolve_collector_credentials("databricks")

    assert result is None
    mock_get.assert_not_called()


def test_resolve_collector_credentials_uses_matching_workspace_id_without_network(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "true")
    monkeypatch.setenv("DATABRICKS_HOST", _HOST)
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", _WORKSPACE_ID)
    config = _make_local_creds_config()
    with (
        mock.patch(f"{_MODULE}._get_databricks_creds_config", return_value=config),
        mock.patch(f"{_MODULE}.requests.get") as mock_get,
    ):
        result = _resolve_collector_credentials("databricks")

    assert result == (_HOST, _WORKSPACE_ID, "my-client-id", "my-client-secret")
    mock_get.assert_not_called()


@pytest.mark.parametrize(
    ("workspace_host", "workspace_id"),
    [(None, None), ("https://other-workspace.azuredatabricks.net", "99999999")],
    ids=["missing", "host-mismatch"],
)
def test_resolve_collector_credentials_discovers_workspace_id_with_bounded_request(
    monkeypatch, workspace_host, workspace_id
):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "true")
    if workspace_host is None:
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_HOST", workspace_host)
    if workspace_id is None:
        monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", workspace_id)
    config = _make_local_creds_config()
    response = _make_json_response({"workspace_id": _WORKSPACE_ID})
    with (
        mock.patch(f"{_MODULE}._get_databricks_creds_config", return_value=config),
        mock.patch(f"{_MODULE}.requests.get", return_value=response) as mock_get,
    ):
        result = _resolve_collector_credentials("databricks")

    assert result == (_HOST, _WORKSPACE_ID, "my-client-id", "my-client-secret")
    mock_get.assert_called_once_with(
        f"{_HOST}/.well-known/databricks-config",
        timeout=_SDK_HTTP_TIMEOUT_SECONDS,
    )


def _make_token(access_token: str):
    token = mock.MagicMock()
    token.access_token = access_token
    token.token_type = "Bearer"
    return token


def _make_mock_token_source(access_token="test-access-token"):
    token_source = mock.MagicMock()
    token_source.token.return_value = _make_token(access_token)
    token_source.refresh.return_value = _make_token(access_token)
    return token_source


def _make_response(status_code: int, content: bytes = b""):
    response = mock.MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.ok = status_code < 400
    response.content = content
    return response


def _make_client(token_source=None, endpoint=_ENDPOINT):
    client = DatabricksOTelClient(
        tracking_uri="databricks",
        token_source=token_source or _make_mock_token_source(),
        table_name=_TABLE_NAME,
        host=_HOST,
        workspace_id=_WORKSPACE_ID,
        endpoint=endpoint,
    )
    client._session = mock.MagicMock()
    client._session.post.return_value = _make_response(200)
    return client


def test_databricks_otel_client_constructor_is_network_free():
    with mock.patch(f"{_MODULE}.requests.get") as mock_get:
        client = DatabricksOTelClient(
            tracking_uri="databricks",
            token_source=_make_mock_token_source(),
            table_name=_TABLE_NAME,
        )

    mock_get.assert_not_called()
    assert client.config_warned is False
    assert client.host is None
    assert client.workspace_id == ""
    client.close()


def test_databricks_otel_client_wraps_pre_send_token_failure():
    token_source = _make_mock_token_source()
    token_source.token.side_effect = requests.Timeout("token mint timed out")
    client = _make_client(token_source)

    assert client.ensure_ready()
    with pytest.raises(ZerobusOtelTokenError, match="Minting") as exc_info:
        client.post(b"payload")

    assert isinstance(exc_info.value.__cause__, requests.Timeout)
    client._session.post.assert_not_called()


def test_databricks_otel_client_does_not_replace_injected_workspace_credentials(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    client = DatabricksOTelClient(
        tracking_uri="databricks",
        token_source=_make_mock_token_source(),
        table_name=_TABLE_NAME,
        host=_HOST,
        workspace_id=_WORKSPACE_ID,
    )

    with (
        mock.patch(f"{_MODULE}._resolve_collector_credentials") as mock_credentials,
        mock.patch(
            f"{_MODULE}._resolve_collector_endpoint_from_metastore", return_value=None
        ) as mock_endpoint,
    ):
        assert not client.ensure_ready()

    mock_credentials.assert_not_called()
    mock_endpoint.assert_called_once_with(
        host=_HOST,
        workspace_id=_WORKSPACE_ID,
        client_id=None,
        client_secret=None,
    )
    assert client.config_warned


def test_databricks_otel_client_explicit_endpoint_override_takes_precedence(monkeypatch):
    explicit_endpoint = f"https://{_ENDPOINT}/"
    monkeypatch.setenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", "badhost.example.com")
    client = _make_client(endpoint=explicit_endpoint)

    with mock.patch(f"{_MODULE}._resolve_collector_endpoint_from_metastore") as mock_resolve:
        assert client.ensure_ready()

    mock_resolve.assert_not_called()
    assert client._collector_url == f"https://{_ENDPOINT}/v1/traces"


def test_databricks_otel_client_resolves_endpoint_lazily_once_per_workspace(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    with mock.patch(
        f"{_MODULE}._resolve_collector_endpoint_from_metastore", return_value=_ENDPOINT
    ) as mock_resolve:
        client1 = _make_client(endpoint=None)
        client2 = _make_client(endpoint=None)
        assert client1.ensure_ready()
        assert client1.ensure_ready()
        assert client2.ensure_ready()

    mock_resolve.assert_called_once_with(
        host=_HOST,
        workspace_id=_WORKSPACE_ID,
        client_id=None,
        client_secret=None,
    )
    assert client1._collector_url == f"https://{_ENDPOINT}/v1/traces"
    assert client2._collector_url == f"https://{_ENDPOINT}/v1/traces"


def test_databricks_otel_client_posts_expected_headers_and_timeout():
    client = _make_client()
    assert client.ensure_ready()
    client.post(b"payload")

    client._session.post.assert_called_once_with(
        f"https://{_ENDPOINT}/v1/traces",
        data=b"payload",
        headers={
            "Authorization": "Bearer test-access-token",
            "Content-Type": "application/x-protobuf",
            "x-databricks-zerobus-table-name": _TABLE_NAME,
        },
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


def test_databricks_otel_client_retries_once_on_401_and_caches_refresh():
    token_source = ClientCredentials(
        client_id="cid",
        client_secret="csecret",
        token_url=f"{_HOST}/oidc/v1/token",
        scopes="all-apis",
        use_header=True,
    )
    client = _make_client(token_source)
    client._session.post.side_effect = [
        _make_response(401),
        _make_response(200),
        _make_response(200),
    ]
    assert client.ensure_ready()

    with mock.patch(
        "databricks.sdk.oauth.retrieve_token",
        side_effect=[Token(access_token="token-1"), Token(access_token="token-2")],
    ) as mock_retrieve:
        client.post(b"payload")
        client.post(b"payload")

    assert client._session.post.call_count == 3
    assert [
        call.kwargs["headers"]["Authorization"] for call in client._session.post.call_args_list
    ] == ["Bearer token-1", "Bearer token-2", "Bearer token-2"]
    assert mock_retrieve.call_count == 2


class _LegacyRefreshableTokenSource:
    def __init__(self, token):
        self._lock = threading.Lock()
        self._token = token
        self.refresh_count = 0

    def token(self):
        return self._token

    def refresh(self):
        self.refresh_count += 1
        return _make_token(f"token-refreshed-{self.refresh_count}")


class _CachelessTokenSource:
    def __init__(self, tokens):
        self._tokens = iter(tokens)

    def token(self):
        return next(self._tokens)

    def refresh(self):
        return next(self._tokens)


def test_databricks_otel_client_retries_401_with_legacy_token_source():
    token_source = _LegacyRefreshableTokenSource(_make_token("token-initial"))
    client = _make_client(token_source)
    client._session.post.side_effect = [
        _make_response(401),
        _make_response(200),
        _make_response(200),
    ]
    assert client.ensure_ready()

    client.post(b"payload")
    client.post(b"payload")

    assert client._session.post.call_count == 3
    assert [
        call.kwargs["headers"]["Authorization"] for call in client._session.post.call_args_list
    ] == ["Bearer token-initial", "Bearer token-refreshed-1", "Bearer token-refreshed-1"]
    assert token_source.refresh_count == 1


def test_databricks_otel_client_retries_401_with_cacheless_token_source():
    token_source = _CachelessTokenSource([_make_token("token-a"), _make_token("token-b")])
    client = _make_client(token_source)
    client._session.post.side_effect = [_make_response(401), _make_response(200)]
    assert client.ensure_ready()

    client.post(b"payload")

    assert client._session.post.call_count == 2
    assert [
        call.kwargs["headers"]["Authorization"] for call in client._session.post.call_args_list
    ] == ["Bearer token-a", "Bearer token-b"]


def test_databricks_otel_client_returns_second_401_after_one_refresh():
    token_source = _make_mock_token_source()
    client = _make_client(token_source)
    response = _make_response(401, b"denied")
    client._session.post.side_effect = [response, response]
    assert client.ensure_ready()

    assert client.post(b"payload") is response
    token_source.refresh.assert_called_once()
    assert client._session.post.call_count == 2


def test_databricks_otel_client_propagates_refresh_failure():
    token_source = _make_mock_token_source()
    token_source.refresh.side_effect = requests.Timeout("token refresh timed out")
    client = _make_client(token_source)
    client._session.post.return_value = _make_response(401)
    assert client.ensure_ready()

    with pytest.raises(ZerobusOtelTokenRefreshError, match="Refreshing") as exc_info:
        client.post(b"payload")

    assert isinstance(exc_info.value.__cause__, requests.Timeout)
    assert client._session.post.call_count == 1


def _make_transport(token_source=None, endpoint=_ENDPOINT):
    transport = DatabricksOtelCollectorSpanExporter(
        tracking_uri="databricks",
        token_source=token_source or _make_mock_token_source(),
        table_name=_TABLE_NAME,
        host=_HOST,
        workspace_id=_WORKSPACE_ID,
        endpoint=endpoint,
    )
    transport._client._session = mock.MagicMock()
    transport._client._session.post.return_value = _make_response(200)
    return transport


def test_collector_transport_sends_serialized_mlflow_spans():
    transport = _make_transport()
    span = Span(create_mock_otel_span(trace_id=1, span_id=2))

    response = transport.send_batch([span])

    assert response is transport._client._session.post.return_value
    transport._client._session.post.assert_called_once()
    request = ExportTraceServiceRequest.FromString(
        transport._client._session.post.call_args.kwargs["data"]
    )
    assert len(request.resource_spans) == 1
    assert len(request.resource_spans[0].scope_spans[0].spans) == 1
    assert request.resource_spans[0].scope_spans[0].spans[0].name == "test_span"


@pytest.mark.parametrize("failure", ["build", "serialize"])
def test_collector_transport_wraps_serialization_failures(failure):
    transport = _make_transport()
    span = Span(create_mock_otel_span(trace_id=7, span_id=8))

    if failure == "build":
        patcher = mock.patch(
            "mlflow.tracing.export.databricks_otel_collector.build_otlp_export_request",
            side_effect=ValueError("invalid span"),
        )
    else:
        request = mock.MagicMock()
        request.SerializeToString.side_effect = ValueError("invalid protobuf")
        patcher = mock.patch(
            "mlflow.tracing.export.databricks_otel_collector.build_otlp_export_request",
            return_value=request,
        )

    with (
        patcher,
        pytest.raises(
            DatabricksOtelSerializationError, match="Failed to (build|serialize)"
        ) as exc_info,
    ):
        transport.send_batch([span])

    assert isinstance(exc_info.value.__cause__, ValueError)
    transport._client._session.post.assert_not_called()


def test_collector_transport_returns_none_when_client_unavailable():
    transport = _make_transport()
    transport._client.ensure_ready = mock.MagicMock(return_value=False)
    transport._client.post = mock.MagicMock()

    assert transport.send_batch([]) is None
    transport._client.post.assert_not_called()


def test_collector_transport_propagates_network_errors():
    transport = _make_transport()
    error = requests.ConnectionError("collector unavailable")
    transport._client.post = mock.MagicMock(side_effect=error)
    span = Span(create_mock_otel_span(trace_id=3, span_id=4))

    with pytest.raises(requests.ConnectionError, match="collector unavailable") as exc_info:
        transport.send_batch([span])

    assert exc_info.value is error


def test_collector_transport_returns_http_response_untouched():
    transport = _make_transport()
    response = _make_response(503, b"server error")
    transport._client.post = mock.MagicMock(return_value=response)
    span = Span(create_mock_otel_span(trace_id=5, span_id=6))

    assert transport.send_batch([span]) is response


def test_collector_transport_forwards_client_properties_and_close():
    transport = _make_transport()
    transport._client._config_warned = True
    transport._client._host = "https://example.databricks.com"
    transport._client._workspace_id = "workspace"

    assert transport.config_warned is True
    assert transport.host == "https://example.databricks.com"
    assert transport.workspace_id == "workspace"

    transport.close()
    transport._client._session.close.assert_called_once()
