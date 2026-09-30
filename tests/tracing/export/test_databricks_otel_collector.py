import contextlib
import json
import threading
from unittest import mock

import pytest
import requests
import urllib3.exceptions
from databricks.sdk.oauth import ClientCredentials, Token
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from mlflow.entities.span import Span
from mlflow.entities.trace_location import UCSchemaLocation, UnityCatalog
from mlflow.environment_variables import MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT
from mlflow.tracing.export.databricks_otel_collector import (
    _REQUEST_TIMEOUT_SECONDS,
    _SDK_HTTP_TIMEOUT_SECONDS,
    _SDK_RETRY_TIMEOUT_SECONDS,
    DatabricksOtelCollectorSpanExporter,
    _get_table_name_from_destination,
    _is_connection_not_established,
    _resolved_endpoints,
    build_databricks_otel_collector_token_source,
    build_table_authorization_details,
    get_databricks_otel_collector_span_exporter,
    is_databricks_otel_collector_host,
    resolve_databricks_otel_collector_endpoint,
)
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter

from tests.tracing.helper import create_mock_otel_span

_MODULE = "mlflow.tracing.export.databricks_otel_collector"

# ---------------------------------------------------------------------------
# is_databricks_otel_collector_host
# ---------------------------------------------------------------------------

_WORKSPACE_ID = "12345678"


@pytest.mark.parametrize(
    ("endpoint", "workspace_id", "expected"),
    [
        # Valid AWS host
        (
            "12345678.zerobus.us-west-2.cloud.databricks.com",
            "12345678",
            True,
        ),
        # Valid Azure host
        (
            "12345678.zerobus.eastus.azuredatabricks.net",
            "12345678",
            True,
        ),
        # Valid GCP host
        (
            "12345678.zerobus.us-central1.gcp.databricks.com",
            "12345678",
            True,
        ),
        # With leading https:// prefix (should be stripped)
        (
            "https://12345678.zerobus.us-west-2.cloud.databricks.com",
            "12345678",
            True,
        ),
        # Staging segment
        (
            "12345678.zerobus.us-west-2.staging.cloud.databricks.com",
            "12345678",
            True,
        ),
        # Wrong workspace ID
        (
            "99999999.zerobus.us-west-2.cloud.databricks.com",
            "12345678",
            False,
        ),
        # Contains port (colon)
        (
            "12345678.zerobus.us-west-2.cloud.databricks.com:443",
            "12345678",
            False,
        ),
        # Contains path
        (
            "12345678.zerobus.us-west-2.cloud.databricks.com/v1/traces",
            "12345678",
            False,
        ),
        # Contains userinfo (@)
        (
            "user@12345678.zerobus.us-west-2.cloud.databricks.com",
            "12345678",
            False,
        ),
        # Wrong suffix
        (
            "12345678.zerobus.us-west-2.example.com",
            "12345678",
            False,
        ),
        # Missing region (empty region between the host segment and suffix)
        (
            "12345678.zerobus.cloud.databricks.com",
            "12345678",
            False,
        ),
        # Trailing dot in region part (malformed)
        (
            "12345678.zerobus..cloud.databricks.com",
            "12345678",
            False,
        ),
    ],
)
def test_is_databricks_otel_collector_host(endpoint: str, workspace_id: str, expected: bool):
    assert is_databricks_otel_collector_host(endpoint, workspace_id) == expected


# ---------------------------------------------------------------------------
# _is_connection_not_established
# ---------------------------------------------------------------------------


def _connect_timeout_error():
    return requests.ConnectTimeout("connect timed out")


def _new_connection_error():
    return requests.ConnectionError(
        urllib3.exceptions.NewConnectionError(None, "Failed to establish a new connection")
    )


def _name_resolution_connection_error():
    # NameResolutionError (DNS failure) subclasses NewConnectionError.
    return requests.ConnectionError(
        urllib3.exceptions.NameResolutionError(
            "adb-12345678.azuredatabricks.net", None, "Name or service not known"
        )
    )


def _context_chained_new_connection_error():
    # Some wrappers drop the urllib3 error from the args but keep it in the
    # exception context chain.
    error = requests.ConnectionError("wrapped connection failure")
    error.__context__ = urllib3.exceptions.NewConnectionError(
        None, "Failed to establish a new connection"
    )
    return error


def _connection_reset_error():
    # A reset after the request was sent: the connection was established and the
    # request may have been delivered, so delivery is ambiguous.
    return requests.ConnectionError(urllib3.exceptions.ProtocolError("Connection reset by peer"))


def _plain_connection_error():
    return requests.ConnectionError("connection refused")


def _read_timeout_error():
    return requests.ReadTimeout("read timed out")


def _generic_timeout_error():
    return requests.Timeout("timed out")


@pytest.mark.parametrize(
    ("error_factory", "expected"),
    [
        (_connect_timeout_error, True),
        (_new_connection_error, True),
        (_name_resolution_connection_error, True),
        (_context_chained_new_connection_error, True),
        (_connection_reset_error, False),
        (_plain_connection_error, False),
        (_read_timeout_error, False),
        (_generic_timeout_error, False),
        (lambda: RuntimeError("boom"), False),
    ],
)
def test_is_connection_not_established(error_factory, expected: bool):
    assert _is_connection_not_established(error_factory()) is expected


# ---------------------------------------------------------------------------
# resolve_databricks_otel_collector_endpoint
# ---------------------------------------------------------------------------


def _make_summary(region="us-west-2", cloud="aws", global_metastore_id=None):
    summary = mock.MagicMock()
    summary.region = region
    summary.cloud = cloud
    summary.global_metastore_id = global_metastore_id
    return summary


@contextlib.contextmanager
def _mock_workspace_client(summary):
    """Patch the SDK client classes used for the metastore summary lookup.

    ``Config`` must be patched as well: constructing the real one would probe
    the (fake) workspace host's well-known metadata endpoint.
    """
    ws = mock.MagicMock(metastores=mock.MagicMock(summary=mock.MagicMock(return_value=summary)))
    with (
        mock.patch("databricks.sdk.core.Config") as mock_config,
        mock.patch("databricks.sdk.WorkspaceClient", return_value=ws) as mock_ws_class,
    ):
        yield mock_config, mock_ws_class, ws


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
    with _mock_workspace_client(_make_summary(region, cloud)) as (mock_config, mock_ws_class, ws):
        result = resolve_databricks_otel_collector_endpoint(
            host=host,
            workspace_id="12345678",
        )

    assert result == expected_endpoint
    mock_config.assert_called_once()
    mock_ws_class.assert_called_once()
    ws.metastores.summary.assert_called_once_with()
    # The one-shot lookup client is released after use.
    ws.close.assert_called_once()


def test_resolve_endpoint_builds_bounded_client_with_sp_credentials(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    with (
        mock.patch("databricks.sdk.core.Config") as mock_config,
        mock.patch("databricks.sdk.WorkspaceClient") as mock_ws_class,
    ):
        mock_ws_class.return_value.metastores.summary.return_value = _make_summary(
            "us-west-2", "aws"
        )
        resolve_databricks_otel_collector_endpoint(
            host="https://adb-12345678.cloud.databricks.com",
            workspace_id="12345678",
            client_id="sp-client-id",
            client_secret="sp-secret",
        )

    # The resolution client runs with an explicitly bounded timeout budget instead
    # of the SDK defaults (60s per request, 300s retry budget).
    mock_config.assert_called_once_with(
        host="https://adb-12345678.cloud.databricks.com",
        client_id="sp-client-id",
        client_secret="sp-secret",
        workspace_id="12345678",
        http_timeout_seconds=_SDK_HTTP_TIMEOUT_SECONDS,
        retry_timeout_seconds=_SDK_RETRY_TIMEOUT_SECONDS,
    )
    mock_ws_class.assert_called_once_with(config=mock_config.return_value)
    mock_ws_class.return_value.close.assert_called_once()


def test_resolve_endpoint_override_takes_precedence(monkeypatch):
    valid_override = "12345678.zerobus.eu-west-1.cloud.databricks.com"
    monkeypatch.setenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", valid_override)
    with _mock_workspace_client(_make_summary("us-west-2", "aws")) as (_, mock_ws_class, _):
        result = resolve_databricks_otel_collector_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result == valid_override
    mock_ws_class.assert_not_called()


def test_resolve_endpoint_override_invalid_returns_none(monkeypatch):
    monkeypatch.setenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", "badhost.example.com")
    with _mock_workspace_client(_make_summary("us-west-2", "aws")) as (_, mock_ws_class, _):
        result = resolve_databricks_otel_collector_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result is None
    mock_ws_class.assert_not_called()


def test_resolve_endpoint_metastore_error_returns_none(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    with (
        mock.patch("databricks.sdk.core.Config") as mock_config,
        mock.patch(
            "databricks.sdk.WorkspaceClient", side_effect=RuntimeError("connection refused")
        ) as mock_ws_class,
    ):
        result = resolve_databricks_otel_collector_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result is None
    mock_config.assert_called_once()
    mock_ws_class.assert_called_once()


def test_resolve_endpoint_cloud_from_global_metastore_id(monkeypatch):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    summary = _make_summary(region="us-west-2", cloud=None, global_metastore_id="aws:us-west-2:abc")
    summary.cloud = None
    with _mock_workspace_client(summary) as (_, _, ws):
        result = resolve_databricks_otel_collector_endpoint(
            host="https://adb-12345678.cloud.databricks.com",
            workspace_id="12345678",
        )
    assert result == "12345678.zerobus.us-west-2.cloud.databricks.com"
    ws.metastores.summary.assert_called_once_with()


# ---------------------------------------------------------------------------
# build_table_authorization_details
# ---------------------------------------------------------------------------


def test_build_table_authorization_details_structure():
    result = build_table_authorization_details(["mycat.myschema.mytable"])
    entries = json.loads(result)
    assert len(entries) == 3

    types = {e["object_type"] for e in entries}
    assert types == {"CATALOG", "SCHEMA", "TABLE"}

    catalog_entry = next(e for e in entries if e["object_type"] == "CATALOG")
    assert catalog_entry["object_full_path"] == "mycat"
    assert catalog_entry["privileges"] == ["USE CATALOG"]
    assert catalog_entry["type"] == "unity_catalog_privileges"

    schema_entry = next(e for e in entries if e["object_type"] == "SCHEMA")
    assert schema_entry["object_full_path"] == "mycat.myschema"
    assert schema_entry["privileges"] == ["USE SCHEMA"]

    table_entry = next(e for e in entries if e["object_type"] == "TABLE")
    assert table_entry["object_full_path"] == "mycat.myschema.mytable"
    assert "SELECT" in table_entry["privileges"]
    assert "MODIFY" in table_entry["privileges"]


def test_build_table_authorization_details_multiple_tables():
    result = build_table_authorization_details(["cat1.sch1.tbl1", "cat2.sch2.tbl2"])
    entries = json.loads(result)
    assert len(entries) == 6  # 3 per table


def test_build_table_authorization_details_malformed_table_skipped():
    result = build_table_authorization_details(["badtable"])
    entries = json.loads(result)
    assert entries == []


# ---------------------------------------------------------------------------
# build_databricks_otel_collector_token_source
# ---------------------------------------------------------------------------


def test_build_databricks_otel_collector_token_source_construction():
    with mock.patch(
        "databricks.sdk.oauth.ClientCredentials",
        autospec=True,
    ) as mock_cc:
        build_databricks_otel_collector_token_source(
            host="https://adb-12345678.azuredatabricks.net",
            client_id="my-client-id",
            client_secret="my-client-secret",
            workspace_id="12345678",
            tables=["cat.sch.tbl"],
        )

    mock_cc.assert_called_once()
    _, kwargs = mock_cc.call_args
    assert kwargs["client_id"] == "my-client-id"
    assert kwargs["client_secret"] == "my-client-secret"
    assert kwargs["token_url"].endswith("/oidc/v1/token")
    assert kwargs["scopes"] == "all-apis"
    assert kwargs["use_header"] is True
    assert "zerobusDirectWriteApi" in kwargs["endpoint_params"]["resource"]
    assert "12345678" in kwargs["endpoint_params"]["resource"]
    # authorization_details should be a JSON string containing the table
    auth_details = json.loads(kwargs["authorization_details"])
    assert any(e["object_full_path"] == "cat.sch.tbl" for e in auth_details)


# ---------------------------------------------------------------------------
# get_databricks_otel_collector_span_exporter
# ---------------------------------------------------------------------------


def _make_host_creds(
    client_id="cid", client_secret="csecret", host="https://host.com", workspace_id="ws123"
):
    creds = mock.MagicMock()
    creds.client_id = client_id
    creds.client_secret = client_secret
    creds.host = host
    creds.workspace_id = workspace_id
    return creds


def _make_uc_destination(table_name="cat.sch.tbl"):
    destination = UnityCatalog(catalog_name="cat", schema_name="sch", table_prefix="pfx")
    destination._otel_spans_table_name = table_name
    return destination


def test_get_exporter_disabled_by_env_var_returns_none(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "false")
    with mock.patch(f"{_MODULE}.get_databricks_host_creds") as mock_creds:
        result = get_databricks_otel_collector_span_exporter(_make_uc_destination(), "databricks")
    assert result is None
    mock_creds.assert_not_called()


def test_get_exporter_disabled_by_env_var_logs_no_warning(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "false")
    with (
        mock.patch(f"{_MODULE}.get_databricks_host_creds"),
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        result = get_databricks_otel_collector_span_exporter(_make_uc_destination(), "databricks")
    assert result is None
    mock_log.warning.assert_not_called()


def test_get_exporter_enabled_by_default_returns_exporter(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    mock_token_source = mock.MagicMock()
    with (
        mock.patch(
            f"{_MODULE}.get_databricks_host_creds",
            return_value=_make_host_creds(workspace_id="ws123"),
        ),
        mock.patch(f"{_MODULE}.resolve_databricks_otel_collector_endpoint") as mock_resolve,
        mock.patch(
            f"{_MODULE}.build_databricks_otel_collector_token_source",
            return_value=mock_token_source,
        ),
        mock.patch.object(
            DatabricksOtelCollectorSpanExporter, "__init__", return_value=None
        ) as mock_init,
    ):
        result = get_databricks_otel_collector_span_exporter(
            _make_uc_destination("cat.sch.tbl"), "databricks"
        )

    mock_init.assert_called_once()
    _, kwargs = mock_init.call_args
    assert kwargs["endpoint"] is None  # resolved lazily on the first export
    assert kwargs["token_source"] is mock_token_source
    assert kwargs["table_name"] == "cat.sch.tbl"
    assert kwargs["host"] == "https://host.com"
    assert kwargs["workspace_id"] == "ws123"
    assert kwargs["client_id"] == "cid"
    assert kwargs["client_secret"] == "csecret"
    assert isinstance(result, DatabricksOtelCollectorSpanExporter)
    # Tracer initialization performs no network I/O for endpoint resolution.
    mock_resolve.assert_not_called()


def test_get_exporter_valid_endpoint_override_is_passed_to_exporter(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)
    monkeypatch.setenv(
        "MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", "ws123.zerobus.us-west-2.cloud.databricks.com"
    )
    with (
        mock.patch(
            f"{_MODULE}.get_databricks_host_creds",
            return_value=_make_host_creds(workspace_id="ws123"),
        ),
        mock.patch(f"{_MODULE}.resolve_databricks_otel_collector_endpoint") as mock_resolve,
        mock.patch(f"{_MODULE}.build_databricks_otel_collector_token_source"),
        mock.patch.object(
            DatabricksOtelCollectorSpanExporter, "__init__", return_value=None
        ) as mock_init,
    ):
        get_databricks_otel_collector_span_exporter(_make_uc_destination(), "databricks")

    _, kwargs = mock_init.call_args
    assert kwargs["endpoint"] == "ws123.zerobus.us-west-2.cloud.databricks.com"
    # The override is validated locally; no resolution network I/O happens.
    mock_resolve.assert_not_called()


def test_get_exporter_invalid_endpoint_override_returns_none(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)
    monkeypatch.setenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", "badhost.example.com")
    with (
        mock.patch(
            f"{_MODULE}.get_databricks_host_creds",
            return_value=_make_host_creds(),
        ),
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        result = get_databricks_otel_collector_span_exporter(_make_uc_destination(), "databricks")
    assert result is None
    mock_log.debug.assert_called_once()
    mock_log.warning.assert_not_called()


def test_get_exporter_uc_schema_location_returns_none(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)
    destination = UCSchemaLocation(catalog_name="cat", schema_name="sch")
    with mock.patch(f"{_MODULE}.get_databricks_host_creds") as mock_creds:
        result = get_databricks_otel_collector_span_exporter(destination, "databricks")
    assert result is None
    mock_creds.assert_not_called()


def test_get_exporter_no_sp_creds_returns_none_with_debug_when_default(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)
    with (
        mock.patch(
            f"{_MODULE}.get_databricks_host_creds",
            return_value=_make_host_creds(client_id=None, client_secret=None),
        ),
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        result = get_databricks_otel_collector_span_exporter(_make_uc_destination(), "databricks")
    assert result is None
    mock_log.debug.assert_called_once()
    mock_log.warning.assert_not_called()


def test_get_exporter_no_sp_creds_returns_none_with_warning_when_explicitly_set(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "true")
    with (
        mock.patch(
            f"{_MODULE}.get_databricks_host_creds",
            return_value=_make_host_creds(client_id=None, client_secret=None),
        ),
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        result = get_databricks_otel_collector_span_exporter(_make_uc_destination(), "databricks")
    assert result is None
    mock_log.warning.assert_called_once()
    mock_log.debug.assert_not_called()


def test_get_exporter_creds_exception_returns_none(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "true")
    with mock.patch(
        f"{_MODULE}.get_databricks_host_creds",
        side_effect=RuntimeError("auth error"),
    ):
        result = get_databricks_otel_collector_span_exporter(_make_uc_destination(), "databricks")
    assert result is None


def test_get_exporter_missing_table_name_returns_none(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    with mock.patch(
        f"{_MODULE}.get_databricks_host_creds",
        return_value=_make_host_creds(),
    ):
        result = get_databricks_otel_collector_span_exporter(
            _make_uc_destination(table_name=None), "databricks"
        )
    assert result is None


# ---------------------------------------------------------------------------
# DatabricksOtelCollectorSpanExporter - lazy endpoint resolution
# ---------------------------------------------------------------------------

_TABLE_NAME = "cat.sch.tbl"
_ENDPOINT = "ws123.zerobus.us-west-2.cloud.databricks.com"
_HOST = "https://adb-12345678.azuredatabricks.net"
_WORKSPACE_ID = "12345678"


@pytest.fixture
def clear_resolved_endpoints():
    _resolved_endpoints.clear()
    yield
    _resolved_endpoints.clear()


def _make_token(access_token: str):
    token = mock.MagicMock()
    token.access_token = access_token
    return token


def _make_mock_token_source(access_token="test-access-token"):
    token_source = mock.MagicMock()
    token_source.token.return_value = _make_token(access_token)
    token_source.refresh.return_value = _make_token(access_token)
    return token_source


class _LegacyRefreshableTokenSource:
    """Mimics older databricks-sdk ``Refreshable`` versions (pre-``_update_token``):
    the cache is a plain ``_token`` attribute guarded by ``_lock``.
    """

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
    """Token source with no cache attributes at all (e.g. a custom TokenSource)."""

    def __init__(self, tokens):
        self._tokens = iter(tokens)

    def token(self):
        return next(self._tokens)

    def refresh(self):
        return next(self._tokens)


def _make_response(status_code: int, content: bytes = b""):
    response = mock.MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.ok = status_code < 400
    response.content = content
    return response


def _make_exporter(
    monkeypatch,
    table_name=_TABLE_NAME,
    endpoint=_ENDPOINT,
    host=_HOST,
    workspace_id=_WORKSPACE_ID,
    client_id="cid",
    client_secret="csecret",
    token_source=None,
    sync_rest=True,
):
    if token_source is None:
        token_source = _make_mock_token_source()

    if sync_rest:
        # Route the REST fallback synchronously through ``_log_spans`` so fallback
        # assertions are deterministic (no batcher worker / async queue involved).
        monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

    exporter = DatabricksOtelCollectorSpanExporter(
        tracking_uri="databricks",
        token_source=token_source,
        table_name=table_name,
        host=host,
        workspace_id=workspace_id,
        client_id=client_id,
        client_secret=client_secret,
        endpoint=endpoint,
    )
    session = mock.MagicMock()
    session.post.return_value = _make_response(200)
    exporter._session = session
    return exporter, session, token_source


def _posted_auth_headers(session):
    return [call.kwargs["headers"]["Authorization"] for call in session.post.call_args_list]


def test_exporter_resolves_endpoint_lazily_once_per_workspace(
    monkeypatch, clear_resolved_endpoints
):
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    otel_span = create_mock_otel_span(trace_id=20, span_id=20)
    with mock.patch(
        f"{_MODULE}.resolve_databricks_otel_collector_endpoint",
        return_value=_ENDPOINT,
    ) as mock_resolve:
        exporter1, session1, _ = _make_exporter(monkeypatch, endpoint=None)
        exporter2, session2, _ = _make_exporter(monkeypatch, endpoint=None)

        exporter1._export_spans_incrementally([otel_span])
        exporter1._export_spans_incrementally([otel_span])
        exporter2._export_spans_incrementally([otel_span])

    # Resolved once (on the first export) and memoized on the exporter and in the
    # module-level cache shared by the second exporter (same host + workspace).
    mock_resolve.assert_called_once()
    _, kwargs = mock_resolve.call_args
    assert kwargs == {
        "host": _HOST,
        "workspace_id": _WORKSPACE_ID,
        "client_id": "cid",
        "client_secret": "csecret",
    }
    assert session1.post.call_count == 2
    assert session2.post.call_count == 1
    assert exporter1._collector_url == f"https://{_ENDPOINT}/v1/traces"
    assert exporter2._collector_url == f"https://{_ENDPOINT}/v1/traces"


def test_exporter_resolution_failure_replays_via_rest_and_sticks_with_debug_by_default(
    monkeypatch, clear_resolved_endpoints
):
    monkeypatch.delenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, raising=False)
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    exporter, session, _ = _make_exporter(monkeypatch, endpoint=None)
    otel_span = create_mock_otel_span(trace_id=21, span_id=21)

    with (
        mock.patch(
            f"{_MODULE}.resolve_databricks_otel_collector_endpoint", return_value=None
        ) as mock_resolve,
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])
        exporter._export_spans_incrementally([otel_span])

    # Resolution is attempted once; both batches go through the REST path.
    mock_resolve.assert_called_once()
    session.post.assert_not_called()
    assert mock_log_spans.call_count == 2
    mock_log.warning.assert_not_called()
    mock_log.debug.assert_called_once()


def test_exporter_resolution_failure_logs_warning_when_export_explicitly_enabled(
    monkeypatch, clear_resolved_endpoints
):
    monkeypatch.setenv(MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.name, "true")
    monkeypatch.delenv("MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT", raising=False)
    exporter, session, _ = _make_exporter(monkeypatch, endpoint=None)
    otel_span = create_mock_otel_span(trace_id=22, span_id=22)

    with (
        mock.patch(f"{_MODULE}.resolve_databricks_otel_collector_endpoint", return_value=None),
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])

    session.post.assert_not_called()
    mock_log_spans.assert_called_once()
    mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# DatabricksOtelCollectorSpanExporter - export behavior
# ---------------------------------------------------------------------------


def test_exporter_posts_otlp_request_with_decoded_attributes(monkeypatch):
    exporter, session, token_source = _make_exporter(monkeypatch)
    otel_span = create_mock_otel_span(trace_id=1, span_id=1)
    # MLflow stores span attributes JSON-encoded on the raw OTel span; the posted
    # payload must contain the decoded values without the JSON quotes.
    otel_span.set_attribute("mlflow.spanType", '"UNKNOWN"')
    otel_span.set_attribute("mlflow.spanFunctionName", '"test_span"')

    exporter._export_spans_incrementally([otel_span])

    token_source.token.assert_called_once()
    session.post.assert_called_once()
    request = ExportTraceServiceRequest.FromString(session.post.call_args.kwargs["data"])
    assert len(request.resource_spans) == 1
    assert len(request.resource_spans[0].scope_spans) == 1
    pb_span = request.resource_spans[0].scope_spans[0].spans[0]
    assert pb_span.name == "test_span"
    attrs = {attr.key: attr.value for attr in pb_span.attributes}
    assert attrs["mlflow.spanType"].WhichOneof("value") == "string_value"
    assert attrs["mlflow.spanType"].string_value == "UNKNOWN"
    assert attrs["mlflow.spanFunctionName"].string_value == "test_span"


def test_exporter_posts_per_request_headers_to_collector_url(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)
    otel_span = create_mock_otel_span(trace_id=1, span_id=1)

    exporter._export_spans_incrementally([otel_span])

    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args[0] == f"https://{_ENDPOINT}/v1/traces"
    assert kwargs["headers"] == {
        "Authorization": "Bearer test-access-token",
        "Content-Type": "application/x-protobuf",
        "x-databricks-zerobus-table-name": _TABLE_NAME,
    }
    assert kwargs["timeout"] == _REQUEST_TIMEOUT_SECONDS


def test_exporter_mints_token_per_request(monkeypatch):
    # The auth header is computed for every POST (not cached on the session), so a
    # refreshed token is picked up by the very next export.
    exporter, session, token_source = _make_exporter(monkeypatch)
    token_source.token.side_effect = [_make_token("token-a"), _make_token("token-b")]
    otel_span = create_mock_otel_span(trace_id=1, span_id=1)

    exporter._export_spans_incrementally([otel_span])
    exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 2
    assert _posted_auth_headers(session) == ["Bearer token-a", "Bearer token-b"]


def test_exporter_retries_once_on_401_with_forced_refresh_and_caches_new_token(monkeypatch):
    # Use a real ClientCredentials token source so the refresh-cache path
    # (Refreshable._update_token) is exercised exactly as in production.
    token_source = ClientCredentials(
        client_id="cid",
        client_secret="csecret",
        token_url="https://adb-12345678.azuredatabricks.net/oidc/v1/token",
        scopes="all-apis",
        use_header=True,
    )
    exporter, session, _ = _make_exporter(monkeypatch, token_source=token_source)
    otel_span = create_mock_otel_span(trace_id=2, span_id=2)
    session.post.side_effect = [_make_response(401), _make_response(200), _make_response(200)]

    # First export is rejected with a 401 and succeeds after the forced refresh; the
    # second export must reuse the refreshed token instead of minting yet another one.
    with mock.patch(
        "databricks.sdk.oauth.retrieve_token",
        side_effect=[Token(access_token="token-1"), Token(access_token="token-2")],
    ) as mock_retrieve:
        exporter._export_spans_incrementally([otel_span])
        exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 3
    assert _posted_auth_headers(session) == [
        "Bearer token-1",
        "Bearer token-2",
        "Bearer token-2",
    ]
    assert mock_retrieve.call_count == 2  # initial mint + forced refresh only


def test_exporter_retries_on_401_with_legacy_token_source(monkeypatch):
    # Older databricks-sdk Refreshable has no ``_update_token``; the retry must
    # still use the freshly minted token, and it must be cached via the plain
    # ``_token`` attribute so later exports reuse it without re-minting.
    token_source = _LegacyRefreshableTokenSource(_make_token("token-initial"))
    exporter, session, _ = _make_exporter(monkeypatch, token_source=token_source)
    otel_span = create_mock_otel_span(trace_id=7, span_id=7)
    session.post.side_effect = [_make_response(401), _make_response(200), _make_response(200)]

    exporter._export_spans_incrementally([otel_span])
    exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 3
    assert _posted_auth_headers(session) == [
        "Bearer token-initial",
        "Bearer token-refreshed-1",
        "Bearer token-refreshed-1",
    ]
    assert token_source.refresh_count == 1


def test_exporter_retries_on_401_with_cacheless_token_source(monkeypatch):
    # A token source with no cache attributes at all must not break the 401 retry.
    token_source = _CachelessTokenSource([_make_token("token-a"), _make_token("token-b")])
    exporter, session, _ = _make_exporter(monkeypatch, token_source=token_source)
    otel_span = create_mock_otel_span(trace_id=8, span_id=8)
    session.post.side_effect = [_make_response(401), _make_response(200)]

    exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 2
    assert _posted_auth_headers(session) == ["Bearer token-a", "Bearer token-b"]


# ---------------------------------------------------------------------------
# DatabricksOtelCollectorSpanExporter - failure classification
# ---------------------------------------------------------------------------
#
# sticky  : the batch and all later batches go via the tracing-server REST path
# replay  : this batch goes via REST, the next batch still tries the collector
# drop    : the batch is dropped (delivery ambiguous), the collector is retried


def _case_connect_timeout():
    return [_connect_timeout_error()]


def _case_new_connection_error():
    return [_new_connection_error()]


def _case_name_resolution_error():
    return [_name_resolution_connection_error()]


def _case_connection_reset():
    return [_connection_reset_error(), _make_response(200)]


def _case_read_timeout():
    return [requests.ReadTimeout("read timed out"), _make_response(200)]


def _case_http_500():
    return [_make_response(500, content=b"server error"), _make_response(200)]


def _case_http_400():
    return [_make_response(400, content=b"bad request")]


def _case_http_403():
    return [_make_response(403, content=b"forbidden")]


def _case_http_404():
    return [_make_response(404, content=b"not found")]


def _case_http_401_after_refresh():
    return [_make_response(401), _make_response(401, content=b"denied")]


def _case_http_408():
    return [_make_response(408), _make_response(200)]


def _case_http_413():
    return [_make_response(413), _make_response(200)]


def _case_http_429():
    return [_make_response(429), _make_response(200)]


_CLASSIFICATION_CASES = [
    # Requests that provably never reached the collector: sticky REST fallback.
    (_case_connect_timeout, "sticky"),
    (_case_new_connection_error, "sticky"),
    (_case_name_resolution_error, "sticky"),
    # Connection established then failed mid-request: delivery ambiguous, drop.
    (_case_connection_reset, "drop"),
    (_case_read_timeout, "drop"),
    # Server-side failure after possible ingestion: drop.
    (_case_http_500, "drop"),
    # Definitive client-side rejections: sticky REST fallback.
    (_case_http_400, "sticky"),
    (_case_http_403, "sticky"),
    (_case_http_404, "sticky"),
    (_case_http_401_after_refresh, "sticky"),
    # Transient or batch-specific rejections: replay this batch via REST only.
    (_case_http_408, "replay"),
    (_case_http_413, "replay"),
    (_case_http_429, "replay"),
]


@pytest.mark.parametrize(("post_side_effect_factory", "expected"), _CLASSIFICATION_CASES)
def test_exporter_failure_classification(monkeypatch, post_side_effect_factory, expected):
    post_side_effect = post_side_effect_factory()
    exporter, session, _ = _make_exporter(monkeypatch)
    session.post.side_effect = post_side_effect
    otel_span = create_mock_otel_span(trace_id=13, span_id=13)

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])
        # A second batch separates a sticky fallback (no further collector POSTs)
        # from a one-batch replay (the collector is retried).
        exporter._export_spans_incrementally([otel_span])

    if expected == "sticky":
        # Only the first batch reached the collector; both batches went via REST.
        assert session.post.call_count == len(post_side_effect)
        assert mock_log_spans.call_count == 2
        assert mock_log.warning.call_count == 1
    elif expected == "replay":
        # The first batch was replayed via REST; the second went to the collector.
        assert session.post.call_count == len(post_side_effect)
        assert mock_log_spans.call_count == 1
        mock_log.warning.assert_not_called()
        mock_log.debug.assert_called_once()
    else:  # drop
        # The first batch was dropped (ambiguous delivery); the collector is retried.
        assert session.post.call_count == len(post_side_effect)
        mock_log_spans.assert_not_called()
        assert mock_log.warning.call_count == 1


def test_exporter_401_after_refresh_falls_back_to_rest(monkeypatch):
    exporter, session, token_source = _make_exporter(monkeypatch)
    otel_span = create_mock_otel_span(trace_id=3, span_id=3)
    session.post.side_effect = [_make_response(401), _make_response(401, content=b"denied")]

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])

    # A 401 that survives the refresh retry is a definitive rejection: exactly one
    # retry, then the batch goes through the parent REST path.
    assert session.post.call_count == 2
    token_source.refresh.assert_called_once()
    mock_log.warning.assert_called_once()
    mock_log_spans.assert_called_once()
    location, spans = mock_log_spans.call_args.args
    assert location == _TABLE_NAME
    assert len(spans) == 1
    assert isinstance(spans[0], Span)


def test_exporter_400_falls_back_to_rest_and_is_sticky(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)
    session.post.return_value = _make_response(400, content=b"bad request")
    otel_span = create_mock_otel_span(trace_id=4, span_id=4)

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])
        # A second batch after the sticky flag is set must not POST to the collector.
        exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 1
    # Both batches were replayed through the parent REST path.
    assert mock_log_spans.call_count == 2
    # Exactly one fallback warning for the whole exporter lifetime.
    mock_log.warning.assert_called_once()


def test_exporter_token_mint_failure_replays_batch_via_rest_without_sticking(monkeypatch):
    exporter, session, token_source = _make_exporter(monkeypatch)
    token_source.token.side_effect = [
        RuntimeError("token endpoint unreachable"),
        _make_token("token-b"),
    ]
    otel_span = create_mock_otel_span(trace_id=6, span_id=6)

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])
        exporter._export_spans_incrementally([otel_span])

    # The mint failure happened before any POST was sent, so the first batch is
    # replayed via REST and the collector is still used for the second batch.
    session.post.assert_called_once()
    mock_log_spans.assert_called_once()
    location, spans = mock_log_spans.call_args.args
    assert location == _TABLE_NAME
    assert isinstance(spans[0], Span)
    mock_log.warning.assert_not_called()


def test_exporter_forced_refresh_failure_falls_back_to_rest(monkeypatch):
    exporter, session, token_source = _make_exporter(monkeypatch)
    token_source.refresh.side_effect = RuntimeError("token endpoint unreachable")
    session.post.return_value = _make_response(401)
    otel_span = create_mock_otel_span(trace_id=9, span_id=9)

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])

    # The 401 definitively rejected the batch and the token cannot be refreshed,
    # so the collector path is unusable: the batch is replayed via REST and the
    # exporter sticks to it.
    token_source.refresh.assert_called_once()
    session.post.assert_called_once()
    mock_log_spans.assert_called_once()
    mock_log.warning.assert_called_once()


def test_exporter_connection_error_falls_back_to_rest_and_metadata_still_exported(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)
    session.post.side_effect = _new_connection_error()
    # Trace metadata export must run even when the span sink fails.
    exporter._export_traces = mock.MagicMock()
    otel_span = create_mock_otel_span(trace_id=5, span_id=5)

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter.export([otel_span])

    session.post.assert_called_once()
    mock_log.warning.assert_called_once()
    mock_log_spans.assert_called_once()
    exporter._export_traces.assert_called_once_with([otel_span])


def test_exporter_connection_error_falls_back_through_span_batcher_when_async(monkeypatch):
    # With async logging enabled, the fallback must reuse the parent's SpanBatcher
    # path rather than calling the REST API synchronously.
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "true")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "1")  # no batching
    exporter, session, _ = _make_exporter(monkeypatch, sync_rest=False)
    session.post.side_effect = _new_connection_error()
    otel_span = create_mock_otel_span(trace_id=12, span_id=12)
    mock_client = mock.MagicMock()
    exporter._client = mock_client

    exporter._export_spans_incrementally([otel_span])
    exporter._async_queue.flush(terminate=True)

    mock_client.log_spans.assert_called_once()
    args = mock_client.log_spans.call_args.args
    assert args[0] == _TABLE_NAME
    assert len(args[1]) == 1
    assert isinstance(args[1][0], Span)


def test_fallback_replays_to_init_time_table_when_active_table_unset_sync(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    exporter, session, _ = _make_exporter(monkeypatch, sync_rest=False)
    session.post.return_value = _make_response(400, content=b"bad request")
    otel_span = create_mock_otel_span(trace_id=30, span_id=30)

    with (
        mock.patch(
            "mlflow.tracing.export.uc_table.get_active_spans_table_name", return_value=None
        ) as mock_active_table,
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
    ):
        exporter._export_spans_incrementally([otel_span])

    # The replay targets the table captured at exporter init, not the (unset)
    # active spans table.
    mock_log_spans.assert_called_once()
    location, spans = mock_log_spans.call_args.args
    assert location == _TABLE_NAME
    assert isinstance(spans[0], Span)
    mock_active_table.assert_not_called()


def test_fallback_replays_to_init_time_table_when_active_table_unset_async(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "true")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "1")
    exporter, session, _ = _make_exporter(monkeypatch, sync_rest=False)
    session.post.return_value = _make_response(400, content=b"bad request")
    otel_span = create_mock_otel_span(trace_id=31, span_id=31)
    mock_client = mock.MagicMock()
    exporter._client = mock_client

    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name", return_value=None
    ) as mock_active_table:
        exporter._export_spans_incrementally([otel_span])
        exporter._async_queue.flush(terminate=True)

    mock_client.log_spans.assert_called_once()
    args = mock_client.log_spans.call_args.args
    assert args[0] == _TABLE_NAME
    assert len(args[1]) == 1
    assert isinstance(args[1][0], Span)
    mock_active_table.assert_not_called()


def test_exporter_500_warns_and_does_not_fall_back(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)
    session.post.return_value = _make_response(500, content=b"server error")
    otel_span = create_mock_otel_span(trace_id=10, span_id=10)

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])
        # Delivery is ambiguous, so the exporter must not pin the REST path: the
        # next batch still goes to the collector.
        exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 2
    mock_log_spans.assert_not_called()
    # The ambiguous-delivery warning fires once per exporter; repeats log at DEBUG.
    assert mock_log.warning.call_count == 1
    assert mock_log.debug.call_count == 1
    assert 500 in mock_log.warning.call_args.args
    assert b"server error" in mock_log.warning.call_args.args


def test_exporter_timeout_warns_and_does_not_fall_back(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)
    session.post.side_effect = requests.Timeout("timed out")
    otel_span = create_mock_otel_span(trace_id=11, span_id=11)

    with (
        mock.patch.object(exporter, "_log_spans") as mock_log_spans,
        mock.patch(f"{_MODULE}._logger") as mock_log,
    ):
        exporter._export_spans_incrementally([otel_span])

    # A read timeout leaves delivery ambiguous (the batch may have been ingested),
    # so the batch is dropped with a warning instead of replayed over REST.
    session.post.assert_called_once()
    mock_log_spans.assert_not_called()
    mock_log.warning.assert_called_once()


def test_exporter_no_op_on_empty_spans(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)
    exporter._export_spans_incrementally([])
    session.post.assert_not_called()


def test_shutdown_closes_session(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)

    with mock.patch.object(DatabricksUCTableSpanExporter, "shutdown") as mock_super_shutdown:
        exporter.shutdown()

    mock_super_shutdown.assert_called_once()
    session.close.assert_called_once()


def test_shutdown_closes_session_even_when_parent_shutdown_raises(monkeypatch):
    exporter, session, _ = _make_exporter(monkeypatch)

    with (
        mock.patch.object(
            DatabricksUCTableSpanExporter, "shutdown", side_effect=RuntimeError("shutdown failed")
        ),
        pytest.raises(RuntimeError, match="shutdown failed"),
    ):
        exporter.shutdown()

    session.close.assert_called_once()


# ---------------------------------------------------------------------------
# _get_table_name_from_destination
# ---------------------------------------------------------------------------


def test_get_table_name_from_uc_schema_location():
    dest = UCSchemaLocation(catalog_name="cat", schema_name="sch")
    # Default table name is set by the constant
    result = _get_table_name_from_destination(dest)
    assert result is not None
    assert result.startswith("cat.sch.")


def test_get_table_name_returns_none_for_unknown():
    result = _get_table_name_from_destination(object())
    assert result is None
