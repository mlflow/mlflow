import json
import threading
from unittest import mock

import pytest
import requests
from databricks.sdk.oauth import ClientCredentials, Token
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.export.zerobus import (
    _REQUEST_TIMEOUT_SECONDS,
    DatabricksZerobusSpanExporter,
    _get_table_name_from_destination,
    build_table_authorization_details,
    build_zerobus_token_source,
    get_zerobus_span_exporter,
    is_zerobus_host,
    resolve_zerobus_endpoint,
)

from tests.tracing.helper import create_mock_otel_span

# ---------------------------------------------------------------------------
# is_zerobus_host
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
        # Missing region (empty region between .zerobus. and suffix)
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
def test_is_zerobus_host(endpoint: str, workspace_id: str, expected: bool):
    assert is_zerobus_host(endpoint, workspace_id) == expected


# ---------------------------------------------------------------------------
# resolve_zerobus_endpoint
# ---------------------------------------------------------------------------


def _make_summary(region="us-west-2", cloud="aws", global_metastore_id=None):
    summary = mock.MagicMock()
    summary.region = region
    summary.cloud = cloud
    summary.global_metastore_id = global_metastore_id
    return summary


def test_resolve_zerobus_endpoint_aws():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=None)),
        ),
        mock.patch(
            "databricks.sdk.WorkspaceClient",
            return_value=mock.MagicMock(
                metastores=mock.MagicMock(
                    summary=mock.MagicMock(return_value=_make_summary("us-west-2", "aws"))
                )
            ),
        ),
    ):
        result = resolve_zerobus_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result == "12345678.zerobus.us-west-2.cloud.databricks.com"


def test_resolve_zerobus_endpoint_azure():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=None)),
        ),
        mock.patch(
            "databricks.sdk.WorkspaceClient",
            return_value=mock.MagicMock(
                metastores=mock.MagicMock(
                    summary=mock.MagicMock(return_value=_make_summary("eastus", "azure"))
                )
            ),
        ),
    ):
        result = resolve_zerobus_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result == "12345678.zerobus.eastus.azuredatabricks.net"


def test_resolve_zerobus_endpoint_gcp():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=None)),
        ),
        mock.patch(
            "databricks.sdk.WorkspaceClient",
            return_value=mock.MagicMock(
                metastores=mock.MagicMock(
                    summary=mock.MagicMock(return_value=_make_summary("us-central1", "gcp"))
                )
            ),
        ),
    ):
        result = resolve_zerobus_endpoint(
            host="https://12345678.gcp.databricks.com",
            workspace_id="12345678",
        )
    assert result == "12345678.zerobus.us-central1.gcp.databricks.com"


def test_resolve_zerobus_endpoint_staging_segment():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=None)),
        ),
        mock.patch(
            "databricks.sdk.WorkspaceClient",
            return_value=mock.MagicMock(
                metastores=mock.MagicMock(
                    summary=mock.MagicMock(return_value=_make_summary("us-west-2", "aws"))
                )
            ),
        ),
    ):
        result = resolve_zerobus_endpoint(
            host="https://adb-12345678.staging.cloud.databricks.com",
            workspace_id="12345678",
        )
    assert result == "12345678.zerobus.us-west-2.staging.cloud.databricks.com"


def test_resolve_zerobus_endpoint_override_takes_precedence():
    valid_override = "12345678.zerobus.eu-west-1.cloud.databricks.com"
    with mock.patch(
        "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
        new=mock.MagicMock(get=mock.MagicMock(return_value=valid_override)),
    ):
        result = resolve_zerobus_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result == valid_override


def test_resolve_zerobus_endpoint_override_invalid_returns_none():
    invalid_override = "badhost.example.com"
    with mock.patch(
        "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
        new=mock.MagicMock(get=mock.MagicMock(return_value=invalid_override)),
    ):
        result = resolve_zerobus_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result is None


def test_resolve_zerobus_endpoint_metastore_error_returns_none():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=None)),
        ),
        mock.patch(
            "databricks.sdk.WorkspaceClient",
            side_effect=RuntimeError("connection refused"),
        ),
    ):
        result = resolve_zerobus_endpoint(
            host="https://adb-12345678.azuredatabricks.net",
            workspace_id="12345678",
        )
    assert result is None


def test_resolve_zerobus_endpoint_cloud_from_global_metastore_id():
    summary = _make_summary(region="us-west-2", cloud=None, global_metastore_id="aws:us-west-2:abc")
    summary.cloud = None
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=None)),
        ),
        mock.patch(
            "databricks.sdk.WorkspaceClient",
            return_value=mock.MagicMock(
                metastores=mock.MagicMock(summary=mock.MagicMock(return_value=summary))
            ),
        ),
    ):
        result = resolve_zerobus_endpoint(
            host="https://adb-12345678.cloud.databricks.com",
            workspace_id="12345678",
        )
    assert result == "12345678.zerobus.us-west-2.cloud.databricks.com"


def test_resolve_zerobus_endpoint_passes_service_principal_credentials():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ZEROBUS_ENDPOINT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=None)),
        ),
        mock.patch(
            "databricks.sdk.WorkspaceClient",
            return_value=mock.MagicMock(
                metastores=mock.MagicMock(
                    summary=mock.MagicMock(return_value=_make_summary("us-west-2", "aws"))
                )
            ),
        ) as mock_ws,
    ):
        resolve_zerobus_endpoint(
            host="https://adb-12345678.cloud.databricks.com",
            workspace_id="12345678",
            client_id="sp-client-id",
            client_secret="sp-secret",
        )
    mock_ws.assert_called_once_with(
        host="https://adb-12345678.cloud.databricks.com",
        client_id="sp-client-id",
        client_secret="sp-secret",
    )


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
# build_zerobus_token_source
# ---------------------------------------------------------------------------


def test_build_zerobus_token_source_construction():
    with mock.patch(
        "databricks.sdk.oauth.ClientCredentials",
        autospec=True,
    ) as mock_cc:
        build_zerobus_token_source(
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
# get_zerobus_span_exporter
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


def _make_destination(table_name="cat.sch.tbl"):
    dest = mock.MagicMock()
    dest.full_otel_spans_table_name = table_name
    return dest


def test_get_zerobus_span_exporter_flag_off_returns_none():
    with mock.patch(
        "mlflow.tracing.export.zerobus.MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT",
        new=mock.MagicMock(get=mock.MagicMock(return_value=False)),
    ):
        result = get_zerobus_span_exporter(_make_destination(), "databricks")
    assert result is None


def test_get_zerobus_span_exporter_no_sp_creds_returns_none():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=True)),
        ),
        mock.patch(
            "mlflow.tracing.export.zerobus.get_databricks_host_creds",
            return_value=_make_host_creds(client_id=None, client_secret=None),
        ),
    ):
        result = get_zerobus_span_exporter(_make_destination(), "databricks")
    assert result is None


def test_get_zerobus_span_exporter_endpoint_unresolvable_returns_none():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=True)),
        ),
        mock.patch(
            "mlflow.tracing.export.zerobus.get_databricks_host_creds",
            return_value=_make_host_creds(),
        ),
        mock.patch(
            "mlflow.tracing.export.zerobus.resolve_zerobus_endpoint",
            return_value=None,
        ),
    ):
        result = get_zerobus_span_exporter(_make_destination(), "databricks")
    assert result is None


def test_get_zerobus_span_exporter_returns_exporter_when_all_present():
    valid_endpoint = "ws123.zerobus.us-west-2.cloud.databricks.com"
    mock_token_source = mock.MagicMock()
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=True)),
        ),
        mock.patch(
            "mlflow.tracing.export.zerobus.get_databricks_host_creds",
            return_value=_make_host_creds(workspace_id="ws123"),
        ),
        mock.patch(
            "mlflow.tracing.export.zerobus.resolve_zerobus_endpoint",
            return_value=valid_endpoint,
        ),
        mock.patch(
            "mlflow.tracing.export.zerobus.build_zerobus_token_source",
            return_value=mock_token_source,
        ),
        mock.patch.object(
            DatabricksZerobusSpanExporter, "__init__", return_value=None
        ) as mock_init,
    ):
        result = get_zerobus_span_exporter(_make_destination("cat.sch.tbl"), "databricks")

    mock_init.assert_called_once()
    _, kwargs = mock_init.call_args
    assert kwargs["endpoint"] == valid_endpoint
    assert kwargs["token_source"] is mock_token_source
    assert kwargs["table_name"] == "cat.sch.tbl"
    # result is the exporter instance (truthy since __init__ patched to return None
    # means object was constructed)
    assert isinstance(result, DatabricksZerobusSpanExporter)


def test_get_zerobus_span_exporter_creds_exception_returns_none():
    with (
        mock.patch(
            "mlflow.tracing.export.zerobus.MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT",
            new=mock.MagicMock(get=mock.MagicMock(return_value=True)),
        ),
        mock.patch(
            "mlflow.tracing.export.zerobus.get_databricks_host_creds",
            side_effect=RuntimeError("auth error"),
        ),
    ):
        result = get_zerobus_span_exporter(_make_destination(), "databricks")
    assert result is None


# ---------------------------------------------------------------------------
# DatabricksZerobusSpanExporter - export behavior
# ---------------------------------------------------------------------------

_TABLE_NAME = "cat.sch.tbl"
_ENDPOINT = "ws123.zerobus.us-west-2.cloud.databricks.com"


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


def _make_exporter(table_name=_TABLE_NAME, endpoint=_ENDPOINT, token_source=None):
    if token_source is None:
        token_source = _make_mock_token_source()

    exporter = DatabricksZerobusSpanExporter(
        tracking_uri="databricks",
        endpoint=endpoint,
        token_source=token_source,
        table_name=table_name,
    )
    session = mock.MagicMock()
    session.post.return_value = _make_response(200)
    exporter._session = session
    return exporter, session, token_source


def _posted_auth_headers(session):
    return [call.kwargs["headers"]["Authorization"] for call in session.post.call_args_list]


def test_exporter_posts_otlp_request_with_decoded_attributes():
    exporter, session, token_source = _make_exporter()
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


def test_exporter_posts_per_request_headers_to_zerobus_url():
    exporter, session, _ = _make_exporter()
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


def test_exporter_mints_token_per_request():
    # The auth header is computed for every POST (not cached on the session), so a
    # refreshed token is picked up by the very next export.
    exporter, session, token_source = _make_exporter()
    token_source.token.side_effect = [_make_token("token-a"), _make_token("token-b")]
    otel_span = create_mock_otel_span(trace_id=1, span_id=1)

    exporter._export_spans_incrementally([otel_span])
    exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 2
    assert _posted_auth_headers(session) == ["Bearer token-a", "Bearer token-b"]


def test_exporter_retries_once_on_401_with_forced_refresh_and_caches_new_token():
    # Use a real ClientCredentials token source so the refresh-cache path
    # (Refreshable._update_token) is exercised exactly as in production.
    token_source = ClientCredentials(
        client_id="cid",
        client_secret="csecret",
        token_url="https://adb-12345678.azuredatabricks.net/oidc/v1/token",
        scopes="all-apis",
        use_header=True,
    )
    exporter, session, _ = _make_exporter(token_source=token_source)
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


def test_exporter_retries_on_401_with_legacy_token_source():
    # Older databricks-sdk Refreshable has no ``_update_token``; the retry must
    # still use the freshly minted token, and it must be cached via the plain
    # ``_token`` attribute so later exports reuse it without re-minting.
    token_source = _LegacyRefreshableTokenSource(_make_token("token-initial"))
    exporter, session, _ = _make_exporter(token_source=token_source)
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


def test_exporter_retries_on_401_with_cacheless_token_source():
    # A token source with no cache attributes at all must not break the 401 retry.
    token_source = _CachelessTokenSource([_make_token("token-a"), _make_token("token-b")])
    exporter, session, _ = _make_exporter(token_source=token_source)
    otel_span = create_mock_otel_span(trace_id=8, span_id=8)
    session.post.side_effect = [_make_response(401), _make_response(200)]

    exporter._export_spans_incrementally([otel_span])

    assert session.post.call_count == 2
    assert _posted_auth_headers(session) == ["Bearer token-a", "Bearer token-b"]


def test_exporter_warns_when_forced_refresh_fails():
    exporter, session, token_source = _make_exporter()
    token_source.refresh.side_effect = RuntimeError("token endpoint unreachable")
    otel_span = create_mock_otel_span(trace_id=9, span_id=9)
    session.post.return_value = _make_response(401)

    with mock.patch("mlflow.tracing.export.zerobus._logger") as mock_log:
        exporter._export_spans_incrementally([otel_span])

    # A failed forced refresh logs a warning and must not send a retry.
    token_source.refresh.assert_called_once()
    session.post.assert_called_once()
    mock_log.warning.assert_called_once()


def test_exporter_warns_when_401_retry_still_fails():
    exporter, session, token_source = _make_exporter()
    otel_span = create_mock_otel_span(trace_id=3, span_id=3)
    session.post.side_effect = [_make_response(401), _make_response(401, content=b"denied")]

    with mock.patch("mlflow.tracing.export.zerobus._logger") as mock_log:
        exporter._export_spans_incrementally([otel_span])

    # Exactly one retry, then degrade to a warning.
    assert session.post.call_count == 2
    token_source.refresh.assert_called_once()
    mock_log.warning.assert_called_once()
    assert 401 in mock_log.warning.call_args.args
    assert b"denied" in mock_log.warning.call_args.args


@pytest.mark.parametrize("status_code", [400, 500])
def test_exporter_no_retry_no_refresh_on_other_http_errors(status_code):
    exporter, session, token_source = _make_exporter()
    otel_span = create_mock_otel_span(trace_id=4, span_id=4)
    session.post.return_value = _make_response(status_code, content=b"bad request")

    with mock.patch("mlflow.tracing.export.zerobus._logger") as mock_log:
        exporter._export_spans_incrementally([otel_span])

    session.post.assert_called_once()
    token_source.token.assert_called_once()
    token_source.refresh.assert_not_called()
    mock_log.warning.assert_called_once()
    assert status_code in mock_log.warning.call_args.args
    assert b"bad request" in mock_log.warning.call_args.args


def test_exporter_warns_on_request_error_and_metadata_still_exported():
    exporter, session, _ = _make_exporter()
    session.post.side_effect = requests.ConnectionError("connection refused")
    # Trace metadata export must run even when the span sink errors out.
    exporter._export_traces = mock.MagicMock()
    otel_span = create_mock_otel_span(trace_id=5, span_id=5)

    with mock.patch("mlflow.tracing.export.zerobus._logger") as mock_log:
        exporter.export([otel_span])

    session.post.assert_called_once()
    mock_log.warning.assert_called_once()
    exporter._export_traces.assert_called_once_with([otel_span])


def test_exporter_warns_when_token_mint_fails():
    exporter, session, token_source = _make_exporter()
    token_source.token.side_effect = RuntimeError("token endpoint unreachable")
    otel_span = create_mock_otel_span(trace_id=6, span_id=6)

    with mock.patch("mlflow.tracing.export.zerobus._logger") as mock_log:
        exporter._export_spans_incrementally([otel_span])

    session.post.assert_not_called()
    mock_log.warning.assert_called_once()


def test_exporter_no_op_on_empty_spans():
    exporter, session, _ = _make_exporter()
    exporter._export_spans_incrementally([])
    session.post.assert_not_called()


def test_shutdown_closes_session():
    exporter, session, _ = _make_exporter()

    with mock.patch.object(DatabricksUCTableSpanExporter, "shutdown") as mock_super_shutdown:
        exporter.shutdown()

    mock_super_shutdown.assert_called_once()
    session.close.assert_called_once()


def test_shutdown_closes_session_even_when_parent_shutdown_raises():
    exporter, session, _ = _make_exporter()

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
    from mlflow.entities.trace_location import UCSchemaLocation

    dest = UCSchemaLocation(catalog_name="cat", schema_name="sch")
    # Default table name is set by the constant
    result = _get_table_name_from_destination(dest)
    assert result is not None
    assert result.startswith("cat.sch.")


def test_get_table_name_returns_none_for_unknown():
    result = _get_table_name_from_destination(object())
    assert result is None
