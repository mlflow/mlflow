import json
from unittest import mock

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from mlflow.tracing.export.zerobus import (
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


def _make_exporter(
    table_name="cat.sch.tbl", endpoint="ws123.zerobus.us-west-2.cloud.databricks.com"
):
    token = mock.MagicMock()
    token.access_token = "test-access-token"
    token_source = mock.MagicMock()
    token_source.token.return_value = token
    token_source.refresh.return_value = token

    with mock.patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        autospec=True,
    ) as mock_otlp_cls:
        mock_otlp = mock.MagicMock()
        mock_otlp._session = mock.MagicMock()
        mock_otlp._session.headers = {}
        mock_otlp.export.return_value = SpanExportResult.SUCCESS
        mock_otlp_cls.return_value = mock_otlp

        exporter = DatabricksZerobusSpanExporter(
            tracking_uri="databricks",
            endpoint=endpoint,
            token_source=token_source,
            table_name=table_name,
        )
        exporter._otlp_exporter = mock_otlp
        exporter._token_source = token_source

    return exporter, mock_otlp, token_source


def test_exporter_sets_auth_and_table_headers_on_export():
    exporter, mock_otlp, token_source = _make_exporter()
    otel_span = create_mock_otel_span(trace_id=1, span_id=1)

    exporter._export_spans_incrementally([otel_span])

    token_source.token.assert_called_once()
    assert mock_otlp._session.headers["Authorization"] == "Bearer test-access-token"
    assert mock_otlp._session.headers["x-databricks-zerobus-table-name"] == "cat.sch.tbl"
    mock_otlp.export.assert_called_once_with([otel_span])


def test_exporter_retries_on_failure_with_refreshed_token():
    exporter, mock_otlp, token_source = _make_exporter()
    otel_span = create_mock_otel_span(trace_id=2, span_id=2)

    # First export returns FAILURE, second returns SUCCESS (after token refresh).
    mock_otlp.export.side_effect = [SpanExportResult.FAILURE, SpanExportResult.SUCCESS]

    exporter._export_spans_incrementally([otel_span])

    assert mock_otlp.export.call_count == 2
    token_source.refresh.assert_called_once()


def test_exporter_warns_on_persistent_failure():
    exporter, mock_otlp, token_source = _make_exporter()
    otel_span = create_mock_otel_span(trace_id=3, span_id=3)

    mock_otlp.export.return_value = SpanExportResult.FAILURE

    with mock.patch("mlflow.tracing.export.zerobus._logger") as mock_log:
        exporter._export_spans_incrementally([otel_span])

    mock_log.warning.assert_called()
    assert mock_otlp.export.call_count == 2  # initial + one retry


def test_exporter_no_op_on_empty_spans():
    exporter, mock_otlp, _ = _make_exporter()
    exporter._export_spans_incrementally([])
    mock_otlp.export.assert_not_called()


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
