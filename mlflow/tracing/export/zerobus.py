"""
Experimental Zerobus OTLP span exporter for Databricks Unity Catalog tracing.

When ``MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT=true``, spans are sent directly to the
Databricks Zerobus OTLP ingest endpoint instead of the MLflow REST ``log_spans``
API. Trace-level metadata (TraceInfo) continues to flow through the standard
MLflow backend path unchanged.

This module is DEFAULT OFF. Set ``MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT=true`` to
opt in.
"""

import json
import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from mlflow.environment_variables import MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT, MLFLOW_ZEROBUS_ENDPOINT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.utils.databricks_utils import get_databricks_host_creds

_logger = logging.getLogger(__name__)

# OTLP token path from databricks-sdk oauth module.
# We re-declare rather than import to avoid a hard dependency path at module level;
# the value is stable across SDK versions.
_OIDC_TOKEN_PATH = "/oidc/v1/token"

# Zerobus OTLP ingest port and path.
_ZEROBUS_OTLP_PORT = 443
_ZEROBUS_OTLP_PATH = "/v1/traces"

# Header names expected by the Zerobus ingest service.
_HEADER_AUTHORIZATION = "Authorization"
_HEADER_TABLE_NAME = "x-databricks-zerobus-table-name"

# Domain suffixes that identify valid Zerobus hosts.
_VALID_ZEROBUS_SUFFIXES = (
    ".cloud.databricks.com",
    ".azuredatabricks.net",
    ".gcp.databricks.com",
)

# Cloud label -> domain mapping.
_CLOUD_DOMAIN = {
    "aws": "cloud.databricks.com",
    "azure": "azuredatabricks.net",
    "gcp": "gcp.databricks.com",
}


def is_zerobus_host(endpoint: str, workspace_id: str) -> bool:
    """Validate that *endpoint* is a legitimate Zerobus OTLP host for *workspace_id*.

    Port of the Go validator. Rules:
    - Strip a leading ``https://``.
    - Reject if the remainder contains any of ``/?#@:``.
    - Must start with ``{workspace_id}.zerobus.``.
    - Must end with one of the known cloud domain suffixes.
    - Must have a non-empty region label between the ``.zerobus.`` prefix and the
      cloud domain suffix.

    Returns ``True`` when all rules pass, ``False`` otherwise.
    """
    host = endpoint
    host = host.removeprefix("https://")

    # Reject any host with authority-separator or path characters.
    if any(c in host for c in "/?#@:"):
        return False

    expected_prefix = f"{workspace_id}.zerobus."
    if not host.startswith(expected_prefix):
        return False

    # After the prefix the remainder must end with a known suffix, and there must be
    # at least one non-empty region segment between the prefix and that suffix.
    remainder = host[len(expected_prefix) :]
    matched_suffix = next(
        (suffix for suffix in _VALID_ZEROBUS_SUFFIXES if remainder.endswith(suffix)),
        None,
    )
    if matched_suffix is None:
        return False

    region_part = remainder[: -len(matched_suffix)]
    # Region must be non-empty and must not itself contain a trailing dot (i.e. not
    # an empty staging segment after the region).
    return bool(region_part) and not region_part.endswith(".")


def resolve_zerobus_endpoint(
    host: str,
    workspace_id: str,
) -> str | None:
    """Assemble and validate the Zerobus OTLP ingest endpoint for a workspace.

    Precedence:
    1. ``MLFLOW_ZEROBUS_ENDPOINT`` env-var override (validated, returned as-is).
    2. Auto-assembled from workspace metastore metadata.

    The assembled form is::

        {workspace_id}.zerobus.{region}.{env_segment}{domain}

    where ``env_segment`` is ``"staging."`` when *host* contains ``.staging.``, and
    ``""`` otherwise.

    Returns the host string (no ``https://`` prefix, no trailing slash) on success,
    or ``None`` on any failure (logged at DEBUG).
    """
    # Explicit override takes highest precedence.
    if override := MLFLOW_ZEROBUS_ENDPOINT.get():
        if is_zerobus_host(override, workspace_id):
            return override
        _logger.debug(
            "MLFLOW_ZEROBUS_ENDPOINT override %r failed is_zerobus_host validation "
            "for workspace_id=%r; ignoring override.",
            override,
            workspace_id,
        )
        return None

    try:
        from databricks.sdk import WorkspaceClient

        ws = WorkspaceClient(host=host)
        summary = ws.metastores.summary()
    except Exception as exc:
        _logger.debug("Failed to fetch metastore summary for Zerobus endpoint resolution: %s", exc)
        return None

    region = summary.region
    cloud_raw = summary.cloud  # e.g. "aws", "azure", "gcp"

    if not region:
        _logger.debug("Metastore summary returned empty region; cannot resolve Zerobus endpoint.")
        return None

    if not cloud_raw:
        # Attempt to parse cloud from global_metastore_id: "cloud:region:id"
        gid = summary.global_metastore_id or ""
        parts = gid.split(":")
        cloud_raw = parts[0] if parts else ""

    cloud_raw = (cloud_raw or "").lower()
    domain = _CLOUD_DOMAIN.get(cloud_raw)
    if not domain:
        _logger.debug(
            "Unrecognised cloud %r from metastore summary; cannot resolve Zerobus endpoint.",
            cloud_raw,
        )
        return None

    env_segment = "staging." if ".staging." in host else ""
    endpoint = f"{workspace_id}.zerobus.{region}.{env_segment}{domain}"

    if not is_zerobus_host(endpoint, workspace_id):
        _logger.debug("Assembled Zerobus endpoint %r failed is_zerobus_host validation.", endpoint)
        return None

    return endpoint


def build_table_authorization_details(tables: list[str]) -> str:
    """Return a JSON string encoding RFC 9396 authorization details for *tables*.

    For each fully-qualified ``catalog.schema.table`` name, three objects are
    emitted:

    - CATALOG ``cat`` - ``["USE CATALOG"]``
    - SCHEMA ``cat.sch`` - ``["USE SCHEMA"]``
    - TABLE ``cat.sch.tbl`` - ``["SELECT", "MODIFY"]``

    Keys: ``type``, ``object_type``, ``object_full_path``, ``privileges``.

    .. note::
        The exact ``type`` string ``"unity_catalog_privileges"`` is used here as the
        RFC 9396 ``type`` field value. This string has not been verified against a
        live Zerobus token endpoint and is an open item pending confirmation.
    """
    entries: list[dict[str, object]] = []
    for table in tables:
        parts = table.split(".")
        if len(parts) != 3:
            _logger.debug("Skipping malformed table name %r (expected cat.sch.tbl).", table)
            continue
        catalog, schema, _ = parts
        entries.append({
            "type": "unity_catalog_privileges",
            "object_type": "CATALOG",
            "object_full_path": catalog,
            "privileges": ["USE CATALOG"],
        })
        entries.append({
            "type": "unity_catalog_privileges",
            "object_type": "SCHEMA",
            "object_full_path": f"{catalog}.{schema}",
            "privileges": ["USE SCHEMA"],
        })
        entries.append({
            "type": "unity_catalog_privileges",
            "object_type": "TABLE",
            "object_full_path": table,
            "privileges": ["SELECT", "MODIFY"],
        })
    return json.dumps(entries)


def build_zerobus_token_source(
    host: str,
    client_id: str,
    client_secret: str,
    workspace_id: str,
    tables: list[str],
):
    """Build a ``ClientCredentials`` token source for Zerobus Direct Write.

    Args:
        host: Workspace host URL (e.g. ``https://adb-xxx.azuredatabricks.net``).
        client_id: Service principal client ID.
        client_secret: Service principal client secret.
        workspace_id: Numeric workspace ID string.
        tables: Fully-qualified UC table names for which authorization is requested.

    Returns:
        A ``databricks.sdk.oauth.ClientCredentials`` instance.
    """
    from databricks.sdk.oauth import ClientCredentials  # lazy: databricks top-level import rule

    token_url = f"{host.rstrip('/')}{_OIDC_TOKEN_PATH}"
    return ClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url,
        scopes="all-apis",
        endpoint_params={
            "resource": f"api://databricks/workspaces/{workspace_id}/zerobusDirectWriteApi"
        },
        authorization_details=build_table_authorization_details(tables),
        use_header=True,
    )


class DatabricksZerobusSpanExporter(DatabricksUCTableSpanExporter):
    """Span exporter that sends spans to Databricks Zerobus via OTLP/HTTP protobuf.

    Trace-level metadata (TraceInfo) continues to flow through the inherited
    MLflow REST backend path (``_export_traces`` / ``_log_trace`` /
    ``client.start_trace``). Only the span data is redirected to Zerobus.

    The exporter holds an internal ``OTLPSpanExporter`` pointed at
    ``https://{endpoint}:{port}/v1/traces``. Before each batch export the
    ``Authorization`` and ``x-databricks-zerobus-table-name`` headers are
    refreshed on its ``requests.Session``. On a 401 response from the OTLP
    exporter, the token source is force-refreshed and the export is retried
    once, then a warning is logged if it still fails (the app is not crashed).
    """

    def __init__(
        self,
        tracking_uri: str | None,
        endpoint: str,
        token_source,
        table_name: str,
    ) -> None:
        super().__init__(tracking_uri=tracking_uri)
        self._zerobus_endpoint = endpoint
        self._token_source = token_source
        self._table_name = table_name

        otlp_url = f"https://{endpoint}:{_ZEROBUS_OTLP_PORT}{_ZEROBUS_OTLP_PATH}"
        # `opentelemetry-exporter-otlp-proto-http` is an optional dependency (mirrors
        # `mlflow/tracing/utils/otlp.py`), so import it lazily here rather than at module top
        # level: this module is imported unconditionally on the UC-table tracing path, and a
        # missing package must not break that path when the Zerobus flag is off. The factory
        # `get_zerobus_span_exporter` catches this and falls back to the REST exporter.
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as e:
            raise MlflowException(
                "The HTTP OTLP exporter is required for Zerobus trace export but is not "
                "installed. Install it with `pip install opentelemetry-exporter-otlp-proto-http`.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e
        # Initial headers are empty; auth + table-name headers are injected before each export.
        self._otlp_exporter = OTLPSpanExporter(endpoint=otlp_url, headers={})

    def _refresh_auth_headers(self) -> None:
        """Inject a fresh token and table-name header onto the OTLP session."""
        try:
            token = self._token_source.token()
            self._otlp_exporter._session.headers[_HEADER_AUTHORIZATION] = (
                f"Bearer {token.access_token}"
            )
            self._otlp_exporter._session.headers[_HEADER_TABLE_NAME] = self._table_name
        except Exception as exc:
            _logger.warning("Failed to refresh Zerobus auth token: %s", exc)
            raise

    def _export_spans_to_zerobus(self, spans: Sequence[ReadableSpan]) -> None:
        """Send *spans* to Zerobus, retrying once on a 401 after forcing token refresh."""
        self._refresh_auth_headers()
        result = self._otlp_exporter.export(spans)
        if result == SpanExportResult.FAILURE:
            # The OTLPSpanExporter does not expose the HTTP status code to callers;
            # we conservatively treat any failure as a potential auth issue and
            # attempt a single retry with a refreshed token.
            _logger.debug("Zerobus OTLP export failed; forcing token refresh and retrying once.")
            try:
                # Force a fresh token by invalidating the cached one. ClientCredentials
                # descends from Refreshable which caches the token; calling refresh()
                # directly bypasses the cache and returns a new Token, but the cached
                # value is not updated unless we call token() afterwards.
                # We update the header directly from the refresh result here.
                new_token = self._token_source.refresh()
                self._otlp_exporter._session.headers[_HEADER_AUTHORIZATION] = (
                    f"Bearer {new_token.access_token}"
                )
                retry_result = self._otlp_exporter.export(spans)
            except Exception as exc:
                _logger.warning(
                    "Zerobus OTLP export failed after token refresh: %s. "
                    "Spans were NOT exported to Zerobus.",
                    exc,
                )
                return

            if retry_result == SpanExportResult.FAILURE:
                _logger.warning(
                    "Zerobus OTLP export failed after token refresh. "
                    "Spans were NOT exported to Zerobus."
                )

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """Override: send raw ReadableSpans to Zerobus instead of the UC table REST path.

        The parent ``_export_traces`` path (TraceInfo / metadata) is NOT touched here;
        metadata continues to flow through the inherited ``MlflowV3SpanExporter``
        implementation.
        """
        if not spans:
            return
        # Never let a span-export failure propagate into the parent ``export()``: doing so would
        # skip ``_export_traces`` and silently drop trace metadata. Span-export failures degrade
        # to a warning; the inherited metadata path is unaffected.
        try:
            self._export_spans_to_zerobus(spans)
        except Exception as exc:
            _logger.warning(
                "Zerobus span export failed: %s. Spans were NOT exported to Zerobus; "
                "trace metadata export via the MLflow backend is unaffected.",
                exc,
            )

    def shutdown(self) -> None:
        super().shutdown()
        self._otlp_exporter.shutdown()


def get_zerobus_span_exporter(
    destination,
    tracking_uri: str | None,
) -> DatabricksZerobusSpanExporter | None:
    """Return a ``DatabricksZerobusSpanExporter`` when all prerequisites are met.

    Returns ``None`` (so the caller falls back to the standard UC table path) when:
    - ``MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT`` is ``False`` (the default).
    - Service-principal credentials are not available on *tracking_uri*.
    - The Zerobus endpoint cannot be resolved or fails validation.

    A warning is logged when the flag is on but a prerequisite is missing, so
    users who opt in learn why the fallback occurred.

    Args:
        destination: The trace destination (``UCSchemaLocation`` or ``UnityCatalog``).
        tracking_uri: MLflow tracking URI to resolve Databricks credentials from.

    Returns:
        A configured ``DatabricksZerobusSpanExporter``, or ``None``.
    """
    if not MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT.get():
        return None

    # Resolve service-principal credentials.
    try:
        host_creds = get_databricks_host_creds(tracking_uri)
    except Exception as exc:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but Databricks credentials "
            "could not be retrieved for tracking URI %r: %s. "
            "Falling back to the standard UC table exporter.",
            tracking_uri,
            exc,
        )
        return None

    client_id = getattr(host_creds, "client_id", None)
    client_secret = getattr(host_creds, "client_secret", None)
    host = getattr(host_creds, "host", None)
    workspace_id = str(getattr(host_creds, "workspace_id", "") or "")

    if not client_id or not client_secret:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but service-principal credentials "
            "(client_id + client_secret) are not available for tracking URI %r. "
            "Falling back to the standard UC table exporter.",
            tracking_uri,
        )
        return None

    if not host:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but the workspace host could not "
            "be resolved from tracking URI %r. "
            "Falling back to the standard UC table exporter.",
            tracking_uri,
        )
        return None

    if not workspace_id:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but the workspace ID could not "
            "be resolved from tracking URI %r. "
            "Falling back to the standard UC table exporter.",
            tracking_uri,
        )
        return None

    endpoint = resolve_zerobus_endpoint(host=host, workspace_id=workspace_id)
    if endpoint is None:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but the Zerobus endpoint could "
            "not be resolved for workspace_id=%r, host=%r. "
            "Falling back to the standard UC table exporter.",
            workspace_id,
            host,
        )
        return None

    # Derive the fully-qualified table name from the destination.
    table_name = _get_table_name_from_destination(destination)
    if not table_name:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but could not determine the "
            "UC table name from the trace destination %r. "
            "Falling back to the standard UC table exporter.",
            destination,
        )
        return None

    try:
        token_source = build_zerobus_token_source(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            workspace_id=workspace_id,
            tables=[table_name],
        )
    except Exception as exc:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but failed to build Zerobus "
            "token source: %s. Falling back to the standard UC table exporter.",
            exc,
        )
        return None

    _logger.debug("Zerobus span exporter configured: endpoint=%r, table=%r", endpoint, table_name)
    try:
        return DatabricksZerobusSpanExporter(
            tracking_uri=tracking_uri,
            endpoint=endpoint,
            token_source=token_source,
            table_name=table_name,
        )
    except Exception as exc:
        _logger.warning(
            "MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT is set but the Zerobus exporter could not be "
            "constructed: %s. Falling back to the standard UC table exporter.",
            exc,
        )
        return None


def _get_table_name_from_destination(destination) -> str | None:
    """Extract a fully-qualified ``catalog.schema.table`` name from a trace destination.

    Delegates to ``destination.full_otel_spans_table_name``, which both
    ``UCSchemaLocation`` and ``UnityCatalog`` expose.  Returns ``None`` when the
    property is not set (e.g. ``UnityCatalog`` before the backend populates it).
    """
    return getattr(destination, "full_otel_spans_table_name", None)
