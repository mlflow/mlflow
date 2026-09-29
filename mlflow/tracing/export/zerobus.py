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
from contextlib import nullcontext
from typing import Sequence

import requests
from opentelemetry.sdk.trace import ReadableSpan

from mlflow.entities.span import Span
from mlflow.environment_variables import MLFLOW_ENABLE_ZEROBUS_TRACE_EXPORT, MLFLOW_ZEROBUS_ENDPOINT
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.utils.otlp import build_otlp_export_request
from mlflow.utils.databricks_utils import get_databricks_host_creds

_logger = logging.getLogger(__name__)

# OTLP token path from databricks-sdk oauth module.
# We re-declare rather than import to avoid a hard dependency path at module level;
# the value is stable across SDK versions.
_OIDC_TOKEN_PATH = "/oidc/v1/token"

# Zerobus OTLP ingest path.
_ZEROBUS_OTLP_PATH = "/v1/traces"

# Header names expected by the Zerobus ingest service.
_HEADER_AUTHORIZATION = "Authorization"
_HEADER_TABLE_NAME = "x-databricks-zerobus-table-name"
_HEADER_CONTENT_TYPE = "Content-Type"

# Content type for OTLP protobuf payloads.
_CONTENT_TYPE_PROTOBUF = "application/x-protobuf"

# Timeout (seconds) for a single Zerobus OTLP POST.
_REQUEST_TIMEOUT_SECONDS = 30

# Max body bytes included in a span-export failure warning log.
_BODY_SNIPPET_BYTES = 512

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
    client_id: str | None = None,
    client_secret: str | None = None,
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

        # Pass the resolved SP credentials explicitly so the metastore lookup uses the same
        # identity the exporter mints its Zerobus token with, rather than the databricks-sdk
        # ambient chain (env vars / ~/.databrickscfg) which may not carry the creds supplied
        # only via the MLflow tracking URI.
        ws = WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret)
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

    Each batch is wrapped into MLflow ``Span`` objects and serialized with the
    same ``build_otlp_export_request`` helper the MLflow REST ``log_spans``
    path uses, then POSTed to ``https://{endpoint}/v1/traces`` with per-request
    auth headers. On a 401 the token source is force-refreshed (the fresh token
    is cached on the source when it supports it, so later exports reuse it) and
    the batch is retried once. Any other failure only logs a warning:
    span-export failures never propagate out of ``_export_spans_incrementally``,
    so the metadata export still runs.
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
        self._zerobus_url = f"https://{endpoint}{_ZEROBUS_OTLP_PATH}"
        # A single session for connection pooling. Auth headers are computed per request
        # (in `_post_spans`) rather than stored on the session, so a token refresh on one
        # thread can never race another thread's in-flight export.
        self._session = requests.Session()

    def _force_token_refresh(self):
        """Force-refresh the Zerobus token and cache it on the token source where possible.

        Returns the freshly minted token, which the caller must use for the
        immediate retry regardless of what the source's cache returns.

        The cache update is best effort and version tolerant: MLflow supports a
        ``databricks-sdk`` range whose ``Refreshable`` cache plumbing changed
        across releases (``_token`` assigned directly on older versions,
        replaced via ``_update_token`` from ~0.100), and custom token sources
        may have no cache at all. A failure to store the token only means later
        exports mint a new one instead of reusing this one; it never loses the
        retry.
        """
        token_source = self._token_source
        new_token = token_source.refresh()

        try:
            if hasattr(token_source, "_update_token"):
                # databricks-sdk >= ~0.100 (Refreshable._update_token): the SDK caches
                # the current token in ``_token`` and replaces it only via
                # ``_update_token`` inside ``token()``. Calling ``refresh()`` alone
                # mints a fresh token without updating that cache, so subsequent
                # ``token()`` calls would keep handing out the rejected one. Replicate
                # the SDK's own cache-update step under the same lock ``token()`` uses,
                # so concurrent ``token()`` callers serialize against this refresh
                # exactly as they do against the SDK's expiry-driven refreshes.
                with getattr(token_source, "_lock", nullcontext()):
                    token_source._update_token(new_token)
            elif hasattr(token_source, "_token"):
                # Older databricks-sdk Refreshable: the cache is the plain ``_token``
                # attribute, assigned under ``_lock`` by ``token()`` itself.
                with getattr(token_source, "_lock", nullcontext()):
                    token_source._token = new_token
            # else: the token source has no cache to update; later exports simply
            # mint a fresh token as usual.
        except Exception:
            _logger.debug(
                "Failed to cache the refreshed Zerobus token on the token source; "
                "later exports will mint a new one.",
                exc_info=True,
            )

        return new_token

    def _post_spans(self, payload: bytes, token=None) -> requests.Response:
        """POST one serialized OTLP request to Zerobus with fresh per-request headers.

        When *token* is None, one is minted from the token source. The 401 retry
        passes the token returned by ``_force_token_refresh`` explicitly so the
        retry always carries the freshly minted token, whatever the source's
        cache returns.
        """
        if token is None:
            token = self._token_source.token()
        return self._session.post(
            self._zerobus_url,
            data=payload,
            headers={
                _HEADER_AUTHORIZATION: f"Bearer {token.access_token}",
                _HEADER_CONTENT_TYPE: _CONTENT_TYPE_PROTOBUF,
                _HEADER_TABLE_NAME: self._table_name,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

    def _export_spans_to_zerobus(self, spans: Sequence[ReadableSpan]) -> None:
        """Send *spans* to Zerobus, retrying once on a 401 after forcing token refresh."""
        # Wrap the raw OTel spans in the MLflow Span interface: `Span.to_otel_proto` decodes
        # the JSON-encoded attribute values (e.g. mlflow.spanType), which the raw ReadableSpan
        # serialization of the stock OTLP exporter would send double-encoded.
        payload = build_otlp_export_request([Span(span) for span in spans]).SerializeToString()

        try:
            response = self._post_spans(payload)
        except Exception as exc:
            _logger.warning(
                "Zerobus span export request failed: %s. Spans were NOT exported to Zerobus.",
                exc,
            )
            return

        if response.status_code == 401:
            # The cached token was rejected (e.g. revoked, or minted for another audience):
            # force a refresh so this retry and all subsequent exports use a new token.
            _logger.debug("Zerobus rejected the token with HTTP 401; forcing token refresh.")
            try:
                response = self._post_spans(payload, self._force_token_refresh())
            except Exception as exc:
                _logger.warning(
                    "Zerobus span export failed after token refresh: %s. "
                    "Spans were NOT exported to Zerobus.",
                    exc,
                )
                return

        # NB: unlike the stock OTLP HTTP exporter, which retries 429/5xx with
        # backoff, we intentionally do not retry rate limits or server errors:
        # this runs on span-processing hot paths, and re-sending a batch to an
        # overloaded ingest only adds load. The batch is dropped with a warning.
        if not response.ok:
            _logger.warning(
                "Zerobus span export failed with HTTP %d: %r. Spans were NOT exported to Zerobus.",
                response.status_code,
                response.content[:_BODY_SNIPPET_BYTES],
            )

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """Override: send spans to Zerobus instead of the UC table REST path.

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
        try:
            super().shutdown()
        finally:
            # Close the HTTP session even if the parent shutdown fails, so the
            # underlying connections are not leaked.
            self._session.close()


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

    endpoint = resolve_zerobus_endpoint(
        host=host,
        workspace_id=workspace_id,
        client_id=client_id,
        client_secret=client_secret,
    )
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
