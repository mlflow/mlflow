"""
Databricks OTel collector span exporter for Unity Catalog trace destinations.

When the trace destination is a Unity Catalog location with service-principal
credentials, spans are sent directly to the Databricks OTel collector OTLP/HTTP
ingest endpoint instead of the MLflow REST ``log_spans`` API. Trace-level metadata
(TraceInfo) continues to flow through the standard MLflow backend path unchanged.

This path is on by default; set ``MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT=false``
to always use the MLflow tracing server path.

For how the OTLP ingestion endpoint and authentication are configured on the
Databricks side, see https://docs.databricks.com/aws/en/ingestion/opentelemetry/configure.

The collector endpoint is resolved lazily on the first span export rather than
during tracer initialization: ``_initialize_tracer_provider`` runs under the
tracer-provider ``Once`` lock, so a workspace metadata lookup there would block
every ``mlflow.tracing.reset()`` and the first traced request. Resolved endpoints
are memoized module-wide so provider re-initializations do not re-resolve them.
"""

import json
import logging
import os
import threading
from contextlib import nullcontext
from datetime import datetime, timedelta
from types import MethodType
from typing import Sequence

import requests
import urllib3.exceptions
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
from opentelemetry.sdk.trace import ReadableSpan

from mlflow.entities.span import Span
from mlflow.entities.trace_location import UnityCatalog
from mlflow.environment_variables import (
    MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT,
    MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT,
    MLFLOW_ENABLE_DB_SDK,
)
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH, build_otlp_export_request
from mlflow.utils.databricks_utils import _get_databricks_creds_config
from mlflow.utils.uri import get_db_info_from_uri

_logger = logging.getLogger(__name__)

# OTLP token path from databricks-sdk oauth module.
# We re-declare rather than import to avoid a hard dependency path at module level;
# the value is stable across SDK versions.
_OIDC_TOKEN_PATH = "/oidc/v1/token"

# Header names for the collector ingest requests.
_HEADER_AUTHORIZATION = "Authorization"
_HEADER_CONTENT_TYPE = "Content-Type"

# Content type for OTLP protobuf payloads.
_CONTENT_TYPE_PROTOBUF = "application/x-protobuf"

# Timeout (seconds) for a single collector OTLP POST.
_REQUEST_TIMEOUT_SECONDS = 30

# Timeout (seconds) for an OAuth token mint or refresh. The SDK's
# ``ClientCredentials`` implementation currently calls ``requests.post`` without a
# timeout, so collector export must provide its own bounded refresh function.
_TOKEN_REQUEST_TIMEOUT_SECONDS = 30

# Max body bytes included in a span-export failure warning log.
_BODY_SNIPPET_BYTES = 512

# HTTP statuses for which the collector rejected this batch but is still worth
# retrying for the next one (timeout, payload too large, rate limit): the batch
# is replayed over the REST path without pinning the exporter to it.
_REPLAYABLE_STATUS_CODES = (408, 413, 429)

# Timeout for each collector metadata request. SDK client construction can
# trigger OIDC discovery with the SDK's much longer default retry budget.
_SDK_HTTP_TIMEOUT_SECONDS = 10.0

# Collector endpoints resolved so far, keyed by (host, workspace_id). Resolution
# is deferred out of tracer initialization, so it is memoized module-wide to
# keep provider re-initialization (mlflow.tracing.reset) from re-resolving it.
_resolved_endpoints: dict[tuple[str, str], str] = {}

# Set once per process after warning about a collector config/resolution failure
# for a user who qualifies for the collector, so later span batches do not repeat
# the warning. Guarded by a lock because exports run on multiple threads.
_collector_config_failure_warned = False
_collector_config_failure_lock = threading.Lock()

# Wire-protocol literals required by the Databricks OTel collector ingest service; the
# exact strings below are mandated by the service and must not be changed unilaterally.
# Segment of the ingest hostname between the workspace ID and the region.
_COLLECTOR_HOST_SEGMENT = ".zerobus."
# OAuth resource for the table-scoped token the ingest service accepts.
_COLLECTOR_TOKEN_RESOURCE = "api://databricks/workspaces/{workspace_id}/zerobusDirectWriteApi"
# Header the ingest service requires to route a payload to the destination UC table.
_COLLECTOR_TABLE_NAME_HEADER = "x-databricks-zerobus-table-name"

# Domain suffixes that identify valid collector ingest hosts.
_VALID_COLLECTOR_HOST_SUFFIXES = (
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


def _normalize_collector_endpoint(endpoint: str) -> str:
    """Return a collector host in the form used by the URL builder."""
    return endpoint.removeprefix("https://")


def _retrieve_token_with_timeout(
    client_id: str,
    client_secret: str,
    token_url: str,
    params: dict[str, str],
    *,
    use_params: bool = False,
    use_header: bool = False,
):
    """Retrieve an OAuth token with a bounded HTTP request timeout."""
    from databricks.sdk.oauth import IgnoreNetrcAuth, Token

    request_params = dict(params)
    if use_params:
        if client_id:
            request_params["client_id"] = client_id
        if client_secret:
            request_params["client_secret"] = client_secret

    auth = (
        requests.auth.HTTPBasicAuth(client_id, client_secret) if use_header else IgnoreNetrcAuth()
    )
    response = requests.post(
        token_url,
        data=request_params,
        auth=auth,
        timeout=_TOKEN_REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            error = response.json()
            code = error.get("errorCode", error.get("error", "unknown"))
            summary = error.get("errorSummary", error.get("error_description", "unknown"))
            summary = summary.replace("\r\n", " ")
            raise ValueError(f"{code}: {summary}")
        raise ValueError(response.content)

    try:
        body = response.json()
        expires_in = int(body["expires_in"])
        expiry = datetime.now() + timedelta(seconds=expires_in)
        return Token(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token"),
            token_type=body["token_type"],
            expiry=expiry,
        )
    except Exception as exc:
        raise NotImplementedError(f"Not supported yet: {exc}") from exc


def _bounded_client_credentials_refresh(token_source):
    """Refresh a databricks-sdk ``ClientCredentials`` source with a timeout."""
    params = {
        "grant_type": "client_credentials",
        "scope": "all-apis",
        "authorization_details": token_source._mlflow_authorization_details,
    }
    if token_source.endpoint_params:
        params.update(token_source.endpoint_params)
    return _retrieve_token_with_timeout(
        token_source.client_id,
        token_source.client_secret,
        token_source.token_url,
        params,
        use_params=token_source.use_params,
        use_header=token_source.use_header,
    )


def is_databricks_otel_collector_host(endpoint: str, workspace_id: str) -> bool:
    """Validate that *endpoint* is a legitimate collector ingest host for *workspace_id*.

    Port of the Go validator. Rules:
    - Strip a leading ``https://``.
    - Reject if the remainder contains any of ``/?#@:``.
    - Must start with the workspace-scoped collector host prefix.
    - Must end with one of the known cloud domain suffixes.
    - Must have a non-empty region label between the host segment and the
      cloud domain suffix.

    Returns ``True`` when all rules pass, ``False`` otherwise.
    """
    host = endpoint
    host = host.removeprefix("https://")

    # Reject any host with authority-separator or path characters.
    if any(c in host for c in "/?#@:"):
        return False

    expected_prefix = f"{workspace_id}{_COLLECTOR_HOST_SEGMENT}"
    if not host.startswith(expected_prefix):
        return False

    # After the prefix the remainder must end with a known suffix, and there must be
    # at least one non-empty region segment between the prefix and that suffix.
    remainder = host[len(expected_prefix) :]
    matched_suffix = next(
        (suffix for suffix in _VALID_COLLECTOR_HOST_SUFFIXES if remainder.endswith(suffix)),
        None,
    )
    if matched_suffix is None:
        return False

    region_part = remainder[: -len(matched_suffix)]
    # Region must be non-empty and must not itself contain a trailing dot (i.e. not
    # an empty staging segment after the region).
    return bool(region_part) and not region_part.endswith(".")


def _get_metastore_summary(
    host: str,
    workspace_id: str,
    client_id: str | None,
    client_secret: str | None,
) -> dict[str, object]:
    """Read the workspace metastore summary using bounded HTTP requests."""
    if not (client_id and client_secret):
        raise ValueError("Service-principal credentials are required for collector discovery")

    token = _retrieve_token_with_timeout(
        client_id,
        client_secret,
        f"{host.rstrip('/')}{_OIDC_TOKEN_PATH}",
        {"grant_type": "client_credentials", "scope": "all-apis"},
        use_header=True,
    )
    response = requests.get(
        f"{host.rstrip('/')}/api/2.1/unity-catalog/metastore_summary",
        headers={
            "Accept": "application/json",
            "Authorization": f"{token.token_type} {token.access_token}",
            "X-Databricks-Workspace-Id": workspace_id,
        },
        timeout=_SDK_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    summary = response.json()
    if not isinstance(summary, dict):
        raise ValueError("Metastore summary is not a JSON object")
    return summary


def _warn_collector_config_failure(reason: str, *args) -> None:
    """Warn once per process about a collector config/resolution failure.

    Reaching endpoint resolution means the user qualifies for the collector: the
    feature flag is on, the destination is a ``UnityCatalog`` location, and
    service-principal credentials were found. A failure here prevents direct
    export and is worth surfacing at WARNING, even when the cause is transient.
    Warn only once per process; the exporter still falls back to REST.

    This is distinct from the ``_log_collector_unavailable`` path, which handles
    the high-volume "not applicable" case (no service-principal credentials:
    PAT/notebook users on the default-on path) and stays quiet by default.
    """
    global _collector_config_failure_warned
    message = reason + " Falling back to the MLflow tracing server span export path."
    with _collector_config_failure_lock:
        already_warned = _collector_config_failure_warned
        _collector_config_failure_warned = True
    if already_warned:
        _logger.debug(message, *args)
    else:
        _logger.warning(message, *args)


def resolve_databricks_otel_collector_endpoint(
    host: str,
    workspace_id: str,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> str | None:
    """Assemble and validate the collector ingest endpoint for a workspace.

    Precedence:
    1. ``MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT`` env-var override (validated,
       normalized to a host without ``https://``).
    2. Auto-assembled from workspace metastore metadata: the workspace ID, the
       collector host segment, the metastore region, an optional ``staging.``
       environment segment, and the cloud domain.

    Returns the host string (no ``https://`` prefix, no trailing slash) on success,
    or ``None`` on failure (logged at WARNING once per process).
    """
    # Explicit override takes highest precedence.
    if override := MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get():
        if is_databricks_otel_collector_host(override, workspace_id):
            return _normalize_collector_endpoint(override)
        _warn_collector_config_failure(
            "MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT override %r failed host validation "
            "for workspace_id=%r; ignoring override.",
            override,
            workspace_id,
        )
        return None

    try:
        summary = _get_metastore_summary(host, workspace_id, client_id, client_secret)
    except Exception as exc:
        _warn_collector_config_failure(
            "Failed to fetch metastore summary for collector endpoint resolution: %s", exc
        )
        return None

    region = summary.get("region")
    cloud_raw = summary.get("cloud")  # e.g. "aws", "azure", "gcp"

    if not region:
        _warn_collector_config_failure(
            "Metastore summary returned empty region; cannot resolve collector endpoint."
        )
        return None

    if not cloud_raw:
        # Attempt to parse cloud from global_metastore_id: "cloud:region:id"
        gid = str(summary.get("global_metastore_id") or "")
        parts = gid.split(":")
        cloud_raw = parts[0] if parts else ""

    cloud_raw = str(cloud_raw or "").lower()
    domain = _CLOUD_DOMAIN.get(cloud_raw)
    if not domain:
        _warn_collector_config_failure(
            "Unrecognised cloud %r from metastore summary; cannot resolve collector endpoint.",
            cloud_raw,
        )
        return None

    env_segment = "staging." if ".staging." in host else ""
    endpoint = f"{workspace_id}{_COLLECTOR_HOST_SEGMENT}{region}.{env_segment}{domain}"

    if not is_databricks_otel_collector_host(endpoint, workspace_id):
        _warn_collector_config_failure(
            "Assembled collector endpoint %r failed host validation.", endpoint
        )
        return None

    return _normalize_collector_endpoint(endpoint)


def build_table_authorization_details(tables: list[str]) -> str:
    """Return a JSON string encoding RFC 9396 authorization details for *tables*.

    For each fully-qualified ``catalog.schema.table`` name, three objects are
    emitted:

    - CATALOG ``cat`` - ``["USE CATALOG"]``
    - SCHEMA ``cat.sch`` - ``["USE SCHEMA"]``
    - TABLE ``cat.sch.tbl`` - ``["SELECT", "MODIFY"]``

    Keys: ``type``, ``object_type``, ``object_full_path``, ``privileges``.

    .. note::
        The exact ``type`` string ``"unity_catalog_privileges"`` used as the
        RFC 9396 ``type`` field value has been verified against a live
        Databricks workspace: the collector accepted a token minted with it.
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


def build_databricks_otel_collector_token_source(
    host: str,
    client_id: str,
    client_secret: str,
    workspace_id: str,
    tables: list[str],
):
    """Build a ``ClientCredentials`` token source for direct span writes to the collector.

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
    token_source = ClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url,
        endpoint_params={"resource": _COLLECTOR_TOKEN_RESOURCE.format(workspace_id=workspace_id)},
        use_header=True,
    )
    # ``scopes`` changed from a list to a string across supported SDK versions,
    # and older versions do not accept ``authorization_details``. The bounded
    # refresh supplies both in the form body regardless of SDK version.
    token_source._mlflow_authorization_details = build_table_authorization_details(tables)
    # ``ClientCredentials.token()`` retains the SDK's cache and locking
    # behavior, while its inherited ``refresh()`` uses an unbounded
    # ``requests.post`` in supported SDK versions. Replace only the refresh
    # implementation on this instance so the collector path has a bounded
    # token request without changing SDK behavior process-wide.
    token_source.refresh = MethodType(_bounded_client_credentials_refresh, token_source)
    return token_source


def _is_connection_not_established(exc: BaseException) -> bool:
    """Whether *exc* proves a request was never delivered to the collector.

    ``requests.ConnectTimeout`` fails while the TCP connection is still being
    established, and a ``requests.ConnectionError`` carrying a
    ``urllib3.exceptions.NewConnectionError`` (DNS and TCP connection failures;
    ``NameResolutionError`` subclasses it) fails before any request bytes are
    sent - in both cases the collector definitely never received the request.

    Any other ``ConnectionError`` (e.g. a reset after the request was sent) does
    NOT prove that: the request may have been delivered and ingested before the
    connection failed.
    """
    if isinstance(exc, requests.ConnectTimeout):
        return True
    if not isinstance(exc, requests.ConnectionError):
        return False

    # requests usually wraps urllib3's MaxRetryError, whose underlying failure
    # is stored in ``reason`` rather than ``args`` or the exception chain.
    # Other wrappers may retain the failure in their args or cause/context.
    # The ``seen`` set guards against cycles in those chains.
    stack = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, urllib3.exceptions.NewConnectionError):
            return True
        if isinstance(current, urllib3.exceptions.MaxRetryError) and isinstance(
            current.reason, BaseException
        ):
            stack.append(current.reason)
        stack.extend(arg for arg in current.args if isinstance(arg, BaseException))
        stack.extend(
            chained for chained in (current.__cause__, current.__context__) if chained is not None
        )
    return False


def _resolve_collector_credentials(tracking_uri: str | None):
    """Resolve local service-principal settings and a workspace ID on first export."""
    if not MLFLOW_ENABLE_DB_SDK.get():
        _logger.debug(
            "MLFLOW_ENABLE_DB_SDK is false; skipping Databricks OTel collector credential "
            "discovery."
        )
        return None

    try:
        # MLflow's existing credential provider reads the tracking URI's profile
        # or environment locally. SDK Config can initiate an OIDC discovery call
        # with its own default retry budget while constructing SP auth, even when
        # its configured host-metadata timeout is short.
        config = _get_databricks_creds_config(tracking_uri)
        profile, _ = get_db_info_from_uri(tracking_uri or "")
    except Exception as exc:
        _logger.debug(
            "Failed to resolve local Databricks credentials for collector export: %s", exc
        )
        return None

    host = getattr(config, "host", None)
    client_id = getattr(config, "client_id", None)
    client_secret = getattr(config, "client_secret", None)
    auth_type = getattr(config, "auth_type", None)
    if auth_type and auth_type != "oauth-m2m":
        _logger.debug("Databricks profile selects %s authentication; using REST export.", auth_type)
        return None
    if auth_type != "oauth-m2m" and (
        getattr(config, "token", None)
        or (getattr(config, "username", None) and getattr(config, "password", None))
    ):
        _logger.debug("Databricks credentials include another auth method; using REST export.")
        return None
    if not (host and client_id and client_secret):
        _logger.debug(
            "Local Databricks credentials do not contain a host and service-principal pair."
        )
        return None

    # The ambient workspace ID is applicable only when its host matches the
    # tracking URI's selected profile. A different profile can point at another
    # workspace, including one behind a shared workspace host.
    workspace_id = (
        os.environ.get("DATABRICKS_WORKSPACE_ID")
        if not (profile or os.environ.get("DATABRICKS_CONFIG_PROFILE"))
        and os.environ.get("DATABRICKS_HOST", "").rstrip("/") == host.rstrip("/")
        else None
    )
    if not workspace_id:
        try:
            response = requests.get(
                f"{host.rstrip('/')}/.well-known/databricks-config",
                timeout=_SDK_HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            workspace_id = response.json().get("workspace_id")
        except Exception as exc:
            _logger.debug("Failed to resolve workspace ID for collector export: %s", exc)
            return None

    if not workspace_id:
        _logger.debug("Workspace ID is unavailable for collector export.")
        return None

    return host, str(workspace_id), client_id, client_secret


class DatabricksOtelCollectorSpanExporter(DatabricksUCTableSpanExporter):
    """Span exporter that sends spans to the Databricks OTel collector via OTLP/HTTP protobuf.

    Trace-level metadata (TraceInfo) continues to flow through the inherited
    MLflow REST backend path (``_export_traces`` / ``_log_trace`` /
    ``client.start_trace``). Only the span data is redirected to the collector.

    Unless an explicit endpoint was provided at construction, the collector
    endpoint is resolved lazily on the first export (a bounded metastore summary
    request) and cached module-wide, so tracer initialization performs no network
    I/O on this path.

    Each batch is wrapped into MLflow ``Span`` objects and serialized with the
    same ``build_otlp_export_request`` helper the MLflow REST ``log_spans``
    path uses, then POSTed to the collector ingest URL with per-request auth
    headers. On a 401 the token source is force-refreshed (the fresh token is
    cached on the source when it supports it, so later exports reuse it) and
    the batch is retried once.

    Export failures are classified by whether the collector can still have
    ingested the batch:

    - Definitive rejection (4xx other than 401/408/413/429; a 401 that still
      fails after the refresh retry; lazy endpoint resolution failing; a request
      that provably never reached the collector): the batch is replayed over
      the inherited tracing-server REST path and the exporter is pinned to that
      path for all later batches, with one warning per exporter.
    - Transient or batch-specific rejection (408/413/429; a token mint failure
      before any POST was sent): the batch is replayed over the REST path, but
      the next batch still goes to the collector.
    - Ambiguous delivery (5xx; read timeouts; a connection failure after the
      request may have been sent): the batch is dropped with a warning, because
      replaying it could duplicate spans. Later batches use the REST path.

    No span-export failure ever propagates out of ``_export_spans_incrementally``,
    so the metadata export still runs.
    """

    def __init__(
        self,
        tracking_uri: str | None,
        token_source,
        table_name: str,
        host: str | None = None,
        workspace_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(tracking_uri=tracking_uri)
        # When *token_source*, *host*, or *workspace_id* are omitted, all
        # credential and workspace discovery is deferred to the first collector
        # batch. This keeps tracer initialization free of SDK metadata probes.
        self._tracking_uri = tracking_uri
        self._endpoint_override = _normalize_collector_endpoint(endpoint) if endpoint else None
        self._collector_endpoint = self._endpoint_override if token_source is not None else None
        self._collector_url = (
            f"https://{self._collector_endpoint}{OTLP_TRACES_PATH}"
            if self._collector_endpoint
            else None
        )
        self._host = host
        self._workspace_id = str(workspace_id or "")
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_source = token_source
        self._table_name = table_name
        # Guards the lazy endpoint resolution so it runs at most once per
        # exporter even with concurrent exporting threads.
        self._resolution_lock = threading.Lock()
        self._collector_config_failed = False
        # Set when a qualified-user config failure already emitted its single
        # WARNING via ``_warn_collector_config_failure``, so the endpoint-unavailable
        # path in ``_export_spans_to_collector`` does not also log for the same
        # failure. Stays False for the not-applicable path (missing SP creds), which
        # keeps the quiet ``_log_collector_unavailable`` behavior.
        self._collector_config_warned = False
        # Set when the collector path is unusable or a batch's delivery is
        # uncertain. Later batches go straight to the inherited tracing-server
        # REST path without contacting the collector again.
        self._collector_rejected = False
        # Set after the first ambiguous-delivery warning so later ones log at
        # DEBUG (mirrors ``_has_raised_span_export_error`` in ``uc_table``).
        self._has_warned_ambiguous_drop = False
        # A single session for connection pooling. Auth headers are computed per request
        # (in `_post_spans`) rather than stored on the session, so a token refresh on one
        # thread can never race another thread's in-flight export.
        self._session = requests.Session()

        # Reuse the inherited batcher so the shared ``flush_exporter`` utility
        # drains collector batches during provider flush and retirement. Its
        # callback dispatches to the collector while that path is active and to
        # the parent REST logger after a permanent fallback. Keep this alias for
        # callers that inspect the collector batcher explicitly.
        self._collector_span_batcher = getattr(self, "_span_batcher", None)
        if self._collector_span_batcher is not None:
            self._collector_span_batcher._log_spans_func = self._export_batch

    def _ensure_collector_endpoint(self) -> str | None:
        """Return the collector endpoint, resolving it lazily on first use.

        The resolved endpoint is memoized on the exporter and in the module-level
        cache keyed by ``(host, workspace_id)``, so resolution runs at most once
        per exporter and once per workspace per process.
        """
        if self._collector_endpoint is not None and self._token_source is not None:
            return self._collector_endpoint
        if self._collector_config_failed:
            return None
        with self._resolution_lock:
            # Another export thread may have resolved the endpoint while this
            # one waited on the lock.
            if self._collector_endpoint is not None and self._token_source is not None:
                return self._collector_endpoint

            if not (self._host and self._workspace_id) or (
                self._token_source is None and not (self._client_id and self._client_secret)
            ):
                credentials = _resolve_collector_credentials(self._tracking_uri)
                if credentials is None:
                    self._collector_config_failed = True
                    return None
                self._host, self._workspace_id, self._client_id, self._client_secret = credentials

            if self._token_source is None:
                try:
                    self._token_source = build_databricks_otel_collector_token_source(
                        host=self._host,
                        client_id=self._client_id,
                        client_secret=self._client_secret,
                        workspace_id=self._workspace_id,
                        tables=[self._table_name],
                    )
                except Exception as exc:
                    # Credentials were found, so this user qualifies for the collector:
                    # a token-source build failure is a real misconfiguration.
                    _warn_collector_config_failure(
                        "Failed to build the Databricks OTel collector token source: %s", exc
                    )
                    self._collector_config_failed = True
                    self._collector_config_warned = True
                    return None

            override = self._endpoint_override or MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get()
            if override:
                if not is_databricks_otel_collector_host(override, self._workspace_id):
                    _warn_collector_config_failure(
                        "MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT override %r failed host "
                        "validation for workspace_id=%r; ignoring override.",
                        override,
                        self._workspace_id,
                    )
                    self._collector_config_failed = True
                    self._collector_config_warned = True
                    return None
                self._collector_endpoint = _normalize_collector_endpoint(override)
                self._collector_url = f"https://{self._collector_endpoint}{OTLP_TRACES_PATH}"
                return self._collector_endpoint

            cache_key = (self._host, self._workspace_id)
            if (cached := _resolved_endpoints.get(cache_key)) is not None:
                self._collector_endpoint = _normalize_collector_endpoint(cached)
            else:
                resolved = resolve_databricks_otel_collector_endpoint(
                    host=self._host,
                    workspace_id=self._workspace_id,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                )
                if resolved is None:
                    self._collector_config_failed = True
                    # resolve_* already emitted the single WARNING for the specific
                    # reason; do not also log for the endpoint-unavailable case.
                    self._collector_config_warned = True
                    return None
                normalized = _normalize_collector_endpoint(resolved)
                _resolved_endpoints[cache_key] = normalized
                self._collector_endpoint = normalized
            self._collector_url = f"https://{self._collector_endpoint}{OTLP_TRACES_PATH}"
        return self._collector_endpoint

    def _force_token_refresh(self):
        """Force-refresh the collector token and cache it on the token source where possible.

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
                "Failed to cache the refreshed collector token on the token source; "
                "later exports will mint a new one.",
                exc_info=True,
            )

        return new_token

    def _post_spans(self, payload: bytes, token) -> requests.Response:
        """POST one serialized OTLP request to the collector with per-request headers.

        The token must be minted by the caller so a mint failure can be told
        apart from a request failure: a mint failure happens before any bytes
        reach the collector, while a request failure may not.
        """
        return self._session.post(
            self._collector_url,
            data=payload,
            headers={
                _HEADER_AUTHORIZATION: f"Bearer {token.access_token}",
                _HEADER_CONTENT_TYPE: _CONTENT_TYPE_PROTOBUF,
                _COLLECTOR_TABLE_NAME_HEADER: self._table_name,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _as_mlflow_spans(spans: Sequence[ReadableSpan | Span]) -> list[Span]:
        return [span if isinstance(span, Span) else Span(span) for span in spans]

    def _send_spans_via_rest(
        self,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        """Send *spans* to the UC table via the inherited tracing-server REST path.

        Replicates ``DatabricksUCTableSpanExporter._export_spans_incrementally``
        for the table captured at exporter init (``self._table_name``) instead of
        the active spans table: a REST replay must target the same destination
        the collector batch was destined for, even if the thread-local active
        table has since changed or is unset. Async batching follows the parent
        behavior (``SpanBatcher`` when async logging is enabled). A collector
        batch worker calls the parent logger directly so the batch cannot be
        routed back into the collector callback.
        """
        spans = self._as_mlflow_spans(spans)
        if self._should_log_async() and not from_collector_batch:
            for span in spans:
                self._span_batcher.add_span(location=self._table_name, span=span)
        elif from_collector_batch:
            super()._log_spans(self._table_name, spans)
        else:
            self._log_spans(self._table_name, spans)

    def _fall_back_to_rest(
        self,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        """Replay a definitively rejected batch over the REST path, permanently.

        Logs one warning per exporter, then pins the exporter to the REST path so
        a rejecting collector is not retried for every subsequent export.
        """
        if not self._collector_rejected:
            self._collector_rejected = True
            _logger.warning(
                "The Databricks OTel collector rejected a span export. MLflow is falling "
                "back to the tracing server path for this and all subsequent span batches."
            )
        self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)

    def _replay_batch_via_rest(
        self,
        spans: Sequence[ReadableSpan | Span],
        reason: str,
        *args,
        from_collector_batch: bool = False,
    ) -> None:
        """Replay this batch over the REST path without abandoning the collector.

        For transient or batch-specific rejections (408/413/429, a token mint
        failure before any POST was sent): the batch was not ingested, so
        replaying it cannot duplicate spans, and the next batch still goes to
        the collector.
        """
        _logger.debug(reason, *args)
        self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)

    def _log_ambiguous_drop(self, reason: str, *args) -> None:
        """Drop an uncertain batch and route later batches through REST.

        Whether the collector ingested the batch before failing is unknown, so
        the batch is dropped rather than replayed over the REST path, which
        could duplicate the spans. Future batches have not been sent yet and
        can safely use REST. Warn loudly the first time so the loss is
        discoverable (mirrors ``_has_raised_span_export_error`` in
        ``DatabricksUCTableSpanExporter``).
        """
        self._collector_rejected = True
        message = reason + (
            " Delivery is ambiguous, so the spans were dropped instead of replayed over "
            "the MLflow tracing server path, which could duplicate them. Future span "
            "batches will use the tracing server path."
        )
        if self._has_warned_ambiguous_drop:
            _logger.debug(message, *args)
        else:
            _logger.warning(message, *args)
            self._has_warned_ambiguous_drop = True

    def _handle_post_exception(
        self,
        exc: BaseException,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        if _is_connection_not_established(exc):
            # The request was never delivered, so the batch was definitely not
            # ingested and replaying it over the REST path cannot duplicate spans.
            self._fall_back_to_rest(spans, from_collector_batch=from_collector_batch)
        else:
            # Read timeouts and mid-request connection failures (e.g. a reset after
            # the request was sent) leave delivery ambiguous.
            self._log_ambiguous_drop(
                "The Databricks OTel collector span export request failed: %s.", exc
            )

    def _handle_collector_response(
        self,
        response: requests.Response,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool,
    ) -> None:
        """Handle an HTTP 200 OTLP response, including partial success."""
        if not response.content:
            return

        response_message = ExportTraceServiceResponse()
        try:
            response_message.ParseFromString(response.content)
        except Exception as exc:
            # The collector may return an empty body for success, but a malformed
            # non-empty body leaves delivery status unknown. Do not replay it.
            self._log_ambiguous_drop(
                "The Databricks OTel collector returned an invalid HTTP 200 response: %s.",
                exc,
            )
            return

        rejected_spans = response_message.partial_success.rejected_spans
        if rejected_spans == 0:
            return

        if rejected_spans < 0 or rejected_spans > len(spans):
            self._log_ambiguous_drop(
                "The Databricks OTel collector reported %d rejected spans for a batch of %d spans.",
                rejected_spans,
                len(spans),
            )
            return

        if rejected_spans == len(spans):
            self._collector_rejected = True
            _logger.warning(
                "The Databricks OTel collector rejected all %d spans in a successful HTTP 200 "
                "partial response (%s). Replaying this batch via the MLflow tracing server "
                "path and using that path for future batches.",
                len(spans),
                response_message.partial_success.error_message,
            )
            self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)
            return

        # The response does not identify which spans were rejected. Replaying
        # would duplicate spans that were accepted, so pin only future batches
        # to REST and leave this partially accepted batch untouched.
        self._collector_rejected = True
        _logger.warning(
            "The Databricks OTel collector partially accepted a span batch: %d of %d spans "
            "were rejected (%s). Future batches will use the MLflow tracing server path.",
            rejected_spans,
            len(spans),
            response_message.partial_success.error_message,
        )

    def _export_spans_to_collector(
        self,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        """Send *spans* to the collector, retrying once on a 401 after forcing token refresh."""
        endpoint = self._ensure_collector_endpoint()
        if endpoint is None:
            # Resolution failed once; do not retry it for later batches either:
            # pin the REST path for this and all subsequent span batches.
            self._collector_rejected = True
            if not self._collector_config_warned:
                # Not-applicable path (no service-principal credentials: PAT/notebook
                # users on the default-on path). This is the high-volume majority, so
                # it stays quiet by default (DEBUG unless the env var is explicitly
                # set). Qualified-user config failures were already surfaced once at
                # WARNING by the specific failure site, which set the flag above.
                _log_collector_unavailable(
                    "The Databricks OTel collector endpoint could not be resolved for "
                    "workspace_id=%r, host=%r.",
                    self._workspace_id,
                    self._host,
                )
            self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)
            return

        # Wrap the raw OTel spans in the MLflow Span interface: `Span.to_otel_proto` decodes
        # the JSON-encoded attribute values (e.g. mlflow.spanType), which the raw ReadableSpan
        # serialization of the stock OTLP exporter would send double-encoded.
        mlflow_spans = self._as_mlflow_spans(spans)
        payload = build_otlp_export_request(mlflow_spans).SerializeToString()

        # Mint the token before the POST: a mint failure means nothing reached the
        # collector, so replaying this batch cannot duplicate spans.
        try:
            token = self._token_source.token()
        except Exception as exc:
            self._replay_batch_via_rest(
                spans,
                "Minting the Databricks OTel collector token failed: %s. Replaying this "
                "batch via the MLflow tracing server path.",
                exc,
                from_collector_batch=from_collector_batch,
            )
            return

        try:
            response = self._post_spans(payload, token)
        except Exception as exc:
            self._handle_post_exception(exc, spans, from_collector_batch=from_collector_batch)
            return

        if response.status_code == 401:
            # The cached token was rejected (e.g. revoked, or minted for another audience):
            # force a refresh so this retry and all subsequent exports use a new token.
            _logger.debug("The collector rejected the token with HTTP 401; forcing token refresh.")
            try:
                refreshed = self._force_token_refresh()
            except Exception:
                # The 401 definitively rejected the batch, and the token cannot be
                # refreshed, so the collector path is unusable.
                self._fall_back_to_rest(spans, from_collector_batch=from_collector_batch)
                return
            try:
                response = self._post_spans(payload, refreshed)
            except Exception as exc:
                self._handle_post_exception(exc, spans, from_collector_batch=from_collector_batch)
                return

        # NB: unlike the stock OTLP HTTP exporter, which retries 429/5xx with backoff, we
        # intentionally do not retry rate limits or server errors: this runs on
        # span-processing hot paths, and re-sending a batch to an overloaded ingest only
        # adds load.
        if response.ok:
            if response.status_code == 200:
                self._handle_collector_response(
                    response,
                    spans,
                    from_collector_batch=from_collector_batch,
                )
            return

        if 400 <= response.status_code < 500:
            # A 4xx means the collector rejected the batch without ingesting it, so
            # replaying it over the REST path cannot duplicate spans.
            if response.status_code in _REPLAYABLE_STATUS_CODES:
                self._replay_batch_via_rest(
                    spans,
                    "The Databricks OTel collector returned HTTP %d for a span export "
                    "(transient or batch-specific rejection). Replaying this batch via the "
                    "MLflow tracing server path.",
                    response.status_code,
                    from_collector_batch=from_collector_batch,
                )
            else:
                self._fall_back_to_rest(spans, from_collector_batch=from_collector_batch)
            return

        # Even retryable OTLP 5xx statuses (502/503/504) do not prove that the
        # collector failed to ingest the batch. Replaying through a different
        # sink could duplicate spans; later batches can safely use REST.
        # See https://opentelemetry.io/docs/specs/otlp/#duplicate-data.
        self._log_ambiguous_drop(
            "The Databricks OTel collector span export failed with HTTP %d: %r.",
            response.status_code,
            response.content[:_BODY_SNIPPET_BYTES],
        )

    def _export_batch(self, location: str, spans: list[Span]) -> None:
        if self._collector_rejected:
            # This callback runs in the inherited batch worker. Calling the
            # parent logger directly avoids re-enqueuing the batch into this
            # same dispatcher and recursively trying the collector.
            super()._log_spans(location, spans)
        else:
            self._export_spans_to_collector(spans, from_collector_batch=True)

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """Override: send spans to the collector instead of the UC table REST path.

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
            if self._collector_rejected:
                # A prior collector batch failed or endpoint resolution failed;
                # use the tracing-server REST path directly.
                self._send_spans_via_rest(spans)
            elif self._collector_span_batcher and self._should_log_async():
                # ``SpanBatcher`` owns the collector worker and performs the
                # actual network call away from the traced application thread.
                for span in self._as_mlflow_spans(spans):
                    self._collector_span_batcher.add_span(
                        location=self._table_name,
                        span=span,
                    )
            else:
                self._export_spans_to_collector(spans)
        except Exception as exc:
            _logger.warning(
                "Databricks OTel collector span export failed: %s. Spans were NOT exported "
                "to the collector; trace metadata export via the MLflow backend is unaffected.",
                exc,
            )

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            try:
                # The provider may call shutdown() directly. Drain any partial
                # span batch and queued metadata before closing the session.
                self.flush(terminate=True)
            finally:
                self._session.close()


def _log_collector_unavailable(reason: str, *args) -> None:
    """Log why the collector export path is not used, at a level matching user intent.

    The collector path is on by default, so a missing prerequisite is expected in
    many environments and logs at DEBUG. When the user explicitly set
    ``MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT``, log at WARNING so the
    explicit opt-in still learns why the fallback occurred.
    """
    message = reason + " Falling back to the MLflow tracing server span export path."
    if MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.is_set():
        _logger.warning(message, *args)
    else:
        _logger.debug(message, *args)


def get_databricks_otel_collector_span_exporter(
    destination,
    tracking_uri: str | None,
) -> DatabricksOtelCollectorSpanExporter | None:
    """Return a ``DatabricksOtelCollectorSpanExporter`` when all prerequisites are met.

    This runs inside tracer initialization, so it performs only local checks and
    defers all credential, workspace, and endpoint lookups. Network lookups run
    with bounded HTTP timeouts on the first collector batch; failures fall back
    to the inherited REST span path.

    Returns ``None`` (so the caller falls back to the standard UC table path) when:
    - ``MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT`` is ``False``.
    - The destination is not a ``UnityCatalog`` location.
    - The UC table name cannot be derived from the destination.

    Args:
        destination: The trace destination; only ``UnityCatalog`` locations are eligible.
        tracking_uri: MLflow tracking URI to resolve Databricks credentials from.

    Returns:
        A configured ``DatabricksOtelCollectorSpanExporter``, or ``None``.
    """
    if not MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.get():
        return None

    # UCSchemaLocation destinations use deprecated v1 span tables (map<string,string>
    # attributes) which the collector rejects with a 400 schema error; only UnityCatalog
    # destinations have the v2 span tables (VARIANT attributes) it accepts.
    if not isinstance(destination, UnityCatalog):
        return None

    # Derive the fully-qualified table name from the destination.
    table_name = _get_table_name_from_destination(destination)
    if not table_name:
        _log_collector_unavailable(
            "Could not determine the UC table name from the trace destination %r.",
            destination,
        )
        return None

    # Keep the override as a local constructor value so accepted
    # ``https://`` forms are normalized before URL construction. Workspace
    # scoped validation still happens after deferred credential discovery.
    endpoint = (
        _normalize_collector_endpoint(MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get())
        if MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get()
        else None
    )

    _logger.debug(
        "Databricks OTel collector span exporter configured with deferred credential and "
        "endpoint resolution: table=%r",
        table_name,
    )
    try:
        return DatabricksOtelCollectorSpanExporter(
            tracking_uri=tracking_uri,
            token_source=None,
            table_name=table_name,
            endpoint=endpoint,
        )
    except Exception as exc:
        _log_collector_unavailable(
            "The Databricks OTel collector span exporter could not be constructed: %s.",
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
