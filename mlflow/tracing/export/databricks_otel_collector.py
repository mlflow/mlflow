"""
Databricks OTel collector span exporter for Unity Catalog trace destinations.

When the trace destination is a Unity Catalog location with service-principal
credentials, spans are sent directly to the Databricks OTel collector OTLP/HTTP
ingest endpoint instead of the MLflow REST ``log_spans`` API. Trace-level metadata
(TraceInfo) continues to flow through the standard MLflow backend path unchanged.

This path is on by default; set ``MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT=false``
to always use the MLflow tracing server path.

The collector endpoint is resolved lazily on the first span export rather than
during tracer initialization: ``_initialize_tracer_provider`` runs under the
tracer-provider ``Once`` lock, so a workspace metadata lookup there would block
every ``mlflow.tracing.reset()`` and the first traced request. Resolved endpoints
are memoized module-wide so provider re-initializations do not re-resolve them.
"""

import json
import logging
import threading
from contextlib import nullcontext
from typing import Sequence

import requests
import urllib3.exceptions
from opentelemetry.sdk.trace import ReadableSpan

from mlflow.entities.span import Span
from mlflow.entities.trace_location import UnityCatalog
from mlflow.environment_variables import (
    MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT,
    MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT,
)
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH, build_otlp_export_request
from mlflow.utils.databricks_utils import get_databricks_host_creds

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

# Max body bytes included in a span-export failure warning log.
_BODY_SNIPPET_BYTES = 512

# HTTP statuses for which the collector rejected this batch but is still worth
# retrying for the next one (timeout, payload too large, rate limit): the batch
# is replayed over the REST path without pinning the exporter to it.
_REPLAYABLE_STATUS_CODES = (408, 413, 429)

# Timeouts for the lazy endpoint-resolution client. The databricks-sdk defaults
# (60s per request, 300s retry budget) would stall the first span export for
# minutes when the workspace host is unreachable, so resolution runs with an
# explicitly bounded budget instead.
_SDK_HTTP_TIMEOUT_SECONDS = 10.0
_SDK_RETRY_TIMEOUT_SECONDS = 15

# Collector endpoints resolved so far, keyed by (host, workspace_id). Resolution
# is deferred out of tracer initialization, so it is memoized module-wide to
# keep provider re-initialization (mlflow.tracing.reset) from re-resolving it.
_resolved_endpoints: dict[tuple[str, str], str] = {}

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


def _build_workspace_client(
    host: str,
    workspace_id: str,
    client_id: str | None,
    client_secret: str | None,
):
    """Build a short-timeout ``WorkspaceClient`` for the metastore summary lookup.

    The client is built from an explicit ``Config`` so the lookup runs with a
    bounded budget instead of the SDK defaults (60s per request, 300s retry
    budget). Passing ``workspace_id`` pins the workspace on the config so its
    resolution does not depend on the SDK's host-metadata probe, and the SP
    credentials are passed explicitly so the lookup uses the same identity the
    exporter mints its collector token with, rather than the SDK's ambient chain
    (env vars / ~/.databrickscfg) which may not carry the creds supplied only via
    the MLflow tracking URI.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.core import Config

    config = Config(
        host=host,
        client_id=client_id,
        client_secret=client_secret,
        workspace_id=workspace_id,
        http_timeout_seconds=_SDK_HTTP_TIMEOUT_SECONDS,
        retry_timeout_seconds=_SDK_RETRY_TIMEOUT_SECONDS,
    )
    return WorkspaceClient(config=config)


def resolve_databricks_otel_collector_endpoint(
    host: str,
    workspace_id: str,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> str | None:
    """Assemble and validate the collector ingest endpoint for a workspace.

    Precedence:
    1. ``MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT`` env-var override (validated,
       returned as-is).
    2. Auto-assembled from workspace metastore metadata: the workspace ID, the
       collector host segment, the metastore region, an optional ``staging.``
       environment segment, and the cloud domain.

    Returns the host string (no ``https://`` prefix, no trailing slash) on success,
    or ``None`` on any failure (logged at DEBUG).
    """
    # Explicit override takes highest precedence.
    if override := MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get():
        if is_databricks_otel_collector_host(override, workspace_id):
            return override
        _logger.debug(
            "MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT override %r failed host validation "
            "for workspace_id=%r; ignoring override.",
            override,
            workspace_id,
        )
        return None

    ws = None
    try:
        ws = _build_workspace_client(host, workspace_id, client_id, client_secret)
        summary = ws.metastores.summary()
    except Exception as exc:
        _logger.debug(
            "Failed to fetch metastore summary for collector endpoint resolution: %s", exc
        )
        return None
    finally:
        # The client is only needed for this one lookup; release its resources.
        if ws is not None and (close := getattr(ws, "close", None)) is not None:
            close()

    region = summary.region
    cloud_raw = summary.cloud  # e.g. "aws", "azure", "gcp"

    if not region:
        _logger.debug("Metastore summary returned empty region; cannot resolve endpoint.")
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
            "Unrecognised cloud %r from metastore summary; cannot resolve endpoint.",
            cloud_raw,
        )
        return None

    env_segment = "staging." if ".staging." in host else ""
    endpoint = f"{workspace_id}{_COLLECTOR_HOST_SEGMENT}{region}.{env_segment}{domain}"

    if not is_databricks_otel_collector_host(endpoint, workspace_id):
        _logger.debug("Assembled collector endpoint %r failed host validation.", endpoint)
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
    return ClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url,
        scopes="all-apis",
        endpoint_params={"resource": _COLLECTOR_TOKEN_RESOURCE.format(workspace_id=workspace_id)},
        authorization_details=build_table_authorization_details(tables),
        use_header=True,
    )


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

    # requests wraps the underlying urllib3 exception as the first positional
    # argument (``raise ConnectionError(e)``), and lower layers may instead nest
    # it in the cause/context chain; walk both until a NewConnectionError turns
    # up. The ``seen`` set guards against cycles in the cause/context chain.
    stack = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, urllib3.exceptions.NewConnectionError):
            return True
        stack.extend(arg for arg in current.args if isinstance(arg, BaseException))
        stack.extend(
            chained for chained in (current.__cause__, current.__context__) if chained is not None
        )
    return False


class DatabricksOtelCollectorSpanExporter(DatabricksUCTableSpanExporter):
    """Span exporter that sends spans to the Databricks OTel collector via OTLP/HTTP protobuf.

    Trace-level metadata (TraceInfo) continues to flow through the inherited
    MLflow REST backend path (``_export_traces`` / ``_log_trace`` /
    ``client.start_trace``). Only the span data is redirected to the collector.

    Unless an explicit endpoint was provided at construction, the collector
    endpoint is resolved lazily on the first export (a bounded metastores.summary
    lookup) and cached module-wide, so tracer initialization performs no network
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
      request may have been sent): the batch is dropped with a warning (once
      per exporter, then debug), because replaying it could duplicate spans.

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
        # When *endpoint* is None it is resolved lazily on the first export from
        # (*host*, *workspace_id*) using the SP credentials, and never during
        # tracer initialization.
        self._collector_endpoint = endpoint
        self._collector_url = f"https://{endpoint}{OTLP_TRACES_PATH}" if endpoint else None
        self._host = host
        self._workspace_id = workspace_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_source = token_source
        self._table_name = table_name
        # Guards the lazy endpoint resolution so it runs at most once per
        # exporter even with concurrent exporting threads.
        self._resolution_lock = threading.Lock()
        # Set when the collector path is definitively unusable (definitive
        # rejection or failed endpoint resolution); all later batches go straight
        # to the inherited tracing-server REST path without contacting the
        # collector again.
        self._collector_rejected = False
        # Set after the first ambiguous-delivery warning so later ones log at
        # DEBUG (mirrors ``_has_raised_span_export_error`` in ``uc_table``).
        self._has_warned_ambiguous_drop = False
        # A single session for connection pooling. Auth headers are computed per request
        # (in `_post_spans`) rather than stored on the session, so a token refresh on one
        # thread can never race another thread's in-flight export.
        self._session = requests.Session()

    def _ensure_collector_endpoint(self) -> str | None:
        """Return the collector endpoint, resolving it lazily on first use.

        The resolved endpoint is memoized on the exporter and in the module-level
        cache keyed by ``(host, workspace_id)``, so resolution runs at most once
        per exporter and once per workspace per process.
        """
        if self._collector_endpoint is not None:
            return self._collector_endpoint
        with self._resolution_lock:
            # Another export thread may have resolved the endpoint while this
            # one waited on the lock.
            if self._collector_endpoint is not None:
                return self._collector_endpoint
            cache_key = (self._host, self._workspace_id)
            if (cached := _resolved_endpoints.get(cache_key)) is not None:
                self._collector_endpoint = cached
            else:
                resolved = resolve_databricks_otel_collector_endpoint(
                    host=self._host,
                    workspace_id=self._workspace_id,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                )
                if resolved is None:
                    return None
                _resolved_endpoints[cache_key] = resolved
                self._collector_endpoint = resolved
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

    def _send_spans_via_rest(self, spans: Sequence[ReadableSpan]) -> None:
        """Send *spans* to the UC table via the inherited tracing-server REST path.

        Replicates ``DatabricksUCTableSpanExporter._export_spans_incrementally``
        for the table captured at exporter init (``self._table_name``) instead of
        the active spans table: a REST replay must target the same destination
        the collector batch was destined for, even if the thread-local active
        table has since changed or is unset. Async batching follows the parent
        behavior (``SpanBatcher`` when async logging is enabled).
        """
        spans = [Span(span) for span in spans]
        if self._should_log_async():
            for span in spans:
                self._span_batcher.add_span(location=self._table_name, span=span)
        else:
            self._log_spans(self._table_name, spans)

    def _fall_back_to_rest(self, spans: Sequence[ReadableSpan]) -> None:
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
        self._send_spans_via_rest(spans)

    def _replay_batch_via_rest(self, spans: Sequence[ReadableSpan], reason: str, *args) -> None:
        """Replay this batch over the REST path without abandoning the collector.

        For transient or batch-specific rejections (408/413/429, a token mint
        failure before any POST was sent): the batch was not ingested, so
        replaying it cannot duplicate spans, and the next batch still goes to
        the collector.
        """
        _logger.debug(reason, *args)
        self._send_spans_via_rest(spans)

    def _log_ambiguous_drop(self, reason: str, *args) -> None:
        """Log an ambiguous-delivery drop once per exporter at WARNING, then DEBUG.

        Whether the collector ingested the batch before failing is unknown, so
        the batch is dropped rather than replayed over the REST path, which
        could duplicate the spans. Warn loudly the first time so the loss is
        discoverable, then stay quiet (mirrors ``_has_raised_span_export_error``
        in ``DatabricksUCTableSpanExporter``).
        """
        message = reason + (
            " Delivery is ambiguous, so the spans were dropped instead of replayed over "
            "the MLflow tracing server path, which could duplicate them."
        )
        if self._has_warned_ambiguous_drop:
            _logger.debug(message, *args)
        else:
            _logger.warning(message, *args)
            self._has_warned_ambiguous_drop = True

    def _handle_post_exception(self, exc: BaseException, spans: Sequence[ReadableSpan]) -> None:
        if _is_connection_not_established(exc):
            # The request was never delivered, so the batch was definitely not
            # ingested and replaying it over the REST path cannot duplicate spans.
            self._fall_back_to_rest(spans)
        else:
            # Read timeouts and mid-request connection failures (e.g. a reset after
            # the request was sent) leave delivery ambiguous.
            self._log_ambiguous_drop(
                "The Databricks OTel collector span export request failed: %s.", exc
            )

    def _export_spans_to_collector(self, spans: Sequence[ReadableSpan]) -> None:
        """Send *spans* to the collector, retrying once on a 401 after forcing token refresh."""
        endpoint = self._ensure_collector_endpoint()
        if endpoint is None:
            # Resolution failed once; do not retry it for later batches either:
            # pin the REST path for this and all subsequent span batches.
            self._collector_rejected = True
            _log_collector_unavailable(
                "The Databricks OTel collector endpoint could not be resolved for "
                "workspace_id=%r, host=%r.",
                self._workspace_id,
                self._host,
            )
            self._send_spans_via_rest(spans)
            return

        # Wrap the raw OTel spans in the MLflow Span interface: `Span.to_otel_proto` decodes
        # the JSON-encoded attribute values (e.g. mlflow.spanType), which the raw ReadableSpan
        # serialization of the stock OTLP exporter would send double-encoded.
        payload = build_otlp_export_request([Span(span) for span in spans]).SerializeToString()

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
            )
            return

        try:
            response = self._post_spans(payload, token)
        except Exception as exc:
            self._handle_post_exception(exc, spans)
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
                self._fall_back_to_rest(spans)
                return
            try:
                response = self._post_spans(payload, refreshed)
            except Exception as exc:
                self._handle_post_exception(exc, spans)
                return

        # NB: unlike the stock OTLP HTTP exporter, which retries 429/5xx with backoff, we
        # intentionally do not retry rate limits or server errors: this runs on
        # span-processing hot paths, and re-sending a batch to an overloaded ingest only
        # adds load.
        if response.ok:
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
                )
            else:
                self._fall_back_to_rest(spans)
            return

        # A 5xx leaves delivery ambiguous (the batch may have been ingested before
        # the server error), so the batch is dropped rather than replayed over the
        # REST path, which could duplicate the spans.
        self._log_ambiguous_drop(
            "The Databricks OTel collector span export failed with HTTP %d: %r.",
            response.status_code,
            response.content[:_BODY_SNIPPET_BYTES],
        )

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
                # The collector definitively rejected an earlier batch (or endpoint
                # resolution failed); skip it entirely and use the tracing-server
                # REST path directly.
                self._send_spans_via_rest(spans)
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
            # Close the HTTP session even if the parent shutdown fails, so the
            # underlying connections are not leaked.
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
    defers all network I/O: the collector endpoint (when not explicitly
    overridden via ``MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT``, which is
    validated here without network access) is resolved lazily by the exporter
    on its first batch.

    Returns ``None`` (so the caller falls back to the standard UC table path) when:
    - ``MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT`` is ``False``.
    - The destination is not a ``UnityCatalog`` location.
    - Service-principal credentials are not available on *tracking_uri*.
    - The explicit endpoint override fails host validation.
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

    # Resolve service-principal credentials.
    try:
        host_creds = get_databricks_host_creds(tracking_uri)
    except Exception as exc:
        _log_collector_unavailable(
            "Databricks credentials could not be retrieved for tracking URI %r: %s.",
            tracking_uri,
            exc,
        )
        return None

    client_id = getattr(host_creds, "client_id", None)
    client_secret = getattr(host_creds, "client_secret", None)
    host = getattr(host_creds, "host", None)
    workspace_id = str(getattr(host_creds, "workspace_id", "") or "")

    if not (client_id and client_secret):
        _log_collector_unavailable(
            "Service-principal credentials (client_id + client_secret) are not available "
            "for tracking URI %r.",
            tracking_uri,
        )
        return None

    if not host:
        _log_collector_unavailable(
            "The workspace host could not be resolved from tracking URI %r.",
            tracking_uri,
        )
        return None

    if not workspace_id:
        _log_collector_unavailable(
            "The workspace ID could not be resolved from tracking URI %r.",
            tracking_uri,
        )
        return None

    # An explicit endpoint override is validated locally (pure string checks, no
    # network); without one the endpoint is resolved lazily on the first export.
    endpoint = None
    if override := MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT.get():
        if is_databricks_otel_collector_host(override, workspace_id):
            endpoint = override
        else:
            _log_collector_unavailable(
                "The MLFLOW_DATABRICKS_OTEL_COLLECTOR_ENDPOINT override %r failed host "
                "validation for workspace_id=%r.",
                override,
                workspace_id,
            )
            return None

    # Derive the fully-qualified table name from the destination.
    table_name = _get_table_name_from_destination(destination)
    if not table_name:
        _log_collector_unavailable(
            "Could not determine the UC table name from the trace destination %r.",
            destination,
        )
        return None

    try:
        token_source = build_databricks_otel_collector_token_source(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            workspace_id=workspace_id,
            tables=[table_name],
        )
    except Exception as exc:
        _log_collector_unavailable(
            "Failed to build the Databricks OTel collector token source: %s.",
            exc,
        )
        return None

    _logger.debug(
        "Databricks OTel collector span exporter configured: endpoint=%r (resolved lazily "
        "when unset), table=%r",
        endpoint,
        table_name,
    )
    try:
        return DatabricksOtelCollectorSpanExporter(
            tracking_uri=tracking_uri,
            token_source=token_source,
            table_name=table_name,
            host=host,
            workspace_id=workspace_id,
            client_id=client_id,
            client_secret=client_secret,
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
