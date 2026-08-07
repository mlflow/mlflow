import logging
import os
import threading
from collections import defaultdict
from contextlib import nullcontext
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import Span
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpansLocation, TraceTagKey
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.export.utils import try_link_prompts_to_trace
from mlflow.tracing.fluent import _EVAL_REQUEST_ID_TO_TRACE_ID
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    add_size_stats_to_trace_metadata,
    encode_span_id,
    get_experiment_id_for_trace,
    maybe_get_request_id,
)
from mlflow.utils.databricks_utils import is_in_databricks_notebook
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
from mlflow.utils.workspace_context import ServerWorkspaceContext

_logger = logging.getLogger(__name__)

# HTTP statuses that mean the caller's credential was rejected.
_AUTH_FAILURE_STATUS_CODES = (401, 403)

# Phrases that identify a credential failure raised before any HTTP status exists,
# such as a local credential provider that cannot build a token at all. These are
# auth-specific wordings, not bare numbers, so they cannot match an unrelated
# message that happens to contain a status-like substring.
_AUTH_FAILURE_MARKERS = (
    "token refresh",
    "unauthorized",
    "invalid access token",
    "expired token",
    "credentials have expired",
    "could not identify databricks workspace configuration",
    "default auth",
)


# Generic gateway messages that carry INTERNAL_ERROR but actually mean the request
# never reached the endpoint, typically because a wrong-workspace or expired token
# was routed to a gateway that returns a short, unhelpful body. A genuine missing
# resource carries RESOURCE_DOES_NOT_EXIST (not INTERNAL_ERROR), so keying on the
# INTERNAL_ERROR code plus a short/empty message avoids matching real server errors,
# which carry a substantive message.
_INTERNAL_ERROR_AUTH_MESSAGES = ("not found", "")

# MLflow raises this exact wording when an API endpoint returns a non-JSON body,
# which on a Databricks workspace is almost always a redirect to a login page.
_NON_JSON_RESPONSE_MARKER = "response body was not in a valid json format"


def _is_auth_failure(exc: Exception) -> bool:
    """Check for an auth or credential failure in an export exception.

    A dropped trace from an auth failure is worth an ERROR rather than a WARNING,
    because the fix is a user re-auth rather than a transient retry. Numeric
    statuses are read from the exception's structured error code, so an unrelated
    message such as ``RESOURCE_DOES_NOT_EXIST: No Experiment with id=403 exists``
    is not misreported as an authentication problem.
    """
    if isinstance(exc, MlflowException):
        if exc.get_http_status_code() in _AUTH_FAILURE_STATUS_CODES:
            return True
        # A gateway that rejects the credential often wraps the failure in
        # INTERNAL_ERROR with a short, generic message ("Not Found" or empty)
        # instead of a proper auth code. Keyed on the code plus a short message so
        # a real INTERNAL_ERROR with a substantive message stays a server error.
        if getattr(exc, "error_code", None) == "INTERNAL_ERROR":
            rest_json = getattr(exc, "json", None)
            raw_message = rest_json.get("message", "") if isinstance(rest_json, dict) else ""
            if raw_message.strip().lower() in _INTERNAL_ERROR_AUTH_MESSAGES:
                return True
        # A non-JSON response to a JSON API endpoint is almost always an auth
        # redirect to a login page.
        if _NON_JSON_RESPONSE_MARKER in str(exc).lower():
            return True
    return any(marker in str(exc).lower() for marker in _AUTH_FAILURE_MARKERS)


def _get_profile_from_uri(tracking_uri: str | None) -> str:
    """Resolve the Databricks CLI profile that a tracking URI selects.

    ``databricks://<profile>`` names the profile directly. A bare ``databricks``
    URI selects the profile named by ``DATABRICKS_CONFIG_PROFILE``, or ``DEFAULT``.
    Naming this in an export-failure message tells the user which credential was
    attempted so they can spot a wrong-profile misconfiguration.
    """
    profile, _ = get_db_info_from_uri(tracking_uri)
    return profile or os.environ.get("DATABRICKS_CONFIG_PROFILE") or "DEFAULT"


class MlflowV3SpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to MLflow Tracking Server
    using the V3 trace schema and API.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._client = TracingClient(tracking_uri)
        self._is_async_enabled = self._should_enable_async_logging()
        if self._is_async_enabled:
            self._async_queue = AsyncTraceExportQueue()

        # Display handler is no-op when running outside of notebooks.
        self._display_handler = get_display_handler()

        # Tracks whether the store supports span-level logging. Set to False at runtime
        # if log_spans() raises NotImplementedError or returns a 501.
        self._store_supports_log_spans = True

        # Root spans deferred when background thread spans are still running at export time.
        # Keyed by OTel trace ID; popped and exported once all spans in the trace have ended.
        # Protected by _deferred_lock because SimpleSpanProcessor calls export() from the
        # span's own thread, so concurrent calls from multiple threads are possible.
        self._deferred_root_spans: dict[int, ReadableSpan] = {}
        self._deferred_lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export the spans to the destination.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. All spans (root and non-root) are exported.
        """

        if self._store_supports_log_spans:
            self._export_spans_incrementally(spans)

        self._export_traces(spans)

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export spans incrementally as they complete.

        Args:
            spans: Sequence of ReadableSpan objects to export.
            manager: The trace manager instance.
        """
        if is_databricks_uri(self._client.tracking_uri):
            _logger.debug(
                "Databricks tracking server only supports logging spans to UC table, "
                "skipping span exporting."
            )
            return

        mlflow_spans_by_experiment_and_workspace = self._collect_mlflow_spans_for_export(spans)
        for (
            (experiment_id, workspace),
            spans_to_log,
        ) in mlflow_spans_by_experiment_and_workspace.items():
            if self._should_log_async():
                self._async_queue.put(
                    task=Task(
                        handler=self._log_spans,
                        args=(experiment_id, spans_to_log, workspace),
                        error_msg="Failed to log spans to the trace server.",
                    )
                )
            else:
                self._log_spans(experiment_id, spans_to_log, workspace=workspace)

    def _collect_mlflow_spans_for_export(
        self, spans: Sequence[ReadableSpan]
    ) -> dict[tuple[str, str | None], list[Span]]:
        """
        Collect MLflow spans from ReadableSpans for export, grouped by (experiment_id, workspace).

        The experiment_id and workspace are resolved from the trace (set during on_start in the
        originating thread) rather than from thread-local ContextVars that are unavailable in the
        batch processor's worker thread.

        Args:
            spans: Sequence of ReadableSpan objects.

        Returns:
            Dictionary mapping (experiment_id, workspace) to list of MLflow Span objects.
        """
        manager = InMemoryTraceManager.get_instance()
        spans_by_experiment_and_workspace = defaultdict(list)

        for span in spans:
            mlflow_trace_id = manager.get_mlflow_trace_id_from_otel_id(span.context.trace_id)
            if mlflow_trace_id is None:
                continue
            span_id = encode_span_id(span.context.span_id)
            mlflow_span = manager.get_span_from_id(mlflow_trace_id, span_id)
            if mlflow_span is None:
                continue
            # Get experiment_id and workspace from trace info (resolved at on_start time in the
            # originating thread) to survive BatchSpanProcessor thread hops.
            workspace = None
            with manager.get_trace(mlflow_trace_id) as trace:
                try:
                    experiment_id = trace.info.experiment_id if trace else None
                except AttributeError:
                    # Remote/distributed traces may have trace_location=None
                    experiment_id = None
                if trace is not None:
                    workspace = trace.workspace
            if experiment_id is None:
                experiment_id = get_experiment_id_for_trace(span)
            spans_by_experiment_and_workspace[(experiment_id, workspace)].append(mlflow_span)

        return spans_by_experiment_and_workspace

    def _export_traces(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export full traces for root spans.

        Args:
            spans: Sequence of ReadableSpan objects.
        """
        manager = InMemoryTraceManager.get_instance()

        # Flush any previously deferred root spans whose background spans have now ended.
        # Copy the keys under the lock, then check has_open_spans() outside the lock to
        # avoid holding _deferred_lock while acquiring InMemoryTraceManager._lock (deadlock risk).
        with self._deferred_lock:
            deferred_ids = list(self._deferred_root_spans.keys())
        for otel_trace_id in deferred_ids:
            if not manager.has_open_spans(otel_trace_id):
                with self._deferred_lock:
                    deferred_span = self._deferred_root_spans.pop(otel_trace_id, None)
                if deferred_span is not None:
                    self._do_export_trace(manager, deferred_span)

        for span in spans:
            if span._parent is not None:
                continue

            # If background-thread child spans are still running, defer the full trace export
            # so that pop_trace is not called until after those spans land in a later batch.
            # This prevents _collect_mlflow_spans_for_export from losing the OTel→MLflow trace
            # ID mapping before those late spans can be logged.
            if manager.has_open_spans(span.context.trace_id):
                with self._deferred_lock:
                    self._deferred_root_spans[span.context.trace_id] = span
                continue

            self._do_export_trace(manager, span)

    def _do_export_trace(self, manager: InMemoryTraceManager, span: ReadableSpan) -> None:
        manager_trace = manager.pop_trace(span.context.trace_id)
        if manager_trace is None:
            _logger.debug(f"Trace for root span {span} not found. Skipping full export.")
            return

        if manager_trace.is_remote_trace and not self._store_supports_log_spans:
            _logger.warning(
                f"Current MLflow server does not support ingesting the span {span.name} "
                "that is created in a remote process. Please upgrade the server version and "
                "use SQL backend to do distributed tracing."
            )
            return

        trace = manager_trace.trace

        # Store mapping from eval request ID to trace ID so that the evaluation
        # harness can access to the trace using mlflow.get_trace(eval_request_id)
        if eval_request_id := trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID):
            _EVAL_REQUEST_ID_TO_TRACE_ID[eval_request_id] = trace.info.trace_id

        if not maybe_get_request_id(is_evaluate=True):
            self._display_handler.display_traces([trace])

        workspace = manager_trace.workspace
        if self._should_log_async():
            self._async_queue.put(
                task=Task(
                    handler=self._log_trace,
                    args=(trace, manager_trace.prompts, workspace),
                    error_msg="Failed to log trace to the trace server.",
                )
            )
        else:
            self._log_trace(trace, prompts=manager_trace.prompts, workspace=workspace)

    def _log_spans(
        self, experiment_id: str, spans: list[Span], workspace: str | None = None
    ) -> None:
        """
        Helper method to log spans with error handling.

        Args:
            experiment_id: The experiment ID to log spans to.
            spans: List of spans to log.
            workspace: Optional workspace name captured on the originating thread.
                When provided, a ServerWorkspaceContext is applied so that HTTP calls
                on the async worker thread include the correct X-MLFLOW-WORKSPACE header
                without mutating process-wide environment variables.
        """
        with ServerWorkspaceContext(workspace) if workspace else nullcontext():
            try:
                self._client.log_spans(experiment_id, spans)
            except NotImplementedError:
                # Silently skip if the store doesn't support log_spans. This is expected for stores
                # that don't implement span-level logging, and we don't want to spam warnings for
                # every span.
                self._store_supports_log_spans = False
            except RestException as e:
                # When the FileStore is behind the tracking server, it returns 501 exception.
                # However, the OTLP endpoint returns general HTTP error, not MlflowException, which
                # does not include error_code in the body and handled as a general server side
                # error. Hence, we need to check the message to handle this case.
                if "REST OTLP span logging is not supported" in e.message:
                    self._store_supports_log_spans = False
                else:
                    _logger.debug(f"Failed to log span to MLflow backend: {e}")
            except Exception as e:
                _logger.debug(f"Failed to log span to MLflow backend: {e}")

    def _log_trace(
        self, trace: Trace, prompts: Sequence[PromptVersion], workspace: str | None = None
    ) -> None:
        """
        Handles exporting a trace to MLflow using the V3 API and blob storage.
        Steps:
        1. Create the trace in MLflow
        2. Upload the trace data to blob storage using the returned trace info.

        Args:
            trace: The trace to export.
            prompts: Prompt versions to link to the trace.
            workspace: Optional workspace name captured on the originating thread.
                When provided, a ServerWorkspaceContext is applied so that HTTP calls
                on the async worker thread include the correct X-MLFLOW-WORKSPACE header
                without mutating process-wide environment variables.
        """
        with ServerWorkspaceContext(workspace) if workspace else nullcontext():
            returned_trace_info = None
            try:
                if trace:
                    add_size_stats_to_trace_metadata(trace)
                    returned_trace_info = self._client.start_trace(trace.info)

                    if self._should_log_spans_to_artifacts(returned_trace_info):
                        self._client._upload_trace_data(returned_trace_info, trace.data)
                else:
                    _logger.warning("No trace or trace info provided, unable to export")
            except Exception as e:
                # Name the tracking URI and resolved profile so the user can see which
                # credential was attempted, which surfaces a wrong-profile misconfiguration.
                creds = (
                    f" (tracking URI: {self._client.tracking_uri!r}, "
                    f"profile: {_get_profile_from_uri(self._client.tracking_uri)!r})"
                    if is_databricks_uri(self._client.tracking_uri)
                    else ""
                )
                if _is_auth_failure(e):
                    # An expired or missing credential silently drops the trace: the app
                    # keeps running, so without a loud signal the user never learns the
                    # trace was lost. Surface it at ERROR with a re-auth hint.
                    _logger.error(
                        "Failed to send trace to MLflow backend because of an "
                        f"authentication error, so the trace was NOT saved: {e}{creds}. "
                        "Refresh your credentials (for example, run `databricks auth login`) "
                        "and retry.",
                        exc_info=_logger.isEnabledFor(logging.DEBUG),
                    )
                else:
                    _logger.warning(
                        f"Failed to send trace to MLflow backend: {e}{creds}",
                        exc_info=_logger.isEnabledFor(logging.DEBUG),
                    )

            # Upload attachments in a separate try-except so trace data still lands
            # even if attachment upload fails. Runs regardless of span storage mode —
            # in TRACKING_STORE mode, spans are in the DB but attachments still go
            # to the artifact repo via the mlflow.artifactLocation tag.
            try:
                if trace and returned_trace_info:
                    attachments = {}
                    for span in trace.data.spans:
                        attachments.update(span._attachments)
                    if attachments:
                        self._client._upload_attachments(returned_trace_info, attachments)
            except Exception as e:
                _logger.warning(
                    f"Failed to upload trace attachments: {e}",
                    exc_info=_logger.isEnabledFor(logging.DEBUG),
                )

            try:
                # Always run prompt linking asynchronously since (1) prompt linking API calls
                # would otherwise add latency to the export procedure and (2) prompt linking is
                # not critical for trace export (if the prompt fails to link, the user's workflow
                # is minorly affected), so we don't have to await successful linking
                try_link_prompts_to_trace(
                    client=self._client,
                    trace_id=trace.info.trace_id,
                    prompts=prompts,
                    synchronous=False,
                )
            except Exception as e:
                _logger.warning(f"Failed to link prompts to trace: {e}")

    def _should_enable_async_logging(self) -> bool:
        if is_in_databricks_notebook():
            # NB: We don't turn on async logging in Databricks notebook by default
            # until we are confident that the async logging is working on the
            # offline workload on Databricks, to derisk the inclusion to the
            # standard image. When it is enabled explicitly via the env var, we
            # will respect that.
            return (
                MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()
                if MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.is_set()
                else False
            )

        return MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()

    def _should_log_async(self) -> bool:
        # During evaluate or assertion tests, the harness relies on the generated
        # trace objects being immediately available, so log synchronously.
        if maybe_get_request_id(is_evaluate=True):
            return False

        return self._is_async_enabled

    def shutdown(self) -> None:
        # Flush any deferred root spans that are still pending (e.g. if a background
        # span never ended). This prevents them from leaking across exporter lifetimes.
        with self._deferred_lock:
            pending = list(self._deferred_root_spans.items())
            self._deferred_root_spans.clear()
        manager = InMemoryTraceManager.get_instance()
        for _, span in pending:
            self._do_export_trace(manager, span)

    def _should_log_spans_to_artifacts(self, trace_info: TraceInfo) -> bool:
        """
        Whether to log spans to artifacts. Overridden by UC table exporter to False.
        """
        # We only log traces to artifacts when the tracking store doesn't support span logging
        return trace_info.tags.get(TraceTagKey.SPANS_LOCATION) != SpansLocation.TRACKING_STORE.value
