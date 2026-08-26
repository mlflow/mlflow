"""
Utility functions for trace exporters.
"""

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Sequence

from mlflow.entities.model_registry import PromptVersion
from mlflow.tracing.client import TracingClient

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

_logger = logging.getLogger(__name__)


def try_link_prompts_to_trace(
    client: TracingClient,
    trace_id: str,
    prompts: Sequence[PromptVersion],
    synchronous: bool = True,
) -> None:
    """
    Attempt to link prompt versions to a trace with graceful error handling.

    This function provides a reusable way to link prompts to traces with consistent
    error handling across different exporters. Errors are caught and logged but do
    not propagate, ensuring that prompt linking failures don't affect trace export.

    Args:
        client: The TracingClient instance to use for linking.
        trace_id: The ID of the trace to link prompts to.
        prompts: Sequence of PromptVersion objects to link.
        synchronous: If True, run the linking synchronously. If False, run in a separate thread.
    """
    if not prompts:
        return

    if synchronous:
        _link_prompts_sync(client, trace_id, prompts)
    else:
        threading.Thread(
            target=_link_prompts_sync,
            args=(client, trace_id, prompts),
            name=f"link_prompts_from_exporter-{uuid.uuid4().hex[:8]}",
        ).start()


def _link_prompts_sync(
    client: TracingClient,
    trace_id: str,
    prompts: Sequence[PromptVersion],
) -> None:
    """
    Synchronously link prompt versions to a trace with error handling.

    This is the core implementation that handles the actual API call and error logging.

    Args:
        client: The TracingClient instance to use for linking.
        trace_id: The ID of the trace to link prompts to.
        prompts: Sequence of PromptVersion objects to link.
    """
    try:
        client.link_prompt_versions_to_trace(
            trace_id=trace_id,
            prompts=prompts,
        )
        _logger.debug(f"Successfully linked {len(prompts)} prompts to trace {trace_id}")
    except Exception as e:
        _logger.warning(f"Failed to link prompts to trace {trace_id}: {e}")


def flush_exporter(exporter: "SpanExporter", terminate: bool = False) -> None:
    """
    Drain every layer an MLflow exporter buffers spans in.

    ``DatabricksUCTableSpanExporter`` batches spans in a ``SpanBatcher`` before they
    reach its async queue, so draining only the queue exports nothing. The batcher has
    to go first: ``AsyncTraceExportQueue.flush()`` returns immediately while the queue
    is inactive, and the queue is only activated by its first ``put()`` — which is the
    batcher draining.

    Both attributes are optional. Neither exists when async trace logging is disabled,
    and a bare ``SpanExporter`` has neither at any time.

    Args:
        exporter: The span exporter to drain.
        terminate: If True, shut the exporter's logging threads down after flushing.
    """
    if span_batcher := getattr(exporter, "_span_batcher", None):
        # SpanBatcher.shutdown() is permanent - add_span() drops spans once the stop
        # event is set - so it is only used when the caller asked to terminate.
        if terminate:
            span_batcher.shutdown()
        else:
            span_batcher.flush()

    if async_queue := getattr(exporter, "_async_queue", None):
        async_queue.flush(terminate=terminate)
