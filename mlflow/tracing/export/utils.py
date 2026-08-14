"""
Utility functions for prompt linking in trace exporters.
"""

import logging
import threading
import uuid
from typing import Sequence

from mlflow.entities.model_registry import PromptVersion
from mlflow.tracing.client import TracingClient

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
