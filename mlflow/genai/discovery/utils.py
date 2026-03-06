from __future__ import annotations

import logging
import threading
from collections import defaultdict

import mlflow
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import LLM_MAX_TOKENS, NUM_RETRIES
from mlflow.genai.discovery.entities import Issue
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm
from mlflow.metrics.genai.model_utils import convert_mlflow_uri_to_litellm
from mlflow.tracing.constant import TraceMetadataKey

_logger = logging.getLogger(__name__)


def get_session_id(trace: Trace) -> str | None:
    return (trace.info.trace_metadata or {}).get(TraceMetadataKey.TRACE_SESSION)


def group_traces_by_session(
    traces: list[Trace],
) -> dict[str, list[Trace]]:
    """
    Group traces by session ID.

    Traces without a session become standalone single-trace "sessions"
    keyed by their trace_id. Each group is sorted by timestamp_ms.

    Note: mlflow.genai.evaluation.session_utils has a similar function, but it
    operates on EvalItem objects and drops traces without sessions. This version
    works on raw Trace objects and keeps sessionless traces as standalone groups,
    which is required for the discovery pipeline's frequency calculations.
    """
    groups: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        session_id = get_session_id(trace) or trace.info.trace_id
        groups[session_id].append(trace)

    for traces_in_group in groups.values():
        traces_in_group.sort(key=lambda trace: trace.info.timestamp_ms)

    return dict(groups)


class _TokenCounter:
    """Thread-safe accumulator for LLM token usage across pipeline phases."""

    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def track(self, response) -> None:
        with self._lock:
            if usage := getattr(response, "usage", None):
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            if hidden := getattr(response, "_hidden_params", None):
                if cost := hidden.get("response_cost"):
                    self.cost_usd += cost

    def to_dict(self) -> dict[str, object]:
        result = {}
        total = self.input_tokens + self.output_tokens
        if total > 0:
            result["input_tokens"] = self.input_tokens
            result["output_tokens"] = self.output_tokens
            result["total_tokens"] = total
        if self.cost_usd > 0:
            result["cost_usd"] = round(self.cost_usd, 6)
        return result


def _call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    token_counter: _TokenCounter | None = None,
) -> object:
    response = _invoke_litellm(
        litellm_model=convert_mlflow_uri_to_litellm(model),
        messages=messages,
        tools=[],
        num_retries=NUM_RETRIES,
        response_format={"type": "json_object"} if json_mode else None,
        include_response_format=json_mode,
        inference_params={"max_tokens": LLM_MAX_TOKENS},
    )
    if token_counter is not None:
        token_counter.track(response)
    return response


def build_summary(issues: list[Issue], total_traces: int) -> str:
    if not issues:
        return f"## Issue Discovery Summary\n\nAnalyzed {total_traces} traces. No issues found."

    lines = [
        "## Issue Discovery Summary\n",
        f"Analyzed **{total_traces}** traces. Found **{len(issues)}** issues:\n",
    ]
    for i, issue in enumerate(issues, 1):
        lines.append(
            f"### {i}. {issue.name} ({issue.frequency:.0%} of traces, "
            f"severity: {issue.severity})\n\n"
            f"{issue.description}\n\n"
            f"**Root cause:** {issue.root_cause}\n"
        )
    return "\n".join(lines)


def log_discovery_artifacts(run_id: str, artifacts: dict[str, str]) -> None:
    if not run_id:
        return
    client = mlflow.MlflowClient()
    for filename, content in artifacts.items():
        try:
            client.log_text(run_id, content, filename)
        except Exception:
            _logger.warning("Failed to log %s to run %s", filename, run_id, exc_info=True)
