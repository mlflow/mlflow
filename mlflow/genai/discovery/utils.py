from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING

import pydantic

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    LLM_MAX_TOKENS,
    MAX_EXAMPLE_TRACE_IDS,
    NUM_RETRIES,
    TRACE_CONTENT_TRUNCATION,
)
from mlflow.genai.discovery.entities import Issue, _ConversationAnalysis, _IdentifiedIssue
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.provider import trace_disabled

if TYPE_CHECKING:
    import litellm

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

    def track(self, response: litellm.ModelResponse) -> None:
        with self._lock:
            if response.usage:
                self.input_tokens += response.usage.prompt_tokens or 0
                self.output_tokens += response.usage.completion_tokens or 0
            if hidden := getattr(response, "_hidden_params", None):
                if cost := hidden.get("response_cost"):
                    self.cost_usd += cost

    def to_dict(self) -> dict[str, int | float]:
        result = {}
        total = self.input_tokens + self.output_tokens
        if total > 0:
            result["input_tokens"] = self.input_tokens
            result["output_tokens"] = self.output_tokens
            result["total_tokens"] = total
        if self.cost_usd > 0:
            result["cost_usd"] = round(self.cost_usd, 6)
        return result


@trace_disabled
def _call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    response_format: type[pydantic.BaseModel] | None = None,
    token_counter: _TokenCounter | None = None,
) -> litellm.ModelResponse:
    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm
    from mlflow.metrics.genai.model_utils import convert_mlflow_uri_to_litellm

    use_format = response_format or ({"type": "json_object"} if json_mode else None)
    response = _invoke_litellm(
        litellm_model=convert_mlflow_uri_to_litellm(model),
        messages=messages,
        tools=[],
        num_retries=NUM_RETRIES,
        response_format=use_format,
        include_response_format=use_format is not None,
        inference_params={"max_completion_tokens": LLM_MAX_TOKENS},
    )
    if token_counter is not None:
        token_counter.track(response)
    return response


def build_summary(issues: list[Issue], total_traces: int) -> str:
    if not issues:
        return f"Analyzed {total_traces} traces. No issues found."

    lines = [
        f"Analyzed **{total_traces}** traces. Found **{len(issues)}** issues:\n",
    ]
    for i, issue in enumerate(issues, 1):
        root_causes = "; ".join(issue.root_causes) if issue.root_causes else "Unknown"
        lines.append(
            f"### {i}. {issue.name} (severity: {issue.severity})\n\n"
            f"{issue.description}\n\n"
            f"**Root causes:** {root_causes}\n"
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


@trace_disabled
def verify_scorer(
    scorer: Scorer,
    trace: Trace,
    session: list[Trace] | None = None,
) -> None:
    """
    Verify a scorer works on a single trace (or session) before running the full pipeline.

    Calls the scorer and checks that the returned Feedback has a non-null value.

    Args:
        scorer: The scorer to test.
        trace: A trace to run the scorer on (used for trace-based scorers).
        session: If provided, pass as ``session=`` to the scorer instead of ``trace=``.
            Used for conversation-based scorers that require ``{{ conversation }}``.

    Raises:
        MlflowException: If the scorer produces no feedback or returns a null value.
    """
    try:
        feedback = scorer(session=session) if session is not None else scorer(trace=trace)
    except Exception as exc:
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' failed verification on trace {trace.info.trace_id}: {exc}"
        ) from exc

    if not isinstance(feedback, Feedback):
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' returned {type(feedback).__name__} instead of Feedback"
        )
    if feedback.value is None:
        error = feedback.error_message or "unknown error (check model API logs)"
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' returned null value: {error}"
        )


def collect_example_trace_ids(
    issue: _IdentifiedIssue,
    analyses: list[_ConversationAnalysis],
) -> list[str]:
    trace_ids = []
    for idx in issue.example_indices:
        if 0 <= idx < len(analyses):
            trace_ids.extend(analyses[idx].affected_trace_ids)
    return trace_ids[:MAX_EXAMPLE_TRACE_IDS]


def format_trace_content(trace: Trace) -> str:
    from mlflow.genai.discovery.extraction import extract_execution_path, extract_span_errors

    parts = []
    if request := trace.data.request:
        parts.append(f"Input: {str(request)[:TRACE_CONTENT_TRUNCATION]}")
    if response := trace.data.response:
        parts.append(f"Output: {str(response)[:TRACE_CONTENT_TRUNCATION]}")
    if (exec_path := extract_execution_path(trace)) and exec_path != "(no routing)":
        parts.append(f"Execution path: {exec_path}")
    if errors := extract_span_errors(trace):
        parts.append(f"Errors: {errors}")
    return "\n".join(parts) if parts else "(trace content not available)"


def format_annotation_prompt(issue: Issue, trace_content: str, triage_rationale: str) -> str:
    return (
        f"=== ISSUE ===\n"
        f"Name: {issue.name}\n"
        f"Description: {issue.description}\n"
        f"Root causes: {'; '.join(issue.root_causes or [])}\n\n"
        f"=== TRACE ===\n"
        f"{trace_content}\n\n"
        f"=== TRIAGE JUDGE RATIONALE ===\n"
        f"{triage_rationale or '(not available)'}"
    )
