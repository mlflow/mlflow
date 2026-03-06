from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from mlflow.entities.assessment import Feedback
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.discovery.constants import (
    FAILURE_LABEL_SYSTEM_PROMPT,
    SURFACE_TRUNCATION_LIMIT,
)
from mlflow.genai.discovery.entities import _ConversationAnalysis
from mlflow.genai.discovery.utils import _call_llm, _TokenCounter

_logger = logging.getLogger(__name__)

# Span types that represent generic LLM plumbing, not meaningful execution steps.
_GENERIC_SPAN_TYPES = {"LLM", "CHAT_MODEL", "EMBEDDING"}


def extract_execution_path(trace: Trace) -> str:
    """
    Extract a compact execution path from a trace's span tree.

    Filters out generic LLM/embedding spans and returns the meaningful
    execution steps: sub-agents routed to, tools called, etc.

    Args:
        trace: The trace to extract the execution path from.

    Returns:
        A string like ``"ask_sports_assistant > get_live_scores, web_search"``
        or ``"(no routing)"`` if only generic spans were found.
    """
    spans = trace.data.spans
    if not spans:
        return "(no spans)"

    # Build a map from parent span ID to child spans for tree traversal
    children_by_parent: dict[str | None, list[object]] = defaultdict(list)
    for span in spans:
        children_by_parent[span.parent_id].append(span)

    # Walk the span tree, collecting non-generic spans with their depth and error status
    execution_steps: list[tuple[str, int, str]] = []

    def _walk(span, depth: int):
        span_type = getattr(span, "span_type", "") or ""
        if span_type not in _GENERIC_SPAN_TYPES:
            status = ""
            if span.status and span.status.status_code == SpanStatusCode.ERROR:
                status = " [ERROR]"
            execution_steps.append((span.name, depth, status))
        for child in children_by_parent.get(span.span_id, []):
            _walk(child, depth + 1)

    roots = children_by_parent.get(None, [])
    if not roots:
        roots = spans[:1]
    for root in roots:
        _walk(root, 0)

    if not execution_steps:
        return "(no routing)"

    # Skip root span (top-level entry point) and organize remaining steps hierarchically.
    # Depth 1 = sub-agents/tools called by orchestrator; depth 2+ = their children.
    non_root = [(name, depth, status) for name, depth, status in execution_steps if depth > 0]
    if not non_root:
        return "(no routing)"

    top_level_entries: list[str] = []
    sub_items: dict[str, list[str]] = defaultdict(list)
    current_parent = None

    for name, depth, status in non_root:
        entry = f"{name}{status}"
        if depth == 1:
            current_parent = name
            top_level_entries.append(entry)
        elif current_parent:
            sub_items[current_parent].append(entry)

    # Format as "parent > child1, child2 | parent2 > child3"
    parts = []
    for entry in top_level_entries:
        entry_name = entry.replace(" [ERROR]", "")
        if child_entries := sub_items.get(entry_name, []):
            unique_children = list(dict.fromkeys(child_entries))
            parts.append(f"{entry} > {', '.join(unique_children)}")
        else:
            parts.append(entry)

    return " | ".join(parts) if parts else "(no routing)"


def extract_execution_paths_for_session(traces: list[Trace]) -> str:
    """
    Extract combined execution paths across all traces in a session.

    Deduplicates paths and joins them, giving a compact summary of
    what the agent did across the entire conversation.
    """
    paths = list(dict.fromkeys(extract_execution_path(trace) for trace in traces))
    return "; ".join(paths) if paths else "(no routing)"


def extract_span_errors(trace: Trace, max_length: int = 500) -> str:
    spans = trace.data.spans
    if not spans:
        return ""

    errors: list[str] = []
    seen: set[str] = set()

    # Walk all spans, collecting error messages from those with ERROR status
    for span in spans:
        if not (span.status and span.status.status_code == SpanStatusCode.ERROR):
            continue

        # Grab the span-level status description (e.g. "tool call failed")
        if span.status.description:
            msg = f"{span.name}: {span.status.description}"
            if msg not in seen:
                seen.add(msg)
                errors.append(msg)

        # Also check exception events for detailed stack traces / error types
        for event in span.events or []:
            if event.name != "exception":
                continue
            attrs = event.attributes or {}
            exc_type = attrs.get("exception.type", "")
            exc_msg = attrs.get("exception.message", "")
            msg = f"{exc_type}: {exc_msg}" if exc_type else exc_msg
            if msg and msg not in seen:
                seen.add(msg)
                errors.append(msg)

    if not errors:
        return ""

    # Join deduplicated errors and truncate to max_length
    return "; ".join(errors)[:max_length]


def extract_failure_labels(
    analyses: list[_ConversationAnalysis],
    model: str,
    token_counter: _TokenCounter | None = None,
) -> list[str]:
    """
    Extract short failure labels that combine execution path with symptom.

    Each label has the format ``[execution_path] symptom description``.
    The execution path comes from the trace spans (which sub-agents/tools
    were called). The symptom is extracted by an LLM from the triage
    rationale. Together they enable clustering by "what the agent did"
    and "how it failed."

    Args:
        analyses: Conversation analyses to generate labels for.
        model: Model URI for the label-generation LLM.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        List of failure label strings, one per analysis.
    """

    # Generate labels in parallel — each label is an independent LLM call
    def _generate_label(analysis: _ConversationAnalysis) -> str:
        rationale = analysis.surface[:SURFACE_TRUNCATION_LIMIT]
        response = _call_llm(
            model,
            [
                {"role": "system", "content": FAILURE_LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": rationale},
            ],
            token_counter=token_counter,
        )
        symptom = response.choices[0].message.content.strip()
        exec_path = analysis.execution_path or "(no routing)"
        return f"[{exec_path}] {symptom}"

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(analyses))
    labels: list[str | None] = [None] * len(analyses)
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoveryLabel"
    ) as executor:
        future_to_idx = {
            executor.submit(_generate_label, analysis): i for i, analysis in enumerate(analyses)
        }
        for future in as_completed(future_to_idx):
            labels[future_to_idx[future]] = future.result()

    return [label for label in labels if label is not None]


def extract_failing_traces(
    scored_traces: list[Trace],
    scorer_names: str | list[str],
) -> tuple[list[Trace], dict[str, str]]:
    """
    Extract failing traces and their rationales from scored trace objects.

    After ``mlflow.genai.evaluate()`` runs, assessments are written to
    traces as Feedback objects. This function walks each trace's
    assessments to find failures.

    ``scorer_names`` can be a single scorer name or a list. A trace is
    considered failing if ANY scorer's most-recent Feedback has a falsy
    value, and rationales from all failing scorers are combined.

    Args:
        scored_traces: Traces with scorer assessments attached.
        scorer_names: One or more scorer names to check for failures.

    Returns:
        A tuple of (failing_traces, rationale_map) where rationale_map
        maps trace_id to the combined rationale string.
    """
    if isinstance(scorer_names, str):
        scorer_names = [scorer_names]

    failing: list[Trace] = []
    rationales: dict[str, str] = {}

    for trace in scored_traces:
        row_failing: list[tuple[str, str]] = []
        for scorer_name in scorer_names:
            # Find the most recent Feedback for this scorer
            assessment = next(
                (
                    assessment
                    for assessment in reversed(trace.info.assessments)
                    if isinstance(assessment, Feedback) and assessment.name == scorer_name
                ),
                None,
            )
            if assessment is None:
                continue
            if assessment.value is not None and not bool(assessment.value):
                row_failing.append((scorer_name, assessment.rationale or ""))

        if not row_failing:
            continue

        failing.append(trace)
        rationales[trace.info.trace_id] = "; ".join(
            rationale for _, rationale in row_failing if rationale
        )

    return failing, rationales
