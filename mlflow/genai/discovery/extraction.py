from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from mlflow.entities.span import Span, SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.discovery.constants import (
    FAILURE_LABEL_SYSTEM_PROMPT,
)
from mlflow.genai.discovery.entities import _ConversationAnalysis
from mlflow.genai.discovery.utils import _call_llm, _TokenCounter

_logger = logging.getLogger(__name__)

# Span types that represent generic LLM plumbing, not meaningful execution steps.
_GENERIC_SPAN_TYPES = {SpanType.LLM, SpanType.CHAT_MODEL, SpanType.EMBEDDING}


def _has_error(span: Span) -> bool:
    return bool(span.status and span.status.status_code == SpanStatusCode.ERROR)


def extract_execution_path(trace: Trace) -> str:
    """
    Extract a compact execution path from a trace's span tree.

    Filters out generic LLM/embedding spans and returns the meaningful
    execution steps: sub-agents routed to, tools called, etc. Only
    depth-1 spans (direct children of root) and their children are
    included — the root span itself is always excluded.

    Args:
        trace: The trace to extract the execution path from.

    Returns:
        A string like ``"ask_sports_assistant > get_live_scores, web_search"``
        or ``"(no routing)"`` if only generic spans were found.
    """
    spans = trace.data.spans
    if not spans:
        return "(no spans)"

    # Build a map from parent span ID to child spans
    children_by_parent: dict[str | None, list[Span]] = defaultdict(list)
    for span in spans:
        children_by_parent[span.parent_id].append(span)

    # Find root spans (parent_id=None). If none found (malformed trace), use the first span.
    roots = children_by_parent.get(None)
    if not roots:
        _logger.debug("Trace has no root span (parent_id=None), falling back to first span")
        roots = spans[:1]

    # Build mapping from depth-1 spans to their non-generic children.
    # Depth-1 spans are the top-level orchestrator calls (sub-agents, tools).
    # Keyed by span_id (not name) to avoid collisions when duplicate names exist.
    top_level: list[tuple[str, str, bool]] = []  # (span_id, name, has_error)
    children_of: dict[str, list[str]] = defaultdict(list)

    for root in roots:
        for depth1_span in children_by_parent.get(root.span_id, []):
            if depth1_span.span_type in _GENERIC_SPAN_TYPES:
                continue
            top_level.append((depth1_span.span_id, depth1_span.name, _has_error(depth1_span)))

            # Collect non-generic descendants (depth 2+) iteratively
            stack = list(children_by_parent.get(depth1_span.span_id, []))
            while stack:
                child = stack.pop()
                if child.span_type not in _GENERIC_SPAN_TYPES:
                    suffix = " [ERROR]" if _has_error(child) else ""
                    children_of[depth1_span.span_id].append(f"{child.name}{suffix}")
                stack.extend(children_by_parent.get(child.span_id, []))

    if not top_level:
        return "(no routing)"

    # Format as "parent > child1, child2 | parent2 > child3"
    parts = []
    for span_id, name, has_error in top_level:
        display = f"{name} [ERROR]" if has_error else name
        if child_entries := children_of.get(span_id, []):
            unique_children = list(dict.fromkeys(child_entries))
            parts.append(f"{display} > {', '.join(unique_children)}")
        else:
            parts.append(display)

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
    """
    Collect deduplicated error messages from all spans in a trace.

    Checks both span status descriptions and exception events, deduplicates
    them, and returns a truncated semicolon-separated string.
    """
    spans = trace.data.spans
    if not spans:
        return ""

    # Use dict for ordered deduplication (preserves first-seen order)
    errors: dict[str, None] = {}

    for span in spans:
        if not _has_error(span):
            continue

        # Grab the span-level status description (e.g. "tool call failed")
        if span.status.description:
            errors.setdefault(f"{span.name}: {span.status.description}", None)

        # Also check exception events for detailed stack traces / error types
        for event in span.events or []:
            if event.name != "exception":
                continue
            attrs = event.attributes or {}
            exc_type = attrs.get("exception.type", "")
            exc_msg = attrs.get("exception.message", "")
            if msg := f"{exc_type}: {exc_msg}" if exc_type else exc_msg:
                errors.setdefault(msg, None)

    if not errors:
        return ""

    return "; ".join(errors)[:max_length]


def extract_assessment_rationale(trace: Trace, scorer_name: str) -> str:
    assessments = trace.search_assessments(scorer_name, type="feedback")
    return next((a.rationale for a in assessments if a.rationale), "")


def collect_session_rationales(
    traces: list[Trace],
    rationale_map: dict[str, str],
    scorer_name: str,
) -> str:
    """Combine triage rationales, human feedback, and span errors for a session.

    Deduplicates identical text across traces and returns a semicolon-separated string.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for trace in traces:
        for text, prefix in [
            (rationale_map.get(trace.info.trace_id, ""), ""),
            (extract_assessment_rationale(trace, scorer_name), "[human feedback] "),
            (extract_span_errors(trace), "[span errors] "),
        ]:
            if text and text not in seen:
                seen.add(text)
                parts.append(f"{prefix}{text}" if prefix else text)
    return "; ".join(parts)


def extract_failure_labels(
    analyses: list[_ConversationAnalysis],
    model: str,
    token_counter: _TokenCounter | None = None,
) -> tuple[list[str], list[int]]:
    """
    Extract short failure labels that combine execution path with symptom.

    Each label has the format ``[execution_path] symptom description``.
    The execution path comes from the trace spans (which sub-agents/tools
    were called). The symptom is extracted by an LLM from the triage
    rationale. A single conversation may produce multiple labels if it
    exhibits multiple distinct failures.

    Args:
        analyses: Conversation analyses to generate labels for.
        model: Model URI for the label-generation LLM.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        A tuple of (labels, label_to_analysis) where label_to_analysis[i]
        is the index into ``analyses`` that produced labels[i].
    """
    if not analyses:
        return [], []

    def _generate_labels(analysis: _ConversationAnalysis) -> list[str]:
        rationale = analysis.rationale_summary
        response = _call_llm(
            model,
            [
                {"role": "system", "content": FAILURE_LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": rationale},
            ],
            token_counter=token_counter,
        )
        content = response.choices[0].message.content.strip()
        symptoms = [line.lstrip("- ").strip() for line in content.splitlines() if line.strip()]
        return [f"[{analysis.execution_path}] {symptom}" for symptom in (symptoms or [content])]

    # Generate labels in parallel — each analysis is an independent LLM call
    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(analyses))
    raw_labels: list[list[str] | None] = [None] * len(analyses)
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoveryLabel"
    ) as executor:
        future_to_idx = {
            executor.submit(_generate_labels, analysis): i for i, analysis in enumerate(analyses)
        }
        for future in as_completed(future_to_idx):
            raw_labels[future_to_idx[future]] = future.result()

    # Flatten into (labels, label_to_analysis) preserving analysis order
    labels: list[str] = []
    label_to_analysis: list[int] = []
    for analysis_idx, analysis_labels in enumerate(raw_labels):
        for label in analysis_labels or []:
            labels.append(label)
            label_to_analysis.append(analysis_idx)

    return labels, label_to_analysis


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
            assessments = trace.search_assessments(scorer_name, type="feedback")
            assessment = assessments[-1] if assessments else None
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
