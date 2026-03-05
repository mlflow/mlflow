from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.discovery.constants import (
    CLUSTER_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_JUDGE_MODEL,
    FAILURE_LABEL_SYSTEM_PROMPT,
    LLM_MAX_TOKENS,
    NUM_RETRIES,
    SAMPLE_POOL_MULTIPLIER,
    SAMPLE_RANDOM_SEED,
    SURFACE_TRUNCATION_LIMIT,
)
from mlflow.genai.discovery.entities import (
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm
from mlflow.genai.scorers.base import Scorer
from mlflow.metrics.genai.model_utils import convert_model_uri_to_litellm
from mlflow.tracing.constant import TraceMetadataKey

if TYPE_CHECKING:
    from mlflow.genai.discovery.entities import Issue
    from mlflow.genai.discovery.pipeline import _TokenCounter
    from mlflow.genai.evaluation.entities import EvaluationResult

_logger = logging.getLogger(__name__)

# Span types that represent generic LLM plumbing, not meaningful execution steps.
_GENERIC_SPAN_TYPES = {"LLM", "CHAT_MODEL", "EMBEDDING"}


def get_session_id(trace: Trace) -> str | None:
    return (trace.info.trace_metadata or {}).get(TraceMetadataKey.TRACE_SESSION)


def sample_traces(
    sample_size: int,
    search_kwargs: dict[str, object],
) -> list[Trace]:
    """Randomly sample traces, grouping by session when session IDs exist.

    Fetches a pool of traces, groups them by session (or treats each trace
    as its own group when no sessions exist), then randomly selects
    ``sample_size`` groups and returns all traces from those groups.

    Args:
        sample_size: Number of groups (sessions or individual traces) to sample.
        search_kwargs: Keyword arguments passed to ``mlflow.search_traces``.

    Returns:
        List of sampled Trace objects.
    """
    pool_size = sample_size * SAMPLE_POOL_MULTIPLIER
    pool = mlflow.search_traces(max_results=pool_size, **search_kwargs)
    if not pool:
        return []

    # Group traces by session; traces without a session become their own group
    groups: dict[str, list[Trace]] = defaultdict(list)
    for trace in pool:
        key = get_session_id(trace) or trace.info.trace_id
        groups[key].append(trace)

    rng = random.Random(SAMPLE_RANDOM_SEED)
    group_keys = sorted(groups.keys())
    num_samples = min(sample_size, len(group_keys))
    selected = rng.sample(group_keys, num_samples)
    result = [trace for key in selected for trace in groups[key]]
    _logger.info(
        "Sampled %d groups (%d traces) from pool of %d groups",
        num_samples,
        len(result),
        len(group_keys),
    )
    return result


def verify_scorer(scorer: Scorer, trace: Trace) -> None:
    """Verify a scorer works on a single trace before running the full pipeline.

    Calls the scorer on the trace, fetches the updated trace, and checks
    that a Feedback assessment with a non-null value was produced.

    Args:
        scorer: The scorer to test.
        trace: A trace to run the scorer on.

    Raises:
        MlflowException: If the scorer produces no feedback or returns a null value.
    """
    try:
        scorer(trace=trace)
        result_trace = mlflow.get_trace(trace.info.trace_id)
        if result_trace is None:
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' produced no feedback on test trace"
            )
        feedback = next(
            (
                assessment
                for assessment in result_trace.info.assessments
                if isinstance(assessment, Feedback) and assessment.name == scorer.name
            ),
            None,
        )
        if feedback is None:
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' produced no feedback on test trace"
            )
        if feedback.value is None:
            error = feedback.error_message or "unknown error (check model API logs)"
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' returned null value: {error}"
            )
    except Exception as exc:
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' failed verification on trace {trace.info.trace_id}: {exc}"
        ) from exc


def extract_execution_path(trace: Trace) -> str:
    """Extract a compact execution path from a trace's span tree.

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
    """Extract combined execution paths across all traces in a session.

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

    for span in spans:
        if not (span.status and span.status.status_code == SpanStatusCode.ERROR):
            continue
        if span.status.description:
            msg = f"{span.name}: {span.status.description}"
            if msg not in seen:
                seen.add(msg)
                errors.append(msg)
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
    return "; ".join(errors)[:max_length]


def group_traces_by_session(
    traces: list[Trace],
) -> dict[str, list[Trace]]:
    """Group traces by session ID.

    Traces without a session become standalone single-trace "sessions"
    keyed by their trace_id. Each group is sorted by timestamp_ms.
    """
    groups: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        session_id = get_session_id(trace) or trace.info.trace_id
        groups[session_id].append(trace)

    for traces_in_group in groups.values():
        traces_in_group.sort(key=lambda trace: trace.info.timestamp_ms)

    return dict(groups)


def extract_failure_labels(
    analyses: list[_ConversationAnalysis],
    model: str,
    token_counter: _TokenCounter | None = None,
) -> list[str]:
    """Extract short failure labels that combine execution path with symptom.

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
    litellm_model = convert_model_uri_to_litellm(model)

    # Generate labels in parallel — each label is an independent LLM call
    def _generate_label(analysis: _ConversationAnalysis) -> str:
        rationale = analysis.surface[:SURFACE_TRUNCATION_LIMIT]
        response = _invoke_litellm(
            litellm_model=litellm_model,
            messages=[
                {"role": "system", "content": FAILURE_LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": rationale},
            ],
            tools=[],
            num_retries=NUM_RETRIES,
            response_format=None,
            include_response_format=False,
            inference_params={"max_tokens": LLM_MAX_TOKENS, "temperature": 0},
        )
        if token_counter is not None:
            token_counter.track(response)
        symptom = response.choices[0].message.content.strip()
        exec_path = analysis.execution_path or "(no routing)"
        return f"[{exec_path}] {symptom}"

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(analyses))
    labels: list[str | None] = [None] * len(analyses)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="label") as executor:
        future_to_idx = {
            executor.submit(_generate_label, analysis): i for i, analysis in enumerate(analyses)
        }
        for future in as_completed(future_to_idx):
            labels[future_to_idx[future]] = future.result()

    return [label for label in labels if label is not None]


def cluster_analyses(
    analyses: list[_ConversationAnalysis],
    max_issues: int,
    labels: list[str],
    label_model: str | None = None,
    token_counter: _TokenCounter | None = None,
) -> list[list[int]]:
    if len(analyses) == 1:
        return [[0]]

    return cluster_by_llm(labels, max_issues, label_model, token_counter=token_counter)


def cluster_by_llm(
    labels: list[str],
    max_issues: int,
    model: str | None = None,
    token_counter: _TokenCounter | None = None,
) -> list[list[int]]:
    """Use an LLM to group failure labels by execution path and symptom.

    Each label has the format ``[execution_path] symptom``, where the
    execution path shows which sub-agents/tools were called. The LLM
    groups labels that share similar execution paths AND similar failure
    symptoms into coherent issue categories.

    Args:
        labels: Failure labels to cluster.
        max_issues: Maximum number of groups to produce.
        model: Model URI for the clustering LLM.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        List of index lists, where each inner list is a cluster of label indices.
    """
    model = model or DEFAULT_JUDGE_MODEL
    litellm_model = convert_model_uri_to_litellm(model)

    numbered = "\n".join(f"[{i}] {lbl}" for i, lbl in enumerate(labels))
    prompt = (
        f"Below are {len(labels)} failure labels from an AI agent.\n"
        "Each label has the format: [execution_path] symptom\n"
        "The execution path shows which sub-agents and tools were called.\n\n"
        "Group these labels into coherent issue categories. Two labels belong "
        "in the same group when:\n"
        "  1. They share the same failure pattern (similar symptom)\n"
        "  2. They involve the same tool, sub-agent, or execution path\n\n"
        "Same tool/path strongly suggests the same root cause — group together "
        "unless symptoms are clearly unrelated. Different paths MAY still be the "
        "same issue if symptoms are very similar.\n\n"
        "Rules:\n"
        "- Each group should have a name prefixed with 'Issue: ' followed by a short "
        "readable description (3-8 words), e.g. 'Issue: Incomplete response details'\n"
        "- A label can only appear in one group\n"
        "- Singleton groups are fine for truly unique issues\n"
        f"- Create at most {max_issues} groups\n\n"
        f"Labels:\n{numbered}\n\n"
        'Return a JSON object with a "groups" key containing an array of objects, '
        'each with "name" (short readable string) and "indices" (list of ints).\n'
        "Return ONLY the JSON, no explanation."
    )

    response = _invoke_litellm(
        litellm_model=litellm_model,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        num_retries=NUM_RETRIES,
        response_format={"type": "json_object"},
        include_response_format=True,
        inference_params={"max_tokens": LLM_MAX_TOKENS, "temperature": 0},
    )
    if token_counter is not None:
        token_counter.track(response)
    content = (response.choices[0].message.content or "").strip()
    if not content:
        _logger.warning(
            "LLM returned empty content for label grouping "
            "(finish_reason=%s), falling back to singletons",
            response.choices[0].finish_reason,
        )
        return [[i] for i in range(len(labels))]
    result = json.loads(content)

    # Normalize response format: accept both {"groups": [...]} and bare list
    groups = (
        result if isinstance(result, list) else result.get("groups", result.get("categories", []))
    )

    # Ensure every index appears exactly once; orphaned indices become singletons
    all_indices = set()
    cluster_groups: list[list[int]] = []
    for group in groups:
        indices = [i for i in group["indices"] if 0 <= i < len(labels)]
        if indices := [i for i in indices if i not in all_indices]:
            cluster_groups.append(indices)
            all_indices.update(indices)

    cluster_groups.extend([i] for i in range(len(labels)) if i not in all_indices)

    # Enforce max_issues limit by keeping the largest groups
    if len(cluster_groups) > max_issues:
        cluster_groups.sort(key=len, reverse=True)
        cluster_groups = cluster_groups[:max_issues]

    return cluster_groups


def summarize_cluster(
    cluster_indices: list[int],
    analyses: list[_ConversationAnalysis],
    analysis_model: str,
    token_counter: _TokenCounter | None = None,
) -> _IdentifiedIssue:
    """Summarize a cluster of analyses into a single identified issue.

    Uses an LLM to synthesize a name, description, root cause, and confidence
    for the cluster. Always returns all cluster indices as example_indices
    (overriding the LLM's selection).

    Args:
        cluster_indices: Indices into ``analyses`` that form this cluster.
        analyses: All conversation analyses from the pipeline.
        analysis_model: Model URI for the summarization LLM.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        An ``_IdentifiedIssue`` with synthesized fields and all cluster indices.
    """
    cluster_analyses = [analyses[i] for i in cluster_indices]
    parts = []
    for i, analysis in zip(cluster_indices, cluster_analyses):
        entry = f"[{i}] {analysis.surface}"
        if analysis.execution_path:
            entry += f"\n  execution_path: {analysis.execution_path}"
        parts.append(entry)
    analyses_text = "\n\n".join(parts)

    schema_json = json.dumps(_IdentifiedIssue.model_json_schema(), indent=2)
    system_prompt = (
        f"{CLUSTER_SUMMARY_SYSTEM_PROMPT}\n\n"
        f"Respond with a JSON object matching this schema:\n{schema_json}"
    )

    litellm_model = convert_model_uri_to_litellm(analysis_model)

    response = _invoke_litellm(
        litellm_model=litellm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Cluster of {len(cluster_indices)} analyses:\n\n{analyses_text}",
            },
        ],
        tools=[],
        num_retries=NUM_RETRIES,
        response_format={"type": "json_object"},
        include_response_format=True,
        inference_params={"max_tokens": LLM_MAX_TOKENS, "temperature": 0},
    )
    if token_counter is not None:
        token_counter.track(response)

    content = (response.choices[0].message.content or "").strip()
    result = _IdentifiedIssue(**json.loads(content))
    result.example_indices = cluster_indices
    return result


def extract_failing_traces(
    eval_result: EvaluationResult,
    scorer_names: str | list[str],
    original_traces: list[Trace] | None = None,
) -> tuple[list[Trace], dict[str, str]]:
    """Extract failing traces and their rationales from an evaluation result.

    ``scorer_names`` can be a single scorer name or a list. When multiple
    names are provided, a trace is considered failing if ANY scorer marks
    it as ``False``, and rationales from all failing scorers are combined.

    When ``original_traces`` is provided, maps results back to the original
    trace objects by DataFrame position (the evaluate framework may assign
    new trace IDs during scoring).

    Args:
        eval_result: The evaluation result containing a DataFrame with
            scorer value and rationale columns.
        scorer_names: One or more scorer names to check for failures.
        original_traces: Optional list of original traces to map results
            back to by position.

    Returns:
        A tuple of (failing_traces, rationale_map) where rationale_map
        maps trace_id to the combined rationale string.
    """
    if isinstance(scorer_names, str):
        scorer_names = [scorer_names]

    failing: list[Trace] = []
    rationales: dict[str, str] = {}
    df = eval_result.result_df
    if df is None:
        return failing, rationales

    # Filter to scorer names whose value column actually exists
    active_scorers = [name for name in scorer_names if f"{name}/value" in df.columns]
    if not active_scorers:
        return failing, rationales

    for idx, (_, row) in enumerate(df.iterrows()):
        row_failing_scorers: list[str] = []
        for scorer_name in active_scorers:
            val = row.get(f"{scorer_name}/value")
            if val is not None and not bool(val):
                row_failing_scorers.append(scorer_name)

        if not row_failing_scorers:
            continue

        if original_traces and idx < len(original_traces):
            trace = original_traces[idx]
        else:
            trace = row.get("trace")
            if trace is None:
                continue
            if isinstance(trace, str):
                trace = Trace.from_json(trace)

        failing.append(trace)

        # Combine rationales from all failing scorers
        row_rationales: list[str] = []
        for scorer_name in row_failing_scorers:
            rationale_col = f"{scorer_name}/rationale"
            rationale = ""
            if rationale_col in df.columns:
                rationale = str(row.get(rationale_col, "") or "")
            if not rationale:
                eval_trace = row.get("trace")
                if eval_trace is not None:
                    if isinstance(eval_trace, str):
                        eval_trace = Trace.from_json(eval_trace)
                    rationale = next(
                        (
                            assessment.rationale
                            for assessment in reversed(eval_trace.info.assessments)
                            if isinstance(assessment, Feedback)
                            and assessment.name == scorer_name
                            and assessment.rationale
                        ),
                        "",
                    )
            if rationale:
                row_rationales.append(rationale)
        rationales[trace.info.trace_id] = "; ".join(row_rationales)

    return failing, rationales


def log_discovery_artifacts(run_id: str, artifacts: dict[str, str]) -> None:
    if not run_id:
        return
    client = mlflow.MlflowClient()
    for filename, content in artifacts.items():
        try:
            client.log_text(run_id, content, filename)
        except Exception:
            _logger.warning("Failed to log %s to run %s", filename, run_id, exc_info=True)


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
            f"confidence: {issue.confidence})\n\n"
            f"{issue.description}\n\n"
            f"**Root cause:** {issue.root_cause}\n"
        )
    return "\n".join(lines)
