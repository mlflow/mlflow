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
from mlflow.genai.discovery.constants import (
    _CLUSTER_SUMMARY_SYSTEM_PROMPT,
    _DEFAULT_SCORER_NAME,
    _build_satisfaction_instructions,
)
from mlflow.genai.discovery.entities import (
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.judges.utils.invocation_utils import (
    get_chat_completions_with_structured_output,
)
from mlflow.genai.scorers.base import Scorer

if TYPE_CHECKING:
    from mlflow.genai.evaluation.entities import EvaluationResult

_logger = logging.getLogger(__name__)


def _get_session_id(trace: Trace) -> str | None:
    return (trace.info.tags or {}).get("mlflow.trace.session_id") or (
        trace.info.trace_metadata or {}
    ).get("mlflow.trace.session")


def _sample_traces(
    sample_size: int,
    search_kwargs: dict[str, object],
    pool_multiplier: int = 5,
) -> list[Trace]:
    """Randomly sample traces, grouping by session when session IDs exist.

    Fetches a pool of traces larger than `sample_size`, then:
    - If sessions exist: randomly selects `sample_size` sessions and returns
      all traces from those sessions.
    - If no sessions: randomly selects `sample_size` individual traces.
    """
    pool_size = sample_size * pool_multiplier
    pool = mlflow.search_traces(max_results=pool_size, **search_kwargs)
    if not pool:
        return []

    sessions: dict[str, list[Trace]] = defaultdict(list)
    no_session: list[Trace] = []
    for trace in pool:
        if sid := _get_session_id(trace):
            sessions[sid].append(trace)
        else:
            no_session.append(trace)

    if sessions:
        session_ids = list(sessions.keys())
        k = min(sample_size, len(session_ids))
        selected = random.sample(session_ids, k)
        result = [t for sid in selected for t in sessions[sid]]
        _logger.info(
            "Sampled %d sessions (%d traces) from pool of %d sessions",
            k,
            len(result),
            len(session_ids),
        )
        return result

    k = min(sample_size, len(no_session))
    result = random.sample(no_session, k)
    _logger.info("Sampled %d traces from pool of %d", k, len(no_session))
    return result


def _has_session_ids(traces: list[Trace]) -> bool:
    for t in traces:
        if (t.info.tags or {}).get("mlflow.trace.session_id"):
            return True
        if (t.info.trace_metadata or {}).get("mlflow.trace.session"):
            return True
    return False


def _build_default_satisfaction_scorer(model: str | None, use_conversation: bool) -> Scorer:
    instructions = _build_satisfaction_instructions(use_conversation=use_conversation)
    return make_judge(
        name=_DEFAULT_SCORER_NAME,
        instructions=instructions,
        model=model,
        feedback_value_type=bool,
    )


def _test_scorer(scorer: Scorer, trace: Trace) -> None:
    result = mlflow.genai.evaluate(data=[trace], scorers=[scorer])

    if result.run_id:
        try:
            mlflow.MlflowClient().delete_run(result.run_id)
        except Exception:
            _logger.debug("Failed to delete test run %s", result.run_id)

    if result.result_df is None:
        return

    value_col = f"{scorer.name}/value"
    if value_col not in result.result_df.columns:
        return

    if result.result_df[value_col].iloc[0] is not None:
        return

    error_msg = None
    if "trace" in result.result_df.columns:
        result_trace = result.result_df["trace"].iloc[0]
        if isinstance(result_trace, str):
            result_trace = Trace.from_json(result_trace)
        if result_trace is not None:
            for assessment in result_trace.info.assessments:
                if isinstance(assessment, Feedback) and assessment.name == scorer.name:
                    error_msg = assessment.error_message
                    break

    raise mlflow.exceptions.MlflowException(
        f"Scorer '{scorer.name}' failed on test trace "
        f"{trace.info.trace_id}: {error_msg or 'unknown error (check model API logs)'}"
    )


# Span types that represent generic LLM plumbing, not meaningful execution steps.
_GENERIC_SPAN_TYPES = {"LLM", "CHAT_MODEL", "EMBEDDING"}


def _extract_execution_path(trace: Trace) -> str:
    """
    Extract a compact execution path from a trace's span tree.

    Filters out generic LLM/embedding spans and returns the meaningful
    execution steps: sub-agents routed to, tools called, etc.
    Returns a string like "ask_sports_assistant > get_live_scores, web_search"
    or "(no routing)" if only generic spans were found.
    """
    spans = trace.data.spans
    if not spans:
        return "(no spans)"

    # Build parent-child map
    children: dict[str | None, list[object]] = defaultdict(list)
    for s in spans:
        children[s.parent_id].append(s)

    # Collect meaningful span names in tree order, tracking depth
    meaningful: list[tuple[str, int, str]] = []  # (name, depth, status)

    def _walk(span, depth: int):
        span_type = getattr(span, "span_type", "") or ""
        if span_type not in _GENERIC_SPAN_TYPES:
            status = ""
            if span.status and span.status.status_code == SpanStatusCode.ERROR:
                status = " [ERROR]"
            meaningful.append((span.name, depth, status))
        for child in children.get(span.span_id, []):
            _walk(child, depth + 1)

    roots = children.get(None, [])
    if not roots:
        roots = spans[:1]
    for root in roots:
        _walk(root, 0)

    if not meaningful:
        return "(no routing)"

    # Skip the root span (it's always the top-level entry point)
    non_root = [(name, depth, status) for name, depth, status in meaningful if depth > 0]
    if not non_root:
        return "(no routing)"

    # Build hierarchical path: group by depth
    # Depth 1 = sub-agent/tool called by orchestrator
    # Depth 2+ = tools called by that sub-agent
    top_level = []
    sub_items: dict[str, list[str]] = defaultdict(list)
    current_top = None

    for name, depth, status in non_root:
        entry = f"{name}{status}"
        if depth == 1:
            current_top = name
            top_level.append(entry)
        elif current_top:
            sub_items[current_top].append(entry)

    parts = []
    for t in top_level:
        base_name = t.replace(" [ERROR]", "")
        if subs := sub_items.get(base_name, []):
            # Deduplicate while preserving order
            seen = set()
            unique_subs = []
            for s in subs:
                if s not in seen:
                    seen.add(s)
                    unique_subs.append(s)
            parts.append(f"{t} > {', '.join(unique_subs)}")
        else:
            parts.append(t)

    return " | ".join(parts) if parts else "(no routing)"


def _extract_execution_paths_for_session(traces: list[Trace]) -> str:
    """
    Extract combined execution paths across all traces in a session.

    Deduplicates paths and joins them, giving a compact summary of
    what the agent did across the entire conversation.
    """
    paths = []
    seen = set()
    for t in traces:
        path = _extract_execution_path(t)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return "; ".join(paths) if paths else "(no routing)"


def _extract_span_errors(trace: Trace, max_length: int = 500) -> str:
    """Collect error messages from trace spans (status descriptions and exception events)."""
    spans = trace.data.spans
    if not spans:
        return ""

    errors: list[str] = []
    seen: set[str] = set()

    for span in spans:
        if not (span.status and span.status.status_code == SpanStatusCode.ERROR):
            continue
        # Status description (e.g., "Exception: Connection failed")
        if span.status.description:
            msg = f"{span.name}: {span.status.description}"
            if msg not in seen:
                seen.add(msg)
                errors.append(msg)
        # Exception events carry type + message
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
    combined = "; ".join(errors)
    return combined[:max_length] if len(combined) > max_length else combined


def _group_traces_by_session(
    traces: list[Trace],
) -> dict[str, list[Trace]]:
    """
    Group traces by session ID. Traces without a session become standalone
    single-trace "sessions" keyed by their trace_id.
    Each group is sorted by timestamp_ms.
    """
    groups: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        sid = _get_session_id(trace) or trace.info.trace_id
        groups[sid].append(trace)

    for traces_in_group in groups.values():
        traces_in_group.sort(key=lambda t: t.info.timestamp_ms)

    return dict(groups)


def _embed_texts(texts: list[str], embedding_model: str) -> list[list[float]]:
    import litellm

    from mlflow.metrics.genai.model_utils import _parse_model_uri

    scheme, path = _parse_model_uri(embedding_model)
    if scheme in ("endpoints", "databricks"):
        litellm_model = f"databricks/{path}"
    else:
        litellm_model = f"{scheme}/{path}"

    # Embedding APIs reject empty strings; replace with a placeholder.
    sanitized = [t or "(empty)" for t in texts]
    response = litellm.embedding(model=litellm_model, input=sanitized)
    return [item["embedding"] for item in response.data]


def _extract_failure_labels(
    analyses: list[_ConversationAnalysis],
    model: str,
) -> list[str]:
    """
    Extract short failure labels that combine execution path with symptom.

    Each label has the format:
    ``[execution_path] symptom description``

    The execution path comes from the trace spans (which sub-agents/tools
    were called). The symptom is extracted by an LLM from the triage
    rationale. Together they enable clustering by "what the agent did"
    and "how it failed."
    """
    import litellm

    from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
    from mlflow.genai.discovery.constants import _FAILURE_LABEL_SYSTEM_PROMPT
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    scheme, path = _parse_model_uri(model)
    if scheme in ("endpoints", "databricks"):
        litellm_model = f"databricks/{path}"
    else:
        litellm_model = f"{scheme}/{path}"

    def _label_one(a: _ConversationAnalysis) -> str:
        rationale = a.surface[:800]
        response = litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": _FAILURE_LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": rationale},
            ],
            max_tokens=1000,
        )
        symptom = response.choices[0].message.content.strip()
        exec_path = a.execution_path or "(no routing)"
        return f"[{exec_path}] {symptom}"

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(analyses))
    labels: list[str | None] = [None] * len(analyses)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="label") as executor:
        future_to_idx = {executor.submit(_label_one, a): i for i, a in enumerate(analyses)}
        for future in as_completed(future_to_idx):
            labels[future_to_idx[future]] = future.result()

    return labels


def _cluster_analyses(
    analyses: list[_ConversationAnalysis],
    embedding_model: str,
    max_issues: int,
    labels: list[str] | None = None,
    label_model: str | None = None,
) -> list[list[int]]:
    """
    Group failure analyses into issue clusters.

    When ``labels`` are provided (short LLM-extracted domain+failure labels),
    uses an LLM to group them by domain/topic — this produces much better
    domain-aware groupings than pure embedding similarity.

    Falls back to embedding-based agglomerative clustering when labels are
    not available.
    """
    if len(analyses) == 1:
        return [[0]]

    if labels:
        return _cluster_by_llm(labels, max_issues, label_model)

    return _cluster_by_embeddings(analyses, embedding_model, max_issues)


def _cluster_by_llm(
    labels: list[str],
    max_issues: int,
    model: str | None = None,
) -> list[list[int]]:
    """
    Use an LLM to group failure labels by execution path and symptom.

    Each label has the format ``[execution_path] symptom``, where the
    execution path shows which sub-agents/tools were called. The LLM
    groups labels that share similar execution paths AND similar failure
    symptoms into coherent issue categories.
    """
    import litellm

    from mlflow.genai.discovery.constants import _DEFAULT_JUDGE_MODEL
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    model = model or _DEFAULT_JUDGE_MODEL
    scheme, path = _parse_model_uri(model)
    if scheme in ("endpoints", "databricks"):
        litellm_model = f"databricks/{path}"
    else:
        litellm_model = f"{scheme}/{path}"

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

    response = litellm.completion(
        model=litellm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        _logger.warning(
            "LLM returned empty content for label grouping "
            "(finish_reason=%s), falling back to singletons",
            response.choices[0].finish_reason,
        )
        return [[i] for i in range(len(labels))]
    result = json.loads(content)

    # Handle both {"groups": [...]} and direct list formats
    groups = (
        result if isinstance(result, list) else result.get("groups", result.get("categories", []))
    )

    # Convert to list of index lists
    all_indices = set()
    cluster_groups: list[list[int]] = []
    for g in groups:
        indices = [i for i in g["indices"] if 0 <= i < len(labels)]
        if indices := [i for i in indices if i not in all_indices]:
            cluster_groups.append(indices)
            all_indices.update(indices)

    # Add any missing indices as singletons
    cluster_groups.extend([i] for i in range(len(labels)) if i not in all_indices)

    # Cap at max_issues
    if len(cluster_groups) > max_issues:
        cluster_groups.sort(key=len, reverse=True)
        cluster_groups = cluster_groups[:max_issues]

    return cluster_groups


def _cluster_by_embeddings(
    analyses: list[_ConversationAnalysis],
    embedding_model: str,
    max_issues: int,
) -> list[list[int]]:
    """Fallback: embedding-based agglomerative clustering on surface text."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    texts = [a.surface for a in analyses]
    vecs = np.array(_embed_texts(texts, embedding_model))

    dists = pdist(vecs, metric="cosine")
    dists = np.nan_to_num(dists, nan=1.0)
    link = linkage(dists, method="average")
    cluster_ids = fcluster(link, t=0.50, criterion="distance")

    clusters: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        clusters.setdefault(int(cid), []).append(idx)

    all_clusters = list(clusters.values())

    if len(all_clusters) > max_issues:
        all_clusters.sort(key=len, reverse=True)
        all_clusters = all_clusters[:max_issues]

    return all_clusters


def _summarize_cluster(
    cluster_indices: list[int],
    analyses: list[_ConversationAnalysis],
    analysis_model: str,
) -> _IdentifiedIssue:
    cluster_analyses = [analyses[i] for i in cluster_indices]
    parts = []
    for i, a in zip(cluster_indices, cluster_analyses):
        entry = f"[{i}] {a.surface}"
        if a.execution_path:
            entry += f"\n  execution_path: {a.execution_path}"
        parts.append(entry)
    analyses_text = "\n\n".join(parts)

    from mlflow.types.llm import ChatMessage

    result = get_chat_completions_with_structured_output(
        model_uri=analysis_model,
        messages=[
            ChatMessage(role="system", content=_CLUSTER_SUMMARY_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(f"Cluster of {len(cluster_indices)} analyses:\n\n{analyses_text}"),
            ),
        ],
        output_schema=_IdentifiedIssue,
    )
    result.example_indices = cluster_indices
    return result


def _extract_failing_traces(
    eval_result: EvaluationResult,
    scorer_names: str | list[str],
    original_traces: list[Trace] | None = None,
) -> tuple[list[Trace], dict[str, str]]:
    """
    Extract failing traces and their rationales from an evaluation result.

    ``scorer_names`` can be a single scorer name or a list. When multiple
    names are provided, a trace is considered failing if ANY scorer marks
    it as ``False``, and rationales from all failing scorers are combined.

    When ``original_traces`` is provided, maps results back to the original
    trace objects by DataFrame position (the evaluate framework may assign
    new trace IDs during scoring).  Rationales are extracted from the
    DataFrame column first; if absent, from the trace assessment.
    """
    if isinstance(scorer_names, str):
        scorer_names = [scorer_names]

    failing: list[Trace] = []
    rationales: dict[str, str] = {}
    df = eval_result.result_df
    if df is None:
        return failing, rationales

    # Filter to scorer names whose value column actually exists
    active_scorers = [s for s in scorer_names if f"{s}/value" in df.columns]
    if not active_scorers:
        return failing, rationales

    for idx, (_, row) in enumerate(df.iterrows()):
        # Check if ANY scorer marks this row as failing
        row_failing_scorers: list[str] = []
        for scorer_name in active_scorers:
            val = row.get(f"{scorer_name}/value")
            if val is not None and not bool(val):
                row_failing_scorers.append(scorer_name)

        if not row_failing_scorers:
            continue

        # Map back to original trace by position when available.
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
                            a.rationale
                            for a in reversed(eval_trace.info.assessments)
                            if isinstance(a, Feedback) and a.name == scorer_name and a.rationale
                        ),
                        "",
                    )
            if rationale:
                row_rationales.append(rationale)
        rationales[trace.info.trace_id] = "; ".join(row_rationales)

    return failing, rationales


def _log_discovery_artifacts(run_id: str, artifacts: dict[str, str]) -> None:
    if not run_id:
        return
    client = mlflow.MlflowClient()
    for filename, content in artifacts.items():
        try:
            client.log_text(run_id, content, filename)
        except Exception:
            _logger.warning("Failed to log %s to run %s", filename, run_id, exc_info=True)


def _build_summary(issues: list[object], total_traces: int) -> str:
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
