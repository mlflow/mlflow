from __future__ import annotations

import logging
import random
from collections import defaultdict

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    _CLUSTER_SUMMARY_SYSTEM_PROMPT,
    _DEEP_ANALYSIS_SYSTEM_PROMPT,
    _DEFAULT_SCORER_NAME,
    _ERROR_CHAR_LIMIT,
    _SCORER_GENERATION_SYSTEM_PROMPT,
    _SPAN_IO_CHAR_LIMIT,
    _TEMPLATE_VARS,
    _TRACE_IO_CHAR_LIMIT,
    _TRIM_MARKER,
    _build_satisfaction_instructions,
)
from mlflow.genai.discovery.entities import (
    _ConversationAnalysis,
    _ConversationAnalysisLLMResult,
    _IdentifiedIssue,
    _ScorerInstructionsResult,
    _ScorerSpec,
)
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.judges.utils.invocation_utils import (
    get_chat_completions_with_structured_output,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.types.llm import ChatMessage

_logger = logging.getLogger(__name__)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + _TRIM_MARKER


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


def _ensure_template_var(instructions: str) -> str:
    if any(var in instructions for var in _TEMPLATE_VARS):
        return instructions
    return f"Analyze the following {{{{ trace }}}} and determine:\n\n{instructions}"


def _get_existing_score(trace: Trace, scorer_name: str) -> bool | None:
    for assessment in trace.info.assessments:
        if isinstance(assessment, Feedback) and assessment.name == scorer_name:
            if isinstance(assessment.value, bool):
                return assessment.value
    return None


def _partition_by_existing_scores(
    traces: list[Trace],
    scorer_name: str,
) -> tuple[list[Trace], list[Trace], list[Trace]]:
    negative: list[Trace] = []
    positive: list[Trace] = []
    needs_scoring: list[Trace] = []
    for trace in traces:
        score = _get_existing_score(trace, scorer_name)
        if score is True:
            positive.append(trace)
        elif score is False:
            negative.append(trace)
        else:
            needs_scoring.append(trace)
    return negative, positive, needs_scoring


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


def _build_span_tree(spans: list[object]) -> str:
    if not spans:
        return "    (no spans)"

    span_by_id: dict[str, object] = {}
    children: dict[str | None, list[object]] = {}
    for span in spans:
        span_by_id[span.span_id] = span
        children.setdefault(span.parent_id, []).append(span)

    def _format_span(span, depth: int) -> list[str]:
        indent = "    " + "  " * depth
        duration_ms = ""
        if span.end_time_ns is not None and span.start_time_ns is not None:
            duration_ms = f", {(span.end_time_ns - span.start_time_ns) // 1_000_000}ms"

        status = span.status.status_code.value if span.status else "UNKNOWN"
        span_type = getattr(span, "span_type", "UNKNOWN") or "UNKNOWN"
        model = getattr(span, "model_name", None)
        model_str = f", model={model}" if model else ""

        line = f"{indent}{span.name} ({span_type}, {status}{duration_ms}{model_str})"
        lines = [line]

        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            if span.status.description:
                lines.append(
                    f"{indent}  ERROR: {_trim(span.status.description, _ERROR_CHAR_LIMIT)}"
                )
            for event in getattr(span, "events", []):
                if event.name == "exception":
                    exc_type = event.attributes.get("exception.type", "")
                    exc_msg = event.attributes.get("exception.message", "")
                    if exc_type or exc_msg:
                        exc_text = _trim(f"{exc_type}: {exc_msg}", _ERROR_CHAR_LIMIT)
                        lines.append(f"{indent}  EXCEPTION: {exc_text}")

        inputs = getattr(span, "inputs", None)
        outputs = getattr(span, "outputs", None)
        if inputs:
            lines.append(f"{indent}  in: {_trim(str(inputs), _SPAN_IO_CHAR_LIMIT)}")
        if outputs:
            lines.append(f"{indent}  out: {_trim(str(outputs), _SPAN_IO_CHAR_LIMIT)}")

        for child in children.get(span.span_id, []):
            lines.extend(_format_span(child, depth + 1))
        return lines

    roots = children.get(None, [])
    if not roots:
        roots = spans

    result_lines: list[str] = []
    for root in roots:
        result_lines.extend(_format_span(root, 0))
    return "\n".join(result_lines)


def _get_root_span_io(trace: Trace) -> tuple[str, str]:
    if trace.data.spans:
        root = next(
            (s for s in trace.data.spans if s.parent_id is None),
            trace.data.spans[0],
        )
        request = str(root.inputs) if root.inputs else ""
        response = str(root.outputs) if root.outputs else ""
    else:
        request = ""
        response = ""
    if not request:
        request = trace.info.request_preview or ""
    if not response:
        response = trace.info.response_preview or ""
    return request, response


def _build_assessments_section(trace: Trace) -> str:
    lines: list[str] = []
    for assessment in trace.info.assessments:
        if not isinstance(assessment, Feedback):
            continue
        entry = f"    {assessment.name}: {assessment.value}"
        if assessment.rationale:
            entry += f" — {assessment.rationale}"
        lines.append(entry)
    if not lines:
        return ""
    return "  Assessments:\n" + "\n".join(lines)


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
    children: dict[str | None, list] = defaultdict(list)
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
        subs = sub_items.get(base_name, [])
        if subs:
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


def _build_enriched_trace_summary(index: int, trace: Trace, rationale: str) -> str:
    request, response = _get_root_span_io(trace)
    request = _trim(request, _TRACE_IO_CHAR_LIMIT)
    response = _trim(response, _TRACE_IO_CHAR_LIMIT)
    duration = trace.info.execution_duration or 0
    span_tree = _build_span_tree(trace.data.spans)
    assessments = _build_assessments_section(trace)
    parts = [
        f"[{index}] trace_id={trace.info.trace_id}",
        f"  Input: {request}",
        f"  Output: {response}",
        f"  Duration: {duration}ms | Failure rationale: {rationale}",
    ]
    if assessments:
        parts.append(assessments)
    parts.append(f"  Span tree:\n{span_tree}")
    return "\n".join(parts)


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


def _build_conversation_for_analysis(
    session_traces: list[Trace],
    rationale_map: dict[str, str],
) -> tuple[str, str]:
    """
    Build a triage preamble and annotated conversation for one session.

    Returns:
        A tuple of (triage_preamble, conversation_text). The preamble lists
        every failing trace ID with its violated expectations. The conversation
        text has triage annotations on the assistant (not user) messages.
    """
    from mlflow.genai.utils.trace_utils import resolve_conversation_from_session

    conversation = resolve_conversation_from_session(
        session_traces,
        include_trace_ids=True,
    )

    # Identify failing trace_ids in this session (preserving insertion order)
    failing_rationales: dict[str, str] = {}
    for t in session_traces:
        tid = t.info.trace_id
        if tid in rationale_map and tid not in failing_rationales:
            failing_rationales[tid] = rationale_map[tid]

    # Build triage preamble
    preamble_lines = ["TRIAGE SUMMARY — investigate ONLY these failures:"]
    for tid, rationale in failing_rationales.items():
        preamble_lines.append(f"  • trace_id={tid}")
        preamble_lines.append(f"    Violated expectations: {rationale}")
    triage_preamble = "\n".join(preamble_lines)

    # Track which trace_ids still need annotation (on assistant message)
    pending_annotation = set(failing_rationales.keys())

    # Build conversation lines with annotations on assistant messages
    lines = []
    for msg in conversation:
        trace_id = msg.get("trace_id", "")
        role = msg["role"]
        content = msg["content"]
        line = f"[trace_id={trace_id}] {role}: {content}"

        # Annotate the assistant response (not the user question)
        if role == "assistant" and trace_id in pending_annotation:
            rationale = failing_rationales[trace_id]
            line += f"\n  ** TRIAGE FAILURE: {rationale} **"
            pending_annotation.discard(trace_id)

        lines.append(line)

    return triage_preamble, "\n\n".join(lines)


def _run_deep_analysis_single(
    triage_preamble: str,
    conversation_text: str,
    analysis_model: str,
    failing_trace_ids: list[str],
) -> _ConversationAnalysis:
    """
    Run deep analysis on a single conversation with tool access.
    The triage preamble tells the LLM exactly which traces failed and why,
    so it can make targeted tool calls only on those traces.
    """
    from mlflow.genai.judges.tools import list_judge_tools

    judge_tools = list_judge_tools()
    tools = [tool.get_definition().to_dict() for tool in judge_tools]

    llm_result = get_chat_completions_with_structured_output(
        model_uri=analysis_model,
        messages=[
            ChatMessage(role="system", content=_DEEP_ANALYSIS_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(f"{triage_preamble}\n\nCONVERSATION:\n\n{conversation_text}"),
            ),
        ],
        output_schema=_ConversationAnalysisLLMResult,
        tools=tools,
    )
    return _ConversationAnalysis(
        surface=llm_result.surface,
        root_cause=llm_result.root_cause,
        symptoms=llm_result.symptoms,
        domain=llm_result.domain,
        affected_trace_ids=failing_trace_ids,
        severity=llm_result.severity,
    )


def _run_deep_analysis(
    session_groups: dict[str, list[Trace]],
    rationale_map: dict[str, str],
    analysis_model: str,
) -> list[_ConversationAnalysis]:
    """
    Run deep analysis per-conversation for sessions with failures, in parallel.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS

    items: list[tuple[str, list[Trace]]] = [
        (session_id, session_traces)
        for session_id, session_traces in session_groups.items()
        if any(t.info.trace_id in rationale_map for t in session_traces)
    ]
    if not items:
        return []

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(items))
    _logger.info(
        "Deep analysis: %d sessions, %d workers",
        len(items),
        max_workers,
    )

    def _analyze(idx: int, session_id: str, session_traces: list[Trace]) -> _ConversationAnalysis:
        t0 = time.time()
        preamble, conv_text = _build_conversation_for_analysis(session_traces, rationale_map)
        failing_trace_ids = [
            t.info.trace_id for t in session_traces if t.info.trace_id in rationale_map
        ]
        _logger.info(
            "  [%d/%d] session %s: built conversation (%d chars) in %.1fs, calling LLM...",
            idx + 1,
            len(items),
            session_id[:20],
            len(conv_text),
            time.time() - t0,
        )
        t1 = time.time()
        result = _run_deep_analysis_single(preamble, conv_text, analysis_model, failing_trace_ids)
        _logger.info(
            "  [%d/%d] session %s: LLM analysis done in %.1fs",
            idx + 1,
            len(items),
            session_id[:20],
            time.time() - t1,
        )
        return result

    analyses: list[_ConversationAnalysis] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_analyze, idx, sid, traces): idx
            for idx, (sid, traces) in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            analyses[idx] = future.result()

    return analyses


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

    from mlflow.genai.discovery.constants import _FAILURE_LABEL_SYSTEM_PROMPT
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    scheme, path = _parse_model_uri(model)
    if scheme in ("endpoints", "databricks"):
        litellm_model = f"databricks/{path}"
    else:
        litellm_model = f"{scheme}/{path}"

    labels: list[str] = []
    for a in analyses:
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
        labels.append(f"[{exec_path}] {symptom}")

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
    import json

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
        "  1. They went through similar execution paths (same sub-agents/tools)\n"
        "  2. They failed in a similar way (same symptom pattern)\n\n"
        "Labels with different execution paths should generally be in DIFFERENT "
        "groups, even if the symptom sounds similar — different paths mean "
        "different parts of the system are involved.\n\n"
        "Rules:\n"
        "- Each group should have a short readable name (3-8 words) followed by domain "
        "keywords in brackets, e.g. 'Incomplete response details [sports, weather]'. "
        "The brackets should list the user-facing domains affected (1-3 keywords)\n"
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
        indices = [i for i in indices if i not in all_indices]
        if indices:
            cluster_groups.append(indices)
            all_indices.update(indices)

    # Add any missing indices as singletons
    for i in range(len(labels)):
        if i not in all_indices:
            cluster_groups.append([i])

    # Cap at max_issues
    if len(cluster_groups) > max_issues:
        cluster_groups.sort(key=len, reverse=True)
        cluster_groups = cluster_groups[:max_issues]

    return cluster_groups


def _merge_similar_clusters(
    groups: list[list[int]],
    labels: list[str],
    max_issues: int,
    model: str | None = None,
) -> list[list[int]]:
    """
    Second pass: merge clusters that describe the same user-facing issue
    despite different execution paths.

    The initial grouping is conservative — it keeps clusters with different
    execution paths separate. This pass asks the LLM to identify groups
    that should be merged because they represent the same issue from the
    user's perspective (e.g., "pause music ignored via end_conversation"
    and "pause music ignored via pause_spotify error").
    """
    import json

    import litellm

    from mlflow.genai.discovery.constants import _DEFAULT_JUDGE_MODEL
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    if len(groups) <= 1:
        return groups

    model = model or _DEFAULT_JUDGE_MODEL
    scheme, path = _parse_model_uri(model)
    if scheme in ("endpoints", "databricks"):
        litellm_model = f"databricks/{path}"
    else:
        litellm_model = f"{scheme}/{path}"

    # Build a summary of each group for the LLM
    group_summaries = []
    for gi, group in enumerate(groups):
        group_labels = [labels[idx] for idx in group]
        summary = f"[Group {gi}] ({len(group)} members):\n"
        for lbl in group_labels:
            summary += f"  - {lbl}\n"
        group_summaries.append(summary)

    numbered_groups = "\n".join(group_summaries)
    prompt = (
        f"Below are {len(groups)} groups of failure labels from an AI agent, "
        "already grouped by execution path and symptom.\n\n"
        "Some groups may describe the SAME user-facing issue despite going through "
        "different execution paths. For example, 'pause music ignored via "
        "end_conversation' and 'pause music ignored via pause_spotify error' are "
        "the same issue from the user's perspective: media control commands not "
        "executed.\n\n"
        "Identify which groups should be MERGED because they represent the same "
        "user-facing problem. Only merge groups where the failure experienced "
        "by the user is genuinely the same — don't merge just because symptoms "
        "sound vaguely similar.\n\n"
        f"Groups:\n{numbered_groups}\n\n"
        'Return a JSON object with a "merges" key containing an array of arrays. '
        "Each inner array lists the group indices (ints) that should be merged. "
        "Groups not listed in any merge set remain unchanged.\n"
        'If no merges are needed, return {"merges": []}.\n'
        "Return ONLY the JSON, no explanation."
    )

    response = litellm.completion(
        model=litellm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        return groups

    result = json.loads(content)
    merge_sets = result.get("merges", [])

    if not merge_sets:
        return groups

    # Apply merges
    merged_group_indices: set[int] = set()
    merged_groups: list[list[int]] = []
    for merge_set in merge_sets:
        valid_gis = [gi for gi in merge_set if 0 <= gi < len(groups)]
        if len(valid_gis) < 2:
            continue
        combined = []
        for gi in valid_gis:
            combined.extend(groups[gi])
            merged_group_indices.add(gi)
        merged_groups.append(combined)

    # Add unmerged groups
    final_groups = []
    for gi, group in enumerate(groups):
        if gi not in merged_group_indices:
            final_groups.append(group)
    final_groups.extend(merged_groups)

    # Cap at max_issues
    if len(final_groups) > max_issues:
        final_groups.sort(key=len, reverse=True)
        final_groups = final_groups[:max_issues]

    _logger.info(
        "Merge pass: %d groups → %d groups (%d merges applied)",
        len(groups),
        len(final_groups),
        len([m for m in merge_sets if len(m) >= 2]),
    )

    return final_groups


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
    analyses_text = "\n\n".join(
        f"[{i}] surface: {a.surface}\n"
        f"  root_cause: {a.root_cause}\n"
        f"  symptoms: {a.symptoms}\n"
        f"  domain: {a.domain}"
        for i, a in zip(cluster_indices, cluster_analyses)
    )

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


def _generate_scorer_specs(
    issue: _IdentifiedIssue,
    example_analyses: list[_ConversationAnalysis],
    judge_model: str,
) -> list[_ScorerSpec]:
    examples_text = "\n".join(
        f"- [surface: {a.surface}] {a.symptoms} (root cause: {a.root_cause[:200]})"
        for a in example_analyses
    )
    result = get_chat_completions_with_structured_output(
        model_uri=judge_model,
        messages=[
            ChatMessage(role="system", content=_SCORER_GENERATION_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Issue: {issue.name}\n"
                    f"Description: {issue.description}\n"
                    f"Root cause: {issue.root_cause}\n\n"
                    f"Example failures:\n{examples_text}\n\n"
                    "Write detection scorer(s) for this issue. Use one scorer if the issue is "
                    "a single criterion, or multiple scorers if it involves independent criteria."
                ),
            ),
        ],
        output_schema=_ScorerInstructionsResult,
    )
    for spec in result.scorers:
        spec.detection_instructions = _ensure_template_var(spec.detection_instructions)
    return result.scorers


def _extract_failing_traces(
    eval_result: EvaluationResult,
    scorer_name: str,
    original_traces: list[Trace] | None = None,
) -> tuple[list[Trace], dict[str, str]]:
    """
    Extract failing traces and their rationales from an evaluation result.

    When ``original_traces`` is provided, maps results back to the original
    trace objects by DataFrame position (the evaluate framework may assign
    new trace IDs during scoring).  Rationales are extracted from the
    DataFrame column first; if absent, from the trace assessment.
    """
    failing: list[Trace] = []
    rationales: dict[str, str] = {}
    df = eval_result.result_df
    if df is None:
        return failing, rationales

    value_col = f"{scorer_name}/value"
    rationale_col = f"{scorer_name}/rationale"
    if value_col not in df.columns:
        return failing, rationales

    has_rationale_col = rationale_col in df.columns

    for idx, (_, row) in enumerate(df.iterrows()):
        val = row.get(value_col)
        if val is None or bool(val):
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

        # Extract rationale from column or from the eval trace's assessment.
        rationale = ""
        if has_rationale_col:
            rationale = str(row.get(rationale_col, "") or "")
        if not rationale:
            eval_trace = row.get("trace")
            if eval_trace is not None:
                if isinstance(eval_trace, str):
                    eval_trace = Trace.from_json(eval_trace)
                # Use reversed() to get the most recent assessment
                # (traces accumulate assessments across runs).
                rationale = next(
                    (
                        a.rationale
                        for a in reversed(eval_trace.info.assessments)
                        if isinstance(a, Feedback) and a.name == scorer_name and a.rationale
                    ),
                    "",
                )
        rationales[trace.info.trace_id] = rationale

    return failing, rationales


def _compute_frequencies(
    eval_result: EvaluationResult,
    scorer_names: list[str],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    frequencies: dict[str, float] = {}
    rationale_examples: dict[str, list[str]] = {}
    df = eval_result.result_df
    if df is None:
        return frequencies, rationale_examples

    total = len(df)
    for name in scorer_names:
        value_col = f"{name}/value"
        rationale_col = f"{name}/rationale"
        if value_col not in df.columns:
            continue

        affected = df[value_col].eq(False).sum()
        frequencies[name] = affected / total if total > 0 else 0.0

        examples = []
        if rationale_col in df.columns:
            for _, row in df.iterrows():
                val = row.get(value_col)
                if val is not None and not bool(val) and len(examples) < 3:
                    if r := row.get(rationale_col):
                        examples.append(str(r))
        rationale_examples[name] = examples

    return frequencies, rationale_examples


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
            f"confidence: {issue.confidence}/100)\n\n"
            f"{issue.description}\n\n"
            f"**Root cause:** {issue.root_cause}\n"
        )
    return "\n".join(lines)
