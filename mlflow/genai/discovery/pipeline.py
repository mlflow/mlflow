from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.discovery.constants import (
    CONFIDENCE_ORDER,
    DEFAULT_ANALYSIS_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TRIAGE_SAMPLE_SIZE,
    MIN_CONFIDENCE,
    MIN_EXAMPLES,
)
from mlflow.genai.discovery.entities import (
    MAX_EXAMPLE_TRACE_IDS,
    DiscoverIssuesResult,
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.utils import (
    _build_default_satisfaction_scorer,
    _build_summary,
    _cluster_analyses,
    _extract_execution_path,
    _extract_execution_paths_for_session,
    _extract_failing_traces,
    _extract_failure_labels,
    _extract_span_errors,
    _get_session_id,
    _group_traces_by_session,
    _log_discovery_artifacts,
    _sample_traces,
    _summarize_cluster,
    _test_scorer,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

_NO_ISSUE_PATTERNS = frozenset(
    {
        "no issues",
        "no issue",
        "no problems",
        "no failures",
        "no errors",
        "no real issue",
        "no failure",
        "no deficiency",
        "no significant issue",
        "nothing wrong",
        "goals were achieved",
        "functioning correctly",
        "operating as expected",
        "working as intended",
        "performed well",
        "no discernible issue",
    }
)


def confidence_gte(a: str, b: str) -> bool:
    return CONFIDENCE_ORDER.get(a, -1) >= CONFIDENCE_ORDER.get(b, 0)


def confidence_max(a: str, b: str) -> str:
    return a if CONFIDENCE_ORDER.get(a, 0) >= CONFIDENCE_ORDER.get(b, 0) else b


class _TokenCounter:
    """Thread-safe accumulator for LLM token usage across pipeline phases."""

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def track(self, response) -> None:
        with self._lock:
            usage = getattr(response, "usage", None)
            if usage:
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            if hidden := getattr(response, "_hidden_params", None):
                if cost := hidden.get("response_cost"):
                    self.cost_usd += cost

    def to_dict(self) -> dict:
        result = {}
        total = self.input_tokens + self.output_tokens
        if total > 0:
            result["input_tokens"] = self.input_tokens
            result["output_tokens"] = self.output_tokens
            result["total_tokens"] = total
        if self.cost_usd > 0:
            result["cost_usd"] = round(self.cost_usd, 6)
        return result


def _is_non_issue(issue: _IdentifiedIssue) -> bool:
    from mlflow.genai.discovery.constants import NO_ISSUE_KEYWORD

    if NO_ISSUE_KEYWORD.lower() in issue.name.lower():
        return True
    combined = f"{issue.name} {issue.description} {issue.root_cause}".lower()
    return any(pattern in combined for pattern in _NO_ISSUE_PATTERNS)


def _extract_assessment_rationale(trace: Trace, scorer_name: str) -> str:
    return next(
        (
            a.rationale
            for a in trace.info.assessments
            if isinstance(a, Feedback) and a.name == scorer_name and a.rationale
        ),
        "",
    )


def _collect_example_trace_ids(
    issue: _IdentifiedIssue,
    analyses: list[_ConversationAnalysis],
) -> list[str]:
    trace_ids = []
    for idx in issue.example_indices:
        if 0 <= idx < len(analyses):
            trace_ids.extend(analyses[idx].affected_trace_ids)
    return trace_ids[:MAX_EXAMPLE_TRACE_IDS]


def _recluster_singletons(
    singletons: list[_IdentifiedIssue],
    labels: list[str],
    analyses: list[_ConversationAnalysis],
    analysis_model: str,
    label_model: str,
    max_issues: int,
    token_counter=None,
) -> list[_IdentifiedIssue]:
    """Re-cluster singleton issues via a second LLM pass to find better groupings."""
    from mlflow.genai.discovery.utils import _cluster_by_llm

    if len(singletons) < 2:
        return list(singletons)

    singleton_labels = []
    for singleton in singletons:
        idx = singleton.example_indices[0]
        singleton_labels.append(labels[idx] if idx < len(labels) else singleton.name)

    new_groups = _cluster_by_llm(
        singleton_labels, max_issues, label_model, token_counter=token_counter
    )

    result: list[_IdentifiedIssue] = []
    for group in new_groups:
        if len(group) == 1:
            result.append(singletons[group[0]])
            continue
        merged_indices = [singletons[g].example_indices[0] for g in group]
        merged_issue = _summarize_cluster(
            merged_indices, analyses, analysis_model, token_counter=token_counter
        )
        if confidence_gte(merged_issue.confidence, MIN_CONFIDENCE):
            result.append(merged_issue)
        else:
            result.extend(singletons[g] for g in group)

    return result


def _format_trace_content(trace: Trace) -> str:
    """Build a compact text representation of a trace for annotation prompts."""
    parts = []
    if request := trace.data.request:
        parts.append(f"Input: {str(request)[:1000]}")
    if response := trace.data.response:
        parts.append(f"Output: {str(response)[:1000]}")
    if (exec_path := _extract_execution_path(trace)) and exec_path != "(no routing)":
        parts.append(f"Execution path: {exec_path}")
    if errors := _extract_span_errors(trace):
        parts.append(f"Errors: {errors}")
    return "\n".join(parts) if parts else "(trace content not available)"


def _annotate_issue_traces(
    issues: list[Issue],
    rationale_map: dict[str, str],
    trace_lookup: dict[str, Trace],
    model: str,
    trace_to_session: dict[str, str] | None = None,
    session_first_trace: dict[str, str] | None = None,
    token_counter=None,
) -> None:
    """Log a Feedback assessment for each issue on affected traces.

    When session information is provided, logs one annotation per (issue,
    session) on the session's first trace. Otherwise annotates each trace
    individually.
    """

    from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
    from mlflow.genai.discovery.constants import TRACE_ANNOTATION_SYSTEM_PROMPT
    from mlflow.genai.discovery.utils import _NUM_RETRIES, _to_litellm_model
    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm

    litellm_model = _to_litellm_model(model)

    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id=model,
    )

    # Build work items: (issue, target_trace_id, triage_rationale, session_id)
    # When sessions are available, log one annotation per (issue, session)
    # on the session's first trace with session metadata.
    work_items: list[tuple[Issue, str, str, str | None]] = []
    for issue in issues:
        if trace_to_session and session_first_trace:
            session_traces: dict[str, list[str]] = {}
            for tid in issue.example_trace_ids:
                sid = trace_to_session.get(tid, tid)
                session_traces.setdefault(sid, []).append(tid)
            for sid, tids in session_traces.items():
                target = session_first_trace.get(sid, tids[0])
                rationale = next((rationale_map[t] for t in tids if t in rationale_map), "")
                work_items.append((issue, target, rationale, sid))
        else:
            work_items.extend(
                (issue, trace_id, rationale_map.get(trace_id, ""), None)
                for trace_id in issue.example_trace_ids
            )

    if not work_items:
        return

    def _annotate_one(
        issue: Issue, trace_id: str, triage_rationale: str, session_id: str | None
    ) -> str | None:
        trace = trace_lookup.get(trace_id)
        trace_content = _format_trace_content(trace) if trace else "(trace not available)"

        user_content = (
            f"=== ISSUE ===\n"
            f"Name: {issue.name}\n"
            f"Description: {issue.description}\n"
            f"Root cause: {issue.root_cause}\n\n"
            f"=== TRACE ===\n"
            f"{trace_content}\n\n"
            f"=== TRIAGE JUDGE RATIONALE ===\n"
            f"{triage_rationale or '(not available)'}"
        )
        try:
            response = _invoke_litellm(
                litellm_model=litellm_model,
                messages=[
                    {"role": "system", "content": TRACE_ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tools=[],
                num_retries=_NUM_RETRIES,
                response_format=None,
                include_response_format=False,
                inference_params={"max_tokens": 4096, "temperature": 0},
            )
            if token_counter is not None:
                token_counter.track(response)
            annotation = (response.choices[0].message.content or "").strip()
        except Exception:
            _logger.debug("Failed to generate annotation for trace %s", trace_id, exc_info=True)
            annotation = (
                f"This trace was flagged for issue '{issue.name}'. "
                f"Triage rationale: {triage_rationale or '(not available)'}"
            )

        name = issue.name.removeprefix("Issue: ").removeprefix("issue: ")
        feedback_name = f"issue: {name}"
        metadata = {TraceMetadataKey.TRACE_SESSION: session_id} if session_id else None
        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=feedback_name,
                value=False,
                source=source,
                rationale=annotation,
                metadata=metadata,
            )
        except Exception:
            _logger.debug("Failed to log feedback for trace %s", trace_id, exc_info=True)
            return None
        return annotation

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(work_items))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="annotate") as executor:
        future_to_item = {
            executor.submit(_annotate_one, issue, trace_id, rationale, sid): (issue, trace_id)
            for issue, trace_id, rationale, sid in work_items
        }
        for future in as_completed(future_to_item):
            future.result()


@experimental(version="3.11.0")
def discover_issues(
    experiment_id: str | None = None,
    traces: list[Trace] | None = None,
    satisfaction_scorer: Scorer | None = None,
    additional_scorers: list[Scorer] | None = None,
    judge_model: str | None = None,
    analysis_model: str | None = None,
    embedding_model: str = "openai:/text-embedding-3-small",
    triage_sample_size: int = DEFAULT_TRIAGE_SAMPLE_SIZE,
    validation_sample_size: int | None = None,
    max_issues: int = 20,
    filter_string: str | None = None,
) -> DiscoverIssuesResult:
    """
    Discover quality and operational issues in traces.

    Runs a multi-phase pipeline:
    1. **Triage**: Scores traces for user satisfaction using the satisfaction
       scorer (and any additional scorers). Traces marked as failing by any
       scorer proceed to analysis.
    2. **Analysis**: Builds per-session analyses from triage rationales and
       human feedback assessments.
    3. **Cluster & Identify**: Embedding-based clustering of analyses into
       coherent issue groups, with LLM-based summarization and refinement.

    Args:
        experiment_id: Experiment to analyze. Defaults to the active experiment.
            Ignored when ``traces`` is provided.
        traces: Traces to analyze directly. When provided, skips sampling
            and uses these traces as the input to the pipeline.
        satisfaction_scorer: Custom scorer for triage. Defaults to a built-in
            conversation-level satisfaction judge.
        additional_scorers: Extra scorers to run alongside the satisfaction
            scorer during triage. A trace is considered failing if *any*
            scorer (satisfaction or additional) marks it as ``False``.
        judge_model: LLM used for scoring traces (satisfaction + issue detection).
            Defaults to ``"openai:/gpt-5.2-mini"``.
        analysis_model: LLM used for analysis and cluster summarization.
            Defaults to ``"openai:/gpt-5.2"``.
        embedding_model: Embedding model for semantic clustering.
            Defaults to ``"openai:/text-embedding-3-small"``.
        triage_sample_size: Number of sessions (or traces, if no session metadata
            exists) to randomly sample for the triage phase. Ignored when
            ``traces`` is provided.
        validation_sample_size: Reserved for future use (scorer generation and
            validation phases are not yet implemented).
        max_issues: Maximum distinct issues to identify.
        filter_string: Filter string passed to ``search_traces``.
            Ignored when ``traces`` is provided.

    Returns:
        A :class:`DiscoverIssuesResult` with discovered issues, run IDs,
        and a summary report.

    Example:

        .. code-block:: python

            import mlflow

            # Option 1: Auto-sample from experiment
            mlflow.set_experiment("my-genai-app")
            result = mlflow.genai.discover_issues()

            # Option 2: Pass traces directly
            traces = mlflow.search_traces(max_results=100, return_type="list")
            result = mlflow.genai.discover_issues(traces=traces)

            for issue in result.issues:
                print(f"{issue.name}: {issue.frequency:.1%} of traces affected")
    """
    pipeline_start = time.time()
    token_counter = _TokenCounter()
    judge_model = judge_model or DEFAULT_JUDGE_MODEL
    analysis_model = analysis_model or DEFAULT_ANALYSIS_MODEL

    exp_id = experiment_id or _get_experiment_id()

    if traces is not None:
        triage_traces = list(traces)
        _logger.info("Phase 1: Using %d provided traces...", len(triage_traces))
    else:
        if exp_id is None:
            raise mlflow.exceptions.MlflowException(
                "No experiment specified. Pass traces, use "
                "mlflow.set_experiment(), or pass experiment_id."
            )
        search_kwargs = {
            "filter_string": filter_string,
            "return_type": "list",
            "locations": [exp_id],
        }
        _logger.info("Phase 1: Sampling %d traces...", triage_sample_size)
        triage_traces = _sample_traces(triage_sample_size, search_kwargs)

    # Ensure all evaluation runs are logged to the source experiment
    if exp_id is not None:
        mlflow.set_experiment(experiment_id=exp_id)

    if not triage_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id="",
            summary="No traces to analyze.",
            total_traces_analyzed=0,
        )

    if satisfaction_scorer is None:
        use_conversation = any(_get_session_id(t) for t in triage_traces)
        if not use_conversation:
            _logger.info(
                "No session IDs found in trace metadata, falling back to trace-level scorer"
            )
        satisfaction_scorer = _build_default_satisfaction_scorer(judge_model, use_conversation)
        _logger.info(
            "Using %s-based satisfaction scorer",
            "session (conversation)" if use_conversation else "trace",
        )

    scorer_name = satisfaction_scorer.name
    all_scorers = [satisfaction_scorer] + (additional_scorers or [])

    _logger.info("Phase 1: Testing scorer on one trace...")
    t0 = time.time()
    _test_scorer(satisfaction_scorer, triage_traces[0])
    _logger.info("Phase 1: Test scorer took %.1fs", time.time() - t0)

    _logger.info("Phase 1: Scoring %d traces...", len(triage_traces))
    t0 = time.time()
    with mlflow.start_run(run_name="discover_issues - triage"):
        triage_eval = mlflow.genai.evaluate(
            data=triage_traces,
            scorers=all_scorers,
        )
    _logger.info("Phase 1: Triage scoring took %.1fs", time.time() - t0)

    # Aggregate judge costs from Phase 1 evaluation assessments.
    from mlflow.tracing.constant import AssessmentMetadataKey

    try:
        scored_traces = mlflow.search_traces(
            return_type="list",
            locations=[exp_id],
        )
        for trace in scored_traces:
            for assessment in trace.info.assessments or []:
                meta = getattr(assessment, "metadata", None) or {}
                if meta.get(AssessmentMetadataKey.SOURCE_RUN_ID) != triage_eval.run_id:
                    continue
                if cost := meta.get(AssessmentMetadataKey.JUDGE_COST):
                    token_counter.cost_usd += float(cost)
                if inp := meta.get(AssessmentMetadataKey.JUDGE_INPUT_TOKENS):
                    token_counter.input_tokens += int(inp)
                if out := meta.get(AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS):
                    token_counter.output_tokens += int(out)
    except Exception:
        _logger.debug("Failed to aggregate Phase 1 judge costs", exc_info=True)

    scorer_names = [s.name for s in all_scorers]
    failing_traces, rationale_map = _extract_failing_traces(
        triage_eval, scorer_names, original_traces=triage_traces
    )

    _logger.info(
        "Phase 1 complete: %d/%d traces unsatisfactory",
        len(failing_traces),
        len(triage_traces),
    )

    if not failing_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 2: Build analyses from triage rationales (one per session)
    session_groups = _group_traces_by_session(triage_traces)
    analyses: list[_ConversationAnalysis] = []
    for session_id, session_traces in session_groups.items():
        session_failing = [t for t in session_traces if t.info.trace_id in rationale_map]
        if not session_failing:
            continue
        rationales: list[str] = []
        seen_rationales: set[str] = set()

        def _add_rationale(text: str, prefix: str = "") -> None:
            if text and text not in seen_rationales:
                seen_rationales.add(text)
                rationales.append(f"{prefix}{text}" if prefix else text)

        for t in session_failing:
            _add_rationale(rationale_map[t.info.trace_id])
            _add_rationale(_extract_assessment_rationale(t, scorer_name), "[human feedback] ")
            _add_rationale(_extract_span_errors(t), "[span errors] ")
        combined_rationale = "; ".join(rationales)
        if not combined_rationale:
            continue
        surface = combined_rationale[:800]
        exec_path = _extract_execution_paths_for_session(session_failing)
        analyses.append(
            _ConversationAnalysis(
                surface=surface,
                root_cause=combined_rationale,
                symptoms=combined_rationale,
                domain="",
                affected_trace_ids=[t.info.trace_id for t in session_failing],
                severity=3,
                execution_path=exec_path,
            )
        )
    _logger.info("Phase 2: Built %d analyses from triage rationales", len(analyses))

    # Phase 3: Cluster — embedding-based agglomerative clustering + LLM refinement
    _logger.info("Phase 3: Extracting failure labels for clustering...")
    t0 = time.time()
    labels = _extract_failure_labels(analyses, judge_model, token_counter=token_counter)
    _logger.info(
        "Phase 3: Label extraction took %.1fs",
        time.time() - t0,
    )
    for i, label in enumerate(labels):
        _logger.info("  [%d] %s", i, label)

    _logger.info("Phase 3: Clustering analyses into issues...")
    t0 = time.time()
    cluster_groups = _cluster_analyses(
        analyses,
        embedding_model,
        max_issues,
        labels=labels,
        label_model=judge_model,
        token_counter=token_counter,
    )
    t_embed = time.time()
    _logger.info(
        "Phase 3: Clustering took %.1fs, produced %d clusters",
        t_embed - t0,
        len(cluster_groups),
    )

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(cluster_groups))

    def _summarize_one(gi: int, group: list[int]) -> _IdentifiedIssue:
        t_s = time.time()
        issue = _summarize_cluster(group, analyses, analysis_model, token_counter=token_counter)
        _logger.info(
            "Phase 3: Summarized cluster %d/%d (%d analyses) in %.1fs",
            gi + 1,
            len(cluster_groups),
            len(group),
            time.time() - t_s,
        )
        return issue

    summaries: list[_IdentifiedIssue] = [None] * len(cluster_groups)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="summarize") as executor:
        future_to_idx = {
            executor.submit(_summarize_one, gi, group): gi
            for gi, group in enumerate(cluster_groups)
        }
        for future in as_completed(future_to_idx):
            summaries[future_to_idx[future]] = future.result()

    # Re-split incoherent clusters: if the summarizer flags a multi-member
    # cluster as low-confidence, break it into singletons and resummarize
    # each one so the individual items get a fair standalone assessment.
    resplit_groups: list[list[int]] = []
    for group, issue in zip(cluster_groups, summaries):
        if not confidence_gte(issue.confidence, MIN_CONFIDENCE) and len(group) > 1:
            _logger.info(
                "Phase 3: Re-splitting incoherent cluster '%s' "
                "(confidence=%s, %d members) into singletons",
                issue.name,
                issue.confidence,
                len(group),
            )
            resplit_groups.extend([idx] for idx in group)

    if resplit_groups:
        resplit_summaries: list[_IdentifiedIssue] = [None] * len(resplit_groups)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="resplit") as executor:
            future_to_idx = {
                executor.submit(_summarize_one, len(cluster_groups) + ri, group): ri
                for ri, group in enumerate(resplit_groups)
            }
            for future in as_completed(future_to_idx):
                resplit_summaries[future_to_idx[future]] = future.result()
        final_groups: list[list[int]] = []
        final_summaries: list[_IdentifiedIssue] = []
        for group, issue in zip(cluster_groups, summaries):
            if not confidence_gte(issue.confidence, MIN_CONFIDENCE) and len(group) > 1:
                continue
            final_groups.append(group)
            final_summaries.append(issue)
        final_groups.extend(resplit_groups)
        final_summaries.extend(resplit_summaries)
        cluster_groups = final_groups
        summaries = final_summaries
        _logger.info("Phase 3: After re-split, %d total clusters", len(cluster_groups))

    identified: list[_IdentifiedIssue] = [
        issue
        for issue in summaries
        if confidence_gte(issue.confidence, MIN_CONFIDENCE)
        and len(issue.example_indices) >= MIN_EXAMPLES
        and not _is_non_issue(issue)
    ]

    # Deduplicate issues with identical names
    seen_names: dict[str, int] = {}
    deduped: list[_IdentifiedIssue] = []
    for issue in identified:
        key = issue.name.strip().lower()
        if key in seen_names:
            existing = deduped[seen_names[key]]
            merged_indices = list(set(existing.example_indices + issue.example_indices))
            existing.example_indices = merged_indices
            existing.confidence = confidence_max(existing.confidence, issue.confidence)
        else:
            seen_names[key] = len(deduped)
            deduped.append(issue)
    identified = deduped

    # Phase 3d: Re-cluster singletons — second LLM pass to merge orphaned
    # single-analysis issues into better groupings
    singletons = [i for i in identified if len(i.example_indices) == 1]
    multi_member = [i for i in identified if len(i.example_indices) > 1]
    if len(singletons) >= 2:
        _logger.info("Phase 3d: Re-clustering %d singleton issues...", len(singletons))
        merged = _recluster_singletons(
            singletons,
            labels,
            analyses,
            analysis_model,
            judge_model,
            max_issues,
            token_counter=token_counter,
        )
        identified = multi_member + merged
        _logger.info(
            "Phase 3d: %d singletons → %d issues after re-clustering",
            len(singletons),
            len(merged),
        )

    _logger.info(
        "Phase 3 complete: %d issues identified (%d clusters filtered out) in %.1fs",
        len(identified),
        len(cluster_groups) - len(identified),
        time.time() - t0,
    )
    for issue in identified:
        example_ids = _collect_example_trace_ids(issue, analyses)
        _logger.info(
            "  %s (confidence %s): %s | root_cause: %s | examples: %s",
            issue.name,
            issue.confidence,
            issue.description,
            issue.root_cause,
            example_ids,
        )

    if not identified:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Build final Issue objects. Compute frequency based on affected sessions
    # (falls back to traces when no session IDs exist).
    num_total = len(triage_traces)
    total_sessions = len(session_groups)
    trace_to_session: dict[str, str] = {}
    for sid, traces_in_session in session_groups.items():
        for t in traces_in_session:
            trace_to_session[t.info.trace_id] = sid

    now_iso = datetime.now(timezone.utc).isoformat()
    issues: list[Issue] = []
    for ident in identified:
        example_ids = _collect_example_trace_ids(ident, analyses)
        affected_sessions = len({trace_to_session.get(tid, tid) for tid in example_ids})
        freq = affected_sessions / max(total_sessions, 1)
        name = ident.name.removeprefix("Issue: ").removeprefix("issue: ")
        issues.append(
            Issue(
                issue_id=str(uuid.uuid4()),
                run_id=triage_eval.run_id,
                name=name,
                description=ident.description,
                root_cause=ident.root_cause,
                example_trace_ids=example_ids,
                frequency=freq,
                confidence=ident.confidence,
                status="open",
                created_at=now_iso,
            )
        )

    issues.sort(
        key=lambda i: (i.frequency, CONFIDENCE_ORDER.get(i.confidence, 0)),
        reverse=True,
    )

    # Phase 4: Annotate affected traces with per-issue feedback assessments
    trace_lookup = {t.info.trace_id: t for t in triage_traces}
    _logger.info(
        "Phase 4: Annotating %d traces across %d issues...", len(rationale_map), len(issues)
    )
    t0 = time.time()
    has_sessions = any(_get_session_id(t) for t in triage_traces)
    session_first_trace: dict[str, str] | None = None
    if has_sessions:
        session_first_trace = {
            sid: traces_in_session[0].info.trace_id
            for sid, traces_in_session in session_groups.items()
        }
    _annotate_issue_traces(
        issues,
        rationale_map,
        trace_lookup,
        judge_model,
        trace_to_session=trace_to_session if has_sessions else None,
        session_first_trace=session_first_trace,
        token_counter=token_counter,
    )
    _logger.info("Phase 4: Annotation took %.1fs", time.time() - t0)

    summary = _build_summary(issues, num_total)
    _logger.info("Done. Found %d issues across %d traces.", len(issues), num_total)

    result = DiscoverIssuesResult(
        issues=issues,
        triage_run_id=triage_eval.run_id,
        summary=summary,
        total_traces_analyzed=num_total,
    )

    # Log artifacts to the triage run
    issues_data = [
        {
            "issue_id": issue.issue_id,
            "name": issue.name,
            "description": issue.description,
            "root_cause": issue.root_cause,
            "frequency": issue.frequency,
            "confidence": issue.confidence,
            "example_trace_ids": issue.example_trace_ids,
            "status": issue.status,
            "created_at": issue.created_at,
        }
        for issue in issues
    ]
    elapsed_seconds = round(time.time() - pipeline_start, 1)

    metadata = {
        "total_traces_analyzed": num_total,
        "num_issues": len(issues),
        "triage_run_id": triage_eval.run_id,
        "elapsed_seconds": elapsed_seconds,
        **token_counter.to_dict(),
    }

    _log_discovery_artifacts(
        triage_eval.run_id,
        {
            "summary.md": summary,
            "issues.json": json.dumps(issues_data, indent=2),
            "metadata.json": json.dumps(metadata, indent=2),
        },
    )

    return result
