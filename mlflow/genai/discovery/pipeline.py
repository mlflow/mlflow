from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.discovery.clustering import (
    cluster_by_llm,
    summarize_cluster,
)
from mlflow.genai.discovery.constants import (
    DEFAULT_MODEL,
    DEFAULT_SCORER_NAME,
    DEFAULT_TRIAGE_SAMPLE_SIZE,
    MAX_EXAMPLE_TRACE_IDS,
    MIN_SEVERITY,
    NO_ISSUE_KEYWORD,
    SEVERITY_ORDER,
    SURFACE_TRUNCATION_LIMIT,
    TRACE_ANNOTATION_SYSTEM_PROMPT,
    TRACE_CONTENT_TRUNCATION,
    build_satisfaction_instructions,
)
from mlflow.genai.discovery.entities import (
    DiscoverIssuesResult,
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.extraction import (
    extract_execution_path,
    extract_execution_paths_for_session,
    extract_failing_traces,
    extract_failure_labels,
    extract_span_errors,
)
from mlflow.genai.discovery.sampling import sample_traces
from mlflow.genai.discovery.utils import (
    _call_llm,
    _TokenCounter,
    build_summary,
    get_session_id,
    group_traces_by_session,
    log_discovery_artifacts,
)
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey
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


def severity_gte(a: str, b: str) -> bool:
    return SEVERITY_ORDER.get(a, -1) >= SEVERITY_ORDER.get(b, 0)


def severity_max(a: str, b: str) -> str:
    return a if SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0) else b


def _is_non_issue(issue: _IdentifiedIssue) -> bool:
    """Check if an identified issue is actually a false positive (no real issue)."""
    if NO_ISSUE_KEYWORD.lower() in issue.name.lower():
        return True
    combined = f"{issue.name} {issue.description} {issue.root_cause}".lower()
    return any(pattern in combined for pattern in _NO_ISSUE_PATTERNS)


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
        if not isinstance(feedback, Feedback):
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' returned {type(feedback).__name__} instead of Feedback"
            )
        if feedback.value is None:
            error = feedback.error_message or "unknown error (check model API logs)"
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' returned null value: {error}"
            )
    except mlflow.exceptions.MlflowException:
        raise
    except Exception as exc:
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' failed verification on trace {trace.info.trace_id}: {exc}"
        ) from exc


def _extract_assessment_rationale(trace: Trace, scorer_name: str) -> str:
    return next(
        (
            assessment.rationale
            for assessment in trace.info.assessments
            if isinstance(assessment, Feedback)
            and assessment.name == scorer_name
            and assessment.rationale
        ),
        "",
    )


def _collect_example_trace_ids(
    issue: _IdentifiedIssue,
    analyses: list[_ConversationAnalysis],
) -> list[str]:
    """Gather trace IDs from analyses referenced by the issue's example indices."""
    trace_ids = []
    for idx in issue.example_indices:
        if 0 <= idx < len(analyses):
            trace_ids.extend(analyses[idx].affected_trace_ids)
    return trace_ids[:MAX_EXAMPLE_TRACE_IDS]


def _recluster_singletons(
    singletons: list[_IdentifiedIssue],
    labels: list[str],
    analyses: list[_ConversationAnalysis],
    model: str,
    max_issues: int,
    token_counter: _TokenCounter | None = None,
) -> list[_IdentifiedIssue]:
    """
    Re-cluster singleton issues via a second LLM pass to find better groupings.

    Args:
        singletons: Single-analysis issues to attempt merging.
        labels: Failure labels from the initial clustering phase.
        analyses: All conversation analyses from the pipeline.
        model: Model URI for clustering and summarization.
        max_issues: Maximum number of groups to produce.
        token_counter: Optional token counter for tracking LLM usage.

    Returns:
        List of issues after re-clustering (merged or original singletons).
    """
    if len(singletons) < 2:
        return list(singletons)

    singleton_labels = []
    for singleton in singletons:
        idx = singleton.example_indices[0]
        singleton_labels.append(labels[idx] if idx < len(labels) else singleton.name)

    new_groups = cluster_by_llm(singleton_labels, max_issues, model, token_counter=token_counter)

    result: list[_IdentifiedIssue] = []
    for group in new_groups:
        if len(group) == 1:
            result.append(singletons[group[0]])
            continue
        merged_indices = [singletons[group_idx].example_indices[0] for group_idx in group]
        merged_issue = summarize_cluster(
            merged_indices, analyses, model, token_counter=token_counter
        )
        if severity_gte(merged_issue.severity, MIN_SEVERITY):
            result.append(merged_issue)
        else:
            result.extend(singletons[group_idx] for group_idx in group)

    return result


def _format_trace_content(trace: Trace) -> str:
    """Build a compact text representation of a trace for annotation prompts."""
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


def _annotate_issue_traces(
    issues: list[Issue],
    rationale_map: dict[str, str],
    trace_lookup: dict[str, Trace],
    model: str,
    trace_to_session: dict[str, str] | None = None,
    session_first_trace: dict[str, str] | None = None,
    token_counter: _TokenCounter | None = None,
) -> None:
    """
    Log a Feedback assessment for each issue on affected traces.

    When session information is provided, logs one annotation per (issue,
    session) on the session's first trace. Otherwise annotates each trace
    individually.

    Args:
        issues: Discovered issues to annotate traces for.
        rationale_map: Mapping of trace_id to triage rationale text.
        trace_lookup: Mapping of trace_id to Trace objects.
        model: Model URI for the annotation LLM.
        trace_to_session: Optional mapping of trace_id to session_id.
        session_first_trace: Optional mapping of session_id to first trace_id.
        token_counter: Optional token counter for tracking LLM usage.
    """
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
                session_id = trace_to_session.get(tid, tid)
                session_traces.setdefault(session_id, []).append(tid)
            for session_id, tids in session_traces.items():
                target = session_first_trace.get(session_id, tids[0])
                rationale = next((rationale_map[t] for t in tids if t in rationale_map), "")
                work_items.append((issue, target, rationale, session_id))
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
            response = _call_llm(
                model,
                [
                    {"role": "system", "content": TRACE_ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                token_counter=token_counter,
            )
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
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoveryAnnotate"
    ) as executor:
        future_to_item = {
            executor.submit(_annotate_one, issue, trace_id, rationale, session_id): (
                issue,
                trace_id,
            )
            for issue, trace_id, rationale, session_id in work_items
        }
        for future in as_completed(future_to_item):
            future.result()


@experimental(version="3.11.0")
def discover_issues(
    experiment_id: str | None = None,
    traces: list[Trace] | None = None,
    scorers: list[Scorer] | None = None,
    model: str | None = None,
    triage_sample_size: int = DEFAULT_TRIAGE_SAMPLE_SIZE,
    max_issues: int = 20,
    filter_string: str | None = None,
) -> DiscoverIssuesResult:
    """
    Discover quality and operational issues in traces.

    Runs a multi-phase pipeline:
    1. **Triage**: Scores traces using the provided scorers (or a default
       satisfaction judge). Traces marked as failing by any scorer proceed
       to analysis.
    2. **Analysis**: Builds per-session analyses from triage rationales and
       human feedback assessments.
    3. **Cluster & Identify**: LLM-based clustering of analyses into
       coherent issue groups, with summarization and refinement.

    Args:
        experiment_id: Experiment to analyze. Defaults to the active experiment.
            Ignored when ``traces`` is provided.
        traces: Traces to analyze directly. When provided, skips sampling
            and uses these traces as the input to the pipeline.
        scorers: Scorers to run during triage. A trace is considered failing
            if *any* scorer marks it as ``False``. When ``None``, a default
            conversation-level satisfaction judge is used.
        model: LLM used for all pipeline phases (scoring, labeling, clustering,
            summarization). Defaults to ``"openai:/gpt-5-mini"``.
        triage_sample_size: Number of sessions (or traces, if no session metadata
            exists) to randomly sample for the triage phase. Ignored when
            ``traces`` is provided.
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
    model = model or DEFAULT_MODEL

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
        triage_traces = sample_traces(triage_sample_size, search_kwargs)

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

    use_conversation = False
    if scorers is None:
        use_conversation = any(get_session_id(trace) for trace in triage_traces)
        if not use_conversation:
            _logger.info(
                "No session IDs found in trace metadata, falling back to trace-level scorer"
            )
        instructions = build_satisfaction_instructions(use_conversation=use_conversation)
        default_scorer = make_judge(
            name=DEFAULT_SCORER_NAME,
            instructions=instructions,
            model=model,
            feedback_value_type=bool,
        )
        _logger.info(
            "Using %s-based satisfaction scorer",
            "session (conversation)" if use_conversation else "trace",
        )
        scorers = [default_scorer]

    scorer_name = scorers[0].name

    _logger.info("Phase 1: Testing scorer on one trace...")
    phase_start = time.time()
    test_session = None
    if use_conversation:
        session_groups = group_traces_by_session(triage_traces)
        # Pick a group with a real session ID (multi-trace groups have actual sessions)
        test_session = next(
            (traces for traces in session_groups.values() if len(traces) > 1),
            next(iter(session_groups.values())),
        )
    verify_scorer(
        scorers[0], test_session[0] if test_session else triage_traces[0], session=test_session
    )
    _logger.info("Phase 1: Test scorer took %.1fs", time.time() - phase_start)

    _logger.info("Phase 1: Scoring %d traces...", len(triage_traces))
    phase_start = time.time()
    with mlflow.start_run(run_name="discover_issues - triage"):
        triage_eval = mlflow.genai.evaluate(
            data=triage_traces,
            scorers=scorers,
        )
    _logger.info("Phase 1: Triage scoring took %.1fs", time.time() - phase_start)

    # Re-fetch traces with scorer assessments attached.
    scored_traces = triage_traces
    try:
        fetched = mlflow.search_traces(
            return_type="list",
            locations=[exp_id],
        )
        scored_lookup = {t.info.trace_id: t for t in fetched}
        scored_traces = [scored_lookup.get(t.info.trace_id, t) for t in triage_traces]
        # Aggregate judge costs from Phase 1 evaluation assessments.
        for trace in fetched:
            for assessment in trace.info.assessments or []:
                meta = getattr(assessment, "metadata", None) or {}
                if meta.get(AssessmentMetadataKey.SOURCE_RUN_ID) != triage_eval.run_id:
                    continue
                if cost := meta.get(AssessmentMetadataKey.JUDGE_COST):
                    token_counter.cost_usd += float(cost)
                if input_tok := meta.get(AssessmentMetadataKey.JUDGE_INPUT_TOKENS):
                    token_counter.input_tokens += int(input_tok)
                if output_tok := meta.get(AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS):
                    token_counter.output_tokens += int(output_tok)
    except Exception:
        _logger.debug("Failed to fetch scored traces", exc_info=True)

    scorer_names = [s.name for s in scorers]
    failing_traces, rationale_map = extract_failing_traces(scored_traces, scorer_names)

    _logger.info(
        "Phase 1 complete: %d/%d traces unsatisfactory",
        len(failing_traces),
        len(triage_traces),
    )

    if not failing_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            summary=build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 2: For each session with failing traces, combine triage rationales,
    # human feedback, and span errors into a single ConversationAnalysis.
    session_groups = group_traces_by_session(triage_traces)
    analyses: list[_ConversationAnalysis] = []
    for session_id, session_traces in session_groups.items():
        session_failing = [
            trace for trace in session_traces if trace.info.trace_id in rationale_map
        ]
        if not session_failing:
            continue
        rationales: list[str] = []
        seen_rationales: set[str] = set()

        def _add_rationale(text: str, prefix: str = "") -> None:
            if text and text not in seen_rationales:
                seen_rationales.add(text)
                rationales.append(f"{prefix}{text}" if prefix else text)

        for trace in session_failing:
            _add_rationale(rationale_map[trace.info.trace_id])
            _add_rationale(_extract_assessment_rationale(trace, scorer_name), "[human feedback] ")
            _add_rationale(extract_span_errors(trace), "[span errors] ")
        combined_rationale = "; ".join(rationales)
        if not combined_rationale:
            continue
        surface = combined_rationale[:SURFACE_TRUNCATION_LIMIT]
        exec_path = extract_execution_paths_for_session(session_failing)
        analyses.append(
            _ConversationAnalysis(
                surface=surface,
                root_cause=combined_rationale,
                affected_trace_ids=[trace.info.trace_id for trace in session_failing],
                execution_path=exec_path,
            )
        )
    _logger.info("Phase 2: Built %d analyses from triage rationales", len(analyses))

    # Phase 3: Cluster — LLM-based label extraction and grouping
    _logger.info("Phase 3: Extracting failure labels for clustering...")
    phase_start = time.time()
    labels = extract_failure_labels(analyses, model, token_counter=token_counter)
    _logger.info(
        "Phase 3: Label extraction took %.1fs",
        time.time() - phase_start,
    )
    for i, label in enumerate(labels):
        _logger.info("  [%d] %s", i, label)

    _logger.info("Phase 3: Clustering analyses into issues...")
    phase_start = time.time()
    if len(analyses) == 1:
        cluster_groups = [[0]]
    else:
        cluster_groups = cluster_by_llm(labels, max_issues, model, token_counter=token_counter)
    cluster_end = time.time()
    _logger.info(
        "Phase 3: Clustering took %.1fs, produced %d clusters",
        cluster_end - phase_start,
        len(cluster_groups),
    )

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(cluster_groups))

    def _summarize_one(cluster_idx: int, group: list[int]) -> _IdentifiedIssue:
        summarize_start = time.time()
        issue = summarize_cluster(group, analyses, model, token_counter=token_counter)
        _logger.info(
            "Phase 3: Summarized cluster %d/%d (%d analyses) in %.1fs",
            cluster_idx + 1,
            len(cluster_groups),
            len(group),
            time.time() - summarize_start,
        )
        return issue

    summaries: list[_IdentifiedIssue | None] = [None] * len(cluster_groups)
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoverySummarize"
    ) as executor:
        future_to_idx = {
            executor.submit(_summarize_one, cluster_idx, group): cluster_idx
            for cluster_idx, group in enumerate(cluster_groups)
        }
        for future in as_completed(future_to_idx):
            summaries[future_to_idx[future]] = future.result()

    # Re-split incoherent clusters: if the summarizer flags a multi-member
    # cluster as low-severity, break it into singletons and resummarize
    # each one so the individual items get a fair standalone assessment.
    resplit_groups: list[list[int]] = []
    for group, issue in zip(cluster_groups, summaries):
        if not severity_gte(issue.severity, MIN_SEVERITY) and len(group) > 1:
            _logger.info(
                "Phase 3: Re-splitting incoherent cluster '%s' "
                "(severity=%s, %d members) into singletons",
                issue.name,
                issue.severity,
                len(group),
            )
            resplit_groups.extend([idx] for idx in group)

    if resplit_groups:
        resplit_summaries: list[_IdentifiedIssue | None] = [None] * len(resplit_groups)
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoveryResplit"
        ) as executor:
            future_to_idx = {
                executor.submit(_summarize_one, len(cluster_groups) + ri, group): ri
                for ri, group in enumerate(resplit_groups)
            }
            for future in as_completed(future_to_idx):
                resplit_summaries[future_to_idx[future]] = future.result()
        final_groups: list[list[int]] = []
        final_summaries: list[_IdentifiedIssue] = []
        for group, issue in zip(cluster_groups, summaries):
            if not severity_gte(issue.severity, MIN_SEVERITY) and len(group) > 1:
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
        if severity_gte(issue.severity, MIN_SEVERITY) and not _is_non_issue(issue)
    ]

    # Merge issues with identical names (case-insensitive), combining their
    # example indices and keeping the higher severity level.
    seen_names: dict[str, int] = {}
    deduped: list[_IdentifiedIssue] = []
    for issue in identified:
        key = issue.name.strip().lower()
        if key in seen_names:
            existing = deduped[seen_names[key]]
            merged_indices = list(set(existing.example_indices + issue.example_indices))
            existing.example_indices = merged_indices
            existing.severity = severity_max(existing.severity, issue.severity)
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
            model,
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
        time.time() - phase_start,
    )
    for issue in identified:
        example_ids = _collect_example_trace_ids(issue, analyses)
        _logger.info(
            "  %s (severity %s): %s | root_cause: %s | examples: %s",
            issue.name,
            issue.severity,
            issue.description,
            issue.root_cause,
            example_ids,
        )

    if not identified:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            summary=build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Build final Issue objects. Compute frequency based on affected sessions
    # (falls back to traces when no session IDs exist).
    total_sessions = len(session_groups)
    trace_to_session: dict[str, str] = {}
    for session_id, traces_in_session in session_groups.items():
        for trace in traces_in_session:
            trace_to_session[trace.info.trace_id] = session_id

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
                severity=ident.severity,
                status="open",
                created_at=now_iso,
            )
        )

    issues.sort(
        key=lambda i: (i.frequency, SEVERITY_ORDER.get(i.severity, 0)),
        reverse=True,
    )

    # Phase 4: Annotate affected traces with per-issue feedback assessments
    trace_lookup = {trace.info.trace_id: trace for trace in triage_traces}
    _logger.info(
        "Phase 4: Annotating %d traces across %d issues...", len(rationale_map), len(issues)
    )
    phase_start = time.time()
    has_sessions = any(get_session_id(trace) for trace in triage_traces)
    session_first_trace: dict[str, str] | None = None
    if has_sessions:
        session_first_trace = {
            session_id: traces_in_session[0].info.trace_id
            for session_id, traces_in_session in session_groups.items()
        }
    _annotate_issue_traces(
        issues,
        rationale_map,
        trace_lookup,
        model,
        trace_to_session=trace_to_session if has_sessions else None,
        session_first_trace=session_first_trace,
        token_counter=token_counter,
    )
    _logger.info("Phase 4: Annotation took %.1fs", time.time() - phase_start)

    summary = build_summary(issues, len(triage_traces))
    _logger.info("Done. Found %d issues across %d traces.", len(issues), len(triage_traces))

    result = DiscoverIssuesResult(
        issues=issues,
        triage_run_id=triage_eval.run_id,
        summary=summary,
        total_traces_analyzed=len(triage_traces),
    )

    # Log artifacts to the triage run
    issues_data = [
        {
            "issue_id": issue.issue_id,
            "name": issue.name,
            "description": issue.description,
            "root_cause": issue.root_cause,
            "frequency": issue.frequency,
            "severity": issue.severity,
            "example_trace_ids": issue.example_trace_ids,
            "status": issue.status,
            "created_at": issue.created_at,
        }
        for issue in issues
    ]
    elapsed_seconds = round(time.time() - pipeline_start, 1)

    metadata = {
        "total_traces_analyzed": len(triage_traces),
        "num_issues": len(issues),
        "model": model,
        "scorer_names": scorer_names,
        "triage_run_id": triage_eval.run_id,
        "elapsed_seconds": elapsed_seconds,
        **token_counter.to_dict(),
    }

    log_discovery_artifacts(
        triage_eval.run_id,
        {
            "summary.md": summary,
            "issues.json": json.dumps(issues_data, indent=2),
            "metadata.json": json.dumps(metadata, indent=2),
        },
    )

    return result
