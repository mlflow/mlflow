from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import mlflow
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.issue import IssueSeverity
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_GENAI_DISCOVERY_TRIAGE_SAMPLE_SIZE,
    MLFLOW_GENAI_EVAL_MAX_WORKERS,
)
from mlflow.genai.discovery.clustering import (
    cluster_by_llm,
    recluster_singletons,
    summarize_cluster,
)
from mlflow.genai.discovery.constants import (
    DEFAULT_MODEL,
    DEFAULT_SCORER_NAME,
    NO_ISSUE_KEYWORD,
    TRACE_ANNOTATION_SYSTEM_PROMPT,
    build_satisfaction_instructions,
)
from mlflow.genai.discovery.entities import (
    DiscoverIssuesResult,
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.extraction import (
    collect_session_rationales,
    extract_execution_paths_for_session,
    extract_failing_traces,
    extract_failure_labels,
)
from mlflow.genai.discovery.sampling import sample_traces
from mlflow.genai.discovery.utils import (
    _call_llm,
    _TokenCounter,
    build_summary,
    collect_example_trace_ids,
    format_annotation_prompt,
    format_trace_content,
    get_session_id,
    group_traces_by_session,
    log_discovery_artifacts,
    verify_scorer,
)
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey
from mlflow.tracking.fluent import _get_experiment_id

_logger = logging.getLogger(__name__)


def _is_non_issue(issue: _IdentifiedIssue) -> bool:
    return issue.severity == "not_an_issue" or NO_ISSUE_KEYWORD.lower() in issue.name.lower()


@dataclass
class _AnnotationWorkItem:
    issue: Issue
    trace_id: str
    triage_rationale: str
    session_id: str | None


def _build_annotation_work_items(
    issues: list[Issue],
    issue_trace_ids: dict[str, list[str]],
    rationale_map: dict[str, str],
    trace_to_session: dict[str, str] | None = None,
    session_first_trace: dict[str, str] | None = None,
) -> list[_AnnotationWorkItem]:
    """Build one work item per (issue, trace) or per (issue, session).

    When session mappings are provided, groups affected traces by session
    and produces one item per session targeting the session's first trace.
    """
    work_items: list[_AnnotationWorkItem] = []
    for issue in issues:
        trace_ids = issue_trace_ids.get(issue.issue_id, [])
        if trace_to_session and session_first_trace:
            by_session: dict[str, list[str]] = {}
            for tid in trace_ids:
                by_session.setdefault(trace_to_session.get(tid, tid), []).append(tid)
            for session_id, tids in by_session.items():
                work_items.append(
                    _AnnotationWorkItem(
                        issue=issue,
                        trace_id=session_first_trace.get(session_id, tids[0]),
                        triage_rationale=next(
                            (rationale_map[t] for t in tids if t in rationale_map), ""
                        ),
                        session_id=session_id,
                    )
                )
        else:
            work_items.extend(
                _AnnotationWorkItem(
                    issue=issue,
                    trace_id=tid,
                    triage_rationale=rationale_map.get(tid, ""),
                    session_id=None,
                )
                for tid in trace_ids
            )
    return work_items


def _annotate_issue_traces(
    issues: list[Issue],
    issue_trace_ids: dict[str, list[str]],
    rationale_map: dict[str, str],
    trace_lookup: dict[str, Trace],
    model: str,
    trace_to_session: dict[str, str] | None = None,
    session_first_trace: dict[str, str] | None = None,
    token_counter: _TokenCounter | None = None,
) -> None:
    """
    Log an Issue assessment for each issue on affected traces.

    When session information is provided, logs one annotation per (issue,
    session) on the session's first trace. Otherwise annotates each trace
    individually.

    Args:
        issues: Discovered issues to annotate traces for.
        issue_trace_ids: Mapping of issue_id to list of affected trace IDs.
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
    work_items = _build_annotation_work_items(
        issues, issue_trace_ids, rationale_map, trace_to_session, session_first_trace
    )
    if not work_items:
        return

    def _annotate_one(item: _AnnotationWorkItem) -> str | None:
        trace = trace_lookup.get(item.trace_id)
        trace_content = format_trace_content(trace) if trace else "(trace not available)"
        user_content = format_annotation_prompt(item.issue, trace_content, item.triage_rationale)

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
            _logger.debug(
                "Failed to generate annotation for trace %s", item.trace_id, exc_info=True
            )
            annotation = (
                f"This trace was flagged for issue '{item.issue.name}'. "
                f"Triage rationale: {item.triage_rationale or '(not available)'}"
            )

        metadata = {TraceMetadataKey.TRACE_SESSION: item.session_id} if item.session_id else None
        try:
            mlflow.log_issue(
                trace_id=item.trace_id,
                issue_id=item.issue.issue_id,
                issue_name=item.issue.name,
                source=source,
                rationale=annotation,
                metadata=metadata,
            )
        except Exception:
            _logger.debug("Failed to log issue for trace %s", item.trace_id, exc_info=True)
            return None
        return annotation

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(work_items))
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoveryAnnotate"
    ) as executor:
        futures = {executor.submit(_annotate_one, item): item for item in work_items}
        for future in as_completed(futures):
            future.result()


def _build_analyses(
    triage_traces: list[Trace],
    rationale_map: dict[str, str],
    scorer_name: str,
) -> tuple[list[_ConversationAnalysis], dict[str, list[Trace]]]:
    """
    Build per-session analyses from triage results.

    For each session with failing traces, combines triage rationales,
    human feedback, and span errors into a single ConversationAnalysis.

    Args:
        triage_traces: All traces from the triage phase (passing and failing).
        rationale_map: Mapping of trace_id to triage rationale for failing traces.
        scorer_name: Name of the triage scorer, used to look up human feedback.

    Returns:
        A tuple of (analyses, session_groups) where session_groups maps
        session_id to the list of traces in that session.
    """
    session_groups = group_traces_by_session(triage_traces)
    analyses: list[_ConversationAnalysis] = []
    for session_id, session_traces in session_groups.items():
        session_failing = [
            trace for trace in session_traces if trace.info.trace_id in rationale_map
        ]
        if not session_failing:
            continue
        combined_rationale = collect_session_rationales(session_failing, rationale_map, scorer_name)
        if not combined_rationale:
            continue
        exec_path = extract_execution_paths_for_session(session_failing)
        analyses.append(
            _ConversationAnalysis(
                full_rationale=combined_rationale,
                affected_trace_ids=[trace.info.trace_id for trace in session_failing],
                execution_path=exec_path,
            )
        )
    _logger.debug("Built %d analyses from triage rationales", len(analyses))
    return analyses, session_groups


def _resplit_incoherent_clusters(
    cluster_groups: list[list[int]],
    summaries: list[_IdentifiedIssue],
    summarize_fn: Callable[[list[int]], _IdentifiedIssue],
) -> list[_IdentifiedIssue]:
    """Re-split multi-member clusters flagged below minimum severity.

    Breaks incoherent clusters into singletons, resummarizes each one,
    then filters all results by severity and non-issue status.
    """
    resplit_groups: list[list[int]] = []
    for group, issue in zip(cluster_groups, summaries):
        if issue.severity < IssueSeverity.LOW and len(group) > 1:
            _logger.debug(
                "Re-splitting incoherent cluster '%s' (severity=%s, %d members)",
                issue.name,
                issue.severity,
                len(group),
            )
            resplit_groups.extend([idx] for idx in group)

    if resplit_groups:
        max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(resplit_groups))
        resplit_summaries: list[_IdentifiedIssue | None] = [None] * len(resplit_groups)
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoveryResplit"
        ) as executor:
            future_to_idx = {
                executor.submit(summarize_fn, group): ri for ri, group in enumerate(resplit_groups)
            }
            for future in as_completed(future_to_idx):
                resplit_summaries[future_to_idx[future]] = future.result()

        # Keep clusters that passed severity, replace incoherent ones with resplit results
        kept = [
            issue
            for group, issue in zip(cluster_groups, summaries)
            if issue.severity >= IssueSeverity.LOW or len(group) <= 1
        ]
        summaries = kept + resplit_summaries

    return [
        issue
        for issue in summaries
        if issue.severity >= IssueSeverity.LOW and not _is_non_issue(issue)
    ]


def _dedup_issues_by_name(issues: list[_IdentifiedIssue]) -> list[_IdentifiedIssue]:
    seen_names: dict[str, int] = {}
    deduped: list[_IdentifiedIssue] = []
    for issue in issues:
        key = issue.name.strip().lower()
        if key in seen_names:
            existing = deduped[seen_names[key]]
            existing.example_indices = list(set(existing.example_indices + issue.example_indices))
            existing.severity = max(existing.severity, issue.severity)
        else:
            seen_names[key] = len(deduped)
            deduped.append(issue)
    return deduped


def _merge_singleton_issues(
    identified: list[_IdentifiedIssue],
    analysis_labels: dict[int, str],
    analyses: list[_ConversationAnalysis],
    model: str,
    max_issues: int,
    token_counter: _TokenCounter | None = None,
) -> list[_IdentifiedIssue]:
    singletons = [i for i in identified if len(i.example_indices) == 1]
    multi_member = [i for i in identified if len(i.example_indices) > 1]
    if len(singletons) < 2:
        return identified
    _logger.debug("Re-clustering %d singleton issues", len(singletons))
    merged = recluster_singletons(
        singletons,
        analysis_labels,
        analyses,
        model,
        max_issues,
        token_counter=token_counter,
    )
    return multi_member + merged


def _cluster_and_identify(
    analyses: list[_ConversationAnalysis],
    model: str,
    max_issues: int,
    token_counter: _TokenCounter | None = None,
) -> list[_IdentifiedIssue]:
    """Cluster analyses into identified issues via LLM-based labeling and grouping."""
    labels, label_to_analysis = extract_failure_labels(analyses, model, token_counter=token_counter)
    for i, label in enumerate(labels):
        _logger.debug("  [%d] %s", i, label)

    if len(labels) == 1:
        cluster_groups = [[0]]
    else:
        cluster_groups = cluster_by_llm(labels, max_issues, model, token_counter=token_counter)
    _logger.debug("Clustering produced %d groups", len(cluster_groups))

    def summarize_fn(group: list[int]) -> _IdentifiedIssue:
        return summarize_cluster(
            group, analyses, model, label_to_analysis=label_to_analysis, token_counter=token_counter
        )

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(cluster_groups))
    summaries: list[_IdentifiedIssue | None] = [None] * len(cluster_groups)
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowIssueDiscoverySummarize"
    ) as executor:
        future_to_idx = {
            executor.submit(summarize_fn, group): idx for idx, group in enumerate(cluster_groups)
        }
        for future in as_completed(future_to_idx):
            summaries[future_to_idx[future]] = future.result()

    identified = _resplit_incoherent_clusters(cluster_groups, summaries, summarize_fn)
    identified = _dedup_issues_by_name(identified)

    analysis_labels: dict[int, str] = {}
    for label, analysis_idx in zip(labels, label_to_analysis):
        analysis_labels.setdefault(analysis_idx, label)
    return _merge_singleton_issues(
        identified, analysis_labels, analyses, model, max_issues, token_counter=token_counter
    )


def _build_issues(
    identified: list[_IdentifiedIssue],
    analyses: list[_ConversationAnalysis],
    exp_id: str,
    source_run_id: str,
) -> tuple[list[Issue], dict[str, list[str]]]:
    """
    Convert identified issues into Issue entities.

    Returns:
        A tuple of (issues, issue_trace_ids) where issue_trace_ids maps
        issue_id to the list of example trace IDs for annotation.
    """
    issues: list[Issue] = []
    issue_trace_ids: dict[str, list[str]] = {}
    for ident in identified:
        example_ids = collect_example_trace_ids(ident, analyses)
        name = ident.name.removeprefix("Issue: ").removeprefix("issue: ")
        issue_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)
        issues.append(
            Issue(
                issue_id=issue_id,
                experiment_id=exp_id or "",
                name=name,
                description=ident.description,
                status="open",
                created_timestamp=now_ms,
                last_updated_timestamp=now_ms,
                severity=ident.severity,
                root_causes=[ident.root_cause],
                source_run_id=source_run_id,
            )
        )
        issue_trace_ids[issue_id] = example_ids

    issues.sort(
        key=lambda i: i.severity,
        reverse=True,
    )
    return issues, issue_trace_ids


def build_issue_discovery_scorer(
    categories: list[str] | None = None,
    model: str | None = None,
    use_conversation: bool = True,
) -> Scorer:
    model = model or DEFAULT_MODEL
    instructions = build_satisfaction_instructions(
        use_conversation=use_conversation, categories=categories
    )
    return make_judge(
        name=DEFAULT_SCORER_NAME,
        instructions=instructions,
        model=model,
        feedback_value_type=bool,
    )


def discover_issues(
    experiment_id: str | None = None,
    traces: list[Trace] | None = None,
    scorers: list[Scorer] | None = None,
    model: str | None = None,
    max_issues: int = 20,
    filter_string: str | None = None,
) -> DiscoverIssuesResult:
    """
    Discover quality and operational issues in traces.

    Runs a multi-phase pipeline:

    1. **Triage** -- scores traces using the provided scorers (or a default
       satisfaction judge). Traces marked as failing proceed to analysis.
    2. **Analysis** -- builds per-session analyses from triage rationales
       and human feedback assessments.
    3. **Cluster & Identify** -- LLM-based clustering of analyses into
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
        max_issues: Maximum distinct issues to identify.
        filter_string: Filter string passed to ``search_traces``.
            Ignored when ``traces`` is provided.

    Returns:
        A :class:`DiscoverIssuesResult` with discovered issues, run IDs,
        and a summary report.
    """
    pipeline_start = time.time()
    token_counter = _TokenCounter()
    model = model or DEFAULT_MODEL

    exp_id = experiment_id or _get_experiment_id()

    # ---- Phase 1: Triage ----
    sample_size = MLFLOW_GENAI_DISCOVERY_TRIAGE_SAMPLE_SIZE.get()
    if traces is not None:
        triage_traces = list(traces)
        _logger.debug("Using %d provided traces", len(triage_traces))
    else:
        if exp_id is None:
            raise mlflow.exceptions.MlflowException(
                "No experiment specified. Pass traces, use "
                "mlflow.set_experiment(), or pass experiment_id."
            )
        search_kwargs = {
            "filter_string": filter_string,
            "locations": [exp_id],
        }
        _logger.debug("Sampling %d traces", sample_size)
        triage_traces = sample_traces(sample_size, search_kwargs)

    if len(triage_traces) > sample_size:
        triage_traces = sample_traces(sample_size, traces=triage_traces)

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
            _logger.debug("No session IDs found, falling back to trace-level scorer")
        instructions = build_satisfaction_instructions(use_conversation=use_conversation)
        default_scorer = make_judge(
            name=DEFAULT_SCORER_NAME,
            instructions=instructions,
            model=model,
            feedback_value_type=bool,
        )
        scorers = [default_scorer]

    scorer_name = scorers[0].name

    test_session = None
    if use_conversation:
        session_groups = group_traces_by_session(triage_traces)
        test_session = next(
            (traces for traces in session_groups.values() if len(traces) > 1),
            next(iter(session_groups.values())),
        )
    verify_scorer(
        scorers[0], test_session[0] if test_session else triage_traces[0], session=test_session
    )

    with mlflow.start_run(run_name="discover_issues"):
        triage_eval = mlflow.genai.evaluate(
            data=triage_traces,
            scorers=scorers,
        )

    # Re-fetch traces with scorer assessments attached.
    scored_traces = triage_traces
    try:
        fetched = [mlflow.get_trace(t.info.trace_id) for t in triage_traces]
        scored_traces = fetched
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
        "Triage complete: %d/%d traces unsatisfactory",
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

    # ---- Phase 2: Build analyses ----
    analyses, session_groups = _build_analyses(triage_traces, rationale_map, scorer_name)

    # ---- Phase 3: Cluster & identify ----
    identified = _cluster_and_identify(analyses, model, max_issues, token_counter=token_counter)

    if not identified:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            summary=build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # ---- Phase 4: Build issues & annotate ----
    issues, issue_trace_ids = _build_issues(identified, analyses, exp_id, triage_eval.run_id)

    trace_to_session: dict[str, str] = {}
    for session_id, traces_in_session in session_groups.items():
        for trace in traces_in_session:
            trace_to_session[trace.info.trace_id] = session_id

    trace_lookup = {trace.info.trace_id: trace for trace in triage_traces}
    session_first_trace: dict[str, str] | None = None
    if use_conversation:
        session_first_trace = {
            session_id: traces_in_session[0].info.trace_id
            for session_id, traces_in_session in session_groups.items()
        }
    _annotate_issue_traces(
        issues,
        issue_trace_ids,
        rationale_map,
        trace_lookup,
        model,
        trace_to_session=trace_to_session if use_conversation else None,
        session_first_trace=session_first_trace,
        token_counter=token_counter,
    )

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
            "root_causes": issue.root_causes,
            "severity": issue.severity,
            "status": issue.status,
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
        "max_issues": max_issues,
        "experiment_id": exp_id,
        "filter_string": filter_string,
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
