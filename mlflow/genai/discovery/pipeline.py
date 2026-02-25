from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.discovery.constants import (
    _DEFAULT_ANALYSIS_MODEL,
    _DEFAULT_JUDGE_MODEL,
    _DEFAULT_TRIAGE_SAMPLE_SIZE,
    _MIN_CONFIDENCE,
    _MIN_EXAMPLES,
)
from mlflow.genai.discovery.entities import (
    DiscoverIssuesResult,
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.utils import (
    _build_default_satisfaction_scorer,
    _build_summary,
    _cluster_analyses,
    _extract_execution_paths_for_session,
    _extract_failing_traces,
    _extract_failure_labels,
    _group_traces_by_session,
    _has_session_ids,
    _log_discovery_artifacts,
    _sample_traces,
    _summarize_cluster,
    _test_scorer,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

_NO_ISSUE_PATTERNS = frozenset({"no issues", "no issue", "no problems", "no failures", "no errors"})


def _is_non_issue(issue: _IdentifiedIssue) -> bool:
    name_lower = issue.name.lower()
    desc_lower = issue.description.lower()
    return any(p in name_lower or p in desc_lower for p in _NO_ISSUE_PATTERNS)


def _extract_assessment_rationale(trace: Trace, scorer_name: str) -> str:
    for assessment in trace.info.assessments:
        if (
            isinstance(assessment, Feedback)
            and assessment.name == scorer_name
            and assessment.rationale
        ):
            return assessment.rationale
    return ""


def _collect_example_trace_ids(
    issue: _IdentifiedIssue,
    analyses: list[_ConversationAnalysis],
) -> list[str]:
    trace_ids = []
    for idx in issue.example_indices:
        if 0 <= idx < len(analyses):
            trace_ids.extend(analyses[idx].affected_trace_ids)
    return trace_ids


def _format_trace_content(trace: Trace) -> str:
    """Build a compact text representation of a trace for annotation prompts."""
    from mlflow.genai.discovery.utils import _extract_execution_path

    parts = []
    request = trace.data.request
    if request:
        parts.append(f"Input: {str(request)[:1000]}")
    response = trace.data.response
    if response:
        parts.append(f"Output: {str(response)[:1000]}")
    exec_path = _extract_execution_path(trace)
    if exec_path and exec_path != "(no routing)":
        parts.append(f"Execution path: {exec_path}")
    return "\n".join(parts) if parts else "(trace content not available)"


def _annotate_issue_traces(
    issues: list[Issue],
    rationale_map: dict[str, str],
    trace_lookup: dict[str, Trace],
    model: str,
) -> None:
    """Log a Feedback assessment on each trace affected by a discovered issue.

    For each (issue, trace_id) pair, calls an LLM with the issue context,
    the trace's actual content (inputs/outputs/execution path), and the
    triage judge's rationale to produce a specific annotation explaining
    how this trace exhibits the issue. Logs it via ``mlflow.log_feedback``.
    """
    import litellm

    from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
    from mlflow.genai.discovery.constants import _TRACE_ANNOTATION_SYSTEM_PROMPT
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    scheme, path = _parse_model_uri(model)
    if scheme in ("endpoints", "databricks"):
        litellm_model = f"databricks/{path}"
    else:
        litellm_model = f"{scheme}/{path}"

    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id=model,
    )

    # Build work items: (issue, trace_id) pairs
    work_items: list[tuple[Issue, str]] = []
    for issue in issues:
        for trace_id in issue.example_trace_ids:
            work_items.append((issue, trace_id))

    if not work_items:
        return

    def _annotate_one(issue: Issue, trace_id: str) -> str | None:
        triage_rationale = rationale_map.get(trace_id, "")
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
            response = litellm.completion(
                model=litellm_model,
                messages=[
                    {"role": "system", "content": _TRACE_ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=4096,
            )
            annotation = (response.choices[0].message.content or "").strip()
        except Exception:
            _logger.debug("Failed to generate annotation for trace %s", trace_id, exc_info=True)
            annotation = (
                f"This trace was flagged for issue '{issue.name}'. "
                f"Triage rationale: {triage_rationale or '(not available)'}"
            )

        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=issue.name,
                value=False,
                source=source,
                rationale=annotation,
            )
        except Exception:
            _logger.debug("Failed to log feedback for trace %s", trace_id, exc_info=True)
            return None
        return annotation

    max_workers = min(MLFLOW_GENAI_EVAL_MAX_WORKERS.get(), len(work_items))
    annotations: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(_annotate_one, issue, trace_id): (issue, trace_id)
            for issue, trace_id in work_items
        }
        for future in as_completed(future_to_item):
            issue, trace_id = future_to_item[future]
            result = future.result()
            if result:
                annotations.setdefault(issue.name, []).append(result)

    # Populate rationale_examples on each issue (up to 3 per issue)
    for issue in issues:
        issue.rationale_examples = annotations.get(issue.name, [])[:3]


@experimental(version="3.11.0")
def discover_issues(
    experiment_id: str | None = None,
    traces: list | None = None,
    satisfaction_scorer: Scorer | None = None,
    additional_scorers: list[Scorer] | None = None,
    judge_model: str | None = None,
    analysis_model: str | None = None,
    embedding_model: str = "openai:/text-embedding-3-small",
    triage_sample_size: int = _DEFAULT_TRIAGE_SAMPLE_SIZE,
    validation_sample_size: int | None = None,
    max_issues: int = 25,
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
            Defaults to ``"openai:/gpt-5-mini"``.
        analysis_model: LLM used for analysis and cluster summarization.
            Defaults to ``"openai:/gpt-5"``.
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
                mlflow.genai.evaluate(data=traces, scorers=[issue.scorer])
    """
    judge_model = judge_model or _DEFAULT_JUDGE_MODEL
    analysis_model = analysis_model or _DEFAULT_ANALYSIS_MODEL

    if traces is not None:
        # Use provided traces directly — experiment_id only needed for validation
        exp_id = experiment_id or _get_experiment_id()
        triage_traces = list(traces)
        _logger.info("Phase 1: Using %d provided traces...", len(triage_traces))
    else:
        exp_id = experiment_id or _get_experiment_id()
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

    if not triage_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id="",
            validation_run_id=None,
            summary="No traces to analyze.",
            total_traces_analyzed=0,
        )

    if satisfaction_scorer is None:
        use_conversation = _has_session_ids(triage_traces)
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
            validation_run_id=None,
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
        rationales = [rationale_map[t.info.trace_id] for t in session_failing]
        # Also include human feedback assessments if available
        for t in session_failing:
            human_rationale = _extract_assessment_rationale(t, scorer_name)
            if human_rationale:
                rationales.append(f"[human feedback] {human_rationale}")
        combined_rationale = "; ".join(r for r in rationales if r)
        if not combined_rationale:
            continue
        surface = combined_rationale[:500]
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
    labels = _extract_failure_labels(analyses, judge_model)
    _logger.info(
        "Phase 3: Label extraction took %.1fs",
        time.time() - t0,
    )
    for i, label in enumerate(labels):
        _logger.info("  [%d] %s", i, label)

    _logger.info("Phase 3: Clustering analyses into issues...")
    t0 = time.time()
    cluster_groups = _cluster_analyses(
        analyses, embedding_model, max_issues, labels=labels, label_model=judge_model
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
        issue = _summarize_cluster(group, analyses, analysis_model)
        _logger.info(
            "Phase 3: Summarized cluster %d/%d (%d analyses) in %.1fs",
            gi + 1,
            len(cluster_groups),
            len(group),
            time.time() - t_s,
        )
        return issue

    summaries: list[_IdentifiedIssue] = [None] * len(cluster_groups)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        if issue.confidence < _MIN_CONFIDENCE and len(group) > 1:
            _logger.info(
                "Phase 3: Re-splitting incoherent cluster '%s' "
                "(confidence=%d, %d members) into singletons",
                issue.name,
                issue.confidence,
                len(group),
            )
            for idx in group:
                resplit_groups.append([idx])

    if resplit_groups:
        resplit_summaries: list[_IdentifiedIssue] = [None] * len(resplit_groups)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_summarize_one, len(cluster_groups) + ri, group): ri
                for ri, group in enumerate(resplit_groups)
            }
            for future in as_completed(future_to_idx):
                resplit_summaries[future_to_idx[future]] = future.result()
        # Replace incoherent clusters with their singleton resummarizations
        final_groups: list[list[int]] = []
        final_summaries: list[_IdentifiedIssue] = []
        for group, issue in zip(cluster_groups, summaries):
            if issue.confidence < _MIN_CONFIDENCE and len(group) > 1:
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
        if issue.confidence >= _MIN_CONFIDENCE
        and len(issue.example_indices) >= _MIN_EXAMPLES
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
            existing.confidence = max(existing.confidence, issue.confidence)
        else:
            seen_names[key] = len(deduped)
            deduped.append(issue)
    identified = deduped

    _logger.info(
        "Phase 3 complete: %d issues identified (%d clusters filtered out) in %.1fs",
        len(identified),
        len(cluster_groups) - len(identified),
        time.time() - t0,
    )
    for issue in identified:
        example_ids = _collect_example_trace_ids(issue, analyses)
        _logger.info(
            "  %s (confidence %d): %s | root_cause: %s | examples: %s",
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
            validation_run_id=None,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Build final Issue objects directly from identified clusters.
    # (Phase 4 scorer generation and Phase 5 validation are skipped for now.)
    num_total = len(triage_traces)
    issues: list[Issue] = []
    for ident in identified:
        example_ids = _collect_example_trace_ids(ident, analyses)
        freq = len(example_ids) / max(num_total, 1)
        issues.append(
            Issue(
                name=ident.name,
                description=ident.description,
                root_cause=ident.root_cause,
                example_trace_ids=example_ids,
                scorer=None,
                frequency=freq,
                confidence=ident.confidence,
                rationale_examples=[],
            )
        )

    issues.sort(key=lambda i: i.frequency, reverse=True)

    # Phase 4: Annotate affected traces with per-issue feedback assessments
    trace_lookup = {t.info.trace_id: t for t in triage_traces}
    _logger.info(
        "Phase 4: Annotating %d traces across %d issues...", len(rationale_map), len(issues)
    )
    t0 = time.time()
    _annotate_issue_traces(issues, rationale_map, trace_lookup, judge_model)
    _logger.info("Phase 4: Annotation took %.1fs", time.time() - t0)

    summary = _build_summary(issues, num_total)
    _logger.info("Done. Found %d issues across %d traces.", len(issues), num_total)

    result = DiscoverIssuesResult(
        issues=issues,
        triage_run_id=triage_eval.run_id,
        validation_run_id=None,
        summary=summary,
        total_traces_analyzed=num_total,
    )

    # Log artifacts to the triage run
    issues_data = [
        {
            "name": issue.name,
            "description": issue.description,
            "root_cause": issue.root_cause,
            "frequency": issue.frequency,
            "confidence": issue.confidence,
            "example_trace_ids": issue.example_trace_ids,
        }
        for issue in issues
    ]
    _log_discovery_artifacts(
        triage_eval.run_id,
        {
            "summary.md": summary,
            "issues.json": json.dumps(issues_data, indent=2),
            "metadata.json": json.dumps(
                {
                    "total_traces_analyzed": num_total,
                    "num_issues": len(issues),
                    "triage_run_id": triage_eval.run_id,
                },
                indent=2,
            ),
        },
    )

    return result
