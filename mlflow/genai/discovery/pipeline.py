from __future__ import annotations

import json
import logging

import mlflow
from mlflow.genai.discovery.constants import (
    _CLUSTERING_SYSTEM_PROMPT,
    _DEFAULT_ANALYSIS_MODEL,
    _DEFAULT_JUDGE_MODEL,
    _DEFAULT_TRIAGE_SAMPLE_SIZE,
    _MAX_SUMMARIES_FOR_CLUSTERING,
    _MIN_CONFIDENCE,
    _MIN_EXAMPLES,
    _MIN_FREQUENCY_THRESHOLD,
)
from mlflow.genai.discovery.entities import DiscoverIssuesResult, Issue, _IssueClusteringResult
from mlflow.genai.discovery.utils import (
    _build_default_satisfaction_scorer,
    _build_enriched_trace_summary,
    _build_summary,
    _compute_frequencies,
    _extract_failing_traces,
    _format_analysis_for_clustering,
    _generate_scorer_specs,
    _has_session_ids,
    _log_discovery_artifacts,
    _partition_by_existing_scores,
    _run_deep_analysis,
    _test_scorer,
)
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.judges.utils.invocation_utils import (
    get_chat_completions_with_structured_output,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.types.llm import ChatMessage
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental(version="3.11.0")
def discover_issues(
    experiment_id: str | None = None,
    satisfaction_scorer: Scorer | None = None,
    judge_model: str | None = None,
    analysis_model: str | None = None,
    triage_sample_size: int = _DEFAULT_TRIAGE_SAMPLE_SIZE,
    validation_sample_size: int | None = None,
    max_issues: int = 10,
    filter_string: str | None = None,
) -> DiscoverIssuesResult:
    """Discover quality and operational issues in an experiment's traces.

    Runs a multi-phase pipeline:
    1. **Triage**: Scores a sample of traces for user satisfaction
    2. **Deep Analysis**: Extracts enriched span data and analyzes each failure
    3. **Cluster**: Groups analyses into distinct issue categories
    4. **Generate Scorers**: Writes detection instructions per issue
    5. **Validate**: Runs generated issue scorers on a broader trace set

    Args:
        experiment_id: Experiment to analyze. Defaults to the active experiment.
        satisfaction_scorer: Custom scorer for triage. Defaults to a built-in
            conversation-level satisfaction judge.
        judge_model: LLM used for scoring traces (satisfaction + issue detection).
            Defaults to ``"openai:/gpt-5-mini"``.
        analysis_model: LLM used for clustering failures into issues.
            Defaults to ``"openai:/gpt-5"``.
        triage_sample_size: Number of traces for the triage phase.
        validation_sample_size: Number of traces for validation.
            Defaults to ``5 * triage_sample_size``.
        max_issues: Maximum distinct issues to identify.
        filter_string: Filter string passed to ``search_traces``.

    Returns:
        A :class:`DiscoverIssuesResult` with discovered issues, run IDs,
        and a summary report.

    Example:

        .. code-block:: python

            import mlflow

            mlflow.set_experiment("my-genai-app")
            result = mlflow.genai.discover_issues()

            for issue in result.issues:
                print(f"{issue.name}: {issue.frequency:.1%} of traces affected")
                # Each issue has a scorer you can reuse
                mlflow.genai.evaluate(data=traces, scorers=[issue.scorer])
    """
    judge_model = judge_model or _DEFAULT_JUDGE_MODEL
    analysis_model = analysis_model or _DEFAULT_ANALYSIS_MODEL
    exp_id = experiment_id or _get_experiment_id()
    if exp_id is None:
        raise mlflow.exceptions.MlflowException(
            "No experiment specified. Use mlflow.set_experiment() or pass experiment_id."
        )

    locations = [exp_id]
    search_kwargs = {
        "filter_string": filter_string,
        "return_type": "list",
        "locations": locations,
    }

    # Phase 1: Triage — score a sample for user satisfaction
    _logger.info("Phase 1: Fetching %d traces...", triage_sample_size)
    triage_traces = mlflow.search_traces(max_results=triage_sample_size, **search_kwargs)
    if not triage_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id="",
            validation_run_id=None,
            summary=f"No traces found in experiment {exp_id}.",
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
    already_negative, _already_positive, needs_scoring = _partition_by_existing_scores(
        triage_traces, scorer_name
    )
    if already_negative:
        _logger.info(
            "Found %d traces with existing '%s' = False, %d need scoring",
            len(already_negative),
            scorer_name,
            len(needs_scoring),
        )

    _logger.info("Phase 1: Testing scorer on one trace...")
    _test_scorer(satisfaction_scorer, triage_traces[0])

    if needs_scoring:
        _logger.info("Phase 1: Scoring %d traces...", len(needs_scoring))
        with mlflow.start_run(run_name="discover_issues - triage"):
            triage_eval = mlflow.genai.evaluate(
                data=needs_scoring,
                scorers=[satisfaction_scorer],
            )
        scored_failing, rationale_map = _extract_failing_traces(
            triage_eval, satisfaction_scorer.name
        )
    else:
        triage_eval = EvaluationResult(run_id="", metrics={}, result_df=None)
        scored_failing = []
        rationale_map = {}

    failing_traces = already_negative + scored_failing
    for trace in already_negative:
        rationale_map.setdefault(trace.info.trace_id, "Previously scored as unsatisfactory")

    _logger.info(
        "Phase 1 complete: %d/%d traces unsatisfactory (%d existing, %d newly scored)",
        len(failing_traces),
        len(triage_traces),
        len(already_negative),
        len(scored_failing),
    )

    if not failing_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            validation_run_id=None,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 2: Deep Analysis — enriched span extraction + batched LLM analysis
    capped_failing = failing_traces[:_MAX_SUMMARIES_FOR_CLUSTERING]
    _logger.info("Phase 2: Deep analysis of %d failing traces...", len(capped_failing))
    enriched_summaries = [
        _build_enriched_trace_summary(i, t, rationale_map.get(t.info.trace_id, ""))
        for i, t in enumerate(capped_failing)
    ]
    analyses = _run_deep_analysis(enriched_summaries, analysis_model)
    _logger.info("Phase 2 complete: %d trace analyses produced", len(analyses))
    for a in analyses:
        _logger.info(
            "  [%d] %s (severity %d/5): %s",
            a.trace_index,
            a.failure_category,
            a.severity,
            a.failure_summary,
        )

    # Phase 3: Cluster — group analyses into distinct issues
    _logger.info("Phase 3: Clustering analyses into issues...")
    analysis_by_index = {a.trace_index: a for a in analyses}
    clustering_texts = "\n\n".join(
        _format_analysis_for_clustering(
            i,
            analysis_by_index.get(i, analyses[i] if i < len(analyses) else analyses[0]),
            enriched_summaries[i],
        )
        for i in range(len(enriched_summaries))
    )

    clustering_result = get_chat_completions_with_structured_output(
        model_uri=analysis_model,
        messages=[
            ChatMessage(
                role="system",
                content=_CLUSTERING_SYSTEM_PROMPT.format(max_issues=max_issues),
            ),
            ChatMessage(
                role="user",
                content=f"Per-trace analyses:\n\n{clustering_texts}",
            ),
        ],
        output_schema=_IssueClusteringResult,
    )

    identified = [
        issue
        for issue in clustering_result.issues[:max_issues]
        if issue.confidence >= _MIN_CONFIDENCE and len(issue.example_indices) >= _MIN_EXAMPLES
    ]
    _logger.info(
        "Phase 3 complete: %d issues identified (%d filtered out by confidence/examples)",
        len(identified),
        len(clustering_result.issues) - len(identified),
    )
    for issue in identified:
        example_ids = [
            capped_failing[idx].info.trace_id
            for idx in issue.example_indices
            if 0 <= idx < len(capped_failing)
        ]
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

    # Phase 4: Generate Scorers — write detection instructions per issue
    _logger.info("Phase 4: Generating scorers for %d issues...", len(identified))
    issue_scorers: list[Scorer] = []
    scorer_to_issue: dict[str, _IssueClusteringResult] = {}
    for issue in identified:
        example_analyses = [
            analysis_by_index[idx] for idx in issue.example_indices if idx in analysis_by_index
        ]
        specs = _generate_scorer_specs(issue, example_analyses, judge_model)
        for spec in specs:
            scorer = make_judge(
                name=spec.name,
                instructions=spec.detection_instructions,
                model=judge_model,
                feedback_value_type=bool,
            )
            issue_scorers.append(scorer)
            scorer_to_issue[spec.name] = issue
            _logger.info(
                "  Generated scorer '%s': %s", spec.name, spec.detection_instructions[:200]
            )

    triage_issues_data = [
        {
            "name": ident.name,
            "description": ident.description,
            "root_cause": ident.root_cause,
            "confidence": ident.confidence,
            "example_trace_ids": [
                capped_failing[idx].info.trace_id
                for idx in ident.example_indices
                if 0 <= idx < len(capped_failing)
            ],
            "scorer_names": [s.name for s in issue_scorers if scorer_to_issue.get(s.name) is ident],
        }
        for ident in identified
    ]
    _log_discovery_artifacts(
        triage_eval.run_id,
        {"issues.json": json.dumps(triage_issues_data, indent=2)},
    )

    # Phase 5: Validate — run issue scorers on a broader sample
    if validation_sample_size is None:
        validation_sample_size = 5 * triage_sample_size

    _logger.info(
        "Phase 5: Validating %d scorers on %d traces...",
        len(issue_scorers),
        validation_sample_size,
    )
    validation_traces = mlflow.search_traces(max_results=validation_sample_size, **search_kwargs)

    validation_eval = None
    try:
        with mlflow.start_run(run_name="discover_issues - validation"):
            validation_eval = mlflow.genai.evaluate(
                data=validation_traces,
                scorers=issue_scorers,
            )
    except Exception:
        _logger.warning(
            "Phase 5 validation failed, continuing with unvalidated issues",
            exc_info=True,
        )

    # Build final Issue objects — one Issue per scorer
    all_scorer_names = [s.name for s in issue_scorers]
    if validation_eval is not None:
        frequencies, rationale_examples_map = _compute_frequencies(
            validation_eval, all_scorer_names
        )
    else:
        num_total = len(triage_traces)
        frequencies = {}
        rationale_examples_map = {}

    scorer_by_name = {s.name: s for s in issue_scorers}
    issues: list[Issue] = []
    for scorer in issue_scorers:
        ident = scorer_to_issue[scorer.name]
        if validation_eval is not None:
            freq = frequencies.get(scorer.name, 0.0)
            if freq < _MIN_FREQUENCY_THRESHOLD:
                continue
        else:
            freq = len(ident.example_indices) / max(num_total, 1)

        example_ids = [
            failing_traces[idx].info.trace_id
            for idx in ident.example_indices
            if 0 <= idx < len(failing_traces)
        ]
        issues.append(
            Issue(
                name=scorer.name,
                description=ident.description,
                root_cause=ident.root_cause,
                example_trace_ids=example_ids,
                scorer=scorer_by_name[scorer.name],
                frequency=freq,
                confidence=ident.confidence,
                rationale_examples=rationale_examples_map.get(scorer.name, []),
            )
        )

    issues.sort(key=lambda i: i.frequency, reverse=True)
    total_analyzed = len(validation_traces) if validation_eval is not None else len(triage_traces)
    summary = _build_summary(issues, total_analyzed)
    _logger.info("Done. Found %d issues across %d traces.", len(issues), total_analyzed)

    validation_run_id = validation_eval.run_id if validation_eval is not None else None
    result = DiscoverIssuesResult(
        issues=issues,
        triage_run_id=triage_eval.run_id,
        validation_run_id=validation_run_id,
        summary=summary,
        total_traces_analyzed=total_analyzed,
    )

    # Log artifacts to validation run
    if validation_eval is not None:
        validation_issues_data = [
            {
                "name": issue.name,
                "description": issue.description,
                "root_cause": issue.root_cause,
                "frequency": issue.frequency,
                "confidence": issue.confidence,
                "example_trace_ids": issue.example_trace_ids,
                "rationale_examples": issue.rationale_examples,
            }
            for issue in issues
        ]
        _log_discovery_artifacts(
            validation_eval.run_id,
            {
                "summary.md": summary,
                "issues.json": json.dumps(validation_issues_data, indent=2),
                "metadata.json": json.dumps(
                    {
                        "total_traces_analyzed": total_analyzed,
                        "num_issues": len(issues),
                        "triage_run_id": triage_eval.run_id,
                        "validation_run_id": validation_eval.run_id,
                    },
                    indent=2,
                ),
            },
        )

    return result
