from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pydantic

import mlflow
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.issue import Issue, IssueSeverity
from mlflow.entities.issue_evidence import IssueEvidence
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
    CATEGORY_COST,
    CATEGORY_LATENCY,
    DEDUP_ISSUES_PROMPT_TEMPLATE,
    DEFAULT_CATEGORIES,
    DEFAULT_MODEL,
    DEFAULT_SCORER_NAME,
    NO_ISSUE_KEYWORD,
    TRACE_ANNOTATION_SYSTEM_PROMPT,
    build_satisfaction_instructions,
)
from mlflow.genai.discovery.entities import (
    CostAnomaly,
    DiscoverIssuesResult,
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.evidence import generate_trace_evidence
from mlflow.genai.discovery.extraction import (
    collect_session_rationales,
    extract_execution_paths_for_session,
    extract_failing_traces,
    extract_failure_labels,
)
from mlflow.genai.discovery.matching import (
    MATCHABLE_STATUSES,
    MatchOutcome,
    match_findings_to_existing_issues,
    read_match_candidates,
)
from mlflow.genai.discovery.sampling import sample_traces
from mlflow.genai.discovery.utils import (
    analyze_cost_anomalies,
    build_summary,
    collect_affected_trace_ids,
    compute_latency_percentiles,
    format_annotation_prompt,
    format_trace_content,
    get_session_id,
    group_traces_by_session,
    log_discovery_artifacts,
    verify_scorer,
)
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.utils.llm_utils import _call_llm, _TokenCounter
from mlflow.telemetry.events import DiscoverIssuesEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_ISSUE_DETECTION
from mlflow.utils.time import get_current_time_millis

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
    categories: list[str] | None = None,
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

    include_timing = CATEGORY_LATENCY in (categories or [])

    def _annotate_one(item: _AnnotationWorkItem) -> str | None:
        trace = trace_lookup.get(item.trace_id)
        trace_content = (
            format_trace_content(trace, include_timing=include_timing)
            if trace
            else "(trace not available)"
        )
        user_content = format_annotation_prompt(
            item.issue,
            trace_content,
            item.triage_rationale,
            categories=categories,
        )

        evidence = None
        annotation = ""
        if trace is not None:
            # A span-anchored walkthrough is what the Issues tab can navigate, so it is tried
            # first. Generation and validation are both fallible, and either way the summary
            # it produced is still the best prose available for the reference's rationale.
            try:
                evidence, annotation = generate_trace_evidence(
                    issue_name=item.issue.name,
                    issue_description=item.issue.description or "",
                    root_causes="; ".join(item.issue.root_causes or []),
                    trace=trace,
                    trace_content=trace_content,
                    triage_rationale=item.triage_rationale,
                    categories=categories,
                    model=model,
                    token_counter=token_counter,
                )
            except Exception:
                _logger.debug(
                    "Failed to generate evidence for trace %s", item.trace_id, exc_info=True
                )

        if not annotation:
            # Prose only. Reached when the trace is unavailable or evidence generation failed,
            # and still records that this trace exhibits the issue.
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

        def _log(with_evidence: IssueEvidence | None) -> None:
            mlflow.log_issue(
                trace_id=item.trace_id,
                issue_id=item.issue.issue_id,
                issue_name=item.issue.name,
                source=source,
                evidence=with_evidence,
                rationale=annotation,
                metadata=metadata,
            )

        try:
            _log(evidence)
        except Exception:
            if evidence is None:
                _logger.debug("Failed to log issue for trace %s", item.trace_id, exc_info=True)
                return None
            # Structured evidence is the only part of the reference big enough to be refused:
            # the assessment metadata column holds 65,535 bytes and nothing bounds the step
            # count that produced it. Dropping the walkthrough costs the reader a span-anchored
            # explanation; dropping the reference costs them the fact that this trace exhibits
            # the issue at all, which is the whole point of annotating it. So retry once
            # without it, and say so rather than leaving a silent gap between what was
            # generated and what was stored.
            _logger.warning(
                "Could not store structured evidence for trace %s; recording the issue "
                "reference with its summary only",
                item.trace_id,
                exc_info=True,
            )
            try:
                _log(None)
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
    categories_map: dict[str, list[str]],
    scorer_name: str,
    categories: list[str] = DEFAULT_CATEGORIES,
) -> tuple[list[_ConversationAnalysis], dict[str, list[Trace]]]:
    """
    Build per-session analyses from triage results.

    For each session with failing traces, combines triage rationales,
    human feedback, and span errors into a single ConversationAnalysis.

    Args:
        triage_traces: All traces from the triage phase (passing and failing), re-fetched so
            they carry the assessments written during evaluation.
        rationale_map: Mapping of trace_id to triage rationale for failing traces.
        categories_map: Mapping of trace_id to category tags from structured output.
        scorer_name: Name of the triage scorer, used to look up human feedback.
        categories: Known valid categories to filter against.

    Returns:
        A tuple of (analyses, session_groups) where session_groups maps
        session_id to the list of traces in that session.
    """
    known_lower = {c.lower() for c in categories}
    session_groups = group_traces_by_session(triage_traces)
    analyses: list[_ConversationAnalysis] = []
    for _, session_traces in session_groups.items():
        session_failing = [
            trace for trace in session_traces if trace.info.trace_id in rationale_map
        ]
        if not session_failing:
            continue
        # The whole session, not just its failing turns: conversation-scoped feedback is often
        # recorded on the first turn, which may itself have passed.
        combined_rationale = collect_session_rationales(
            session_failing, rationale_map, scorer_name, session_traces=session_traces
        )
        if not combined_rationale:
            continue
        exec_path = extract_execution_paths_for_session(session_failing)
        session_cats = [
            cat
            for trace in session_failing
            for cat in categories_map.get(trace.info.trace_id, [])
            if cat.lower() in known_lower
        ]
        analyses.append(
            _ConversationAnalysis(
                full_rationale=combined_rationale,
                affected_trace_ids=[trace.info.trace_id for trace in session_failing],
                execution_path=exec_path,
                categories=list(dict.fromkeys(session_cats)),
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


class _DedupGroup(pydantic.BaseModel):
    indices: list[int] = pydantic.Field(
        description=(
            "List of issue indices (0-based) that represent the same underlying problem "
            "and should be merged. Must contain 2 or more indices."
        )
    )
    name: str = pydantic.Field(
        description=(
            "Consolidated title for this group. "
            "Use the format 'Issue: <short description>' (3-8 words), "
            "e.g. 'Issue: Incomplete response details'."
        )
    )
    description: str = pydantic.Field(
        description="A unified description of the shared symptom across all issues in the group."
    )
    root_cause: str = pydantic.Field(
        description="The common root cause across all issues in the group."
    )


class _DedupGroups(pydantic.BaseModel):
    groups: list[_DedupGroup] = pydantic.Field(
        description=(
            "List of duplicate groups. Each group contains the indices of issues "
            "that represent the same underlying problem and should be merged, "
            "along with a consolidated name, description, and root cause for the group. "
            "Only include groups with 2 or more indices. "
            "Issues that have no duplicates should NOT appear in any group."
        )
    )


def _dedup_issues(
    issues: list[_IdentifiedIssue],
    model: str = DEFAULT_MODEL,
    token_counter: _TokenCounter | None = None,
) -> list[_IdentifiedIssue]:
    """Deduplicate issues by asking the LLM to identify groups of equivalent issues."""
    if len(issues) < 2:
        return issues

    issue_list = "\n".join(
        f"[{i}] {issue.name}: {issue.description} (root cause: {issue.root_cause})"
        for i, issue in enumerate(issues)
    )
    prompt = DEDUP_ISSUES_PROMPT_TEMPLATE.format(issue_list=issue_list)

    try:
        response = _call_llm(
            model,
            [{"role": "user", "content": prompt}],
            response_format=_DedupGroups,
            token_counter=token_counter,
        )
        result = _DedupGroups.model_validate_json(response.choices[0].message.content)
    except Exception:
        _logger.debug("Failed to deduplicate issues via LLM; skipping dedup", exc_info=True)
        return issues

    parent = list(range(len(issues)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    group_by_root: dict[int, _DedupGroup] = {}
    for group in result.groups:
        if (
            len(group.indices) < 2
            or len(set(group.indices)) != len(group.indices)
            or any(idx < 0 or idx >= len(issues) for idx in group.indices)
        ):
            _logger.debug("Skipping invalid deduplication group: %s", group.indices)
            continue
        root = min(group.indices)
        group_by_root[root] = group
        for idx in group.indices:
            ra = find(group.indices[0])
            rb = find(idx)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    # Map each index to the minimum index in its component (canonical representative)
    canonical = {i: find(i) for i in range(len(issues)) if find(i) != i}

    merged: dict[int, _IdentifiedIssue] = {}
    for i, issue in enumerate(issues):
        head = canonical.get(i, i)
        if head not in merged:
            merged[head] = issue
        else:
            target = merged[head]
            target.example_indices = list(set(target.example_indices + issue.example_indices))
            target.severity = max(target.severity, issue.severity)
            target.categories = list(dict.fromkeys(target.categories + issue.categories))

    for root, group in group_by_root.items():
        if root in merged:
            merged[root].name = group.name
            merged[root].description = group.description
            merged[root].root_cause = group.root_cause

    # Return issues in their original order
    return [merged[i] for i in sorted(merged)]


def _merge_singleton_issues(
    identified: list[_IdentifiedIssue],
    analysis_labels: dict[int, str],
    analyses: list[_ConversationAnalysis],
    model: str,
    max_issues: int,
    categories: list[str],
    token_counter: _TokenCounter | None = None,
    instructions: str | None = None,
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
        categories=categories,
        token_counter=token_counter,
        instructions=instructions,
    )
    return multi_member + merged


def _cluster_and_identify(
    analyses: list[_ConversationAnalysis],
    model: str,
    max_issues: int,
    categories: list[str],
    token_counter: _TokenCounter | None = None,
    instructions: str | None = None,
) -> list[_IdentifiedIssue]:
    """Cluster analyses into identified issues via LLM-based labeling and grouping."""
    labels, label_to_analysis = extract_failure_labels(
        analyses,
        model,
        token_counter=token_counter,
    )
    for i, label in enumerate(labels):
        _logger.debug("  [%d] %s", i, label)

    if len(labels) == 1:
        cluster_groups = [[0]]
    else:
        cluster_groups = cluster_by_llm(
            labels,
            max_issues,
            model,
            categories=categories,
            token_counter=token_counter,
        )
    _logger.debug("Clustering produced %d groups", len(cluster_groups))

    def summarize_fn(group: list[int]) -> _IdentifiedIssue:
        return summarize_cluster(
            group,
            analyses,
            model,
            label_to_analysis=label_to_analysis,
            categories=categories,
            token_counter=token_counter,
            instructions=instructions,
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
    identified = _dedup_issues(identified, model=model, token_counter=token_counter)

    analysis_labels: dict[int, str] = {}
    for label, analysis_idx in zip(labels, label_to_analysis):
        analysis_labels.setdefault(analysis_idx, label)
    return _merge_singleton_issues(
        identified,
        analysis_labels,
        analyses,
        model,
        max_issues,
        categories=categories,
        token_counter=token_counter,
        instructions=instructions,
    )


class _MatchNoLongerValid(Exception):
    """The matched issue stopped being matchable between the candidate read and the write."""


def _build_issues(
    identified: list[_IdentifiedIssue],
    analyses: list[_ConversationAnalysis],
    exp_id: str,
    source_run_id: str,
    match_outcome: MatchOutcome | None = None,
) -> tuple[list[Issue], dict[str, list[str]]]:
    """
    Convert identified issues into Issue entities.

    A finding matched to an issue this experiment already carries refreshes that issue rather
    than filing a second one: its last-seen time moves forward and this run's traces are
    annotated against it. The user's triage is left alone -- a rejected issue stays rejected,
    and the name and description someone edited are not overwritten by this run's wording.

    Returns:
        A tuple of (issues, issue_trace_ids) where issue_trace_ids maps
        issue_id to the list of affected trace IDs for annotation.
    """
    from mlflow.tracing.client import TracingClient

    client = TracingClient()
    issues: list[Issue] = []
    issue_trace_ids: dict[str, list[str]] = {}
    for index, ident in enumerate(identified):
        affected_trace_ids = collect_affected_trace_ids(ident, analyses)
        name = ident.name.removeprefix("Issue: ").removeprefix("issue: ")
        if matched_id := (match_outcome.issue_id_for(index) if match_outcome else None):
            try:
                # Re-read rather than trust the candidate: matching takes a model call, and the
                # issue can be resolved in that window. Refreshing it then would bury a
                # regression in the issue someone had just closed, which is the outcome
                # excluding resolved issues from the candidates exists to prevent.
                current = client._get_issue(matched_id)
                if current.status not in MATCHABLE_STATUSES:
                    _logger.info(
                        "Issue %s became %s while matching; reporting the finding as new",
                        matched_id,
                        current.status,
                    )
                    raise _MatchNoLongerValid
                issue = client._update_issue(
                    issue_id=matched_id,
                    # Monotonic: a run over older traces must not drag the issue's last-seen
                    # time backwards past what a later run already recorded.
                    last_seen_timestamp=max(
                        get_current_time_millis(),
                        current.last_seen_timestamp or 0,
                    ),
                )
                # The re-read above closes the window the model call opens, but not the gap
                # between that read and this write: nothing in the store conditions the update on
                # status. Judge the issue the store handed back rather than the one that was read,
                # so a resolution landing in that gap still sends the finding to a new issue
                # instead of annotating this run's traces onto the row someone just closed.
                if issue.status not in MATCHABLE_STATUSES:
                    _logger.info(
                        "Issue %s became %s while being refreshed; reporting the finding as new",
                        matched_id,
                        issue.status,
                    )
                    raise _MatchNoLongerValid
            except _MatchNoLongerValid:
                pass
            except Exception:
                # Losing the match costs a duplicate; losing the finding costs the user the
                # problem, so it falls through and is filed.
                _logger.warning(
                    "Could not refresh already-reported issue %s; reporting the finding as new",
                    matched_id,
                    exc_info=True,
                )
            else:
                # Several findings in one run can re-find one Issue: it is refreshed once and
                # carries all of their traces, rather than the last assignment winning and the
                # earlier findings' traces going unannotated.
                if issue.issue_id in issue_trace_ids:
                    known = issue_trace_ids[issue.issue_id]
                    issue_trace_ids[issue.issue_id] = known + [
                        trace_id for trace_id in affected_trace_ids if trace_id not in known
                    ]
                else:
                    issues.append(issue)
                    issue_trace_ids[issue.issue_id] = affected_trace_ids
                continue
        issue = client._create_issue(
            experiment_id=exp_id,
            name=name,
            description=ident.description,
            severity=ident.severity,
            categories=ident.categories,
            root_causes=[ident.root_cause],
            source_run_id=source_run_id,
        )
        issues.append(issue)
        issue_trace_ids[issue.issue_id] = affected_trace_ids

    issues.sort(key=_severity_sort_key, reverse=True)
    return issues, issue_trace_ids


# Declaration order is the severity order, so the index is the rank.
_SEVERITY_RANK = {severity.value: rank for rank, severity in enumerate(IssueSeverity)}
# Below every known severity, so an issue whose severity cannot be read sorts last. An absent
# severity is not evidence of a severe issue.
_UNKNOWN_SEVERITY_RANK = -1


def _severity_sort_key(issue: Issue) -> int:
    """
    Rank for ordering issues by severity, most severe first under ``reverse=True``.

    Ranked explicitly rather than by comparing the field, for two reasons the review surfaced.

    ``Issue`` keeps whatever severity it was given, so a row read back from the store holds a
    plain string rather than an :class:`IssueSeverity`. ``IssueSeverity.__lt__`` returns
    ``NotImplemented`` for a non-member, and the enum subclasses ``str``, so Python fell back
    to string comparison -- which is alphabetical. ``"high" < "low"`` is true, so the most
    severe issues sorted to the *bottom* of the list.

    The column is also nullable, so a single untyped row raised ``TypeError`` from this sort.
    It runs after every database write but before evidence annotation, so the run published
    its issues and then lost all of their trace annotations.
    """
    if issue.severity is None:
        return _UNKNOWN_SEVERITY_RANK
    return _SEVERITY_RANK.get(str(issue.severity), _UNKNOWN_SEVERITY_RANK)


def build_issue_discovery_scorer(
    categories: list[str] | None = None,
    model: str | None = None,
    use_conversation: bool = True,
    latency_stats: dict[str, float] | None = None,
    instructions: str | None = None,
) -> Scorer:
    model = model or DEFAULT_MODEL
    categories = categories if categories is not None else DEFAULT_CATEGORIES

    include_timing = use_conversation and CATEGORY_LATENCY in categories

    judge_instructions = build_satisfaction_instructions(
        use_conversation=use_conversation,
        categories=categories,
        latency_stats=latency_stats,
        instructions=instructions,
    )
    return make_judge(
        name=DEFAULT_SCORER_NAME,
        instructions=judge_instructions,
        model=model,
        feedback_value_type=dict[str, str],
        include_timing_in_conversation=include_timing,
    )


def _lookup_by_trace_id(traces: list[Trace]) -> dict[str, Trace]:
    """Traces by ID, skipping any whose ID cannot be read rather than keying them positionally."""
    lookup = {}
    for trace in traces:
        try:
            lookup[trace.info.trace_id] = trace
        except Exception:
            continue
    return lookup


def _merge_cost_findings(
    triage,
    cost_anomalies: list[CostAnomaly],
    trace_lookup: dict[str, Trace],
) -> int:
    """
    Carry cost anomalies into triage so a trace that only costs too much is still a finding.

    A trace can achieve the user's goal, return ``OK``, and still spend many times what its
    own agent label normally does. Triage judges output quality, so such a trace passes and
    the anomaly would never reach clustering. Each anomaly is added as a failing trace with a
    deterministic rationale -- the observed spend, what it is compared against, and the driver
    -- so it groups and reports through the same path as every other finding.

    An unpriced trace never reaches here: it is excluded while grouping and disclosed in the
    run's log, because a missing cost is not a cheap trace.

    Returns:
        How many traces this added to the failing set.
    """
    already_failing = {trace.info.trace_id for trace in triage.failing_traces}
    added = 0
    for anomaly in cost_anomalies:
        trace = trace_lookup.get(anomaly.trace_id)
        if trace is None:
            continue
        rationale = f"[cost anomaly] spend is {anomaly.describe()}"
        if anomaly.trace_id in already_failing:
            # The trace already failed on quality; append the cost signal so clustering sees
            # both, rather than replacing the quality rationale with it.
            existing = triage.rationale_map.get(anomaly.trace_id, "")
            triage.rationale_map[anomaly.trace_id] = (
                f"{existing}; {rationale}" if existing else rationale
            )
        else:
            triage.failing_traces.append(trace)
            triage.rationale_map[anomaly.trace_id] = rationale
            added += 1
        categories = triage.categories_map.setdefault(anomaly.trace_id, [])
        if CATEGORY_COST not in categories:
            categories.append(CATEGORY_COST)
    return added


@record_usage_event(DiscoverIssuesEvent)
def discover_issues(
    experiment_id: str | None = None,
    traces: list[Trace] | None = None,
    scorers: list[Scorer] | None = None,
    model: str | None = None,
    max_issues: int = 20,
    filter_string: str | None = None,
    run_id: str | None = None,
    categories: list[str] = DEFAULT_CATEGORIES,
    instructions: str | None = None,
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
        run_id: Run ID to attach issues to. If not provided, a new run will be created.
        categories: Issue categories to search for.
        instructions: Free-text detection priorities saved on the experiment, applied to the
            stages that decide what to flag and how severe it is. Steers the run; it cannot
            redefine the task.

    Returns:
        A :class:`DiscoverIssuesResult` with discovered issues, run IDs,
        and a summary report.

    Raises:
        CandidateReadError: If the issues this experiment already carries could not be read.
            Deliberately fatal: publishing without that list would report every finding as new
            and file a duplicate of everything already known, so the run fails instead. The job
            wrapper marks the run FAILED and re-raises, which is the outcome a caller wants --
            a detection run that could not be deduplicated is not a result worth keeping.
    """
    from mlflow.server.jobs.progress import update_status_details

    pipeline_start = time.time()
    model = model or DEFAULT_MODEL
    token_counter = _TokenCounter(model=model)

    exp_id = experiment_id or _get_experiment_id()

    # ---- Phase 1: Triage ----
    update_status_details({"stage": "Sampling traces for analysis..."})
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
            total_cost_usd=0.0,
        )

    use_conversation = False
    if scorers is None:
        use_conversation = any(get_session_id(trace) for trace in triage_traces)
        if not use_conversation:
            _logger.debug("No session IDs found, falling back to trace-level scorer")

        latency_stats = (
            compute_latency_percentiles(triage_traces) if CATEGORY_LATENCY in categories else None
        )

        default_scorer = build_issue_discovery_scorer(
            categories=categories,
            model=model,
            use_conversation=use_conversation,
            latency_stats=latency_stats,
            instructions=instructions,
        )
        scorers = [default_scorer]

    scorer_name = scorers[0].name

    test_session = None
    if use_conversation:
        session_groups = group_traces_by_session(triage_traces)
        test_session = next(
            (traces for traces in session_groups.values() if get_session_id(traces[0])),
            None,
        )

    update_status_details({"stage": "Verifying configuration..."})
    verify_scorer(
        scorers[0], test_session[0] if test_session else triage_traces[0], session=test_session
    )

    update_status_details({"stage": "Identifying issues from traces..."})
    with mlflow.start_run(run_id=run_id, tags={MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_ISSUE_DETECTION}):
        triage_eval = mlflow.genai.evaluate(
            data=triage_traces,
            scorers=scorers,
        )

    # Re-fetch traces with scorer assessments attached.
    scored_traces = triage_traces
    try:
        fetched = [mlflow.get_trace(t.info.trace_id) for t in triage_traces]
        fetched = [f for f in fetched if f is not None]
        if len(fetched) == len(triage_traces):
            scored_traces = fetched
        else:
            _logger.debug(
                "Could not re-fetch %d/%d traces, using originals",
                len(triage_traces) - len(fetched),
                len(triage_traces),
            )
        # Aggregate judge costs from Phase 1 evaluation assessments.
        for trace in scored_traces:
            for assessment in trace.info.assessments or []:
                meta = getattr(assessment, "metadata", None) or {}
                if meta.get(AssessmentMetadataKey.SOURCE_RUN_ID) != triage_eval.run_id:
                    continue
                if cost := meta.get(AssessmentMetadataKey.JUDGE_COST):
                    token_counter.add_cost(float(cost))
                if input_tok := meta.get(AssessmentMetadataKey.JUDGE_INPUT_TOKENS):
                    token_counter.input_tokens += int(input_tok)
                if output_tok := meta.get(AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS):
                    token_counter.output_tokens += int(output_tok)
    except Exception:
        _logger.debug("Failed to fetch scored traces", exc_info=True)

    scorer_names = [s.name for s in scorers]
    triage = extract_failing_traces(scored_traces, scorer_names)

    # Cost is judged across the sample, because it is the one category a per-trace judge
    # cannot reach. This is the traced application's spend; the detector's own is owned by
    # `token_counter`.
    cost_anomalies = analyze_cost_anomalies(scored_traces) if CATEGORY_COST in categories else []

    # Keyed on the real trace ID: an anomaly only ever carries one, because a trace whose ID
    # could not be read was excluded while grouping.
    cost_findings = _merge_cost_findings(triage, cost_anomalies, _lookup_by_trace_id(scored_traces))
    if cost_findings:
        _logger.info(
            "%d traces were flagged for cost alone, as outliers against their own peers",
            cost_findings,
        )

    if triage.scorer_error_trace_ids:
        # The messages are included because the count alone does not say whether this was a
        # rate limit, a bad key or a malformed trace -- and that is the difference between
        # rerunning the detection and fixing something first. Bounded so one broken scorer
        # cannot flood the run's log with the same line.
        sampled = "; ".join(
            f"{trace_id}: {triage.scorer_errors[trace_id]}"
            for trace_id in triage.scorer_error_trace_ids[:3]
            if trace_id in triage.scorer_errors
        )
        _logger.warning(
            "%d/%d traces could not be judged because their scorer errored, so they are "
            "excluded from trace-quality classification rather than counted as passing%s",
            len(triage.scorer_error_trace_ids),
            len(triage_traces),
            f". First errors -- {sampled}" if sampled else "",
        )
    if triage.unclassified_trace_ids:
        _logger.warning(
            "%d/%d traces were left unclassified because their scorer returned no value, so "
            "neither pass nor fail was established and they support no conclusion either way",
            len(triage.unclassified_trace_ids),
            len(triage_traces),
        )
    _logger.info(
        "Triage complete: %d/%d traces unsatisfactory",
        len(triage.failing_traces),
        len(triage_traces),
    )

    if not triage.failing_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            summary=build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
            total_cost_usd=token_counter.cost_usd,
        )

    # ---- Phase 2: Build analyses ----
    update_status_details({"stage": "Analyzing results..."})
    analyses, session_groups = _build_analyses(
        scored_traces,
        triage.rationale_map,
        triage.categories_map,
        scorer_name,
        categories=categories,
    )

    # ---- Phase 3: Cluster & identify ----
    update_status_details({"stage": "Clustering issues..."})
    identified = _cluster_and_identify(
        analyses,
        model,
        max_issues,
        categories=categories,
        token_counter=token_counter,
        instructions=instructions,
    )

    if not identified:
        return DiscoverIssuesResult(
            issues=[],
            triage_run_id=triage_eval.run_id,
            summary=build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
            total_cost_usd=token_counter.cost_usd,
        )

    # ---- Phase 4: Match against what is already reported, then build issues & annotate ----
    update_status_details({"stage": "Checking already-reported issues..."})
    # Read before publishing, and fail rather than publish against an unread list: treating a
    # failed read as "nothing reported yet" would file a duplicate of every known issue.
    candidates = read_match_candidates(exp_id)
    match_outcome = match_findings_to_existing_issues(
        identified, candidates, model=model, token_counter=token_counter
    )
    if match_outcome.already_reported:
        _logger.info(
            "%d of %d findings were already reported, out of %d issues considered",
            match_outcome.already_reported,
            len(identified),
            match_outcome.candidates_considered,
        )

    update_status_details({"stage": "Annotating issues..."})
    issues, issue_trace_ids = _build_issues(
        identified, analyses, exp_id, triage_eval.run_id, match_outcome
    )

    trace_to_session: dict[str, str] = {}
    for session_id, traces_in_session in session_groups.items():
        for trace in traces_in_session:
            trace_to_session[trace.info.trace_id] = session_id

    trace_lookup = {trace.info.trace_id: trace for trace in scored_traces}
    session_first_trace: dict[str, str] | None = None
    if use_conversation:
        session_first_trace = {
            session_id: traces_in_session[0].info.trace_id
            for session_id, traces_in_session in session_groups.items()
        }
    _annotate_issue_traces(
        issues,
        issue_trace_ids,
        triage.rationale_map,
        trace_lookup,
        model,
        categories=categories,
        trace_to_session=trace_to_session if use_conversation else None,
        session_first_trace=session_first_trace,
        token_counter=token_counter,
    )

    update_status_details({"stage": "Generating summary..."})
    summary = build_summary(issues, len(triage_traces))
    _logger.info("Done. Found %d issues across %d traces.", len(issues), len(triage_traces))

    result = DiscoverIssuesResult(
        issues=issues,
        triage_run_id=triage_eval.run_id,
        summary=summary,
        total_traces_analyzed=len(triage_traces),
        total_cost_usd=token_counter.cost_usd,
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
