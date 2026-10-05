"""Utilities for session-level (multi-turn) evaluation."""

from __future__ import annotations

import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.rate_limiter import (
    NoOpRateLimiter,
    RateLimiter,
    call_with_retry,
    eval_retry_context,
)
from mlflow.genai.evaluation.utils import (
    add_scorer_metadata,
    make_code_type_assessment_source,
    standardize_scorer_value,
)
from mlflow.genai.scorers import Scorer
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.utils.session_hierarchy import (
    parse_session_hierarchy_levels,
    session_group_key,
)
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_SESSION_HIERARCHY

if TYPE_CHECKING:
    from mlflow.genai.evaluation.entities import EvalItem, EvalResult, ScorerStat


def classify_scorers(scorers: list[Scorer]) -> tuple[list[Scorer], list[Scorer]]:
    """
    Separate scorers into single-turn and multi-turn categories.

    Args:
        scorers: List of scorer instances.

    Returns:
        tuple: (single_turn_scorers, multi_turn_scorers)
    """
    single_turn_scorers = []
    multi_turn_scorers = []

    for scorer in scorers:
        if scorer.is_session_level_scorer:
            multi_turn_scorers.append(scorer)
        else:
            single_turn_scorers.append(scorer)

    return single_turn_scorers, multi_turn_scorers


def _get_item_session_id(item: "EvalItem") -> str | None:
    """Raw session value of an item: trace metadata first, dataset record as fallback."""
    session_id = None

    # First, try to get session_id from the trace metadata if trace exists
    if item.trace:
        trace_metadata = item.trace.info.trace_metadata
        session_id = trace_metadata.get(TraceMetadataKey.TRACE_SESSION)

    # If no session_id found in trace, check the source data (for dataset records)
    if not session_id and item.source is not None:
        session_id = item.source.source_data.get("session_id")

    return session_id


def group_traces_by_session(
    eval_items: list["EvalItem"],
    level_index: int | None = None,
) -> dict[str, list["EvalItem"]]:
    """
    Group evaluation items containing traces by session_id.

    Args:
        eval_items: List of EvalItem objects.
        level_index: Optional 0-based session-hierarchy level to group by. When given,
            hierarchical session IDs (compact JSON arrays) are grouped by their first
            ``level_index + 1`` elements; plain session IDs and shorter paths form
            their own groups. Defaults to grouping by the full session value.

    Returns:
        dict: {session_id: [eval_item, ...]} where eval items are grouped by session.
              Only items with traces that have a session_id are included in the output.
    """
    session_groups = defaultdict(list)

    for item in eval_items:
        if session_id := _get_item_session_id(item):
            if level_index is not None:
                session_id = session_group_key(session_id, level_index)
            session_groups[session_id].append(item)

    return dict(session_groups)


def resolve_session_level_index(
    level_name: str,
    eval_items: list["EvalItem"],
    run_experiment_id: str | None,
    scorer_names: list[str],
    hierarchy_cache: dict[str, list[str] | None] | None = None,
) -> int:
    """
    Resolve a session-hierarchy level name to its 0-based position.

    The position is read from the ``mlflow.experiment.sessionHierarchy`` tag of each
    trace's experiment. The evaluation run's experiment is only used as a fallback
    for items whose experiment cannot be determined (no trace, or a trace without
    one, e.g. UC-located), so a run in an untagged experiment can still evaluate
    traces from a tagged one. Tag lookups are memoized per experiment in
    ``hierarchy_cache``, which callers may share across levels in one evaluation run.

    Raises:
        MlflowException: If an experiment has no valid hierarchy tag, the level name
            is not among its levels, or the name resolves to different positions
            across the experiments in this run.
    """
    experiment_ids: set[str | None] = set()
    needs_run_fallback = False
    for item in eval_items:
        if not _get_item_session_id(item):
            # Items without a session never form a group, so they must not drag
            # the (possibly untagged) run experiment into resolution.
            continue
        if experiment_id := item.trace.info.experiment_id if item.trace else None:
            experiment_ids.add(experiment_id)
        else:
            # No trace (dataset-record fallback) or no trace experiment (e.g. UC).
            needs_run_fallback = True
    if needs_run_fallback and run_experiment_id:
        experiment_ids.add(run_experiment_id)

    if not experiment_ids:
        if not eval_items:
            # No items to group, so no group can form; any index is equivalent.
            return 0
        raise MlflowException.invalid_parameter_value(
            f"Session-level scorer(s) {scorer_names} set session_level={level_name!r}, but no "
            "experiment could be determined to read the session hierarchy from."
        )

    hierarchy_cache = hierarchy_cache if hierarchy_cache is not None else {}
    client = MlflowClient()
    resolved: dict[str, int] = {}
    for experiment_id in sorted(experiment_ids):
        if experiment_id in hierarchy_cache:
            levels = hierarchy_cache[experiment_id]
        else:
            # A failed experiment fetch (deleted experiment, permissions) propagates
            # its own error rather than masquerading as a missing hierarchy tag.
            tags = client.get_experiment(experiment_id).tags
            levels = parse_session_hierarchy_levels(tags.get(MLFLOW_EXPERIMENT_SESSION_HIERARCHY))
            hierarchy_cache[experiment_id] = levels
        if not levels:
            raise MlflowException.invalid_parameter_value(
                f"Session-level scorer(s) {scorer_names} set session_level={level_name!r}, but "
                f"experiment {experiment_id} does not have a valid "
                f"'{MLFLOW_EXPERIMENT_SESSION_HIERARCHY}' tag. Declare the hierarchy with "
                "mlflow.genai.set_session_hierarchy([...])."
            )
        if level_name not in levels:
            raise MlflowException.invalid_parameter_value(
                f"Session-level scorer(s) {scorer_names} set session_level={level_name!r}, but "
                f"that level is not declared by experiment {experiment_id}. Its hierarchy "
                f"levels are {levels}. Declare it with mlflow.genai.set_session_hierarchy([...])."
            )
        resolved[experiment_id] = levels.index(level_name)

    positions = set(resolved.values())
    if len(positions) > 1:
        raise MlflowException.invalid_parameter_value(
            f"Session-level scorer(s) {scorer_names} set session_level={level_name!r}, but that "
            f"level resolves to different positions across the experiments in this run: "
            f"{resolved}. All traces must share a hierarchy where '{level_name}' has one "
            "position."
        )
    return positions.pop()


def group_sessions_by_scorer_level(
    eval_items: list["EvalItem"],
    multi_turn_scorers: list[Scorer],
    run_experiment_id: str | None,
) -> list[tuple[dict[str, list["EvalItem"]], list[Scorer]]]:
    """
    Bucket multi-turn scorers by ``session_level`` and group traces for each bucket.

    Scorers with ``session_level=None`` keep today's full-session-value grouping. Each
    distinct level name yields one (session_groups, scorers) pair, with groups keyed
    by the level prefix of hierarchical session IDs.

    Returns:
        list: One (session_groups, scorers) tuple per distinct session_level.
    """
    scorers_by_level: dict[str | None, list[Scorer]] = defaultdict(list)
    for scorer in multi_turn_scorers:
        scorers_by_level[scorer.session_level].append(scorer)

    hierarchy_cache: dict[str, list[str] | None] = {}
    buckets = []
    for level, scorers in scorers_by_level.items():
        if level is None:
            groups = group_traces_by_session(eval_items)
        else:
            level_index = resolve_session_level_index(
                level,
                eval_items,
                run_experiment_id,
                [s.name for s in scorers],
                hierarchy_cache,
            )
            groups = group_traces_by_session(eval_items, level_index=level_index)
        buckets.append((groups, scorers))
    return buckets


def get_first_trace_in_session(session_items: list["EvalItem"]) -> "EvalItem":
    """
    Find the chronologically first trace in a session based on request_time.

    Args:
        session_items: List of EvalItem objects from the same session.

    Returns:
        EvalItem: The eval item with the earliest trace in chronological order.
    """
    return min(session_items, key=lambda x: x.trace.info.request_time)


def evaluate_session_level_scorers(
    session_id: str,
    session_items: list["EvalItem"],
    multi_turn_scorers: list[Scorer],
    scorer_rate_limiter: RateLimiter = NoOpRateLimiter(),
    max_retries: int = 0,
) -> EvalResult:
    """
    Evaluate all multi-turn scorers for a single session.

    Args:
        session_id: The session identifier
        session_items: List of EvalItem objects from the same session
        multi_turn_scorers: List of multi-turn scorer instances
        scorer_rate_limiter: Rate limiter to throttle scorer invocations.
        max_retries: Max 429-retry attempts per scorer call.

    Returns:
        EvalResult containing the assessments from all multi-turn scorers for this session.
        The result is associated with the first item in the session (chronologically by
        trace timestamp), and multi-turn assessments will be logged to that trace.
    """
    # Import lazily here since mlflow.genai.evaluation.entities imports pandas at the top level
    # (needed for EvalResult.to_pd_series() and result DataFrame operations). By importing inside
    # the function, we avoid loading pandas when this module is imported, improving startup time.
    from mlflow.genai.evaluation.entities import EvalResult, ScorerStat

    first_item = get_first_trace_in_session(session_items)
    session_traces = [item.trace for item in session_items]

    def run_scorer(scorer: Scorer) -> list[Feedback]:
        try:
            with eval_retry_context():
                value = call_with_retry(
                    lambda: scorer.run(session=session_traces),
                    scorer_rate_limiter,
                    max_retries,
                )
            feedbacks = standardize_scorer_value(scorer.name, value)

            # Add session_id to metadata for each feedback
            for feedback in feedbacks:
                if feedback.metadata is None:
                    feedback.metadata = {}
                feedback.metadata[TraceMetadataKey.TRACE_SESSION] = session_id

            add_scorer_metadata(scorer, feedbacks)

            return feedbacks
        except Exception as e:
            feedbacks = [
                Feedback(
                    name=scorer.name,
                    source=make_code_type_assessment_source(scorer.name),
                    error=AssessmentError(
                        error_code="SCORER_ERROR",
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                    ),
                    metadata={TraceMetadataKey.TRACE_SESSION: session_id},
                )
            ]
            add_scorer_metadata(scorer, feedbacks)
            return feedbacks

    # Run scorers in parallel (similar to _compute_eval_scores for single-turn)
    with ThreadPoolExecutor(
        max_workers=len(multi_turn_scorers),
        thread_name_prefix="MlflowGenAIEvalMultiTurnScorer",
    ) as executor:
        futures = {executor.submit(run_scorer, scorer): scorer for scorer in multi_turn_scorers}

        try:
            results = [(scorer, future.result()) for future, scorer in futures.items()]
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise

    # Track scorer stats
    scorer_stats: dict[str, ScorerStat] = {}
    all_feedbacks = []
    for scorer, feedbacks in results:
        scorer_name = scorer.name
        if scorer_name not in scorer_stats:
            scorer_stats[scorer_name] = ScorerStat()
        failed = len(feedbacks) == 1 and feedbacks[0].error is not None
        scorer_stats[scorer_name].record_invocation(failed=failed)
        all_feedbacks.extend(feedbacks)

    return EvalResult(
        eval_item=first_item,
        assessments=all_feedbacks,
        scorer_stats=scorer_stats,
    )


def validate_session_level_evaluation_inputs(scorers: list[Scorer], predict_fn: Any) -> None:
    """
    Validate input parameters when session-level scorers are present.

    Args:
        scorers: List of scorer instances
        predict_fn: Prediction function (if provided)

    Raises:
        MlflowException: If invalid configuration is detected
    """
    # `session_level` is validated here (not at scorer construction) because
    # subclasses like InstructionsJudge only know their level after init.
    for scorer in scorers:
        scorer._validate_session_level()

    if session_level_scorers := [scorer for scorer in scorers if scorer.is_session_level_scorer]:
        if predict_fn is not None:
            scorer_names = [scorer.name for scorer in session_level_scorers]
            raise MlflowException.invalid_parameter_value(
                f"Session-level scorers require traces with session IDs. "
                f"The following scorers are session-level: {scorer_names}. "
                f"Either pass a ConversationSimulator to `data` with `predict_fn`, "
                f"or pass existing traces containing session IDs to `data` "
                f"(e.g., `data=mlflow.search_traces()`) without `predict_fn`."
            )
