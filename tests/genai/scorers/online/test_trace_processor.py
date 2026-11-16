import json
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import Feedback, Trace, TraceData, TraceInfo
from mlflow.entities.assessment import Assessment, AssessmentError
from mlflow.entities.trace_location import (
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.genai.scorers.builtin_scorers import Completeness
from mlflow.genai.scorers.online.constants import (
    MAX_SCORER_FAILURE_RETRY_MS,
    MAX_TRACE_COMPLETION_WAIT_MS,
)
from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.trace_checkpointer import (
    OnlineTraceCheckpointManager,
    OnlineTraceScoringCheckpoint,
    OnlineTraceScoringTimeWindow,
)
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.genai.scorers.online.trace_processor import (
    OnlineTraceScoringProcessor,
)
from mlflow.tracing.constant import AssessmentMetadataKey

_PROCESSOR_MODULE = "mlflow.genai.scorers.online.trace_processor"


def make_online_scorer(scorer, sample_rate: float = 1.0, filter_string: str | None = None):
    config = OnlineScoringConfig(
        online_scoring_config_id=uuid.uuid4().hex,
        scorer_id=uuid.uuid4().hex,
        sample_rate=sample_rate,
        experiment_id="exp1",
        filter_string=filter_string,
    )
    return OnlineScorer(
        name=scorer.name,
        serialized_scorer=json.dumps(scorer.model_dump()),
        online_config=config,
    )


def make_trace_info(
    trace_id: str,
    timestamp_ms: int = 1000,
    state: TraceState = TraceState.OK,
    assessments: list[Assessment] | None = None,
    execution_duration: int | None = None,
) -> TraceInfo:
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="exp1"),
        ),
        request_time=timestamp_ms,
        execution_duration=execution_duration,
        state=state,
        assessments=assessments or [],
    )


def make_trace(trace_id: str, timestamp_ms: int = 1000) -> Trace:
    return Trace(info=make_trace_info(trace_id, timestamp_ms), data=TraceData(spans=[]))


def make_online_assessment(scorer_name: str) -> Feedback:
    """An assessment as trace-level online scoring would have persisted it."""
    return Feedback(
        name=scorer_name,
        value=True,
        metadata={AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME: scorer_name},
    )


def eval_result(*feedbacks: Feedback):
    """Minimal stand-in for the EvalResult returned by _compute_eval_scores."""
    return SimpleNamespace(assessments=list(feedbacks))


def fetch_traces_for(trace_infos: list[TraceInfo]):
    """Build a fetch_traces side effect returning full traces for the given infos."""
    by_id = {t.trace_id: t for t in trace_infos}

    def _fetch(trace_ids):
        return [make_trace(tid, by_id[tid].timestamp_ms) for tid in trace_ids if tid in by_id]

    return _fetch


def incomplete_error_logged_for(mock_log, trace_id: str) -> list[Feedback]:
    """Error assessments recorded on a trace that online scoring gave up waiting for."""
    return [
        a
        for c in mock_log.call_args_list
        if c[1]["trace"].info.trace_id == trace_id
        for a in c[1]["assessments"]
        if a.error is not None and a.error.error_code == "TRACE_INCOMPLETE"
    ]


@pytest.fixture
def mock_trace_loader():
    loader = MagicMock(spec=OnlineTraceLoader)
    loader.fetch_trace_infos_in_range.return_value = []
    loader.fetch_traces.return_value = []
    return loader


@pytest.fixture
def mock_checkpoint_manager():
    manager = MagicMock(spec=OnlineTraceCheckpointManager)
    manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=1000, max_trace_timestamp_ms=2000
    )
    manager.get_checkpoint.return_value = None
    return manager


@pytest.fixture
def sampler_with_scorers():
    configs = [make_online_scorer(Completeness(), sample_rate=1.0)]
    return OnlineScorerSampler(configs)


@pytest.fixture
def empty_sampler():
    return OnlineScorerSampler([])


@pytest.fixture
def mock_compute():
    with patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock:
        mock.return_value = eval_result(Feedback(name="completeness", value=True))
        yield mock


@pytest.fixture
def mock_log():
    with patch("mlflow.genai.evaluation.harness._log_assessments") as mock:
        yield mock


def make_processor(loader, checkpoint_manager, sampler):
    return OnlineTraceScoringProcessor(
        trace_loader=loader,
        checkpoint_manager=checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
    )


def persisted_checkpoint(mock_checkpoint_manager) -> OnlineTraceScoringCheckpoint | None:
    if not mock_checkpoint_manager.persist_checkpoint.called:
        return None
    return mock_checkpoint_manager.persist_checkpoint.call_args[0][0]


def test_process_traces_skips_when_no_scorers(
    mock_trace_loader, mock_checkpoint_manager, empty_sampler
):
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, empty_sampler)

    processor.process_traces()

    # When there are no scorers there is nothing to discover, so the whole window is covered
    # and the checkpoint advances to the window end without fetching any traces.
    checkpoint = persisted_checkpoint(mock_checkpoint_manager)
    assert checkpoint == OnlineTraceScoringCheckpoint(timestamp_ms=2000, trace_id=None)
    mock_trace_loader.fetch_traces.assert_not_called()


def test_process_traces_updates_checkpoint_when_no_traces(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    checkpoint = persisted_checkpoint(mock_checkpoint_manager)
    assert checkpoint == OnlineTraceScoringCheckpoint(timestamp_ms=2000, trace_id=None)


def test_process_traces_passes_checkpoint_tiebreaker_to_loader(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    """
    The checkpoint trace ID is a tie-breaker inside the checkpoint millisecond. It is applied
    by the loader while paging (so a page of already-processed traces cannot stall discovery),
    and only when the window still starts at the checkpoint timestamp.
    """
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineTraceScoringCheckpoint(
        timestamp_ms=1000, trace_id="tr-002"
    )
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)
    processor.process_traces()
    assert mock_trace_loader.fetch_trace_infos_in_range.call_args[1]["after_trace_id"] == "tr-002"

    # Lookback moved the window start past the checkpoint: the tie-breaker no longer applies.
    mock_trace_loader.reset_mock()
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=1500, max_trace_timestamp_ms=2000
    )
    processor.process_traces()
    assert mock_trace_loader.fetch_trace_infos_in_range.call_args[1]["after_trace_id"] is None


def test_process_traces_groups_by_filter(mock_trace_loader, mock_checkpoint_manager):
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.env = 'staging'"),
    ]
    sampler = OnlineScorerSampler(configs)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    assert mock_trace_loader.fetch_trace_infos_in_range.call_count == 2
    call_args = [c[0] for c in mock_trace_loader.fetch_trace_infos_in_range.call_args_list]
    filters = [args[3] for args in call_args]
    assert any("tags.env = 'prod'" in f for f in filters)
    assert any("tags.env = 'staging'" in f for f in filters)


def test_process_traces_excludes_eval_run_traces(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    call_args = mock_trace_loader.fetch_trace_infos_in_range.call_args[0]
    filter_string = call_args[3]
    assert "metadata.mlflow.sourceRun IS NULL" in filter_string


def test_process_traces_samples_scores_and_marks_assessments(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    infos = [make_trace_info("tr-001", 1500)]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    mock_compute.assert_called_once()
    assert [s.name for s in mock_compute.call_args[1]["scorers"]] == ["completeness"]
    mock_log.assert_called_once()
    logged = mock_log.call_args[1]["assessments"]
    # Every persisted assessment carries the scorer name so a re-run can recognise it.
    assert logged[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME] == "completeness"
    # The whole window was covered and everything is durable: checkpoint = window end.
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=2000, trace_id=None
    )


def test_scorers_already_applied_to_a_trace_are_not_rerun(
    mock_trace_loader, mock_checkpoint_manager, mock_compute, mock_log
):
    """
    Re-running a window (after a crash, a job retry, or a held checkpoint) must not create a
    second visible assessment: scorers that already wrote one are skipped.
    """
    configs = [
        make_online_scorer(Completeness()),
        make_online_scorer(Completeness(name="c2")),
    ]
    sampler = OnlineScorerSampler(configs)
    infos = [
        make_trace_info("tr-001", 1500, assessments=[make_online_assessment("completeness")]),
        # An assessment with the same name but written by something other than online scoring
        # (e.g. a user, or an eval run) does not count.
        make_trace_info("tr-002", 1600, assessments=[Feedback(name="completeness", value=True)]),
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    ran = {
        (c[1]["eval_item"].trace.info.trace_id, c[1]["scorers"][0].name)
        for c in mock_compute.call_args_list
    }
    assert ran == {("tr-001", "c2"), ("tr-002", "completeness"), ("tr-002", "c2")}


def test_trace_with_all_scorers_already_applied_is_not_fetched(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    infos = [make_trace_info("tr-001", 1500, assessments=[make_online_assessment("completeness")])]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    mock_trace_loader.fetch_traces.assert_not_called()
    mock_compute.assert_not_called()
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=2000, trace_id=None
    )


def _error_feedback(name: str) -> Feedback:
    """What _compute_eval_scores returns when a scorer raises."""
    return Feedback(
        name=name,
        error=AssessmentError(error_code="SCORER_ERROR", error_message="judge timed out"),
    )


@pytest.mark.parametrize("failure_mode", ["raises", "error_feedback"])
def test_scorer_failure_holds_checkpoint_and_persists_nothing(
    mock_trace_loader,
    mock_checkpoint_manager,
    sampler_with_scorers,
    mock_compute,
    mock_log,
    failure_mode,
):
    """
    A judge failure (an exception, or an error feedback from the harness) is treated as
    transient: no assessment is written for that trace, and the checkpoint stops before it so
    the next run retries it. Other traces in the batch are still scored.
    """
    now_ms = int(time.time() * 1000)
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=now_ms - 10_000, max_trace_timestamp_ms=now_ms
    )
    infos = [make_trace_info("tr-001", now_ms - 5000), make_trace_info("tr-002", now_ms - 4000)]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    def compute(eval_item, scorers):
        if eval_item.trace.info.trace_id == "tr-001":
            if failure_mode == "raises":
                raise Exception("Scorer failed")
            return eval_result(_error_feedback("completeness"))
        return eval_result(Feedback(name="completeness", value=True))

    mock_compute.side_effect = compute
    processor.process_traces()

    assert mock_compute.call_count == 2
    # Only the successful trace was written; no error assessment for the one we will retry.
    assert [c[1]["trace"].info.trace_id for c in mock_log.call_args_list] == ["tr-002"]
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=now_ms - 5001, trace_id=None
    )


@pytest.mark.parametrize("failure_mode", ["raises", "error_feedback"])
def test_scorer_failure_is_recorded_once_retry_budget_is_exhausted(
    mock_trace_loader,
    mock_checkpoint_manager,
    sampler_with_scorers,
    mock_compute,
    mock_log,
    failure_mode,
):
    """A judge that keeps failing must not hold the pipeline forever: after the retry budget
    the error is persisted as an error feedback and the checkpoint moves on.
    """
    now_ms = int(time.time() * 1000)
    stale_ts = now_ms - MAX_SCORER_FAILURE_RETRY_MS - 60_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=stale_ts - 1000, max_trace_timestamp_ms=now_ms
    )
    infos = [make_trace_info("tr-001", stale_ts)]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    if failure_mode == "raises":
        mock_compute.side_effect = Exception("Scorer failed")
    else:
        mock_compute.return_value = eval_result(_error_feedback("completeness"))
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    logged = mock_log.call_args[1]["assessments"]
    assert len(logged) == 1
    assert logged[0].name == "completeness"
    assert logged[0].error is not None
    assert logged[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME] == "completeness"
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=now_ms, trace_id=None
    )


def test_checkpoint_is_held_before_trace_whose_assessment_could_not_be_persisted(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    """
    Partial batch success: the assessment for tr-002 could not be written (neither the result
    nor an error feedback). The checkpoint must stop before tr-002 so it is retried, while
    tr-001 and tr-003 are still scored in this run.
    """
    infos = [
        make_trace_info("tr-001", 1000),
        make_trace_info("tr-002", 1500),
        make_trace_info("tr-003", 1800),
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    def log(trace, assessments, run_id):
        if trace.info.trace_id == "tr-002":
            raise ConnectionError("store unavailable")

    mock_log.side_effect = log
    processor.process_traces()

    assert {c[1]["eval_item"].trace.info.trace_id for c in mock_compute.call_args_list} == {
        "tr-001",
        "tr-002",
        "tr-003",
    }
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=1499, trace_id=None
    )


def test_checkpoint_not_persisted_when_it_would_not_advance(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineTraceScoringCheckpoint(
        timestamp_ms=1000, trace_id="tr-000"
    )
    infos = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    mock_log.side_effect = ConnectionError("store unavailable")
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    # Stopping before tr-001 would mean (999, None), which is behind the existing checkpoint.
    mock_checkpoint_manager.persist_checkpoint.assert_not_called()


def test_checkpoint_does_not_jump_past_unfetched_traces_when_page_is_full(
    mock_trace_loader, mock_checkpoint_manager, monkeypatch, mock_compute, mock_log
):
    """
    Regression: with more matching traces than fit in one job and a low sample rate, a run can
    sample nothing from the page it fetched. The checkpoint must then advance only to the last
    fetched trace, not to the end of the window, or the traces beyond the page are never seen.
    """
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_TRACES_PER_JOB", 3)
    sampler = OnlineScorerSampler([make_online_scorer(Completeness(), sample_rate=1.0)])
    monkeypatch.setattr(sampler, "sample", lambda trace_id, scorers: [])  # nothing sampled
    infos = [
        make_trace_info("tr-001", 1100),
        make_trace_info("tr-002", 1200),
        make_trace_info("tr-003", 1300),
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    mock_compute.assert_not_called()
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=1300, trace_id="tr-003"
    )


def test_full_page_in_one_filter_group_bounds_checkpoint_for_all_groups(
    mock_trace_loader, mock_checkpoint_manager, monkeypatch, mock_compute, mock_log
):
    """
    Filter group A's page is full (there may be more A traces after 1300); group B returned
    later traces. B's traces beyond A's last row must be deferred, otherwise the checkpoint
    would move past A traces that were never fetched.
    """
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_TRACES_PER_JOB", 3)
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'a'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.env = 'b'"),
    ]
    sampler = OnlineScorerSampler(configs)
    group_a = [make_trace_info(f"tr-a{i}", 1100 + i * 100) for i in range(3)]  # full page
    group_b = [make_trace_info("tr-b1", 1900)]

    def fetch_infos(exp_id, min_ts, max_ts, filter_str, limit, after_trace_id=None, time_key=None):
        return group_a if "'a'" in filter_str else group_b

    mock_trace_loader.fetch_trace_infos_in_range.side_effect = fetch_infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(group_a + group_b)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    scored = {c[1]["eval_item"].trace.info.trace_id for c in mock_compute.call_args_list}
    assert scored == {"tr-a0", "tr-a1", "tr-a2"}  # tr-b1 (1900) deferred to the next run
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=1300, trace_id="tr-a2"
    )


def test_process_traces_truncates_and_sorts_across_filters(
    mock_trace_loader, mock_checkpoint_manager, mock_compute, mock_log
):
    """
    Test that when multiple filters return MAX_TRACES_PER_JOB traces each,
    we truncate to MAX_TRACES_PER_JOB total while preserving chronological order.
    This ensures we don't skip earlier traces from certain filters.

    Uses overlapping timestamp ranges to verify correct chronological sorting
    across filters when truncating.
    """
    from mlflow.genai.scorers.online.constants import MAX_TRACES_PER_JOB

    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.env = 'staging'"),
    ]
    sampler = OnlineScorerSampler(configs)

    # Staging: starts at 0, covers full MAX_TRACES_PER_JOB range (0 to MAX-1)
    # Prod: starts at 80% through staging range, creating 20% overlap
    staging_start = 0
    prod_start = int(0.8 * MAX_TRACES_PER_JOB)

    filter1_traces = [
        make_trace_info(f"tr-prod-{i}", prod_start + i) for i in range(MAX_TRACES_PER_JOB)
    ]
    filter2_traces = [
        make_trace_info(f"tr-staging-{i}", staging_start + i) for i in range(MAX_TRACES_PER_JOB)
    ]

    def mock_fetch_trace_infos(
        exp_id, min_ts, max_ts, filter_str, limit, after_trace_id=None, time_key=None
    ):
        if "tags.env = 'prod'" in filter_str:
            return filter1_traces
        elif "tags.env = 'staging'" in filter_str:
            return filter2_traces
        return []

    mock_trace_loader.fetch_trace_infos_in_range.side_effect = mock_fetch_trace_infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(filter1_traces + filter2_traces)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    fetched_trace_ids = mock_trace_loader.fetch_traces.call_args[0][0]
    assert len(fetched_trace_ids) == MAX_TRACES_PER_JOB

    # Expected order: staging-only traces first, then interleaved overlap region.
    staging_only_count = prod_start
    overlap_pairs = (MAX_TRACES_PER_JOB - staging_only_count) // 2
    expected_trace_ids = [f"tr-staging-{i}" for i in range(staging_only_count)]
    for i in range(overlap_pairs):
        expected_trace_ids.append(f"tr-prod-{i}")
        expected_trace_ids.append(f"tr-staging-{staging_only_count + i}")
    assert fetched_trace_ids == expected_trace_ids

    # Traces at timestamp prod_start + overlap_pairs were deferred by the per-job cap, so the
    # checkpoint stops just before them.
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=prod_start + overlap_pairs - 1, trace_id=None
    )


def test_process_traces_deduplicates_scorers_across_filters(
    mock_trace_loader, mock_checkpoint_manager, mock_compute, mock_log
):
    """
    Test that when a trace matches multiple filters that would select the same scorer,
    the scorer only runs once on that trace.
    """
    completeness_scorer = Completeness()
    configs = [
        make_online_scorer(completeness_scorer, filter_string="tags.env = 'prod'"),
        make_online_scorer(completeness_scorer, filter_string="tags.priority = 'high'"),
    ]
    sampler = OnlineScorerSampler(configs)
    shared_trace = make_trace_info("tr-001", 1500)
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [shared_trace]
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for([shared_trace])
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    assert mock_compute.call_count == 1
    scorers_used = mock_compute.call_args[1]["scorers"]
    assert [s.name for s in scorers_used] == [completeness_scorer.name]


def test_in_progress_trace_holds_checkpoint_until_it_completes(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    """
    A trace that is still running is visible to search (incremental span export) but cannot
    be scored yet. It must not be scored on partial data, and the checkpoint must stop before
    it so the next run picks it up once it has completed.
    """
    now_ms = int(time.time() * 1000)
    infos = [
        make_trace_info("tr-done", now_ms - 5000),
        make_trace_info("tr-running", now_ms - 4000, state=TraceState.IN_PROGRESS),
        make_trace_info("tr-later", now_ms - 3000),
    ]
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=now_ms - 10_000, max_trace_timestamp_ms=now_ms
    )
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    assert mock_trace_loader.fetch_traces.call_args[0][0] == ["tr-done", "tr-later"]
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=now_ms - 4001, trace_id=None
    )


def test_stale_in_progress_trace_is_given_up_on(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    """A trace still in progress long after it started must not stall scoring forever."""
    now_ms = int(time.time() * 1000)
    stale_ts = now_ms - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    infos = [
        make_trace_info("tr-stuck", stale_ts, state=TraceState.IN_PROGRESS),
        make_trace_info("tr-later", stale_ts + 1000),
    ]
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=stale_ts - 1000, max_trace_timestamp_ms=now_ms
    )
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    assert mock_trace_loader.fetch_traces.call_args[0][0] == ["tr-later"]
    # Giving up is recorded on the trace, not just in a log line.
    recorded = incomplete_error_logged_for(mock_log, "tr-stuck")
    assert [a.name for a in recorded] == ["completeness"]
    assert recorded[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME] == "completeness"
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=now_ms, trace_id=None
    )


def test_stale_in_progress_trace_is_not_dropped_if_the_error_cannot_be_recorded(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    now_ms = int(time.time() * 1000)
    stale_ts = now_ms - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=stale_ts - 1000, max_trace_timestamp_ms=now_ms
    )
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-stuck", stale_ts, state=TraceState.IN_PROGRESS)
    ]
    mock_log.side_effect = ConnectionError("store unavailable")
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    # Nothing durable exists for tr-stuck, so the checkpoint must not pass it.
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=stale_ts - 1, trace_id=None
    )


def test_unloadable_trace_is_retried_while_recent_and_skipped_once_stale(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    """
    fetch_traces returns nothing for a trace whose info was found (spans not yet exported, or
    deleted in between). A recent one holds the checkpoint; an old one is skipped.
    """
    now_ms = int(time.time() * 1000)
    recent_ts = now_ms - 5000
    stale_ts = now_ms - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=stale_ts - 1000, max_trace_timestamp_ms=now_ms
    )
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-stale-missing", stale_ts),
        make_trace_info("tr-recent-missing", recent_ts),
    ]
    mock_trace_loader.fetch_traces.return_value = []
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    mock_compute.assert_not_called()
    assert [a.name for a in incomplete_error_logged_for(mock_log, "tr-stale-missing")] == [
        "completeness"
    ]
    assert incomplete_error_logged_for(mock_log, "tr-recent-missing") == []
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=recent_ts - 1, trace_id=None
    )


def test_per_job_cap_defers_later_traces(
    mock_trace_loader, mock_checkpoint_manager, monkeypatch, mock_compute, mock_log
):
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_TRACES_PER_JOB", 2)
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'a'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.env = 'b'"),
    ]
    sampler = OnlineScorerSampler(configs)
    group_a = [make_trace_info("tr-a", 1100), make_trace_info("tr-c", 1300)]  # full page
    group_b = [make_trace_info("tr-b", 1200)]

    def fetch_infos(exp_id, min_ts, max_ts, filter_str, limit, after_trace_id=None, time_key=None):
        return group_a if "'a'" in filter_str else group_b

    mock_trace_loader.fetch_trace_infos_in_range.side_effect = fetch_infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(group_a + group_b)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    # Group A's page (2 rows) is full, so the watermark is (1300, tr-c); the cap of 2 then
    # defers tr-c itself, so the checkpoint stops before it.
    scored = {c[1]["eval_item"].trace.info.trace_id for c in mock_compute.call_args_list}
    assert scored == {"tr-a", "tr-b"}
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=1299, trace_id=None
    )


def test_scoring_runs_one_unit_per_trace_and_scorer_in_parallel(
    mock_trace_loader, mock_checkpoint_manager, mock_compute, mock_log, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS", "4")
    configs = [make_online_scorer(Completeness()), make_online_scorer(Completeness(name="c2"))]
    sampler = OnlineScorerSampler(configs)
    infos = [make_trace_info("tr-001", 1100), make_trace_info("tr-002", 1200)]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)
    threads = set()

    def compute(eval_item, scorers):
        threads.add(threading.current_thread().name)
        return eval_result(Feedback(name=scorers[0].name, value=True))

    mock_compute.side_effect = compute
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler)

    processor.process_traces()

    assert mock_compute.call_count == 4
    assert all(name.startswith("OnlineScoring") for name in threads)


def test_create_factory_method():
    mock_store = MagicMock()
    configs = [make_online_scorer(Completeness())]

    processor = OnlineTraceScoringProcessor.create(
        experiment_id="exp1",
        online_scorers=configs,
        tracking_store=mock_store,
    )

    assert processor._experiment_id == "exp1"
    assert isinstance(processor._trace_loader, OnlineTraceLoader)
    assert isinstance(processor._checkpoint_manager, OnlineTraceCheckpointManager)
    assert isinstance(processor._sampler, OnlineScorerSampler)


def test_discovery_and_checkpoint_are_keyed_on_trace_end_time(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    """
    Traces are written when they finish, so the pipeline discovers and checkpoints them by end
    time (start + duration). Two recent traces whose data cannot be loaded yet are held; the
    checkpoint stops before the earliest *end* time, not the earliest start time, so a trace
    that ran for a long time is never left behind the checkpoint.
    """
    now_ms = int(time.time() * 1000)
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=now_ms - 10_000, max_trace_timestamp_ms=now_ms
    )
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        # Started 9s ago and ran for 7.5s: ended 1.5s ago.
        make_trace_info("tr-long", now_ms - 9_000, execution_duration=7_500),
        # Started 1.4s ago and ran for 10ms.
        make_trace_info("tr-short", now_ms - 1_400, execution_duration=10),
    ]
    mock_trace_loader.fetch_traces.return_value = []
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    assert mock_trace_loader.fetch_trace_infos_in_range.call_args[1]["time_key"] == "end_time_ms"
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=now_ms - 1_501, trace_id=None
    )


def test_completed_trace_replaces_its_incomplete_placeholder(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    """
    A trace that online scoring gave up on (TRACE_INCOMPLETE recorded) later completes and is
    re-discovered by its new end time. The placeholder must not block scoring, and once the real
    score is persisted the placeholder is removed so the user sees one, correct assessment.
    """
    placeholder = Feedback(
        name="completeness",
        error=AssessmentError(error_code="TRACE_INCOMPLETE", error_message="gave up"),
        metadata={AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME: "completeness"},
    )
    placeholder.assessment_id = "a-placeholder"
    info = make_trace_info("tr-late", 1500, assessments=[placeholder])
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [info]
    mock_trace_loader.fetch_traces.return_value = [Trace(info=info, data=TraceData(spans=[]))]
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    with patch("mlflow.genai.scorers.online.trace_processor.mlflow.delete_assessment") as delete:
        processor.process_traces()

    mock_compute.assert_called_once()
    assert mock_log.call_args[1]["assessments"][0].error is None
    delete.assert_called_once_with(trace_id="tr-late", assessment_id="a-placeholder")
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=2000, trace_id=None
    )


def test_giving_up_twice_does_not_record_a_second_placeholder(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_compute, mock_log
):
    now_ms = int(time.time() * 1000)
    stale_ts = now_ms - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    placeholder = Feedback(
        name="completeness",
        error=AssessmentError(error_code="TRACE_INCOMPLETE", error_message="gave up"),
        metadata={AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME: "completeness"},
    )
    placeholder.assessment_id = "a-placeholder"
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=stale_ts - 1000, max_trace_timestamp_ms=now_ms
    )
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info(
            "tr-stuck", stale_ts, state=TraceState.IN_PROGRESS, assessments=[placeholder]
        )
    ]
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    mock_log.assert_not_called()
    assert persisted_checkpoint(mock_checkpoint_manager) == OnlineTraceScoringCheckpoint(
        timestamp_ms=now_ms, trace_id=None
    )


def _paged_loader(mock_trace_loader, infos, page):
    """Make the mock loader honour the checkpoint the way the real store does."""

    def fetch_infos(exp_id, min_ts, max_ts, filter_str, limit, after_trace_id=None, time_key=None):
        eligible = [
            t
            for t in infos
            if min_ts <= t.timestamp_ms <= max_ts
            and not (
                after_trace_id is not None
                and t.timestamp_ms == min_ts
                and t.trace_id <= after_trace_id
            )
        ]
        return eligible[:page]

    mock_trace_loader.fetch_trace_infos_in_range.side_effect = fetch_infos
    mock_trace_loader.fetch_traces.side_effect = fetch_traces_for(infos)


def _live_checkpoint_manager(mock_checkpoint_manager, window_min, window_max):
    """Make the mock checkpoint manager reflect what was persisted, like the real one."""
    state = {"cp": None}

    def persist(cp):
        state["cp"] = cp

    def window():
        lo = window_min if state["cp"] is None else max(window_min, state["cp"].timestamp_ms)
        return OnlineTraceScoringTimeWindow(
            min_trace_timestamp_ms=lo, max_trace_timestamp_ms=window_max
        )

    mock_checkpoint_manager.persist_checkpoint.side_effect = persist
    mock_checkpoint_manager.get_checkpoint.side_effect = lambda: state["cp"]
    mock_checkpoint_manager.calculate_time_window.side_effect = window
    return state


def test_one_job_drains_a_backlog_larger_than_one_batch(
    mock_trace_loader,
    mock_checkpoint_manager,
    sampler_with_scorers,
    mock_compute,
    mock_log,
    monkeypatch,
):
    """
    With more eligible traces than fit in a batch, a job must keep going rather than leave the
    rest for the next scheduler tick; otherwise a sustained backlog would age out of the
    lookback window before it was reached.
    """
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_TRACES_PER_JOB", 2)
    infos = [make_trace_info(f"tr-{i}", 1000 + i) for i in range(5)]
    _paged_loader(mock_trace_loader, infos, page=2)
    _live_checkpoint_manager(mock_checkpoint_manager, 1000, 2000)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    scored = [c[1]["eval_item"].trace.info.trace_id for c in mock_compute.call_args_list]
    assert sorted(scored) == [f"tr-{i}" for i in range(5)]
    assert len(scored) == 5  # nothing scored twice
    checkpoints = [c[0][0] for c in mock_checkpoint_manager.persist_checkpoint.call_args_list]
    assert checkpoints == [
        OnlineTraceScoringCheckpoint(timestamp_ms=1001, trace_id="tr-1"),
        OnlineTraceScoringCheckpoint(timestamp_ms=1003, trace_id="tr-3"),
        OnlineTraceScoringCheckpoint(timestamp_ms=2000, trace_id=None),
    ]


def test_job_stops_when_the_checkpoint_cannot_move(
    mock_trace_loader,
    mock_checkpoint_manager,
    sampler_with_scorers,
    mock_compute,
    mock_log,
    monkeypatch,
):
    """A full page whose first trace is held (write failed) must not spin the job."""
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_TRACES_PER_JOB", 2)
    infos = [make_trace_info(f"tr-{i}", 1000 + i) for i in range(5)]
    _paged_loader(mock_trace_loader, infos, page=2)
    _live_checkpoint_manager(mock_checkpoint_manager, 1000, 2000)
    mock_log.side_effect = ConnectionError("store unavailable")
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    # The first batch holds before tr-0 at (999, None). The second batch recomputes the same
    # key, which is not progress, so the job stops instead of spinning until its runtime budget.
    assert mock_trace_loader.fetch_trace_infos_in_range.call_count == 2
    persisted = [c[0][0] for c in mock_checkpoint_manager.persist_checkpoint.call_args_list]
    assert persisted == [OnlineTraceScoringCheckpoint(timestamp_ms=999, trace_id=None)]


def test_job_stops_when_persisted_checkpoint_is_not_reflected(
    mock_trace_loader,
    mock_checkpoint_manager,
    sampler_with_scorers,
    mock_compute,
    mock_log,
    monkeypatch,
):
    """If the store keeps returning the old checkpoint after a write, each batch would
    recompute the same one; the job must notice it made no progress and stop.
    """
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_TRACES_PER_JOB", 2)
    infos = [make_trace_info(f"tr-{i}", 1000 + i) for i in range(5)]
    _paged_loader(mock_trace_loader, infos, page=2)
    # Default fixture: get_checkpoint always None, window fixed -> same batch every time.
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    assert mock_trace_loader.fetch_trace_infos_in_range.call_count == 2
    persisted = [c[0][0] for c in mock_checkpoint_manager.persist_checkpoint.call_args_list]
    assert persisted == [OnlineTraceScoringCheckpoint(timestamp_ms=1001, trace_id="tr-1")] * 2


def test_job_stops_at_runtime_budget_and_leaves_the_rest_for_the_next_job(
    mock_trace_loader,
    mock_checkpoint_manager,
    sampler_with_scorers,
    mock_compute,
    mock_log,
    monkeypatch,
):
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_TRACES_PER_JOB", 2)
    monkeypatch.setattr(f"{_PROCESSOR_MODULE}.MAX_JOB_RUNTIME_MS", 0)
    infos = [make_trace_info(f"tr-{i}", 1000 + i) for i in range(5)]
    _paged_loader(mock_trace_loader, infos, page=2)
    _live_checkpoint_manager(mock_checkpoint_manager, 1000, 2000)
    processor = make_processor(mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers)

    processor.process_traces()

    assert mock_trace_loader.fetch_trace_infos_in_range.call_count == 1
    assert mock_checkpoint_manager.persist_checkpoint.call_args[0][
        0
    ] == OnlineTraceScoringCheckpoint(timestamp_ms=1001, trace_id="tr-1")
