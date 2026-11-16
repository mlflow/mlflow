import json
import time
import uuid
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
from mlflow.genai.evaluation.entities import EvalItem, EvalResult
from mlflow.genai.scorers.builtin_scorers import ConversationCompleteness
from mlflow.genai.scorers.online.constants import (
    MAX_SCORER_FAILURE_RETRY_MS,
    MAX_TRACE_COMPLETION_WAIT_MS,
)
from mlflow.genai.scorers.online.entities import CompletedSession, OnlineScorer, OnlineScoringConfig
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.session_checkpointer import (
    OnlineSessionCheckpointManager,
    OnlineSessionScoringCheckpoint,
    OnlineSessionScoringTimeWindow,
)
from mlflow.genai.scorers.online.session_processor import OnlineSessionScoringProcessor
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.tracing.constant import AssessmentMetadataKey


def make_online_scorer(scorer, sample_rate: float = 1.0, filter_string: str | None = None):
    config = OnlineScoringConfig(
        online_scoring_config_id=uuid.uuid4().hex,
        scorer_id=uuid.uuid4().hex,
        sample_rate=sample_rate,
        experiment_id="exp1",
        filter_string=filter_string,
    )
    return OnlineScorer(
        name=scorer.name, serialized_scorer=json.dumps(scorer.model_dump()), online_config=config
    )


def make_completed_session(
    session_id: str, first_trace_timestamp_ms: int, last_trace_timestamp_ms: int
):
    return CompletedSession(
        session_id=session_id,
        first_trace_timestamp_ms=first_trace_timestamp_ms,
        last_trace_timestamp_ms=last_trace_timestamp_ms,
    )


def make_trace_info(trace_id: str, timestamp_ms: int = 1000, state: TraceState = TraceState.OK):
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="exp1"),
        ),
        request_time=timestamp_ms,
        state=state,
    )


def make_trace(trace_id: str, timestamp_ms: int = 1000, assessments=None):
    return Trace(
        info=TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation(
                type=TraceLocationType.MLFLOW_EXPERIMENT,
                mlflow_experiment=MlflowExperimentLocation(experiment_id="exp1"),
            ),
            request_time=timestamp_ms,
            state=TraceState.OK,
            assessments=assessments or [],
        ),
        data=TraceData(spans=[]),
    )


def make_processor(loader, checkpoint_mgr, sampler, store):
    return OnlineSessionScoringProcessor(
        trace_loader=loader,
        checkpoint_manager=checkpoint_mgr,
        sampler=sampler,
        experiment_id="exp1",
        tracking_store=store,
    )


def make_assessment(assessment_id: str, name: str, session_id: str | None = None):
    assessment = MagicMock(spec=Assessment)
    assessment.assessment_id = assessment_id
    assessment.name = name
    assessment.error = None
    assessment.metadata = (
        {AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID: session_id} if session_id else {}
    )
    return assessment


@pytest.fixture
def mock_trace_loader():
    return MagicMock(spec=OnlineTraceLoader)


@pytest.fixture
def mock_checkpoint_manager():
    manager = MagicMock(spec=OnlineSessionCheckpointManager)
    manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=1000, max_last_trace_timestamp_ms=2000
    )
    manager.get_checkpoint.return_value = None
    return manager


@pytest.fixture
def mock_tracking_store():
    return MagicMock()


@pytest.fixture
def sampler_with_scorers():
    return OnlineScorerSampler([make_online_scorer(ConversationCompleteness(), sample_rate=1.0)])


@pytest.fixture
def empty_sampler():
    return OnlineScorerSampler([])


@pytest.fixture
def mock_evaluate():
    with patch("mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers") as mock:
        yield mock


@pytest.fixture
def mock_log_assessments():
    with patch("mlflow.genai.evaluation.harness._log_assessments") as mock:
        yield mock


@pytest.fixture
def mock_score_session():
    with patch(
        "mlflow.genai.scorers.online.session_processor.OnlineSessionScoringProcessor._score_session"
    ) as mock:
        yield mock


def test_process_sessions_skips_when_no_scorers(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, empty_sampler
):
    """
    Scenario: No online scorers are configured.

    When there are no scorers, the processor should skip all processing work
    (no time window calculation, no session fetching) to avoid unnecessary overhead.
    """
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, empty_sampler, mock_tracking_store
    )
    processor.process_sessions()

    mock_checkpoint_manager.calculate_time_window.assert_not_called()
    mock_tracking_store.find_completed_sessions.assert_not_called()


def test_process_sessions_updates_checkpoint_when_no_sessions(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    """
    Scenario: Scorers are configured but no completed sessions are found in the time window.

    The processor should advance the checkpoint to the end of the time window
    (max_last_trace_timestamp_ms) to avoid reprocessing the same empty window on the next run.
    """
    mock_tracking_store.find_completed_sessions.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 2000
    assert checkpoint.session_id is None


def test_process_sessions_filters_checkpoint_boundary(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
):
    """
    Scenario: The previous run stopped at (timestamp=1000, session_id=sess-002). Sessions at
    that timestamp with an id at or before sess-002 were already scored.

    The tie-break is delegated to the store (``after_session_id``) so that it is evaluated
    with the same collation as the store's ordering, and applied before the page limit. The
    processor scores whatever the store returns without re-filtering in Python.
    """
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineSessionScoringCheckpoint(
        timestamp_ms=1000, session_id="sess-002"
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-003", 500, 1000),
        make_completed_session("sess-004", 500, 1500),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    call_kwargs = mock_tracking_store.find_completed_sessions.call_args[1]
    assert call_kwargs["min_last_trace_timestamp_ms"] == 1000
    assert call_kwargs["after_session_id"] == "sess-002"
    assert mock_score_session.call_count == 2
    scored_tasks = [call[0][0] for call in mock_score_session.call_args_list]
    assert scored_tasks[0].session.session_id == "sess-003"
    assert scored_tasks[1].session.session_id == "sess-004"


def test_checkpoint_tiebreaker_not_used_when_lookback_moved_window_start(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
):
    """
    Scenario: The checkpoint is older than the maximum lookback, so the window now starts
    later than the checkpoint timestamp. The checkpoint session_id is a tie-breaker for the
    checkpoint timestamp only and must not be applied to the new window start.
    """
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineSessionScoringCheckpoint(
        timestamp_ms=500, session_id="sess-old"
    )
    mock_tracking_store.find_completed_sessions.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    assert mock_tracking_store.find_completed_sessions.call_args[1]["after_session_id"] is None


def test_session_rescored_when_new_trace_added_after_checkpoint(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
):
    """
    Scenario: A session (sess-001) was previously scored when its last trace was at
    timestamp 1000, but a new trace (tr-002) was added at timestamp 2000, making the
    session "complete" again with a new last_trace_timestamp.

    The processor should re-score the session with all traces (including the new one)
    and update old assessments with new scores that incorporate the additional trace.
    """
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineSessionScoringCheckpoint(
        timestamp_ms=1000, session_id="sess-001"
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 2000)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", 500),
        make_trace_info("tr-002", 2000),
    ]
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-001", 500),
        make_trace("tr-002", 2000),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(make_trace("tr-001", 500)),
        assessments=[
            make_assessment("assess-1", "ConversationCompleteness/v1"),
            make_assessment("assess-2", "ConversationCompleteness/v1"),
        ],
        scorer_stats={},
    )
    processor.process_sessions()

    call_kwargs = mock_evaluate.call_args[1]
    assert call_kwargs["session_id"] == "sess-001"
    assert len(call_kwargs["session_items"]) == 2
    assert len(call_kwargs["multi_turn_scorers"]) == 1


def test_process_sessions_samples_and_scores(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: A completed session is found with traces, and sampling selects scorers to run.

    The processor should evaluate the session with the selected scorers and log the
    resulting assessments to the trace. This is the happy path for session scoring.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 1000)]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(make_trace("tr-001", 1000)),
        assessments=[MagicMock()],
        scorer_stats={},
    )
    processor.process_sessions()

    call_kwargs = mock_evaluate.call_args[1]
    assert call_kwargs["session_id"] == "sess-001"
    assert len(call_kwargs["session_items"]) == 1
    assert len(call_kwargs["multi_turn_scorers"]) == 1
    mock_log_assessments.assert_called_once()


def test_process_sessions_updates_checkpoint_on_success(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
):
    """
    Scenario: Two sessions are successfully scored (sess-001 at timestamp 1000,
    sess-002 at timestamp 1500) and the store returned fewer sessions than the page limit.

    Every completed session in the window was therefore seen and durably processed, so the
    checkpoint advances to the end of the window.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    assert mock_score_session.call_count == 2
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=2000, session_id=None)


def test_checkpoint_does_not_jump_past_unfetched_sessions_when_page_is_full(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
    monkeypatch,
):
    """
    Scenario: More completed sessions exist than fit in one job. The store returned a full
    page; there may be more sessions after the last one returned.

    The checkpoint must advance exactly to the last fetched session, never to the end of the
    window, even if sampling selected nothing from the page.
    """
    monkeypatch.setattr("mlflow.genai.scorers.online.session_processor.MAX_SESSIONS_PER_JOB", 2)
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1200),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=1200, session_id="sess-002")

    # Same, but nothing is sampled: the checkpoint must still stop at the page boundary.
    mock_checkpoint_manager.reset_mock()
    monkeypatch.setattr(sampler_with_scorers, "sample", lambda entity_id, scorers: [])
    processor.process_sessions()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=1200, session_id="sess-002")


def test_full_page_in_one_filter_group_defers_later_sessions_from_other_groups(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    mock_score_session,
    monkeypatch,
):
    """
    Scenario: Scorer A's filter returned a full page ending at timestamp 1200; scorer B's
    filter returned a session at 1800. There may be unfetched A sessions between 1200 and
    1800, so B's session must wait for the next run and the checkpoint stops at A's last row.
    """
    monkeypatch.setattr("mlflow.genai.scorers.online.session_processor.MAX_SESSIONS_PER_JOB", 2)
    scorer_a, scorer_b = ConversationCompleteness(), ConversationCompleteness()
    scorer_a.name, scorer_b.name = "scorer_a", "scorer_b"
    sampler = OnlineScorerSampler([
        make_online_scorer(scorer_a, filter_string="tags.env = 'a'"),
        make_online_scorer(scorer_b, filter_string="tags.env = 'b'"),
    ])

    def find_sessions(**kwargs):
        if kwargs["filter_string"] == "tags.env = 'a'":
            return [
                make_completed_session("sess-a1", 500, 1000),
                make_completed_session("sess-a2", 500, 1200),
            ]
        return [make_completed_session("sess-b1", 500, 1800)]

    mock_tracking_store.find_completed_sessions.side_effect = find_sessions
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler, mock_tracking_store
    )
    processor.process_sessions()

    scored = {call[0][0].session.session_id for call in mock_score_session.call_args_list}
    assert scored == {"sess-a1", "sess-a2"}
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=1200, session_id="sess-a2")


def test_execute_session_scoring_handles_failures(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
):
    """
    Scenario: Two sessions need scoring, but the first one (sess-001) fails in a way that
    leaves nothing durable behind (e.g. the store rejected the writes) while the second one
    (sess-002) succeeds.

    The processor should continue processing sess-002 in this run, but the checkpoint must
    stop just before sess-001 so it is retried on the next run: the checkpoint never moves
    past a session that was not durably assessed.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )

    def score(task):
        if task.session.session_id == "sess-001":
            raise Exception("Session failed")

    mock_score_session.side_effect = score
    processor.process_sessions()

    assert mock_score_session.call_count == 2
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=999, session_id=None)


def test_checkpoint_not_persisted_when_failed_session_is_at_checkpoint_timestamp(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
):
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineSessionScoringCheckpoint(
        timestamp_ms=1000, session_id="sess-000"
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
    ]
    mock_score_session.side_effect = Exception("Session failed")
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    # Stopping before sess-001 would be (999, None), behind the current checkpoint.
    mock_checkpoint_manager.persist_checkpoint.assert_not_called()


def test_score_session_logs_assessments_to_first_trace(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: A session has multiple traces. The scorer evaluates the session and
    produces assessments. Session-level assessments are logged to the first (chronologically
    earliest) trace in the session.

    The processor should log all assessments to the first trace (tr-001). If logging fails
    nothing durable exists for this session, so it must log a warning with trace_id and
    session_id and hold the checkpoint before the session so it is retried.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", 1000),
        make_trace_info("tr-002", 1200),
    ]
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-001", 1000),
        make_trace("tr-002", 1200),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(make_trace("tr-001", 1000)),
        assessments=[
            make_assessment("assess-1", "ConversationCompleteness/v1"),
            make_assessment("assess-2", "ConversationCompleteness/v1"),
        ],
        scorer_stats={},
    )
    # Logging fails
    mock_log_assessments.side_effect = [Exception("Failed to log")]

    with patch("mlflow.genai.scorers.online.session_processor._logger") as mock_logger:
        processor.process_sessions()

        # Assessments should be logged once (to the first trace)
        assert mock_log_assessments.call_count == 1
        assert mock_log_assessments.call_args[1]["trace"].info.trace_id == "tr-001"

        # Verify a warning was logged with trace_id and session_id
        warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("tr-001" in msg and "sess-001" in msg for msg in warning_messages)

        # Nothing durable was written for sess-001, so the checkpoint stops before it.
        checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
        assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=1499, session_id=None)


def test_create_factory_method(mock_tracking_store):
    """
    Scenario: Creating a processor using the factory method instead of direct instantiation.

    The factory method should properly initialize all dependencies (trace loader,
    checkpoint manager, sampler) with the correct configuration.
    """
    configs = [make_online_scorer(ConversationCompleteness())]
    processor = OnlineSessionScoringProcessor.create(
        experiment_id="exp1", online_scorers=configs, tracking_store=mock_tracking_store
    )

    assert processor._experiment_id == "exp1"
    assert isinstance(processor._trace_loader, OnlineTraceLoader)
    assert isinstance(processor._checkpoint_manager, OnlineSessionCheckpointManager)
    assert isinstance(processor._sampler, OnlineScorerSampler)
    assert processor._tracking_store == mock_tracking_store
    assert processor._checkpoint_manager._tracking_store == mock_tracking_store
    assert processor._checkpoint_manager._experiment_id == "exp1"
    assert processor._sampler._online_scorers == configs


def test_score_session_excludes_eval_run_traces(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    """
    Scenario: A session exists, but we need to ensure eval-generated traces are filtered out.

    The processor should apply the "metadata.mlflow.sourceRun IS NULL" filter when
    fetching traces to exclude traces generated during MLflow evaluation runs
    (which already have assessments and shouldn't be scored again).
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    filter_string = mock_trace_loader.fetch_trace_infos_in_range.call_args[1]["filter_string"]
    assert (
        filter_string
        == "metadata.mlflow.sourceRun IS NULL AND metadata.`mlflow.trace.session` = 'sess-001'"
    )


def test_score_session_adds_session_metadata_to_assessments(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: Session scoring produces assessments that need to be logged.

    The processor should add session metadata (ONLINE_SCORING_SESSION_ID) to each
    assessment before logging it, enabling cleanup of old assessments when the session
    is re-scored later.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 1000)]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    feedback = make_assessment("new-id", "ConversationCompleteness/v1")
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(make_trace("tr-001", 1000)),
        assessments=[feedback],
        scorer_stats={},
    )
    processor.process_sessions()

    logged_feedbacks = mock_log_assessments.call_args[1]["assessments"]
    assert (
        logged_feedbacks[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID] == "sess-001"
    )


def test_score_session_skips_when_no_traces_found(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
):
    """
    Scenario: A session is marked as completed, but no traces are found for it
    (possibly all traces were filtered out as eval-generated traces).

    The processor should skip scoring this session since there's no data to evaluate,
    logging a warning and moving on to process other sessions.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    mock_evaluate.assert_not_called()


def test_score_session_skips_when_no_applicable_scorers(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, mock_evaluate
):
    """
    Scenario: A session exists with traces, but sampling excludes all scorers
    (e.g., all scorers have sample_rate=0.0 for this session).

    The processor should skip scoring since no scorers were selected to run,
    avoiding unnecessary evaluation work.
    """
    sampler = OnlineScorerSampler([make_online_scorer(ConversationCompleteness(), sample_rate=0.0)])
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 1000)]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler, mock_tracking_store
    )
    processor.process_sessions()

    mock_evaluate.assert_not_called()


def test_checkpoint_advances_when_all_traces_are_from_eval_runs(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    """
    Scenario: Two sessions exist, but all their traces are from evaluation runs
    (filtered out by the eval run exclusion filter).

    Sessions with nothing scorable are not eligible, so the processor should still advance
    the checkpoint (to the window end, since the page was not full) even though no actual
    scoring occurred, preventing the processor from getting stuck on eval-only sessions.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=2000, session_id=None)


def test_clean_up_old_assessments_removes_duplicates(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: A session is re-scored, and there's an old assessment from a previous
    scoring run for the same session/scorer combination.

    The processor should delete the old assessment (old-id) after logging the new one,
    preventing accumulation of duplicate assessments when sessions are re-scored
    (e.g., when new traces are added).
    """
    old_assessment = make_assessment("old-id", "ConversationCompleteness/v1", session_id="sess-001")
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    trace_with_old_assessment = make_trace("tr-001", 1000, assessments=[old_assessment])
    mock_trace_loader.fetch_traces.return_value = [trace_with_old_assessment]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    new_assessment = make_assessment("new-id", "ConversationCompleteness/v1")
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(trace_with_old_assessment),
        assessments=[new_assessment],
        scorer_stats={},
    )
    order = []
    mock_tracking_store.delete_assessment.side_effect = lambda **kw: order.append("delete")
    mock_log_assessments.side_effect = lambda **kw: order.append("log")
    processor.process_sessions()

    mock_tracking_store.delete_assessment.assert_called_once_with(
        trace_id="tr-001", assessment_id="old-id"
    )
    # The stale assessment is removed *before* the new one is logged, so a failure in between
    # leads to a retry (re-log) rather than two visible assessments for the same scorer.
    assert order == ["delete", "log"]


def test_failed_cleanup_of_old_assessments_holds_checkpoint_for_retry(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    old_assessment = make_assessment("old-id", "ConversationCompleteness/v1", session_id="sess-001")
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    trace_with_old_assessment = make_trace("tr-001", 1000, assessments=[old_assessment])
    mock_trace_loader.fetch_traces.return_value = [trace_with_old_assessment]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(trace_with_old_assessment),
        assessments=[make_assessment("new-id", "ConversationCompleteness/v1")],
        scorer_stats={},
    )
    mock_tracking_store.delete_assessment.side_effect = ConnectionError("store unavailable")
    processor.process_sessions()

    mock_log_assessments.assert_not_called()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=1499, session_id=None)


def test_session_with_recent_in_progress_trace_is_deferred(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
):
    """
    Scenario: A session looks complete by trace start timestamps, but one of its traces is
    still in progress (its spans are incomplete). Scoring now would evaluate partial data and
    the session would never be revisited, so the session is deferred and the checkpoint held.
    """
    now_ms = int(time.time() * 1000)
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=now_ms - 20_000, max_last_trace_timestamp_ms=now_ms
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", now_ms - 10_000, now_ms - 5_000)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", now_ms - 10_000),
        make_trace_info("tr-002", now_ms - 5_000, state=TraceState.IN_PROGRESS),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    mock_evaluate.assert_not_called()
    mock_trace_loader.fetch_traces.assert_not_called()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(
        timestamp_ms=now_ms - 5_001, session_id=None
    )


def test_stale_in_progress_trace_is_excluded_from_session_scoring(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """A trace still in progress long after it started is dropped rather than waited for."""
    now_ms = int(time.time() * 1000)
    stale_ts = now_ms - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=stale_ts - 20_000, max_last_trace_timestamp_ms=now_ms
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", stale_ts - 10_000, stale_ts)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", stale_ts - 10_000),
        make_trace_info("tr-stuck", stale_ts, state=TraceState.IN_PROGRESS),
    ]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", stale_ts - 10_000)]
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(make_trace("tr-001", stale_ts - 10_000)),
        assessments=[make_assessment("new-id", "ConversationCompleteness/v1")],
        scorer_stats={},
    )
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    mock_trace_loader.fetch_traces.assert_called_once_with(["tr-001"])
    assert len(mock_evaluate.call_args[1]["session_items"]) == 1
    # The excluded trace is not dropped silently: the exclusion is recorded on it.
    recorded = [
        a
        for c in mock_log_assessments.call_args_list
        if c[1]["trace"].info.trace_id == "tr-stuck"
        for a in c[1]["assessments"]
    ]
    assert [a.name for a in recorded] == ["conversation_completeness"]
    assert recorded[0].error.error_code == "TRACE_INCOMPLETE"
    assert recorded[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID] == "sess-001"
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=now_ms, session_id=None)


def test_session_whose_trace_data_cannot_be_loaded_is_held_then_recorded(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: the session's trace infos exist but none of the span data can be loaded (still
    being exported, or archived/deleted). Recent sessions are deferred; once the session is old
    enough, an error assessment is recorded on its first trace so it is not dropped silently.
    """
    now_ms = int(time.time() * 1000)
    mock_trace_loader.fetch_traces.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )

    # Recent: deferred, nothing recorded.
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=now_ms - 60_000, max_last_trace_timestamp_ms=now_ms
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", now_ms - 20_000, now_ms - 10_000)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", now_ms - 20_000)
    ]
    processor.process_sessions()
    mock_log_assessments.assert_not_called()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(
        timestamp_ms=now_ms - 10_001, session_id=None
    )

    # Stale: recorded on the first trace, checkpoint advances.
    mock_checkpoint_manager.reset_mock()
    stale_ts = now_ms - MAX_TRACE_COMPLETION_WAIT_MS - 60_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=stale_ts - 20_000, max_last_trace_timestamp_ms=now_ms
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-002", stale_ts - 10_000, stale_ts)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-002", stale_ts - 10_000),
        make_trace_info("tr-003", stale_ts),
    ]
    processor.process_sessions()
    mock_evaluate.assert_not_called()
    assert mock_log_assessments.call_args[1]["trace"].info.trace_id == "tr-002"
    logged = mock_log_assessments.call_args[1]["assessments"]
    assert [a.name for a in logged] == ["conversation_completeness"]
    assert logged[0].error.error_code == "TRACE_INCOMPLETE"
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=now_ms, session_id=None)

    # If even that cannot be written, the session is held rather than dropped.
    mock_checkpoint_manager.reset_mock()
    mock_log_assessments.side_effect = ConnectionError("store unavailable")
    processor.process_sessions()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=stale_ts - 1, session_id=None)


def _recent_window(mock_checkpoint_manager, now_ms: int) -> None:
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=now_ms - 60_000, max_last_trace_timestamp_ms=now_ms
    )


def _error_assessment(name: str) -> Feedback:
    return Feedback(
        name=name,
        error=AssessmentError(error_code="SCORER_ERROR", error_message="judge timed out"),
    )


def test_scorer_failure_defers_session_and_keeps_successful_results(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: Two session scorers run; one succeeds and one fails (the harness reports it as an
    error feedback). Judge failures are usually transient, so the session is deferred and the
    checkpoint held for a retry. The successful result is persisted now (a later re-run replaces
    it), and no error assessment is written for the scorer we are going to try again. A previous
    result for the failed scorer is left in place rather than deleted.
    """
    now_ms = int(time.time() * 1000)
    _recent_window(mock_checkpoint_manager, now_ms)
    scorer_a, scorer_b = ConversationCompleteness(), ConversationCompleteness()
    scorer_a.name, scorer_b.name = "scorer_a", "scorer_b"
    sampler = OnlineScorerSampler([make_online_scorer(scorer_a), make_online_scorer(scorer_b)])
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", now_ms - 20_000, now_ms - 10_000)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", now_ms - 20_000)
    ]
    previous_b = make_assessment("old-b", "scorer_b", session_id="sess-001")
    trace = make_trace("tr-001", now_ms - 20_000, assessments=[previous_b])
    mock_trace_loader.fetch_traces.return_value = [trace]
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(trace),
        assessments=[
            make_assessment("new-a", "scorer_a"),
            _error_assessment("scorer_b"),
        ],
        scorer_stats={},
    )
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler, mock_tracking_store
    )

    processor.process_sessions()

    logged = mock_log_assessments.call_args[1]["assessments"]
    assert [a.name for a in logged] == ["scorer_a"]
    mock_tracking_store.delete_assessment.assert_not_called()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(
        timestamp_ms=now_ms - 10_001, session_id=None
    )


def test_scorer_failure_is_recorded_once_retry_budget_is_exhausted(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """A judge that keeps failing must not hold the session watermark forever: once the
    session is older than the retry budget the error feedback is persisted and the checkpoint
    moves on.
    """
    now_ms = int(time.time() * 1000)
    stale_ts = now_ms - MAX_SCORER_FAILURE_RETRY_MS - 60_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=stale_ts - 20_000, max_last_trace_timestamp_ms=now_ms
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", stale_ts - 10_000, stale_ts)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", stale_ts - 10_000)
    ]
    trace = make_trace("tr-001", stale_ts - 10_000)
    mock_trace_loader.fetch_traces.return_value = [trace]
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(trace),
        assessments=[_error_assessment("ConversationCompleteness/v1")],
        scorer_stats={},
    )
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    logged = mock_log_assessments.call_args[1]["assessments"]
    assert len(logged) == 1
    assert logged[0].error is not None
    assert logged[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID] == "sess-001"
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=now_ms, session_id=None)


def test_unexpected_evaluation_failure_is_retried_then_recorded(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: evaluate_session_level_scorers raises (scorer errors are normally converted to
    error feedbacks inside it, so this is an unexpected failure).

    Within the retry budget the session is deferred with nothing persisted. Once the budget is
    exhausted the failure is persisted as an error feedback per scorer on the first trace so a
    deterministic failure cannot hold the checkpoint forever.
    """
    now_ms = int(time.time() * 1000)
    mock_evaluate.side_effect = ValueError("malformed trace")
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )

    # Recent session: deferred.
    _recent_window(mock_checkpoint_manager, now_ms)
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", now_ms - 20_000, now_ms - 10_000)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", now_ms - 20_000)
    ]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", now_ms - 20_000)]
    processor.process_sessions()
    mock_log_assessments.assert_not_called()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(
        timestamp_ms=now_ms - 10_001, session_id=None
    )

    # Stale session: error recorded, checkpoint advances.
    mock_checkpoint_manager.reset_mock()
    stale_ts = now_ms - MAX_SCORER_FAILURE_RETRY_MS - 60_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineSessionScoringTimeWindow(
        min_last_trace_timestamp_ms=stale_ts - 20_000, max_last_trace_timestamp_ms=now_ms
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-002", stale_ts - 10_000, stale_ts)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-002", stale_ts - 10_000)
    ]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-002", stale_ts - 10_000)]
    processor.process_sessions()

    logged = mock_log_assessments.call_args[1]["assessments"]
    assert len(logged) == 1
    assert logged[0].name == "conversation_completeness"
    assert logged[0].error is not None
    assert logged[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID] == "sess-002"
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint == OnlineSessionScoringCheckpoint(timestamp_ms=now_ms, session_id=None)


def test_clean_up_old_assessments_preserves_different_sessions(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """
    Scenario: A trace has an old assessment from a different session (sess-002),
    and we're now scoring sess-001 which produces a new assessment.

    The processor should NOT delete the old assessment since it belongs to a different
    session, preserving assessments from all sessions that the trace participates in.
    """
    old_assessment = make_assessment("old-id", "ConversationCompleteness/v1", session_id="sess-002")
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1500)
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    trace_with_old_assessment = make_trace("tr-001", 1000, assessments=[old_assessment])
    mock_trace_loader.fetch_traces.return_value = [trace_with_old_assessment]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    new_assessment = make_assessment("new-id", "ConversationCompleteness/v1")
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(trace_with_old_assessment),
        assessments=[new_assessment],
        scorer_stats={},
    )
    processor.process_sessions()

    mock_tracking_store.delete_assessment.assert_not_called()


def test_fetch_sessions_calls_once_per_filter_when_scorers_have_different_filters(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store
):
    """
    Scenario: Two scorers have different filter strings (scorer1 with 'tag.env=prod',
    scorer2 with 'tag.env=dev').

    The processor should call find_completed_sessions once per unique filter to
    efficiently fetch sessions that match each filter, avoiding duplicate work
    and ensuring each session is evaluated by the correct scorers.
    """
    scorer1 = ConversationCompleteness()
    scorer2 = ConversationCompleteness()
    scorer1.name = "scorer1"
    scorer2.name = "scorer2"
    sampler = OnlineScorerSampler([
        make_online_scorer(scorer1, filter_string="tag.env = 'prod'"),
        make_online_scorer(scorer2, filter_string="tag.env = 'dev'"),
    ])
    mock_tracking_store.find_completed_sessions.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler, mock_tracking_store
    )
    processor.process_sessions()

    assert mock_tracking_store.find_completed_sessions.call_count == 2
    filter_strings = [
        call[1]["filter_string"]
        for call in mock_tracking_store.find_completed_sessions.call_args_list
    ]
    assert set(filter_strings) == {"tag.env = 'prod'", "tag.env = 'dev'"}


def test_fetch_sessions_calls_once_per_filter_when_any_scorer_has_no_filter(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store
):
    """
    Scenario: Two scorers, one with a filter ('tag.env=prod') and one without (None).

    The processor should call find_completed_sessions twice: once with the specific
    filter for scorer1, and once with no filter for scorer2. This ensures both filtered
    and unfiltered scorers get the appropriate sessions.
    """
    scorer1 = ConversationCompleteness()
    scorer2 = ConversationCompleteness()
    scorer1.name = "scorer1"
    scorer2.name = "scorer2"
    sampler = OnlineScorerSampler([
        make_online_scorer(scorer1, filter_string="tag.env = 'prod'"),
        make_online_scorer(scorer2, filter_string=None),
    ])
    mock_tracking_store.find_completed_sessions.return_value = []
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler, mock_tracking_store
    )
    processor.process_sessions()

    assert mock_tracking_store.find_completed_sessions.call_count == 2
    filter_strings = [
        call[1]["filter_string"]
        for call in mock_tracking_store.find_completed_sessions.call_args_list
    ]
    assert set(filter_strings) == {"tag.env = 'prod'", None}


def test_one_job_drains_a_session_backlog_larger_than_one_batch(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
    monkeypatch,
):
    monkeypatch.setattr("mlflow.genai.scorers.online.session_processor.MAX_SESSIONS_PER_JOB", 2)
    sessions = [make_completed_session(f"sess-{i}", 500, 1000 + i) for i in range(5)]
    state = {"cp": None}

    def find_sessions(**kw):
        lo, hi, after = (
            kw["min_last_trace_timestamp_ms"],
            kw["max_last_trace_timestamp_ms"],
            kw["after_session_id"],
        )
        eligible = [
            s
            for s in sessions
            if lo <= s.last_trace_timestamp_ms <= hi
            and not (
                after is not None and s.last_trace_timestamp_ms == lo and s.session_id <= after
            )
        ]
        return eligible[: kw["max_results"]]

    def window():
        lo = 1000 if state["cp"] is None else max(1000, state["cp"].timestamp_ms)
        return OnlineSessionScoringTimeWindow(
            min_last_trace_timestamp_ms=lo, max_last_trace_timestamp_ms=2000
        )

    mock_tracking_store.find_completed_sessions.side_effect = find_sessions
    mock_checkpoint_manager.persist_checkpoint.side_effect = lambda cp: state.__setitem__("cp", cp)
    mock_checkpoint_manager.get_checkpoint.side_effect = lambda: state["cp"]
    mock_checkpoint_manager.calculate_time_window.side_effect = window
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )

    processor.process_sessions()

    scored = [c[0][0].session.session_id for c in mock_score_session.call_args_list]
    assert scored == [f"sess-{i}" for i in range(5)]
    assert [c[0][0] for c in mock_checkpoint_manager.persist_checkpoint.call_args_list] == [
        OnlineSessionScoringCheckpoint(timestamp_ms=1001, session_id="sess-1"),
        OnlineSessionScoringCheckpoint(timestamp_ms=1003, session_id="sess-3"),
        OnlineSessionScoringCheckpoint(timestamp_ms=2000, session_id=None),
    ]


def test_long_session_is_scored_on_all_of_its_traces(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_evaluate,
    mock_log_assessments,
):
    """A session with more turns than the loader's default page cap must be loaded in full;
    silently scoring the first 500 turns would be wrong and invisible to the user.
    """
    n = 650
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-long", 1000, 1000 + n)
    ]
    infos = [make_trace_info(f"tr-{i:04d}", 1000 + i) for i in range(n)]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = infos
    mock_trace_loader.fetch_traces.side_effect = lambda ids: [make_trace(t, 1000) for t in ids]
    mock_evaluate.return_value = EvalResult(
        eval_item=EvalItem.from_trace(make_trace("tr-0000", 1000)),
        assessments=[make_assessment("a", "ConversationCompleteness/v1")],
        scorer_stats={},
    )
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )

    processor.process_sessions()

    assert mock_trace_loader.fetch_trace_infos_in_range.call_args[1]["max_traces"] is None
    assert len(mock_evaluate.call_args[1]["session_items"]) == n
