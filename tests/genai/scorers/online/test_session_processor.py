import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_location import (
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.genai.scorers.builtin_scorers import ConversationCompleteness
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


def make_trace_info(trace_id: str, timestamp_ms: int = 1000):
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="exp1"),
        ),
        request_time=timestamp_ms,
        state=TraceState.OK,
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
    Scenario: Four sessions exist at the checkpoint boundary (timestamp=1000),
    where sessions sess-001 and sess-002 were already scored in a previous run
    (checkpoint at timestamp=1000, session_id=sess-002).

    The processor should only score sess-003 (at same timestamp but after checkpoint
    session_id) and sess-004 (at later timestamp), filtering out sess-001 and sess-002
    to avoid duplicate scoring.
    """
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineSessionScoringCheckpoint(
        timestamp_ms=1000, session_id="sess-002"
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1000),
        make_completed_session("sess-003", 500, 1000),
        make_completed_session("sess-004", 500, 1500),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    assert mock_score_session.call_count == 2
    scored_tasks = [call[0][0] for call in mock_score_session.call_args_list]
    assert scored_tasks[0].session.session_id == "sess-003"
    assert scored_tasks[1].session.session_id == "sess-004"


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
    mock_evaluate.return_value = {
        "tr-001": [make_assessment("assess-1", "ConversationCompleteness/v1")],
        "tr-002": [make_assessment("assess-2", "ConversationCompleteness/v1")],
    }
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
    mock_evaluate.return_value = {"tr-001": [MagicMock()]}
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
    sess-002 at timestamp 1500).

    The processor should update the checkpoint to the last scored session (sess-002
    at timestamp 1500) to resume from this point in the next run.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    processor.process_sessions()

    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 1500
    assert checkpoint.session_id == "sess-002"


def test_execute_session_scoring_handles_failures(
    mock_trace_loader,
    mock_checkpoint_manager,
    mock_tracking_store,
    sampler_with_scorers,
    mock_score_session,
):
    """
    Scenario: Two sessions need scoring, but the first one (sess-001) fails with an error
    while the second one (sess-002) succeeds.

    The processor should log the failure, continue processing sess-002, and still update
    the checkpoint after both attempts. This ensures one failing session doesn't block
    progress on other sessions.
    """
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    mock_score_session.side_effect = [Exception("Session failed"), None]
    processor.process_sessions()

    assert mock_score_session.call_count == 2
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 1500
    assert checkpoint.session_id == "sess-002"


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
    mock_evaluate.return_value = {"tr-001": [feedback]}
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

    The processor should still advance the checkpoint to the last session (sess-002)
    even though no actual scoring occurred, preventing the processor from getting
    stuck repeatedly attempting to score eval-only sessions.
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
    assert checkpoint.timestamp_ms == 1500
    assert checkpoint.session_id == "sess-002"


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
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-001", 1000, assessments=[old_assessment])
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    new_assessment = make_assessment("new-id", "ConversationCompleteness/v1")
    mock_evaluate.return_value = {"tr-001": [new_assessment]}
    processor.process_sessions()

    mock_tracking_store.delete_assessment.assert_called_once_with(
        trace_id="tr-001", assessment_id="old-id"
    )


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
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-001", 1000, assessments=[old_assessment])
    ]
    processor = make_processor(
        mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, mock_tracking_store
    )
    new_assessment = make_assessment("new-id", "ConversationCompleteness/v1")
    mock_evaluate.return_value = {"tr-001": [new_assessment]}
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
    sampler = OnlineScorerSampler(
        [
            make_online_scorer(scorer1, filter_string="tag.env = 'prod'"),
            make_online_scorer(scorer2, filter_string="tag.env = 'dev'"),
        ]
    )
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
    sampler = OnlineScorerSampler(
        [
            make_online_scorer(scorer1, filter_string="tag.env = 'prod'"),
            make_online_scorer(scorer2, filter_string=None),
        ]
    )
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
