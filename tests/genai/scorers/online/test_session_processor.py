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
        name=scorer.name,
        serialized_scorer=json.dumps(scorer.model_dump()),
        online_config=config,
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
    configs = [make_online_scorer(ConversationCompleteness(), sample_rate=1.0)]
    return OnlineScorerSampler(configs)


@pytest.fixture
def empty_sampler():
    return OnlineScorerSampler([])


def test_process_sessions_skips_when_no_scorers(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, empty_sampler
):
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=empty_sampler,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    processor.process_sessions()

    mock_checkpoint_manager.calculate_time_window.assert_not_called()
    mock_tracking_store.find_completed_sessions.assert_not_called()


def test_process_sessions_updates_checkpoint_when_no_sessions(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    mock_tracking_store.find_completed_sessions.return_value = []
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    processor.process_sessions()

    mock_checkpoint_manager.persist_checkpoint.assert_called_once()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 2000
    assert checkpoint.session_id is None


def test_process_sessions_filters_checkpoint_boundary(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineSessionScoringCheckpoint(
        timestamp_ms=1000, session_id="sess-002"
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1000),
        make_completed_session("sess-003", 500, 1000),
        make_completed_session("sess-004", 500, 1500),
    ]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with patch(
        "mlflow.genai.scorers.online.session_processor.OnlineSessionScoringProcessor._score_session"
    ) as mock_score:
        processor.process_sessions()

        assert mock_score.call_count == 2
        scored_tasks = [call[0][0] for call in mock_score.call_args_list]
        assert scored_tasks[0].session.session_id == "sess-003"
        assert scored_tasks[1].session.session_id == "sess-004"


def test_session_rescored_when_new_trace_added_after_checkpoint(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineSessionScoringCheckpoint(
        timestamp_ms=1000, session_id="sess-001"
    )
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 2000),
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", 500),
        make_trace_info("tr-002", 2000),
    ]
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-001", 500),
        make_trace("tr-002", 2000),
    ]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with patch(
        "mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers"
    ) as mock_evaluate:
        mock_evaluate.return_value = {}
        processor.process_sessions()

        mock_evaluate.assert_called_once()


def test_process_sessions_samples_and_scores(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    session = make_completed_session("sess-001", 500, 1500)
    mock_tracking_store.find_completed_sessions.return_value = [session]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 1000)]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with (
        patch(
            "mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers"
        ) as mock_evaluate,
        patch("mlflow.genai.evaluation.harness._log_assessments") as mock_log,
    ):
        mock_evaluate.return_value = {"tr-001": [MagicMock()]}
        processor.process_sessions()

        mock_evaluate.assert_called_once()
        mock_log.assert_called_once()


def test_process_sessions_updates_checkpoint_on_success(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with patch(
        "mlflow.genai.scorers.online.session_processor.OnlineSessionScoringProcessor._score_session"
    ):
        processor.process_sessions()

    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 1500
    assert checkpoint.session_id == "sess-002"


def test_execute_session_scoring_handles_failures(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with patch(
        "mlflow.genai.scorers.online.session_processor.OnlineSessionScoringProcessor._score_session"
    ) as mock_score:
        mock_score.side_effect = [Exception("Session failed"), None]
        processor.process_sessions()

        assert mock_score.call_count == 2

    mock_checkpoint_manager.persist_checkpoint.assert_called_once()


def test_create_factory_method(mock_tracking_store):
    configs = [make_online_scorer(ConversationCompleteness())]

    processor = OnlineSessionScoringProcessor.create(
        experiment_id="exp1",
        online_scorers=configs,
        tracking_store=mock_tracking_store,
    )

    assert processor._experiment_id == "exp1"
    assert isinstance(processor._trace_loader, OnlineTraceLoader)
    assert isinstance(processor._checkpoint_manager, OnlineSessionCheckpointManager)
    assert isinstance(processor._sampler, OnlineScorerSampler)
    assert processor._tracking_store == mock_tracking_store


def test_score_session_excludes_eval_run_traces(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    session = make_completed_session("sess-001", 500, 1500)
    mock_tracking_store.find_completed_sessions.return_value = [session]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    processor.process_sessions()

    call_args = mock_trace_loader.fetch_trace_infos_in_range.call_args[1]
    filter_string = call_args["filter_string"]
    assert "metadata.mlflow.sourceRun IS NULL" in filter_string


def test_score_session_adds_session_metadata_to_assessments(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    session = make_completed_session("sess-001", 500, 1500)
    mock_tracking_store.find_completed_sessions.return_value = [session]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 1000)]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with (
        patch(
            "mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers"
        ) as mock_evaluate,
        patch("mlflow.genai.evaluation.harness._log_assessments") as mock_log,
    ):
        feedback = MagicMock(spec=Assessment)
        feedback.metadata = {}
        mock_evaluate.return_value = {"tr-001": [feedback]}
        processor.process_sessions()

        logged_feedbacks = mock_log.call_args[1]["assessments"]
        assert (
            logged_feedbacks[0].metadata[AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID]
            == "sess-001"
        )


def test_score_session_skips_when_no_traces_found(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    session = make_completed_session("sess-001", 500, 1500)
    mock_tracking_store.find_completed_sessions.return_value = [session]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with patch(
        "mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers"
    ) as mock_evaluate:
        processor.process_sessions()

        mock_evaluate.assert_not_called()


def test_score_session_skips_when_no_applicable_scorers(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store
):
    configs = [make_online_scorer(ConversationCompleteness(), sample_rate=0.0)]
    sampler = OnlineScorerSampler(configs)
    session = make_completed_session("sess-001", 500, 1500)
    mock_tracking_store.find_completed_sessions.return_value = [session]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 1000)]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with patch(
        "mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers"
    ) as mock_evaluate:
        processor.process_sessions()

        mock_evaluate.assert_not_called()


def test_checkpoint_advances_when_all_traces_are_from_eval_runs(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    mock_tracking_store.find_completed_sessions.return_value = [
        make_completed_session("sess-001", 500, 1000),
        make_completed_session("sess-002", 500, 1500),
    ]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    processor.process_sessions()

    mock_checkpoint_manager.persist_checkpoint.assert_called_once()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 1500
    assert checkpoint.session_id == "sess-002"


def test_clean_up_old_assessments_removes_duplicates(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    old_assessment = MagicMock(spec=Assessment)
    old_assessment.assessment_id = "old-id"
    old_assessment.name = "ConversationCompleteness/v1"
    old_assessment.metadata = {AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID: "sess-001"}

    trace = make_trace("tr-001", 1000, assessments=[old_assessment])
    session = make_completed_session("sess-001", 500, 1500)
    mock_tracking_store.find_completed_sessions.return_value = [session]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [trace]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with (
        patch(
            "mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers"
        ) as mock_evaluate,
        patch("mlflow.genai.evaluation.harness._log_assessments"),
    ):
        new_assessment = MagicMock(spec=Assessment)
        new_assessment.assessment_id = "new-id"
        new_assessment.name = "ConversationCompleteness/v1"
        new_assessment.metadata = {}
        mock_evaluate.return_value = {"tr-001": [new_assessment]}
        processor.process_sessions()

        mock_tracking_store.delete_assessment.assert_called_once_with(
            trace_id="tr-001", assessment_id="old-id"
        )


def test_clean_up_old_assessments_preserves_different_sessions(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store, sampler_with_scorers
):
    old_assessment = MagicMock(spec=Assessment)
    old_assessment.assessment_id = "old-id"
    old_assessment.name = "ConversationCompleteness/v1"
    old_assessment.metadata = {AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID: "sess-002"}

    trace = make_trace("tr-001", 1000, assessments=[old_assessment])
    session = make_completed_session("sess-001", 500, 1500)
    mock_tracking_store.find_completed_sessions.return_value = [session]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1000)]
    mock_trace_loader.fetch_traces.return_value = [trace]
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    with (
        patch(
            "mlflow.genai.evaluation.session_utils.evaluate_session_level_scorers"
        ) as mock_evaluate,
        patch("mlflow.genai.evaluation.harness._log_assessments"),
    ):
        new_assessment = MagicMock(spec=Assessment)
        new_assessment.assessment_id = "new-id"
        new_assessment.name = "ConversationCompleteness/v1"
        new_assessment.metadata = {}
        mock_evaluate.return_value = {"tr-001": [new_assessment]}
        processor.process_sessions()

        mock_tracking_store.delete_assessment.assert_not_called()


def test_fetch_sessions_calls_once_per_filter_when_scorers_have_different_filters(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store
):
    scorer1 = ConversationCompleteness()
    scorer1.name = "scorer1"
    scorer2 = ConversationCompleteness()
    scorer2.name = "scorer2"
    configs = [
        make_online_scorer(scorer1, filter_string="tag.env = 'prod'"),
        make_online_scorer(scorer2, filter_string="tag.env = 'dev'"),
    ]
    sampler = OnlineScorerSampler(configs)
    mock_tracking_store.find_completed_sessions.return_value = []
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    processor.process_sessions()

    # Should call find_completed_sessions twice, once for each filter
    assert mock_tracking_store.find_completed_sessions.call_count == 2
    filter_strings = [
        call[1]["filter_string"]
        for call in mock_tracking_store.find_completed_sessions.call_args_list
    ]
    assert set(filter_strings) == {"tag.env = 'prod'", "tag.env = 'dev'"}


def test_fetch_sessions_calls_once_per_filter_when_any_scorer_has_no_filter(
    mock_trace_loader, mock_checkpoint_manager, mock_tracking_store
):
    scorer1 = ConversationCompleteness()
    scorer1.name = "scorer1"
    scorer2 = ConversationCompleteness()
    scorer2.name = "scorer2"
    configs = [
        make_online_scorer(scorer1, filter_string="tag.env = 'prod'"),
        make_online_scorer(scorer2, filter_string=None),
    ]
    sampler = OnlineScorerSampler(configs)
    mock_tracking_store.find_completed_sessions.return_value = []
    processor = OnlineSessionScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
        tracking_store=mock_tracking_store,
    )

    processor.process_sessions()

    # Should call find_completed_sessions twice, once with filter and once with None
    assert mock_tracking_store.find_completed_sessions.call_count == 2
    filter_strings = [
        call[1]["filter_string"]
        for call in mock_tracking_store.find_completed_sessions.call_args_list
    ]
    assert set(filter_strings) == {"tag.env = 'prod'", None}
