import json
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.genai.scorers.builtin_scorers import Completeness
from mlflow.genai.scorers.online.entities import OnlineScorer
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


def make_online_scorer(scorer, sample_rate: float = 1.0, filter_string: str | None = None):
    return OnlineScorer(
        name=scorer.name,
        experiment_id="exp1",
        serialized_scorer=json.dumps(scorer.model_dump()),
        sample_rate=sample_rate,
        filter_string=filter_string,
    )


def make_trace_info(trace_id: str, timestamp_ms: int = 1000):
    return MagicMock(spec=TraceInfo, trace_id=trace_id, timestamp_ms=timestamp_ms)


def make_trace(trace_id: str, timestamp_ms: int = 1000):
    info = MagicMock(spec=TraceInfo, trace_id=trace_id, timestamp_ms=timestamp_ms)
    data = MagicMock(spec=TraceData)
    return MagicMock(spec=Trace, info=info, data=data)


@pytest.fixture
def mock_trace_loader():
    return MagicMock(spec=OnlineTraceLoader)


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


def test_process_traces_skips_when_no_scorers(
    mock_trace_loader, mock_checkpoint_manager, empty_sampler
):
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=empty_sampler,
        experiment_id="exp1",
    )

    processor.process_traces()

    mock_checkpoint_manager.calculate_time_window.assert_not_called()
    mock_trace_loader.fetch_trace_infos_in_range.assert_not_called()


def test_process_traces_updates_checkpoint_when_no_traces(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    processor.process_traces()

    mock_checkpoint_manager.persist_checkpoint.assert_called_once()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 2000
    assert checkpoint.trace_id is None


def test_process_traces_filters_checkpoint_boundary(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineTraceScoringCheckpoint(
        timestamp_ms=1000, trace_id="tr-002"
    )
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", 1000),
        make_trace_info("tr-002", 1000),
        make_trace_info("tr-003", 1000),
        make_trace_info("tr-004", 1500),
    ]
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-003", 1000),
        make_trace("tr-004", 1500),
    ]
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    with patch("mlflow.genai.scorers.online.trace_processor._compute_eval_scores") as mock_compute:
        mock_compute.return_value = []
        processor.process_traces()

    mock_trace_loader.fetch_traces.assert_called_once_with(["tr-003", "tr-004"])


def test_process_traces_groups_by_filter(mock_trace_loader, mock_checkpoint_manager):
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.env = 'staging'"),
    ]
    sampler = OnlineScorerSampler(configs)
    mock_trace_loader.fetch_trace_infos_in_range.return_value = []
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
    )

    processor.process_traces()

    assert mock_trace_loader.fetch_trace_infos_in_range.call_count == 2
    call_args = [c[0] for c in mock_trace_loader.fetch_trace_infos_in_range.call_args_list]
    filters = [args[3] for args in call_args]
    assert any("tags.env = 'prod'" in f for f in filters)
    assert any("tags.env = 'staging'" in f for f in filters)


def test_process_traces_samples_and_scores(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    trace = make_trace("tr-001", 1500)
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1500)]
    mock_trace_loader.fetch_traces.return_value = [trace]
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    with (
        patch("mlflow.genai.scorers.online.trace_processor._compute_eval_scores") as mock_compute,
        patch("mlflow.genai.scorers.online.trace_processor._log_assessments") as mock_log,
    ):
        mock_compute.return_value = [MagicMock()]
        processor.process_traces()

        mock_compute.assert_called_once()
        mock_log.assert_called_once()


def test_process_traces_updates_checkpoint_on_success(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", 1000),
        make_trace_info("tr-002", 1500),
    ]
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-001", 1000),
        make_trace("tr-002", 1500),
    ]
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    with patch("mlflow.genai.scorers.online.trace_processor._compute_eval_scores") as mock_compute:
        mock_compute.return_value = []
        processor.process_traces()

    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 1500
    assert checkpoint.trace_id == "tr-002"


def test_execute_scoring_handles_failures(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", 1000),
        make_trace_info("tr-002", 1500),
    ]
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-001", 1000),
        make_trace("tr-002", 1500),
    ]
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    with (
        patch("mlflow.genai.scorers.online.trace_processor._compute_eval_scores") as mock_compute,
        patch("mlflow.genai.scorers.online.trace_processor._log_assessments") as mock_log,
    ):
        mock_compute.side_effect = [Exception("Scorer failed"), [MagicMock()]]
        processor.process_traces()

        assert mock_compute.call_count == 2
        assert mock_log.call_count == 1

    mock_checkpoint_manager.persist_checkpoint.assert_called_once()


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
