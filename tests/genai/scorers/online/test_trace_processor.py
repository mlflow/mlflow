import contextlib
import json
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.genai.evaluation.rate_limiter import NoOpRateLimiter, RPSRateLimiter
from mlflow.genai.judges.adapters.rate_limit_retry_adapters import (
    _RETRY_ADAPTER_REGISTRY,
    RateLimitRetryAdapter,
    register_retry_adapter,
)
from mlflow.genai.scorers.builtin_scorers import Completeness
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

    # When there are no scorers, _build_scoring_tasks returns empty dict,
    # so checkpoint is still advanced but no trace fetching occurs
    mock_checkpoint_manager.persist_checkpoint.assert_called_once()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 2000
    assert checkpoint.trace_id is None
    mock_trace_loader.fetch_traces.assert_not_called()


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


def test_process_traces_skips_until_buffered_window_advances(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=1000, max_trace_timestamp_ms=900
    )
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineTraceScoringCheckpoint(
        timestamp_ms=1000, trace_id="tr-001"
    )
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    processor.process_traces()
    processor.process_traces()

    mock_trace_loader.fetch_trace_infos_in_range.assert_not_called()
    mock_checkpoint_manager.persist_checkpoint.assert_not_called()


def test_process_traces_repairs_future_checkpoint(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    current_time_ms = 1_000_000
    future_timestamp_ms = current_time_ms + 86_400_000
    mock_checkpoint_manager.calculate_time_window.return_value = OnlineTraceScoringTimeWindow(
        min_trace_timestamp_ms=future_timestamp_ms,
        max_trace_timestamp_ms=future_timestamp_ms,
    )
    mock_checkpoint_manager.get_checkpoint.return_value = OnlineTraceScoringCheckpoint(
        timestamp_ms=future_timestamp_ms, trace_id="tr-001"
    )
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    with patch("mlflow.genai.scorers.online.trace_processor.time.time", return_value=1000):
        processor.process_traces()

    mock_trace_loader.fetch_trace_infos_in_range.assert_not_called()
    mock_checkpoint_manager.persist_checkpoint.assert_called_once()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == current_time_ms
    assert checkpoint.trace_id is None


def test_process_traces_updates_checkpoint_when_full_traces_empty(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    """
    Test that when traces are sampled but fetch_traces returns empty
    (e.g., traces deleted between sampling and fetching), we still
    advance the checkpoint using the time window end.
    """
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info("tr-001", 1500),
        make_trace_info("tr-002", 1800),
    ]
    mock_trace_loader.fetch_traces.return_value = []
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    with patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute:
        mock_compute.return_value = []
        processor.process_traces()

    mock_checkpoint_manager.persist_checkpoint.assert_called_once()
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 2000
    assert checkpoint.trace_id is None


def test_process_traces_uses_search_timestamp_for_checkpoint(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [make_trace_info("tr-001", 1500)]
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 86_401_500)]
    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    with patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute:
        mock_compute.return_value = []
        processor.process_traces()

    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    assert checkpoint.timestamp_ms == 1500
    assert checkpoint.trace_id == "tr-001"


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

    with patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute:
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


def test_process_traces_excludes_eval_run_traces(
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

    call_args = mock_trace_loader.fetch_trace_infos_in_range.call_args[0]
    filter_string = call_args[3]
    assert "metadata.mlflow.sourceRun IS NULL" in filter_string


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
        patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute,
        patch("mlflow.genai.evaluation.harness._log_assessments") as mock_log,
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

    with patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute:
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
        patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute,
        patch("mlflow.genai.evaluation.harness._log_assessments") as mock_log,
    ):
        mock_compute.side_effect = [Exception("Scorer failed"), [MagicMock()]]
        processor.process_traces()

        assert mock_compute.call_count == 2
        # Now we log error assessments for failures + successful assessments
        assert mock_log.call_count == 2

        # Verify error assessments were logged for the failed trace
        error_log_call = mock_log.call_args_list[0]
        error_assessments = error_log_call[1]["assessments"]
        assert len(error_assessments) == 1  # One error per scorer
        assert error_assessments[0].error is not None

    mock_checkpoint_manager.persist_checkpoint.assert_called_once()


def test_process_traces_truncates_and_sorts_across_filters(
    mock_trace_loader, mock_checkpoint_manager
):
    """
    Test that when multiple filters return MAX_TRACES_PER_JOB traces each,
    we truncate to MAX_TRACES_PER_JOB total while preserving chronological order.
    This ensures we don't skip earlier traces from certain filters.

    Uses overlapping timestamp ranges to verify correct chronological sorting
    across filters when truncating.
    """
    from mlflow.genai.scorers.online.constants import MAX_TRACES_PER_JOB

    # Create two scorers with different filters
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.env = 'staging'"),
    ]
    sampler = OnlineScorerSampler(configs)

    # Compute timestamp ranges dynamically based on MAX_TRACES_PER_JOB
    # Staging: starts at 0, covers full MAX_TRACES_PER_JOB range (0 to MAX-1)
    # Prod: starts at 80% through staging range, creating 20% overlap
    staging_start = 0
    prod_start = int(0.8 * MAX_TRACES_PER_JOB)

    # Each filter returns MAX_TRACES_PER_JOB traces
    filter1_traces = [
        make_trace_info(f"tr-prod-{i}", prod_start + i) for i in range(MAX_TRACES_PER_JOB)
    ]
    filter2_traces = [
        make_trace_info(f"tr-staging-{i}", staging_start + i) for i in range(MAX_TRACES_PER_JOB)
    ]

    def mock_fetch_trace_infos(exp_id, min_ts, max_ts, filter_str, limit):
        if "tags.env = 'prod'" in filter_str:
            return filter1_traces
        elif "tags.env = 'staging'" in filter_str:
            return filter2_traces
        return []

    mock_trace_loader.fetch_trace_infos_in_range.side_effect = mock_fetch_trace_infos

    # Mock fetch_traces to return full traces for sampled IDs
    def mock_fetch_traces(trace_ids):
        result = []
        for tid in trace_ids:
            if tid.startswith("tr-prod-"):
                idx = int(tid.split("-")[-1])
                result.append(make_trace(tid, prod_start + idx))
            elif tid.startswith("tr-staging-"):
                idx = int(tid.split("-")[-1])
                result.append(make_trace(tid, staging_start + idx))
        return result

    mock_trace_loader.fetch_traces.side_effect = mock_fetch_traces

    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
    )

    with patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute:
        mock_compute.return_value = []
        processor.process_traces()

    # Verify we only processed MAX_TRACES_PER_JOB traces total
    fetched_trace_ids = mock_trace_loader.fetch_traces.call_args[0][0]
    assert len(fetched_trace_ids) == MAX_TRACES_PER_JOB

    # Build expected list of traces in chronological order:
    # 1. Staging-only traces (timestamps 0 to prod_start-1)
    # 2. Interleaved traces in overlap region (prod_start onwards)
    staging_only_count = prod_start  # traces before overlap
    overlap_pairs = (MAX_TRACES_PER_JOB - staging_only_count) // 2

    expected_trace_ids = [f"tr-staging-{i}" for i in range(staging_only_count)]
    for i in range(overlap_pairs):
        expected_trace_ids.append(f"tr-prod-{i}")
        expected_trace_ids.append(f"tr-staging-{staging_only_count + i}")

    # Verify exact traces processed in exact order
    assert fetched_trace_ids == expected_trace_ids

    # Verify checkpoint was updated to the last processed trace
    checkpoint = mock_checkpoint_manager.persist_checkpoint.call_args[0][0]
    last_timestamp = prod_start + overlap_pairs - 1
    last_trace_id = f"tr-staging-{staging_only_count + overlap_pairs - 1}"
    assert checkpoint.timestamp_ms == last_timestamp
    assert checkpoint.trace_id == last_trace_id


def test_process_traces_deduplicates_scorers_across_filters(
    mock_trace_loader, mock_checkpoint_manager
):
    """
    Test that when a trace matches multiple filters that would select the same scorer,
    the scorer only appears once in the task's scorer list.
    """
    # Create the same scorer with two different filters
    completeness_scorer = Completeness()
    configs = [
        make_online_scorer(completeness_scorer, filter_string="tags.env = 'prod'"),
        make_online_scorer(completeness_scorer, filter_string="tags.priority = 'high'"),
    ]
    sampler = OnlineScorerSampler(configs)

    # Same trace returned by both filters (trace matches both conditions)
    shared_trace = make_trace_info("tr-001", 1500)

    def mock_fetch_trace_infos(exp_id, min_ts, max_ts, filter_str, limit):
        # Both filters return the same trace
        return [shared_trace]

    mock_trace_loader.fetch_trace_infos_in_range.side_effect = mock_fetch_trace_infos
    mock_trace_loader.fetch_traces.return_value = [make_trace("tr-001", 1500)]

    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
    )

    with patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute:
        mock_compute.return_value = []
        processor.process_traces()

        # Verify the scorer was called only once per trace (not duplicated)
        assert mock_compute.call_count == 1
        call_args = mock_compute.call_args[1]
        scorers_used = call_args["scorers"]

        # Should only have one instance of Completeness scorer despite matching 2 filters
        assert len(scorers_used) == 1
        assert scorers_used[0].name == completeness_scorer.name


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


# ── Rate limiting tests ──


def test_execute_scoring_uses_aimd_rate_limiter_by_default(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, monkeypatch
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

    captured_limiter = {}

    def capture_compute(eval_item, scorers, rate_limiter=None, max_retries=0):
        captured_limiter["rate_limiter"] = rate_limiter
        captured_limiter["max_retries"] = max_retries
        return MagicMock(assessments=[])

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT", "auto")
    with (
        patch("mlflow.genai.evaluation.harness._compute_eval_scores", side_effect=capture_compute),
        patch("mlflow.genai.evaluation.rate_limiter.eval_retry_context"),
    ):
        processor.process_traces()

    assert isinstance(captured_limiter["rate_limiter"], RPSRateLimiter)
    assert captured_limiter["rate_limiter"]._adaptive is True


def test_execute_scoring_uses_noop_rate_limiter_when_disabled(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, monkeypatch
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

    captured_limiter = {}

    def capture_compute(eval_item, scorers, rate_limiter=None, max_retries=0):
        captured_limiter["rate_limiter"] = rate_limiter
        return MagicMock(assessments=[])

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT", "0")
    with (
        patch("mlflow.genai.evaluation.harness._compute_eval_scores", side_effect=capture_compute),
        patch("mlflow.genai.evaluation.rate_limiter.eval_retry_context"),
    ):
        processor.process_traces()

    assert isinstance(captured_limiter["rate_limiter"], NoOpRateLimiter)


def test_execute_scoring_passes_max_retries_from_env(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, monkeypatch
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

    captured = {}

    def capture_compute(eval_item, scorers, rate_limiter=None, max_retries=0):
        captured["max_retries"] = max_retries
        return MagicMock(assessments=[])

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_RETRIES", "5")
    with (
        patch("mlflow.genai.evaluation.harness._compute_eval_scores", side_effect=capture_compute),
        patch("mlflow.genai.evaluation.rate_limiter.eval_retry_context"),
    ):
        processor.process_traces()

    assert captured["max_retries"] == 5


def test_execute_scoring_uses_fixed_rate_limiter_for_numeric_rate(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, monkeypatch
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

    captured = {}

    def capture_compute(eval_item, scorers, rate_limiter=None, max_retries=0):
        captured["rate_limiter"] = rate_limiter
        return MagicMock(assessments=[])

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT", "25")
    with (
        patch("mlflow.genai.evaluation.harness._compute_eval_scores", side_effect=capture_compute),
        patch("mlflow.genai.evaluation.rate_limiter.eval_retry_context"),
    ):
        processor.process_traces()

    assert isinstance(captured["rate_limiter"], RPSRateLimiter)
    assert captured["rate_limiter"]._adaptive is False
    assert captured["rate_limiter"]._rps == pytest.approx(25.0)


def test_execute_scoring_honors_scorer_rate_limit_env(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers, monkeypatch
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

    captured = {}

    def capture_compute(eval_item, scorers, rate_limiter=None, max_retries=0):
        captured["rate_limiter"] = rate_limiter
        return MagicMock(assessments=[])

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT", "25")
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT", "7")
    with (
        patch("mlflow.genai.evaluation.harness._compute_eval_scores", side_effect=capture_compute),
        patch("mlflow.genai.evaluation.rate_limiter.eval_retry_context"),
    ):
        processor.process_traces()

    assert captured["rate_limiter"]._rps == pytest.approx(7.0)


def test_execute_scoring_scales_rate_by_distinct_scorers(
    mock_trace_loader, mock_checkpoint_manager, monkeypatch
):
    # Disjoint scorer subsets across traces: 3 distinct scorers run, so rate is 10 * 3, not the
    # per-trace peak of 2.
    configs = [
        make_online_scorer(Completeness(name="a"), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="b"), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="c"), filter_string="tags.env = 'staging'"),
    ]
    sampler = OnlineScorerSampler(configs)

    def mock_fetch_trace_infos(exp_id, min_ts, max_ts, filter_str, limit):
        if "prod" in filter_str:
            return [make_trace_info("tr-prod", 1500)]
        return [make_trace_info("tr-staging", 1600)]

    mock_trace_loader.fetch_trace_infos_in_range.side_effect = mock_fetch_trace_infos
    mock_trace_loader.fetch_traces.return_value = [
        make_trace("tr-prod", 1500),
        make_trace("tr-staging", 1600),
    ]

    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler,
        experiment_id="exp1",
    )

    captured = {}

    def capture_compute(eval_item, scorers, rate_limiter=None, max_retries=0):
        captured["rate_limiter"] = rate_limiter
        return MagicMock(assessments=[])

    monkeypatch.setenv("MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT", "10")
    with (
        patch("mlflow.genai.evaluation.harness._compute_eval_scores", side_effect=capture_compute),
        patch("mlflow.genai.evaluation.rate_limiter.eval_retry_context"),
    ):
        processor.process_traces()

    assert captured["rate_limiter"]._rps == pytest.approx(30.0)


def test_execute_scoring_wraps_in_eval_retry_context(
    mock_trace_loader, mock_checkpoint_manager, sampler_with_scorers
):
    traces = [make_trace(f"tr-{i:03d}", 1000 + i * 100) for i in range(3)]
    mock_trace_loader.fetch_trace_infos_in_range.return_value = [
        make_trace_info(f"tr-{i:03d}", 1000 + i * 100) for i in range(3)
    ]
    mock_trace_loader.fetch_traces.return_value = traces

    processor = OnlineTraceScoringProcessor(
        trace_loader=mock_trace_loader,
        checkpoint_manager=mock_checkpoint_manager,
        sampler=sampler_with_scorers,
        experiment_id="exp1",
    )

    entered_threads: list[str] = []
    main_thread_name = threading.current_thread().name

    class _TrackingCtx:
        def __enter__(self):
            entered_threads.append(threading.current_thread().name)
            return self

        def __exit__(self, *args):
            return False

    with (
        patch("mlflow.genai.evaluation.harness._compute_eval_scores") as mock_compute,
        patch(
            "mlflow.genai.evaluation.rate_limiter.eval_retry_context",
            side_effect=lambda: _TrackingCtx(),
        ),
    ):
        mock_compute.return_value = MagicMock(assessments=[])
        processor.process_traces()

    # One context entry per trace
    assert len(entered_threads) == 3
    # Every entry happened on a worker thread, not the main thread
    assert all(t != main_thread_name for t in entered_threads)


def test_register_retry_adapter_deduplicates_by_name():
    original = list(_RETRY_ADAPTER_REGISTRY)
    try:
        adapter_v1 = RateLimitRetryAdapter(
            name="test-dedup",
            is_adapter_active=lambda: False,
            disable_internal_retries=contextlib.nullcontext,
        )
        adapter_v2 = RateLimitRetryAdapter(
            name="test-dedup",
            is_adapter_active=lambda: True,
            disable_internal_retries=contextlib.nullcontext,
        )
        register_retry_adapter(adapter_v1)
        register_retry_adapter(adapter_v2)

        names = [a.name for a in _RETRY_ADAPTER_REGISTRY]
        assert names.count("test-dedup") == 1
        match = next(a for a in _RETRY_ADAPTER_REGISTRY if a.name == "test-dedup")
        # v2 should have replaced v1
        assert match.is_adapter_active() is True
    finally:
        _RETRY_ADAPTER_REGISTRY[:] = original
