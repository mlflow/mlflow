from unittest import mock

import pytest

from mlflow.entities.run_status import RunStatus
from mlflow.genai.discovery.checkpoint import (
    MAX_SCHEDULED_SAMPLE_SIZE,
    CheckpointTrace,
    parse_detection_checkpoint,
)
from mlflow.genai.discovery.job import run_scheduled_issue_detection_job
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_CHECKPOINT,
    MLFLOW_ISSUE_DETECTION_TRIGGER,
    MLFLOW_RUN_TYPE,
    MLFLOW_RUN_TYPE_ISSUE_DETECTION,
)

WINDOW_LOWER_MS = 1_800_000_000_000
WINDOW_UPPER_MS = 1_800_003_600_000
CATEGORIES = ["correctness", "cost"]


@pytest.fixture
def client():
    created = mock.MagicMock()
    created.info.run_id = "run-scheduled"
    client = mock.MagicMock()
    client.create_run.return_value = created
    return client


def _trace(trace_id: str):
    trace = mock.MagicMock()
    trace.info.trace_id = trace_id
    return trace


def _result(cost: float = 0.25):
    result = mock.MagicMock()
    result.summary = "Found 1 issue"
    result.issues = [mock.MagicMock()]
    result.total_traces_analyzed = 2
    result.total_cost_usd = cost
    return result


def test_job_has_metadata_and_is_exclusive_per_experiment():
    metadata = run_scheduled_issue_detection_job._job_fn_metadata
    assert metadata.name == "scheduled_issue_detection"
    assert metadata.exclusive == ["experiment_id"]


def test_an_empty_window_creates_no_run_and_leaves_the_watermark_alone(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch("mlflow.genai.discovery.job._newest_trace_matching", return_value=None),
        mock.patch("mlflow.genai.discovery.job.sample_traces") as sample,
    ):
        result = run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    assert result == {"traces_in_window": 0}
    client.create_run.assert_not_called()
    client.set_experiment_tag.assert_not_called()
    sample.assert_not_called()


def test_a_window_whose_sample_comes_back_empty_creates_no_run(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching",
            return_value=CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1),
        ),
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[]) as sample,
    ):
        result = run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    assert result == {"traces_in_window": 0}
    client.create_run.assert_not_called()
    client.set_experiment_tag.assert_not_called()
    sample.assert_called_once()


def test_a_successful_occurrence_records_a_run_and_advances_the_watermark(client):
    newest = CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1)
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch("mlflow.genai.discovery.job._newest_trace_matching", return_value=newest),
        mock.patch(
            "mlflow.genai.discovery.job.sample_traces",
            return_value=[_trace("tr-1"), _trace("tr-2")],
        ) as sample,
        mock.patch(
            "mlflow.genai.discovery.job.discover_issues", return_value=_result()
        ) as discover,
    ):
        client._tracing_client.batch_get_traces.return_value = [_trace("tr-1"), _trace("tr-2")]
        result = run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
            trace_filter="tags.`team` = 'search'",
        )

    sample.assert_called_once()
    assert sample.call_args.args[0] == MAX_SCHEDULED_SAMPLE_SIZE
    # The occurrence's own window, ANDed with the schedule's clauses.
    assert sample.call_args.kwargs["search_kwargs"]["filter_string"] == (
        f"timestamp_ms > {WINDOW_LOWER_MS} AND timestamp_ms <= {WINDOW_UPPER_MS} "
        "AND tags.`team` = 'search'"
    )

    tags = client.create_run.call_args.kwargs["tags"]
    assert tags[MLFLOW_RUN_TYPE] == MLFLOW_RUN_TYPE_ISSUE_DETECTION
    assert tags[MLFLOW_ISSUE_DETECTION_TRIGGER] == "schedule"
    assert tags["categories"] == "correctness,cost"
    assert tags["total_traces"] == 2

    discover.assert_called_once()
    assert discover.call_args.kwargs["categories"] == CATEGORIES
    assert discover.call_args.kwargs["run_id"] == "run-scheduled"
    client.set_terminated.assert_called_once_with(
        "run-scheduled", RunStatus.to_string(RunStatus.FINISHED)
    )

    client.set_experiment_tag.assert_called_once()
    experiment_id, key, value = client.set_experiment_tag.call_args.args
    assert (experiment_id, key) == ("exp-1", MLFLOW_ISSUE_DETECTION_CHECKPOINT)
    checkpoint = parse_detection_checkpoint(value)
    # The newest trace in the window, not the newest one sampled.
    assert checkpoint.last_scanned_trace_time_ms == newest.time_ms
    assert checkpoint.last_scanned_trace_id == "tr-newest"
    assert checkpoint.last_run_id == "run-scheduled"

    assert result["traces_in_window"] == 2
    assert result["run_id"] == "run-scheduled"


def test_a_window_with_no_schedule_filter_uses_only_its_own_bounds(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching",
            return_value=CheckpointTrace(trace_id="tr", time_ms=WINDOW_UPPER_MS),
        ),
        mock.patch(
            "mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr")]
        ) as sample,
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=_result()),
    ):
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    assert sample.call_args.kwargs["search_kwargs"]["filter_string"] == (
        f"timestamp_ms > {WINDOW_LOWER_MS} AND timestamp_ms <= {WINDOW_UPPER_MS}"
    )


def test_a_failed_occurrence_fails_its_run_and_leaves_the_watermark_alone(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching",
            return_value=CheckpointTrace(trace_id="tr", time_ms=WINDOW_UPPER_MS),
        ),
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr")]),
        mock.patch(
            "mlflow.genai.discovery.job.discover_issues",
            side_effect=RuntimeError("the judge was unreachable"),
        ) as discover,
        pytest.raises(RuntimeError, match="the judge was unreachable"),
    ):
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    discover.assert_called_once()
    client.set_terminated.assert_called_once_with(
        "run-scheduled", RunStatus.to_string(RunStatus.FAILED)
    )
    # The next occurrence must cover this window again.
    client.set_experiment_tag.assert_not_called()


def test_the_default_model_is_recorded_when_the_schedule_names_none(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching",
            return_value=CheckpointTrace(trace_id="tr", time_ms=WINDOW_UPPER_MS),
        ),
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr")]),
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=_result()),
    ):
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    from mlflow.genai.discovery.constants import DEFAULT_MODEL

    assert client.create_run.call_args.kwargs["tags"]["model"] == DEFAULT_MODEL


def test_the_experiments_saved_priorities_reach_the_pipeline(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching",
            return_value=CheckpointTrace(trace_id="tr", time_ms=WINDOW_UPPER_MS),
        ),
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr")]),
        mock.patch(
            "mlflow.genai.discovery.job.discover_issues", return_value=_result()
        ) as discover,
    ):
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
            instructions="watch the checkout flow",
        )

    # A scheduled run steers the judge the same way a manual run with the same settings would.
    assert discover.call_args.kwargs["instructions"] == "watch the checkout flow"


def test_an_experiment_with_no_saved_priorities_steers_nothing(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching",
            return_value=CheckpointTrace(trace_id="tr", time_ms=WINDOW_UPPER_MS),
        ),
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr")]),
        mock.patch(
            "mlflow.genai.discovery.job.discover_issues", return_value=_result()
        ) as discover,
    ):
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    assert discover.call_args.kwargs["instructions"] is None
