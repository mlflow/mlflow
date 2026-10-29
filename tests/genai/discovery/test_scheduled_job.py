from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.entities.run_status import RunStatus
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.checkpoint import (
    MAX_SCHEDULED_SESSION_SAMPLE_SIZE,
    CheckpointTrace,
    DetectionCheckpoint,
    parse_detection_checkpoint,
)
from mlflow.genai.discovery.constants import SAMPLE_POOL_MULTIPLIER
from mlflow.genai.discovery.job import (
    MAX_ERROR_TAG_LENGTH,
    _describe_failure,
    run_scheduled_issue_detection_job,
)
from mlflow.genai.discovery.sampling import sample_traces
from mlflow.server.jobs.utils import MLFLOW_SERVER_JOB_ID_ENV_VAR
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_CHECKPOINT,
    MLFLOW_ISSUE_DETECTION_ERROR,
    MLFLOW_ISSUE_DETECTION_JOB_ID,
    MLFLOW_ISSUE_DETECTION_TRIGGER,
    MLFLOW_ISSUE_DETECTION_WINDOW_END,
    MLFLOW_ISSUE_DETECTION_WINDOW_START,
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


NEWEST = CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1)


@pytest.fixture
def occurrence(client):
    """
    One scheduled occurrence with every collaborator patched and a default happy path.

    Tests override what they are about -- ``occurrence.sample.return_value = []`` for an empty
    sample, ``occurrence.newest.return_value = None`` for an empty window -- and call
    ``occurrence.run()`` with any parameter overrides. The defaults are a window that holds
    traces, two of them sampled, and a successful analysis.
    """
    with ExitStack() as stack:
        patch = lambda target, **kw: stack.enter_context(  # noqa: E731
            mock.patch(f"mlflow.genai.discovery.job.{target}", **kw)
        )
        patch("MlflowClient", return_value=client)
        mocks = SimpleNamespace(
            client=client,
            newest=patch("_newest_trace_matching", return_value=NEWEST),
            sample=patch("sample_traces", return_value=[_trace("tr-1"), _trace("tr-2")]),
            discover=patch("discover_issues", return_value=_result()),
            write_checkpoint=patch("write_detection_checkpoint"),
        )
        client._tracing_client.batch_get_traces.return_value = [_trace("tr-1"), _trace("tr-2")]
        client.get_experiment.return_value = _experiment_with_checkpoint(None)
        mocks.run = lambda **overrides: run_scheduled_issue_detection_job(**{
            "experiment_id": "exp-1",
            "window_lower_bound_ms": WINDOW_LOWER_MS,
            "window_upper_bound_ms": WINDOW_UPPER_MS,
            "categories": CATEGORIES,
            **overrides,
        })
        yield mocks


def _experiment_with_checkpoint(watermark_ms: int | None):
    """An experiment whose checkpoint tag reads back as ``watermark_ms``, or carries none."""
    experiment = mock.MagicMock()
    experiment.tags = (
        {}
        if watermark_ms is None
        else {
            MLFLOW_ISSUE_DETECTION_CHECKPOINT: DetectionCheckpoint(
                last_scanned_trace_time_ms=watermark_ms
            ).to_json()
        }
    )
    return experiment


def test_job_has_metadata_and_is_exclusive_per_experiment():
    metadata = run_scheduled_issue_detection_job._job_fn_metadata
    assert metadata.name == "scheduled_issue_detection"
    assert metadata.exclusive == ["experiment_id"]


def test_an_empty_window_records_a_finished_run_and_leaves_the_watermark_alone(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch("mlflow.genai.discovery.job._newest_trace_matching", return_value=None),
        mock.patch("mlflow.genai.discovery.job.sample_traces") as sample,
        mock.patch("mlflow.genai.discovery.job.discover_issues") as discover,
    ):
        result = run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    # A quiet window still leaves a run behind: "fired and found nothing" has to be
    # distinguishable from "never fired". Nothing is analyzed, so the model is never called.
    assert result == {"traces_in_window": 0, "run_id": "run-scheduled"}
    tags = client.create_run.call_args.kwargs["tags"]
    assert tags["total_traces"] == "0"
    assert tags[MLFLOW_RUN_TYPE] == MLFLOW_RUN_TYPE_ISSUE_DETECTION
    assert tags[MLFLOW_ISSUE_DETECTION_TRIGGER] == "schedule"
    client.set_terminated.assert_called_once_with(
        "run-scheduled", RunStatus.to_string(RunStatus.FINISHED)
    )
    discover.assert_not_called()
    client.link_traces_to_run.assert_not_called()
    client.set_experiment_tag.assert_not_called()
    sample.assert_not_called()


def test_a_window_whose_sample_comes_back_empty_records_a_run_over_zero_traces(client):
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching",
            return_value=CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1),
        ),
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[]) as sample,
        mock.patch("mlflow.genai.discovery.job.write_detection_checkpoint") as write_checkpoint,
    ):
        result = run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    assert result == {"traces_in_window": 0, "run_id": "run-scheduled"}
    assert client.create_run.call_args.kwargs["tags"]["total_traces"] == "0"
    sample.assert_called_once()
    # The window was still covered, so the watermark moves, and it names the run that covered it.
    checkpoint = write_checkpoint.call_args.args[2]
    assert checkpoint.last_scanned_trace_time_ms == WINDOW_UPPER_MS - 1
    assert checkpoint.last_run_id == "run-scheduled"


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
    assert sample.call_args.args[0] == MAX_SCHEDULED_SESSION_SAMPLE_SIZE
    # The occurrence's own window, ANDed with the schedule's clauses.
    assert sample.call_args.kwargs["search_kwargs"]["filter_string"] == (
        f"timestamp_ms > {WINDOW_LOWER_MS} AND timestamp_ms <= {WINDOW_UPPER_MS} "
        "AND tags.`team` = 'search'"
    )

    tags = client.create_run.call_args.kwargs["tags"]
    assert tags[MLFLOW_RUN_TYPE] == MLFLOW_RUN_TYPE_ISSUE_DETECTION
    assert tags[MLFLOW_ISSUE_DETECTION_TRIGGER] == "schedule"
    assert tags["categories"] == "correctness,cost"
    assert tags["total_traces"] == "2"
    # Tags go to create_run, which stores values verbatim: a non-string value is a TypeError at
    # run creation, not a cosmetic difference.
    assert all(isinstance(value, str) for value in tags.values())

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


@pytest.mark.parametrize("instructions", ["watch the checkout flow", None])
def test_the_experiments_saved_priorities_reach_the_pipeline(occurrence, instructions):
    # A scheduled run steers the judge the same way a manual run with the same settings would,
    # and passes nothing on when the experiment saved nothing.
    occurrence.run(instructions=instructions)
    assert occurrence.discover.call_args.kwargs["instructions"] == instructions


@pytest.mark.parametrize("job_id", ["job-42", None])
def test_the_run_carries_the_job_id_when_it_has_one(occurrence, monkeypatch, job_id):
    # Without the tag the Issues tab treats a RUNNING detection run as a submission that never
    # started, drops it after two minutes, and lets a second run start over the top. Outside a
    # job worker there is no id to carry, and the run is created without it rather than failing.
    if job_id is None:
        monkeypatch.delenv(MLFLOW_SERVER_JOB_ID_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(MLFLOW_SERVER_JOB_ID_ENV_VAR, job_id)

    occurrence.run()

    tags = occurrence.client.create_run.call_args.kwargs["tags"]
    assert tags.get(MLFLOW_ISSUE_DETECTION_JOB_ID) == job_id


def test_the_watermark_lookup_ignores_the_schedule_filter(client):
    # The filter scopes what gets analyzed, not how far the occurrence covered. Applying it to the
    # watermark lookup too would stall the checkpoint whenever a window holds traces but none that
    # match, leaving every later occurrence to re-scan the same stretch.
    newest = CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1)
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching", return_value=newest
        ) as newest_lookup,
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr-1")]),
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=_result()),
    ):
        client._tracing_client.batch_get_traces.return_value = [_trace("tr-1")]
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
            trace_filter="tags.`team` = 'search'",
        )

    assert newest_lookup.call_args.args[1] == (
        f"timestamp_ms > {WINDOW_LOWER_MS} AND timestamp_ms <= {WINDOW_UPPER_MS}"
    )


def test_a_window_whose_traces_all_fail_the_filter_still_advances_the_watermark(client):
    # Nothing to analyze, but the occurrence did cover the window, so the next one starts after it.
    newest = CheckpointTrace(trace_id="tr-unmatched", time_ms=WINDOW_UPPER_MS - 1)
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch("mlflow.genai.discovery.job._newest_trace_matching", return_value=newest),
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[]),
        mock.patch("mlflow.genai.discovery.job.write_detection_checkpoint") as write_checkpoint,
        mock.patch("mlflow.genai.discovery.job.discover_issues") as discover,
    ):
        result = run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
            trace_filter="tags.`team` = 'search'",
        )

    # No analysis for an empty sample -- but a run records that the occurrence happened, and the
    # watermark moves to the window's newest.
    discover.assert_not_called()
    assert client.create_run.call_args.kwargs["tags"]["total_traces"] == "0"
    assert result == {"traces_in_window": 0, "run_id": "run-scheduled"}
    assert write_checkpoint.call_args.args[2].last_scanned_trace_time_ms == newest.time_ms


def test_a_queued_occurrence_skips_what_ran_ahead_of_it(client):
    # One worker, two occurrences for the same experiment: `exclusive` serializes them but does
    # not refresh the bounds the second was dispatched with. Without raising its lower bound, the
    # second re-analyzes everything the first already covered -- paying twice for the same traces.
    covered_to = WINDOW_LOWER_MS + 30 * 60 * 1000
    client.get_experiment.return_value = _experiment_with_checkpoint(covered_to)
    newest = CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1)
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching", return_value=newest
        ) as newest_lookup,
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr-1")]),
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=_result()),
    ):
        client._tracing_client.batch_get_traces.return_value = [_trace("tr-1")]
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    # Lower bound moved up to the watermark; the upper bound is still this occurrence's fire time,
    # so a delayed job cannot swallow the next occurrence's window.
    assert newest_lookup.call_args.args[1] == (
        f"timestamp_ms > {covered_to} AND timestamp_ms <= {WINDOW_UPPER_MS}"
    )


def test_an_occurrence_that_waited_for_nothing_keeps_its_dispatched_window(client):
    client.get_experiment.return_value = _experiment_with_checkpoint(None)
    newest = CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1)
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching", return_value=newest
        ) as newest_lookup,
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr-1")]),
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=_result()),
    ):
        client._tracing_client.batch_get_traces.return_value = [_trace("tr-1")]
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    assert newest_lookup.call_args.args[1] == (
        f"timestamp_ms > {WINDOW_LOWER_MS} AND timestamp_ms <= {WINDOW_UPPER_MS}"
    )


def test_a_stale_watermark_does_not_drag_the_window_backwards(client):
    # A watermark behind the dispatched bound (a checkpoint written by an older, wider occurrence)
    # must not widen this window -- `max` not "whatever the checkpoint says".
    client.get_experiment.return_value = _experiment_with_checkpoint(WINDOW_LOWER_MS - 60_000)
    newest = CheckpointTrace(trace_id="tr-newest", time_ms=WINDOW_UPPER_MS - 1)
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=client),
        mock.patch(
            "mlflow.genai.discovery.job._newest_trace_matching", return_value=newest
        ) as newest_lookup,
        mock.patch("mlflow.genai.discovery.job.sample_traces", return_value=[_trace("tr-1")]),
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=_result()),
    ):
        client._tracing_client.batch_get_traces.return_value = [_trace("tr-1")]
        run_scheduled_issue_detection_job(
            experiment_id="exp-1",
            window_lower_bound_ms=WINDOW_LOWER_MS,
            window_upper_bound_ms=WINDOW_UPPER_MS,
            categories=CATEGORIES,
        )

    assert newest_lookup.call_args.args[1] == (
        f"timestamp_ms > {WINDOW_LOWER_MS} AND timestamp_ms <= {WINDOW_UPPER_MS}"
    )


def _session_trace(trace_id: str, session_id: str, time_ms: int):
    """A real ``Trace`` carrying session metadata, so the actual sampler groups it."""
    return Trace(
        info=TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id("exp-1"),
            request_time=time_ms,
            execution_duration=1,
            state=TraceState.OK,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: session_id},
        ),
        data=TraceData(spans=[]),
    )


def _honest_search(store: list[Trace]):
    """A ``search_traces`` stand-in that honors ``max_results``, as a real store does."""
    asked: list[int | None] = []

    def search(max_results: int | None = None, **kwargs):
        asked.append(max_results)
        return store[:max_results]

    return search, asked


def test_an_occurrence_samples_with_the_session_cap(occurrence):
    occurrence.run()

    assert occurrence.sample.call_args.args[0] == MAX_SCHEDULED_SESSION_SAMPLE_SIZE


@pytest.mark.parametrize(
    ("traces_per_session", "expected"),
    [
        # One trace per session: the sample is the session cap itself.
        (1, MAX_SCHEDULED_SESSION_SAMPLE_SIZE),
        # Long sessions: the pool the sampler fetched is the ceiling, since every returned trace
        # comes out of that pool however many sessions are selected.
        (10, MAX_SCHEDULED_SESSION_SAMPLE_SIZE * SAMPLE_POOL_MULTIPLIER),
    ],
)
def test_the_session_cap_bounds_traces_too(
    traces_per_session: int, expected: int, monkeypatch: pytest.MonkeyPatch
):
    # Why there is no separate trace cap: ``sample_traces`` fetches a pool of
    # ``sample_size * SAMPLE_POOL_MULTIPLIER`` traces and returns a subset of it, so an
    # occurrence cannot analyze more than that however long its sessions are. Driven against a
    # store far larger than the bound with a stub that honors ``max_results`` -- a stub that
    # ignores it would "prove" a bound the real path can never reach.
    ceiling = MAX_SCHEDULED_SESSION_SAMPLE_SIZE * SAMPLE_POOL_MULTIPLIER
    store = [
        _session_trace(f"tr-{session}-{index}", f"sess-{session}", 1_800_000_000_000 + session)
        for session in range(4 * ceiling)
        for index in range(traces_per_session)
    ]
    search, asked = _honest_search(store)
    monkeypatch.setattr("mlflow.genai.discovery.sampling.mlflow.search_traces", search)

    analyzed = sample_traces(
        MAX_SCHEDULED_SESSION_SAMPLE_SIZE,
        search_kwargs={"experiment_ids": ["exp-1"], "filter_string": "timestamp_ms > 0"},
    )

    assert len(store) > ceiling
    assert asked == [ceiling]
    assert len(analyzed) == expected
    # Whole sessions only -- never half a conversation.
    per_session: dict[str, int] = {}
    for trace in analyzed:
        key = trace.info.trace_metadata[TraceMetadataKey.TRACE_SESSION]
        per_session[key] = per_session.get(key, 0) + 1
    assert set(per_session.values()) == {traces_per_session}


def test_a_store_smaller_than_the_bound_is_sampled_whole(monkeypatch: pytest.MonkeyPatch):
    store = [
        _session_trace(f"tr-{session}-{index}", f"sess-{session}", 1_800_000_000_000 + session)
        for session in range(5)
        for index in range(3)
    ]
    search, _ = _honest_search(store)
    monkeypatch.setattr("mlflow.genai.discovery.sampling.mlflow.search_traces", search)

    assert (
        len(
            sample_traces(
                MAX_SCHEDULED_SESSION_SAMPLE_SIZE,
                search_kwargs={"experiment_ids": ["exp-1"], "filter_string": "timestamp_ms > 0"},
            )
        )
        == 15
    )


def _created_tags(client) -> dict[str, str]:
    client.create_run.assert_called_once()
    return client.create_run.call_args.kwargs["tags"]


def test_an_occurrence_records_the_window_it_covered(occurrence):
    occurrence.run()

    tags = _created_tags(occurrence.client)
    assert tags[MLFLOW_ISSUE_DETECTION_WINDOW_START] == str(WINDOW_LOWER_MS)
    assert tags[MLFLOW_ISSUE_DETECTION_WINDOW_END] == str(WINDOW_UPPER_MS)


def test_an_occurrence_with_nothing_to_analyze_still_records_its_window(occurrence):
    # The one case where the window is the only thing the run can say: without these tags a quiet
    # occurrence carries no evidence of which stretch it found empty.
    occurrence.newest.return_value = None

    occurrence.run()

    tags = _created_tags(occurrence.client)
    assert tags[MLFLOW_ISSUE_DETECTION_WINDOW_START] == str(WINDOW_LOWER_MS)
    assert tags[MLFLOW_ISSUE_DETECTION_WINDOW_END] == str(WINDOW_UPPER_MS)


def test_a_queued_occurrence_records_the_window_it_actually_covered(occurrence):
    # Its lower bound was raised to the watermark because another occurrence covered that stretch
    # while this one sat in the queue, so the tag has to carry the raised bound rather than the
    # one it was dispatched with -- otherwise the page claims coverage that never happened.
    raised_to = WINDOW_LOWER_MS + 600_000
    occurrence.client.get_experiment.return_value = _experiment_with_checkpoint(raised_to)

    occurrence.run()

    assert _created_tags(occurrence.client)[MLFLOW_ISSUE_DETECTION_WINDOW_START] == str(raised_to)


def test_an_occurrence_records_how_many_traces_it_analyzed(occurrence):
    # Two were sampled and handed in; the sampler left one of them to analyze. The run has to
    # carry both, because the submitted count is the one the run already recorded at creation and
    # reporting it as the analyzed count overstates what the model was shown.
    result = _result()
    result.total_traces_analyzed = 1
    occurrence.discover.return_value = result

    occurrence.run()

    assert _created_tags(occurrence.client)["total_traces"] == "2"
    occurrence.client.set_tag.assert_any_call("run-scheduled", "total_traces_analyzed", 1)


def test_an_occurrence_with_nothing_to_analyze_says_it_analyzed_nothing(occurrence):
    # Not left absent: a blank count reads as unknown, and this one is known to be zero.
    occurrence.newest.return_value = None

    occurrence.run()

    occurrence.client.set_tag.assert_any_call("run-scheduled", "total_traces_analyzed", 0)


def test_a_failed_occurrence_records_why(occurrence):
    occurrence.discover.side_effect = RuntimeError("the judge was unreachable")

    with pytest.raises(RuntimeError, match="the judge was unreachable"):
        occurrence.run()

    occurrence.client.set_tag.assert_any_call(
        "run-scheduled",
        MLFLOW_ISSUE_DETECTION_ERROR,
        "RuntimeError: the judge was unreachable",
    )


def test_a_failure_that_cannot_be_recorded_still_raises_the_original(occurrence):
    # The tag write is best-effort: a store that refuses it must not replace the failure an
    # operator needs to see with one about tagging.
    occurrence.discover.side_effect = RuntimeError("the judge was unreachable")
    occurrence.client.set_tag.side_effect = MlflowException("tag rejected")

    with pytest.raises(RuntimeError, match="the judge was unreachable"):
        occurrence.run()

    occurrence.client.set_terminated.assert_called_once_with(
        "run-scheduled", RunStatus.to_string(RunStatus.FAILED)
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("plain"), "RuntimeError: plain"),
        # A bare raise carries no message, and "RuntimeError: " would read as a truncation.
        (RuntimeError(), "RuntimeError"),
        (RuntimeError("   "), "RuntimeError"),
    ],
)
def test_a_failure_is_described_by_type_and_message(error, expected):
    assert _describe_failure(error) == expected


def test_a_long_failure_message_is_truncated_rather_than_dropped():
    described = _describe_failure(RuntimeError("x" * 2000))

    assert len(described) == MAX_ERROR_TAG_LENGTH
    assert described.startswith("RuntimeError: xxx")
    assert described.endswith("…")
