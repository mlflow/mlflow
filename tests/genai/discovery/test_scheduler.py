import json
from pathlib import Path
from unittest import mock

import pytest

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.genai.discovery.checkpoint import (
    FIRST_RUN_LOOKBACK_MS,
    STALE_WATERMARK_CAP_MS,
    DetectionCheckpoint,
)
from mlflow.genai.discovery.constants import DEFAULT_CATEGORIES
from mlflow.genai.discovery.schedule import (
    DetectionCadence,
    DetectionSchedule,
    DetectionScheduleState,
    ScheduleModel,
    ScheduleTraceFilter,
    parse_detection_schedule_state,
)
from mlflow.genai.discovery.scheduler import (
    MAX_EXPERIMENTS_PER_PASS,
    run_issue_detection_scheduler,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_CHECKPOINT,
    MLFLOW_ISSUE_DETECTION_PRIORITIES,
    MLFLOW_ISSUE_DETECTION_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
)

# 2026-05-04 is a Monday; 13:00 and 14:00 UTC bracket one hourly occurrence.
ONE_PM_MS = 1777899600000
HOUR_MS = 60 * 60 * 1000


@pytest.fixture
def store(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def dispatch(store):
    """Run one dispatcher pass against ``store`` with job submission captured."""

    def run(now_ms: int):
        submitted = mock.MagicMock()
        submitted.job_id = "job-1"
        with (
            mock.patch(
                "mlflow.server.handlers._get_tracking_store", return_value=store
            ) as tracking_store,
            mock.patch("mlflow.server.jobs.submit_job", return_value=submitted) as submit_job,
        ):
            run_issue_detection_scheduler(now_ms=now_ms)
        tracking_store.assert_called_once()
        return submit_job

    return run


def _experiment_with(store, name: str, **tags: str) -> str:
    return store.create_experiment(
        name, tags=[ExperimentTag(key, value) for key, value in tags.items()]
    )


def _schedule_tags(schedule: DetectionSchedule) -> dict[str, str]:
    return {
        MLFLOW_ISSUE_DETECTION_SCHEDULE: schedule.to_json(),
        MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER: schedule.trace_filters_to_json(),
    }


def _state(store, experiment_id: str) -> DetectionScheduleState | None:
    return parse_detection_schedule_state(
        store.get_experiment(experiment_id).tags.get(MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE)
    )


def test_an_experiment_with_no_schedule_is_never_dispatched(store, dispatch):
    _experiment_with(store, "unscheduled")
    assert dispatch(ONE_PM_MS).call_count == 0


def test_the_first_pass_anchors_a_new_schedule_without_running_it(store, dispatch):
    experiment_id = _experiment_with(
        store, "fresh", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )

    assert dispatch(ONE_PM_MS).call_count == 0
    assert _state(store, experiment_id) == DetectionScheduleState(last_fire_time_ms=ONE_PM_MS)


def test_the_occurrence_after_the_anchor_is_dispatched(store, dispatch):
    experiment_id = _experiment_with(
        store, "hourly", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)

    # Woken half a minute after the fire time, as a once-a-minute dispatcher usually is.
    submit_job = dispatch(ONE_PM_MS + HOUR_MS + 30_000)
    submit_job.assert_called_once()
    params = submit_job.call_args.kwargs["params"]
    assert params["experiment_id"] == experiment_id
    assert params["trace_filter"] == ""
    # The window's upper bound is the fire time, not the moment the dispatcher woke up, so a late
    # pass does not quietly widen what the occurrence covers.
    assert params["window_upper_bound_ms"] == ONE_PM_MS + HOUR_MS
    assert _state(store, experiment_id).last_fire_time_ms == ONE_PM_MS + HOUR_MS


def test_an_occurrence_is_not_dispatched_twice(store, dispatch):
    _experiment_with(
        store, "hourly", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)
    assert dispatch(ONE_PM_MS + HOUR_MS).call_count == 1
    assert dispatch(ONE_PM_MS + HOUR_MS + 60_000).call_count == 0


def test_a_paused_schedule_is_anchored_but_never_dispatched(store, dispatch):
    experiment_id = _experiment_with(
        store,
        "paused",
        **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY, paused=True)),
    )
    dispatch(ONE_PM_MS)
    assert dispatch(ONE_PM_MS + 10 * HOUR_MS).call_count == 0
    assert _state(store, experiment_id).last_fire_time_ms == ONE_PM_MS


def test_a_first_run_window_looks_back_one_cadence_interval(store, dispatch):
    _experiment_with(
        store, "daily", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.DAILY))
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + 25 * HOUR_MS)
    params = submit_job.call_args.kwargs["params"]
    assert (
        params["window_upper_bound_ms"] - params["window_lower_bound_ms"]
        == FIRST_RUN_LOOKBACK_MS["daily"]
    )


def test_a_later_window_starts_at_the_watermark(store, dispatch):
    watermark = ONE_PM_MS - 5 * 60_000
    experiment_id = _experiment_with(
        store,
        "hourly",
        **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY)),
        **{
            MLFLOW_ISSUE_DETECTION_CHECKPOINT: DetectionCheckpoint(
                last_scanned_trace_time_ms=watermark
            ).to_json()
        },
    )
    dispatch(ONE_PM_MS)

    params = dispatch(ONE_PM_MS + HOUR_MS).call_args.kwargs["params"]
    assert params["window_lower_bound_ms"] == watermark
    assert _state(store, experiment_id) is not None


def test_a_long_paused_schedule_does_not_rescan_a_huge_backlog(store, dispatch):
    _experiment_with(
        store,
        "stale",
        **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.DAILY)),
        **{
            MLFLOW_ISSUE_DETECTION_CHECKPOINT: DetectionCheckpoint(
                last_scanned_trace_time_ms=ONE_PM_MS - 90 * 24 * HOUR_MS
            ).to_json()
        },
    )
    dispatch(ONE_PM_MS)

    params = dispatch(ONE_PM_MS + 25 * HOUR_MS).call_args.kwargs["params"]
    assert (
        params["window_upper_bound_ms"] - params["window_lower_bound_ms"] == STALE_WATERMARK_CAP_MS
    )


def test_the_schedules_clauses_reach_the_job(store, dispatch):
    schedule = DetectionSchedule(
        cadence=DetectionCadence.HOURLY,
        trace_filters=(
            ScheduleTraceFilter(field="metadata", key="environment", value="production"),
            ScheduleTraceFilter(field="tag", key="team", value="search"),
        ),
    )
    _experiment_with(store, "filtered", **_schedule_tags(schedule))
    dispatch(ONE_PM_MS)

    params = dispatch(ONE_PM_MS + HOUR_MS).call_args.kwargs["params"]
    assert params["trace_filter"] == (
        "request_metadata.`environment` = 'production' AND tags.`team` = 'search'"
    )


def test_the_experiments_saved_categories_reach_the_job(store, dispatch):
    _experiment_with(
        store,
        "with-priorities",
        **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY)),
        **{
            MLFLOW_ISSUE_DETECTION_PRIORITIES: json.dumps({
                "instructions": "watch the checkout flow",
                "categories": {"cost": False},
            })
        },
    )
    dispatch(ONE_PM_MS)

    params = dispatch(ONE_PM_MS + HOUR_MS).call_args.kwargs["params"]
    assert "cost" not in params["categories"]
    assert "correctness" in params["categories"]
    assert params["instructions"] == "watch the checkout flow"


def test_an_experiment_with_no_saved_priorities_sends_none(store, dispatch):
    _experiment_with(
        store, "bare", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)

    params = dispatch(ONE_PM_MS + HOUR_MS).call_args.kwargs["params"]
    assert params["instructions"] == ""
    assert set(params["categories"]) == set(DEFAULT_CATEGORIES)


def test_a_schedule_tag_this_server_cannot_read_is_skipped(store, dispatch):
    _experiment_with(store, "broken", **{MLFLOW_ISSUE_DETECTION_SCHEDULE: "{not json"})
    assert dispatch(ONE_PM_MS).call_count == 0


def test_one_unreadable_schedule_does_not_stop_the_others(store, dispatch):
    _experiment_with(store, "broken", **{MLFLOW_ISSUE_DETECTION_SCHEDULE: "{not json"})
    _experiment_with(
        store, "good", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)
    assert dispatch(ONE_PM_MS + HOUR_MS).call_count == 1


def test_a_failed_submission_does_not_stop_the_pass(store, dispatch):
    _experiment_with(
        store, "first", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    _experiment_with(
        store, "second", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)

    submitted = mock.MagicMock()
    submitted.job_id = "job-1"
    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
        mock.patch(
            "mlflow.server.jobs.submit_job",
            side_effect=[RuntimeError("job execution is disabled"), submitted],
        ) as submit_job,
    ):
        run_issue_detection_scheduler(now_ms=ONE_PM_MS + HOUR_MS)

    assert submit_job.call_count == 2


def test_an_occurrence_is_claimed_before_it_is_submitted(store, dispatch):
    experiment_id = _experiment_with(
        store, "hourly", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)

    claimed_when_submitted: list[int | None] = []

    def _record(**kwargs):
        state = _state(store, experiment_id)
        claimed_when_submitted.append(state.last_fire_time_ms if state else None)
        raise RuntimeError("job execution is disabled")

    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
        mock.patch("mlflow.server.jobs.submit_job", side_effect=_record),
    ):
        run_issue_detection_scheduler(now_ms=ONE_PM_MS + HOUR_MS)

    # The claim was already durable when the submission was attempted, so the crash loses this
    # occurrence instead of running it twice.
    assert claimed_when_submitted == [_state(store, experiment_id).last_fire_time_ms]
    # And the lost occurrence is not retried.
    assert dispatch(ONE_PM_MS + HOUR_MS + 60_000).call_count == 0


def test_a_pass_pages_through_more_experiments_than_one_page_holds(store, dispatch):
    for index in range(3):
        _experiment_with(
            store,
            f"exp-{index}",
            **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY)),
        )
    with mock.patch("mlflow.genai.discovery.scheduler._EXPERIMENTS_PAGE_SIZE", 1):
        dispatch(ONE_PM_MS)
        assert dispatch(ONE_PM_MS + HOUR_MS).call_count == 3


def test_a_pass_is_bounded(store, dispatch):
    for index in range(3):
        _experiment_with(
            store,
            f"exp-{index}",
            **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY)),
        )
    with (
        mock.patch("mlflow.genai.discovery.scheduler.MAX_EXPERIMENTS_PER_PASS", 2),
        mock.patch("mlflow.genai.discovery.scheduler._EXPERIMENTS_PAGE_SIZE", 1),
    ):
        dispatch(ONE_PM_MS)
        assert dispatch(ONE_PM_MS + HOUR_MS).call_count == 2
    assert MAX_EXPERIMENTS_PER_PASS > 2


class _RecordingHuey:
    def __init__(self):
        self.periodic_tasks = {}

    def periodic_task(self, *_args, **_kwargs):
        def decorator(fn):
            self.periodic_tasks[fn.__name__] = fn
            return fn

        return decorator

    def lock_task(self, *_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator


def test_the_dispatcher_is_registered_as_a_periodic_task():
    from mlflow.server.jobs.utils import register_periodic_tasks

    huey = _RecordingHuey()
    register_periodic_tasks(huey)

    assert "issue_detection_scheduler" in huey.periodic_tasks


def test_a_failing_pass_does_not_bring_down_the_periodic_task_consumer():
    from mlflow.server.jobs.utils import register_periodic_tasks

    huey = _RecordingHuey()
    register_periodic_tasks(huey)

    with mock.patch(
        "mlflow.genai.discovery.scheduler.run_issue_detection_scheduler",
        side_effect=RuntimeError("the tracking store was unreachable"),
    ) as scheduler:
        huey.periodic_tasks["issue_detection_scheduler"]()

    scheduler.assert_called_once()


def test_the_schedule_model_is_what_the_job_is_asked_to_run(store, dispatch):
    _experiment_with(
        store,
        "with-model",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.HOURLY,
                model=ScheduleModel(provider="anthropic", model="claude-opus-4-8"),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + HOUR_MS)
    assert submit_job.call_args.kwargs["params"]["model"] == "anthropic:/claude-opus-4-8"


def test_a_gateway_endpoint_is_passed_as_a_gateway_model_uri(store, dispatch):
    _experiment_with(
        store,
        "with-endpoint",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.HOURLY, model=ScheduleModel(endpoint_name="prod-gateway")
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + HOUR_MS)
    kwargs = submit_job.call_args.kwargs
    assert kwargs["params"]["model"] == "gateway:/prod-gateway"
    # An endpoint carries its own credentials server-side, so none are handed to the job.
    assert kwargs["extra_envs"] == {}


def test_a_schedule_naming_no_model_leaves_the_job_on_the_server_default(store, dispatch):
    _experiment_with(
        store, "no-model", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + HOUR_MS)
    kwargs = submit_job.call_args.kwargs
    assert kwargs["params"]["model"] is None
    assert kwargs["extra_envs"] == {}


def test_the_stored_key_for_the_schedules_provider_is_handed_to_the_job(store, dispatch):
    store.create_gateway_secret(
        secret_name="openai-key", secret_value={"api_key": "sk-stored"}, provider="openai"
    )
    _experiment_with(
        store,
        "with-key",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.HOURLY,
                model=ScheduleModel(provider="openai", model="gpt-5.6-sol"),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + HOUR_MS)
    assert submit_job.call_args.kwargs["extra_envs"] == {"OPENAI_API_KEY": "sk-stored"}


def test_a_key_stored_for_another_provider_is_not_used(store, dispatch):
    store.create_gateway_secret(
        secret_name="anthropic-key", secret_value={"api_key": "sk-anthropic"}, provider="anthropic"
    )
    _experiment_with(
        store,
        "other-provider",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.HOURLY,
                model=ScheduleModel(provider="openai", model="gpt-5.6-sol"),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + HOUR_MS)
    # Nothing is stored for OpenAI, so the run falls back to the server's own environment rather
    # than being handed another provider's key.
    assert submit_job.call_args.kwargs["extra_envs"] == {}


def test_an_unresolvable_credential_still_dispatches_the_occurrence(store, dispatch):
    _experiment_with(
        store,
        "unknown-provider",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.HOURLY,
                model=ScheduleModel(provider="not-a-provider", model="m"),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + HOUR_MS)
    submit_job.assert_called_once()
    assert submit_job.call_args.kwargs["extra_envs"] == {}


def test_the_most_recently_saved_key_for_the_provider_is_the_one_used(store, dispatch):
    # Stamped explicitly: two keys saved in the same millisecond would leave "most recent"
    # decided by insertion order rather than by time.
    with mock.patch(
        "mlflow.store.tracking.gateway.sqlalchemy_mixin.get_current_time_millis",
        side_effect=[ONE_PM_MS - HOUR_MS, ONE_PM_MS],
    ):
        store.create_gateway_secret(
            secret_name="openai-old", secret_value={"api_key": "sk-old"}, provider="openai"
        )
        store.create_gateway_secret(
            secret_name="openai-new", secret_value={"api_key": "sk-rotated"}, provider="openai"
        )
    _experiment_with(
        store,
        "rotated-key",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.HOURLY,
                model=ScheduleModel(provider="openai", model="gpt-5.6-sol"),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + HOUR_MS)
    assert submit_job.call_args.kwargs["extra_envs"] == {"OPENAI_API_KEY": "sk-rotated"}
