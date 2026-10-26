import json
import logging
from pathlib import Path
from unittest import mock

import pytest

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.entities.run_status import RunStatus
from mlflow.entities.run_tag import RunTag
from mlflow.environment_variables import MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT
from mlflow.exceptions import MlflowException
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
    scheduled_detection_available,
)
from mlflow.genai.discovery.scheduler import (
    MAX_EXPERIMENTS_PER_PASS,
    run_issue_detection_scheduler,
)
from mlflow.server import (
    ARTIFACT_ROOT_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_CHECKPOINT,
    MLFLOW_ISSUE_DETECTION_PRIORITIES,
    MLFLOW_ISSUE_DETECTION_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
    MLFLOW_RUN_TYPE,
    MLFLOW_RUN_TYPE_ISSUE_DETECTION,
)

# 2026-05-04 is a Monday; 13:00 and 14:00 UTC bracket one hourly occurrence.
ONE_PM_MS = 1777899600000
HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS


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
    state = _state(store, experiment_id)
    assert state.last_fire_time_ms == ONE_PM_MS


def _detection_run(store, experiment_id: str, start_time_ms: int):
    return store.create_run(
        experiment_id,
        "someone",
        start_time_ms,
        [RunTag(MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_ISSUE_DETECTION)],
        "detection",
    )


def test_an_occurrence_due_while_detection_is_running_is_left_unclaimed(store, dispatch):
    experiment_id = _experiment_with(
        store, "overlapping", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)
    due_ms = ONE_PM_MS + HOUR_MS
    run = _detection_run(store, experiment_id, due_ms - 60_000)

    assert dispatch(due_ms).call_count == 0
    # Unclaimed on purpose: the watermark has not moved, so the occurrence is still due and a
    # later pass runs it rather than the window being skipped.
    assert _state(store, experiment_id).last_fire_time_ms == ONE_PM_MS

    store.update_run_info(run.info.run_id, RunStatus.FINISHED, due_ms, None)
    assert dispatch(due_ms).call_count == 1
    assert _state(store, experiment_id).last_fire_time_ms == due_ms


def test_a_run_left_running_past_the_job_timeout_does_not_block_the_schedule(store, dispatch):
    # Otherwise a worker that died mid-detection would stop every later occurrence forever.
    experiment_id = _experiment_with(
        store, "stale", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)
    due_ms = ONE_PM_MS + HOUR_MS
    stale_start_ms = due_ms - int(MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT.get() * 1000) - 1
    _detection_run(store, experiment_id, stale_start_ms)

    assert dispatch(due_ms).call_count == 1


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
                # The shape the tab writes: every built-in, with the unticked ones false.
                "categories": dict.fromkeys(DEFAULT_CATEGORIES, True) | {"cost": False},
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


def test_a_claim_without_a_submission_says_so(store, dispatch, caplog):
    # Otherwise the only trace of a lost occurrence is a claim with no matching submission log,
    # which reads like a silent drop rather than the deliberate at-most-once behaviour.
    experiment_id = _experiment_with(
        store, "hourly", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)

    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
        mock.patch("mlflow.server.jobs.submit_job", side_effect=RuntimeError("submit failed")),
        caplog.at_level(logging.WARNING, logger="mlflow.genai.discovery.scheduler"),
    ):
        run_issue_detection_scheduler(now_ms=ONE_PM_MS + HOUR_MS)

    assert any(
        "claimed at fire time" in record.message and experiment_id in record.message
        for record in caplog.records
    )


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


def test_a_pass_dispatches_no_more_than_the_bound(store, dispatch):
    for index in range(5):
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
        # Which experiments a pass reaches is random, so the bound is asserted on every pass
        # rather than on a particular one.
        for pass_index in range(5):
            assert dispatch(ONE_PM_MS + (pass_index + 1) * HOUR_MS).call_count <= 2
    assert MAX_EXPERIMENTS_PER_PASS > 2


def test_every_scheduled_experiment_is_eventually_dispatched_despite_the_bound(store, dispatch):
    # The regression the bound used to cause: it was applied while paging, so the experiments the
    # store returned last were never fetched, never shuffled, and never dispatched at all.
    ids = {
        _experiment_with(
            store,
            f"exp-{index}",
            **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY)),
        )
        for index in range(6)
    }
    with (
        mock.patch("mlflow.genai.discovery.scheduler.MAX_EXPERIMENTS_PER_PASS", 2),
        mock.patch("mlflow.genai.discovery.scheduler._EXPERIMENTS_PAGE_SIZE", 2),
    ):
        dispatch(ONE_PM_MS)

        dispatched: set[str] = set()
        # Each pass dispatches at most the bound, so several are needed; a shuffled subset means
        # the count needed is probabilistic, and 60 passes makes a miss vanishingly unlikely.
        for pass_index in range(60):
            submit_job = dispatch(ONE_PM_MS + (pass_index + 1) * HOUR_MS)
            assert submit_job.call_count <= 2
            dispatched.update(
                call.kwargs["params"]["experiment_id"] for call in submit_job.call_args_list
            )
            if dispatched == ids:
                break

    assert dispatched == ids


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


def test_before_the_first_success_each_window_looks_back_from_its_own_fire_time(store, dispatch):
    # With no checkpoint yet, the lower bound is one cold-start lookback before this fire time,
    # the same as upstream. A failed occurrence therefore leaves its own stretch unanalyzed, which
    # is the accepted cost of resolving the bound from the watermark alone.
    _experiment_with(
        store, "fresh", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.HOURLY))
    )
    dispatch(ONE_PM_MS)

    first = dispatch(ONE_PM_MS + HOUR_MS).call_args.kwargs["params"]
    # The job failed, so no checkpoint was ever written for this experiment.
    second = dispatch(ONE_PM_MS + 2 * HOUR_MS).call_args.kwargs["params"]

    lookback = FIRST_RUN_LOOKBACK_MS["hourly"]
    assert first["window_lower_bound_ms"] == first["window_upper_bound_ms"] - lookback
    assert second["window_lower_bound_ms"] == second["window_upper_bound_ms"] - lookback
    # Each window still moves forward with its fire time, and they abut rather than overlap.
    assert second["window_lower_bound_ms"] == first["window_upper_bound_ms"]


def test_the_secret_the_schedule_names_is_handed_to_the_job(store, dispatch):
    secret = store.create_gateway_secret(
        secret_name="openai-key", secret_value={"api_key": "sk-chosen"}, provider="openai"
    )
    _experiment_with(
        store,
        "with-key",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.DAILY,
                model=ScheduleModel(
                    provider="openai", model="gpt-5.6-sol", secret_id=secret.secret_id
                ),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + DAY_MS)
    assert submit_job.call_args.kwargs["extra_envs"] == {"OPENAI_API_KEY": "sk-chosen"}


def test_another_users_key_is_never_substituted(store, dispatch):
    # The regression this replaces: resolving by provider picked the newest secret, so a schedule
    # ran on whichever key was saved last -- on a multi-user server, somebody else's.
    mine = store.create_gateway_secret(
        secret_name="my-key", secret_value={"api_key": "sk-mine"}, provider="openai"
    )
    store.create_gateway_secret(
        secret_name="someone-elses-key", secret_value={"api_key": "sk-theirs"}, provider="openai"
    )
    _experiment_with(
        store,
        "named-secret",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.DAILY,
                model=ScheduleModel(
                    provider="openai", model="gpt-5.6-sol", secret_id=mine.secret_id
                ),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    submit_job = dispatch(ONE_PM_MS + DAY_MS)
    assert submit_job.call_args.kwargs["extra_envs"] == {"OPENAI_API_KEY": "sk-mine"}


def test_a_schedule_naming_a_deleted_secret_fails_rather_than_substituting(store, dispatch):
    secret = store.create_gateway_secret(
        secret_name="openai-key", secret_value={"api_key": "sk-gone"}, provider="openai"
    )
    store.create_gateway_secret(
        secret_name="still-here", secret_value={"api_key": "sk-other"}, provider="openai"
    )
    _experiment_with(
        store,
        "dangling-secret",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.DAILY,
                model=ScheduleModel(
                    provider="openai", model="gpt-5.6-sol", secret_id=secret.secret_id
                ),
            )
        ),
    )
    dispatch(ONE_PM_MS)
    store.delete_gateway_secret(secret.secret_id)

    # The occurrence is not submitted. Failing visibly beats running on a key nobody designated.
    assert dispatch(ONE_PM_MS + DAY_MS).call_count == 0


def test_a_schedule_with_no_named_secret_leaves_the_run_on_the_server_environment(store, dispatch):
    # A schedule saved before the id was recorded, which is what it did when it was saved.
    _experiment_with(
        store,
        "no-secret-id",
        **_schedule_tags(
            DetectionSchedule(
                cadence=DetectionCadence.DAILY,
                model=ScheduleModel(provider="openai", model="gpt-5.6-sol"),
            )
        ),
    )
    dispatch(ONE_PM_MS)

    assert dispatch(ONE_PM_MS + DAY_MS).call_args.kwargs["extra_envs"] == {}


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


def test_the_real_submission_rejects_the_credential_environment_on_the_executor_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    # Drives the real `submit_job` validation rather than the mock the dispatcher tests use, which
    # is what makes this a contract test: if that engine ever learns to carry `extra_envs`, this
    # goes green and `scheduled_detection_available` should stop reporting it unavailable.
    from mlflow.genai.discovery.job import run_scheduled_issue_detection_job
    from mlflow.server.jobs import submit_job

    monkeypatch.setenv("MLFLOW_SERVER_JOB_EXECUTION_ENGINE", "executor")
    # Every gate ahead of the environment check has to be satisfied for the submission to reach
    # it: job execution enabled (CI runs with it off, so relying on the default made this pass
    # locally and fail there), then the job backend's own requirement check.
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, f"sqlite:///{tmp_path}/jobs.db")
    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setenv(HUEY_STORAGE_PATH_ENV_VAR, str(tmp_path))
    params = {
        "experiment_id": "1",
        "window_lower_bound_ms": ONE_PM_MS,
        "window_upper_bound_ms": ONE_PM_MS + HOUR_MS,
        "categories": ["correctness"],
    }
    with pytest.raises(MlflowException, match="extra_envs is not yet supported"):
        submit_job(
            function=run_scheduled_issue_detection_job,
            params=params,
            extra_envs={"OPENAI_API_KEY": "sk-test"},
        )

    # And the capability check agrees, which is the whole reason the control is disabled there.
    monkeypatch.setenv("MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS", "true")
    with mock.patch("mlflow.server.jobs.utils._check_requirements"):
        assert scheduled_detection_available() is False


def test_a_scheduled_job_records_no_creator_so_only_an_admin_can_inspect_it(store, dispatch):
    # A background pass has no authenticated request, so `submit_job` persists `creator=None` and
    # `validate_is_job_owner` -- which requires a non-null creator matching the caller -- denies
    # every non-admin. Verified against a basic-auth server; pinned here so the limitation is
    # visible in the suite and this test turns red when it is fixed rather than silently staying
    # true. See the PR description for why the fix is deferred.
    from mlflow.server.auth import validate_is_job_owner

    _experiment_with(
        store, "no-creator", **_schedule_tags(DetectionSchedule(cadence=DetectionCadence.DAILY))
    )
    dispatch(ONE_PM_MS)
    submitted = dispatch(ONE_PM_MS + DAY_MS)
    job_id = submitted.return_value.job_id

    job = mock.MagicMock()
    job.creator = None
    with (
        mock.patch("mlflow.server.jobs.get_job", return_value=job),
        mock.patch("mlflow.server.auth._get_request_param", return_value=job_id),
        mock.patch("mlflow.server.auth.authenticate_request") as authenticate,
    ):
        authenticate.return_value.username = "alice"
        assert validate_is_job_owner() is False
        authenticate.return_value.username = "admin"
        # Not even the admin passes the *ownership* check -- admins reach these routes by
        # bypassing RBAC entirely, not by owning the job.
        assert validate_is_job_owner() is False
