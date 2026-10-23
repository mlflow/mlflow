"""The dispatcher that turns a saved detection schedule into runs.

Registered as a periodic task, it wakes every minute, finds the experiments carrying a schedule,
and submits one detection job for each schedule whose next fire time has arrived. It does no
analysis itself: deciding what is due is cheap and must stay so, while detection is slow and
belongs in the job system where it is already bounded, retried and observable.
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import nullcontext

from mlflow.entities.experiment import Experiment
from mlflow.entities.view_type import ViewType
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.genai.discovery.checkpoint import (
    FIRST_RUN_LOOKBACK_MS,
    read_detection_checkpoint,
    sample_window,
)
from mlflow.genai.discovery.schedule import (
    DetectionSchedule,
    DetectionScheduleState,
    ScheduleModel,
    claim_first_occurrence,
    claim_occurrence,
    due_occurrence_ms,
    read_detection_schedule,
    read_detection_schedule_state,
    write_detection_schedule_state,
)
from mlflow.genai.discovery.settings import read_detection_settings
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import MLFLOW_ISSUE_DETECTION_SCHEDULE
from mlflow.utils.workspace_context import ServerWorkspaceContext

_logger = logging.getLogger(__name__)

# How many experiments one pass will dispatch. A pass runs every minute and must finish well
# inside that, so an installation with a very large number of scheduled experiments spreads its
# work over several passes rather than overrunning one. The bound is applied after shuffling the
# whole set, which is what keeps a schedule from being starved by always landing beyond it.
MAX_EXPERIMENTS_PER_PASS = 500

_EXPERIMENTS_PAGE_SIZE = 100

# Only experiments that carry a schedule tag at all are considered, so the cost of a pass tracks
# the number of scheduled experiments rather than the size of the installation.
_SCHEDULED_EXPERIMENTS_FILTER = f"tags.`{MLFLOW_ISSUE_DETECTION_SCHEDULE}` != ''"


def _workspace_contexts():
    if not MLFLOW_ENABLE_WORKSPACES.get():
        return [nullcontext()]

    from mlflow.server.workspace_helpers import _get_workspace_store

    workspaces = list(_get_workspace_store().list_workspaces())
    if not workspaces:
        _logger.info("Issue detection scheduler found no workspaces; skipping.")
        return []

    # The dispatcher runs on a thread of the periodic-tasks consumer, next to other periodic
    # tasks, so each workspace is set for this thread only rather than in the process environment.
    return [ServerWorkspaceContext(workspace.name) for workspace in workspaces]


def _scheduled_experiments(store: AbstractStore) -> list[Experiment]:
    """
    Every active experiment carrying a detection schedule, shuffled.

    The whole set is read rather than the first page's worth. Stopping at the dispatch bound
    would shuffle only the experiments already fetched, and ``search_experiments`` orders them
    deterministically, so everything past the bound would never be anchored or dispatched at
    all. The set is small by construction -- one entry per experiment that has a schedule --
    which is how the online scoring scheduler reads its active scorers too.
    """
    experiments: list[Experiment] = []
    page_token = None
    while True:
        page = store.search_experiments(
            view_type=ViewType.ACTIVE_ONLY,
            max_results=_EXPERIMENTS_PAGE_SIZE,
            filter_string=_SCHEDULED_EXPERIMENTS_FILTER,
            page_token=page_token,
        )
        experiments.extend(page)
        if not (page_token := page.token):
            break
    # Shuffled before the dispatch bound is applied, so a pass that cannot reach all of them
    # drops a different subset every time rather than the same tail forever.
    random.shuffle(experiments)
    return experiments


def _provider_credentials(store: AbstractStore, model: ScheduleModel | None) -> dict[str, str]:
    """
    The credential environment a scheduled run needs for its chosen model.

    Resolved from the provider at each fire time rather than from a secret id stored with the
    schedule, so rotating a key keeps the schedule running. An AI Gateway endpoint needs nothing
    here: it carries its own credentials server-side.

    Returns an empty mapping when there is no stored secret to use, which leaves the run on the
    credentials in the server's own environment -- what every scheduled run used before a model
    could be chosen. The occurrence is still dispatched either way: a run that fails visibly for
    want of a key is easier to act on than a schedule that quietly stops firing.
    """
    from mlflow.genai.discovery.job import _fetch_provider_credentials

    if model is None or model.endpoint_name or not model.provider:
        return {}
    if not hasattr(store, "list_secret_infos"):
        return {}
    provider = model.provider.lower()
    try:
        secrets = store.list_secret_infos(provider=provider)
        if not secrets:
            return {}
        # Neither dialog offers a choice between several keys for one provider, so the most
        # recently touched one wins: saving a replacement is how a key gets rotated, and a
        # schedule should pick that up rather than stay on the one it was created with.
        newest = max(secrets, key=lambda secret: secret.last_updated_at or secret.created_at or 0)
        return _fetch_provider_credentials(store, provider, newest.secret_id)
    except Exception:
        _logger.warning(
            "Could not resolve a stored %s credential for scheduled issue detection; the run "
            "will fall back to the credentials in this server's environment",
            provider,
            exc_info=True,
        )
        return {}


def _submit_occurrence(
    store: AbstractStore,
    experiment: Experiment,
    schedule: DetectionSchedule,
    state: DetectionScheduleState,
    fire_time_ms: int,
) -> str:
    from mlflow.genai.discovery.job import run_scheduled_issue_detection_job
    from mlflow.server.jobs import submit_job

    # The window is fixed here, at claim time, so the occurrence covers exactly the stretch the
    # dispatcher decided it covers -- not whatever had arrived by the time a queued job ran.
    window = sample_window(
        read_detection_checkpoint(experiment),
        now_ms=fire_time_ms,
        first_run_lookback_ms=FIRST_RUN_LOOKBACK_MS[schedule.cadence.value],
        # Until a run succeeds there is no watermark, so the floor pinned when this schedule was
        # anchored is what stops a failed occurrence taking its traces with it.
        coverage_floor_ms=state.coverage_floor_ms,
    )
    settings = read_detection_settings(experiment)
    submitted = submit_job(
        function=run_scheduled_issue_detection_job,
        params={
            "experiment_id": experiment.experiment_id,
            "window_lower_bound_ms": window.lower_bound_ms,
            "window_upper_bound_ms": window.now_ms,
            "categories": list(settings.categories),
            "instructions": settings.instructions,
            "trace_filter": schedule.to_filter_string(),
            "model": schedule.model.to_model_uri() if schedule.model else None,
        },
        extra_envs=_provider_credentials(store, schedule.model),
    )
    return submitted.job_id


def _dispatch_if_due(store: AbstractStore, experiment: Experiment, now_ms: int) -> None:
    schedule = read_detection_schedule(experiment)
    if schedule is None:
        return
    state = read_detection_schedule_state(experiment)
    if state is None:
        # First time this server has seen the schedule. Anchor it and wait: its first run belongs
        # at the next fire time, not the moment it was noticed.
        write_detection_schedule_state(
            store,
            experiment.experiment_id,
            claim_first_occurrence(now_ms, schedule, FIRST_RUN_LOOKBACK_MS[schedule.cadence.value]),
        )
        return
    fire_time_ms = due_occurrence_ms(schedule, state, now_ms)
    if fire_time_ms is None:
        return

    # Claim before submitting. A crash between the two loses this occurrence rather than running
    # it twice, and losing one costs nothing: the watermark only advances when a run finishes, so
    # the next occurrence's window still covers the traces this one would have analyzed.
    write_detection_schedule_state(
        store, experiment.experiment_id, claim_occurrence(state, fire_time_ms, now_ms)
    )
    job_id = _submit_occurrence(store, experiment, schedule, state, fire_time_ms)
    _logger.info(
        "Submitted scheduled issue detection job %s for experiment %s (%s occurrence at %d)",
        job_id,
        experiment.experiment_id,
        schedule.cadence.value,
        fire_time_ms,
    )


def run_issue_detection_scheduler(now_ms: int | None = None) -> None:
    """
    Dispatch every issue-detection schedule whose fire time has arrived.

    One experiment's failure never stops the pass: a schedule that cannot be read or whose job
    cannot be submitted is logged and skipped, so one bad experiment tag cannot stop detection
    across an installation.

    Args:
        now_ms: The instant to evaluate schedules against. Defaults to now; a caller passes one
            to drive the clock in tests.
    """
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    for workspace_ctx in _workspace_contexts():
        with workspace_ctx:
            experiments = _scheduled_experiments(store)
            _logger.debug(
                "Issue detection scheduler found %d experiment(s) with a schedule",
                len(experiments),
            )
            if len(experiments) > MAX_EXPERIMENTS_PER_PASS:
                _logger.info(
                    "Issue detection scheduler is dispatching %d of %d scheduled experiments this "
                    "pass; the rest are picked up by later passes",
                    MAX_EXPERIMENTS_PER_PASS,
                    len(experiments),
                )
            for experiment in experiments[:MAX_EXPERIMENTS_PER_PASS]:
                try:
                    _dispatch_if_due(store, experiment, now_ms)
                except Exception:
                    _logger.exception(
                        "Could not dispatch issue detection for experiment %s",
                        experiment.experiment_id,
                    )
