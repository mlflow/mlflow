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

# How many experiments one pass will page through. A pass runs every minute and must finish well
# inside that, so an installation with a very large number of scheduled experiments spreads its
# work over several passes rather than overrunning one. Experiments are shuffled per pass, so no
# schedule is starved by always landing beyond the bound.
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
    experiments: list[Experiment] = []
    page_token = None
    while len(experiments) < MAX_EXPERIMENTS_PER_PASS:
        page = store.search_experiments(
            view_type=ViewType.ACTIVE_ONLY,
            max_results=min(_EXPERIMENTS_PAGE_SIZE, MAX_EXPERIMENTS_PER_PASS - len(experiments)),
            filter_string=_SCHEDULED_EXPERIMENTS_FILTER,
            page_token=page_token,
        )
        experiments.extend(page)
        if not (page_token := page.token):
            break
    # Shuffled so a pass that hits the bound does not always drop the same experiments.
    random.shuffle(experiments)
    return experiments


def _submit_occurrence(
    experiment: Experiment, schedule: DetectionSchedule, fire_time_ms: int
) -> str:
    from mlflow.genai.discovery.job import run_scheduled_issue_detection_job
    from mlflow.server.jobs import submit_job

    # The window is fixed here, at claim time, so the occurrence covers exactly the stretch the
    # dispatcher decided it covers -- not whatever had arrived by the time a queued job ran.
    window = sample_window(
        read_detection_checkpoint(experiment),
        now_ms=fire_time_ms,
        first_run_lookback_ms=FIRST_RUN_LOOKBACK_MS[schedule.cadence.value],
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
        },
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
            store, experiment.experiment_id, claim_first_occurrence(now_ms)
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
    job_id = _submit_occurrence(experiment, schedule, fire_time_ms)
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
            for experiment in experiments:
                try:
                    _dispatch_if_due(store, experiment, now_ms)
                except Exception:
                    _logger.exception(
                        "Could not dispatch issue detection for experiment %s",
                        experiment.experiment_id,
                    )
