import logging
import re
import threading
import time
import warnings
from datetime import timedelta

from mlflow.entities import ViewType
from mlflow.exceptions import InvalidUrlException, MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)


def _gc_once(older_than_ms: int | None = None):
    """Permanently delete runs and experiments marked for deletion."""
    store = _get_tracking_store()

    if not hasattr(store, "_hard_delete_run"):
        _logger.warning("Tracking store does not support hard-deleting runs")
        return

    skip_experiments = False
    if not hasattr(store, "_hard_delete_experiment"):
        warnings.warn(
            "The specified backend does not allow hard-deleting experiments. Experiments "
            "will be skipped.",
            FutureWarning,
            stacklevel=2,
        )
        skip_experiments = True

    if older_than_ms is None:
        older_than_ms = 0

    deleted_run_ids_older_than = getattr(store, "_get_deleted_runs", lambda older_than=0: [])(
        older_than=older_than_ms
    )
    run_ids = list(deleted_run_ids_older_than)

    if not skip_experiments:
        filter_string = None
        if older_than_ms is not None:
            filter_string = f"last_update_time < {get_current_time_millis() - older_than_ms}"

        def fetch_experiments(token=None):
            page = store.search_experiments(
                view_type=ViewType.DELETED_ONLY,
                filter_string=filter_string,
                page_token=token,
            )
            return (page + fetch_experiments(page.token)) if page.token else page

        experiment_ids = [exp.experiment_id for exp in fetch_experiments()]

        def fetch_runs(token=None):
            page = store.search_runs(
                experiment_ids=experiment_ids,
                filter_string="",
                run_view_type=ViewType.DELETED_ONLY,
                page_token=token,
            )
            return (page + fetch_runs(page.token)) if page.token else page

        run_ids.extend([run.info.run_id for run in fetch_runs()])

    for run_id in set(run_ids):
        run = store.get_run(run_id)
        try:
            artifact_repo = get_artifact_repository(run.info.artifact_uri)
            artifact_repo.delete_artifacts()
        except InvalidUrlException:
            _logger.warning(
                "Unable to resolve the provided artifact URL: '%s'. The gc process will continue "
                "and bypass artifact deletion.",
                run.info.artifact_uri,
            )
        except MlflowException as exc:
            _logger.warning(
                "Failed to delete artifacts for run %s: %s. Skipping artifact deletion.",
                run_id,
                exc,
            )
        store._hard_delete_run(run_id)
        _logger.info("Run with ID %s has been permanently deleted.", run_id)

    if not skip_experiments:
        for experiment_id in experiment_ids:
            store._hard_delete_experiment(experiment_id)
            _logger.info("Experiment with ID %s has been permanently deleted.", experiment_id)


def _parse_older_than(older_than: str) -> int:
    regex = re.compile(
        r"^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)"
        r"?((?P<seconds>[\.\d]+?)s)?$"
    )
    parts = regex.match(older_than)
    if parts is None:
        raise MlflowException(
            f"Could not parse any time information from '{older_than}'. "
            "Examples of valid strings: '8h', '2d8h5m20s', '2m4s'",
            error_code=INVALID_PARAMETER_VALUE,
        )
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return int(timedelta(**time_params).total_seconds() * 1000)


def start_gc_worker(interval: float, older_than: str | None = None) -> threading.Thread:
    """Start a background thread that runs garbage collection periodically."""

    older_than_ms = _parse_older_than(older_than) if older_than else 0

    def loop():
        while True:
            try:
                _gc_once(older_than_ms)
            except Exception:
                _logger.exception("Failed to run mlflow gc")
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="MLflowGCWorker")
    thread.start()
    _logger.info("Started MLflow GC worker with interval %s seconds", interval)
    return thread
