import atexit
import os
import signal
import threading

from mlflow.entities.run_status import RunStatus
from mlflow.utils import process
from mlflow.utils.logging_utils import eprint

launched_runs = []
lock = threading.Lock()

_is_exit_handler_registered = False


def _add_run(submitted_run_obj):
    global _is_exit_handler_registered
    with lock:
        # Note: we wait until we've created a run to register our handler, since the multiprocessing
        # module registers its exit handler only when a subprocess is run. This assumes that we
        # launch a monitoring subprocess for each run.
        if not _is_exit_handler_registered:
            atexit.register(_wait_runs)
            _is_exit_handler_registered = True
        launched_runs.append(submitted_run_obj)


def _wait_runs():
    try:
        eprint("=== Waiting for active runs to complete (interrupting will kill active runs) ===")
        with lock:
            for run in launched_runs:
                run.wait()
    except KeyboardInterrupt:
        _do_kill_runs()


def _do_kill_runs():
    with lock:
        for run in launched_runs:
            run.cancel()


class SubmittedRun(object):
    """
    Class exposing information about a run submitted for execution. Note that the run ID
    may be None if it is unknown, e.g. if we launched a run against a tracking server that our
    local client cannot access.
    """
    def __init__(self, active_run, pollable_run):
        self._active_run = active_run
        self._monitoring_process = process.exec_fn(
            target=pollable_run.monitor_run, args=(self._active_run,))
        _add_run(self)


    @property
    def run_id(self):
        """Returns the MLflow run ID of the current run"""
        return self._active_run.run_info.run_uuid

    def get_status(self):
        """Gets the human-readable status of the MLflow run from the tracking server."""
        if not self._active_run:
            raise Exception("Can't get MLflow run status; the run's status has not been "
                            "persisted to an accessible tracking server.")
        return RunStatus.to_string(self._active_run.get_run().info.status)

    def wait(self):
        """
        Waits for the run to complete. Note that in some cases (e.g. remote execution on
        Databricks), we may wait until the remote job completes rather than until the MLflow run
        completes.
        """
        self._monitoring_process.join()

    def cancel(self):
        """
        Attempts to cancel the current run by interrupting the monitoring process; note that this
        will not cancel the run if it has already completed.
        """
        try:
            self._monitoring_process.terminate()
            # os.kill(self._monitoring_process.pid, signal.SIGINT)
        except OSError:
            pass
        self._monitoring_process.join()
