import sys

from mlflow.entities.run_status import RunStatus
from mlflow.utils.logging_utils import eprint

_all_runs = []


def _add_run(run):
    _all_runs.append(run)


old_hook = sys.excepthook


def _kill_active_runs(type, value, traceback):
    """
    Hook that runs when the program exits with an exception - attempts to cancel all ongoing runs.
    Note that the addition of this hook makes the project execution APIs not fork-safe, in that
    a forked process
    """
    old_hook(type, value, traceback)
    for run in _all_runs:
        run.cancel()


sys.excepthook = _kill_active_runs


class SubmittedRun(object):
    """
    Class exposing information about an MLflow project run submitted for execution.
    Note that methods that return run information (e.g. `run_id` and `get_status`) may return None
    if we launched a run against a tracking server that our local client cannot access.
    """
    def __init__(self, active_run, pollable_run_obj):
        self._active_run = active_run
        self._pollable_run_obj = pollable_run_obj
        _add_run(self)

    @property
    def run_id(self):
        """Returns the MLflow run ID of the current run"""
        # TODO: we handle the case where the local client can't access the tracking server to
        # support e.g. running projects on Databricks without specifying a tracking server.
        # Should be able to remove this logic once Databricks has a hosted tracking server.
        if self._active_run:
            return self._active_run.run_info.run_uuid
        return None

    def get_status(self):
        """Gets the human-readable status of the MLflow run from the tracking server."""
        if not self._active_run:
            eprint("Can't get MLflow run status; the run's status has not been "
                   "persisted to an accessible tracking server.")
            return None
        return RunStatus.to_string(self._active_run.get_run().info.status)

    def wait(self):
        """
        Waits for the run to complete. Note that in some cases (e.g. remote execution on
        Databricks), we may wait until the remote job completes rather than until the MLflow run
        completes.
        """
        self._pollable_run_obj.wait()

    def cancel(self):
        """
        Attempts to cancel the current run by interrupting the monitoring process; note that this
        will not cancel the run if it has already completed.
        """
        self._pollable_run_obj.cancel()
