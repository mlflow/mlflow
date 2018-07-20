from mlflow.entities.run_status import RunStatus
from mlflow.utils.logging_utils import eprint


def maybe_set_run_terminated(active_run, status):
    """
    If the passed-in active run is defined and still running (i.e. hasn't already been terminated
    within user code), mark it as terminated with the passed-in status.
    """
    if active_run and not RunStatus.is_terminated(active_run.get_run().info.status):
        active_run.set_terminated(status)


class SubmittedRun(object):
    """
    Class exposing information about an MLflow project run submitted for execution.
    Note that methods that return run information (e.g. `run_id` and `get_status`) may return None
    if we launched a run against a tracking server that our local client cannot access.
    """
    def __init__(self, active_run, pollable_run_obj):
        self._active_run = active_run
        self._pollable_run_obj = pollable_run_obj

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
        maybe_set_run_terminated(self._active_run, "FAILED")
