import multiprocessing
import os
import signal

from mlflow.entities.run_status import RunStatus
from mlflow.projects.pollable_run import monitor_run, maybe_set_run_terminated
from mlflow.utils.logging_utils import eprint


def _run_in_subprocess(target, args, **kwargs):
    """
    Runs a Python function as a child process. The function's output will be streamed
    to the current process's stdout/stderr
    :param target: Function to run
    :param args: Iterable of arguments to pass to the function
    :param kwargs: Additional arguments to pass to the `multiprocessing.Process` launched to run the
                   function.
    :return: The `multiprocessing.Process` used to run the function
    """
    def wrapper():
        target(*args)
    p = multiprocessing.Process(target=wrapper, args=[], **kwargs)
    p.start()
    return p


class SubmittedRun(object):
    """
    Class exposing information about an MLflow project run submitted for execution. Note that the
    run ID may be None if it is unknown, e.g. if we launched a run against a tracking server that
    our local client cannot access - in this case it's also not possible to get the run's status.
    """
    # TODO: we handle the case where the local client can't access the tracking server to support
    # e.g. running projects on Databricks without specifying a tracking server. Should be able
    # to remove this logic once Databricks has a hosted tracking server.
    def __init__(self, active_run, pollable_run_obj):
        self._active_run = active_run
        self._monitoring_process = _run_in_subprocess(
            target=monitor_run, args=(pollable_run_obj, self._active_run,))

    @property
    def run_id(self):
        """Returns the MLflow run ID of the current run"""
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
        self._monitoring_process.join()

    def cancel(self):
        """
        Attempts to cancel the current run by interrupting the monitoring process; note that this
        will not cancel the run if it has already completed.
        """
        try:
            os.kill(self._monitoring_process.pid, signal.SIGTERM)
        except OSError:
            pass
        self._monitoring_process.join()
        # In rare cases, it's possible that we cancel the monitoring subprocess before it has a
        # chance to set up a signal handler. In this case we should update the status of the MLflow
        # run here.
        maybe_set_run_terminated(self._active_run, "FAILED")
