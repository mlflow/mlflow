import multiprocessing
import os
import signal
import sys

from mlflow.entities.run_status import RunStatus
from mlflow.projects.pollable_run import maybe_set_run_terminated
from mlflow.utils.logging_utils import eprint


def monitor_run(pollable_run, active_run):
    """
    Polls the run for termination, sending updates on the run's status to a tracking server via
    the passed-in `ActiveRun` instance. This function is intended to be run asynchronously
    in a subprocess.
    """
    # Add a SIGTERM & SIGINT handler to the current process that cancels the run
    def handler(signal_num, stack_frame):  # pylint: disable=unused-argument
        eprint("=== Run (%s) was interrupted, cancelling run... ===" % pollable_run.describe())
        pollable_run.cancel()
        maybe_set_run_terminated(active_run, "FAILED")
        sys.exit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    # Perform any necessary setup for the pollable run, then wait on it to finish
    pollable_run.setup()
    run_succeeded = pollable_run.wait()
    if run_succeeded:
        eprint("=== Run (%s) succeeded ===" % pollable_run.describe())
        maybe_set_run_terminated(active_run, "FINISHED")
    else:
        eprint("=== Run (%s) failed ===" % pollable_run.describe())
        maybe_set_run_terminated(active_run, "FAILED")


class SubmittedRun(object):
    """
    Class exposing information about an MLflow project run submitted for execution.
    Note that methods that return run information (e.g. `run_id` and `get_status`) may return None
    if we launched a run against a tracking server that our local client cannot access.
    """
    def __init__(self, active_run, pollable_run_obj):
        self._active_run = active_run
        # Launch subprocess that watches our pollable run & sends status updates to the tracking
        # server
        self._monitoring_subprocess = multiprocessing.Process(
            target=monitor_run, args=(pollable_run_obj, self._active_run,))
        self._monitoring_subprocess.start()

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
        self._monitoring_subprocess.join()

    def cancel(self):
        """
        Attempts to cancel the current run by interrupting the monitoring process; note that this
        will not cancel the run if it has already completed.
        """
        try:
            os.kill(self._monitoring_subprocess.pid, signal.SIGTERM)
        except OSError:
            pass
        self._monitoring_subprocess.join()
        # In rare cases, it's possible that we cancel the monitoring subprocess before it has a
        # chance to set up a signal handler. In this case we should update the status of the MLflow
        # run here.
        maybe_set_run_terminated(self._active_run, "FAILED")
