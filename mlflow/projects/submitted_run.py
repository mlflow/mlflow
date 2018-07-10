from abc import abstractmethod
import atexit
import os
import sys
import threading

from mlflow.utils.logging_utils import eprint

launched_runs = []
lock = threading.Lock()


def _add_run(submitted_run_obj):
    with lock:
        launched_runs.append(submitted_run_obj)


@atexit.register
def _wait_runs():
    try:
        eprint("=== Waiting for active runs to complete (interrupting will kill active runs) ===")
        with lock:
            for run in launched_runs:
                run.wait()
    except KeyboardInterrupt:
        # TODO: Why don't we need to call _do_kill_runs here? Would be good to understand...
        # Answer: because the SIGINT gets forwarded to all processes in the process group,
        # see https://unix.stackexchange.com/questions/365463/can-ctrlc-send-the-sigint-signal-to-multiple-processes
        # But is there a race condition where we'll somehow orphan/abandon the monitoring processes
        # here? No, since we wait for all multiprocessing subprocesses to terminate
        sys.exit(1)


def _do_kill_runs():
    with lock:
        for run in launched_runs:
            run.cancel()


# Override the current exception hook with a new hook that calls the existing hook & also kills
# all active runs. TODO does this work in Jupyter noteboboks where sys.excepthook is potentially
# overridden?
old_hook = sys.excepthook


def _kill_runs(type, value, traceback):
    old_hook(type, value, traceback)
    if type != KeyboardInterrupt:
        eprint("=== Main thread exited with uncaught exception of type %s. Killing active runs. "
               "===" % type)
        _do_kill_runs()


sys.excepthook = _kill_runs


class SubmittedRun(object):
    """
    Abstract class exposing information about a run submitted for execution. Note that the run ID
    may be None if it is unknown, e.g. if we launched a run against a tracking server that our
    local client cannot access.
    """
    def __init__(self):
        _add_run(self)

    @abstractmethod
    def get_status(self):
        """Gets the status of the MLflow run from the tracking server."""
        pass

    @abstractmethod
    def wait(self):
        """
        Waits for the run to complete. Note that in some cases (e.g. remote execution on
        Databricks), we may wait until the remote job completes rather than until the MLflow run
        completes.
        """
        pass

    @abstractmethod
    def run_id(self):
        """Returns the MLflow run ID of the current run"""
        pass

    @abstractmethod
    def cancel(self):
        """Cancels and cleans up the resources for the current run, if the run is still active."""
        pass


class LocalSubmittedRun(SubmittedRun):
    """Implementation of SubmittedRun corresponding to a local project run."""
    def __init__(self, active_run, monitoring_process):
        super(LocalSubmittedRun, self).__init__()
        self._active_run = active_run
        self._monitoring_process = monitoring_process

    @property
    def run_id(self):
        return self._active_run.run_info.run_uuid

    def get_status(self):
        return self._active_run.get_run().info.status

    def wait(self):
        # Note: this is written with the assumption that the main source of interrupts will
        # be e.g. Ctrl+C from the terminal / "cancel" from an IPython notebook
        self._monitoring_process.join()

    def cancel(self):
        """
        Attempts to cancel the current run by interrupting the monitoring process; note that this
        will not cancel the run if it has already completed.
        """
        import signal
        try:
            os.kill(self._monitoring_process.pid, signal.SIGINT)
        except OSError:
            pass
        self._monitoring_process.join()


class DatabricksSubmittedRun(SubmittedRun):
    """Implementation of SubmittedRun corresponding to a project run on Databricks."""
    def __init__(self, active_run, databricks_run_id):
        super(DatabricksSubmittedRun, self).__init__()
        self._active_run = active_run
        self._databricks_run_id = databricks_run_id

    @property
    def run_id(self):
        return None if self._active_run is None else self._active_run.run_info.run_uuid

    def get_status(self):
        if not self._active_run:
            raise Exception("Can't get MLflow run status for run launched on Databricks; the run's "
                            "status has not been persisted to an accessible tracking server.")
        return self._active_run.get_run().run_info.status

    def wait(self):
        from mlflow.projects import databricks
        return databricks.wait_databricks(databricks_run_id=self._databricks_run_id)

    def cancel(self):
        from mlflow.projects import databricks
        databricks.cancel_databricks(databricks_run_id=self._databricks_run_id)
