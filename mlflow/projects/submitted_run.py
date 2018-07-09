from abc import abstractmethod
import atexit
import os
import sys
import threading

from mlflow.utils.logging_utils import eprint
from mlflow.utils import process
from mlflow.utils.process import ShellCommandException

launched_runs = []
lock = threading.Lock()


def _add_run(submitted_run_obj):
    with lock:
        launched_runs.append(submitted_run_obj)


@atexit.register
def _wait_runs():
    try:
        eprint("=== Main thread completed successfully. Waiting for active runs to complete "
               "interrupting will kill active runs) ===")
        with lock:
            for run in launched_runs:
                run.wait()
    finally:
        _do_kill_runs()


old_hook = sys.excepthook


def _do_kill_runs():
    with lock:
        for run in launched_runs:
            run.cancel()


def _kill_runs(type, value, traceback):
    eprint("=== Main thread exited with uncaught exception of type %s. Killing active runs." % type)
    old_hook(type, value, traceback)
    _do_kill_runs()
    os._exit(1)


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
        pass

    @abstractmethod
    def cancel(self):
        """Cancels and cleans up the resources for the current run."""
        pass


class LocalSubmittedRun(SubmittedRun):
    """Implementation of SubmittedRun corresponding to a local project run."""
    def __init__(self, active_run, monitoring_process, command_proc):
        super(LocalSubmittedRun, self).__init__()
        self._active_run = active_run
        self._monitoring_process = monitoring_process
        self._command_proc = command_proc

    @property
    def run_id(self):
        return self._active_run.run_info.run_uuid

    def get_status(self):
        return self._active_run.get_run().info.status

    def wait(self):
        exit_code = process._wait_polling(self._command_proc.pid)
        eprint("@SID: Got process (pid %s) exit code %s" % (self._command_proc.pid, exit_code))
        self._monitoring_process.join()
        if exit_code != 0:
            raise ShellCommandException("Command failed with non-zero exit code %s" % exit_code)

    def cancel(self):
        """
        Cancels the command process, which should cause the monitoring thread to record its status
        as failed & terminate.
        """
        try:
            self._command_proc.terminate()
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
