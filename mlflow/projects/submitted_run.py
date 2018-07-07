from abc import abstractmethod
import signal
import multiprocessing
import multiprocessing.process
import os
import threading

launched_runs = []
lock = threading.Lock()


def _add_run(submitted_run_obj):
    with lock:
        launched_runs.append(submitted_run_obj)

import atexit
@atexit.register
def _wait_procs():
    print("@SID in handler waiting for %s runs" % len(launched_runs))
    with lock:
        try:
            for run in launched_runs:
                print("@SID waiting (curr PID %s)" % os.getpid())
                run.wait()
        except KeyboardInterrupt:
            print("@SID killing all active runs")
            for run in launched_runs:
                run.cancel()
            raise


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
        Databricks), we may wait until the remote job completes rather than
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
        try:
            self._monitoring_process.join()
        finally:
            pass
            # if self._monitoring_process.is_alive():
            #     os.killpg(self._monitoring_process.pid, signal.SIGTERM)

    def cancel(self):
        """"""
        # """Cancels the monitoring process, which we expect to terminate the command subprocess."""
        signal.signal(self._monitoring_process.pid, signal.SIGINT)
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
        """Add internal cancel API?"""
        from mlflow.projects import databricks
        databricks.cancel_databricks(databricks_run_id=self._databricks_run_id)
