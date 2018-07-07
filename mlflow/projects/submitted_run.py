from abc import abstractmethod
import os
import signal


class SubmittedRun(object):
    """
    Abstract class exposing information about a run submitted for execution. Note that the run ID
    may be None if it is unknown, e.g. if we launched a run against a tracking server that our
    local client cannot access.
    """
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
            if self._monitoring_process.is_alive():
                os.killpg(self._monitoring_process.pid, signal.SIGTERM)


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
