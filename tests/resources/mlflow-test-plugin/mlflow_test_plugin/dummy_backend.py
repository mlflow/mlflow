from mlflow.entities import RunStatus
from mlflow.projects.utils import prepare_project_and_run
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import SubmittedRun


class DummySubmittedRun(SubmittedRun):
    """
    A run that just does nothing
    """

    def __init__(self, run_id):
        self._run_id = run_id

    def wait(self):
        return True

    def get_status(self):
        return RunStatus.FINISHED

    def cancel(self):
        pass

    @property
    def run_id(self):
        return self._run_id


class PluginDummyProjectBackend(AbstractBackend):
    def run(self, run_id, experiment_id, project_uri, entry_point, params,
            version, backend_config):
        prepare_project_and_run(project_uri, experiment_id, run_id, entry_point, params, version)
        return DummySubmittedRun(run_id)
