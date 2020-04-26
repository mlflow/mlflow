from mlflow.entities import RunStatus
from mlflow.projects.utils import create_run, fetch_project, load_project
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
    def run(self, project_uri, entry_point, params,
            version, backend_config, tracking_store_uri, experiment_id):
        work_dir = fetch_project(uri, version)
        project = load_project(work_dir)
        project.get_entry_point(entry_point).validate_parameters(params)
        active_run = create_run(project_uri, experiment_id, work_dir, version, entry_point, params)
        return DummySubmittedRun(active_run.run_id)
