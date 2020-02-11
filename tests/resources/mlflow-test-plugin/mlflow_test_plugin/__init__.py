from six.moves import urllib

from mlflow.entities import RunStatus
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import SubmittedRun


class PluginFileStore(FileStore):
    """FileStore provided through entrypoints system"""

    def __init__(self, store_uri=None, artifact_uri=None):
        path = urllib.parse.urlparse(store_uri).path if store_uri else None
        self.is_plugin = True
        super(PluginFileStore, self).__init__(path, artifact_uri)


class PluginLocalArtifactRepository(LocalArtifactRepository):
    """LocalArtifactRepository provided through plugin system"""
    is_plugin = True


class PluginRunContextProvider(RunContextProvider):
    """RunContextProvider provided through plugin system"""

    def in_context(self):
        return False

    def tags(self):
        return {"test": "tag"}


class PluginRegistrySqlAlchemyStore(SqlAlchemyStore):
    def __init__(self, store_uri=None):
        path = urllib.parse.urlparse(store_uri).path if store_uri else None
        self.is_plugin = True
        super(PluginRegistrySqlAlchemyStore, self).__init__(path)


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
    def run(self, active_run, uri, entry_point, work_dir, parameters,
            experiment_id, cluster_spec, project):
        return DummySubmittedRun(active_run)

    @property
    def name(self):
        return "dummy"
