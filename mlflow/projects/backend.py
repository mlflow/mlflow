from abc import ABC, abstractmethod


class ProjectBackend(ABC):
    """
    Wrapper around an MLflow project remote backend (e.g. databricks, azure)
    for methods
    """

    def __init__(self, project, active_run, work_dir, experiment_id, entry_point="main",
                 parameters=None, backend_config=None, uri=None, storage_dir=None):
        self.project = project
        self.active_run = active_run
        self.work_dir = work_dir
        self.experiment_id = experiment_id
        self.entry_point = entry_point
        self.parameters = parameters
        self.backend_config = backend_config
        self.uri = uri
        self.storage_dir = storage_dir

    @abstractmethod
    def validate(self):
        """
        Validates that the configuration is good. Also checks if supported
        """
        pass

    @abstractmethod
    def configure(self):
        """
        Set up things like logging
        """
        pass

    @abstractmethod
    def submit_run(self):
        """
        returns a SubmittedRun, like DatabricksSubmittedRun
        """
        pass

    @staticmethod
    @abstractmethod
    def _parse_config(backend_config):
        pass

    @property
    @abstractmethod
    def backend_type(self):
        pass
