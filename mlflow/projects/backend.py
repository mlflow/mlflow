from abc import abstractmethod


class ProjectBackend():
    """
    Wrapper around an MLflow project remote backend (e.g. databricks, azure)
    This class defines the interface for validateing, setting up logging,
    and submitting a run.

    NOTE:

        Subclasses of ``ProjectBackend`` must expose a ``backend_type`` member
        containing the str reference name of the backend.
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
        Submits the run to the remote compute, returns SubmittedRun obj
        """
        pass

    @staticmethod
    @abstractmethod
    def _parse_config(backend_config):
        """
        Parse the backend config
        """
        pass

    @property
    @abstractmethod
    def backend_type(self):
        """
        Returns the type of execution backend, i.e kubernetes, azure, ect.
        """
        pass
