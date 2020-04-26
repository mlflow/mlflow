from abc import abstractmethod, ABCMeta


class AbstractBackend():
    """
    Abstract class for MLflow Project Execution Backend
    A backend must implement the run method to launch a new job
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, project_uri, entry_point, params,
            version, backend_config, tracking_store_uri, experiment_id):
        """
        Submit an entrypoint. It must return a SubmittedRun object to track the execution

        :param project_uri: URI of the project to execute, e.g. a local filesystem path
               or a Git repository URI like https://github.com/mlflow/mlflow-example
        :param entry_point: Entry point to run within the project.
        :param params: Dict of parameters to pass to the entry point
        :param version: For git-based projects, either a commit hash or a branch name.
        :param backend_config: Dict to pass parameters to the backend
        :param tracking_store_uri: Uri to the tracking store
        :param experiment_id: Experiment id where to add the run

        :return: A :py:class:`mlflow.projects.SubmittedRun`. This function is expected to run
                 the project asynchronously, i.e. it should trigger project execution and then
                 immediately return a `SubmittedRun` to track execution status.
        """
        raise NotImplementedError()
