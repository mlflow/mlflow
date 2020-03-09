from abc import abstractproperty, abstractmethod, ABCMeta


class AbstractBackend():
    """
    Abstract class for MLflow Project Execution Backend
    A backend must implement the run method to launch a new job
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, run_id, experiment_id, project_uri, entry_point, params,
            version, backend_config):
        """
        Submit an entrypoint. It must returns a SubmittedRun object to track the execution

        :param run_id: Current MLflow run run_id. Everything will registred in this run.
        :param experiment_id: Experiment id where to add the run
        :param project_uri: URI to the project (could be a local or git URI).
                    This is the parameter given to mlflow run command.
        :param entry_point: name of the entry point to execute.
        :param params: Dict of parameters to pass to the entry point
        :param backend_config: Dict to pass parameters to the backend
        :param version: for Git-based projects, either a commit hash or a branch name.

        :return: A :py:class:`mlflow.projects.SubmittedRun`. This function is expected to run
                 the project asynchronously, i.e. it should trigger project execution and then
                 immediately return a `SubmittedRun` to track execution status.
        """
        raise NotImplementedError()
