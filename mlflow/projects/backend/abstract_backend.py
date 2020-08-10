from abc import abstractmethod, ABCMeta


class AbstractBackend:
    """
    Abstract plugin class defining the interface needed to execute MLflow projects. You can define
    subclasses of ``AbstractBackend`` and expose them as third-party plugins to enable running
    MLflow projects against custom execution backends (e.g. to run projects against your team's
    in-house cluster or job scheduler). See `MLflow Plugins <../../plugins.html>`_ for more
    information.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def run(
        self, project_uri, entry_point, params, version, backend_config, tracking_uri, experiment_id
    ):
        """
        Submit an entrypoint. It must return a SubmittedRun object to track the execution

        :param project_uri: URI of the project to execute, e.g. a local filesystem path
               or a Git repository URI like https://github.com/mlflow/mlflow-example
        :param entry_point: Entry point to run within the project.
        :param params: Dict of parameters to pass to the entry point
        :param version: For git-based projects, either a commit hash or a branch name.
        :param backend_config: A dictionary, or a path to a JSON file (must end in '.json'), which
                               will be passed as config to the backend. The exact content which
                               should be provided is different for each execution backend and is
                               documented at https://www.mlflow.org/docs/latest/projects.html.
        :param tracking_uri: URI of tracking server against which to log run information related
                             to project execution.
        :param experiment_id: ID of experiment under which to launch the run.

        :return: A :py:class:`mlflow.projects.SubmittedRun`. This function is expected to run
                 the project asynchronously, i.e. it should trigger project execution and then
                 immediately return a `SubmittedRun` to track execution status.
        """
        raise NotImplementedError()
