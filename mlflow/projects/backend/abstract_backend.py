from abc import abstractproperty, abstractmethod, ABCMeta


class AbstractBackend():
    """
    Abstract class for MLflow Project Execution Backend
    A backend must implement the run method to launch a new job
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, active_run, uri, entry_point, work_dir, parameters,
            backend_config, command_args, env_vars):
        """
        Submit an entrypoint. It must returns a SubmittedRun object to track the execution

        :param active_run: Current MLflow run object. Everything will registred in this run.
        :param uri: uri to the project (could be local or git uri).
                    This is the parameter given to mlflow run command.
        :param entry_point: name of the entry point to execute.
        :param work_dir: path to directory that contains the MLProject file
        :param parameters: Dict of parameters to pass to the entry point
        :param backend_config: Dict to pass parameters to the backend
        :param command_args: list of arguments to the command line to launch
        :param env_vars: environment variables to set
        """
        pass

    @abstractproperty
    def name(self):
        """
        Return the name of the current backend. It will be used to tag the run in MLFlow
        """
        pass
