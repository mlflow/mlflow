from abc import abstractproperty, abstractmethod, ABCMeta


class AbstractBackend():
    """
    Abstract class for MLflow Project Execution Backend
    A backend must implement the run method to launch a new job
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, active_run, uri, entry_point, work_dir, parameters,
            experiment_id, cluster_spec, project):
        """
        Submit an entrypoint. It must returns a SubmittedRun object to track the execution
        """
        pass

    @abstractproperty
    def name(self):
        pass
